import argparse
import hashlib
import json
import os
from pathlib import Path
from threading import Event, Thread
import time

import numpy as np

from pd_allocation import cpu_sets, occupants, reference_probe, verify_and_warm
from pd_serving import ROOT, nvlink, smi, source_info, telemetry
from quickinfer import PDPool, SamplingParams
from quickinfer.cli import add_draft_args, engine_kwargs
from quickinfer.llmengine import pd_pool
from quickinfer.llmengine.pd_pool import gpu_worker


GPUS = list(range(8))
LAYOUTS = {"dp8": [], "4p4d": [0, 4, 3, 7], "3p5d": [0, 4, 3]}
US = 1_000_000
STATE_FIELDS = ("time_us", "gpu", "waiting", "running", "ready_to_transfer", "reserved_blocks",
                "free_blocks", "compute_inflight", "prefill_inflight", "transfer_inflight", "preemptions")
STEP_FIELDS = ("gpu", "prefill", "dispatch_us", "reply_us", "requests", "input_positions")
REQ_FIELDS = ("arrival_us", "admit_us", "dispatch_us", "prefill_done_us", "kv_submit_us", "p_gpu", "d_gpu")
KV_FIELDS = ("request", "gpu", "sender", "ack_us", "stream_us")
HW_FIELDS = ("time_us", "gpu", "util_pct", "memory_mib", "power_w", "query_us")


def traced_worker(conn, config, gpu, prefill, decode, cpus):
    # NVTX 只在 nsys 对照轮开启。CPU 调用区间、CUDA kernel 和 NCCL stream 可分别查看。
    import torch
    from quickinfer.layers.sampler import Sampler
    from quickinfer.llmengine.draft import Model_Draft
    from quickinfer.llmengine.model_runner import ModelRunner
    from quickinfer.llmengine.prefill_worker import KVTransfer

    def annotate(cls, method, label):
        original = getattr(cls, method)

        def timed(*args, **kwargs):
            with torch.cuda.nvtx.range(label):
                return original(*args, **kwargs)
        setattr(cls, method, staticmethod(timed) if isinstance(vars(cls).get(method), staticmethod) else timed)

    run = ModelRunner.run
    def annotated_run(self, seqs, is_prefill, *args):
        stage = "P" if is_prefill else "D"
        with torch.cuda.nvtx.range(f"QI_STEP {stage} {gpu} {len(seqs)}"):
            return run(self, seqs, is_prefill, *args)
    ModelRunner.run = annotated_run
    for name in ("prepare_prefill", "prepare_decode", "prepare_sample"):
        annotate(ModelRunner, name, "QI_INPUT")
    annotate(ModelRunner, "run_model", "QI_MODEL")
    annotate(Sampler, "__call__", "QI_SAMPLE")
    annotate(Model_Draft, "gen", "QI_DRAFT")
    annotate(Model_Draft, "verify", "QI_VERIFY")
    annotate(KVTransfer, "transfer_spans", "QI_KV_SEND" if gpu in prefill else "QI_KV_RECV")
    return gpu_worker(conn, config, gpu, prefill, decode, cpus)


def plan(rate, rep, segments):
    # 各段分别固定请求数，在段内随机到达。三种分卡复用同一份输入和计划时刻。
    rng = np.random.default_rng(4701 + rep)
    offsets, start = [], 0
    for seconds in segments:
        n = round(rate * seconds)
        offsets.append(start + np.sort(rng.integers(0, seconds * US, size=n, dtype=np.int64))
                       if n else np.array([], dtype=np.int64))
        start += seconds * US
    offsets = np.concatenate(offsets)
    prompts = rng.integers(1, 30_000, size=(len(offsets), 128), dtype=np.int32)
    digest = hashlib.sha256(offsets.tobytes() + prompts.tobytes()).hexdigest()
    return offsets, prompts.tolist(), digest


class MeasuredPipe:
    def __init__(self, st, clock, steps, transfers):
        self.st, self.pipe, self.clock = st, st.pipe, clock
        self.steps, self.transfers = steps, transfers
        self.fileno = self.pipe.fileno
        self.pending = None

    def send(self, cmd):
        if cmd[0] == "run":
            seqs, prefill = cmd[1:3]
            positions = sum(s.num_scheduled_tokens for s in seqs) if prefill else len(seqs)
            self.pending = (self.clock(), int(prefill), len(seqs), positions)
        self.pipe.send(cmd)

    def recv(self):
        msg = self.pipe.recv()
        if msg[0] == "tokens":
            start, kind, size, positions = self.pending
            if len(msg) == 3:
                positions = size + sum(msg[2])  # 锚点加上 D 实际送验的草稿行。
            self.steps.append((self.st.gpu, kind, start, self.clock(), size, positions))
            self.pending = None
        elif msg[0] == "kv":
            p, d, seq = self.st.transfer[:3]
            self.transfers.append((seq.bench_index, self.st.gpu, int(self.st is p),
                                   self.clock(), round(msg[1][0] * 1000)))
        return msg

    def close(self):
        self.pipe.close()


def replay(pool, mode, rate, rep, segments, capture, dataset):
    import torch
    offsets, prompts, plan_hash = dataset
    n = len(prompts)
    stages = list(pool.stages.values())
    devices = list(pool.stages)
    pids = {st.pid for st in stages}
    if foreign := occupants(devices) - pids:
        raise RuntimeError(f"Other GPU processes: {foreign}")
    graphs = pool.reset(4701 + rep)
    before = nvlink(devices)
    req = np.full((n, len(REQ_FIELDS)), -1, dtype=np.int64)
    req[:, 0] = offsets
    emits = np.full((n, 512), -1, dtype=np.int32)
    emitted = np.zeros(n, dtype=np.int32)
    state, steps, transfers, hardware, errors = [], [], [], [], []
    started = time.monotonic_ns() + 200_000_000
    clock = lambda: (time.monotonic_ns() - started) // 1000
    stop = Event()

    def sample_hw():
        last_check = -20 * US
        try:
            while not stop.is_set():
                begin = clock()
                info = telemetry(devices)
                end = clock()
                hardware.extend(((begin + end) // 2, int(g), *vals, end-begin) for g, vals in info.items())
                if end-last_check >= 15*US:
                    if foreign := occupants(devices)-pids:
                        raise RuntimeError(f"Other GPU processes during trial: {foreign}")
                    last_check = end
                stop.wait(.3)
        except BaseException as exc:
            errors.append(str(exc))

    def sample_state(now):
        ready = dict.fromkeys(devices, 0)
        for st, seq, token in pool.ready:
            ready[st.gpu] += 1
        for st in stages:
            sched = st.scheduler
            state.append((now, st.gpu, len(sched.waiting), len(sched.running), ready[st.gpu], st.reserved,
                          len(sched.block_manager.free_block_ids), int(bool(st.batch)),
                          int(bool(st.batch) and st.prefill), int(bool(st.transfer)), st.stats["preemptions"]))

    for st in stages:
        st.pipe = MeasuredPipe(st, clock, steps, transfers)
    thread = Thread(target=sample_hw)
    thread.start()
    index, finished, next_sample, next_sync = 0, 0, 0, 0
    horizon = sum(segments) * US
    armed = capturing = False
    interval = 5_000 if capture else 50_000
    params = SamplingParams(.6, 512, True)
    try:
        while index < n or pool.live or clock() < horizon:
            now = clock()
            if errors:
                raise RuntimeError(errors[0])
            if capture and not armed and now >= (sum(segments)-8)*US:
                torch.cuda.nvtx.range_push("QI_CAPTURE")
                armed = capturing = True
            if capturing and now >= next_sync:
                torch.cuda.nvtx.mark(f"QI_SYNC {clock()}")
                next_sync = now + 100_000
            if capturing and now >= horizon+6*US:
                torch.cuda.nvtx.range_pop()
                capturing = False
            if now >= next_sample:
                sample_state(now)
                next_sample = now + interval
            while index < n and offsets[index] <= now:
                seq = pool.add_request(prompts[index], params)
                seq.bench_index = index
                req[index, 1] = clock()
                index += 1
            next_us = min(next_sample, offsets[index]) if index < n else next_sample
            timeout = max(0., (next_us-clock()) / US)
            if not pool.live:
                time.sleep(timeout)
                continue
            for seq, ns in pool.step(timeout):
                i = seq.bench_index
                count = seq.num_completion_tokens
                # 一轮 SD 可以提交多枚，共用这一轮真实的可见时刻。
                emits[i, emitted[i]:count] = (ns-started)//1000
                emitted[i] = count
                if seq.is_finished:
                    if count != 512:
                        raise RuntimeError(f"Request {i} finished at {count} tokens")
                    req[i, 2] = (seq.pd_dispatch_ns-started)//1000
                    if seq.pd_prefill_done_ns is not None:
                        req[i, 3:5] = ((seq.pd_prefill_done_ns-started)//1000, (seq.pd_transfer_ns-started)//1000)
                    req[i, 5:] = seq.pd_p_gpu, seq.pd_d_gpu
                    finished += 1
        sample_state(clock())
    finally:
        if capturing:
            torch.cuda.nvtx.mark(f"QI_SYNC {clock()}")
            torch.cuda.nvtx.range_pop()
        stop.set()
        thread.join()
        for st in stages:
            st.pipe = st.pipe.pipe
    if errors or finished != n or np.any(emits < 0) or np.any(emits[:, 1:] < emits[:, :-1]):
        raise RuntimeError(f"Incomplete events: {errors}, {finished}/{n}")
    if any(st.scheduler.block_manager.used_block_ids or st.reserved for st in stages):
        raise RuntimeError("KV still allocated after drain")
    if len(transfers) != (2*n if pool.prefill else 0):
        raise RuntimeError("KV acknowledgement count differs from request count")
    after = nvlink(devices)
    new = pool.graph_counts(4701+rep)
    meta = dict(mode=mode, rate=rate, repeat=rep, segments_s=segments, plan_hash=plan_hash, profiled=capture,
                time_origin_monotonic_ns=started, requests=n, workers={st.gpu:st.pid for st in stages},
                p_gpus=pool.prefill, d_gpus=pool.decode, step_fields=STEP_FIELDS, req_fields=REQ_FIELDS,
                bytes_per_token=stages[0].bytes_per_token, kv_blocks={str(st.gpu):st.blocks for st in stages},
                blocks_per_request=(len(prompts[0]) + params.max_tokens - 2)//pool.size + 1,
                kv_fields=KV_FIELDS, state_fields=STATE_FIELDS, hw_fields=HW_FIELDS,
                lanes={st.gpu: st.stats for st in stages}, nvlink_bytes={k:v-before[k] for k,v in after.items() if k in before},
                new_graphs=sum(sum(b-a for a,b in zip(graphs[g],new[g])) for g in devices))
    return meta, dict(req=req, emits=emits, steps=np.asarray(steps, dtype=np.int64),
                      kv=np.asarray(transfers, dtype=np.int64).reshape(-1, len(KV_FIELDS)),
                      state=np.asarray(state, dtype=np.int64), hw=np.asarray(hardware, dtype=np.float64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    add_draft_args(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Existing single-token reference for ordinary decode")
    parser.add_argument("--rates", nargs="+", type=int, default=[8, 12, 16, 20, 48])
    parser.add_argument("--modes", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lead", type=int, default=20)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--guard", type=int, default=10)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    draft = engine_kwargs(args)
    if draft["draft_model"]:
        draft["draft_model"] = str(Path(draft["draft_model"]).resolve())
    if min(args.rates) <= 0 or args.window < 1 or min(args.lead, args.guard) < 0 or args.repeats < 1:
        parser.error("Invalid rates, windows or repeat count")
    if args.capture:
        if len(args.modes)*len(args.rates)*args.repeats != 1 or args.window < 8:
            parser.error("Use one layout/rate/repeat per capture")
        pd_pool.gpu_worker = traced_worker
    args.output.mkdir(parents=True, exist_ok=True)
    hashed = source_info()
    for f in (Path(__file__), ROOT/"benchmark/pd_allocation.py"):
        hashed[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    cpus = cpu_sets(GPUS)
    signature = dict(hashes=hashed, model=str(Path(args.model).resolve()), lead=args.lead,
                     window=args.window, guard=args.guard, capture=args.capture, draft=draft,
                     gpu_info=smi(GPUS, "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader").splitlines())
    cfg = args.output/"config.json"
    saved = json.loads(cfg.read_text()) if cfg.exists() else None
    if (draft["enabled_draft"] and draft["num_draft"]) or args.reference is None:
        # SD 的多行 GEMM 和逐枚 decode 的 BF16 舍入可能不同，使用同宽度的单卡结果作对照。
        if saved is None and (busy := occupants([GPUS[0]])):
            raise RuntimeError(f"GPU {GPUS[0]} occupied: {busy}")
        signature["reference"] = saved["reference"] if saved else reference_probe(args.model, GPUS[0], draft)
    else:
        signature["reference"] = json.loads(args.reference.read_text())["metadata"]["reference_tokens"]
    if saved is not None:
        if saved != signature:
            raise RuntimeError("Directory contains another build or measurement protocol")
    else:
        cfg.write_text(json.dumps(signature, indent=2))
    for rep in range(args.repeats):
        modes = args.modes[rep % len(args.modes):] + args.modes[:rep % len(args.modes)]
        rates = list(args.rates)
        np.random.default_rng(5301+rep).shuffle(rates)
        for mode in modes:
            todo = [(rate,args.output/f"{mode}-r{rate}-rep{rep+1}.npz") for rate in rates]
            todo = [(rate,path) for rate,path in todo if not path.exists()]
            if not todo:
                continue
            if busy := occupants(GPUS):
                raise RuntimeError(f"GPUs occupied: {busy}")
            pg = LAYOUTS[mode]
            dg = [g for g in GPUS if g not in pg]
            print(f"LOAD {mode} rep={rep+1} P={pg} D={dg}", flush=True)
            with PDPool(args.model, pg, dg, cpu_sets=cpus, num_kvcache_blocks=256,
                        max_model_len=4608, max_num_batched_tokens=512, max_num_seqs=32, **draft) as pool:
                ref, verified, ties = verify_and_warm(pool, signature["reference"])
                if ref != signature["reference"]:
                    raise RuntimeError("Preflight tokens differ from direct model")
                # 各卡先遇到多行采样，避免首个正式窗口才编译 sampler。
                rng = np.random.default_rng(19037)
                for ids in rng.integers(1,30000,(320,128)).tolist():
                    pool.add_request(ids, SamplingParams(.6,32,True))
                while pool.live:
                    pool.step()
                for rate, path in todo:
                    segments = (args.lead,args.window,args.guard)
                    data = plan(rate, rep, segments)
                    print(f"RUN {mode} rate={rate} rep={rep+1} requests={len(data[0])}", flush=True)
                    for attempt in range(3):
                        meta, arrays = replay(pool, mode, rate, rep, segments, args.capture, data)
                        if not meta["new_graphs"]:
                            break
                        if args.capture:
                            raise RuntimeError("New graph in Nsight replay")
                        print(f"REWARM graphs={meta['new_graphs']}", flush=True)
                    else:
                        raise RuntimeError("Graph count did not settle")
                    meta.update(preflight_kv=len(verified), preflight_ties=ties, warm_retries=attempt)
                    tmp = path.with_suffix(".partial")
                    with tmp.open("wb") as dest:
                        np.savez_compressed(dest, meta=json.dumps(meta), **arrays)
                    os.replace(tmp,path)
                    start, stop = args.lead*US, (args.lead+args.window)*US
                    emits = arrays["emits"]
                    toks = np.count_nonzero((emits>=start)&(emits<stop))/args.window
                    drain = max(0.,emits[:,-1].max()/US-sum(segments))
                    print(f"DONE {path.stem} output={toks:.1f} drain={drain:.2f}s",flush=True)


if __name__ == "__main__":
    main()
