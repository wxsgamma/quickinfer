import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
from random import Random
import re
from statistics import mean
import subprocess
import time
import traceback
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from pd_metrics import distribution, report, summarize


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).with_name("data") / "pd_replay.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def workloads(data, seed, vocab, arena_output):
    rng = Random(seed)

    def tokens(n):
        return [rng.randrange(min(vocab, 30000)) for _ in range(n)]

    arena = [(key, "chat", ids, arena_output, 0.) for key, ids in data["arena"]]
    if digest([r[2] for r in arena]) != data["sources"]["arena"]["ordered_input_ids_sha256"]:
        raise ValueError("Arena input token digest mismatch")
    azure = []
    origin = datetime.fromisoformat(data["azure"][0][0])
    for i, (timestamp, n, output) in enumerate(data["azure"]):
        offset = (datetime.fromisoformat(timestamp) - origin).total_seconds()
        azure.append((f"azure-{i}", "conversation", tokens(n), output, offset))
    # 先让短 prompt 的请求持续出字，200 ms 后插入一批长 prompt。
    burst = [(f"probe-{i}", "decode-probe", tokens(32), 192, 0.) for i in range(8)]
    burst += [(f"long-{i}", "long-burst", tokens(4096), 16, .2) for i in range(8)]
    return {"arena": arena, "azure": azure, "interference": burst}


def arrivals(name, rows, rate, seed):
    rng = Random(seed)
    if name == "arena":
        gaps = [0.]
        for _ in rows[1:]:
            gaps.append(gaps[-1] + rng.expovariate(1))
    else:
        gaps = [row[4] for row in rows]
    # Azure 保留原轨迹的相对间隔；Arena 使用固定种子的指数间隔。缩放后实发均值为指定 RPS。
    scale = 1. if rate == "burst" else (len(rows) - 1) / (float(rate) * gaps[-1])
    return [round(t * scale * 1_000_000) for t in gaps]


def warmup(llm, args):
    from quickinfer import SamplingParams
    rng = Random(39017)
    sizes = (11, 58, 127, 509, 1025, 4096)
    prompts = [[rng.randrange(20000) for _ in range(min(sizes[i % len(sizes)], args.context - 8))]
               for i in range(min(args.max_seqs, 32))]
    llm.generate(prompts, SamplingParams(.6, 8, True), use_tqdm=False)


def replay(llm, rows, offsets, indices, start, seed):
    import torch
    from torch._dynamo.utils import counters
    from quickinfer import SamplingParams
    scheduler, pd = llm.scheduler, llm.prefill_worker
    live, records, transfers = {}, [], []
    stats = dict.fromkeys(("prefill_steps", "decode_steps", "prefill_tokens_computed",
                          "decode_steps_with_prefill_pending", "preemptions", "kv_p_to_d_bytes",
                          "kv_d_to_p_bytes", "kv_p_to_d_stream_ms", "kv_d_to_p_stream_ms"), 0)
    schedule, preempt = scheduler.schedule_prefill, scheduler.preempt
    clock = time.monotonic_ns

    def dispatch(*args):
        seqs = schedule(*args)
        now = (clock() - start) // 1000
        for seq in seqs:
            record = live[seq.seq_id][1]
            if record["dispatch_us"] is None:
                record["dispatch_us"] = now
        return seqs

    def evict(seq):
        stats["preemptions"] += 1
        return preempt(seq)

    if pd:
        link = pd.link
        original_transfer = link.transfer
        bytes_per_slot = sum(layer[0].numel() * layer.element_size() for layer in link.layers)

        def transfer(ranges, *, send):
            if not ranges:
                return original_transfer(ranges, send=send)
            direction = "kv_d_to_p" if send else "kv_p_to_d"
            stats[direction + "_bytes"] += sum(end - begin for begin, end in ranges) * bytes_per_slot
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(link.stream)
            original_transfer(ranges, send=send)
            end.record(link.stream)
            transfers.append((direction, begin, end))

    from contextlib import ExitStack
    index = 0
    graph_sizes = set(getattr(llm.model_runner, "graphs", {}))
    compiled = counters["stats"]["unique_graphs"]
    torch.manual_seed(seed)
    with ExitStack() as stack:
        stack.enter_context(patch.object(scheduler, "schedule_prefill", dispatch))
        stack.enter_context(patch.object(scheduler, "preempt", evict))
        if pd:
            stack.enter_context(patch.object(link, "transfer", transfer))
        while index < len(indices) or live:
            now = (clock() - start) // 1000
            while index < len(indices) and offsets[indices[index]] <= now:
                i = indices[index]
                key, kind, ids, limit, _ = rows[i]
                llm.add_request(ids, SamplingParams(.6, limit, True))
                seq = scheduler.waiting[-1]
                r = dict(id=key, **{"class": kind}, input_tokens=len(ids), output_tokens=limit,
                         arrival_us=offsets[i], admit_us=(clock() - start) // 1000, dispatch_us=None, emits=[])
                records.append(r)
                live[seq.seq_id] = seq, r
                index += 1
            if not live:
                time.sleep(max(0., (start + offsets[indices[index]] * 1000 - clock()) / 1e9))
                continue
            _, count = llm.step()
            now = (clock() - start) // 1000
            if count > 0:
                stats["prefill_steps"] += 1
                stats["prefill_tokens_computed"] += count
            else:
                stats["decode_steps"] += 1
                stats["decode_steps_with_prefill_pending"] += bool(pd and pd.pending)
            for key, (seq, r) in list(live.items()):
                emits = r["emits"]
                if seq.num_completion_tokens > (emits[-1][1] if emits else 0):
                    emits.append([now, seq.num_completion_tokens])
                if seq.is_finished:
                    if seq.num_completion_tokens != r["output_tokens"]:
                        raise RuntimeError(f"Wrong output length for {r['id']}")
                    del live[key]
    if not llm.is_finished() or scheduler.block_manager.used_block_ids:
        raise RuntimeError("Requests or KV blocks remain after replay")
    # 在计时窗口外读 CUDA event，包括本通信流等生产者或对端的时间。
    torch.cuda.synchronize()
    for direction, begin, end in transfers:
        stats[direction + "_stream_ms"] += begin.elapsed_time(end)
    stats["new_runner_graphs"] = len(set(getattr(llm.model_runner, "graphs", {})) - graph_sizes)
    stats["new_dynamo_graphs"] = counters["stats"]["unique_graphs"] - compiled
    return {"requests": records, "engine": stats}


def worker(pipe, visible, lane, limit, args, mode):
    # 每个副本有独立的 CUDA 设备 0。PD / TP 两个进程在各自的一组可见卡上启动。
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible))
    try:
        if limit:
            os.sched_setaffinity(0, limit)
        import torch
        from quickinfer import LLM
        from quickinfer.llmengine.scheduler import Scheduler
        data = json.loads(DATA.read_text())
        vocab = json.loads((Path(args.model) / "config.json").read_text())["vocab_size"]
        traces = workloads(data, args.seed, vocab, args.arena_output)
        torch.manual_seed(args.seed)
        with LLM(args.model, prefill_gpu=0 if mode == "pd" else None, decode_gpu=1 if mode == "pd" else None,
                 tensor_parallel_size=2 if mode == "tp2" else 1, max_model_len=args.context,
                 max_num_batched_tokens=args.chunk, max_num_seqs=args.max_seqs, num_kvcache_blocks=args.blocks,
                 enabled_draft=bool(args.draft_model), draft_method="modeldraft", draft_model=args.draft_model) as llm:
            warmup(llm, args)
            children = [p.pid for p in llm.ps]
            if llm.prefill_worker:
                children.append(llm.prefill_worker.process.pid)
            pipe.send(("ok", dict(lane=lane, pids=[os.getpid(), *children], visible=visible,
                                  blocks=llm.config.num_kvcache_blocks, dtype=str(llm.config.hf_config.dtype),
                                  torch=str(torch.__version__), cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
                                  draft_verify_cost=llm.model_runner.verify_cost)))
            while (case := pipe.recv()) is not None:
                name, offsets, seed, workers = case
                llm.scheduler = Scheduler(llm.config)  # 每轮仅清空逻辑块表，保留已预热的模型和图。
                indices = list(range(lane, len(offsets), workers))
                pipe.send(("ok", None))
                start = pipe.recv()
                result = replay(llm, traces[name], offsets, indices, start, seed)
                pipe.send(("ok", result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()


def receive(pipe):
    ok, value = pipe.recv()
    if ok != "ok":
        raise RuntimeError(value)
    return value


def smi(gpus, *args):
    return subprocess.check_output(["nvidia-smi", *args, "-i", ",".join(map(str, gpus))], text=True)


def nvlink(gpus):
    values = {}
    try:
        for line in smi(gpus, "nvlink", "-gt", "d").splitlines():
            if line.startswith("GPU "):
                gpu = int(line.split()[1][:-1])
            elif match := re.search(r"Data (Tx|Rx): (\d+) KiB", line):
                key = f"gpu{gpu}_{match[1].lower()}"
                values[key] = values.get(key, 0) + int(match[2]) * 1024
    except subprocess.CalledProcessError:
        return {}
    return values


def telemetry(gpus):
    rows = csv.reader(smi(gpus, "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                          "--format=csv,noheader,nounits").splitlines(), skipinitialspace=True)
    return {gpu: [float(x) if x not in ("[N/A]", "N/A") else None for x in vals] for gpu, *vals in rows}


def run_case(pipes, ready, mode, name, rate, rep, rows, offsets, gpus, seed):
    for pipe in pipes:
        pipe.send((name, offsets, seed, len(pipes)))
    for pipe in pipes:
        receive(pipe)
    allowed = {pid for lane in ready for pid in lane["pids"]}
    occupied = smi(gpus, "--query-compute-apps=pid", "--format=csv,noheader,nounits")
    foreign = {int(p) for p in occupied.split() if p.isdigit()} - allowed
    if foreign:
        raise RuntimeError(f"Unrelated GPU processes {foreign} on devices {gpus}")
    before = nvlink(gpus)
    start = time.monotonic_ns() + 200_000_000
    for pipe in pipes:
        pipe.send(start)
    remaining, results, samples = set(pipes), {}, []
    next_sample = start + 100_000_000
    while remaining:
        for pipe in wait(remaining, timeout=max(0., (next_sample - time.monotonic_ns()) / 1e9)):
            results[pipe] = receive(pipe)
            remaining.remove(pipe)
        if remaining and time.monotonic_ns() >= next_sample:
            samples.append(((time.monotonic_ns() - start) / 1e9, telemetry(gpus)))
            next_sample = time.monotonic_ns() + 400_000_000
    after = nvlink(gpus)
    records = [r for pipe in pipes for r in results[pipe]["requests"]]
    if len(records) != len(rows) or {r["id"] for r in records} != {r[0] for r in rows}:
        raise RuntimeError("Missing or duplicated request")
    seconds = max(r["emits"][-1][0] for r in records) / 1e6
    samples = [s for t, s in samples if t <= seconds]
    devices = {}
    for gpu in map(str, gpus):
        cols = [[s[gpu][i] for s in samples if s[gpu][i] is not None] for i in range(3)]
        devices[gpu] = {"mean_gpu_util_pct": mean(cols[0]) if cols[0] else None,
                        "peak_used_mib": max(cols[1]) if cols[1] else None,
                        "mean_power_w": mean(cols[2]) if cols[2] else None}
    return dict(mode=mode, workload=name, rate=rate, repeat=rep, seconds=seconds, gpu_count=len(gpus),
                plan_sha256=digest([(r[0], len(r[2]), r[3], t) for r, t in zip(rows, offsets)]),
                samples=len(samples), gpu=devices, nvlink_bytes={k: after[k] - b for k, b in before.items() if k in after},
                lanes=[results[p]["engine"] for p in pipes], requests=records)


def source_info():
    paths = [Path(__file__), Path(__file__).with_name("pd_metrics.py"), DATA]
    paths.extend(sorted((ROOT / "src/quickinfer").rglob("*.py")))
    return {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def read_trials(archive):
    with ZipFile(archive) as z:
        meta = json.loads(z.read("metadata.json"))
        return meta, [json.loads(z.read(n)) for n in z.namelist() if n.startswith("trial-")]


def write_report(path, meta, trials, ttft, tpot, attainment, max_gap_ms=None):
    result = report(trials, ttft, tpot, attainment, max_gap_ms)
    result.update(metadata=meta, complete=len(trials) == meta["expected_trials"],
                  trial_count=len(trials), request_replays=sum(len(t["requests"]) for t in trials),
                  trials=[{k: v for k, v in t.items() if k != "requests"} for t in trials])
    paired = {}
    for t in trials:
        key = (t["workload"], t["rate"], t["repeat"])
        if paired.setdefault(key, t["plan_sha256"]) != t["plan_sha256"]:
            raise ValueError(f"Unpaired arrivals or lengths in {key}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    return result


def run(args):
    args.model = str(Path(args.model).resolve())
    if args.draft_model:
        args.draft_model = str(Path(args.draft_model).resolve())
    data = json.loads(DATA.read_text())
    vocab = json.loads((Path(args.model) / "config.json").read_text())["vocab_size"]
    traces = workloads(data, args.seed, vocab, args.arena_output)
    grid = [(name, r) for name in args.workloads for r in (["burst"] if name == "interference" else args.rates)]
    for name in args.workloads:
        max_length = max(len(r[2]) + r[3] for r in traces[name])
        if max_length > min(args.context, args.blocks * 256):
            raise ValueError(f"{name} needs at least {max_length} context and KV slots")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "trials.zip"
    config = {k: v for k, v in vars(args).items() if k not in ("output", "command", "ttft_ms", "tpot_ms", "attainment")}
    meta = dict(config=config, source_sha256=source_info(), sources=data["sources"],
                expected_trials=len(grid) * len(args.modes) * args.repeats,
                gpu_info=smi(args.gpus, "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader").splitlines(),
                workloads={n: {"requests": len(traces[n]), "input": distribution([len(r[2]) for r in traces[n]]),
                               "output": distribution([r[3] for r in traces[n]]),
                               "input_sha256": digest([r[2] for r in traces[n]])} for n in args.workloads},
                protocol={
                    "boundary": "Pretokenized engine arrival to observable LLM.step token commits; includes queuing and drain.",
                    "arena": "Frozen original 64 prompts; normalized exponential gaps; fixed output budget, ignore_eos.",
                    "azure": "First 128 consecutive production timestamps and input/output lengths; seeded synthetic IDs, scaled time.",
                    "interference": "8 x (32 in,192 out) at 0s; 8 x (4096 in,16 out) at 0.2s.",
                    "placement": "DP2 uses independent simultaneous replicas, round-robin by arrival; TP2 and 1P1D each use both GPUs.",
                    "warmup": "Warm CUDA graphs and representative prompt sizes; reset scheduler/prefix cache per replay. Retry if runner graphs or local Dynamo graph count grows.",
                    "metrics": "Pooled request quantiles; TPOT=(last-first)/(N-1); inter_emit is between visible chunks. Single-token TPOT omitted. Goodput=joint hits / arrival-to-last-completion wall.",
                    "capacity": "Highest tested offered rate reaching joint attainment in every repeat, including finite-trace drain in achieved throughput.",
                    "telemetry": "nvidia-smi samples about every 0.4s, NVLink counter deltas per GPU; NCCL event times include peer/dependency waits.",
                })
    seen = set()
    if archive.exists():
        stored, old = read_trials(archive)
        if not all(stored[k] == meta[k] for k in ("config", "source_sha256", "gpu_info")):
            raise ValueError("Output contains a different benchmark configuration or source revision")
        seen = {(t["workload"], t["rate"], t["mode"], t["repeat"]) for t in old}
        del old
    else:
        with ZipFile(archive, "x", ZIP_DEFLATED) as z:
            z.writestr("metadata.json", json.dumps(meta))
    ctx = mp.get_context("spawn")
    for rep in range(args.repeats):
        cases = list(grid)
        Random(args.seed + rep).shuffle(cases)
        shift = rep % len(args.modes)
        for mode in args.modes[shift:] + args.modes[:shift]:
            todo = [(n, r) for n, r in cases if (n, r, mode, rep) not in seen]
            if not todo:
                continue
            gpus = args.gpus[:1] if mode == "single" else args.gpus
            devices = [[g] for g in gpus] if mode == "dp2" else [gpus]
            pipes, processes = [], []
            print(f"loading {mode}, repeat {rep + 1}", flush=True)
            try:
                cpus = sorted(os.sched_getaffinity(0))
                for i, ds in enumerate(devices):
                    parent, child = ctx.Pipe()
                    cpuset = cpus[i::len(devices)] or cpus
                    p = ctx.Process(target=worker, args=(child, ds, i, cpuset, args, mode))
                    p.start()
                    child.close()
                    pipes.append(parent)
                    processes.append(p)
                ready = [receive(p) for p in pipes]
                for name, rate in todo:
                    seed = args.seed + rep
                    rows = traces[name]
                    offsets = arrivals(name, rows, rate, seed)
                    for attempt in range(1, 5):
                        trial = run_case(pipes, ready, mode, name, rate, rep, rows, offsets, gpus, seed)
                        growth = sum(l["new_runner_graphs"] + l["new_dynamo_graphs"] for l in trial["lanes"])
                        if not growth:
                            break
                        print(f"settling {mode} {name} {rate}: {growth} new graphs", flush=True)
                    else:
                        raise RuntimeError("Timed replay still compiled new runner graphs")
                    trial.update(warm_retries=attempt - 1, workers=ready)
                    with ZipFile(archive, "a", ZIP_DEFLATED, compresslevel=3) as z:
                        z.writestr(f"trial-{name}-{rate}-{mode}-{rep}.json", json.dumps(trial, separators=(",", ":")))
                    m = summarize([trial], args.ttft_ms, args.tpot_ms)
                    print(f"{mode} {name} {rate} repeat={rep+1} n={m['completed']} "
                          f"out={m['output_tok_s']:.1f} tok/s SLO={m['slo_attainment']:.1%}", flush=True)
            finally:
                for pipe in pipes:
                    try:
                        pipe.send(None)
                    except (EOFError, OSError):
                        pass
                    pipe.close()
                for p in processes:
                    p.join(timeout=120)
                    if p.is_alive():
                        p.terminate()
                        p.join()
    meta, trials = read_trials(archive)
    result = write_report(args.output / "summary.json", meta, trials, args.ttft_ms, args.tpot_ms, args.attainment)
    if not result["complete"]:
        raise RuntimeError("Incomplete trial grid")
    print(f"{result['trial_count']} trials, {result['request_replays']} completed requests", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Open-loop PD serving benchmark on the same GPUs")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--draft-model")
    run_parser.add_argument("--gpus", type=int, nargs=2, required=True, metavar=("P", "D"))
    run_parser.add_argument("--modes", nargs="+", choices=("single", "dp2", "tp2", "pd"), default=["dp2", "pd", "tp2"])
    run_parser.add_argument("--workloads", nargs="+", choices=("arena", "azure", "interference"),
                            default=["arena", "azure", "interference"])
    run_parser.add_argument("--rates", nargs="+", type=float, default=[4., 12., 32.])
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument("--seed", type=int, default=4701)
    run_parser.add_argument("--chunk", type=int, default=512)
    run_parser.add_argument("--context", type=int, default=4608)
    run_parser.add_argument("--max-seqs", type=int, default=32)
    run_parser.add_argument("--blocks", type=int, default=256)
    run_parser.add_argument("--arena-output", type=int, default=128)
    run_parser.add_argument("--output", type=Path, required=True, help="Directory for summary.json and trials.zip")
    rescore = commands.add_parser("report")
    rescore.add_argument("--trials", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    rescore.add_argument("--max-gap-ms", type=float, help="Also limit the longest pause between token commits")
    for sub in (run_parser, rescore):
        sub.add_argument("--ttft-ms", type=float, default=500.)
        sub.add_argument("--tpot-ms", type=float, default=50.)
        sub.add_argument("--attainment", type=float, default=.90)
    args = parser.parse_args()
    if not (math.isfinite(args.ttft_ms) and args.ttft_ms > 0 and math.isfinite(args.tpot_ms)
            and args.tpot_ms > 0 and 0 < args.attainment <= 1):
        parser.error("SLO thresholds must be positive and attainment in (0,1]")
    if args.command == "report":
        if args.max_gap_ms is not None and not (math.isfinite(args.max_gap_ms) and args.max_gap_ms > 0):
            parser.error("Maximum pause must be positive and finite")
        meta, trials = read_trials(args.trials)
        write_report(args.output, meta, trials, args.ttft_ms, args.tpot_ms, args.attainment, args.max_gap_ms)
        return
    if (len(set(args.gpus)) != 2 or min(args.gpus) < 0 or min(args.repeats, args.arena_output, args.blocks,
            args.max_seqs, args.chunk, args.context) <= 0 or any(not math.isfinite(r) or r <= 0 for r in args.rates)):
        parser.error("Select two distinct GPUs and positive sizes and request rates")
    if any(len(x) != len(set(x)) for x in (args.modes, args.workloads, args.rates)):
        parser.error("Modes, workloads and rates must be unique")
    run(args)


if __name__ == "__main__":
    main()
