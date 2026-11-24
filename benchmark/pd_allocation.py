import argparse
from collections import defaultdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from multiprocessing import get_context
from random import Random
import subprocess
from threading import Event, Thread
import time
import traceback
from zipfile import ZipFile, ZIP_DEFLATED

from pd_serving import DATA, ROOT, arrivals, digest, nvlink, read_trials, receive, smi, source_info, telemetry, workloads, write_report
from pd_metrics import summarize
from quickinfer.cli import add_draft_args, engine_kwargs


def cpu_sets(gpus):
    # 同一 NUMA 节点上的卡均分可用 CPU，切换 P:D 比例时保持各张卡的 CPU 不变。
    groups = defaultdict(list)
    for line in smi(gpus, "--query-gpu=index,pci.bus_id", "--format=csv,noheader").splitlines():
        gpu, bus = line.lower().split(", ")
        domain, bus, dev = bus.split(":")
        path = Path(f"/sys/bus/pci/devices/{int(domain, 16):04x}:{bus}:{dev}/local_cpulist")
        local = set()
        for field in path.read_text().strip().split(","):
            a, _, b = field.partition("-")
            local.update(range(int(a), int(b or a) + 1))
        allowed = tuple(sorted(local & os.sched_getaffinity(0))) or tuple(sorted(os.sched_getaffinity(0)))
        groups[allowed].append(int(gpu))
    return {g: list(cpus[i::len(devices)] or cpus) for cpus, devices in groups.items() for i, g in enumerate(devices)}


def traces_for_run(args):
    data = json.loads(DATA.read_text())
    vocab = json.loads((Path(args.model) / "config.json").read_text())["vocab_size"]
    traces = workloads(data, args.seed, vocab, 128)
    for name, n_in, n_out in (("long_input", 2048, 32), ("long_output", 128, 512)):
        rng = Random(args.seed + n_in)
        traces[name] = [(f"{name}-{i}", name, [rng.randrange(30000) for _ in range(n_in)], n_out, row[4])
                        for i, row in enumerate(traces["azure"])]
    return traces


def probe_prompts():
    rng = Random(19021)
    return [([rng.randrange(20000) for _ in range(n)], length)
            for n, length in ((33, 1), (257, 8), (769, 12), (513, 8))]


def direct_probe(conn, model, gpu, draft):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from quickinfer import LLM, SamplingParams
        with LLM(model, max_model_len=4608, max_num_batched_tokens=512, max_num_seqs=32,
                 num_kvcache_blocks=256, **draft) as llm:
            greedy = lambda logits, temperatures: logits.argmax(dim=-1)
            llm.model_runner.run = partial(llm.model_runner.run, target_sampler=greedy)
            tokens = [llm.generate([ids], SamplingParams(1., n, True), use_tqdm=False)[0]["token_ids"]
                      for ids, n in probe_prompts()]
        conn.send(("ok", tokens))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def reference_probe(model, gpu, draft):
    ctx = get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=direct_probe, args=(send, model, gpu, draft))
    proc.start()
    send.close()
    try:
        return receive(recv)
    finally:
        recv.close()
        proc.join()


def verify_and_warm(pool, reference):
    from quickinfer import SamplingParams
    rng = Random(19021)
    pool.check_mode(True)
    probe = []
    for (ids, outputs), target in zip(probe_prompts(), reference, strict=True):
        seq = pool.add_request(ids, SamplingParams(1., outputs, True))
        seq.pd_expected = target
        while pool.live:
            pool.step()
        probe.append(seq.completion_token_ids)
    # 复用前缀、跨块和并发交接另外验 KV。逐字节比对在计时窗口以外。
    shared = [rng.randrange(20000) for _ in range(256)]
    for i in range(8):
        pool.add_request(shared + [i + 1] * (i + 1), SamplingParams(1., 3, True))
    while pool.live:
        pool.step()
    checked = list(pool.verified)
    if pool.prefill and len(checked) != len(probe) + 8:
        raise RuntimeError("Missing KV verification")
    ties = pool.check_mode(False)
    for i in range(32):
        n = (11, 59, 127, 509, 1025, 4096)[i % 6]
        pool.add_request([rng.randrange(20000) for _ in range(n)], SamplingParams(.6, 12, True))
    while pool.live:
        pool.step()
    return probe, checked, ties


def occupants(gpus):
    rows = smi(gpus, "--query-compute-apps=pid", "--format=csv,noheader,nounits")
    return {int(x) for x in rows.split() if x.isdigit()}


def replay(pool, name, rows, offsets, rate, rep, mode, seed):
    from quickinfer import SamplingParams
    graphs = pool.reset(seed)
    devices = list(pool.stages)
    allowed = {st.pid for st in pool.stages.values()}
    foreign = occupants(devices) - allowed
    if foreign:
        raise RuntimeError(f"Other GPU processes appeared: {foreign}")
    before = nvlink(devices)
    start = time.monotonic_ns() + 100_000_000
    clock = lambda: (time.monotonic_ns() - start) // 1000
    samples, errors = [], []
    stop = Event()

    def monitor():
        try:
            if not stop.wait(.2):
                while True:
                    timestamp = clock()
                    samples.append((timestamp, telemetry(devices)))
                    if stop.wait(.4):
                        break
        except Exception as err:
            errors.append(err)

    thread = Thread(target=monitor)
    thread.start()
    index, live, records = 0, {}, []
    cpu_start = time.process_time()
    try:
        while index < len(rows) or pool.live:
            now = clock()
            while index < len(rows) and offsets[index] <= now:
                key, kind, ids, length, _ = rows[index]
                seq = pool.add_request(ids, SamplingParams(.6, length, True))
                row = dict(id=key, **{"class": kind}, input_tokens=len(ids), output_tokens=length,
                           arrival_us=offsets[index], admit_us=clock(), emits=[])
                records.append(row)
                live[seq.seq_id] = row
                index += 1
            timeout = max(0., (offsets[index] - clock()) / 1e6) if index < len(rows) else None
            if not pool.live:
                time.sleep(timeout)
                continue
            for seq, ns in pool.step(timeout):
                row = live[seq.seq_id]
                row["emits"].append([(ns - start) // 1000, seq.num_completion_tokens])
                if seq.is_finished:
                    row.update(dispatch_us=(seq.pd_dispatch_ns - start) // 1000, p_gpu=seq.pd_p_gpu,
                               d_gpu=seq.pd_d_gpu, p_done_us=(seq.pd_prefill_done_ns - start) // 1000
                               if seq.pd_prefill_done_ns else None,
                               kv_start_us=(seq.pd_transfer_ns - start) // 1000 if seq.pd_transfer_ns else None)
                    if seq.num_completion_tokens != row["output_tokens"]:
                        raise RuntimeError(f"Wrong output count: {row['id']}")
                    del live[seq.seq_id]
    finally:
        coordinator_cpu = time.process_time() - cpu_start
        stop.set()
        thread.join()
    if errors:
        raise errors[0]
    if live or len(records) != len(rows) or any(st.scheduler.block_manager.used_block_ids or st.reserved
                                                for st in pool.stages.values()):
        raise RuntimeError("Unfinished requests or unreleased KV after replay")
    after = nvlink(devices)
    seconds = max(r["emits"][-1][0] for r in records) / 1e6
    samples = [s for t, s in samples if 0 <= t <= seconds * 1e6]
    gpu = {}
    for g in map(str, devices):
        cols = [[s[g][i] for s in samples if s[g][i] is not None] for i in range(3)]
        gpu[g] = {"mean_gpu_util_pct": sum(cols[0]) / len(cols[0]) if cols[0] else None,
                  "peak_used_mib": max(cols[1], default=None),
                  "mean_power_w": sum(cols[2]) / len(cols[2]) if cols[2] else None}
    new = pool.graph_counts(seed)
    return dict(mode=mode, workload=name, rate=rate, repeat=rep, seconds=seconds, gpu_count=len(devices),
                plan_sha256=digest([(r[0], len(r[2]), r[3], t) for r, t in zip(rows, offsets)]),
                gpu=gpu, samples=len(samples), coordinator_cpu_s=coordinator_cpu,
                nvlink_bytes={k: after[k]-v for k, v in before.items() if k in after},
                lanes=[dict(gpu=g, **s.stats) for g, s in pool.stages.items()],
                p_to_d_requests={f"{p}->{d}": n for (p, d), n in pool.routes.items()},
                new_graphs=sum(sum(b-a for a,b in zip(graphs[g],new[g])) for g in devices), requests=records)


def main():
    parser = argparse.ArgumentParser(description="Replay serving traffic across prefill/decode GPU allocations")
    parser.add_argument("--model", required=True)
    add_draft_args(parser)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--prefill-counts", nargs="+", type=int)
    parser.add_argument("--workloads", nargs="+", choices=("arena", "azure", "long_input", "long_output", "interference"),
                        default=["arena", "azure", "long_input", "long_output", "interference"])
    parser.add_argument("--rates", nargs="+", type=float, default=[12., 48., 96.])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4701)
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from quickinfer.llmengine.pd_pool import PDPool
    n = len(args.gpus)
    counts = args.prefill_counts if args.prefill_counts is not None else [0, n//4, n//2, 3*n//4]
    if (len(set(args.gpus)) != n or min(args.gpus) < 0 or len(set(counts)) != len(counts)
            or min(counts) < 0 or max(counts) >= n or min(args.rates) <= 0 or args.repeats <= 0):
        parser.error("Invalid devices, prefill counts or replay grid")
    args.model = str(Path(args.model).resolve())
    draft = engine_kwargs(args)
    if draft["draft_model"]:
        draft["draft_model"] = str(Path(draft["draft_model"]).resolve())
    # 交错选 P 卡，P/D 尽量分布在两个 CPU 插槽上。各比例在同一组物理 GPU 上比较。
    positions = {4: [0, 2, 1, 3], 8: [0, 4, 3, 7, 1, 5, 2, 6]}.get(n, range(n))
    order = [args.gpus[i] for i in positions]
    layouts = {f"{p}p{n-p}d" if p else f"dp{n}": (order[:p], [g for g in args.gpus if g not in order[:p]])
               for p in counts}
    args.output.mkdir(exist_ok=True, parents=True)
    archive = args.output / "trials.zip"
    traces = traces_for_run(args)
    grid = [(w, r) for w in args.workloads for r in (["burst"] if w == "interference" else args.rates)]
    cpus = cpu_sets(args.gpus)
    hashes = source_info()
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta = dict(config=dict(model=args.model, **draft, gpus=args.gpus, modes=list(layouts), rates=args.rates,
                            workloads=args.workloads, repeats=args.repeats, seed=args.seed, blocks=args.blocks,
                            chunk=512, context=4608, max_seqs=32), source_sha256=hashes,
                layouts={k: {"prefill": p, "decode": d} for k,(p,d) in layouts.items()}, cpu_sets=cpus,
                expected_trials=len(grid)*len(layouts)*args.repeats, sources=json.loads(DATA.read_text())["sources"],
                gpu_info=smi(args.gpus, "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader").splitlines(),
                input_sha256={w: digest([r[2] for r in traces[w]]) for w in args.workloads},
                topology=subprocess.check_output(["nvidia-smi","topo","-m"],text=True),
                runtime="One model replica per GPU. Least pending prompt tokens for P, least active requests for D. "
                        "PD hands off the completed prompt over NCCL. DP uses the normal prefill-first scheduler. "
                        "First output becomes observable after D owns its KV. Open-loop engine commits, drain included.")
    seen = set()
    if archive.exists():
        old, trials = read_trials(archive)
        if any(old[k] != meta[k] for k in ("config","source_sha256","gpu_info","input_sha256","layouts")):
            raise ValueError("Existing output belongs to a different run")
        seen = {(t["mode"], t["repeat"], t["workload"], t["rate"]) for t in trials}
        meta["reference_tokens"] = old["reference_tokens"]
    else:
        if busy := occupants(args.gpus):
            raise RuntimeError(f"Selected GPUs occupied: {busy}")
        meta["reference_tokens"] = reference_probe(args.model, args.gpus[0], draft)
        with ZipFile(archive, "x", ZIP_DEFLATED) as z:
            z.writestr("metadata.json", json.dumps(meta))
    for rep in range(args.repeats):
        modes = list(layouts)
        rotation = rep % len(modes)
        cells = list(grid)
        Random(args.seed + rep).shuffle(cells)
        for mode in modes[rotation:] + modes[:rotation]:
            todo = [(w,r) for w,r in cells if (mode,rep,w,r) not in seen]
            if not todo:
                continue
            if busy := occupants(args.gpus):
                raise RuntimeError(f"Selected GPUs occupied: {busy}")
            print(f"loading {mode} repeat={rep+1} layout={layouts[mode]}", flush=True)
            with PDPool(args.model, *layouts[mode], cpu_sets=cpus, num_kvcache_blocks=args.blocks,
                        max_model_len=4608, max_num_batched_tokens=512, max_num_seqs=32, **draft) as pool:
                probe, checked, ties = verify_and_warm(pool, meta["reference_tokens"])
                if probe != meta["reference_tokens"]:
                    raise ValueError(f"Probe output mismatch on {mode}: {probe} != {meta['reference_tokens']}")
                for name, rate in todo:
                    rows = traces[name]
                    offsets = arrivals(name, rows, rate, args.seed + rep)
                    for attempt in range(5):
                        t = replay(pool, name, rows, offsets, rate, rep, mode, args.seed+rep)
                        if not t["new_graphs"]:
                            break
                        print(f"settling {mode} {name} {rate}: {t['new_graphs']} graphs", flush=True)
                    else:
                        raise RuntimeError("Graph count did not settle")
                    t.update(warm_retries=attempt, probe_tokens=probe, verified_kv=checked, probe_ties=ties,
                             workers={g: dict(blocks=st.blocks, bytes_per_token=st.bytes_per_token, **st.runtime)
                                      for g,st in pool.stages.items()})
                    with ZipFile(archive, "a", ZIP_DEFLATED, compresslevel=3) as z:
                        z.writestr(f"trial-{mode}-{rep}-{name}-{rate}.json", json.dumps(t,separators=(",",":")))
                    score = summarize([t], 500, 50, 100)
                    print(f"{mode} {name} {rate} repeat={rep+1} output={score['output_tok_s']:.0f} "
                          f"stream_goodput={score['goodput_req_s']:.2f}", flush=True)
    meta,trials = read_trials(archive)
    for filename, gap in (("summary.json",None),("streaming_slo.json",100)):
        out = write_report(args.output/filename, meta,trials,500,50,.9,gap)
        if not out["complete"]:
            raise RuntimeError("Incomplete allocation grid")
    print(f"{out['trial_count']} trials, {out['request_replays']} requests", flush=True)


if __name__ == "__main__":
    main()
