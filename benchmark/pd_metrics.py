from collections import defaultdict
from math import floor
from statistics import median


def distribution(values):
    values = sorted(values)
    if not values:
        return None

    def percentile(p):
        pos = (len(values) - 1) * p / 100
        i = floor(pos)
        return round(values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (pos - i), 3)

    return {"n": len(values), **{f"p{p}": percentile(p) for p in (50, 90, 95, 99)},
            "max": round(values[-1], 3)}


def summarize(trials, ttft_ms, tpot_ms, max_gap_ms=None):
    # 每个请求从计划到达时开始计时，包括上一轮 GPU 忙时等候进入引擎的时间。
    values = defaultdict(list)
    satisfied = finished = inputs = outputs = 0
    elapsed = sum(t["seconds"] for t in trials)
    gpu_count = trials[0]["gpu_count"]
    burst_gaps = []
    for trial in trials:
        rows = trial["requests"]
        long_rows = [r for r in rows if r["class"] == "long-burst"]
        burst_start = min((r["arrival_us"] for r in long_rows), default=None)
        burst_end = max((r["emits"][0][0] for r in long_rows), default=None)
        for r in rows:
            emits = r["emits"]  # [本轮提交时刻(us), 累计输出 token 数]
            first, last = emits[0][0], emits[-1][0]
            count = emits[-1][1]
            if (count != r["output_tokens"] or not r["arrival_us"] <= r["admit_us"] <= r["dispatch_us"] <= first
                    or emits[0][1] < 1 or any(b[0] < a[0] or b[1] <= a[1] for a, b in zip(emits, emits[1:]))):
                raise ValueError(f"Incomplete or out-of-order request {r['id']}")
            ttft = (first - r["arrival_us"]) / 1000
            tpot = (last - first) / (1000 * (count - 1)) if count > 1 else None
            gaps = [(b[0] - a[0]) / 1000 for a, b in zip(emits, emits[1:])]
            longest_gap = max(gaps, default=0.)
            values["arrival_to_admit_ms"].append((r["admit_us"] - r["arrival_us"]) / 1000)
            values["queue_to_first_prefill_ms"].append((r["dispatch_us"] - r["arrival_us"]) / 1000)
            values["ttft_ms"].append(ttft)
            values["e2e_ms"].append((last - r["arrival_us"]) / 1000)
            if tpot is not None:
                values["tpot_ms"].append(tpot)
            if gaps:
                values["inter_emit_ms"].extend(gaps)
                values["request_max_gap_ms"].append(longest_gap)
            if r["class"] == "decode-probe" and first < burst_start < last:
                # 只统计长 prompt 到达前已经出字的请求，窗口到最后一条长请求出首 token 为止。
                burst_gaps.append(max((b[0] - a[0]) / 1000 for a, b in zip(emits, emits[1:])
                                      if a[0] <= burst_end and b[0] >= burst_start))
            # 草稿可能一次提交多个 token，流式停顿按两次可见输出之间的实际间隔算。
            satisfied += (ttft <= ttft_ms and (tpot is None or tpot <= tpot_ms)
                          and (max_gap_ms is None or longest_gap <= max_gap_ms))
            finished += 1
            inputs += r["input_tokens"]
            outputs += count
    return {
        "repeats": len(trials), "completed": finished, "failed": 0, "seconds_total": elapsed,
        "input_tokens": inputs, "output_tokens": outputs, "request_per_s": finished / elapsed,
        "input_tok_s": inputs / elapsed, "output_tok_s": outputs / elapsed,
        "total_tok_s": (inputs + outputs) / elapsed, "output_tok_s_per_gpu": outputs / elapsed / gpu_count,
        "slo_hits": satisfied, "slo_attainment": satisfied / finished,
        "goodput_req_s": satisfied / elapsed, "goodput_req_s_per_gpu": satisfied / elapsed / gpu_count,
        **{k: distribution(v) for k, v in values.items()},
        "active_probe_max_gap_during_burst_ms": distribution(burst_gaps),
    }


def report(trials, ttft_ms, tpot_ms, attainment, max_gap_ms=None):
    groups = defaultdict(list)
    for trial in trials:
        groups[trial["workload"], trial["rate"], trial["mode"]].append(trial)
    cells, passed = [], {}
    for (workload, rate, mode), runs in groups.items():
        pooled = summarize(runs, ttft_ms, tpot_ms, max_gap_ms)
        individual = [summarize([t], ttft_ms, tpot_ms, max_gap_ms) for t in runs]
        pooled["by_repeat"] = [{"repeat": t["repeat"], **{k: m[k] for k in
                               ("output_tok_s", "goodput_req_s", "slo_attainment")}}
                               for t, m in zip(runs, individual)]
        pooled["output_tok_s_median"] = median(v["output_tok_s"] for v in individual)
        transport = [lane for t in runs for lane in t["lanes"]]
        pooled["engine"] = {k: sum(lane[k] for lane in transport) for k in (
            "prefill_steps", "decode_steps", "prefill_tokens_computed", "decode_steps_with_prefill_pending",
            "preemptions", "kv_p_to_d_bytes", "kv_d_to_p_bytes", "kv_p_to_d_stream_ms", "kv_d_to_p_stream_ms")}
        if transport and "draft_tokens_verified" in transport[0]:
            for key in ("verify_steps", "draft_tokens_verified", "draft_tokens_accepted", "decode_tokens"):
                pooled["engine"][key] = sum(lane[key] for lane in transport)
            verified = pooled["engine"]["draft_tokens_verified"]
            pooled["engine"]["draft_acceptance"] = pooled["engine"]["draft_tokens_accepted"] / verified if verified else None
        pooled["nvlink_bytes"] = {k: sum(t["nvlink_bytes"][k] for t in runs) for k in runs[0]["nvlink_bytes"]}
        pooled["gpu"] = {}
        energy = 0.
        for gpu in runs[0]["gpu"]:
            samples = [(t["samples"], t["gpu"][gpu]) for t in runs]
            info = {}
            for k in ("mean_gpu_util_pct", "mean_power_w"):
                vals = [(n, v[k]) for n, v in samples if n and v[k] is not None]
                info[k] = sum(n * x for n, x in vals) / sum(n for n, _ in vals) if vals else None
            info["peak_used_mib"] = max((v["peak_used_mib"] for _, v in samples if v["peak_used_mib"] is not None),
                                        default=None)
            pooled["gpu"][gpu] = info
            watts = [t["gpu"][gpu]["mean_power_w"] for t in runs]
            if energy is not None and all(w is not None for w in watts):
                energy += sum(t["seconds"] * w for t, w in zip(runs, watts))
            else:
                energy = None
        pooled["estimated_joules_per_output_token"] = energy / pooled["output_tokens"] if energy is not None else None
        cells.append(dict(workload=workload, offered_rps=rate, mode=mode, **pooled))
        if rate != "burst":
            layout = passed.setdefault(workload, {}).setdefault(mode, [])
            if all(v["slo_attainment"] >= attainment for v in individual):
                layout.append(float(rate))
    slo = {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms, "required_fraction_each_repeat": attainment}
    if max_gap_ms is not None:
        slo["max_gap_ms"] = max_gap_ms
    return {"slo": slo,
            "highest_tested_rps_meeting_slo": {w: {m: max(r) if r else None for m, r in modes.items()}
                                               for w, modes in passed.items()}, "cells": cells}
