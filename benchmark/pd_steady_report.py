import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


US = 1_000_000
PARTS = ("arrival_to_admit", "p_queue", "p_service", "wait_to_submit_kv", "kv_submit_to_ack",
         "ack_to_first", "wait_to_next_dispatch", "scheduled_after_first", "between_decode_steps")
GAP_PARTS = ("gap_other_prefill", "gap_other_decode", "gap_between_batches", "gap_own_step", "gap_commit")


def stats(values):
    if not len(values):
        return None
    p50, p95, p99 = np.percentile(values, (50,95,99))
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(p50), p95=float(p95), p99=float(p99))


def longest_gap_parts(emits, batches):
    # 取每条请求停顿最久的两个 token，看它在同一张卡上等了多少轮别人的 P 或 D。
    rows = np.arange(len(emits))
    pos = np.diff(emits, axis=1).argmax(axis=1)
    begin, end = emits[rows,pos], emits[rows,pos+1]
    starts, replies = batches[:,2], batches[:,3]
    service = replies-starts
    own = np.searchsorted(replies,end,side="right")-1
    assert (own>=0).all() and (starts[own]>=begin).all()
    # 首 token 可能在其他 D batch 运行途中发布，那一轮只计发布后的部分。
    initial = np.maximum(0,np.searchsorted(starts,begin,side="right")-1)
    partial = np.clip(begin-starts[initial],0,service[initial])
    def other_steps(mask):
        prefix = np.r_[0,np.cumsum(service*mask)]
        return prefix[own]-prefix[initial]-partial*mask[initial]
    prefill = other_steps(batches[:,1]==1)
    decode = other_steps(batches[:,1]==0)
    parts = (prefill, decode, starts[own]-begin-prefill-decode, service[own], end-replies[own])
    assert all((part>=0).all() for part in parts) and np.array_equal(sum(parts),end-begin)
    return parts


def request_times(req, emit, steps, kv, split, cutoff):
    # 全部时间点都来自同一协调进程的单调时钟。一次前向的往返耗时可以同时属于多条请求。
    first, last = emit[:,0].astype(np.int64), emit[:,-1].astype(np.int64)
    arrival, admit, dispatch = req[:,:3].T
    assert np.all((arrival<=admit) & (admit<=dispatch) & (dispatch<=first))
    p_done, submit = (req[:,3],req[:,4]) if split else (first,first)
    ack = first.copy()
    gpu_kv = {}
    if split:
        acks = []
        for sender, name in ((1,"kv_stream_send"),(0,"kv_stream_recv")):
            rows = kv[kv[:,2]==sender]
            rows = rows[np.argsort(rows[:,0])]
            assert np.array_equal(rows[:,0],np.arange(len(req)))
            assert np.array_equal(rows[:,1],req[:,5 if sender else 6])
            gpu_kv[name] = rows[:,4]/1000.
            acks.append(rows[:,3])
        ack = np.maximum(*acks)
    total_step = np.zeros(len(req),np.int64)
    step_after_stop = np.zeros(len(req),np.int64)
    first_dispatch = np.zeros(len(req),np.int64)
    gap = {key: np.zeros(len(req),np.int64) for key in GAP_PARTS}
    # 每个后续 token 对应同一张卡刚返回的一次前向；核对每个 D step 的请求数。
    for g in np.unique(req[:,6]):
        ids = np.flatnonzero(req[:,6]==g)
        batches = steps[steps[:,0]==g]
        assert np.all(batches[1:,2]>=batches[:-1,3])
        counts = np.zeros(len(batches),np.int64)
        for group in np.array_split(ids,max(1,(len(ids)+255)//256)):
            times = emit[group,1:]
            at = np.searchsorted(batches[:,3],times,side="right")-1
            assert at.min()>=0
            assert np.all(times<=np.r_[batches[1:,2],np.iinfo(np.int64).max][at])
            first_dispatch[group] = batches[at[:,0],2]
            # 同一请求在同一轮提交的 token 只占一次前向耗时、一个 batch 名额。
            first_in_step = np.ones_like(at,dtype=bool)
            first_in_step[:,1:] = at[:,1:]!=at[:,:-1]
            total_step[group] = ((batches[:,3]-batches[:,2])[at]*first_in_step).sum(axis=1)
            step_after_stop[group] = (np.maximum(0,batches[at,3]-np.maximum(batches[at,2],cutoff))*first_in_step).sum(axis=1)
            counts += np.bincount(at[first_in_step],minlength=len(batches))
        decode = batches[:,1]==0
        assert np.array_equal(counts[decode],batches[decode,4])
        for key,part in zip(GAP_PARTS,longest_gap_parts(emit[ids],batches),strict=True):
            gap[key][ids] = part
    components = (admit-arrival,dispatch-admit,p_done-dispatch,submit-p_done,ack-submit,
                  first-ack,first_dispatch-first,total_step,last-first_dispatch-total_step)
    assert all(np.all(x>=0) for x in components)
    assert np.array_equal(sum(components),last-arrival)
    # 排空只从停流时刻起算。跨过停流边界的排队或前向只保留后半段。
    edges = (arrival,admit,dispatch,p_done,submit,ack,first,first_dispatch)
    residual = [np.maximum(0,b-np.maximum(a,cutoff)) for a,b in zip(edges,edges[1:])]
    residual += [step_after_stop,np.maximum(0,last-np.maximum(first_dispatch,cutoff))-step_after_stop]
    assert all(np.all(x>=0) for x in residual)
    assert np.array_equal(sum(residual),np.maximum(0,last-np.maximum(arrival,cutoff)))
    metrics = {k:v/1000. for k,v in zip(PARTS,components,strict=True)}
    metrics.update({k:v/1000. for k,v in gap.items()})
    metrics.update(gpu_kv)
    metrics.update(ttft=(first-arrival)/1000., tpot=(last-first)/511000.,
                   worst_gap=np.diff(emit,axis=1).max(axis=1)/1000., e2e=(last-arrival)/1000.,
                   decode=(last-first)/1000., first_decode_gap=(emit[:,1]-first)/1000.)
    return metrics,dict(zip(PARTS,residual,strict=True))


def phase_data(begin,end,req,emit,hit,steps,kv,state,hw,meta):
    seconds = (end-begin)/US
    if seconds<=0:
        return None
    def inside(t):
        return (t>=begin)&(t<end)
    arrived, done = req[:,0],emit[:,-1]
    live_begin = int((arrived<begin).sum()-(done<begin).sum())
    live_end = int((arrived<end).sum()-(done<end).sum())
    result = dict(seconds=seconds, requests_in=int(inside(arrived).sum()), requests_out=int(inside(done).sum()),
                  good_out=int((inside(done)&hit).sum()), tokens=int(inside(emit).sum()),
                  live_start=live_begin,live_end=live_end,
                  avg_live=float(np.maximum(0,np.minimum(done,end)-np.maximum(arrived,begin)).sum()/(end-begin)))
    boundaries=np.arange(begin,end,US,dtype=np.int64)
    live=np.searchsorted(arrived,boundaries)-np.searchsorted(np.sort(done),boundaries)
    result["live_each_second"]=live.tolist()
    result["live_slope_req_s"]=float(np.polyfit(np.arange(len(live)),live,1)[0]) if len(live)>=3 else None
    lanes = {}
    devices = meta["p_gpus"] + meta["d_gpus"]
    for g in devices:
        s = state[state[:,1]==g]
        # 对调度快照按持续时间加权。无在途任务是协调器的状态，GPU 是否在执行由 Nsight 单独核对。
        w = np.maximum(0,np.minimum(np.r_[s[1:,0],end],end)-np.maximum(s[:,0],begin))
        h = hw[(hw[:,1]==g)&inside(hw[:,0])]
        psteps = steps[(steps[:,0]==g)&inside(steps[:,2])&(steps[:,1]==1)]
        dsteps = steps[(steps[:,0]==g)&inside(steps[:,2])&(steps[:,1]==0)]
        free = (s[:,7]==0)&(s[:,9]==0)
        lanes[g] = dict(state_us=int(w.sum()), idle_host_us=int(w[free].sum()),
                        idle_host_p_waiting_us=int(w[free&(s[:,2]>0)].sum()),
                        p_queue_us=int(w[s[:,2]>0].sum()), p_queue_without_blocks_us=int(w[(s[:,2]>0)&(s[:,6]==0)].sum()),
                        free_block_us=int(w @ s[:,6]),
                        p_waiting_req_us=int(w @ s[:,2]),running_req_us=int(w @ s[:,3]),
                        ready_req_us=int(w @ s[:,4]),reserved_block_us=int(w @ s[:,5]),
                        hw_samples=len(h),util_sum=float(h[:,2].sum()),power_sum=float(h[:,4].sum()),
                        p_steps=len(psteps),p_requests=int(psteps[:,4].sum()),p_tokens=int(psteps[:,5].sum()),
                        d_steps=len(dsteps),d_requests=int(dsteps[:,4].sum()),d_positions=int(dsteps[:,5].sum()),
                        p_roundtrip_us=int((psteps[:,3]-psteps[:,2]).sum()),
                        d_roundtrip_us=int((dsteps[:,3]-dsteps[:,2]).sum()))
    result["lanes"] = lanes
    if meta["p_gpus"]:
        snapshots=state.reshape(-1,len(devices),state.shape[1])
        times=snapshots[:,0,0]
        assert np.all(snapshots[:,:,0]==times[:,None])
        weights=np.maximum(0,np.minimum(np.r_[times[1:],end],end)-np.maximum(times,begin))
        ds=snapshots[:,np.isin(snapshots[0,:,1],meta["d_gpus"]),:]
        ps=snapshots[:,np.isin(snapshots[0,:,1],meta["p_gpus"]),:]
        ready=snapshots[:,:,4].sum(axis=1)>0
        # 用各张 D 卡实际的 KV 块数判断能否接收。旧记录是每卡 256 块，每请求预留 3 块。
        capacity=[meta.get("kv_blocks",{}).get(str(g),256) for g in ds[0,:,1]]
        has_space=ds[:,:,5]+meta.get("blocks_per_request",3)<=capacity
        receiver=has_space & (ds[:,:,9]==0)
        sender=(ps[:,:,4]>0) & (ps[:,:,9]==0)
        result["handoff_state_us"]=dict(sampled=int(weights.sum()), ready=int(weights[ready].sum()),
            ready_no_capacity=int(weights[ready & ~has_space.any(axis=1)].sum()),
            ready_receivers_busy=int(weights[ready & has_space.any(axis=1) & ~receiver.any(axis=1)].sum()),
            ready_senders_busy=int(weights[ready & receiver.any(axis=1) & ~sender.any(axis=1)].sum()),
            all_decode_at_least_32=int(weights[(ds[:,:,3]>=32).all(axis=1)].sum()))
    result["kv_bytes"] = int(np.sum(inside(kv[:,3])&(kv[:,2]==0)))*128*meta["bytes_per_token"]
    return result


def read_case(path):
    with np.load(path,allow_pickle=False) as z:
        m = json.loads(str(z["meta"]))
        req,emit,steps,kv,state,hw = (z[k] for k in ("req","emits","steps","kv","state","hw"))
    assert m["new_graphs"]==0 and emit.shape==(m["requests"],512) and (emit>=0).all()
    assert np.all(emit[:,1:]>=emit[:,:-1]) and np.all(req[1:,0]>=req[:-1,0])
    assert set(hw[:,1])==set(m["p_gpus"]+m["d_gpus"]) and np.isfinite(hw).all()
    split = bool(m["p_gpus"])
    payload = sum(v["kv_p_to_d_bytes"] for v in m["lanes"].values())
    # 旧记录只传 8B 主模型；新记录由 worker 给出主模型和草稿模型的实际 KV 大小。
    m.setdefault("bytes_per_token",147456)
    if split:
        assert m["preflight_kv"]==12 and m["bytes_per_token"]>0 and payload==128*len(req)*m["bytes_per_token"]
        assert np.all(np.isin(req[:,5],m["p_gpus"])&np.isin(req[:,6],m["d_gpus"]))
    else:
        assert np.all(req[:,5]==req[:,6])
    decode = steps[steps[:,1]==0]
    verified = int((decode[:,5]-decode[:,4]).sum())
    accepted = sum(v.get("draft_tokens_accepted",0) for v in m["lanes"].values())
    assert 0<=accepted<=verified
    if "decode_tokens" in next(iter(m["lanes"].values())):
        # 本负载忽略 EOS；一次 D 调度提交一枚主模型 token 和接受的草稿。
        assert sum(v["decode_tokens"] for v in m["lanes"].values())==int(decode[:,4].sum())+accepted
        assert sum(v["draft_tokens_verified"] for v in m["lanes"].values())==verified
    start,stop,horizon = np.cumsum(m["segments_s"],dtype=np.int64)*US
    costs,residual = request_times(req,emit,steps,kv,split,horizon)
    hit = (costs["ttft"]<=500)&(costs["tpot"]<=50)&(costs["worst_gap"]<=100)
    end = max(horizon,int(emit[:,-1].max())+1)
    phases = {name:phase_data(int(a),int(b),req,emit,hit,steps,kv,state,hw,m)
              for name,a,b in (("lead",0,start),("window",start,stop),("guard",stop,horizon),
                               ("drain",horizon,end),("all",0,end))}
    assert sum(p["tokens"] for k,p in phases.items() if k!="all" and p)==emit.size
    assert phases["all"]["requests_out"]==len(req) and phases["all"]["live_end"]==0
    last_request = int(np.argmax(emit[:,-1]))
    drain_n = int((emit[:,-1]>=horizon).sum())
    assert drain_n == (phases["drain"]["live_start"] if phases["drain"] else 0)
    cohort = (req[:,0]>=start)&(req[:,0]<stop)
    assert np.sum(cohort)==m["rate"]*m["segments_s"][1]
    obs = {k:v[cohort] for k,v in costs.items()}
    gaps=np.diff(emit[cohort],axis=1)
    causes=dict(ttft_over_500=int((obs["ttft"]>500).sum()),tpot_over_50=int((obs["tpot"]>50).sum()),
                gap_over_100=int((obs["worst_gap"]>100).sum()),first_decode_gap_over_100=int((gaps[:,0]>100_000).sum()),
                later_gap_over_100=int((gaps[:,1:].max(axis=1)>100_000).sum()),
                max_gap_at_first_decode=int((np.argmax(gaps,axis=1)==0).sum()))
    out = dict(mode=m["mode"],rate=m["rate"],repeat=m["repeat"],requests=len(req),plan_hash=m["plan_hash"],
               profiled=m["profiled"],segments_s=m["segments_s"],p_gpus=m["p_gpus"],d_gpus=m["d_gpus"],
               draft_verified=verified,draft_accepted=accepted,
               preflight_kv=m["preflight_kv"],new_graphs=m["new_graphs"],warm_retries=m["warm_retries"],
               phases=phases, cohort_n=int(cohort.sum()),cohort_hits=int(hit[cohort].sum()),cohort_counts=causes,
               cohort_finishes_in_drain=int((cohort&(emit[:,-1]>=horizon)).sum()),
               drain_n=drain_n,drain_stage_us={k:int(v.sum()) for k,v in residual.items()},
               last_request_in_drain_ms={k:float(v[last_request])/1000 for k,v in residual.items()},
               last_prefill_done_s=float((req[:,3] if split else emit[:,0]).max()/US),
               last_handoff_or_first_s=float(emit[:,0].max()/US),last_arrival_s=float(req[-1,0]/US),
               preemptions=sum(v["preemptions"] for v in m["lanes"].values()),nvlink_bytes=m["nvlink_bytes"])
    return out,obs


def merge(cases):
    r = [c for c,_ in cases]
    first = r[0]
    def summarize_phase(name):
        ps = [v["phases"][name] for v in r if v["phases"][name]]
        if not ps:
            return None
        duration = sum(v["seconds"] for v in ps)
        slopes=[p["live_slope_req_s"] for p in ps if p["live_slope_req_s"] is not None]
        phase = dict(seconds=duration,output_tok_s=sum(v["tokens"] for v in ps)/duration,
                     completed_s=sum(v["requests_out"] for v in ps)/duration,
                     goodput_s=sum(v["good_out"] for v in ps)/duration,
                     mean_live=sum(v["avg_live"]*v["seconds"] for v in ps)/duration,
                     mean_live_start=np.mean([v["live_start"] for v in ps]).item(),
                     mean_live_end=np.mean([v["live_end"] for v in ps]).item(),
                     live_slope_req_s=float(np.mean(slopes)) if slopes else None,
                     kv_gib_s=sum(v["kv_bytes"] for v in ps)/duration/2**30)
        groups = {"P":first["p_gpus"],"D":first["d_gpus"]} if first["p_gpus"] else {"DP":first["d_gpus"]}
        if first["p_gpus"]:
            handoff={k:sum(p["handoff_state_us"][k] for p in ps) for k in ps[0]["handoff_state_us"]}
            phase["handoff_state"]={k+"_pct":(100*v/handoff["sampled"] if handoff["sampled"] else None)
                                    for k,v in handoff.items() if k!="sampled"}
        for role,gpus in groups.items():
            ls = [p["lanes"][g] for p in ps for g in gpus]
            def ratio(n,d,scale=1):
                denominator=sum(l[d] for l in ls)
                return sum(l[n] for l in ls)*scale/denominator if denominator else None
            phase[role]=dict(util_pct=ratio("util_sum","hw_samples"),mean_power_w=ratio("power_sum","hw_samples"),
                             p_batch=ratio("p_requests","p_steps"),d_batch=ratio("d_requests","d_steps"),
                             p_positions=ratio("p_tokens","p_steps"),d_positions=ratio("d_positions","d_steps"),
                             p_host_step_ms=ratio("p_roundtrip_us","p_steps",.001),
                             d_host_step_ms=ratio("d_roundtrip_us","d_steps",.001),
                             host_idle_pct=ratio("idle_host_us","state_us",100),
                             waiting_while_host_idle_pct=ratio("idle_host_p_waiting_us","idle_host_us",100),
                             p_queue_without_blocks_pct_of_queue_time=ratio("p_queue_without_blocks_us","p_queue_us",100),
                             mean_waiting=ratio("p_waiting_req_us","state_us"),mean_running=ratio("running_req_us","state_us"),
                             mean_ready=ratio("ready_req_us","state_us"),mean_reserved_blocks=ratio("reserved_block_us","state_us"),
                             mean_free_physical_blocks=ratio("free_block_us","state_us"))
        return phase
    obs={k:stats(np.concatenate([v[k] for _,v in cases])) for k in cases[0][1]}
    stalled={k:stats(np.concatenate([v[k][v["worst_gap"]>100] for _,v in cases])) for k in ("worst_gap",*GAP_PARTS)}
    drain_n=sum(c["drain_n"] for c in r)
    verified=sum(c["draft_verified"] for c in r)
    accepted=sum(c["draft_accepted"] for c in r)
    return dict(mode=first["mode"],rate=first["rate"],repeats=len(r),request_replays=sum(x["requests"] for x in r),
                draft_verified=verified,draft_accepted=accepted,draft_acceptance=accepted/verified if verified else None,
                window=summarize_phase("window"),drain=summarize_phase("drain"),whole=summarize_phase("all"),
                drain_seconds=[c["phases"]["drain"]["seconds"] if c["phases"]["drain"] else 0 for c in r],
                remaining_ms_per_request_at_stop={k:sum(c["drain_stage_us"][k] for c in r)/(1000*drain_n)
                                                  if drain_n else 0 for k in PARTS},
                last_request_in_drain_ms=[c["last_request_in_drain_ms"] for c in r],
                wait_at_stop=[c["phases"]["drain"]["live_start"] if c["phases"]["drain"] else 0 for c in r],
                last_first_after_stop_s=[max(0.,c["last_handoff_or_first_s"]-sum(c["segments_s"])) for c in r],
                arrival_cohort_n=sum(c["cohort_n"] for c in r),
                attainment=sum(c["cohort_hits"] for c in r)/sum(c["cohort_n"] for c in r),
                cohort_counts={k:sum(c["cohort_counts"][k] for c in r) for k in first["cohort_counts"]},
                cohort_finishes_in_drain=sum(c["cohort_finishes_in_drain"] for c in r),
                preemptions=sum(c["preemptions"] for c in r),cohort_ms=obs,stalled_request_worst_gap_ms=stalled,
                by_repeat=[dict(rep=c["repeat"],goodput=c["phases"]["window"]["good_out"]/c["segments_s"][1],
                                output_tok_s=c["phases"]["window"]["tokens"]/c["segments_s"][1],
                                queue_delta=c["phases"]["window"]["live_end"]-c["phases"]["window"]["live_start"],
                                live_slope_req_s=c["phases"]["window"]["live_slope_req_s"],
                                attainment=c["cohort_hits"]/c["cohort_n"]) for c in r])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--complete",action="store_true")
    parser.add_argument("--first-repeats",type=int)
    args=parser.parse_args()
    groups=defaultdict(list)
    plans={}
    for file in sorted(args.directory.glob("*.npz")):
        case,values=read_case(file)
        if args.first_repeats is not None and case["repeat"]>=args.first_repeats:
            continue
        key=case["rate"],case["repeat"]
        assert plans.setdefault(key,case["plan_hash"])==case["plan_hash"]
        groups[case["mode"],case["rate"]].append((case,values))
    if not groups:
        raise ValueError("No completed cases")
    if args.complete:
        assert set(groups)=={(m,r) for m in ("dp8","4p4d","3p5d") for r in (8,12,16,20,48)}
        assert all(sorted(c["repeat"] for c,_ in runs)==[0,1,2] for runs in groups.values())
        assert all(c["segments_s"]==[20,120,10] and not c["profiled"] for runs in groups.values() for c,_ in runs)
    cells=[merge(v) for v in groups.values()]
    out=dict(trials=sum(c["repeats"] for c in cells),requests=sum(c["request_replays"] for c in cells),
             plan_hashes={f"{r}-{rep+1}":v for (r,rep),v in plans.items()},cells=cells)
    tmp=args.out.with_suffix(".partial")
    tmp.write_text(json.dumps(out,indent=2,allow_nan=False))
    tmp.replace(args.out)
    print(f"Validated {out['trials']} replays, {out['requests']} requests",flush=True)


if __name__=="__main__":
    main()
