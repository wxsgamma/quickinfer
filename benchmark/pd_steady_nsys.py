import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np


def capture():
    import torch
    import pd_steady

    original = pd_steady.replay
    def timed(*args, **kwargs):
        # CUPTI 在开始抓取时会停顿；先启动抓取，再让 replay 建立到达时钟。
        with torch.cuda.nvtx.range("QI_FULL"):
            return original(*args, **kwargs)
    pd_steady.replay = timed
    sys.argv.pop(1)
    if "--capture" not in sys.argv:
        sys.argv.append("--capture")
    pd_steady.main()


def union(spans):
    result = []
    for left,right in sorted(spans):
        if right<=left:
            continue
        if result and left<=result[-1][1]:
            result[-1]=(result[-1][0],max(right,result[-1][1]))
        else:
            result.append((left,right))
    return result


def intersection(a,b):
    i=j=total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:
            i+=1
        else:
            j+=1
    return total


def duration(spans):
    return sum(b-a for a,b in spans)


def lookup(index, tid, stamp):
    if tid not in index:
        return None
    starts, ranges=index[tid]
    k=bisect_right(starts,stamp)-1
    if k>=0 and stamp<ranges[k][1]:
        return ranges[k]


def make_index(by_tid):
    result={}
    for tid,rows in by_tid.items():
        rows.sort()
        result[tid]=([r[0] for r in rows],rows)
    return result


def profile(sqlite_path, raw_path):
    with np.load(raw_path,allow_pickle=False) as z:
        meta=json.loads(str(z["meta"]))
        state=z["state"]
        coordinator_steps,req,kv=z["steps"],z["req"],z["kv"]
        last=int(z["emits"][:,-1].max()+1)
    assert meta["profiled"]
    workers={int(pid):int(g) for g,pid in meta["workers"].items()}
    db=sqlite3.connect(f"file:{sqlite_path.resolve().as_posix()}?mode=ro",uri=True)
    strings=dict(db.execute("select id,value from StringIds"))
    pidmap={glob:workers[pid] for glob,pid in db.execute("select globalPid,pid from PROCESSES") if pid in workers}
    uuids=dict(db.execute("select id,uuid from TARGET_INFO_GPU"))
    expected=json.loads((raw_path.parent/"config.json").read_text())["gpu_info"]
    assert all(uuids[int(line.split(",")[0])].lower() in line.lower() for line in expected)
    labels="select n.start,n.end,n.globalTid,coalesce(n.text,s.value) as label from NVTX_EVENTS n left join StringIds s on s.id=n.textId "
    markers=[(a,int(label.split()[1])) for a,_,_,label in db.execute(labels+"where label like 'QI_SYNC %'")]
    delta=np.asarray([a-t*1000 for a,t in markers],dtype=np.int64)
    assert len(delta)>=5
    origin=int(np.median(delta))
    outer="QI_FULL"
    capture_row=db.execute(labels+"where label=? and n.end is not null order by n.start limit 1",(outer,)).fetchone()
    if capture_row is None:
        outer="QI_CAPTURE"
        capture_row=db.execute(labels+"where label=? and n.end is not null order by n.start limit 1",(outer,)).fetchone()
    cap_a,cap_b,_,_=capture_row
    warm,window,guard=meta["segments_s"]
    stop=(warm+window)*1_000_000
    cutoff=stop+guard*1_000_000
    wanted={"flow":(stop-8_000_000,stop),"last_feed":(cutoff-8_000_000,cutoff),"drain":(cutoff,min(last,cutoff+6_000_000))}
    bands={k:(max(cap_a,origin+a*1000),min(cap_b,origin+b*1000)) for k,(a,b) in wanted.items()}
    bands={k:v for k,v in bands.items() if v[1]>v[0]}
    lo=min(v[0] for v in bands.values())-1_000_000_000
    hi=max(v[1] for v in bands.values())+1_000_000_000
    roots, nested=defaultdict(list),defaultdict(list)
    root_info,host_parts=[],defaultdict(lambda:defaultdict(int))
    for a,b,tid,label in db.execute(labels+"where n.end>=? and n.start<=? and label like 'QI_%'",(lo,hi)):
        if label.startswith("QI_STEP "):
            _,stage,g,n=label.split()
            idx=len(root_info)
            root_info.append((a,b,int(g),stage,int(n)))
            roots[tid].append((a,b,idx))
        elif label in ("QI_INPUT","QI_MODEL","QI_SAMPLE","QI_DRAFT","QI_VERIFY","QI_KV_SEND","QI_KV_RECV"):
            nested[tid].append((a,b,label))
    roots=make_index(roots)
    nested=make_index(nested)
    host_by_gpu={g:coordinator_steps[coordinator_steps[:,0]==g] for g in range(8)}
    ipc={}
    for idx,(x,y,g,stage,n) in enumerate(root_info):
        s=host_by_gpu[g]
        j=np.searchsorted(s[:,2],((x+y)//2-origin)/1000,side="right")-1
        if j>=0 and s[j,1]==(stage=="P") and s[j,4]==n and (x+y)//2<=origin+s[j,3]*1000:
            ipc[idx]=(x-origin-int(s[j,2])*1000,origin+int(s[j,3])*1000-y)
    transfer_by_gpu={}
    for g in range(8):
        rows=kv[kv[:,1]==g]
        # 一张卡每次只允许一笔交接。按协调器的提交和双端回报给 NCCL kernel 对上请求。
        rows=rows[np.argsort(req[rows[:,0],4])]
        transfer_by_gpu[g]=(rows,origin+req[rows[:,0],4]*1000,origin+rows[:,3]*1000)
    for tid,(_,items) in nested.items():
        for a,b,label in items:
            root=lookup(roots,tid,a)
            if root is not None:
                host_parts[root[2]][label]+=b-a
    events,needed=[],set()
    queries=(("KERNEL","shortName"),("GRAPH_TRACE","0"),("MEMCPY","copyKind"),("MEMSET","0"))
    for name,extra in queries:
        table="CUPTI_ACTIVITY_KIND_"+name
        for a,b,g,pid,corr,detail in db.execute(f"select start,end,deviceId,globalPid,correlationId,{extra} from {table} where end>=? and start<=?",(lo,hi)):
            if pid not in pidmap:
                continue
            assert pidmap[pid]==g
            events.append((a,b,g,pid,corr,name,detail))
            needed.add((pid,corr))
    launches={}
    for a,tid,corr in db.execute("select start,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<=?",(lo-1_000_000_000,hi)):
        key=tid>>24<<24,corr
        if key in needed:
            launches[key]=(a,tid)
    by_gpu=defaultdict(list)
    gpu_steps=defaultdict(lambda:defaultdict(list))
    transfer_kernels=defaultdict(list)
    no_launch=0
    for a,b,g,pid,corr,name,detail in events:
        launch=launches.get((pid,corr))
        root=inner=None
        if launch:
            stamp,tid=launch
            root=lookup(roots,tid,stamp)
            inner=lookup(nested,tid,stamp)
        else:
            no_launch+=1
        kernel=strings.get(detail,"") if name=="KERNEL" else ""
        if "nccl" in kernel.lower() or (inner and inner[2] in ("QI_KV_SEND","QI_KV_RECV")):
            kind="kv"
        elif inner:
            kind={"QI_INPUT":"input","QI_MODEL":"model","QI_SAMPLE":"sample","QI_DRAFT":"draft","QI_VERIFY":"verify"}[inner[2]]
        elif name=="GRAPH_TRACE":
            kind="model"
        elif root and name=="MEMCPY" and detail==2:
            kind="return_copy"
        else:
            kind="other"
        by_gpu[g].append((a,b,kind,name,kernel))
        if "nccl" in kernel.lower():
            rows,starts,acks=transfer_by_gpu[g]
            j=np.searchsorted(starts,(a+b)//2,side="right")-1
            if j>=0 and (a+b)//2<=acks[j]:
                transfer_kernels[(g,int(rows[j,0]))].append((a,b))
        if root and kind!="kv":
            gpu_steps[root[2]][kind].append((a,b))
    assert set(by_gpu)==set(range(8))
    groups={"P":meta["p_gpus"],"D":meta["d_gpus"]} if meta["p_gpus"] else {"DP":meta["d_gpus"]}
    result={}
    for band,(a,b) in bands.items():
        record=dict(start_s=(a-origin)/1e9,seconds=(b-a)/1e9,roles={},steps={},kv_per_transfer={})
        for role,gpus in groups.items():
            totals=defaultdict(int)
            stage_ns=defaultdict(int)
            counts=defaultdict(int)
            top_kernel=defaultdict(int)
            for g in gpus:
                spans=defaultdict(list)
                for x,y,kind,name,kernel in by_gpu[g]:
                    x,y=max(x,a),min(y,b)
                    if y<=x:
                        continue
                    spans[kind].append((x,y))
                    counts[name]+=1
                    if kernel:
                        top_kernel[kernel]+=y-x
                compute=union([i for k,vals in spans.items() if k!="kv" for i in vals])
                comm=union(spans["kv"])
                busy=union(compute+comm)
                idle=[]
                prev=a
                for x,y in busy:
                    if prev<x:
                        idle.append((prev,x))
                    prev=y
                if prev<b:
                    idle.append((prev,b))
                sg=state[state[:,1]==g]
                snap=sg[:,0]*1000+origin
                def observed(column):
                    return union((max(a,int(x)),min(b,int(y))) for x,y,v in zip(snap,snap[1:],sg[:-1,column]) if v>0 and x<b and y>a)
                totals["idle"]+=duration(idle)
                totals["pending_p_while_idle"]+=intersection(idle,observed(2))
                totals["pending_handoff_while_idle"]+=intersection(idle,observed(4))
                overlap=intersection(compute,comm)
                totals["kv_only"]+=duration(comm)-overlap
                totals["compute_only"]+=duration(compute)-overlap
                totals["kv_and_compute"]+=overlap
                for kind,vals in spans.items():
                    stage_ns[kind]+=duration(union(vals))
            elapsed=(b-a)*len(gpus)
            assert sum(totals[k] for k in ("idle","compute_only","kv_only","kv_and_compute"))==elapsed
            display={k+"_pct":100*v/elapsed for k,v in totals.items()}
            display["waiting_fraction_of_gpu_idle_pct"]=100*totals["pending_p_while_idle"]/totals["idle"] if totals["idle"] else None
            display.update(active_by_stage_pct={k:100*v/elapsed for k,v in stage_ns.items()},activity_counts=dict(counts),
                           top_kernel_gpu_ms={k:round(v/1e6,3) for k,v in sorted(top_kernel.items(),key=lambda i:-i[1])[:7]})
            record["roles"][role]=display
            kv_parts=[]
            for g in gpus:
                rows,starts,acks=transfer_by_gpu[g]
                for j in np.flatnonzero((starts>=a)&(acks<=b)):
                    kernels=union(transfer_kernels[(g,int(rows[j,0]))])
                    if not kernels:
                        continue
                    first_k,last_k=kernels[0][0],kernels[-1][1]
                    kv_parts.append((first_k-starts[j],last_k-first_k,acks[j]-last_k,
                                     duration(kernels),rows[j,4]*1000,acks[j]-starts[j]))
            if kv_parts:
                cols=("submit_to_first_nccl","first_to_last_nccl","last_nccl_to_ack","nccl_kernel_busy",
                      "cuda_stream_event","submit_to_ack")
                means=np.mean(kv_parts,axis=0)/1e6
                record["kv_per_transfer"][role]=dict(n=len(kv_parts),ms=dict(zip(cols,means.tolist(),strict=True)))
            for stage in ("P","D"):
                ids=[i for i,(x,y,g,k,n) in enumerate(root_info) if g in gpus and k==stage and x>=a and y<=b]
                if not ids:
                    continue
                size=len(ids)
                gpu_total=defaultdict(int)
                cpu_total=defaultdict(int)
                for i in ids:
                    for kind,vals in gpu_steps[i].items():
                        gpu_total[kind]+=duration(union(vals))
                    for kind,dt in host_parts[i].items():
                        cpu_total[kind]+=dt
                wall=sum(root_info[i][1]-root_info[i][0] for i in ids)
                cpu_total["unmarked_including_to_host_sync"]=wall-sum(cpu_total.values())
                assert cpu_total["unmarked_including_to_host_sync"]>=0
                ipc_rows=[ipc[i] for i in ids if i in ipc]
                record["steps"][role+"_"+stage]=dict(steps=size,batch=np.mean([root_info[i][4] for i in ids]).item(),
                    cpu_wall_ms=wall/size/1e6,cpu_ms={k:v/size/1e6 for k,v in cpu_total.items()},
                    gpu_ms={k:v/size/1e6 for k,v in gpu_total.items()},matched_host_steps=len(ipc_rows),
                    dispatch_to_worker_ms=np.mean(ipc_rows,axis=0).tolist()[0]/1e6 if ipc_rows else None,
                    worker_to_reply_ms=np.mean(ipc_rows,axis=0).tolist()[1]/1e6 if ipc_rows else None)
        result[band]=record
    db.close()
    return dict(mode=meta["mode"],rate=meta["rate"],plan_hash=meta["plan_hash"],capture_range=outer,
                physical_gpu_uuids=uuids,clock_anchors=len(delta),clock_residual_p95_us=float(np.percentile(abs(delta-origin),95)/1000),
                matched_gpu_events=len(events)-no_launch,total_gpu_events=len(events),windows=result)


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="capture":
        capture()
    else:
        parser=argparse.ArgumentParser()
        parser.add_argument("database",type=Path)
        parser.add_argument("raw",type=Path)
        parser.add_argument("--out",type=Path,required=True)
        args=parser.parse_args()
        data=profile(args.database,args.raw)
        args.out.write_text(json.dumps(data,indent=2,allow_nan=False))
        print(data["mode"],data["rate"],data["capture_range"],data["matched_gpu_events"],data["total_gpu_events"],list(data["windows"]))
