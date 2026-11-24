import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = {"dp8":("8 replicas","#49576b"),"4p4d":("4P + 4D","#126fad"),"3p5d":("3P + 5D","#c25524")}


def line(ax, x, center, repeats, label, color):
    low,high=np.min(repeats,axis=1),np.max(repeats,axis=1)
    ax.plot(x,center,color=color,marker="o",lw=2.1,ms=4.5,label=label)
    ax.vlines(x,low,high,color=color,lw=1.1)
    for x0,a,b in zip(x,low,high):
        ax.hlines([a,b],x0-.045,x0+.045,color=color,lw=1.1)


def throughput(path, result):
    cells={(c["mode"],c["rate"]):c for c in result["cells"]}
    rates=sorted({r for mode,r in cells})
    ticks=np.arange(len(rates))
    fig,axes=plt.subplots(2,2,figsize=(12,7.0))
    for offset,(mode,(label,color)) in enumerate(MODES.items()):
        cs=[cells[mode,r] for r in rates]
        x=ticks+(offset-1)*.055
        for ax,key,out in ((axes[0,0],"output_tok_s","output_tok_s"),(axes[0,1],"goodput_s","goodput")):
            line(ax,x,[c["window"][key] for c in cs],[[b[out] for b in c["by_repeat"]] for c in cs],label,color)
        deltas=[[b["queue_delta"] for b in c["by_repeat"]] for c in cs]
        line(axes[1,0],x,np.mean(deltas,axis=1),deltas,label,color)
        drains=[c["drain_seconds"] for c in cs]
        line(axes[1,1],x,np.mean(drains,axis=1),drains,label,color)
    axes[0,0].plot(ticks,np.array(rates)*512,"--",lw=1,color="#929292",label="Offered work")
    axes[0,1].plot(ticks,rates,"--",lw=1,color="#929292")
    axes[1,0].axhline(0,color="#777777",lw=.7)
    for ax,title,unit in zip(axes.flat,("Output during sustained arrivals","Requests meeting all streaming SLOs",
        "Change in unfinished requests over 120 s","Time to finish after arrivals stop"),
        ("Output tokens/s","Good requests/s","Requests","Seconds")):
        ax.set_title(title,loc="left",fontsize=12,fontweight="bold",pad=8)
        ax.set_xticks(ticks,rates)
        ax.set_ylabel(unit)
        ax.grid(axis="y",alpha=.22)
        ax.spines[["top","right"]].set_visible(False)
    for ax in axes[1,:]:
        ax.set_xlabel("Offered requests/s")
    for ax in axes[0,:]:
        ax.set_ylim(bottom=0)
    axes[1,1].set_ylim(bottom=0)
    fig.suptitle("Eight H100s serving Qwen3-8B  |  128 input / 512 output tokens",x=.07,ha="left",fontsize=18,fontweight="bold")
    axes[0,0].legend(loc="upper left",frameon=False,fontsize=9)
    repeats=sorted({c["repeats"] for c in cells.values()})
    count=str(repeats[0]) if len(repeats)==1 else f"{repeats[0]}–{repeats[-1]}"
    fig.text(.07,.018,f"{count} repeats per cell. Whiskers show the observed range. Goodput requires TTFT ≤ 500 ms, TPOT ≤ 50 ms and every pause ≤ 100 ms.",fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.87,bottom=.11,wspace=.24,hspace=.48)
    fig.savefig(path/"steady_and_drain.png",dpi=175,bbox_inches="tight")
    plt.close(fig)


def gpu_time(path,rate):
    profiles=[json.loads((path/"profiles"/f"{mode}-r{rate}-timeline.json").read_text()) for mode in MODES]
    labels=["8 replicas","4P + 4D  /  P","4P + 4D  /  D","3P + 5D  /  P","3P + 5D  /  D"]
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    colors=("#186f9c","#e8ab3c","#7655a1","#dbe0e6")
    keys=("compute_only_pct","kv_only_pct","kv_and_compute_pct","idle_pct")
    for ax,band,title in zip(axes,("flow","drain"),("Sustained arrivals  ·  8-second slice","After arrivals stop  ·  first ≤6 seconds")):
        rows=[role for item in profiles for role in item["windows"][band]["roles"].values()]
        y=np.arange(len(labels))
        left=np.zeros(len(labels))
        for key,color,label in zip(keys,colors,("Model / other CUDA","KV communication","Overlap","GPU idle")):
            values=np.array([r[key] for r in rows])
            ax.barh(y,values,left=left,height=.64,color=color,label=label)
            left+=values
        ax.set_xlim(0,100)
        ax.set_yticks(y,labels)
        ax.set_xlabel("Share of per-GPU wall time (%)")
        ax.set_title(title,loc="left",fontsize=12,fontweight="bold")
        ax.spines[["top","right","left"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(f"GPU timelines from Nsight Systems  |  {rate} requests/s",x=.02,ha="left",fontsize=17,fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.subplots_adjust(left=.15,right=.98,top=.79,bottom=.24,wspace=.2)
    fig.savefig(path/"nsight_gpu_time.png",dpi=175,bbox_inches="tight")
    plt.close(fig)


def request_time(path,result):
    cells={(c["mode"],c["rate"]):c for c in result["cells"]}
    stages=(
        ("Admission / P queue",("arrival_to_admit","p_queue"),"#a0aec0"),
        ("P batch",("p_service",),"#426282"),
        ("Wait before KV submit",("wait_to_submit_kv",),"#dfb15e"),
        ("KV submit to first token",("kv_submit_to_ack","ack_to_first"),"#b88323"),
        ("Wait for first D batch",("wait_to_next_dispatch",),"#bc5636"),
        ("511 D batch round trips",("scheduled_after_first",),"#1c7397"),
        ("Between D batches",("between_decode_steps",),"#79b7bb"),
    )
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    labels=[v[0] for v in MODES.values()]
    for ax,rate in zip(axes,(20,48)):
        costs=[cells[m,rate]["cohort_ms"] for m in MODES]
        y=np.arange(len(costs))
        left=np.zeros(len(costs))
        for label,keys,color in stages:
            width=np.array([sum(c[k]["mean"] for k in keys)/1000 for c in costs])
            ax.barh(y,width,left=left,height=.53,color=color,label=label)
            left+=width
        for i,total in enumerate(left):
            ax.annotate(f"{total:.2f} s",(total,i),xytext=(4,0),textcoords="offset points",va="center",fontsize=9)
        ax.set_xlim(0,left.max()*1.18)
        ax.set_yticks(y,labels)
        ax.set_xlabel("Mean elapsed seconds per request")
        ax.set_title(f"{rate} offered requests/s",loc="left",fontweight="bold")
        ax.spines[["top","right","left"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Where the request spends time  |  arrivals in the 120-second window",x=.02,ha="left",fontsize=16,fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.10,right=.98,top=.78,bottom=.30,wspace=.21)
    fig.savefig(path/"request_time.png",dpi=175,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("result_dir",type=Path)
    p.add_argument("--profile-rate",type=int)
    p.add_argument("--summary-file",default="summary.json")
    args=p.parse_args()
    result=json.loads((args.result_dir/args.summary_file).read_text())
    throughput(args.result_dir,result)
    request_time(args.result_dir,result)
    if args.profile_rate is not None:
        gpu_time(args.result_dir,args.profile_rate)
