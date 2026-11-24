import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from pd_metrics import summarize


def main():
    parser = argparse.ArgumentParser(description="Plot PD serving latency and SLO-goodput from a completed run")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", default="summary.json")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    summary = json.loads((args.directory / args.summary).read_text())
    slo, config = summary["slo"], summary["metadata"]["config"]
    if not summary["complete"]:
        raise ValueError("Incomplete benchmark")
    samples = {}
    with ZipFile(args.directory / "trials.zip") as archive:
        for name in archive.namelist():
            if name.startswith("trial-"):
                t = json.loads(archive.read(name))
                samples.setdefault((t["workload"], t["rate"], t["mode"]), []).append(
                    summarize([t], slo["ttft_ms"], slo["tpot_ms"], slo.get("max_gap_ms")))
    names = [w for w in ("arena", "azure", "long_input", "long_output") if w in config["workloads"]]
    if not names:
        raise ValueError("Select a rate-sweep workload to plot")
    style = {"dp2": ("2 independent replicas", "#246498", "s"),
             "pd": ("1 prefill + 1 decode", "#9B4124", "o"),
             "tp2": ("Tensor parallel, TP=2", "#16705A", "^"),
             "single": ("1 GPU", "#6A6480", "D")}
    colors = ["#246498", "#9B4124", "#16705A", "#76518F", "#987623"]
    markers = ["s", "o", "^", "D", "P"]
    for i, (mode, layout) in enumerate(summary["metadata"].get("layouts", {}).items()):
        p, d = len(layout["prefill"]), len(layout["decode"])
        label = f"{p} prefill + {d} decode" if p else f"{d} independent replicas"
        style[mode] = label, colors[i % len(colors)], markers[i % len(markers)]
    columns = [("ttft_ms", "P95 time to first token", "ms"),
               ("tpot_ms", "P95 time per output token", "ms"),
               ("request_max_gap_ms", "P95 worst pause per request", "ms"),
               ("goodput_req_s", "SLO-compliant completions", "requests/s")]
    fig, axes = plt.subplots(len(names), len(columns), figsize=(14, 2.65 * len(names) + 1.45), squeeze=False)
    ylabel = {"arena": "Arena-Hard, 64 requests", "azure": "Azure conversations, 128 requests",
              "long_input": "2,048 in, 32 out", "long_output": "128 in, 512 out"}
    for row, workload in enumerate(names):
        for col, (metric, title, unit) in enumerate(columns):
            ax = axes[row][col]
            for mode in config["modes"]:
                cells = sorted((c for c in summary["cells"] if c["workload"] == workload and c["mode"] == mode),
                               key=lambda c: c["offered_rps"])
                values = [c[metric] if unit == "requests/s" else c[metric]["p95"] for c in cells]
                xs = [c["offered_rps"] for c in cells]
                per_run = [[r[metric] if unit == "requests/s" else r[metric]["p95"]
                            for r in samples[workload, x, mode]] for x in xs]
                label, color, marker = style[mode]
                ax.fill_between(xs, [min(y) for y in per_run], [max(y) for y in per_run],
                                color=color, alpha=.10, linewidth=0)
                ax.plot(xs, values, color=color, label=label, marker=marker, linewidth=1.8, markersize=5)
            if unit == "ms":
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(subs=(1., 2., 5.)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylim(bottom=0)
            if metric in ("ttft_ms", "tpot_ms"):
                ax.axhline(slo[metric], color="#555555", linestyle=":", linewidth=.9)
            elif metric == "request_max_gap_ms" and "max_gap_ms" in slo:
                ax.axhline(slo["max_gap_ms"], color="#555555", linestyle=":", linewidth=.9)
            ax.set_xticks(config["rates"])
            ax.grid(axis="y", alpha=.20)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel((ylabel[workload] + "\n" if col == 0 else "") + unit)
            if row == 0:
                ax.set_title(title, fontsize=11, loc="left", pad=7)
            if row == len(names) - 1:
                ax.set_xlabel("Offered requests/s")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    model = Path(config["model"]).name
    if config["draft_model"]:
        model += " + " + Path(config["draft_model"]).name
    fig.suptitle(f"{model}  |  Serving under open-loop arrivals", x=.065, y=.99,
                 ha="left", fontsize=16, fontweight="bold")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.058, .943), ncol=len(config["modes"]),
               frameon=False, fontsize=10)
    limits = f"TTFT {slo['ttft_ms']:g} ms and TPOT {slo['tpot_ms']:g} ms"
    if "max_gap_ms" in slo:
        limits += f" with a maximum pause of {slo['max_gap_ms']:g} ms"
    fig.text(.065, .012, f"Lines pool {config['repeats']} replays. Bands show the range across replays. "
             f"Dotted lines mark {limits}. "
             "Latency is measured at engine token commits.", fontsize=9, color="#424242")
    fig.subplots_adjust(left=.075, right=.985, top=.80, bottom=.13, hspace=.42, wspace=.43)
    fig.savefig(args.output or args.directory / "latency_goodput.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
