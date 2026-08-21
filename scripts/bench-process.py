#!/usr/bin/env python3
"""Post-process llama-benchy JSON output → MD report + PNG graph.

Usage: python3 bench-process.py <json-file>
Output: <json-file> → <json-file-without-ext>.md, <json-file-without-ext>.png
"""
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any


# ── PNG generation (adapted from graph.py) ─────────────────────────────────

def fmt_depth(d: int) -> str:
    if d == 0:
        return "0"
    if d < 1024:
        return str(d)
    return f"{d // 1024}k"

def mean_std(vals: list) -> tuple:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(var)


def filter_outlier_values(values):
    """Drop per-run values that differ from the median by more than 100%.

    A value more than 100% away from the median (i.e. < median/2 or
    > median*2) is treated as an anomaly (e.g. a corrupted peak) and excluded
    from mean/std. Requires at least 3 runs so a single outlier never
    collapses to an empty set.
    """
    if len(values) < 3:
        return values
    vals = sorted(values)
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    if med == 0:
        return values
    return [v for v in values if med * 0.5 <= v <= med * 2.0]


_METRIC_KEYS = [
    "pp_throughput", "pp_req_throughput", "tg_throughput", "tg_req_throughput",
    "peak_throughput", "peak_req_throughput", "ttfr", "est_ppt", "e2e_ttft",
]


def _sanitize(benchmarks):
    """Return a copy of benchmarks with anomalous per-run values removed and
    mean/std recomputed from the surviving runs."""
    cleaned = []
    for b in benchmarks:
        nb = dict(b)
        for k in _METRIC_KEYS:
            metric = b.get(k)
            if metric and metric.get("values"):
                kept = filter_outlier_values(metric["values"])
                if len(kept) != len(metric["values"]):
                    m, s = mean_std(kept)
                    nb[k] = dict(metric, values=kept, mean=m, std=s)
        cleaned.append(nb)
    return cleaned

def make_label(val: float, unit: str = "s") -> str:
    if val >= 1:
        return f"{val:.1f}{unit}"
    return f"{val*1000:.1f}m{unit}"

def _agg_to_data(benchmarks: list) -> dict:
    data = defaultdict(lambda: {"pp": [], "tg": [], "ttfr": [], "e2e_ttft": [], "peak": []})
    for b in benchmarks:
        key = (b["concurrency"], b["context_size"])
        if b.get("pp_throughput"):
            data[key]["pp"].append(b["pp_throughput"]["mean"])
        if b.get("tg_throughput"):
            data[key]["tg"].append(b["tg_throughput"]["mean"])
        if b.get("ttfr"):
            data[key]["ttfr"].append(b["ttfr"]["mean"])
        if b.get("e2e_ttft"):
            data[key]["e2e_ttft"].append(b["e2e_ttft"]["mean"])
        if b.get("peak_throughput"):
            data[key]["peak"].append(b["peak_throughput"]["mean"])
    return dict(data)

def gen_png(benchmarks: list, out_path: str, model_name: str = "Benchmark", max_concurrency: int = 1) -> str:
    data = _agg_to_data(benchmarks)
    all_depths = sorted(set(b["context_size"] for b in benchmarks))
    all_concs = sorted(set(b["concurrency"] for b in benchmarks))
    if not all_depths:
        raise ValueError("No benchmark data to plot")
    if all_depths == [0]:
        raise ValueError("No depth variation for line plot")
    if len(all_concs) <= 1:
        _single_c_plot(all_depths, data, out_path, model_name)
    else:
        _multi_c_plot(all_depths, all_concs, data, out_path, model_name)
    return out_path

def _single_c_plot(depths: list, data: dict, out_path: str, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150, gridspec_kw={"width_ratios": [1.3, 1]})
    pp_m, pp_s, tg_m, tg_s, depths_fmt = [], [], [], [], []
    for d in sorted(depths):
        entry = data.get((1, d), {})
        m1, s1 = mean_std(entry.get("pp", []))
        m2, s2 = mean_std(entry.get("tg", []))
        pp_m.append(m1); pp_s.append(s1); tg_m.append(m2); tg_s.append(s2)
        depths_fmt.append(fmt_depth(d))
    c_pp, c_tg, c_e2e = "#2E86AB", "#A23B72", "#2B9348"
    ax1_twin = ax1.twinx()
    ax1.plot(depths_fmt, pp_m, "s-", color=c_pp, linewidth=2.5, markersize=7, label="Prefill", zorder=3)
    ax1.errorbar(depths_fmt, pp_m, yerr=pp_s, fmt="none", color=c_pp, capsize=3, capthick=1.5, alpha=0.6, zorder=2)
    ax1_twin.plot(depths_fmt, tg_m, "o-", color=c_tg, linewidth=2.5, markersize=7, label="Generation", zorder=3)
    ax1_twin.errorbar(depths_fmt, tg_m, yerr=tg_s, fmt="none", color=c_tg, capsize=3, capthick=1.5, alpha=0.6, zorder=2)
    ax1.set_xlabel("Context Depth", fontsize=12, fontweight="medium")
    ax1.set_ylabel("Prefill Throughput (t/s)", color=c_pp, fontsize=11, fontweight="medium")
    ax1_twin.set_ylabel("Generation Throughput (t/s)", color=c_tg, fontsize=11, fontweight="medium")
    ax1.tick_params(axis="y", labelcolor=c_pp); ax1_twin.tick_params(axis="y", labelcolor=c_tg)
    ax1.grid(True, alpha=0.3, linestyle="--")
    max_pp = max(pp_m) if pp_m else 0; min_pp = min(pp_m) if pp_m else 0
    max_gen = max(tg_m) if tg_m else 0; min_gen = min(tg_m) if tg_m else 0
    ax1.set_ylim(bottom=min_pp * 0.7, top=max_pp * 1.15)
    ax1_twin.set_ylim(bottom=min_gen * 0.7, top=max_gen * 1.15)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=10, framealpha=0.9)
    ttft_vals, ttft_err = [], []
    for d in sorted(depths):
        m, s = mean_std(data.get((1, d), {}).get("e2e_ttft", [0]))
        ttft_vals.append(m / 1000.0); ttft_err.append(s / 1000.0)
    ax2.plot(depths_fmt, ttft_vals, "o-", color=c_e2e, linewidth=2.5, markersize=7, zorder=3)
    ax2.errorbar(depths_fmt, ttft_vals, yerr=ttft_err, fmt="none", color=c_e2e, capsize=3, capthick=1.5, alpha=0.6, zorder=2)
    ax2.set_xlabel("Context Depth", fontsize=12, fontweight="medium")
    ax2.set_ylabel("End-to-End TTFT (seconds)", color=c_e2e, fontsize=11, fontweight="medium")
    ax2.tick_params(axis="y", labelcolor=c_e2e); ax2.grid(True, alpha=0.3, linestyle="--")
    for dfmt, v in zip(depths_fmt, ttft_vals):
        if v > 0.01:
            ax2.annotate(make_label(v), (dfmt, v), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8.5, color=c_e2e, fontweight="medium")
    fig.suptitle(f"{model_name} (C1)", fontsize=13, fontweight="bold", y=1.02)
    ax1.set_ylim(bottom=min_pp * 0.7, top=max_pp * 1.15)
    ax1_twin.set_ylim(bottom=min_gen * 0.7, top=max_gen * 1.15)
    ttft_raw = [v for v in ttft_vals if v > 0]
    if ttft_raw:
        ax2.set_ylim(bottom=min(ttft_raw) * 0.7, top=max(ttft_raw) * 1.15)
    else:
        ax2.set_ylim(bottom=0, top=1)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.20)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def _multi_c_plot(depths: list, concs: list, data: dict, out_path: str, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150, gridspec_kw={"width_ratios": [1.3, 1]})
    depths_fmt = [fmt_depth(d) for d in sorted(depths)]
    aggregated = {}
    for c in concs:
        aggregated[c] = {}
        for d in sorted(depths):
            entry = data.get((c, d), {})
            m1, s1 = mean_std(entry.get("pp", []))
            m2, s2 = mean_std(entry.get("tg", []))
            aggregated[c][d] = {"pp": m1 or 0, "tg": m2 or 0, "pp_s": max(0, s1), "tg_s": max(0, s2)}
    cmap = ["#0000FF", "#FF0000", "#00AA00", "#FF7F00", "#9932CC", "#C0C0C0", "#1E90FF", "#DC143C"]
    ax1_twin = ax1.twinx()
    for ci, c in enumerate(concs):
        col = cmap[ci % len(cmap)]
        pp_m = [aggregated[c][d]["pp"] for d in sorted(depths)]
        tg_m = [aggregated[c][d]["tg"] for d in sorted(depths)]
        pp_s = [aggregated[c][d]["pp_s"] for d in sorted(depths)]
        tg_s = [aggregated[c][d]["tg_s"] for d in sorted(depths)]
        ax1.plot(depths_fmt, pp_m, "o--", color=col, linewidth=2.0, markersize=6, label=f"Prefill (C={c})", zorder=3)
        ax1.errorbar(depths_fmt, pp_m, yerr=pp_s, fmt="none", color=col, capsize=3, capthick=1.2, alpha=0.5, zorder=2)
        ax1_twin.plot(depths_fmt, tg_m, "s-", color=col, linewidth=2.0, markersize=6, label=f"Generation (C={c})", zorder=3)
        ax1_twin.errorbar(depths_fmt, tg_m, yerr=tg_s, fmt="none", color=col, capsize=3, capthick=1.2, alpha=0.5, zorder=2)
    ax1.set_xlabel("Context Depth", fontsize=12, fontweight="medium")
    ax1.set_ylabel("Prefill Throughput (t/s)", fontsize=11, fontweight="medium")
    ax1_twin.set_ylabel("Generation Throughput (t/s)", fontsize=11, fontweight="medium")
    ax1.grid(True, alpha=0.3, linestyle="--")
    all_pp, all_tg = [], []
    for c in concs:
        for d in sorted(depths):
            all_pp.append(aggregated[c][d]["pp"])
            all_tg.append(aggregated[c][d]["tg"])
    max_pp = max(all_pp) if all_pp else 0; min_pp = min(all_pp) if all_pp else 0
    max_gen = max(all_tg) if all_tg else 0; min_gen = min(all_tg) if all_tg else 0
    ax1.set_ylim(bottom=min_pp * 0.7, top=max_pp * 1.15)
    ax1_twin.set_ylim(bottom=min_gen * 0.7, top=max_gen * 1.15)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9, framealpha=0.9, ncol=2)
    c_e2e = "#2B9348"
    for ci, c in enumerate(concs):
        col = cmap[ci % len(cmap)]
        ttft_m, ttft_s = [], []
        for d in sorted(depths):
            m, s = mean_std(data.get((c, d), {}).get("e2e_ttft", [0]))
            ttft_m.append(m / 1000.0); ttft_s.append(s / 1000.0)
        if any(v > 0.01 for v in ttft_m):
            ax2.plot(depths_fmt, ttft_m, "o-", color=col, linewidth=2.0, markersize=6, label=f"TTFT (C={c})", zorder=3)
            ax2.errorbar(depths_fmt, ttft_m, yerr=ttft_s, fmt="none", color=col, capsize=3, capthick=1.2, alpha=0.5, zorder=2)
            if ttft_m and ttft_m[-1] > 0.01:
                ax2.annotate(make_label(ttft_m[-1]), (fmt_depth(depths[-1]), ttft_m[-1]), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8, color=col, fontweight="medium")
    ax2.set_xlabel("Context Depth", fontsize=12, fontweight="medium")
    ax2.set_ylabel("End-to-End TTFT (seconds)", color=c_e2e, fontsize=11, fontweight="medium")
    ax2.tick_params(axis="y", labelcolor=c_e2e); ax2.grid(True, alpha=0.3, linestyle="--")
    ttft_raw = []
    for c in concs:
        for d in sorted(depths):
            m, _ = mean_std(data.get((c, d), {}).get("e2e_ttft", [0]))
            ttft_raw.append(m / 1000.0)
    if ttft_raw:
        ax2.set_ylim(bottom=min(ttft_raw) * 0.7, top=max(ttft_raw) * 1.15)
    conc_str = "C1" if len(concs) == 1 else f"C{'-'.join(str(x) for x in concs)}"
    fig.suptitle(f"{model_name} ({conc_str})", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


# ── MD report generation (adapted from results.py) ─────────────────────────

def gen_md(benchmarks: list, meta: dict, model_name: str = "Benchmark") -> str:
    max_concurrency = meta.get("max_concurrency", 1)
    from tabulate import tabulate
    rows = []
    for b in benchmarks:
        c_suffix = ""
        if max_concurrency > 1:
            c_suffix = f" (c{b['concurrency']})"
        d_suffix = f" @ d{b['context_size']}" if b["context_size"] > 0 else ""
        if b.get("pp_throughput"):
            rows.append(_mkrow(model_name, f"pp{b['prompt_size']}{d_suffix}{c_suffix}", b, "pp_throughput", "pp_req_throughput", "", "", "ttfr", "est_ppt", "e2e_ttft"))
        if b.get("tg_throughput"):
            rows.append(_mkrow(model_name, f"tg{b['response_size']}{d_suffix}{c_suffix}", b, "tg_throughput", "tg_req_throughput", "peak_throughput", "peak_req_throughput", "", "", ""))
    if not rows:
        return "No results collected."
    if max_concurrency == 1:
        data = [[r["model"], r["test"], r["t_s"], r["peak_ts"], r["ttfr"], r["est_ppt"], r["e2e_ttft"]] for r in rows]
        headers = ["model", "test", "t/s", "peak t/s", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"]
    else:
        data = [[r["model"], r["test"], r["t_s"], r["t_s_req"], r["peak_ts"], r["peak_ts_req"], r["ttfr"], r["est_ppt"], r["e2e_ttft"]] for r in rows]
        headers = ["model", "test", "t/s (total)", "t/s (req)", "peak t/s", "peak t/s (req)", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"]
    return tabulate(data, headers=headers, tablefmt="pipe")

def _fmt(metric: Any) -> str:
    if not metric:
        return ""
    return f"{metric['mean']:.2f} ± {metric['std']:.2f}"

def _mkrow(model: str, test: str, b: dict, *keys: str) -> dict:
    row: dict[str, str] = {"model": model, "test": test}
    key_names = ["t_s", "t_s_req", "peak_ts", "peak_ts_req", "ttfr", "est_ppt", "e2e_ttft"]
    for i, kn in enumerate(key_names):
        row[kn] = _fmt(b.get(keys[i])) if i < len(keys) and keys[i] else ""
    return row


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <benchmark-json-path>", file=sys.stderr)
        sys.exit(1)
    json_path = sys.argv[1]
    with open(json_path) as f:
        report = json.load(f)
    benchmarks: list = report.get("benchmarks", [])
    benchmarks = _sanitize(benchmarks)
    if not benchmarks:
        print("No benchmark data found in JSON", file=sys.stderr)
        sys.exit(1)
    model_name = report.get("model", "Benchmark")
    max_concurrency = report.get("max_concurrency", 1)
    base, _ = os.path.splitext(json_path)
    md_path = base + ".md"
    png_path = base + ".png"
    md = gen_md(benchmarks, report, model_name)
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Generated: {md_path}")
    gen_png(benchmarks, png_path, model_name, max_concurrency)
    print(f"Generated: {png_path}")

if __name__ == "__main__":
    main()
