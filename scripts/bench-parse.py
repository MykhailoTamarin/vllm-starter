#!/usr/bin/env python3
"""Parse llama-benchy benchmark JSON outputs into a markdown table."""
import json, argparse, os, sys, math
from pathlib import Path
from collections import defaultdict

def fmt(vals):
    if len(vals) == 0:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    mean = sum(vals) / len(vals)
    variance = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(variance)
    return f"{mean:.2f} ± {std:.2f}"

def main():
    ap = argparse.ArgumentParser(description="Parse benchmark JSON into markdown")
    ap.add_argument('-d', '--dir', required=True)
    ap.add_argument('-o', '--out', default=None, help='Save MD to this file (default: stdout)')
    ap.add_argument('-p', '--pp', type=int, default=2048)
    args, _ = ap.parse_known_args()

    d = Path(args.dir)
    if not d.is_dir():
        print(f"Error: {args.dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(d.glob('*.json'))
    if not json_files:
        print(f"Error: no JSON files in {args.dir}", file=sys.stderr)
        sys.exit(1)

    model = 'unknown'

    # Each entry has BOTH pp and tg metrics.
    # Group by (depth, conc) to average across runs
    pp_vals = defaultdict(list)   # (depth, conc) -> [pp_mean]
    tg_vals = defaultdict(list)   # (depth, conc) -> [tg_mean]
    ttfr_vals = defaultdict(list)  # (depth, conc) -> [ttfr]
    est_vals = defaultdict(list)  # (depth, conc) -> [est_ppt]
    peak_vals = defaultdict(list)  # (depth, conc) -> [peak_throughput]
    all_concs = set()
    all_depths = set()

    for jf in json_files:
        try:
            data = json.loads(jf.read_bytes())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get('model'):
            model = data['model']

        for bench in data.get('benchmarks', []):
            depth = bench.get('context_size', 0)
            conc = bench.get('concurrency', 1)
            pp_mean = bench.get('pp_throughput', {}).get('mean', 0)
            tg_mean = bench.get('tg_throughput', {}).get('mean', 0)
            est = bench.get('est_ppt', {}).get('mean', 0)

            all_concs.add(conc)
            all_depths.add(depth)
            key = (depth, conc)
            if pp_mean:
                pp_vals[key].append(pp_mean)
            if tg_mean:
                tg_vals[key].append(tg_mean)
            ttfr_mean = bench.get('ttfr', {}).get('mean', 0)
            if ttfr_mean:
                ttfr_vals[key].append(ttfr_mean)
            peak_mean = bench.get('peak_throughput', {}).get('mean', 0)
            if peak_mean:
                peak_vals[key].append(peak_mean)
            if est:
                est_vals[key].append(est)

    concs = sorted(all_concs)
    depths = sorted(all_depths)
    is_multi = len(concs) > 1

    lines = []
    lines.append("")
    conc_label = "single (c=1)" if not is_multi else f"multiple (c{'-'.join(str(c) for c in concs)})"
    lines.append(f"Model: {model}")
    lines.append(f"Concurrency: {conc_label}")
    lines.append(f"Depths: {', '.join(str(x) for x in depths)}")
    lines.append("")
    lines.append("| test | t/s | est_ppt (ms) | ttfr (ms) | peak t/s |")
    lines.append("|------|-----|--------------|-----------|----------|")

    for depth in depths:
        for conc in concs:
            key = (depth, conc)
            if not pp_vals.get(key):
                continue
            name = f"pp{args.pp}" if depth == 0 else f"pp{args.pp} @ d{depth}"
            if is_multi:
                name = f"{name} (c{conc})"
            pp_str = fmt(pp_vals[key])
            est_str = fmt(est_vals[key])
            ttfr_str = fmt(ttfr_vals.get(key, [0]))
            peak_str = fmt(peak_vals.get(key, [pp_str])) if peak_vals.get(key) else ""
            lines.append(f"| {name} | {pp_str} | {est_str} | {ttfr_str} | {peak_str} |")

        # Generation rows
        if is_multi:
            for conc in concs:
                key = (depth, conc)
                if not tg_vals.get(key):
                    continue
                tg_str = fmt(tg_vals[key])
                lines.append(f"| tg32 (c{conc}) | {tg_str} | - | - | - |")
        else:
            key = (depth, 1)
            if tg_vals.get(key):
                tg_str = fmt(tg_vals[key])
                lines.append(f"| tg32 | {tg_str} | - | - | - |")

    output = "\n".join(lines) + f"\n\nBenchmarks from: {args.dir}/\n"

    if args.out:
        Path(args.out).write_text(output)
        print(f"Saved to: {args.out}")
    else:
        print(output)

if __name__ == '__main__':
    main()
