#!/usr/bin/env python3
"""Generate publication-quality PNG graphs from llama-benchy benchmark JSON data.

Reads benchmark JSON files from a directory, groups by (concurrency, depth),
produces a 2-panel figure matching the example output format.

Usage:
  bench-graph.py -d <dir> [-m <model>] [-o <out.png>]
  bench-graph.py -d models/benchmarks/<model>/c1_d0-1024
"""
import json, argparse, os, sys, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def fmt_depth(d):
    """Format depth as human-readable label (0, 1k, 4k, 8k, ...)."""
    if d == 0:
        return '0'
    if d < 1024:
        return f'{d}'
    if d < 10000:
        return f'{d // 1024}k'
    return f'{d // 1024}k'


def mean_std(vals):
    """Return (mean, std) for a list of values. Std = population std dev."""
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(var)


def format_val(val, unit='', precision=1):
    """Format a value with unit, returning '0' for zero."""
    if val == 0:
        return '0'
    return f"{val:.{precision}f}{unit}"


def make_label(val, unit='s'):
    """Create a compact annotation label (e.g. '24.8s', '1.24s')."""
    if val >= 1:
        return f"{val:.1f}{unit}"
    return f"{val*1000:.1f}m{unit}"


# ── Data extraction ─────────────────────────────────────────────────────────

def parse_json_files(bench_dir):
    """Read all JSON files, group data by (concurrency, depth)."""
    d = Path(bench_dir)
    if not d.is_dir():
        print(f"Error: {bench_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(d.glob('*.json'))
    if not json_files:
        print(f"Error: no JSON files in {bench_dir}", file=sys.stderr)
        sys.exit(1)

    model = 'unknown'
    # key: (concurrency, depth) → agg dict
    data = defaultdict(lambda: {
        'pp': [],    # prefill throughput
        'tg': [],    # generation throughput
        'ttfr': [],  # time-to-first-resp (ms)
        'e2e_ttft': [],  # end-to-end TTFT (ms)
        'peak': [],  # peak throughput
    })

    for jf in json_files:
        try:
            dat = json.loads(jf.read_bytes())
        except Exception:
            continue
        if not isinstance(dat, dict):
            continue
        if dat.get('model'):
            model = dat['model']

        for bench in dat.get('benchmarks', []):
            conc = bench.get('concurrency', 1)
            depth = bench.get('context_size', 0)
            key = (conc, depth)

            pp_m = bench.get('pp_throughput', {}).get('mean', 0)
            tg_m = bench.get('tg_throughput', {}).get('mean', 0)
            ttfr_m = bench.get('ttfr', {}).get('mean', 0)
            e2e_m = bench.get('e2e_ttft', {}).get('mean', 0)
            peak_m = bench.get('peak_throughput', {}).get('mean', 0)

            if pp_m:
                data[key]['pp'].append(pp_m)
            if tg_m:
                data[key]['tg'].append(tg_m)
            if ttfr_m:
                data[key]['ttfr'].append(ttfr_m)
            if e2e_m:
                data[key]['e2e_ttft'].append(e2e_m)
            if peak_m:
                data[key]['peak'].append(peak_m)

    return model, data


# ── Single concurrency plots ────────────────────────────────────────────────

def single_c_plot(depths, data, out_path, model_name):
    """Two-panel: throughput (dual axis) + TTFT."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150,
                                     gridspec_kw={'width_ratios': [1.3, 1]})

    pp_means = []
    pp_stds = []
    tg_means = []
    tg_stds = []
    depths_fmt = []

    for d in sorted(depths):
        key = (1, d)
        entry = data.get(key, {})
        pp_m, pp_s = mean_std(entry.get('pp', []))
        tg_m, tg_s = mean_std(entry.get('tg', []))
        pp_means.append(pp_m)
        pp_stds.append(pp_s)
        tg_means.append(tg_m)
        tg_stds.append(tg_s)
        depths_fmt.append(fmt_depth(d))

    # ── Color palette ────────────────────────────────────────────────────
    c_pp  = '#2E86AB'     # blue    - prefill
    c_tg  = '#A23B72'     # magenta - generation
    c_e2e = '#2B9348'     # green   - TTFT

    # ── Left: Throughput vs Depth (dual y-axis) ──────────────────────────
    ax1_twin = ax1.twinx()

    ax1.plot(depths_fmt, pp_means, 'o-', color=c_pp, linewidth=2.5,
             markersize=7, label='Prefill', zorder=3)
    ax1.errorbar(depths_fmt, pp_means, yerr=pp_stds, fmt='none',
                 color=c_pp, capsize=3, capthick=1.5, alpha=0.6, zorder=2)

    ax1_twin.plot(depths_fmt, tg_means, 's-', color=c_tg, linewidth=2.5,
                  markersize=7, label='Generation', zorder=3)
    ax1_twin.errorbar(depths_fmt, tg_means, yerr=tg_stds, fmt='none',
                      color=c_tg, capsize=3, capthick=1.5, alpha=0.6, zorder=2)

    ax1.set_xlabel('Context Depth', fontsize=12, fontweight='medium')
    ax1.set_ylabel('Prefill Throughput (t/s)', color=c_pp, fontsize=11, fontweight='medium')
    ax1_twin.set_ylabel('Generation Throughput (t/s)', color=c_tg, fontsize=11, fontweight='medium')
    ax1.tick_params(axis='y', labelcolor=c_pp)
    ax1_twin.tick_params(axis='y', labelcolor=c_tg)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Combine legend from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='lower left', fontsize=10, framealpha=0.9)

    # ── Right: TTFT Curve (e2e_ttft in seconds) ──────────────────────────
    ttft_vals = []
    ttft_err  = []
    for d in sorted(depths):
        m, s = mean_std(data.get((1, d), {}).get('e2e_ttft', [0]))
        ttft_vals.append(m / 1000.0)  # ms → s
        ttft_err.append(s / 1000.0)

    ax2.plot(depths_fmt, ttft_vals, 'o-', color=c_e2e, linewidth=2.5,
             markersize=7, zorder=3)
    ax2.errorbar(depths_fmt, ttft_vals, yerr=ttft_err, fmt='none',
                 color=c_e2e, capsize=3, capthick=1.5, alpha=0.6, zorder=2)

    ax2.set_xlabel('Context Depth', fontsize=12, fontweight='medium')
    ax2.set_ylabel('End-to-End TTFT (seconds)', color=c_e2e, fontsize=11, fontweight='medium')
    ax2.tick_params(axis='y', labelcolor=c_e2e)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Put value labels on data points
    for i, (dfmt, v) in enumerate(zip(depths_fmt, ttft_vals)):
        if v > 0.01:
            label = make_label(v)
            ax2.annotate(label, (dfmt, v), textcoords='offset points',
                        xytext=(0, 14), ha='center', fontsize=8.5, color=c_e2e,
                        fontweight='medium')

    # ── Title ────────────────────────────────────────────────────────────
    fig.suptitle(f'{model_name} (Single DGX Spark, C1 Only)',
                 fontsize=13, fontweight='bold', y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ── Multi-concurrency plots ────────────────────────────────────────────────

def multi_c_plot(depths, concs, data, out_path, model_name):
    """Multi-panel but overlay all concurrency levels on shared axes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150,
                                     gridspec_kw={'width_ratios': [1.3, 1]})

    depths_fmt = [fmt_depth(d) for d in sorted(depths)]

    # Pre-compute aggregates for each {C, D}
    aggregated = {}
    for c in concs:
        aggregated[c] = {}
        for d in sorted(depths):
            key = (c, d)
            entry = data.get(key, {})
            pp_m, pp_s = mean_std(entry.get('pp', [0]))
            tg_m, tg_s = mean_std(entry.get('tg', [0]))
            pp_m = pp_m if pp_m else 0
            tg_m = tg_m if tg_m else 0
            aggregated[c][d] = {'pp': pp_m, 'tg': tg_m,
                                'pp_s': max(0, pp_s), 'tg_s': max(0, tg_s)}

    n_concs = len(concs)
    cmap_colors = ['#0000FF', '#FF0000', '#00AA00', '#FF7F00',   # blue, red, green, orange
                   '#9932CC', '#C0C0C0', '#1E90FF', '#DC143C']  # purple, grey, blue, dark red

    # ── Left: Throughput vs Depth (dual y-axis) ──────────────────────────
    ax1_twin = ax1.twinx()

    for ci, c in enumerate(concs):
        col = cmap_colors[ci % len(cmap_colors)]
        pp_m = [aggregated[c][d]['pp'] for d in sorted(depths)]
        tg_m = [aggregated[c][d]['tg'] for d in sorted(depths)]
        pp_s = [aggregated[c][d]['pp_s'] for d in sorted(depths)]
        tg_s = [aggregated[c][d]['tg_s'] for d in sorted(depths)]

        # Prefill on primary axis
        ax1.plot(depths_fmt, pp_m, 'o-', color=col, linewidth=2.0,
                 markersize=6,
                 label=f'Prefill (C={c})', zorder=3)
        ax1.errorbar(depths_fmt, pp_m, yerr=pp_s, fmt='none',
                     color=col, capsize=3, capthick=1.2, alpha=0.5, zorder=2)

        # Generation on twin axis
        ax1_twin.plot(depths_fmt, tg_m, 'o-', color=col, linewidth=2.0,
                      markersize=6,
                      label=f'Gen (C={c})', zorder=3)
        ax1_twin.errorbar(depths_fmt, tg_m, yerr=tg_s, fmt='none',
                          color=col, capsize=3, capthick=1.2, alpha=0.5, zorder=2)

    ax1.set_xlabel('Context Depth', fontsize=12, fontweight='medium')
    ax1.set_ylabel('Prefill Throughput (t/s)', fontsize=11, fontweight='medium')
    ax1_twin.set_ylabel('Generation Throughput (t/s)', fontsize=11, fontweight='medium')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='lower left', fontsize=9, framealpha=0.9, ncol=2)

    # ── Right: TTFT Curve ────────────────────────────────────────────────
    c_e2e = '#2B9348'
    for ci, c in enumerate(concs):
        col = cmap_colors[ci % len(cmap_colors)]
        ttft_m = []
        ttft_s = []
        for d in sorted(depths):
            key = (c, d)
            m, s = mean_std(data.get(key, {}).get('e2e_ttft', [0]))
            ttft_m.append(m / 1000.0)
            ttft_s.append(s / 1000.0)

        if any(v > 0.01 for v in ttft_m):
            ax2.plot(depths_fmt, ttft_m, 'o-', color=col, linewidth=2.0,
                     markersize=6, label=f'TTFT (C={c})', zorder=3)
            ax2.errorbar(depths_fmt, ttft_m, yerr=ttft_s, fmt='none',
                         color=col, capsize=3, capthick=1.2, alpha=0.5, zorder=2)
            # Put value labels only on the last point to avoid clutter
            if ttft_m:
                last_d = fmt_depth(depths[-1])
                last_v = ttft_m[-1]
                if last_v > 0.01:
                    label = make_label(last_v)
                    ax2.annotate(label, (last_d, last_v), textcoords='offset points',
                                xytext=(0, 14), ha='center', fontsize=8, color=col,
                                fontweight='medium')

    ax2.set_xlabel('Context Depth', fontsize=12, fontweight='medium')
    ax2.set_ylabel('End-to-End TTFT (seconds)', color=c_e2e, fontsize=11, fontweight='medium')
    ax2.tick_params(axis='y', labelcolor=c_e2e)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # ── Title ────────────────────────────────────────────────────────────
    conc_str = 'C1' if n_concs == 1 else f'C{"-".join(str(x) for x in concs)}'
    fig.suptitle(f'{model_name} ({conc_str})',
                 fontsize=13, fontweight='bold', y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Generate benchmark PNG graphs from JSON')
    ap.add_argument('-d', '--dir', required=True,
                    help='Directory with benchmark JSON files')
    ap.add_argument('-m', '--model', default='',
                    help='Model name for title')
    ap.add_argument('-o', '--out', default=None,
                    help='Output PNG path (default: auto in benchmark dir)')
    args, _ = ap.parse_known_args()

    model_name, data = parse_json_files(args.dir)

    if not model_name or model_name == 'unknown':
        model_name = args.model or 'Benchmark'

    all_concs = sorted(set(k[0] for k in data.keys()))
    all_depths = sorted(set(k[1] for k in data.keys()))

    if not all_depths:
        print('Error: no benchmark data found', file=sys.stderr)
        sys.exit(1)

    # Default output path
    if not args.out:
        conc_part = f'c{"_".join(str(c) for c in all_concs)}' if all_concs else 'c1'
        depth_min = min(all_depths)
        depth_max = max(all_depths)
        depth_part = (f'd{depth_min}-{depth_max}'
                      if depth_min != depth_max else f'd{depth_min}')
        from datetime import datetime
        ts = datetime.now().strftime('%d_%m_%y_%H_%M')
        base = Path(args.dir)
        args.out = str(base / f'benchmark_{ts}_{conc_part}_{depth_part}.png')

    out_path = args.out

    if len(all_concs) <= 1:
        single_c_plot(all_depths, data, out_path, model_name)
    else:
        multi_c_plot(all_depths, all_concs, data, out_path, model_name)

    print(f'Generated: {out_path}')


if __name__ == '__main__':
    main()
