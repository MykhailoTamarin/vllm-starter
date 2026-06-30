"""Generate PNG graphs from in-memory benchmark data.

Works with list of BenchmarkRun objects from BenchmarkResults.
"""
import math
import warnings
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from .results import BenchmarkRun

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def fmt_depth(d: int) -> str:
    """Format depth as human-readable label (0, 1k, 4k, 8k, ...)."""
    if d == 0:
        return '0'
    if d < 1024:
        return f'{d}'
    return f'{d // 1024}k'


def mean_std(vals: list) -> Tuple[float, float]:
    """Return (mean, std) for a list of values. Std = population std dev."""
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(var)


def make_label(val: float, unit: str = 's') -> str:
    """Create a compact annotation label (e.g. '24.8s', '1.24s')."""
    if val >= 1:
        return f"{val:.1f}{unit}"
    return f"{val*1000:.1f}m{unit}"


def _aggregate_to_data(runs: List[BenchmarkRun]) -> Dict[Tuple[int, int], Dict[str, list]]:
    """Convert BenchmarkRun list to the dict format expected by plot functions."""
    data = defaultdict(lambda: {
        'pp': [], 'tg': [], 'ttfr': [], 'e2e_ttft': [], 'peak': [],
    })
    for run in runs:
        key = (run.concurrency, run.context_size)
        if run.pp_throughput:
            data[key]['pp'].append(run.pp_throughput.mean)
        if run.tg_throughput:
            data[key]['tg'].append(run.tg_throughput.mean)
        if run.ttfr:
            data[key]['ttfr'].append(run.ttfr.mean)
        if run.e2e_ttft:
            data[key]['e2e_ttft'].append(run.e2e_ttft.mean)
        if run.peak_throughput:
            data[key]['peak'].append(run.peak_throughput.mean)
    return dict(data)


def generate_png(
    runs: List[BenchmarkRun],
    out_path: str,
    model_name: str = 'Benchmark',
    max_concurrency: int = 1,
) -> str:
    """Generate a PNG graph from benchmark runs. Returns the output path."""
    data = _aggregate_to_data(runs)

    all_concs = sorted(set(run.concurrency for run in runs))
    all_depths = sorted(set(run.context_size for run in runs))

    if not all_depths:
        raise ValueError("No benchmark data to plot")

    if all_depths == [0]:
        # Single point - nothing to graph meaningful as a line plot
        # Still generate a minimal plot or skip
        raise ValueError("No depth variation for line plot")

    if len(all_concs) <= 1:
        _single_c_plot(all_depths, data, out_path, model_name)
    else:
        _multi_c_plot(all_depths, all_concs, data, out_path, model_name)

    return out_path


def _single_c_plot(
    depths: list, data: dict, out_path: str, model_name: str,
) -> None:
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

    c_pp  = '#2E86AB'
    c_tg  = '#A23B72'
    c_e2e = '#2B9348'

    ax1_twin = ax1.twinx()

    ax1.plot(depths_fmt, pp_means, 's-', color=c_pp, linewidth=2.5,
             markersize=7, label='Prefill', zorder=3)
    ax1.errorbar(depths_fmt, pp_means, yerr=pp_stds, fmt='none',
                 color=c_pp, capsize=3, capthick=1.5, alpha=0.6, zorder=2)

    ax1_twin.plot(depths_fmt, tg_means, 'o-', color=c_tg, linewidth=2.5,
                   markersize=7, label='Generation', zorder=3)
    ax1_twin.errorbar(depths_fmt, tg_means, yerr=tg_stds, fmt='none',
                      color=c_tg, capsize=3, capthick=1.5, alpha=0.6, zorder=2)

    ax1.set_xlabel('Context Depth', fontsize=12, fontweight='medium')
    ax1.set_ylabel('Prefill Throughput (t/s)', color=c_pp, fontsize=11, fontweight='medium')
    ax1_twin.set_ylabel('Generation Throughput (t/s)', color=c_tg, fontsize=11, fontweight='medium')
    ax1.tick_params(axis='y', labelcolor=c_pp)
    ax1_twin.tick_params(axis='y', labelcolor=c_tg)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add padding: -30% bottom, +15% top
    max_pp = max(pp_means) if pp_means else 0
    min_pp = min(pp_means) if pp_means else 0
    max_gen = max(tg_means) if tg_means else 0
    min_gen = min(tg_means) if tg_means else 0
    ax1.set_ylim(bottom=min_pp * 0.7, top=max_pp * 1.15)
    ax1_twin.set_ylim(bottom=min_gen * 0.7, top=max_gen * 1.15)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='lower left', fontsize=10, framealpha=0.9)

    ttft_vals = []
    ttft_err = []
    for d in sorted(depths):
        m, s = mean_std(data.get((1, d), {}).get('e2e_ttft', [0]))
        ttft_vals.append(m / 1000.0)
        ttft_err.append(s / 1000.0)

    ax2.plot(depths_fmt, ttft_vals, 'o-', color=c_e2e, linewidth=2.5,
             markersize=7, zorder=3)
    ax2.errorbar(depths_fmt, ttft_vals, yerr=ttft_err, fmt='none',
                 color=c_e2e, capsize=3, capthick=1.5, alpha=0.6, zorder=2)

    ax2.set_xlabel('Context Depth', fontsize=12, fontweight='medium')
    ax2.set_ylabel('End-to-End TTFT (seconds)', color=c_e2e, fontsize=11, fontweight='medium')
    ax2.tick_params(axis='y', labelcolor=c_e2e)
    ax2.grid(True, alpha=0.3, linestyle='--')

    for i, (dfmt, v) in enumerate(zip(depths_fmt, ttft_vals)):
        if v > 0.01:
            label = make_label(v)
            ax2.annotate(label, (dfmt, v), textcoords='offset points',
                        xytext=(0, 14), ha='center', fontsize=8.5, color=c_e2e,
                        fontweight='medium')

    fig.suptitle(f'{model_name} (C1)', fontsize=13, fontweight='bold', y=1.02)

    # Add padding: -30% bottom, +15% top
    max_pp = max(pp_means) if pp_means else 0
    min_pp = min(pp_means) if pp_means else 0
    max_gen = max(tg_means) if tg_means else 0
    min_gen = min(tg_means) if tg_means else 0
    ax1.set_ylim(bottom=min_pp * 0.7, top=max_pp * 1.15)
    ax1_twin.set_ylim(bottom=min_gen * 0.7, top=max_gen * 1.15)

    # TTFT axis also needs padding
    ttft_raw = [v for v in ttft_vals if v > 0]
    if ttft_raw:
        ax2.set_ylim(bottom=min(ttft_raw) * 0.7, top=max(ttft_raw) * 1.15)
    else:
        ax2.set_ylim(bottom=0, top=1)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _multi_c_plot(
    depths: list, concs: list, data: dict, out_path: str, model_name: str,
) -> None:
    """Overlay all concurrency levels on shared axes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150,
                                     gridspec_kw={'width_ratios': [1.3, 1]})

    depths_fmt = [fmt_depth(d) for d in sorted(depths)]

    aggregated = {}
    for c in concs:
        aggregated[c] = {}
        for d in sorted(depths):
            key = (c, d)
            entry = data.get(key, {})
            pp_m, pp_s = mean_std(entry.get('pp', []))
            tg_m, tg_s = mean_std(entry.get('tg', []))
            aggregated[c][d] = {
                'pp': pp_m or 0, 'tg': tg_m or 0,
                'pp_s': max(0, pp_s), 'tg_s': max(0, tg_s),
            }

    cmap_colors = ['#0000FF', '#FF0000', '#00AA00', '#FF7F00',
                   '#9932CC', '#C0C0C0', '#1E90FF', '#DC143C']

    ax1_twin = ax1.twinx()

    for ci, c in enumerate(concs):
        col = cmap_colors[ci % len(cmap_colors)]
        pp_m = [aggregated[c][d]['pp'] for d in sorted(depths)]
        tg_m = [aggregated[c][d]['tg'] for d in sorted(depths)]
        pp_s = [aggregated[c][d]['pp_s'] for d in sorted(depths)]
        tg_s = [aggregated[c][d]['tg_s'] for d in sorted(depths)]

        ax1.plot(depths_fmt, pp_m, 'o--', color=col, linewidth=2.0,
                 markersize=6,
                 label=f'Prefill (C={c})', zorder=3)
        ax1.errorbar(depths_fmt, pp_m, yerr=pp_s, fmt='none',
                     color=col, capsize=3, capthick=1.2, alpha=0.5, zorder=2)

        ax1_twin.plot(depths_fmt, tg_m, 's-', color=col, linewidth=2.0,
                      markersize=6,
                      label=f'Generation (C={c})', zorder=3)
        ax1_twin.errorbar(depths_fmt, tg_m, yerr=tg_s, fmt='none',
                          color=col, capsize=3, capthick=1.2, alpha=0.5, zorder=2)

    ax1.set_xlabel('Context Depth', fontsize=12, fontweight='medium')
    ax1.set_ylabel('Prefill Throughput (t/s)', fontsize=11, fontweight='medium')
    ax1_twin.set_ylabel('Generation Throughput (t/s)', fontsize=11, fontweight='medium')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # All generation values for multi-C (right axis)
    all_pp_values = []
    all_tg_values = []
    for c in concs:
        for d_val in sorted(depths):
            entry = aggregated[c][d_val]
            all_pp_values.append(entry['pp'])
            all_tg_values.append(entry['tg'])
    max_pp = max(all_pp_values) if all_pp_values else 0
    min_pp = min(all_pp_values) if all_pp_values else 0
    max_gen = max(all_tg_values) if all_tg_values else 0
    min_gen = min(all_tg_values) if all_tg_values else 0
    ax1.set_ylim(bottom=min_pp * 0.7, top=max_pp * 1.15)
    ax1_twin.set_ylim(bottom=min_gen * 0.7, top=max_gen * 1.15)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='lower left', fontsize=9, framealpha=0.9, ncol=2)

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

    # TTFT axis padding
    ttft_raw = []
    for ci, c in enumerate(concs):
        for d in sorted(depths):
            m, _ = mean_std(data.get((c, d), {}).get('e2e_ttft', [0]))
            ttft_raw.append(m / 1000.0)
    if ttft_raw:
        ax2.set_ylim(bottom=min(ttft_raw) * 0.7, top=max(ttft_raw) * 1.15)

    conc_str = 'C1' if len(concs) == 1 else f'C{"-".join(str(x) for x in concs)}'
    fig.suptitle(f'{model_name} ({conc_str})', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
