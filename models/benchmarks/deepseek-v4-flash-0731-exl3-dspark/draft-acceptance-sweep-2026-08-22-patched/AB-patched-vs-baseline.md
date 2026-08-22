# A/B: patched image vs unpatched baseline — K160 acceptance sweep

- date: 2026-08-22 (baseline 22:39, patched 23:18)
- model: deepseek-v4-flash-0731-exl3-dspark
- images: `ghcr.io/mykhailotamarin/vllm-starter:vllm-0_26_0-exl3-0.1.7` (baseline) vs `vllm-exl3-v26:latest` (patched: #52823 + #51967 + #52084 backports)
- harness: `sweep_dspark_acceptance.py --ks 160 --repeats 3 --max-tokens 512` (same script; patched run used `--tag patched` to isolate output dir)
- identical rep order in both runs → paired comparison is valid
- prefix-cache bypassed (unique salt per request), cold KV per request

## Headline numbers

| task | metric | baseline mean | patched mean | Δ | baseline median | patched median |
|---|---|---|---|---|---|---|
| coding | acceptance % | 45.3 | 46.8 | +1.5 | 42.5 | 45.4 |
| coding | gen t/s | 39.3 | 39.5 | +0.2 | 37.3 | 39.1 |
| coding | TTFT ms | 1803.7 | 1907.0 | +103.3 | 1750.0 | 1825.0 |
| chat | acceptance % | 35.0 | 34.3 | −0.7 | 35.1 | 34.0 |
| chat | gen t/s | 31.9 | 31.1 | −0.8 | 32.0 | 30.7 |
| chat | TTFT ms | 1744.7 | 1816.0 | +71.3 | 1771.0 | 1797.0 |
| text | acceptance % | 39.7 | 38.1 | −1.6 | 40.8 | 38.3 |
| text | gen t/s | 35.0 | 33.4 | −1.6 | 35.7 | 33.8 |
| text | TTFT ms | 1808.3 | 1744.7 | −63.7 | 1768.0 | 1765.0 |
| **ALL** | acceptance % | 40.0 | 39.7 | **−0.27** | 40.8 | 38.3 |
| **ALL** | gen t/s | 35.4 | 34.7 | **−0.73** | 35.7 | 33.8 |
| **ALL** | TTFT ms | 1785.6 | 1822.6 | **+37.0** | 1768.0 | 1790.0 |

## Verdict

**No measurable performance difference** — every delta is inside the run-to-run noise band:

- Within-run baseline spread was already ±2.6–9.4pp acceptance (chat 27.3→42.6 across its own 3 reps), i.e. ±1–2pp and ±1–2 t/s deltas are indistinguishable from noise.
- The direction of the small deltas is inconsistent across tasks (coding slightly up, chat/text slightly down) — a signature of noise, not an effect.
- TTFT did **not** improve despite the C128A adaptive-width backport theoretically shrinking the per-token indexer width to 128 at these short contexts. At 63–81 prompt tokens, TTFT (~1.8s) is dominated by per-request engine overhead/JIT, not by indexer width; any savings are below measurement resolution here. The 2106 ms patched-coding rep1 is a first-request outlier (same pattern as baseline's 2007 ms rep1).

## Interpretation

- The three backports remain **net-positive to keep**: they are upstream hotfixes that vLLM adopted *after* v0.26.0 (adaptive C128A width #52823, constexpr topk kernel #51967, 256 parallel combine workers #52084), and they show **no regression** in this harness. They align v0.26.0-exl3 with upstream correctness/robustness fixes and future-proof the code path.
- **Where gains could still appear but were not measured here:** long-context prefill (8k–128k). Adaptive width 128 vs full capacity (2048 at max-model-len 262144/128) matters proportionally more as the compressed sequence grows — the fixed full-width topk is wasteful exactly in the 128..2048 range the backport targets. The short-context synthetic harness cannot see that.
- Recommended follow-up if chasing prefill gains: `llama-bench.sh --model deepseek-v4-flash-0731-exl3-dspark --depth 0 4096 16384 65536 131072 --runs 3` (C1-only MD files) on the patched image, and compare prefill t/s at depth against the ghcr baseline — that is the instrument that isolates the indexer.

## Files

- baseline: `draft-acceptance-sweep-2026-08-22/k160.md`
- patched: `draft-acceptance-sweep-2026-08-22-patched/k160.md`
- this doc: `AB-patched-vs-baseline.md`