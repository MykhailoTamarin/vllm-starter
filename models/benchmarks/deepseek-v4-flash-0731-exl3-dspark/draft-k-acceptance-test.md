# DSPark draft K — acceptance-rate comparison (chat + coding)

Test date: 2026-08-21
Model: `deepseek-v4-flash-0731-exl3-dspark` (vllm-exl3-v26:latest, EXL3 3.0bpw, REAP K216)
Spec decode: native DSPark, `num_speculative_tokens=5`, `draft_sample_method=probabilistic`
Request settings: temperature 0.6, `max_tokens=700`, `chat_template_kwargs.thinking=false`,
`--max-num-seqs 2` (reduced from 3), `--gpu-memory-utilization 0.9`, `--max-model-len 262144`.

## How acceptance rate is measured

vLLM emits `SpecDecoding metrics:` INFO lines every `VLLM_LOG_STATS_INTERVAL` (set to 10s for this
test) while the engine is generating. Per window it reports drafted/accepted tokens and per-position
(1..5) acceptance. Two chat samples + two coding samples were run per draft size; the windows from
all samples are aggregated into one rate per task:

> **acceptance rate = Σ accepted draft tokens / Σ drafted tokens**

Mean acceptance length is `1 + accepted / drafts` (vLLM convention; drafts = drafted/5).

## Test prompts

- **Chat:** open-ended explanatory prompt (~700 generated tokens) — prose / natural language.
- **Coding:** "write `find_anagrams(word, candidates)` + asserts" (~300 generated tokens) — code.

## Results — acceptance rate

| Draft | Chat draft tokens | Chat accepted | **Chat rate** | Chat MAL | Code draft tokens | Code accepted | **Code rate** | Code MAL |
|---|---|---|---|---|---|---|---|---|
| K64  | 2640 | 907 | **34.4%** | 2.72 | 1030 | 374 | **36.3%** | 2.82 |
| K128 | 2485 | 852 | **34.3%** | 2.71 | 1025 | 454 | **44.3%** | 3.21 |
| K144 | 2580 | 957 | **37.1%** | 2.85 | 1000 | 450 | **45.0%** | 3.25 |
| K160 | 2480 | 978 | **39.4%** | 2.97 | 990  | 460 | **46.5%** | 3.32 |
| K176 | 2085 | 937 | **44.9%** | 3.25 | 1035 | 409 | **39.5%** | 2.98 |
| K192 | 2045 | 792 | **38.7%** | 2.94 | 1015 | 512 | **50.4%** | 3.52 |

MAL = mean acceptance length. Draft tokens counts are the accumulated windows captured for that size.

## Per-position acceptance rate (positions 1–5)

| Draft | Chat p1 | p2 | p3 | p4 | p5 | Code p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|---|---|---|---|
| K64  | 0.720 | 0.496 | 0.267 | 0.153 | 0.082 | 0.752 | 0.480 | 0.301 | 0.189 | 0.092 |
| K128 | 0.716 | 0.487 | 0.274 | 0.153 | 0.085 | 0.795 | 0.600 | 0.414 | 0.239 | 0.166 |
| K144 | 0.744 | 0.527 | 0.308 | 0.178 | 0.097 | 0.805 | 0.570 | 0.390 | 0.280 | 0.205 |
| K160 | 0.770 | 0.534 | 0.345 | 0.216 | 0.107 | 0.813 | 0.601 | 0.420 | 0.298 | 0.192 |
| K176 | 0.782 | 0.590 | 0.413 | 0.276 | 0.187 | 0.763 | 0.565 | 0.334 | 0.198 | 0.116 |
| K192 | 0.760 | 0.528 | 0.350 | 0.196 | 0.103 | 0.803 | 0.665 | 0.483 | 0.340 | 0.232 |

## Memory / KV trade-off (startup log, `--max-num-seqs 2`, util 0.9)

| Draft | Weight (GiB) | GPU KV cache | Concurrency @ 262k |
|---|---|---|---|
| K64  | 93.87 | 1,072,756 | 4.09x |
| K128 | 96.27 | 922,381 | 3.52x |
| K144 | 96.86 | 755,298 | 2.88x |
| K160 | 97.46 | 815,386 | 3.11x |
| K176 | 98.06 | 624,952 | 2.38x |
| K192 | 98.66 | 674,247 | 2.57x |

All sizes keep ≥2x concurrency @ 262k, so `--max-num-seqs 2` (2 concurrent full-context requests)
is satisfied by every K. At `--max-num-seqs 3`, K176 (2.38x) and K192 (2.57x) would NOT support 3
concurrent full-context requests — consistent with "192 was too much" at concurrency 3.

## Interpretation

- **Bigger drafts raise acceptance.** Chat: ~34% (K64/K128) → 37–39% (K144/K160) → peak ~45% (K176).
  Code: ~36% (K64) → 44–47% (K128–K160) → peak ~50% (K192).
- **Per-position gains concentrate in later positions** (p3–p5) — the region the smaller drafts miss.
  E.g. chat p5: 0.082 (K64) → 0.187 (K176).
- **Sample noise is significant** (single 10s windows span ~27–58%; ~2–2.6k draft tokens/size for chat,
  ~1k for code). The K176 chat and K192 code peaks should be treated as ±~3–5pp.
- **Memory cost per +16 experts ≈ 0.6 GiB weight.** K160 adds ~1.3 GiB vs K128; K176 ~1.8 GiB; K192 ~2.4 GiB.
- **Sweet spot:** K160 is the largest size that keeps ≥3x full-context concurrency (if ever needed)
  and improves acceptance vs K128 by ~+5pp (chat) / ~+2pp (code). K176 gives the best chat acceptance
  (~45%) at 2.38x concurrency. K192 gives the best code acceptance (~50%) at 2.57x.

## Raw per-window data

### Chat (accepted/drafted, per 10s window)

| Draft | w1 | w2 | w3 | w4 | w5 |
|---|---|---|---|---|---|
| K64  | 249/515 (48.3%) | 147/540 (27.2%) | 157/535 (29.3%) | 181/515 (35.1%) | 173/535 (32.3%) |
| K128 | 135/375 (36.0%) | 158/535 (29.5%) | 258/510 (50.6%) | 150/535 (28.0%) | 151/530 (28.5%) |
| K144 | 273/510 (53.5%) | 144/535 (26.9%) | 149/520 (28.7%) | 224/500 (44.8%) | 167/515 (32.4%) |
| K160 | 290/500 (58.0%) | 167/530 (31.5%) | 189/530 (35.7%) | 164/390 (42.1%) | 168/530 (31.7%) |
| K176 | 270/510 (52.9%) | 182/535 (34.0%) | 291/510 (57.1%) | 194/530 (36.6%) | — |
| K192 | 175/465 (37.6%) | 206/535 (38.5%) | 209/515 (40.6%) | 202/530 (38.1%) | — |

### Code (accepted/drafted, per 10s window)

| Draft | w1 | w2 |
|---|---|---|
| K64  | 228/520 (43.8%) | 146/510 (28.6%) |
| K128 | 190/510 (37.3%) | 264/515 (51.3%) |
| K144 | 256/510 (50.2%) | 194/490 (39.6%) |
| K160 | 259/480 (54.0%) | 201/510 (39.4%) |
| K176 | 192/515 (37.3%) | 217/520 (41.7%) |
| K192 | 266/505 (52.7%) | 246/510 (48.2%) |