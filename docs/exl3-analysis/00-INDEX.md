# EXL3 vLLM Image — Deep Performance & Optimization Analysis

**Date:** 2026-08-22 (EEST)
**Analyzed against:** `vllm/vllm-openai:v0.26.0` base, image `vllm-exl3-v26` (repo `images/vllm-0_26_0-exl3/`), model `deepseek-v4-flash-0731-exl3-dspark` (REAP-K216, EXL3 3.0 bpw, 43 layers), DSPark compact K192 draft.
**Scope:** read-only analysis — **nothing was changed or benchmarked**. Latest upstream vLLM (`origin/main` @ `236f78cc5c`, ≈ `v0.27.2rc0-419`, heading to v0.28) was diffed against `v0.26.0` (`f2654939e6`) for adoptable changes.

---

## 1. Headline results

| Question | Answer |
|---|---|
| **Are we losing performance today?** | Not catastrophically, but there are **two structural inefficiencies on the decode hot path**: (1) the DSPark draft was near full-size (K192 = ~89% of target), roughly doubling per-step weight traffic — **fixed by the K160 switch** (measured 2026-08-22); (2) the target decode path is a single-token `m=1` Trellis MoE with `block_m=8`, i.e. the kernel is planned for 8-row blocks. Both are tunable/configurable (no code change). |
| **Is the router slow?** | **No for 40/43 layers.** `patch_dsv4_topk_k216.py` is correct and active: the model is `topk_method=noaux_tc` + fp32 gating + `num_experts_per_tok=6` + `norm_topk_prob=True`, so `can_use_dsv4_topk()` returns True and the **fused Triton `dsv4_topk` kernel runs** on layers ≥3. The **first 3 hash layers** (`num_hash_layers=3`) inherently take the slower pure-Torch fallback (hash routing cannot use `dsv4_topk`). Minor, ~7% of layers. |
| **Is the router patch still needed upstream?** | **Yes.** The `(256, 384)` whitelist in `dsv4_topk.py` is **byte-identical on latest main** — upstream has not relaxed it. `patch_dsv4_topk_k216` remains required; no upstream alternative exists yet. |
| **What draft size is best?** | **K160** (sweep, 2026-08-22): coding 45.3% acceptance / 39.3 t/s, text 39.7% / 35.0, chat 35.0% / 31.9. K192 only beats it on chat (39.5% / 34.5). The server now serves **K160**. |
| **Space for optimization?** | Yes — ranked in `03-optimization-levers.md`. Next lever: **`num_speculative_tokens {5,7,9} × draft_sample_method {probabilistic,greedy}` at K160** — the official DeepSeek-V4-Flash card runs 7+greedy (we serve 5+probabilistic); then 4 small already-upstream attention/indexer patches (`#52823`, `#51967`, `#52084`, `#50911`). |
| **Anything upstream to adopt?** | Yes — 4–6 small attention/spec-decode patches (see `04`). The 3 big ones the repo already backported (`#49486`, `#50298`, `#50365`) are current and correct. |

---

## 2. Current performance baseline (source of truth: `models/benchmarks/.../benchmark_21_08_26_15_26_c1_d0-253952.md`)

| Metric | Value |
|---|---|
| Prefill (pp2048) | **1290–1320 t/s** (0–16k ctx); degrades to ~1086 t/s @ 253k ctx |
| TTFT (llama-bench) | ~1.6s @ no ctx → 235.8s @ 253k ctx |
| Decode (tg32) | **22.6–34.6 t/s**, mean ~27–31; high variance (±2–5) |
| **Decode, measured sweep (K160, 3 cold runs, 512 tok)** | **coding 39.3, text 35.0, chat 31.9 t/s** (incl. reasoning tokens); K192: 38.2 / 34.2 / 34.5 |
| **Acceptance (K160)** | coding 45.3%, text 39.7%, chat 35.0% (K192: 43.9 / 38.2 / 39.5) |
| Model card (root README) | 0.98–1.13k t/s prefill, 24–33 t/s gen, TTFT 62.3s @ 64k |

Decode variance is high (`±2.9–5.6`), suggesting sensitivity to batch `m`, CUDA-graph bucket switching, and possibly the draft verify pattern — not a stable memory-bound plateau predictably. See `02` and `03`.

---

## 3. Report map

| File | Contents |
|---|---|
| `01-patch-and-image-inventory.md` | Complete inventory of every patch + overlay component, what it does, its perf impact |
| `02-decode-hotpath-and-router-analysis.md` | Deep dive: router engagement, EXL3 Trellis decode path, DSPark two-pass, prefill |
| `03-optimization-levers.md` | Ranked, actionable levers with measured sweep results (K160 applied) + A-B test plans |
| `04-upstream-v026-vs-main-adoption.md` | Latest-vLLM diff: adoptable candidates, already-backported set, non-transferable GEMM wins |
| `05-maintenance-and-risks.md` | Dead code, fragilities, README/version drift, upgrade-anchor risks |
| `../../models/benchmarks/<model>/draft-acceptance-sweep-2026-08-22/` | Raw per-K reports (`k64.md`…`k192.md`) + combined summary |

---

## 4. One-paragraph summary

The image is well-engineered and already carries the three most impactful upstream attention hotfixes for v0.26.0 (`#49486`, `#50298`, `#50365`) plus working 216-expert fused routing. The decode hot path is dominated by the EXL3 Trellis MoE executed twice per output token (draft forward + target verify) at tiny `m`, which is bandwidth-inefficient by design. An acceptance sweep (K64→K192, coding/chat/text) showed **K160 is the sweet spot** — coding 45.3% acceptance / 39.3 t/s, text 39.7% / 35.0, chat 35.0% / 31.9 — and the server now runs **K160** (`DSPARK_DRAFT_EXPERTS=160`; K192 only wins on chat at ~89% draft cost). The next config lever is **`num_speculative_tokens` 5→7/9 at K160**; the fastest *code* wins remain the four small upstream patches the repo lacks (`#52823`, `#51967`, `#52084`, `#50911`). The `v0.28`/KV-layout refactor is not portable — it belongs to a full version bump, not a backport.
