# 03 — Optimization Levers (ranked, evidence-based)

**Ground rule honored:** nothing below was changed. These are config/env sweeps or small, well-scoped code/backport candidates — each with a one-command test. All are **EXL3-stack validations to run on the bench harness** (`llama-bench.sh --model deepseek-v4-flash-0731-exl3-dspark`), **not** raw model-card claims.

---

## L1 — Sweep the DSPark draft size K (highest expected leverage on decode)

- **Why.** At `K192/216 ≈ 89%`, the draft forward costs almost as much weight traffic as the target verify, so speculative decode reads ≈ 2× active weights per output token. The skill previously recommended K64→K96/K128; the current repo is at **K192** (deepest so far). K192 trades barely any memory and nearly doubles per-step cost, so the acceptance-rate gain must be large enough to win — measure it.
- **Do (no rebuild — env-only, restart):** set `DSPARK_DRAFT_EXPERTS ∈ {64, 96, 128, 160}` in the YAML, rebuild draft dir, re-bench `tg32 @ d1024` + a long-context decode, pick the max **accepted/verified t/s**.
- **Watch:** draft quality (acceptance) vs draft cost (weight bytes/step) — the sweet spot is where `accepted_tokens/step / draft_cost` peaks, not where acceptance is highest.
- **Expected:** if acceptance at K128 is within noise of K192, K128 (or lower) can raise decode t/s materially by cutting the draft pass.

## L2 — Raise `num_speculative_tokens` above the floor (5)

- **Why.** `num_speculative_tokens=5` is the `dspark_block_size` floor (YAML comment). With a strong draft, longer spec runs amortize the fixed per-step Trellis launch + bind overhead and yield more than one verified token per target pass.
- **Do:** set `num_speculative_tokens ∈ {5, 7, 9}` in the speculative-config; re-bench. **Constraint:** must be ≥ `dspark_block_size`; verify SWA pad `patch_sparse_swa_sm12x` still hits a dispatchable topk ({128,512,1024}) at the chosen value (the 256→512 padding already covers 5; larger spec tokens push `window+spec` up — re-check the `_raw_noncausal_width` lands in {128,512,1024}).
- **L1×L2 is the joint sweep that matters** — run as a small grid, not individually.

## L3 — Profile the Trellis decode kernel efficiency (`m=1`, `block_m=8`)

- **Why.** The single-token decode is the throughput ceiling; if SparkInfer's Trellis `bind`/`run` (or the tiled kernel at `m=1`) over-fetches expert tiles or re-derives descriptors per layer, that is the loss. We cannot run it here (no remote vLLM), so this is the one thing only a profile can confirm.
- **Do (offline, local):** `nsys profile` / `ncu --set full` (or `CUDA_LAUNCH_BLOCKING` + `torch.profiler`) a short decode burst in the container; look for: kernel utilization at `m=1`, HBM read vs theoretical `13B-active × 3.0bpw` per token, and per-layer bind/run gaps.
- **Door opened:** if `m=1` Trellis efficiency is poor, evaluate `VLLM_EXL3_TRELLIS_BLOCK_M` (reduce decode block to 4/2?) and whether capture sizes `1 2 4 6` should be `1 2 3 5 6` to match actual speculative batch distribution.

## L4 — Verify `swiglu_limit`/`shared_experts` handling (correctness gate, not perf)

- **Why (evidence).** `config.json` has `n_shared_experts=1` and `swiglu_limit=10.0`, but `Exl3MoEMethod.apply` does `del shared_experts, shared_experts_input` and has **no `activation_clamp` parameter** (stock `FusedMoE.apply` applies `swiglu_limit`). The model "works", so either the shared expert and clamp are applied elsewhere/inside the Trellis kernel, or output drifts slightly.
- **Do:** confirm in the Trellis `run` path + `sparse_mla`/model wrapper that (a) the single shared expert's contribution is added Hunger and (b) the `swiglu_limit=10.0` clamp is applied. If not, that's a **quality** bug to fix (not t/s). Low urgency, but should be settled before trusting long outputs.

## L5 — Add the 4 small upstream attention/spec-decode patches the repo lacks

The repo already has `#49486`, `#50298`, `#50365`. The remaining small, drop-in candidates from latest main (details in `04`):
- **`#52823` adaptive topk width** (aliases `#50004`) — `sparse_mla.py`, ~24 lines. **Highest-value un-adopted attention patch.**
- **`#51967` top-k index kernel compile-time constants** — `cache_utils.py`, +5.
- **`#52084` sparse top-k metadata prefill** — `cache_utils.py`, +1.
- **`#50911` fused non-causal TokenSpeed MLA for DSpark** — `tokenspeed_mla.py`, +8 (DSpark decode path).
Port each as a new `patch_*.py` in the image dir (same fail-closed backport discipline), then A/B prefill/decode.

## L6 — Prefill knobs (untuned)

- `VLLM_EXL3_PREFILL_BLOCK_M` (default 64) — sweep {32, 64, 128}; `VLLM_EXL3_PREFILL_CHUNK` (default 128) — sweep {128, 256}.
- Prefill already strong (≈1.3k t/s short-ctx); expect single-digit % TTFT improvements at most capital — low priority but cheap to try.

## L7 — Memory headroom (not t/s, but context/KV ceiling)

- `gpu-memory-utilization 0.9` is near the KV cliff at 262k / 2 seq. Two ~1 GiB prefill Trellis arenas (target + draft) are resident. If L1 lowers the draft to a smaller plan, the draft arena shrinks too, freeing context/KV headroom (raise `--max-num-seqs` or `--max-model-len`).

---

## Decision priority (readiness)

| # | Lever | Type | Effort | Payoff | Do first? |
|---|---|---|---|---|---|
| L1 | Draft K sweep | config | trivial | high | ✅ |
| L2 | Spec-token sweep | config | trivial | high | ✅ (joint w/ L1) |
| L3 | Trellis decode profile | tooling | medium | diagnostic | ✅ (informs L1/L2) |
| L5 | 4 small upstream patches | code | low–med | medium | after L1–L3 |
| L4 | shared/clamp verify | code | low | correctness | with L5 |
| L6 | Prefill knobs | config | trivial | low | optional |
| L7 | Memory/headroom | config | trivial | indirect | after L1 |

The fastest win is **L1×L2 (25%+ of a config edit; restart; bench)** — no image rebuild, no code. L5 is the best *code* ROI. Everything else is fine-tuning or diagnostic.
