# 03 — Optimization Levers (ranked, evidence-based)

**Ground rule honored:** nothing below was changed. These are config/env sweeps or small, well-scoped code/backport candidates — each with a one-command test. All are **EXL3-stack validations to run on the bench harness** (`llama-bench.sh --model deepseek-v4-flash-0731-exl3-dspark`), **not** raw model-card claims.

---

## L1 — DSPark draft size K: **measured** (2026-08-22 sweep, K64→K192)

Empirical results from 3 cold-KV repeats per task (see `models/benchmarks/
deepseek-v4-flash-0731-exl3-dspark/draft-acceptance-sweep-2026-08-22/`):

| K | coding acc / t/s | chat acc / t/s | text acc / t/s |
|---|---|---|---|
| 64  | 35.6% / 33.3 | 29.2% / 28.4 | 35.7% / 32.6 |
| 96  | 39.1% / 35.1 | 29.9% / 28.7 | 34.9% / 32.2 |
| 128 | 40.6% / 36.2 | 30.2% / 29.0 | 38.5% / 34.2 |
| **160** | **45.3% / 39.3** | 35.0% / 31.9 | **39.7% / 35.0** |
| 192 | 43.9% / 38.2 | **39.5% / 34.5** | 38.2% / 34.2 |

- **K160 is the sweet spot** for coding/text (best acceptance + generation
  t/s). The stack now serves **K160** (`DSPARK_DRAFT_EXPERTS=160`).
- **K192 only wins on chat** (39.5% vs 35.0%) at ~89%-of-target draft cost —
  use it only if chat-heavy workloads dominate.
- Acceptance by draft position (K160, % of all draft tokens; sums to the
  total): coding 15.5/12.0/7.9/5.7/4.2, chat 15.1/9.2/5.9/3.0/1.8,
  text 14.9/9.9/6.7/5.0/3.2 — pos0–1 carry most of the gain.
- **KV tokens per request are K-independent** (~574–593 for these prompts).
- Method: per-request `vllm:spec_decode_num_accepted_tokens_total` /
  `num_draft_tokens_total` counter diffs, salted prompts (no prefix-cache
  reuse), 2×64-token JIT warmup, `max_tokens=512`, temperature 0.6.
- **Remaining L1 question:** `num_speculative_tokens` is K5 (REAP-validated);
  a 7-vs-5 spot check at K160 is optional (per-position acceptance decays
  fast: pos0 15.5% → pos4 4.2% at K160, so wider spec blocks add little).

## L2 — `num_speculative_tokens` (REAP K5 is validated; spot-check only)

- **K5 is the REAP-validated recipe.** The 0xSero REAP card requires the
  startup log to show "fixed K5 DSpark verification with a six-row C1 graph"
  and pins the whole runtime — K5 with a compact draft is their measured sweet
  spot on this EXL3/GB10 stack. (The base DeepSeek-V4-Flash-0731 card runs
  `num_speculative_tokens=7, draft_sample_method=greedy` — but that is the
  unquantized GB300/mega-moe recipe, not this stack.)
- **Our per-position data supports K5:** acceptance decays sharply across
  draft positions (K160: pos0 15.5% → pos4 4.2%), so positions 5+ add little.
- v0.26.0 permits any `>= dspark_block_size (5)`; the `n_predict` divisibility
  rule is MTP-only, so 7/9 are legal — but with low expected value here.
- **Do (cheap, optional):** single spot-check **7 vs 5** at K160 (same
  harness, one restart) to confirm no gain on this stack; skip the full
  {5,7,9}×{prob,greedy} grid unless the spot-check surprises.
- **SWA pad check** (`patch_sparse_swa_sm12x.py`): raw noncausal width stays
  `cdiv(128+9,128)*128 = 256` → padded to 512; the patch survives for 7/9.

## L3 — Profile the Trellis decode kernel efficiency (`m=1`, `block_m=8`)

### 3.0 Trellis parameter cheat-sheet (the knobs at play)

All values below are the **served config** (YAML `models/deepseek-v4-flash-0731-exl3-dspark.yaml` + defaults); the runtime banner confirms them at boot:

```
[exl3.py:1976] EXL3 rank-sliced runtime planned: Trellis m=1..32 block_m=8,
prefill trellis block_m=64 arena=392.1MiB capacity=2048 chunk=128 topk=6
```

| Env knob | Served | Default | Effect |
|---|---|---|---|
| `VLLM_EXL3_TRELLIS_MIN_M` | **1** (YAML) | 1 | Window start for the Trellis plan. Global `1` is **redundant for the draft** (`_rank_sliced_runtime` auto-defaults drafts to `MIN_CAPTURABLE_TRELLIS_M=1`); it matters only to force the **target** window to start at 1 — which is what unlocks FULL-cudagraph decode capture (see `05` §2). |
| `VLLM_EXL3_TRELLIS_MAX_M` | **32** (YAML) | 32 | Window end. `m > 32` and `≤ capacity(2048)` falls to the prefill Trellis plan. |
| `VLLM_EXL3_TRELLIS_BLOCK_M` | **8** (default; unset) | 8 | Decode-plan block rows. A single-token (`m=1`) decode runs through a kernel planned for 8-row blocks → sublinear per-token efficiency. **Primary remaining decode lever (candidate: 4/2).** |
| `VLLM_EXL3_PREFILL_TRELLIS` | **1** (YAML) | 1 | Enable the prefill Trellis plan (vs fallback path). |
| `VLLM_EXL3_PREFILL_CHUNK` | **128** (default; unset) | 128 | Prefill chunk size (tokens/step). |
| `VLLM_EXL3_PREFILL_BLOCK_M` | **64** (default; unset) | 64 | Prefill-plan block rows (arena 392.1 MiB/runtime at capacity 2048). |

Only `MIN_M`/`MAX_M`/`PREFILL_TRELLIS` are set in the YAML; the two `BLOCK_M` knobs and `PREFILL_CHUNK` ride the code defaults and are not yet pinned — a future tuner should pin them explicitly once a sweep decides values (see `05` §6).

### 3.1 Status after the attention-backport A/B

The **attention side is now closed**: `#52823` (adaptive C128A width), `#51967` (constexpr topk kernel) and `#52084` (256 combine workers) were backported (`7904d73`), the image rebuilt and A/B'd against the ghcr baseline on the K160 acceptance harness (`da3c1db`, 2026-08-22) — **no measurable difference** (all deltas inside the ±1–2pp / ±1–2 t/s noise band; see `../../models/benchmarks/deepseek-v4-flash-0731-exl3-dspark/draft-acceptance-sweep-2026-08-22-patched/AB-patched-vs-baseline.md`). Keep them (upstream-aligned, zero regression), but do **not** expect short-context gains from the attention side.

That leaves **the Trellis decode params as the only untested decode-side lever** — the `block_m=8`-at-`m=1` over-planning identified here, plus the capture-size set.

### 3.1b A/B outcome (2026-08-23) — the Trellis param levers are now **closed, empirically**

Two config A/B legs were executed on the bench harness (K160, 3 tasks × 3 reps, cold KV, override support added to `sweep_dspark_acceptance.py` via `--env-extra`/`--replace-arg`):

1. **`VLLM_EXL3_TRELLIS_BLOCK_M=4` — FAILED TO BOOT.** SparkInfer's planner rejects it:
   `ValueError: block_size_m must be one of (8, 16, 32, 48, 64), got 4`
   (container logs; engine-core crash-loop, `/health` never up — two attempts, 36 min and 10 min).
   **Consequence: `block_m=8` is already the minimum legal value.** The only legal alternatives (16/32/48/64) are strictly larger — worse for the `m=1` decode shape — so the "reduce decode block" idea has no valid setting on this SparkInfer build. Evidence: `draft-acceptance-sweep-2026-08-23-blockm4/FAILED-blockm4-evidence.md`.
2. **Capture sizes `1 2 4 6` → `1 2 3 5 6` — NEUTRAL.** All deltas inside the harness noise band (ALL: acc −0.8pp, gen −0.26 t/s, TTFT −10.6 ms; per-task spreads −2.0…+0.6). Evidence: `draft-acceptance-sweep-2026-08-23-capsiz/k160.md`.

**Bottom line for L3:** the `m=1`-vs-`block_m=8` sublinearity is real but **not config-addressable** — the decode block is pinned at its legal minimum. Only a kernel profile (`nsys`/`ncu`) can quantify it further; treat it as a known, accepted inefficiency of the single-token spec-decode path (it applies to every token, draft+verify, so it is by design on this stack). Remaining untested config levers: **L2** (spec-token count/sampling) and **L6** (prefill `BLOCK_M`/`CHUNK`).

### 3.2 Why it matters

- **Why.** The single-token decode is the throughput ceiling; if SparkInfer's Trellis `bind`/`run` (or the tiled kernel at `m=1`) over-fetches expert tiles or re-derives descriptors per layer, that is the loss. We cannot confirm it from the repo alone — this is the one thing only a profile or an env A/B can settle.
- **Do (offline, local):** `nsys profile` / `ncu --set full` (or `CUDA_LAUNCH_BLOCKING` + `torch.profiler`) a short decode burst in the container; look for: kernel utilization at `m=1`, HBM read vs theoretical `13B-active × 3.0bpw` per token, and per-layer bind/run gaps. **Still the only open item in L3** — everything config-level below is now executed.
- **Done (env A/Bs, 2026-08-23):**
  1. `VLLM_EXL3_TRELLIS_BLOCK_M=4` — **rejected by the planner** (`block_size_m must be one of (8, 16, 32, 48, 64)`); `8` is the legal minimum → sub-lever closed (§3.1b).
  2. Capture sizes `1 2 4 6` → `1 2 3 5 6` — **neutral** (§3.1b).

## L4 — `shared_experts` / `swiglu_limit` handling — **verified 2026-08-23 (read-only)**

**Verdict: shared expert ✅ correctly executed; routed-expert `swiglu_limit` clamp ⚠️ not applied in the visible path (opaque kernel caveat).**

### 4.1 Shared expert — RESOLVED, correctly handled

- **Config/weights:** `config.json` has `n_shared_experts=1`, and the checkpoint carries
  `layers.{0..42}.ffn.shared_experts.w{1,2,3}` + FP8 `.scale` (276 tensors in the index) —
  they are **FP8 block-quantized**, and `Exl3Config` explicitly exposes the FP8 attrs so
  they route to `Fp8LinearMethod` (exl3.py:319–332). They load and run.
- **Execution:** the DSV4 wrapper (`vllm/models/deepseek_v4/nvidia/model.py:585–590, 728–730`)
  builds `DeepseekV4MLP` (its own `SiluAndMulWithClamp(10.0)` act_fn, line 130–131) for the
  shared expert. On the served CUDA/TP1 path (`use_mega_moe=False`, `_forward_fused_moe`),
  the **MoERunner** computes the shared expert separately
  (`runner/moe_runner.py:538–545 _maybe_apply_shared_experts`) and adds it to the routed
  output — in-kernel shared-expert *fusion* is ROCm-only
  (`is_fusion_moe_shared_experts_enabled`, layer.py:84–96) and inactive on CUDA.
- **The `del shared_experts, shared_experts_input` in `Exl3MoEMethod.apply` (exl3.py:2138)
  is benign**: it only refuses in-kernel fusion; the runner adds the shared output.
- **Draft:** `build_dspark_draft.py` emits **no** shared-expert tensors (0 hits) — the K160
  compact draft has none, consistent with the MTP design. Nothing missing.

### 4.2 `swiglu_limit=10.0` clamp on ROUTED experts — NOT applied (visible path)

- **Reference semantics** (stock vLLM): `hidden = silu(clamp(gate, ±10)) * clamp(up, ±10)`
  — `fused_moe/activation.py:118–122`; stock modular kernels pass `layer.swiglu_limit` as
  `clamp_limit` (`modular_kernel.py:889–899`). `RoutedExperts` stores it (routed_experts.py:111).
- **EXL3 path:** neither the Trellis hot path (`_apply_rank_sliced` → `api.bind(plan, scratch,
  a, weights, topk_weights, topk_ids)` — no clamp/act argument) nor the eager parity path
  (`hidden = silu(gate) * up`, exl3.py:2176) applies any clamp. The overlay never reads
  `layer.swiglu_limit`.
- **Caveat:** the SiLU itself must already be baked inside SparkInfer's Trellis kernel
  (opaque, not in this repo) — so a hardtanh/clamp *could* also live there. It cannot be
  confirmed or excluded from the repo alone.
- **Assessment:** the clamp is a magnitude-safety device for the reference (fp8/fp16
  overflow avoidance on rare outlier tokens, |gate|/|up|>10). With EXL3 3.0 bpw weights the
  practical effect of dropping it is low-frequency numeric deviation, typically small — a
  **fidelity nit, not a correctness bug**, and it is **not actionable from this repo**
  (the gate/up happen inside the kernel; no hook exists to clamp between the two stages).

### 4.3 Recommended follow-up (cheap, build-time)

Add a build-time assertion in the image Dockerfile (or extend `verify_exl3.py`) that greps
the installed `sparkinfer/moe/trellis_moe/_impl.py` for a clamp/hardtanh in the run path —
turning the one opaque unknown into a pass/fail check at image build. Until then: **accepted
deviation**, low urgency.

## L5 — Upstream attention/spec-decode patches (backported, A/B'd — closed)

The repo already has `#49486`, `#50298`, `#50365`. The remaining small, drop-in candidates from latest main (details in `04`):
- **`#52823` adaptive topk width** (aliases `#50004`) — `sparse_mla.py`, ~24 lines. **Highest-value un-adopted attention patch.**
- **`#51967` top-k index kernel compile-time constants** — `cache_utils.py`, +5.
- **`#52084` sparse top-k metadata prefill** — `cache_utils.py`, +1.
- **`#50911` fused non-causal TokenSpeed MLA for DSpark** — `tokenspeed_mla.py`, +8 (DSpark decode path).

**Status (2026-08-22): closed.** `#52823` + `#51967` + `#52084` are backported (`7904d73`, each a tested `patch_*.py` with fail-closed anchors), the image was rebuilt and A/B'd against the ghcr baseline on the K160 acceptance harness — **no measurable difference, no regression** (see `AB-patched-vs-baseline.md` in the `-patched` sweep dir). `#50911` remains N/A (TokenSpeedMLA is not on the DSv4 path in v0.26.0 and its capability gate is `major==10`; GB10 is `sm_121a` → not selectable). No further action needed on this lever; the attention side is considered closed for short-context workloads.

## L6 — Prefill knobs (untuned)

- `VLLM_EXL3_PREFILL_BLOCK_M` (default 64) — sweep {32, 64, 128}; `VLLM_EXL3_PREFILL_CHUNK` (default 128) — sweep {128, 256}.
- Prefill already strong (≈1.3k t/s short-ctx); expect single-digit % TTFT improvements at most capital — low priority but cheap to try.

## L7 — Memory headroom (not t/s, but context/KV ceiling)

- `gpu-memory-utilization 0.9` is near the KV cliff at 262k / 2 seq. Two prefill Trellis arenas (target + draft) are resident (banner: 392.1 MiB each at capacity 2048; code sizes them ~1 GiB). If L1 lowers the draft to a smaller plan, the draft arena shrinks too, freeing context/KV headroom (raise `--max-num-seqs` or `--max-model-len`).

---

## Decision priority (readiness)

| # | Lever | Type | Effort | Payoff | Status |
|---|---|---|---|---|---|
| L1 | Draft K sweep | config | done | **measured** | ✅ **K160 applied** (coding 45.3%/39.3, text 39.7%/35.0, chat 35.0%/31.9) |
| L2 | Spec-token × sampling (5 vs 7/greedy) | config | trivial | low | ✅ **K5 is the REAP-validated recipe** (startup-log pin) + our per-position decay data support it — only an optional cheap 7-vs-5 spot-check remains (skip unless desired) |
| L3 | Trellis decode params (`BLOCK_M`, capture sizes) | config/env | done | — | ✅ **closed empirically** — `BLOCK_M=4` rejected by planner (legal set 8/16/32/48/64; 8 = minimum, already served); capture sizes `1 2 3 5 6` neutral. Only a kernel profile remains (diagnostic) |
| L5 | 3 small upstream patches (#52823/#51967/#52084) | code | done | medium | ✅ **backported (`7904d73`), rebuilt, A/B'd — no measurable diff, no regression** (`da3c1db`). #50911 N/A (SM121 / not on DSv4 path) |
| L4 | shared/clamp verify | code | done | correctness | ✅ **verified 2026-08-23** — shared expert executed (FP8, runner path, own clamp); routed-expert `swiglu_limit` clamp not applied in visible path (opaque-kernel caveat; build-time grep suggested, §4.3) |
| L6 | Prefill knobs (`PREFILL_BLOCK_M`/`CHUNK`) | config | trivial | low | optional |
| L7 | Memory/headroom | config | trivial | indirect | after L2 |

The fastest win is **L1×L2 (25%+ of a config edit; restart; bench)** — no image rebuild, no code. Decode-side levers are now **exhausted at the config level**: the attention patches (L5) measured neutral in A/B, and the Trellis decode params (L3) are closed empirically — `BLOCK_M=8` is the legal minimum (4 is rejected by the planner) and capture-size alignment is neutral. L4 (shared/clamp) is **verified** (shared expert OK; routed clamp dropped — fidelity nit, build-time grep suggested in §4.3). What remains: **L2** optional 7-vs-5 spot-check, L6 (prefill knobs), kernel profile (diagnostic). Everything else is fine-tuning or housekeeping (`05`).
