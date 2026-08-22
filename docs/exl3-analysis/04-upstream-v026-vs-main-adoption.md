# 04 — Latest vLLM (origin/main) vs v0.26.0 — what to adopt

**Compared:** `v0.26.0` (`f2654939e6`) → `origin/main` (`236f78cc5c`, ≈`v0.27.2rc0-419`; v0.28.0rc1/rc2 tagged). Read-only `git log/diff` — **no checkout, no writes** (worktree stayed on v0.26.0).

**Adoptability rule applied** (user-stated): EXL3 replaces only the weight-quant GEMM; native DeepSeek-V4 attention/router/spec-decode orchestration are shared ⇒ attention-path, router, and spec-decode orchestration changes = **CANDIDATES**; weight/GEMM-kernel changes = **not transferable**.

## 1. Already on the EXL3 image (do not re-add)

| Commit | PR | What | Status |
|---|---|---|---|
| `837eae6458` | #50298 | remove redundant full `combined_indices` kernel | ✅ backported (`patch_attn_combined_indices.py`) |
| `9e6be4a72b` | #50365 | drop index-remap atomics | ✅ backported (`patch_attn_index_remap.py`) |
| (`b0cb1da1`) | #49486 | short-context topk skip | ✅ backported (`patch_attn_short_ctx_topk.py`) |

These three are the biggest attention-path wins in this window and the repo already has them — i.e. the image is **not missing the 0.27 attention hotfix set**. (Note: `#50004` adaptive topk width — not yet present; see §2.)

## 2. Small adoptable candidates (recommended next)

| Commit | PR | File(s) touched | Δ | Why adopt |
|---|---|---|---|---|
| `e6f35d3c69` | **#52823** "Adaptive topk width for dsv4" (re-back of #50004) | `deepseek_v4/sparse_mla.py` | +24 | The skill's #1 un-adopted attention candidate; TTFT/prefill indexer win. **Highest value.** |
| `83f591d7f6` | #51967 "top-k index kernel w/ compile-time constants" | `deepseek_v4/common/ops/cache_utils.py` | +5 | Tiny, indexer hot path. |
| `836aac92ff` | #52084 "sparse top-k metadata kernels (prefill)" | `deepseek_v4/common/ops/cache_utils.py` | +1 | Tiny, prefill metadata. |
| `8ae8337ffa` | #50911 "fused non-causal TokenSpeed MLA for DSpark" | `v1/attention/backends/mla/tokenspeed_mla.py` | +8 | DSpark decode attention path. |
| `bb3b61f2fd` | #49618 "dispatch bias-less topk routing to fused path" | `fused_moe/router/router_factory.py` | +66 | Routing orchestration — but low value for the 216-expert sqrtsoftplus path (uses `FusedTopKBiasRouter`+`dsv4_topk`, not the no-bias factory path). Marginal. |

All five are **small, fail-closable backports** onto v0.26.0 — same `patch_*.py` pattern already used. **`#52823` is the clear first one** (it resurrects exactly the `#50004` the skill flagged as a candidate and the repo skipped).

## 3. Larger spec-decode candidates (bigger engineering, A/B needed)

| Commit | PR | What | Verdict |
|---|---|---|---|
| `7f7a32cfec` | **#47808** "DSpark confidence-scheduled verification" | 477-line adaptive-verification infra; touches `model_runner.py`, `cudagraph_utils.py`, `dspark/speculator.py`, `sparse_mla.py`, indexer | **Candidate but large.** Changes DSPark verification timing — could raise verified decode throughput, but it is a substantial port and **must be A/B-tested against EXL3's FULL-cudagraph** decode before adopting. Not a weekend backport. |
| `5df31ea52d` | #52795 "Enable adaptive verification on DSv4 + sm90" | `mla/indexer.py` +32 | Depends on #47808 infra; standalone value is nil. |

Recommend: **track these** for a v0.28 bump rather than backporting to v0.26.0.

## 4. Not adoptable (EXL3 replaces the GEMM / wrong platform)

- `4f6885fffc` #53040 fused shared experts into **MegaMoE** — GEMM/weight path (and MegaMoE off here).
- `fe76112ff2`, `d626108b18`, `6b68db441e`, `ef43e3101b`, `12a34a6bc7`, `5d8a4cf976`, `9b0ab5dd53`, `fe1c317157`, `8bdc70ec7b`(ROCm/KV-layout) — **ROCm/AITER or AMD-only**; N/A on NVIDIA SM121.
- `788a…`/deep_gemm, `da788334bc` #47972 NVFP4 emulation, `4eef91c03c` R3+DeepGEMM MegaMoE — weight/GEMM path.
- `bf2b45b5d6` #42669 FlashAttention4 SM100 — SM100 (Blackwell Ultra), not SM121; FlashInfer MLA path here.
- `e4d61d0d22`, `13726c80fe` CPU MLA backends — N/A (GPU image).

## 5. Structural refactors — do NOT backport (wait for a version bump)

- KV-cache layout refactor series: `57bd0ed441` [5/N], `8bdc70ec7b` [6/N] (#51704/#51718), `d6c2fec9fd` (#53139 remove FlashInfer DSpark DCP), `a0cd2b69b3` (#50302 block-table width align), `967e104fad` (#52550 indexer cache dtype).
- `6e311c6e20` #44941 FusedMoE→FusedMoEFactory refactor.
- `b38e111d3e` #50613 per-request MLA chunked-context scheduling (medium; candidate on a bump).
- `38a466e7b6` #46789 DSV4 Sequence Parallelism (feature, N/A single-GPU TP1).

These are invasive/structural; cherry-picking them onto v0.26.0 would fight the patch-anchor model (`vllm-version-upgrade` skill). They rightfully belong to a **v0.28 (or v0.27.x) full-image bump** — which, per the version-upgrade skill, requires a full rebuild (PyTorch 2.13 in v0.27.x is a breaking env change; **never copy upstream anchors raw — re-derive them**).

## 6. Confirmation: `dsv4_topk` whitelist still `(256, 384)` on main

`git diff v0.26.0 origin/main -- vllm/model_executor/layers/fused_moe/router/dsv4_topk.py` → **0 lines**. Upstream has NOT relaxed the whitelist for arbitrary expert counts (e.g. 216). So `patch_dsv4_topk_k216.py` remains **required and non-superseded**; there is no upstream-native 216-expert fused topk to adopt instead)Skip. Good — the repo's approach is still the right one.

## 7. Bottom line for adoption

| Priority | Candidate | Effort |
|---|---|---|
| 1 | **#52823** adaptive topk width | small backport |
| 2 | #51967, #52084 (index/metadata kernels) | 5–6 lines each |
| 3 | #50911 (TokenSpeed MLA for DSpark) | tiny |
| 4 | #49618 (router dispatch) | optional/marginal |
| track | #47808 + #52795 (adaptive verification) | large; A/B required |
| don't | KV-layout refactor, GEMM/MegaMoE, ROCm set | bump-only / N/A |
