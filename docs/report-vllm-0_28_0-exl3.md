# vLLM 0.28 Rebase Report — `images/vllm-0_28_0-exl3`

**Date:** 2026-09-02
**Status:** ✅ Model up and serving on the new image (`vllm-exl3-v28:latest`)

---

## 1. Summary

Created `images/vllm-0_28_0-exl3/` — a **from-scratch** vLLM image built from
the latest `vllm-project/vllm` main (pinned commit
`62588e0592ad5af8d3d4a536ace01afb44ebaed5`, i.e. v0.28.0 + 289 commits past
v0.28.1rc0). The wheel is compiled in a builder stage, then installed into a
minimal `nvidia/cuda:13.0.3-base-ubuntu24.04` runtime — **no
`vllm/vllm-openai` base image**, per the requirement to build everything from
scratch, as light and as performant as possible.

| Component | 0.26 image | 0.28 image (new) |
|---|---|---|
| vLLM | 0.26.0-exl3.dspark.sm121 | **0.28.0+exl3.dspark.sm121** |
| vLLM pin | — | `62588e05` (289 commits past v0.28.1rc0; DSML parser #54838, DSV4 SWA RoPE #54815, FlashInfer warmup #396c5a563 native) |
| torch | 2.11.0+cu130 | **2.13.0+cu130** (0.28's pin) |
| FlashInfer | 0.6.14 | **0.6.18** (dispatch table now includes 192/256) |
| ExLlamaV3 | v1.4.1 (ARM64 patches) | **v1.4.6** (ARM64 patch 0004 re-anchored) |
| InstantTensor | 0.1.5 (+patches) | **0.1.9** (fixes upstreamed) |
| SparkInfer | b12x@669a12dd | b12x@669a12dd (same pin) |
| triton | 3.6.0 | 3.7.1 |
| xgrammar | 0.2.3 | 0.2.3 |
| Base | vllm/vllm-openai:v0.26.0 (Ubuntu 22.04) | nvidia/cuda:13.0.3-base-ubuntu24.04 |

## 2. Patch triage — what is dead, what is kept

All 26 `patch_*.py` scripts from the 0.26 image were analyzed against the
pinned commit's source (verified via `git log --grep` for upstream PRs and
anchor-diffing for custom patches).

### 2.1 DEAD — merged upstream (12 patches) 🔴

These upstream backports are now first-class in latest main and were dropped:

| 0.26 patch | Upstream PR | Verified merged |
|---|---|---|
| `patch_attn_short_ctx_topk.py` | #49486 | `b0cb1da1b` in history |
| `patch_attn_combined_indices.py` | #50298 | `837eae645` |
| `patch_attn_index_remap.py` | #50365 | `9e6be4a72` |
| `patch_attn_c128a_adaptive_width.py` | #52823 | `e6f35d3c6` |
| `patch_cache_topk_kernel_constexpr.py` | #51967 | `83f591d7f` |
| `patch_cache_combine_workers.py` | #52084 | `836aac92f` |
| `patch_attn_indexer_seqlen_clamp.py` | #51538 | `97388c44f` |
| `patch_dflash_speculator_51538.py` | #51538 | `97388c44f` |
| `patch_dspark_attn_backend_52288.py` | #52288 | `acb0f1dcd` |
| `patch_attn_prefill_workspace_51733.py` | #51733 | `608c12473` |
| `patch_spec_budget_adaptive_51725.py` | #51725 | `0914ed2e8` |
| `patch_xgrammar_termination.py` | #52805 + #53046 | `12f64b39d` + `c6e19b3be` |

### 2.2 DEAD — superseded by newer deps / upstream refactors (6 patches) 🔴

| 0.26 patch | Why dead |
|---|---|
| `patch_sparse_swa_sm12x.py` | Upstream `get_dspark_swa_index_width()` (align-64) yields width 192 for window 128 + K5, and **flashinfer 0.6.18's `_DECODE_DSV4_DISPATCH` now instantiates `(64, 192)`** — the SM12x pad-to-{128,512,1024} is unnecessary. Verified at runtime: boot completes with no decode fallback and the autotuner tunes `sparse_mla_sm120_decode_dsv4` cleanly. |
| `instanttensor.patch` (0.1.5 scalar/clone) | Fixed upstream in **instanttensor 0.1.9** (`view(torch.Size(shape))` + `.clone()`). |
| `patch_instanttensor_loader.py` (copy=True hunk) | vllm 0.28 pins `instanttensor>=0.1.9`; 0.1.9's `safe_open` accepts `copy=True` natively. |
| `patch_model_exl3.py` hunk 2 (packed_modules_mapping) | Upstream model.py now ships `packed_modules_mapping` for DeepSeek V4. |
| `patch_model_exl3.py` hunk 4 (skip mtp.*) | Upstream mapper drops `mtp.*` natively (`orig_to_new_substr={"mtp.": None}`). |
| `patch_version.py` | Replaced by `SETUPTOOLS_SCM_PRETEND_VERSION` (banner shows the stamped version, no post-install script needed). |

### 2.3 KEPT / REWORKED — still needed (11 patches) 🟢

`vllm-patches/` (12 unified diffs, applied to the **source tree before the wheel
build** — `git apply` fails loudly on upstream drift, which is what makes
bumping maintainable):

| # | Patch | Notes |
|---|---|---|
| 001 | `quant-init-exl3.patch` | exl3 registration in the quantization registry |
| 002 | `model-config-exl3.patch` | `"exl3"` in the DSV4 quant override list |
| 003 | `envs-exl3.patch` | `VLLM_EXL3_*` env registration (0 unknown-env warnings at boot ✓) |
| 004 | `dsv4-model-exl3.patch` | EXL3 `_rankN.` strip in fp4 mapper + expert name normalization (re-anchored; 2 of the old 4 hunks dropped as dead) |
| 005 | `router-k216.patch` | **Reworked for the 0.28 router**: the CUDA op `topk_hash_softplus_sqrt` still lacks 216 experts; upstream deleted the old `_topk_softplus_sqrt_torch`, so a fresh pure-Torch fallback was ported in |
| 006 | `dsv4-topk-k216.patch` | `can_use_dsv4_topk()` whitelist `(256,384)` → `(216,256,384)` — anchor identical to 0.26 |
| 007 | `dspark-draft-experts.patch` | Compact draft `n_routed_experts` override (re-anchored for the `topk_indices_buffer` param) |
| 008 | `detokenizer-stops-reasoning.patch` | Stop-in-reasoning guard — anchors verified, verified live (stop=["the"] doesn't truncate) |
| 009 | `strip-skip-artifact.patch` | `)Skip` (id 83480) strip — anchors verified |
| 010 | `loader-empty-cache.patch` | InstantTensor clone-buffer `empty_cache()` — still relevant with 0.1.9 |
| 011 | `dspark-quant-inherit.patch` | **New — found during bring-up** (see §3). Baked into the wheel with the other 10 |

### 2.4 Unchanged (sparkinfer package patches) 🟢

`sparkinfer-patches/patch_w4a16_expert_counts.py` (b12x #150) and
`sparkinfer-patches/patch_tiny_decode_route_clamp.py` (b12x #228) target the
pinned SparkInfer fork, not vllm — kept as fail-closed anchor scripts, applied
post-install (sparkinfer is pip-installed, not source-built).

## 3. Bring-up findings (new in 0.28, not present in 0.26)

### 3.1 Draft quant-config regression — patch 011 (the important one)

v0.28's `load_dspark_model()` builds a **fresh** draft quantization config via
`get_draft_quant_config()` instead of inheriting the target's hydrated
`Exl3Config` (v0.26 behavior). The compact draft dir does not ship
`quantization_config.json`, so the draft's `Exl3Config` has an empty
`tensor_storage` → draft attention linears registered **unquantized** → crash
at weight load:

```
KeyError: 'model.layers.0.attn.fused_wqa_wkv.weight_scale_inv'
```

Root-caused live in the container (three compounding issues):
1. `tensor_storage` must be inherited from the target config;
2. `rank_sliced_metadata` comes from the draft's own `hybrid_tr3_tail` ✓ (present);
3. `packed_modules_mapping` must also be inherited — upstream's
   `configure_quant_config()` leaves it unset for `DSparkDeepseekV4ForCausalLM`
   (it defines none), and `_linear_prefix_is_fp8()` needs it to resolve
   `fused_wqa_wkv` → `wq_a/wkv`.

Fix (patch 011): after the quant-config override, copy the target's
`tensor_storage`, `rank_sliced_metadata`, `packed_modules_mapping`, and FP8
attrs onto the draft config when both are the same class and the draft's
storage is empty. Applied live to the container to unblock, then added to
`vllm-patches/011-dspark-quant-inherit.patch` and baked into the wheel on the
next rebuild.

### 3.2 SM12x sparse-MLA dispatch — no patch needed (confirmed dead)

The 0.26 image padded the DSpark non-causal SWA index width 256 → 512 because
flashinfer 0.6.14 only dispatched `{128, 512, 1024}`. On 0.28 the width comes
from upstream's `get_dspark_swa_index_width()` (192 for window 128 + K5) and
flashinfer **0.6.18** dispatches `(64, 192)` — verified in
`flashinfer/mla/_sparse_mla_sm120.py` and at runtime (clean boot, autotune
tuned `sparse_mla_sm120_decode_dsv4`, no prefill-orchestrator fallback).

### 3.3 Draft/quant module tree numbering (no patch needed)

The draft model's `nn.ModuleList` names its layers `model.layers.0..2`
(pytorch auto-indexing) while construction-time prefixes are
`model.layers.43..45` (used for storage/FP8 detection, which resolves via the
overlay's `mtp.{N}` mapping). The 0.28 `load_weights` remaps `mtp.{i}` →
`model.layers.{i}`, which matches the module tree — consistent after 3.1.

## 4. Build recipe (from scratch, no vendored source)

```
images/vllm-0_28_0-exl3/
├── Dockerfile            # 2 stages
│   ├── vllm-builder      # nvidia/cuda:13.0.3-devel-ubuntu24.04
│   │                     #   git init + shallow fetch of VLLM_COMMIT (no vendored source)
│   │                     #   git apply vllm-patches/*.patch  (fail-closed)
│   │                     #   + overlay exl3.py into the tree
│   │                     #   torch==2.13.0, setuptools-rust (vllm-rs, rust 1.95 via rustup)
│   │                     #   TORCH_CUDA_ARCH_LIST=12.1 (single-arch wheel: SM121 only)
│   │                     #   SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0+exl3.dspark.sm121
│   │                     #   python3 setup.py bdist_wheel
│   └── runtime           # nvidia/cuda:13.0.3-base-ubuntu24.04
│                         #   python3.12 + cuda-nvcc-13-0 + dev libs (runtime JIT)
│                         #   requirements/cuda.txt (torch, flashinfer 0.6.18, ...)
│                         #   wheel install → sparkinfer (b12x@669a12dd) + 2 patches
│                         #   → exllamav3 v1.4.6 (MAX_JOBS=8) → instanttensor 0.1.9
│                         #   → verify_exl3.py
├── vllm-patches/         # 011 unified diffs (all baked into the wheel)
├── overlay/…/exl3.py     # EXL3 backend (API-compat verified vs 0.28)
├── exllamav3-patches/    # ARM64 SM121 patch set (unchanged)
├── sparkinfer-patches/   # patch_w4a16_expert_counts.py + patch_tiny_decode_route_clamp.py (unchanged)
└── build.sh              # -> vllm-exl3-v28:latest
```

Build gotchas hit and fixed during the run:
- `python3-pip` missing in the builder (Ubuntu 24.04 `EXTERNALLY-MANAGED` removed)
- `-r common.txt` relative include in `requirements/cuda.txt` (copied files
  into a flat `/tmp` with original names)
- exllamav3 OOM (`ninja exit 137`) while the old model still held ~115 GB
  unified memory → stop model; the exllamav3 layer builds with `MAX_JOBS=8`
  to stay within memory
- `libcusparse-dev-13-0` / `libcusolver-dev-13-0` needed for ATen headers

## 5. Verification (live)

- ✅ `Application startup complete.` — version banner `0.28.0+exl3.dspark.sm121`
- ✅ Draft: compact K160 built once, `DSpark draft model loaded: 97 params`,
  dspark CUDA graphs captured
- ✅ EXL3 runtime planned: `Trellis m=1..32 block_m=8, prefill trellis block_m=64
  arena=392.1MiB capacity=2048 chunk=128 topk=6`
- ✅ FlashInfer 0.6.18 autotune: fresh cache created
  (`…/flashinfer_autotune_cache/0.6.18/121a/…`, 30 configs), no decode fallback
- ✅ Correctness: `17*19 → 323`; code-gen with thinking (117 reasoning tokens);
  stop-in-reasoning guard (stop=["the"] does not truncate mid-think)
- ✅ 0 unknown-vllm-env warnings (VLLM_EXL3_* registered)
- ✅ Weight load via InstantTensor: 99.4 GB target in ~48 s (warm cache)

### 5.1 Memory/KV delta vs 0.26 (watch item)

| | 0.26 image | 0.28 image |
|---|---|---|
| Consumed memory | ~102.7 GiB | 103.43 GiB |
| KV cache | 7.12 GiB / 1,047,542 tokens | 5.05 GiB / 758,800 tokens |
| Max concurrency @ 524 K | 2.00x | 1.45x |

The newer stack (torch 2.13 + vllm 0.28) keeps ~3 GiB more non-torch memory.
YAML params were **not changed** per instruction; if the old KV capacity is
required, options (for later, explicitly approved): raise
`--gpu-memory-utilization` to ~0.92, or lower `--max-model-len`.

## 6. Maintainability

- **Bump = one line:** change `VLLM_COMMIT` in the Dockerfile → rebuild.
  `git apply` fails loudly on any drifted anchor; drop a patch file when
  upstream adopts it (README table lists the triage ledger).
- No vendored source: vllm, exllamav3, sparkinfer are cloned/pulled at build
  time at pinned commits/tags.
- Patches are plain unified diffs against the source tree (all 011 baked
  into the wheel).

## 7. Files changed

- `images/vllm-0_28_0-exl3/` (new) — Dockerfile, build.sh, README.md,
  vllm-patches/ (011), overlay/, exllamav3-patches/, sparkinfer-patches/,
  verify_exl3.py
- `models/deepseek-v4-flash-0731-exl3-dspark.yaml` — `image:` bumped to the
  v0.28 image, `serve_ds4_exl3.sh` mount added (no runtime params changed)
- `models/logs/deepseek-v4-flash-0731-exl3-dspark.log` — refreshed

Nothing committed (per repo rules).
## 8. Memory forensics — why KV is smaller (analysis only, nothing changed)

### 8.1 The numbers

| Metric | 0.26 image | 0.28 image | Δ |
|---|---|---|---|
| Init snapshot free (pre-load) | 115.05 GiB | 114.48 GiB | **+0.57 GiB resident at init** |
| Weights | 97.46 GiB | 97.47 GiB | ~0 (identical checkpoint) |
| vLLM non-KV budget (consumed+activation+graphs) | ~102.65 GiB | ~104.53 GiB | **+1.88 GiB** |
| KV pool | 7.12 GiB | 5.05 GiB | −2.07 GiB |
| KV tokens / concurrency @ 524 K | 1,047,542 / 2.00x | 758,800 / 1.45x | −28% |
| Per-token KV cost (fp8, block 256) | ~7.3 KB | ~7.1 KB | **unchanged** |

vLLM sizes the KV pool as `requested(0.9 × 121.63 = 109.46) − non_kv_memory`. The
KV format did not change — the pool is smaller purely because the 0.28 stack's
non-KV footprint is ~1.9 GiB bigger.

### 8.2 Where the extra ~1.9 GiB went

**a) +0.57 GiB at worker init (before any weights load).** The 0.28 stack is
resident-heavier from the start: torch 2.13.0+cu130 vs 2.11.0+cu130 (CUDA 13.0.3
runtime, cuBLAS/cuDNN handles), flashinfer 0.6.18 (imports + sm120 kernel
modules), and vllm 0.28's workspace manager (created before profiling,
`gpu_worker.py:454`).

**b) +1.3 GiB during load/warmup.** Weights are identical; the delta is in the
non-weight resident set:
- flashinfer 0.6.18 autotune/JIT: 30 tuned `sparse_mla_sm120_decode_dsv4`
  configs compiled + resident (dispatch set grew: 192/256/512/1024 buckets vs
  {128,512,1024} in 0.6.14);
- vllm 0.28 persistent workspaces (DSA indexer `use_flattening=True` buffers,
  sparse-MLA combined-indices workspace, Model Runner V2 buffers);
- instanttensor 0.1.9's ring buffer: now defaults to `io_depth=512` (0.1.5 had
  a smaller default) — sized as `free × max_free_mem_usage(0.5)`; it requested
  ~4 GiB, got shrunk to 108–126 (`Shrink io_depth … due to memory limit`
  warnings, new in this boot) and is freed on `__exit__` (verified
  `_C.close()` in 0.1.9) — so no leak, but the peak counts against the
  free-memory-delta accounting;
- torch 2.13 caching-allocator retained segments.

**c) Accounting method changed between versions.** 0.26 reported a
torch-stats-based split (`weights + peak-activation-ESTIMATE 2.1 + non-torch
2.79 + graphs 0.3`); 0.28 computes `total_consumed` as a physical
`mem_get_info()` free-memory delta (`vllm/utils/mem_utils.py:317`), which
captures driver-side reservations the old accounting under-counted. 0.28's
peak-activation estimate also dropped (2.1 → 0.99 GiB, measured transient
headroom vs estimated worst-case) — an accounting improvement, not a real
release. So part of the +1.9 GiB is measurement-method, part is real.

### 8.3 What it is NOT

- **opencode / other processes:** `nvidia-smi --query-compute-apps` shows a
  single GPU consumer (VLLM::EngineCore, 107,099 MiB); the docker build and
  agent processes are CPU-only. Host RAM is 116 GB free.
- **flashinfer caches:** autotune JSONs are 20 KB; JIT cubins 9.1 MB on disk —
  negligible.
- **instanttensor leaks:** 0.1.9 frees the loader ring buffer on context exit,
  and our 0.26-era `empty_cache` cleanup is baked into the 0.28 wheel
  (patch 010) — the clone buffers are returned to the driver after load.
- **KV format:** per-token cost identical (~7.2 KB/token).

### 8.4 If the old KV capacity is ever needed (nothing changed)

- `--gpu-memory-utilization 0.90 → 0.93` would add ~3.6 GiB (≈ 1.05 M tokens,
  ~2.0x) — the log's own suggestion `--kv-cache-memory=10529114112` (9.81 GiB)
  equals exactly that ceiling;
- or keep 0.90 and reduce `--max-model-len` if 512 K contexts are never used.
