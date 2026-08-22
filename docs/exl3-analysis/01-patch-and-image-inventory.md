# 01 — Patch & Image Inventory

Everything in `images/vllm-0_26_0-exl3/` as of `develop` @ `b563686`, verified against the v0.26.0 source in `~/projects/vllm` (tag `v0.26.0`).

## 1. Base & layer stack (Dockerfile)

| Layer | Source | Notes |
|---|---|---|
| 1. Base | `vllm/vllm-openai:v0.26.0` | Native DSPark, DeepSeek-V4, EPLB, Eagle3 |
| 2. SparkInfer | `brandonmmusic-max/b12x@669a12dd` (branch `exl3-trellis-fused`, sparkinfer 1.0.1) | Trellis MoE runtime (`api.plan/bind/run`, `prepare_weights`) |
| 3. ExLlamaV3 | tag `v1.4.1` + `exllamav3-patches/` (ARM64) | Compiled for SM121 (`TORCH_CUDA_ARCH_LIST=12.1`) |
| 4. InstantTensor | sdist `0.1.5` + `instanttensor.patch` | `--load-format instanttensor` (fast mmap load) |
| 5. EXL3 overlay | `overlay/vllm/model_executor/layers/quantization/exl3.py` (2234 lines) | The EXL3 quant backend |

SparkInfer is additionally patched in-image by two backports (see §5).

## 2. Patch set (in-place `str.replace` on dist-packages)

| # | Patch | Target (dist-packages) | Purpose | Hot-path impact |
|---|---|---|---|---|
| 1 | `patch_exl3_init.py` | `quantization/__init__.py` | register `Exl3Config`/method | bootstrap |
| 2 | `patch_model_config.py` | `config/model.py` | add `"exl3"` to DSv4 quant-override list | bootstrap |
| 3 | `patch_envs_exl3.py` | `envs.py` | register `VLLM_EXL3_*` envs | bootstrap |
| 4 | `patch_model_exl3.py` | `deepseek_v4/nvidia/model.py` | `packed_modules_mapping`, `_rankN` mapper strip, rank-name normalize, `skip_weight_name_before_load` (keeps DSPark) | load-time |
| 5 | `patch_router_k216.py` | `fused_moe/router/fused_topk_bias_router.py` | pure-Torch `_topk_softplus_sqrt_torch` fallback for 216 (CUDA op supports ∉216) | safety net (see `02`) |
| 6 | `patch_dsv4_topk_k216.py` | `fused_moe/router/dsv4_topk.py` | whitelist `(256,384)` → `(216,256,384)` so fused `dsv4_topk` runs for 216 | **engaged — the good path** |
| 7 | `patch_dspark_exl3.py` | `deepseek_v4/nvidia/dspark.py` | build draft `DecoderLayer`s with draft's `n_routed_experts` | draft load |
| 8 | `patch_sparse_swa_sm12x.py` | `v1/attention/backends/mla/sparse_swa.py` | pad draft noncausal SWA width 256→512 for FlashInfer `sparse_mla_sm120_decode_dsv4` dispatch | **draft decode correctness** |
| 9 | `patch_version.py` | `_version.py` | stamp `0.26.0-exl3.dspark.sm121` | cosmetics |
| 10 | `instanttensor.patch` | instanttensor pkg | scalar-shape + buffer-clone for EXL3 mcg/mul1 | load-time |
| 11 | `patch_attn_short_ctx_topk.py` | `deepseek_v4/attention.py` | **upstream `#49486`** backport: skip topk/router when candidates ≤ topk (Triton fill) | TTFT ~3.4% |
| 12 | `patch_attn_combined_indices.py` | `deepseek_v4/common/ops/cache_utils.py`, `nvidia/flashmla.py` | **upstream `#50298`** backport: preallocate `combined_indices`/`combined_lens` workspace; drop per-launch `torch.full` | kernel 1.88× |
| 13 | `patch_attn_index_remap.py` | `v1/attention/backends/mla/sparse_utils.py` | **upstream `#50365`** backport: drop atomic-add in index remap when single tile owns a row | less atomic contention |
| 14 | `patch_detokenizer_stops_reasoning.py` | detokenizer/stop logic | don't match client stop strings inside reasoning | output correctness |
| 15 | `patch_w4a16_expert_counts.py` | `sparkinfer/moe/fused_moe/_impl.py` | **b12x#150** backport: preallocate `expert_counts` route histogram for CUDA-graph capture (fast parallel count on hot path) | MoE graph replay |
| 16 | `patch_tiny_decode_route_clamp.py` | `sparkinfer/moe/_shared/kernels/tiny_decode.py` | **b12x#228** backport: clamp inactive/graph-padding expert ids in range (avoid OOB reads) | MoE graph replay |

## 3. Overlay — `exl3.py` (the EXL3 quant backend)

- `Exl3Config` — config; `_configure_rank_sliced`, `get_quant_method`, `normalize_rank_sliced_weight_name`, `_storage_entry`, FP8/EXL3 prefix routing.
- `Exl3Parameter` / `Exl3MoEParameter` — vLLM-weight parameter shims with `load_exl3_weight`.
- `Exl3LinearMethod` — ATTN/QKV/MLP dense EXL3 GEMM (`apply` → `_exl3_gemm`), sign unpack, sharding for TP.
- `Exl3MoEMethod(apply)` — the MoE entry:
  - **rank-sliced**: `_apply_rank_sliced` → SparkInfer Trellis `bind`+`run` (decode `m∈[1,32]`, prefill `m>32` up to capacity). **This is the hot path.**
  - **parity** (`exl3_moe` per-expert argsort path): reachable only for `m < min_trellis_m`; with `VLLM_EXL3_TRELLIS_MIN_M=1` it is **dead at runtime** (every `m≥1` is trellis/prefill).
  - monolithic / `_apply_expert` (per-expert `_exl3_gemm`) — kept for non-trellis configs.
- Environment knobs read per-runtime: `VLLM_EXL3_TRELLIS_MIN_M`, `_MAX_M`, `_BLOCK_M`, `_PREFILL_CHUNK`, `_PREFILL_TRELLIS`, `_PREFILL_BLOCK_M`, `VLLM_EXL3_EXT_PATH`, `VLLM_EXL3_ABI_SHIM`.

## 4. Runtime (per model scope) cost drivers

- One Trellis runtime per owning quant-config scope (`_runtime_scope_id`) × {target, draft} isolation (`_runtime_owner_token`).
- Each runtime reserves: `trellis_scratch` (decode), **`prefill_scratch` ~1 GiB** per code reading (`VLLM_EXL3_PREFILL_TRELLIS=1`; the runtime boot banner reports **392.1 MiB** per runtime at capacity 2048 — see `03` L3 cheat-sheet), plus parity staging.
- Since target **and** draft each own a runtime, there are **two ~1 GiB prefill arenas** ≈ 2 GiB of scratch (memory, not speed — see `03`).

## 5. SparkInfer in-image backports (`fused_moe/_impl.py`, `tiny_decode.py`)

- `patch_w4a16_expert_counts.py` (#150): threads `expert_counts` through `TPW4A16Workspace`/`TPMoEFP4Binding`/arena so `pack_topk_routes_by_expert` always takes the fast parallel-count path, including during CUDA-graph capture (no capture-time alloc).
- `patch_tiny_decode_route_clamp.py` (#228): `route_active` guard in tiny-decode FC1/FC2 so graphed padding/EP-inactive routes read weights at `eid=0` (in range) instead of OOB.

## 6. Draft machinery (config-driven, no rebuild)

- `serve_ds4_exl3.sh` builds a compact DSPark draft from the checkpoint's REAP plan → `DRAFT_DIR`, then `exec vllm serve`.
- `build_dspark_draft.py` selects `DSPARK_DRAFT_EXPERTS` experts (32 structured per category + REAP fill), slices gate rows, renumbers `0..N-1`, emits EXL3-trellis tensors (`mtp.{0,1,2}.ffn.experts.*`). The draft is **EXL3-quantized** → runs through `Exl3MoEMethod._apply_rank_sliced` (own runtime scope).
- Env knobs: `DSPARK_DRAFT_EXPERTS` (currently **192**), `DSPARK_STRUCTURED_EXPERTS_PER_CATEGORY` (32).

## 7. Verified against upstream (this analysis)

All patch anchors in §2 were validated against the real v0.26.0 files in `~/projects/vllm` (tag `v0.26.0`); patch scripts are fail-closed (abort on ambiguous/missing anchors) and `py_compile`-verified. No drift found.
