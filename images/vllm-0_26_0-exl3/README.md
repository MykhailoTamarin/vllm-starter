# vllm-0_26_0-exl3 — vLLM 0.26.0 + EXL3 backend image

Self-contained Docker image for serving the EXL3-quantized DeepSeek-V4-Flash
Spark checkpoint (`0xSero/deepseek-v4-flash-0731-spark`) on a DGX Spark
(SM121 / GB10, ARM64, CUDA 13), with native DSPark/DeepSeek-V4 support kept
intact.

This folder lives under `~/vllm-starters/images/<image-name>/` — each custom
image gets its own folder there.

**Image:** `vllm-exl3-v26:latest` — `vllm.__version__ = 0.26.0-exl3.dspark.sm121`

## Credits

This image is built on the shoulders of an incredible open-source community.
We did not write any of this from scratch — every piece below was created,
debugged and shared by someone else first, and we are genuinely grateful for
their work. Nothing here is taken as-is: each component was **ported,
backported or adapted** to our v0.26.0 / SM121 (GB10, ARM64) base — re-anchored
against our exact sources, re-verified in-container, and A/B-tested on our
model before shipping. Where a patch is a source-exact upstream backport
(e.g. the XGrammar termination fixes), the upstream code is credited verbatim
and its provenance is stated; where we adapted it, the changes are documented
in the patch headers and the patch table below. Each component keeps its own
license:

| Component | Source | Used for (see Patches table for details) |
| --- | --- | --- |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) — base image `vllm/vllm-openai:v0.26.0` | Engine, native DSPark / DeepSeek-V4 / Eagle3 support + 10 upstream community PR backports (patch table rows 11–23), re-anchored onto v0.26.0 |
| SparkInfer | [efeslab/SparkInfer](https://github.com/efeslab/SparkInfer) (Microsoft Research Asia + PKU), via the [`brandonmmusic-max/b12x`](https://github.com/brandonmmusic-max/b12x) `exl3-trellis-fused` fork @ `669a12dd` (sparkinfer 1.0.1) | Trellis MoE + prefill kernels for the EXL3 backend, integrated into our custom exl3.py overlay |
| ExLlamaV3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) v1.4.1 (turboderp) | EXL3 trellis/MCG quant format + `exl3_moe` fused kernel. We patched it for ARM64/SM121 (`exllamav3-patches/`) and compiled it into this image |
| InstantTensor | [vllm-project/instanttensor](https://github.com/vllm-project/instanttensor) 0.1.5 | Fast mmap weight loading (`--load-format instanttensor`); we adapted the vLLM loader integration for 0.1.5 |
| b12x kernel backports | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) PRs #150, #228 | Two kernel fixes (patch table rows 25–26), re-anchored onto our pinned SparkInfer tree |
| Stop-in-reasoning guard | tonyd2wild — community "Patch 5" (detokenizer) | Guard concept adapted to our v0.26.0 detokenizer and DeepSeek-V4 tokenizer quirks (patch table row 14) |
| XGrammar termination fixes | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) — port of vLLM #52805/#53046 | Their overlay was written for a newer GLM fork; we re-verified the source-exact upstream hunks against v0.26.0 and re-validated all anchors before baking it in (patch table row 24) |
| Checkpoint | [0xSero/deepseek-v4-flash-0731-spark](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) (EXL3 3.0 bpw, REAP K216) on [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | The model this image serves; REAP plan vendored at `models/files/REAP_K216_PLAN.json` |

Thank you to every author and maintainer above — and to the wider vLLM,
ExLlama, SparkInfer and DGX-Spark community whose issues, PRs and write-ups
made this deployment possible. If any of this work is yours and you would like
different attribution or have concerns, please open an issue.

## Build

```bash
cd ~/vllm-starters/images/vllm-0_26_0-exl3
./build.sh                    # -> vllm-exl3-v26:latest
# or
docker build -t vllm-exl3-v26:latest .
```

Build time: the ExLlamaV3 CUDA extension compiles from source (~15-20 min on
first build; cached afterwards).

## Structure

```
images/vllm-0_26_0-exl3/       # this custom image's build context
├── build.sh                   # docker build wrapper
├── Dockerfile
├── instanttensor.patch        # scalar-shape + buffer-clone fix for the fast loader
├── exllamav3-patches/         # ARM64 SM121 patch set (applied to cloned upstream)
├── overlay/                   # EXL3 vLLM backend overlay
│   └── vllm/model_executor/layers/quantization/exl3.py
├── patch_*.py                 # in-place vLLM patches (see below)
├── scripts/
│   ├── build_dspark_draft.py  # REAP draft builder (target-preserving, expert count from YAML)
│   └── serve_ds4_exl3.sh      # container entrypoint: build draft, then serve
└── verify_exl3.py             # build-time smoke test
```

## Layers (in order)

1. `vllm/vllm-openai:v0.26.0` (native DSPark, DeepSeek-V4, EPLB, Eagle3)
2. SparkInfer — `brandonmmusic-max/b12x@669a12dd` (exl3-trellis-fused → sparkinfer 1.0.1, pinned commit)
3. ExLlamaV3 v1.4.1 cloned at build time from upstream tag, then patched
   (x86 sources excluded on ARM64, AVX guards) and compiled for SM121
4. InstantTensor 0.1.5 (built from sdist on ARM64) — fast mmap weight loading
   via v0.26.0's native `--load-format instanttensor`
5. EXL3 vLLM overlay (exl3.py quantization + model deltas + router fallback)

## Patches

| # | Patch | File | Fixes |
|---|-------|------|-------|
| 1 | exl3 registration | `patch_exl3_init.py` | Registers `Exl3Config` in the quantization registry (`Exl3MoEMethod`/`Exl3LinearMethod` live in `exl3.py` and are handed out by `get_quant_method`) |
| 2 | Config override | `patch_model_config.py` | Adds `"exl3"` to the DeepSeek V4 quant override list |
| 3 | EXL3 env vars | `patch_envs_exl3.py` | Registers `VLLM_EXL3_*` (suppresses unknown-env warnings) |
| 4 | Model deltas | `patch_model_exl3.py` | `packed_modules_mapping`, `_rankN` mapper strip, expert rank-name normalize, `skip_weight_name_before_load` (keeps native DSPark) |
| 5 | Router fallback | `patch_router_k216.py` | Pure-Torch `_topk_softplus_sqrt_torch` for REAP K216 (216 experts, unsupported by the CUDA op) |
| 6 | Fused DSv4 router kernel | `patch_dsv4_topk_k216.py` | Relaxes `can_use_dsv4_topk()`'s `(256, 384)` whitelist to `(216, 256, 384)` so the fused Triton `dsv4_topk` replaces the pure-Torch fallback on the routing hot path |
| 7 | Compact DSPark draft | `patch_dspark_exl3.py` | Build draft `DecoderLayer`s with the draft's `n_routed_experts` (from the draft's own `config.json`) instead of the target's 216, so compact draft weights load regardless of expert count |
| 8 | SM12x SWA decode fix | `patch_sparse_swa_sm12x.py` | Pads non-causal SWA index width (256) up to a FlashInfer `sparse_mla_sm120_decode_dsv4`-dispatchable topk (512) so the 5-token draft decode doesn't fall through to the prefill orchestrator |
| 9 | Version stamp | `patch_version.py` | `vllm.__version__` → `0.26.0-exl3.dspark.sm121` |
| 10 | InstantTensor scalar fix | `instanttensor.patch` | `view(*shape)` → `view(tuple(shape)).clone()` so scalar (0-dim) EXL3 `mcg`/`mul1` sentinels load and views survive the loader's buffer reuse |
| 11 | Short-context topk skip | `patch_attn_short_ctx_topk.py` | Backport of upstream #49486: when `max_seq_len // compress_ratio` fits within `topk_tokens`, skip the topk/router/indexer op and fill the index buffer directly (~3.4% TTFT) |
| 12 | Combined-indices workspace | `patch_attn_combined_indices.py` | Backport of upstream #50298: stop re-allocating the fused `combined_indices`/`combined_lens` buffer per launch; pass a pre-allocated workspace from the DSv4 attention layer (~1.88× kernel perf) |
| 13 | Index-remap atomic drop | `patch_attn_index_remap.py` | Backport of upstream #50365: when a single Triton tile owns a full row, replace the atomic-add slot allocator/counter with a plain store (removes atomic contention on the sparse-MLA index remap) |
| 14 | Stop-in-reasoning guard | `patch_detokenizer_stops_reasoning.py` | Backport of tonyd2wild Patch 5: don't match client stop strings inside the reasoning segment on think-in-prompt models (fixes null/empty content when a harness e.g. lm-evaluation-harness sends stop strings). Reads markers from `--reasoning-config`; opt out with `VLLM_SUPPRESS_STOPS_IN_REASONING=0` |
| 15 | Adaptive C128A topk width | `patch_attn_c128a_adaptive_width.py` | Backport of upstream `#52823`: short-context C128A metadata runs the topk/index kernel at a 128-aligned active width (from `max_seq_len // compress_ratio`) instead of the full planned capacity — fewer inactive columns on the sparse-MLA path |
| 16 | Top-k index kernel constants | `patch_cache_topk_kernel_constexpr.py` | Backport of upstream `#51967`: `_compute_global_topk_indices_and_lens_kernel` strides/topk/block_size become `tl.constexpr` so Triton unrolls/specializes the kernel |
| 17 | Combine-kernel workers | `patch_cache_combine_workers.py` | Backport of upstream `#52084`: `_combine_topk_swa_indices_kernel` workers 128 → 256 (prefill throughput) |
| 18 | Strip `)Skip` artifact (narrow) | `patch_strip_skip_artifact.py` | Removes only the exact DSPark corrupt-draft token `)Skip` (id 83480) from chat content (streaming + non-streaming, parser + plain paths). Deliberately narrow: bare `Skip`/`skip` (legit English), other punctuation forms, and whitespace are untouched — no global space collapsing, so tool/YAML content is not corrupted |
| 19 | Indexer seq-len clamp | `patch_attn_indexer_seqlen_clamp.py` | Backport of upstream #51538: clamp padded-request per-token seq-len at 0 (kernel + host). `seq_len == 0` on padding rows made token 0 negative (−1); downstream kernels read it as uint32 → ~4e9 corruption on the padded decode path |
| 20 | DFlash/DSPark speculator fixes | `patch_dflash_speculator_51538.py` | Backport of upstream #51538: `sample_idx_mapping` inert rows init to −1 (CUDA-graph capture scatter), `is_query`/`query_off` from `num_valid_ctx` (rejected-suffix boundary), and null-block guards so evicted SWA/rejected rows never write draft KV into physical block 0 (`PAD_SLOT_ID`) |
| 21 | DSPark draft backend inherit | `patch_dspark_attn_backend_52288.py` | Backport of upstream #52288: when `--speculative-config` omits `attention_backend`, the draft now inherits the target's backend (`FLASHINFER_MLA`) instead of re-running auto-selection (which could pick a different sparse-MLA class) |
| 22 | Sparse-MLA prefill workspace | `patch_attn_prefill_workspace_51733.py` | Backport of upstream #51733: prefill workspace floor is `block_size` instead of `max_num_seqs * block_size` (GPU memory headroom for the 512K KV at 0.9 utilization) |
| 23 | Adaptive spec budget | `patch_spec_budget_adaptive_51725.py` | Backport of upstream #51725: scheduler tracks an `input_budget` alongside `token_budget`, consuming `num_new_tokens + draft_slots` per request — stops DSPark draft slots from starving prefill (upstream ~60% Kimi-K3 DSPark TTFT win) |
| 24 | XGrammar termination fixes | `patch_xgrammar_termination.py` | Source-exact backports of upstream #52805 (stop token batches at grammar termination) and #53046 (validate pre-reasoning-end speculative drafts before FSM advance). Model-agnostic; protects DSPark spec decode under grammar constraints |
| 25 | W4A16 route histogram prealloc | `patch_w4a16_expert_counts.py` | Backport of local-inference-lab/b12x #150: preallocate the W4A16 route histogram (`expert_counts`) so the fast count+prefix kernel runs during CUDA-graph capture without allocation |
| 26 | Tiny-decode route clamp | `patch_tiny_decode_route_clamp.py` | Backport of local-inference-lab/b12x #228: keep graphed tiny-decode routes with inactive expert ids in range (graph padding / non-local EP routes) to avoid OOB weight reads |

## Compact DSPark draft

`serve_ds4_exl3.sh` (the container entrypoint used by the model YAML) serves the
EXL3 target with a **compact DSPark draft** (expert count set per-model via
`DSPARK_DRAFT_EXPERTS` in the YAML) instead of the full 216-expert embedded MTP
draft:

1. Resolves the target snapshot under the HF cache
   (`/root/.cache/huggingface/hub/models--0xSero--deepseek-v4-flash-0731-spark`).
2. If `DRAFT_DIR/model.safetensors.index.json` is missing, runs
   `build_dspark_draft.py` against the REAP plan: `DSPARK_DRAFT_EXPERTS` experts
   per MTP stage selected as top structured specialists (32 per category) then
   the global REAP fill, gates sliced to `DSPARK_DRAFT_EXPERTS` rows, tensors
   renumbered `0..N-1`. The target checkpoint is never modified. The REAP plan
   is vendored at `models/files/REAP_K216_PLAN.json` (upstream dropped it from
   newer snapshots, which broke the draft build) and mount-readonly into the
   container at `/opt/recipe/REAP_K216_PLAN.json`; the builder falls back to
   that mounted copy when the snapshot lacks its own plan.
3. `exec vllm serve "$@"` with `--speculative-config` pointing `model` at
   `DRAFT_DIR`.

The draft is built once and reused (idempotent). Drafting fewer experts than
216 frees unified memory that can go toward `--max-model-len` / KV capacity.
`patch_dspark_exl3.py` makes `DSparkDeepseekV4Model` build its `DecoderLayer`s
with the draft's `n_routed_experts` (temporarily overriding the target config
during construction, then restoring it) — required so the compact draft weights
load into matching shapes.

## exl3.py runtime deltas vs stock v0.26.0

- **FP8 routing:** attention QKV/O + shared experts + indexer linears route to
  `Fp8LinearMethod` (not `UnquantizedLinearMethod`) so float8 bytes load with
  their E8M0 block scales.
- **Non-EXL3 MoE → `Mxfp4MoEMethod`:** layers outside the rank-sliced EXL3 range
  (e.g. the embedded DSPark MTP draft) are stored **native FP4-packed**, not EXL3
  trellis; routing them to the unquantized fallback OOM'd (full BF16 expert
  buffers). Now loaded as fp4 via `Mxfp4MoEMethod` so the packed I8 weights pick
  up their ue8m0 scales.
- **Draft-layer storage mapping:** `model.layers.{43+i}` prefixes map back to
  the checkpoint's `mtp.{i}` storage keys so FP8 detection works for the draft.

## Build gotchas (v0.26.0 base is Ubuntu 22.04 / glibc 2.35)

- `apt-get install git` (needed to pip-install sparkinfer from GitHub, and to clone exllamav3)
- `git clone --depth 1 --branch v1.4.1 ... && git apply /patches/*.patch` (applies the ARM64 SM121 patch set before `pip install`)
- `pip install --no-build-isolation` (setup.py must see torch to build the CUDA ext)
- `CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include`
  (the base lacks cusparse/cublas headers in `/usr/local/cuda/include`)
- `TORCH_CUDA_ARCH_LIST="12.1"`

## Pinned dependency versions

| Dependency | Source | Version / Commit |
|---|---|---|
| Base image | `vllm/vllm-openai:v0.26.0` | tag `v0.26.0` |
| SparkInfer (`sparkinfer` pkg) | `brandonmmusic-max/b12x` | commit `669a12ddc7cf3021e91a25f398b1a883b703fd12` (branch `exl3-trellis-fused` → sparkinfer 1.0.1) |
| ExLlamaV3 | cloned at build from upstream tag `v1.4.1` + `exllamav3-patches/` (ARM64 SM121) | tag `v1.4.1` |
| InstantTensor | PyPI sdist (no aarch64 wheel; built with the image's nvcc) | `instanttensor==0.1.5` |

## Usage

Served via `vllm-manager.sh` in `~/vllm-starters`:

```bash
cd ~/vllm-starters
DRY_RUN=false ./vllm-manager.sh start --model deepseek-v4-flash-0731-exl3-dspark
```

**Config:** `~/vllm-starters/models/deepseek-v4-flash-0731-exl3-dspark.yaml`

First-acceptance profile (per the model card): FLASHINFER_MLA attention,
gpu-memory-utilization 0.9, max-model-len 262144, FP8 KV cache, DSPark
compact draft (expert count from the YAML `DSPARK_DRAFT_EXPERTS`; native
`dspark` method, `num_speculative_tokens=5`).

**API:** `http://localhost:8000/v1/chat/completions` (key: `sk-dgx-spark-qwen-777`)
