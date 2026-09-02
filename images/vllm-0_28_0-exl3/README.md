# vllm-0_28_0-exl3 — vLLM latest-main (0.28.x) + EXL3 backend image

Self-contained Docker image for serving the EXL3-quantized DeepSeek-V4-Flash
Spark checkpoint (`0xSero/deepseek-v4-flash-0731-spark`) on a DGX Spark
(SM121 / GB10, ARM64, CUDA 13), with native DSPark/DeepSeek-V4 support kept
intact.

**This image is built entirely from source** — the vLLM wheel is compiled in
a builder stage from a pinned upstream commit, then installed into a minimal
`nvidia/cuda:13.0.3-base-ubuntu24.04` runtime. There is no
`vllm/vllm-openai` base image (we do not inherit its ~20 GB of examples,
benchmarks, modelscope, streamers, etc.).

**Image:** `vllm-exl3-v28:latest` — `vllm.__version__ = 0.28.0+exl3.dspark.sm121`

## Credits

This image is built on the shoulders of an incredible open-source community.
We did not write any of this from scratch — every piece below was created,
debugged and shared by someone else first, and we are genuinely grateful for
their work. Nothing here is taken as-is: each component was **ported,
backported or adapted** to our v0.28.0+ / SM121 (GB10, ARM64) base — built
from source at a pinned upstream commit, re-verified in-container, and
A/B-tested on our model before shipping. Where a patch is a source-exact
upstream backport, the upstream code is credited verbatim and its provenance
is stated; where we adapted it, the changes are documented in the patch
headers and the patch table below. Each component keeps its own license:

| Component | Source | Used for (see Patches table for details) |
| --- | --- | --- |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) — compiled from source at pinned main commit `62588e05` (v0.28.0 + 289 commits) | Engine, native DSPark / DeepSeek-V4 / Eagle3 support; the 12 upstream community backports the 0.26 image carried (rows 11–23 of its table) are now native (patch table rows 1–11 are the remaining custom deltas) |
| SparkInfer | [efeslab/SparkInfer](https://github.com/efeslab/SparkInfer) (Microsoft Research Asia + PKU), via the [`brandonmmusic-max/b12x`](https://github.com/brandonmmusic-max/b12x) `exl3-trellis-fused` fork @ `669a12dd` (sparkinfer 1.0.1) | Trellis MoE + prefill kernels for the EXL3 backend, integrated into our custom exl3.py overlay |
| ExLlamaV3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) v1.4.6 (turboderp) | EXL3 trellis/MCG quant format + `exl3_moe` fused kernel. We patched it for ARM64/SM121 (`exllamav3-patches/`) and compiled it into this image |
| InstantTensor | [vllm-project/instanttensor](https://github.com/vllm-project/instanttensor) 0.1.9 | Fast mmap weight loading (`--load-format instanttensor`); 0.1.9 upstreams the scalar-shape + clone fixes we used to patch into 0.1.5 |
| b12x kernel backports | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) PRs #150, #228 | Two kernel fixes (patch table rows 12–13), re-anchored onto our pinned SparkInfer tree |
| Stop-in-reasoning guard | tonyd2wild — community "Patch 5" (detokenizer) | Guard concept adapted to our v0.28.0 detokenizer and DeepSeek-V4 tokenizer quirks (patch table row 8) |
| Checkpoint | [0xSero/deepseek-v4-flash-0731-spark](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) (EXL3 3.0 bpw, REAP K216) on [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | The model this image serves; REAP plan vendored at `models/files/REAP_K216_PLAN.json` |

Thank you to every author and maintainer above — and to the wider vLLM,
ExLlama, SparkInfer and DGX-Spark community whose issues, PRs and write-ups
made this deployment possible. If any of this work is yours and you would like
different attribution or have concerns, please open an issue.

## Layout

```
images/vllm-0_28_0-exl3/
├── build.sh                   # docker build wrapper (-> vllm-exl3-v28:latest)
├── Dockerfile                 # builder stage (wheel) + minimal runtime stage
├── vllm-patches/              # 11 .patch files applied to the vllm source tree
│                               # BEFORE the wheel build (see table below)
├── overlay/
│   └── vllm/model_executor/layers/quantization/exl3.py   # EXL3 backend module
├── exllamav3-patches/         # ARM64 SM121 patch set for ExLlamaV3 v1.4.6
├── sparkinfer-patches/          # sparkinfer kernel backports (applied post-install)
│   ├── patch_w4a16_expert_counts.py     # b12x #150: route histogram prealloc
│   └── patch_tiny_decode_route_clamp.py # b12x #228: tiny-decode route clamp
└── verify_exl3.py             # build-time smoke test

The container entrypoint (`serve_ds4_exl3.sh`) and the DSPark draft builder
(`build_dspark_draft.py`) are NOT baked in — they live in `models/files/` and
are volume-mounted read-only by the model YAML, so serving/draft logic can
change without an image rebuild.
```

## Build

```bash
cd ~/vllm-starters/images/vllm-0_28_0-exl3
./build.sh                    # -> vllm-exl3-v28:latest
```

Build time: ~45–75 min first run (vllm wheel compile for SM121 + ExLlamaV3
CUDA ext + InstantTensor sdist). Rebuilds are fast — only layers invalidated
by the changed files re-run.

## Pinned versions

| Component | Pin | Notes |
|---|---|---|
| vLLM source | `VLLM_COMMIT=62588e0592ad5af8d3d4a536ace01afb44ebaed5` (main @ 2026-09-02; v0.28.0 + 289 commits past v0.28.1rc0) | cloned shallow at build time |
| torch | `2.13.0` (cu130) | vllm 0.28's pin |
| FlashInfer | `flashinfer-python==0.6.18` + `flashinfer-cubin==0.6.18` | cubin from `flashinfer.ai/whl/` |
| SparkInfer | `brandonmmusic-max/b12x@669a12dd` (exl3-trellis-fused → sparkinfer 1.0.1) | pip git install |
| ExLlamaV3 | tag `v1.4.6` + `exllamav3-patches/` | cloned at build time |
| InstantTensor | `0.1.9` (PyPI sdist, built with image nvcc) | 0.1.9 upstreams our old 0.1.5 fixes |

## Patches

| # | Patch | File | Fixes |
|---|-------|------|-------|
| 1 | exl3 registration | `vllm-patches/001-quant-init-exl3.patch` | Registers `Exl3Config` in the quantization registry (`Exl3MoEMethod`/`Exl3LinearMethod` live in `exl3.py` and are handed out by `get_quant_method`) |
| 2 | Config override | `vllm-patches/002-model-config-exl3.patch` | Adds `"exl3"` to the DeepSeek V4 quant override list |
| 3 | EXL3 env vars | `vllm-patches/003-envs-exl3.patch` | Registers `VLLM_EXL3_*` (suppresses unknown-env warnings) |
| 4 | Model deltas | `vllm-patches/004-dsv4-model-exl3.patch` | EXL3 rank-sliced name handling: `_rankN.` strip in the fp4 weights mapper + expert rank-name normalization in `load_weights` (upstream now ships `packed_modules_mapping` and drops `mtp.*` natively, so those old hunks were dropped) |
| 5 | Router fallback | `vllm-patches/005-router-k216.patch` | Pure-Torch `topk_softplus_sqrt` fallback for REAP K216 (216 experts, unsupported by the CUDA op). Reworked for the 0.28 router, which no longer ships a torch fallback |
| 6 | Fused DSv4 router kernel | `vllm-patches/006-dsv4-topk-k216.patch` | Relaxes `can_use_dsv4_topk()`'s `(256, 384)` whitelist to `(216, 256, 384)` so the fused Triton `dsv4_topk` replaces the pure-Torch fallback on the routing hot path |
| 7 | Compact DSPark draft | `vllm-patches/007-dspark-draft-experts.patch` | Build draft `DecoderLayer`s with the draft's `n_routed_experts` (temporary target-config override) so compact draft weights load regardless of expert count |
| 8 | Stop-in-reasoning guard | `vllm-patches/008-detokenizer-stops-reasoning.patch` | Backport of tonyd2wild Patch 5: don't match client stop strings inside the reasoning segment on think-in-prompt models (fixes null/empty content when a harness e.g. lm-evaluation-harness sends stop strings). Reads markers from `--reasoning-config`; opt out with `VLLM_SUPPRESS_STOPS_IN_REASONING=0` |
| 9 | Strip `)Skip` artifact (narrow) | `vllm-patches/009-strip-skip-artifact.patch` | Removes only the exact DSPark corrupt-draft token `)Skip` (id 83480) from chat content (streaming + non-streaming, parser + plain paths). Deliberately narrow: bare `Skip`/`skip` (legit English), other punctuation forms, and whitespace are untouched — no global space collapsing, so tool/YAML content is not corrupted |
| 10 | Loader empty-cache | `vllm-patches/010-loader-empty-cache.patch` | `torch.cuda.empty_cache()` after InstantTensor load so clone buffers don't shrink the KV profiler's pool |
| 11 | Draft quant-config inherit | `vllm-patches/011-dspark-quant-inherit.patch` | v0.28 builds a fresh draft `Exl3Config` whose `quantization_config.json` is absent from the compact draft dir; inherits `tensor_storage` + `packed_modules_mapping` from the target config so draft FP8 attention loads correctly (fixes `KeyError: fused_wqa_wkv.weight_scale_inv` at draft load) |
| 12 | W4A16 route histogram prealloc | `sparkinfer-patches/patch_w4a16_expert_counts.py` | Backport of local-inference-lab/b12x #150: preallocate the W4A16 route histogram (`expert_counts`) so the fast count+prefix kernel runs during CUDA-graph capture without allocation |
| 13 | Tiny-decode route clamp | `sparkinfer-patches/patch_tiny_decode_route_clamp.py` | Backport of local-inference-lab/b12x #228: keep graphed tiny-decode routes with inactive expert ids in range (graph padding / non-local EP routes) to avoid OOB weight reads |

## Compact DSPark draft

`serve_ds4_exl3.sh` (the container entrypoint used by the model YAML, living
in `models/files/` and volume-mounted read-only over `/opt/recipe/`) serves
the EXL3 target with a **compact DSPark draft** (expert count set per-model
via `DSPARK_DRAFT_EXPERTS` in the YAML) instead of the full 216-expert
embedded MTP draft:

1. Resolves the target snapshot under the HF cache
   (`/root/.cache/huggingface/hub/models--0xSero--deepseek-v4-flash-0731-spark`).
2. If `DRAFT_DIR/model.safetensors.index.json` is missing, runs
   `build_dspark_draft.py` (also mounted from `models/files/`) against the
   REAP plan: `DSPARK_DRAFT_EXPERTS` experts per MTP stage selected as top
   structured specialists (32 per category) then the global REAP fill, gates
   sliced to `DSPARK_DRAFT_EXPERTS` rows, tensors renumbered `0..N-1`. The
   target checkpoint is never modified. The REAP plan is vendored at
   `models/files/REAP_K216_PLAN.json` (upstream dropped it from newer
   snapshots, which broke the draft build) and mount-readonly into the
   container at `/opt/recipe/REAP_K216_PLAN.json`; the builder falls back to
   that mounted copy when the snapshot lacks its own plan.
3. `exec vllm serve "$@"` with `--speculative-config` pointing `model` at
   `DRAFT_DIR`.

The draft is built once and reused (idempotent). Drafting fewer experts than
216 frees unified memory that can go toward `--max-model-len` / KV capacity.
`vllm-patches/007-dspark-draft-experts.patch` makes `DSparkDeepseekV4Model`
build its `DecoderLayer`s with the draft's `n_routed_experts` (temporarily
overriding the target config during construction, then restoring it) —
required so the compact draft weights load into matching shapes.

## How to bump vLLM (maintainability)

```bash
# 1. Pick the new commit
git ls-remote https://github.com/vllm-project/vllm.git HEAD

# 2. Update VLLM_COMMIT in the Dockerfile

# 3. Check the patches still apply (fail-closed):
cd /tmp && git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-new
cd /tmp/vllm-new && git fetch --depth 1 origin <NEW_COMMIT> && git checkout -q FETCH_HEAD
git apply --check ~/vllm-starters/images/vllm-0_28_0-exl3/vllm-patches/*.patch
#   -> if a patch fails: upstream likely changed the anchor (fix the hunk),
#      or adopted the fix (drop the patch from the table + README).

# 4. Rebuild and smoke-test (see the boot markers in the repo AGENTS.md).
```

If a newer upstream commit contains one of our custom patches (detokenizer
guard, strip-skip, K216 router support, etc.), delete the corresponding
`vllm-patches/*.patch` file and note it in this README's table.

## exl3.py runtime deltas vs stock v0.28.0

The EXL3 overlay (`overlay/vllm/model_executor/layers/quantization/exl3.py`)
keeps the native DSPark/DeepSeek-V4 paths intact and only adds the EXL3
rank-sliced checkpoint handling:

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
- **Rank-sliced name handling:** `_rankN.` segments are stripped by the weights
  mapper and the expert loader (`vllm-patches/004`) so the per-expert
  trellis/suh/svh/mcg/mul1 tensors load into their `Exl3MoEParameter`s; the
  TP4 checkpoint is coalesced to runtime TP1 in memory at load.

## Usage

Served via `vllm-manager.sh` in `~/vllm-starters`:

```bash
cd ~/vllm-starters
DRY_RUN=false ./vllm-manager.sh start --model deepseek-v4-flash-0731-exl3-dspark
```

**Config:** `~/vllm-starters/models/deepseek-v4-flash-0731-exl3-dspark.yaml`
(unchanged from the 0.26 image: FLASHINFER_MLA, gpu-memory-utilization 0.9,
max-model-len 524288, FP8 KV, DSPark compact K160 draft, `--load-format
instanttensor`, `--quantization exl3`).

**API:** `http://localhost:8000/v1/chat/completions` (key: `sk-dgx-spark-qwen-777`)