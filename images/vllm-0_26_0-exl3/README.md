# vllm-0_26_0-exl3 — vLLM 0.26.0 + EXL3 backend image

Self-contained Docker image for serving the EXL3-quantized DeepSeek-V4-Flash
Spark checkpoint (`0xSero/deepseek-v4-flash-0731-spark`) on a DGX Spark
(SM121 / GB10, ARM64, CUDA 13), with native DSPark/DeepSeek-V4 support kept
intact.

This folder lives under `~/vllm-starters/images/<image-name>/` — each custom
image gets its own folder there.

**Image:** `vllm-exl3-v26:latest` — `vllm.__version__ = 0.26.0-exl3.dspark.sm121`

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
├── exllamav3-src/             # ExLlamaV3 v1.3.0 + ARM64 SM121 patches
├── overlay/                   # EXL3 vLLM backend overlay
│   └── vllm/model_executor/layers/quantization/exl3.py
├── patch_*.py                 # in-place vLLM patches (see below)
└── verify_exl3.py             # build-time smoke test
```

## Layers (in order)

1. `vllm/vllm-openai:v0.26.0` (native DSPark, DeepSeek-V4, EPLB, Eagle3)
2. SparkInfer — `brandonmmusic-max/b12x@669a12dd` (exl3-trellis-fused → sparkinfer 1.0.1, pinned commit)
3. ExLlamaV3 v1.3.0 compiled from source (ARM64 SM121, x86 sources excluded, AVX guards)
4. EXL3 vLLM overlay (exl3.py quantization + model deltas + router fallback)

## Patches

| # | Patch | File | Fixes |
|---|-------|------|-------|
| 1 | exl3 registration | `patch_exl3_init.py` | Registers `Exl3Config`/`Exl3MoEMethod` in the quantization registry |
| 2 | Config override | `patch_model_config.py` | Adds `"exl3"` to the DeepSeek V4 quant override list |
| 3 | EXL3 env vars | `patch_envs_exl3.py` | Registers `VLLM_EXL3_*` (suppresses unknown-env warnings) |
| 4 | Model deltas | `patch_model_exl3.py` | `packed_modules_mapping`, `_rankN` mapper strip, expert rank-name normalize, `skip_weight_name_before_load` (keeps native DSPark) |
| 5 | Router fallback | `patch_router_k216.py` | Pure-Torch `_topk_softplus_sqrt_torch` for REAP K216 (216 experts, unsupported by the CUDA op) |
| 6 | Version stamp | `patch_version.py` | `vllm.__version__` → `0.26.0-exl3.dspark.sm121` |

## exl3.py runtime deltas vs stock v0.26.0

- **FP8 routing:** attention QKV/O + shared experts + indexer linears route to
  `Fp8LinearMethod` (not `UnquantizedLinearMethod`) so float8 bytes load with
  their E8M0 block scales.
- **Non-EXL3 MoE → `Fp8MoEMethod`:** layers outside the rank-sliced EXL3 range
  (e.g. the embedded DSPark MTP draft) are FP8 block-quantized; routing them to
  the unquantized fallback OOM'd (full BF16 expert buffers). Now built from the
  checkpoint's `base_quantization_config`.
- **Draft-layer storage mapping:** `model.layers.{43+i}` prefixes map back to
  the checkpoint's `mtp.{i}` storage keys so FP8 detection works for the draft.

## Build gotchas (v0.26.0 base is Ubuntu 22.04 / glibc 2.35)

- `apt-get install git` (needed to pip-install sparkinfer from GitHub)
- `pip install --no-build-isolation` (setup.py must see torch to build the CUDA ext)
- `CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include`
  (the base lacks cusparse/cublas headers in `/usr/local/cuda/include`)
- `TORCH_CUDA_ARCH_LIST="12.1"`

## Pinned dependency versions

| Dependency | Source | Version / Commit |
|---|---|---|
| Base image | `vllm/vllm-openai:v0.26.0` | tag `v0.26.0` |
| SparkInfer (`sparkinfer` pkg) | `brandonmmusic-max/b12x` | commit `669a12ddc7cf3021e91a25f398b1a883b703fd12` (branch `exl3-trellis-fused` → sparkinfer 1.0.1) |
| ExLlamaV3 | `exllamav3-src/` (committed) | upstream `0b9745c526a13d5b30f1b58a864efc1932d3d9eb` (v1.3.0) + ARM64 SM121 patches |

## Usage

Served via `vllm-manager.sh` in `~/vllm-starters`:

```bash
cd ~/vllm-starters
DRY_RUN=false ./vllm-manager.sh start --model deepseek-v4-flash-0731-spark-v26
```

**Config:** `~/vllm-starters/models/deepseek-v4-flash-0731-spark-v26.yaml`

First-acceptance profile (per the model card): FLASHINFER_MLA attention,
gpu-memory-utilization 0.88, max-model-len 8192, no speculative config.

**API:** `http://localhost:8000/v1/chat/completions` (key: `sk-dgx-spark-qwen-777`)

## Status

- **Working:** EXL3 model serves correctly on v0.26.0. Verified `"2 + 2 = 4."`
  and coherent poem generation. Decode ~18.5 t/s (memory-bound), prefill
  TTFT ~0.2s on short prompts.
- **DSPark speculative decoding:** not yet enabled. The model card recommends
  "MTP remain disabled for first acceptance". 3 DSpark/MTP expert scopes are
  present in the checkpoint; the native v0.26.0 draft loader needs additional
  work (draft-layer FP8 expert routing) before it can run.
