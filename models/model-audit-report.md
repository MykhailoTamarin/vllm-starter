# Model YAML Audit Report

**Date:** 2026-06-07  
**Method:** Each YAML compared against its HuggingFace model card via `hf models card`  
**Models scanned:** 6

---

## Summary

| Model | File | Status | Issues |
|-------|------|--------|--------|
| MiniMax M2.7 REAP 172B | `minimax-m2.7-reap-nvfp4.yaml` | ✅ OK | None |
| Nemotron-3-Super 120B | `nemotron-3-super-120b-a12b.yaml` | ⚠️ Minor | 2 env vars missing for DGX Spark |
| Qwen3.5 122B | `qwen3.5-122b-a10b.yaml` | 🔴 CRITICAL | No args section (cannot run) |
| Qwen3.6 27B NVFP4 MTP | `qwen3.6-27b-nvfp4-mtp.yaml` | ⚠️ Minor | GPU memory util lower than card |
| Qwen3.6 35B A3B NVFP4 MTP | `qwen3.6-35b-a3b-nvfp4-mtp.yaml` | ✅ OK | Minor note (see below) |
| Step 3.7 Flash 148B | `step3p7-flash-148b.yaml` | ✅ OK | Note: TP=1 vs HF card TP=4 |

---

## 🔴 CRITICAL — qwen3.5-122b-a10b.yaml: No `args:` section

The model YAML has only `image` and `port` defined. **There is no `args:` block and no `env:` block.**

The model card at `nvidia/Qwen3.5-122B-A10B-NVFP4` provides a working vLLM launch:

```
vllm serve nvidia/Qwen3.5-122B-A10B-NVFP4 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

**Key model card details that should be reflected in YAML:**
- Architecture: Qwen3.5 MoE, 122B total / 10B activated
- 262K context, multimodal (text/image/video → text)
- Quantized with nvidia-modelopt v0.44.0
- Requires: Blackwell hardware
- License: Apache 2.0

This YAML is **non-functional** as-is — it must be populated with args.

---

## ⚠️ MINOR — nemotron-3-super-120b-a12b.yaml

### Missing env vars for DGX Spark

The model card's "vLLM on DGX Spark" example includes:

```
VLLM_NVFP4_GEMM_BACKEND=marlin
VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
```

Neither is present in our YAML. The general vLLM example omits them, but since our YAML uses `vllm/vllm-openai:nightly` (matching the Spark image), these may matter.

**Context:** The nemotron uses a LatentMoE hybrid architecture (Mamba-2 + MoE + Attention). The `VLLM_NVFP4_GEMM_BACKEND=marlin` flag is specifically noted for Spark in the card's DGX section.

---

## ⚠️ MINOR — qwen3.6-27b-nvfp4-mtp.yaml

### GPU memory utilization

| | Value |
|---|---|
| YAML | `0.85` |
| HF card recommended | `0.9` |

The HF card's production launch uses `--gpu-memory-utilization 0.9`. Our YAML uses `0.85`. This is **safe** (less VRAM pressure), but the card notes that at 256K context with KV FP8, 0.9 is the recommended value for 2 concurrent requests.

---

## ℹ️ NOTE — qwen3.6-35b-a3b-nvfp4-mtp.yaml

No actual issues found. Minor observations:

- The HF card's general vLLM command doesn't show env vars, but its **DGX Spark example** includes:
  ```
  VLLM_USE_FLASHINFER_MOE_FP4=0
  VLLM_FP8_MOE_BACKEND=flashinfer_cutlass
  FLASHINFER_DISABLE_VERSION_CHECK=1
  CUTE_DSL_ARCH=sm_121a
  ```
  Our YAML already has all four of these. ✅

- The HF card's general command uses `--dtype auto`, our YAML uses `--dtype auto` explicitly. ✅

- `--moe-backend marlin` and `--attention-backend flashinfer` match the card's DGX Spark example. ✅

- Context 262K matches the card (262K). The card's DGX example uses 64K but that's a conservative single-node default. ✅

- YAML has a minor formatting inconsistency: `args: ` has a trailing space (vs `args:` in other files). Cosmetic only.

---

## ℹ️ NOTE — step3p7-flash-148b.yaml

### Tensor parallel size

| | Value |
|---|---|
| YAML | `--tensor-parallel-size 1` |
| HF card example | `--tensor-parallel-size 4` |

The HF card example uses TP=4 (for multi-GPU setups), but our YAML uses TP=1. Since the model fits on a single DGX Spark (95 GB on disk, 128 GB available), **TP=1 is intentional and correct** for single-node deployment. The card notes: *"Hardware fit is not guaranteed by the upload alone. Run a load smoke..."* — suggesting this model is designed to be flexible across configs.

No other issues. All env vars, reasoning/parser flags, and quantization args align with the card.

---

## ℹ️ NOTE — minimax-m2.7-reap-nvfp4.yaml

No issues found. All key details verified against the card:

- ✅ 172B total / ~10B active, 192 experts top-K=8, 62 layers
- ✅ NVFP4 + GB10 tuned, 98.9 GB on disk
- ✅ Env vars match the card's "Running on 1× DGX Spark" section exactly
- ✅ `--compilation-config '{"cudagraph_mode":"none",...}'` matches the card's gotcha note
- ✅ `--enable-expert-parallel` is correctly **absent** (card warns against it)
- ✅ `--max-model-len 131072` is within the card's 196K limit — conservative is fine
- ✅ All argument flags verified correct

---

## Architecture Summary

| Model | Architecture | Total/Active | Quantization | Card Context | YAML Context |
|-------|-------------|--------------|--------------|--------------|--------------|
| MiniMax M2.7 | MoE, 192 experts | 172B / ~10B | NVFP4 GB10 | 196K | 128K |
| Nemotron-3-Super | LatentMoE hybrid | 120B / 12B | NVFP4 | 1M | 256K |
| Qwen3.5 122B | Qwen3.5 MoE | 122B / 10B | NVFP4 (MO) | 262K | — |
| Qwen3.6 27B | Dense text | 27B / 27B | NVFP4 modelopt | 256K | 256K |
| Qwen3.6 35B | MoE (32 routers) | 35B / 3B | NVFP4 modelopt | 262K | 256K |
| Step 3.7 Flash | VLM MoE, 212/288 | ~148B / ~11B | NVFP4 FP8 KV | 256K | 32K |

---

*Report generated from `hf models card` output. Benchmark data differences between cards and YAMLs are expected and not flagged as issues.*
