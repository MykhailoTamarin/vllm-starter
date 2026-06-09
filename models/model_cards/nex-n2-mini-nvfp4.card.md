
# Nex-N2-mini NVFP4

NVFP4-quantized [Nex-N2-mini](https://huggingface.co/Nex-AGI/Nex-N2-mini) (Qwen3.5-MoE-35B fine-tune) optimized for NVIDIA Blackwell (SM 12.1) serving via vLLM with FlashInfer CUTLASS kernels.

**3.2× compression** (70 GB BF16 → 22.1 GiB NVFP4) with zero quality loss on capability tests.

## Quick Start (Docker)

The easiest way to serve this model — auto-downloads on first run:

```bash
docker run -d --name nex-n2-mini-nvfp4 \
  --gpus all \
  --shm-size=8g \
  -e HF_TOKEN=hf_xxxxx \
  -v nex-n2-model:/mnt/model \
  -p 8000:8000 \
  ghcr.io/r0b0tlab/nex-n2-mini-nvfp4:latest
```

Then query:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"r0b0tlab/nex-n2-mini-nvfp4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

**Docker Compose:**
```bash
echo "HF_TOKEN=hf_xxxxx" > .env
docker compose up -d
```

See the [GitHub repo](https://github.com/r0b0tlab/nex-n2-mini-nvfp4) for full documentation, environment variables, and AGENTS.md.

## Model Details

| Property | Value |
|---|---|
| Base model | [Nex-AGI/Nex-N2-mini](https://huggingface.co/Nex-AGI/Nex-N2-mini) |
| Architecture | Qwen3_5MoeForConditionalGeneration |
| Parameters | 35B total / 3B active (MoE, 256 experts, top-8 routing) |
| Layers | 40 (30 linear attention + 10 full attention) |
| Vocabulary | 248,320 tokens |
| Vision encoder | ViT (27 blocks, 1152 hidden) — kept BF16 |
| Original size | ~70 GB (BF16) |
| NVFP4 size | ~22.1 GiB |

## Quantization

- **Method**: NVFP4 via [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 0.44.0
- **Scope**: MLP-only (expert projections + shared expert)
- **Group size**: 16
- **Calibration**: 128 samples from CNN/DailyMail
- **KV cache**: FP8 e4m3 with per-layer calibrated scales
- **Expert calibration**: 10,240/10,240 (100%)

**Included in checkpoint:**
- NVFP4 quantized MoE weights (256 experts × 40 layers)
- BF16 attention weights (self_attn, linear_attn)
- BF16 vision encoder weights
- BF16 lm_head.weight
- Calibrated FP8 KV cache scales (k: 0.016–0.038, v: 0.010–0.040, q: 0.043)

## Benchmarks

Tested on NVIDIA GB10 (Blackwell SM 12.1), vLLM v0.22.0, FlashInfer CUTLASS NVFP4.

### Throughput (llama-benchy, 3 runs per test)

| Test | Throughput | Peak t/s |
|---|---|---|
| pp2048 | 1,974 t/s | — |
| tg128 | **33.35 t/s** | 38.33 |
| pp2048 @ d4096 | 4,007 t/s | — |
| pp2048 @ d8192 | 4,793 t/s | — |
| pp2048 @ d16384 | 5,017 t/s | — |

Decode is rock-stable at 32–33 t/s across all context depths (0–16K). Only 2.8% degradation.

### Concurrency Scaling

| Concurrency | Aggregate t/s | Per-request t/s | Power | Temp |
|---|---|---|---|---|
| C1 | 28.5 | 28.6 | 20.4 W | 45°C |
| C2 | 51.6 | 25.8 | 18.4 W | 46°C |
| C4 | 105.3 | 26.3 | 20.2 W | 47°C |
| C8 | **185.5** | 23.2 | 22.1 W | 48°C |

6.5× scaling at C8. 8.42 t/s/W efficiency. Peak 23.3W at 48°C.

### Capability Tests: 13/13 Passed

Math (3/3), Reasoning (3/3), Coding (3/3), Knowledge (2/2), Instruction (2/2).

## Serving Stack

- **vLLM** v0.22.0 (Docker, aarch64)
- **FlashInfer CUTLASS** NVFP4 GEMM kernel + MoE backend
- **FP8 KV cache** with per-layer calibrated scales (~10–15% throughput improvement)

## Known Issues

- `max_num_batched_tokens` must be >= 2096 (Mamba block alignment)
- NVFP4 KV cache unavailable (requires `torch.nvfp4`, NVIDIA-internal only)
- Vision encoder profiling takes ~4–5 min on first startup
- Requires NVIDIA Blackwell GPU (SM 12.1) for NVFP4 kernels

## Links

- **Container**: [ghcr.io/r0b0tlab/nex-n2-mini-nvfp4](https://github.com/r0b0tlab/nex-n2-mini-nvfp4/pkgs/container/nex-n2-mini-nvfp4)
- **Source code**: [github.com/r0b0tlab/nex-n2-mini-nvfp4](https://github.com/r0b0tlab/nex-n2-mini-nvfp4)
- **Base model**: [Nex-AGI/Nex-N2-mini](https://huggingface.co/Nex-AGI/Nex-N2-mini)
- **Quantization**: [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- **Serving**: [vLLM](https://github.com/vllm-project/vllm)

## License

Apache 2.0.

