# Qwen3.6-35B-A3B-NVFP4-MTP — CUDA Graph Crash Fix (Option B)

**Date:** 2026-06-10
**Status:** Applied fix B — waiting for validation on server

## Problem

Engine core segfaults inside `at::cuda::CUDAGraph::replay()` → `cuGraphLaunch` during MTP speculative decoding. Happens ~10s after a request. Process dies with `EngineDeadError`.

Root cause: **Triton MOE kernel inside CUDA graph replay path** is unstable on Blackwell (sm_121a) with NVFP4 + fp8 kv-cache.

## Applied Fix: Option B — MTP moe_backend → marlin

Changed `--speculative-config` from:

```json
{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}
```

to:

```json
{"method":"mtp","num_speculative_tokens":3,"moe_backend":"marlin"}
```

Rationale: The same model instance runs regular (non-MTP) traffic fine with `--moe-backend marlin`. The triton backend was only chosen for MTP — marlin is already proven stable on this hardware.

## NOT Applied (yet)

- **Option C:** `VLLM_USE_CUDA_GRAPH=0` — nuclear option, disables graph capture entirely. Slower. Applied only if B alone doesn't fix it.

## To Test

```bash
# Deploy updated YAML to server
./vllm-manager.sh --remote restart --model qwen3.6-35b-a3b-nvfp4-mtp

# Run load — monitor for segfault (same ~10s interval between requests)
# Check: no "Segfault encountered", no EngineDeadError
```

## If B Doesn't Work

Add `VLLM_USE_CUDA_GRAPH=0` to env block → Option C.
If that still crashes → consider Option A (disable MTP entirely).

## Reference: Local YAML

`models/qwen3.6-35b-a3b-nvfp4-mtp.yaml`
