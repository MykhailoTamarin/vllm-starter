#!/usr/bin/env python3
"""Runtime patch: extend flashinfer_b12x backend for mixed-quant models.

The Unsloth NVFP4 model mixes NVFP4 (MoE experts) and W8A8 FP8 (GDN attention,
shared expert) quantization. The flashinfer_b12x backend was NVFP4-only. This
patch adds the missing kernel/mapping entries so both quant types work.

Patch 1 — linear/__init__.py: Add FP8 kernels to the flashinfer_b12x backend set
  so --linear-backend=flashinfer_b12x still finds a kernel for W8A8 FP8 layers.

Patch 2 — linear/__init__.py: Uncomment FlashInferB12xNvFp4LinearKernel from
  _POSSIBLE_NVFP4_KERNELS so --linear-backend=flashinfer_b12x finds it for
  NVFP4 linear layers (it was excluded from auto-selection only).

Patch 3 — fused_moe/oracle/fp8.py: Map flashinfer_b12x -> flashinfer_cutlass in
  map_fp8_backend so the FP8 shared-expert MoE path doesn't reject it.
"""

from pathlib import Path

# ── Targets ──────────────────────────────────────────────────────────────

LINEAR_INIT = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/kernels/linear/__init__.py"
)

FP8_ORACLE = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/layers/fused_moe/oracle/fp8.py"
)

# ── Patch 1: Add FP8 kernels to flashinfer_b12x backend set ─────────────

MAP_OLD = (
    '    "flashinfer_b12x": {\n'
    "        FlashInferB12xNvFp4LinearKernel,\n"
    "    },"
)

MAP_NEW = (
    '    "flashinfer_b12x": {\n'
    "        FlashInferB12xNvFp4LinearKernel,\n"
    "        FlashInferFP8ScaledMMLinearKernel,\n"
    "        FlashInferFp8DeepGEMMDynamicBlockScaledKernel,\n"
    "        CutlassFP8ScaledMMLinearKernel,\n"
    "    },"
)

# ── Patch 2: Uncomment FlashInferB12xNvFp4LinearKernel in NVFP4 list ────

NVFP4_OLD = (
    "        FlashInferCuteDslNvFp4LinearKernel,\n"
    "        # FlashInferB12xNvFp4LinearKernel excluded from auto-selection until\n"
    "        # upstream CUTLASS SM121 MMA op guard is resolved; use\n"
    "        # --linear-backend flashinfer_b12x to opt in explicitly.\n"
    "        FlashInferCutlassNvFp4LinearKernel,"
)

NVFP4_NEW = (
    "        FlashInferCuteDslNvFp4LinearKernel,\n"
    "        FlashInferB12xNvFp4LinearKernel,\n"
    "        FlashInferCutlassNvFp4LinearKernel,"
)

# ── Patch 3: Map flashinfer_b12x -> flashinfer_cutlass in FP8 MoE oracle ─

FP8_MAP_OLD = (
    '        "flashinfer_cutlass": Fp8MoeBackend.FLASHINFER_CUTLASS,'
)

FP8_MAP_NEW = (
    '        "flashinfer_b12x": Fp8MoeBackend.TRITON,\n'
    '        "flashinfer_cutlass": Fp8MoeBackend.FLASHINFER_CUTLASS,'
)

# ── Apply patches ────────────────────────────────────────────────────────

def _apply_text(target: Path, old: str, new: str, label: str) -> int:
    if not target.is_file():
        print(f"SKIP {label}: {target} not found")
        return 0
    src = target.read_text()
    if old not in src:
        if new in src:
            print(f"SKIP {label}: already applied")
            return 0
        print(f"FAIL {label}: old text not found in {target}")
        return 0
    src = src.replace(old, new)
    target.write_text(src)
    print(f"OK   {label}")
    return 1


changes = 0
changes += _apply_text(LINEAR_INIT, MAP_OLD, MAP_NEW,
                        "Patch 1: FP8 kernels in flashinfer_b12x backend set")
changes += _apply_text(LINEAR_INIT, NVFP4_OLD, NVFP4_NEW,
                        "Patch 2: B12x kernel in _POSSIBLE_NVFP4_KERNELS")
changes += _apply_text(FP8_ORACLE, FP8_MAP_OLD, FP8_MAP_NEW,
                        "Patch 3: flashinfer_b12x -> triton (FP8 shared expert MoE fallback)")

if changes:
    print(f"\nApplied {changes}/3 patches.")
else:
    print("\nNothing to patch.")
