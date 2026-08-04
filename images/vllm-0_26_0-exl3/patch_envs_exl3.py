"""Register VLLM_EXL3_* env vars in v0.26.0 envs.py.

These are read by exl3.py directly via os.environ, but registering them
suppresses the "Unknown vLLM environment variable" warning at startup.
"""

from pathlib import Path

ENVS = Path("/usr/local/lib/python3.12/dist-packages/vllm/envs.py")

with open(ENVS) as f:
    content = f.read()

anchor = '    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": lambda: int(\n        os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")\n    ),'
add = """    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": lambda: int(
        os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    ),
    # EXL3 Trellis backend knobs (read by model_executor/layers/quantization/exl3.py).
    "VLLM_EXL3_TRELLIS_MIN_M": lambda: os.getenv("VLLM_EXL3_TRELLIS_MIN_M"),
    "VLLM_EXL3_TRELLIS_MAX_M": lambda: os.getenv("VLLM_EXL3_TRELLIS_MAX_M"),
    "VLLM_EXL3_TRELLIS_BLOCK_M": lambda: os.getenv("VLLM_EXL3_TRELLIS_BLOCK_M"),
    "VLLM_EXL3_PREFILL_CHUNK": lambda: os.getenv("VLLM_EXL3_PREFILL_CHUNK"),
    "VLLM_EXL3_PREFILL_TRELLIS": lambda: os.getenv("VLLM_EXL3_PREFILL_TRELLIS"),
    "VLLM_EXL3_PREFILL_BLOCK_M": lambda: os.getenv("VLLM_EXL3_PREFILL_BLOCK_M"),
    "VLLM_EXL3_EXT_PATH": lambda: os.getenv("VLLM_EXL3_EXT_PATH"),
    "VLLM_EXL3_ABI_SHIM": lambda: os.getenv("VLLM_EXL3_ABI_SHIM"),"""

if anchor not in content:
    print("FAIL  envs: anchor not found")
    raise SystemExit(1)

if "VLLM_EXL3_TRELLIS_MIN_M" in content:
    print("SKIP  envs: already applied")
else:
    content = content.replace(anchor, add, 1)
    with open(ENVS, "w") as f:
        f.write(content)
    print("OK    envs: VLLM_EXL3_* registered")
