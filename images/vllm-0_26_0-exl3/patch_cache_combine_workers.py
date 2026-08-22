"""Backport vLLM PR #52084 onto v0.26.0: sparse top-k metadata kernel workers.

Upstream:   836aac92ff "[Perf][DSV4] Optimize sparse top-k metadata kernels for
            higher prefill throughput (#52084)"

What it does
------------
Raises the ``_combine_topk_swa_indices_kernel`` worker count 128 -> 256.
Upstream renamed the local constant to ``_COMBINE_TOPK_SWA_NUM_WORKERS``;
v0.26.0 keeps a function-local ``NUM_WORKERS``, so the port applies the same
value change at the combine-kernel launch site. The anchor includes the
launch line so it cannot collide with the identical ``NUM_WORKERS = 128`` in
the unrelated ``_dequantize_and_gather_k`` launch (~90 lines above).

Fail-closed: aborts BEFORE writing if the anchor is missing/ambiguous.
"""

import os
import sys
from pathlib import Path

import py_compile  # noqa: E402

TARGET = Path(
    os.environ.get(
        "VLLM_CACHE_UTILS",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/common/ops/cache_utils.py",
    )
)

content = TARGET.read_text()

OLD = """    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)]("""

NEW = """    NUM_WORKERS = 256
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)]("""

if NEW in content:
    print("OK    combine workers: already applied")
elif content.count(OLD) != 1:
    print(f"FAIL  combine workers: expected 1 anchor, found {content.count(OLD)}")
    raise SystemExit(1)
else:
    content = content.replace(OLD, NEW, 1)
    TARGET.write_text(content)
    print("OK    combine workers: applied (128 -> 256)")

py_compile.compile(str(TARGET), doraise=True)
print("OK    compile")
print("ALL HUNKS APPLIED AND VERIFIED")