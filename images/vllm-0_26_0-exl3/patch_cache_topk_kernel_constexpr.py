"""Backport vLLM PR #51967 onto v0.26.0: topk index kernel compile-time
constants.

Upstream:   83f591d7f6 "[Perf][DSV4] Optimize global top-k index kernel with
            compile-time constants (#51967)"

What it does
------------
Makes five loop/size parameters of
``_compute_global_topk_indices_and_lens_kernel`` ``tl.constexpr`` so Triton
can fully unroll/specialize the kernel instead of treating them as runtime
values. All five anchors below are VERBATIM v0.26.0 text (verified against
git show v0.26.0:vllm/models/deepseek_v4/common/ops/cache_utils.py).

Fail-closed: aborts BEFORE writing if any anchor is missing/ambiguous.
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

OLD = """def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):"""

NEW = """def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride: tl.constexpr,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride: tl.constexpr,
    topk: tl.constexpr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride: tl.constexpr,
    block_size: tl.constexpr,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):"""

if NEW in content:
    print("OK    kernel constexpr: already applied")
elif content.count(OLD) != 1:
    print(f"FAIL  kernel constexpr: expected 1 anchor, found {content.count(OLD)}")
    raise SystemExit(1)
else:
    content = content.replace(OLD, NEW, 1)
    TARGET.write_text(content)
    print("OK    kernel constexpr: applied")

py_compile.compile(str(TARGET), doraise=True)
print("OK    compile")
print("ALL HUNKS APPLIED AND VERIFIED")