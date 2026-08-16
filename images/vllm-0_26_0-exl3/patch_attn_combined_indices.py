"""Backport vLLM PR #50298 onto v0.26.0: drop redundant full kernels for DSv4.

Upstream:  vllm-project/vllm commit 837eae64580c885101ee95b073aafb27a485e7ce
           "[DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel
            performance improvement (#50298)"  (Wentao Ye, Jul 30 2026)

The PR stops allocating a fresh fused `combined_indices`/`combined_lens`
buffer inside `combine_topk_swa_indices` and instead lets the caller pass a
pre-allocated workspace (`out=`). The DSv4 attention layer reserves two extra
int32 workspaces via `workspace_manager.get_simultaneous` (the combined-topk
matrix and the per-token lengths vector) once per forward instead of having
every kernel launch do a `torch.full`/`torch.empty` alloc (the "redundant full
kernel"). On the warmup pass it reserves the same shapes so workspace
availability matches the real prefill path.

This script is a fail-closed, verified backport onto all **five** locations
the commit touched that exist in v0.26.0:

  cache_utils.py  vllm/models/deepseek_v4/common/ops/cache_utils.py
    A) combine_topk_swa_indices gains `out: tuple[torch.Tensor,
       torch.Tensor] | None = None`.
    B) The `combined_indices = torch.full(...)` /
       `combined_lens = torch.empty(...)` allocs are wrapped in
       `if out is None: <alloc> else: combined_indices, combined_lens = out`.

  flashmla.py     vllm/models/deepseek_v4/nvidia/flashmla.py
    C) import `round_up` from vllm.utils.math_utils.
    D-warmup)  In `forward`'s warmup branch (attn_metadata is None), the
       workspace reservation now also requests the two int32 workspaces
       `((max_num_batched_tokens, combined_topk), int32)` and
       `((max_num_batched_tokens,), int32)`, with `top_k`/`combined_topk`
       derived from `topk_indices_buffer`/`round_up`.
    D)  In `_forward_prefill`: declare `combined_topk =
       round_up(top_k + self.window_size, 128)` before the chunk loop; change
       `kv = workspace_manager.get_simultaneous(bf16)[0]` into a single
       `get_simultaneous` for the bf16 kv + two int32 workspaces unpacked as
       `kv, combined_indices_out, combined_lens_out`; slice each out-buffer to
       `[: query_end - query_start]` before the combine call; pass
       `out=(combined_indices_out, combined_lens_out)` to
       `combine_topk_swa_indices`.

Each OLD anchor is taken verbatim from v0.26.0 (git tag v0.26.0) text so
indentation matches exactly. NEW text is copied from upstream commit
837eae64580c885101ee95b073aafb27a485e7ce except hunk C's anchor spacing,
which is v0.26.0's (functionally identical).

Maximum allowed deviation from upstream: `combined_indices_out`/`combined_lens_out`
are still slices of the reserved workspace passed back in as `out=`; the kernel
itself and the `get_simultaneous` shapes are byte-for-byte upstream.

Fail-closed behaviour: if any anchor is missing or ambiguous (count != 1), the
script aborts BEFORE writing any file Real edits, prints FAIL per hunk)Skip.

Verification: run against copies of both v0.26.0 files with path overrides
(VLLM_CACHE_UTILS / VLLM_FLASHMLA env vars), confirm all hunks OK and both
patched outputs py_compile clean. See head of commit message of this file's
sibling script patch_dsv4_topk_k216.py for the established pattern.
"""

import os
import sys
from pathlib import Path

import py_compile  # noqa: E402

_BASE = "/usr/local/lib/python3.12/dist-packages/vllm"

# Path override knobs so this can be exercised against /tmp copies of the real
# v0.26.0 files without touching the container (see verification section).
CACHE_UTILS = Path(
    os.environ.get(
        "VLLM_CACHE_UTILS",
        f"{_BASE}/models/deepseek_v4/common/ops/cache_utils.py",
    )
)
FLASHMLA = Path(
    os.environ.get(
        "VLLM_FLASHMLA",
        f"{_BASE}/models/deepseek_v4/nvidia/flashmla.py",
    )
)

# --- cache_utils.py -------------------------------------------------------
# A) signature gains `out=`.
A_OLD = (
    "    M: int,\n"
    "    N: int,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)
A_NEW = (
    "    M: int,\n"
    "    N: int,\n"
    "    out: tuple[torch.Tensor, torch.Tensor] | None = None,\n"
    ") -> tuple[torch.Tensor, torch.Tensor]:\n"
)

# B) wrap the two allocs behind `if out is None:`.
B_OLD = (
    "    combined_indices = torch.full(\n"
    "        (num_tokens, combined_topk),\n"
    "        fill_value=-1,\n"
    "        dtype=torch.int32,\n"
    "        device=topk_indices.device,\n"
    "    )\n"
    "    combined_lens = torch.empty(\n"
    "        num_tokens, dtype=torch.int32, device=topk_indices.device\n"
    "    )\n"
)
B_NEW = (
    "    if out is None:\n"
    "        combined_indices = torch.full(\n"
    "            (num_tokens, combined_topk),\n"
    "            fill_value=-1,\n"
    "            dtype=torch.int32,\n"
    "            device=topk_indices.device,\n"
    "        )\n"
    "        combined_lens = torch.empty(\n"
    "            num_tokens, dtype=torch.int32, device=topk_indices.device\n"
    "        )\n"
    "    else:\n"
    "        combined_indices, combined_lens = out\n"
)

# --- flashmla.py ----------------------------------------------------------
# C) import round_up (anchor spacing is v0.26.0's).
C_OLD = (
    "from vllm.models.deepseek_v4.sparse_mla import (\n"
    "    DeepseekV4FlashMLABackend,\n"
    "    DeepseekV4FlashMLAMetadata,\n"
    ")\n"
    "from vllm.v1.attention.ops.flashmla import (\n"
)
C_NEW = (
    "from vllm.models.deepseek_v4.sparse_mla import (\n"
    "    DeepseekV4FlashMLABackend,\n"
    "    DeepseekV4FlashMLAMetadata,\n"
    ")\n"
    "from vllm.utils.math_utils import round_up\n"
    "from vllm.v1.attention.ops.flashmla import (\n"
)

# D-warmup) warmup branch in `forward`: derive top_k/combined_topk and reserve
# the two int32 workspaces alongside the bf16 kv gather workspace.
DW_OLD = (
    "            M = N + self.window_size + self.max_num_batched_tokens\n"
    "            current_workspace_manager().get_simultaneous(\n"
    "                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),\n"
    "            )\n"
)
DW_NEW = (
    "            M = N + self.window_size + self.max_num_batched_tokens\n"
    "            assert self.topk_indices_buffer is not None\n"
    "            top_k = 0 if swa_only else self.topk_indices_buffer.shape[-1]\n"
    "            combined_topk = round_up(top_k + self.window_size, 128)\n"
    "            current_workspace_manager().get_simultaneous(\n"
    "                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),\n"
    "                ((self.max_num_batched_tokens, combined_topk), torch.int32),\n"
    "                ((self.max_num_batched_tokens,), torch.int32),\n"
    "            )\n"
)

# D) _forward_prefill: combined_topk before loop; get_simultaneous for kv + 2
# int32 workspaces unpacked into kv / combined_indices_out / combined_lens_out.
D_OLD = (
    "        workspace_manager = current_workspace_manager()\n"
    "        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:\n"
    "            chunk_size = chunk_end - chunk_start\n"
    "            kv = workspace_manager.get_simultaneous(\n"
    "                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),\n"
    "            )[0]\n"
)
D_NEW = (
    "        workspace_manager = current_workspace_manager()\n"
    "        combined_topk = round_up(top_k + self.window_size, 128)\n"
    "        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:\n"
    "            chunk_size = chunk_end - chunk_start\n"
    "            workspace = workspace_manager.get_simultaneous(\n"
    "                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),\n"
    "                ((self.max_num_batched_tokens, combined_topk), torch.int32),\n"
    "                ((self.max_num_batched_tokens,), torch.int32),\n"
    "            )\n"
    "            kv, combined_indices_out, combined_lens_out = workspace\n"
)

# D-slice) trim the out-buffers to the chunk's token span before combine.
DS_OLD = (
    "            query_end = (\n"
    "                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base\n"
    "            )\n"
    "\n"
    "            combined_indices, combined_lens = combine_topk_swa_indices(\n"
)
DS_NEW = (
    "            query_end = (\n"
    "                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base\n"
    "            )\n"
    "            combined_indices_out = combined_indices_out[: query_end - query_start]\n"
    "            combined_lens_out = combined_lens_out[: query_end - query_start]\n"
    "\n"
    "            combined_indices, combined_lens = combine_topk_swa_indices(\n"
)

# D-out) pass the pre-allocated workspace into the combine kernel.
DO_OLD = (
    "                top_k,\n"
    "                chunk_M,\n"
    "                chunk_N,\n"
    "            )\n"
)
DO_NEW = (
    "                top_k,\n"
    "                chunk_M,\n"
    "                chunk_N,\n"
    "                out=(combined_indices_out, combined_lens_out),\n"
    "            )\n"
)

HUNKS = [
    (CACHE_UTILS, "A cache_utils: sig out= param", A_OLD, A_NEW),
    (CACHE_UTILS, "B cache_utils: if out is None alloc", B_OLD, B_NEW),
    (FLASHMLA, "C flashmla: import round_up", C_OLD, C_NEW),
    (FLASHMLA, "D-warmup flashmla: reserve int32 ws", DW_OLD, DW_NEW),
    (FLASHMLA, "D flashmla: prefill chunk ws unpack", D_OLD, D_NEW),
    (FLASHMLA, "D-slice flashmla: trim out bufs", DS_OLD, DS_NEW),
    (FLASHMLA, "D-out flashmla: pass out=", DO_OLD, DO_NEW),
]

# Pre-scan every anchor before writing anything (fail-closed).
patched: dict[Path, str] = {}
for path, label, old, new in HUNKS:
    content = patched.get(path)
    if content is None:
        content = path.read_text()
    count = content.count(old)
    if count != 1:
        print(f"FAIL  {label}: expected 1 anchor, found {count}")
        sys.exit(1)
    patched[path] = content.replace(old, new, 1)
    print(f"OK    {label}")

# Also fail closed if a file would end up byte-identical to the anchor source.
for path, content in patched.items():
    if content == path.read_text():
        print(f"FAIL  {path}: patch produced no change")
        sys.exit(1)

# Apply.
for path, content in patched.items():
    path.write_text(content)
    print(f"OK    write {path}")

# Compile-check both patched files.
for path in patched:
    try:
        py_compile.compile(str(path), doraise=True)
        print(f"OK    compile {path}")
    except py_compile.PyCompileError as exc:
        print(f"FAIL  compile {path}: {exc}")
        sys.exit(1)

print("ALL HUNKS APPLIED AND VERIFIED")
