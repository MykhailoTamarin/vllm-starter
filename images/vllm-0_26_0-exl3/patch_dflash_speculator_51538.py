"""Backport vLLM PR #51538 onto v0.26.0: DFlash/DSPark draft speculator fixes.

Upstream: commit 97388c44f9 ("[Bugfix] Make DSV4 sparse MLA work end-to-end for
          plain decode, MTP, and DSpark (#51538)") — dflash/speculator.py
          sub-hunks only.

What it fixes (all on the DSPark/DFlash draft path EXL3 uses)
--------------------------------------------------------------
1. ``sample_idx_mapping`` was initialized to zeros; under CUDA graph capture
   the full pre-populated buffer is executed before a real batch populates it,
   so every padding row would scatter into request slot 0. Now initialized to
   -1 (inert sampling row).
2. ``_prepare_dflash_inputs_kernel`` treated the rejected-token suffix as valid
   context: ``is_query``/``query_off`` were computed from ``num_ctx`` instead of
   ``num_valid_ctx``, and context loads read the rejected suffix rows.
3. Null-block guard: old sliding-window context positions can map to physical
   block 0 after eviction, and rejected suffix rows are invalid context —
   neither may write draft KV into block 0. ``ctx_slot``/``q_slot`` now map to
   ``PAD_SLOT_ID`` when the block id is 0.

EXL3 relevance: DSPark (a DFlashSpeculator subclass) runs under FULL CUDA
graph (VLLM_USE_BREAKABLE_CUDAGRAPH=0) with a compact K160 draft — both the
capture-window scatter and the null-block writes are on its hot path.

Scope ported
------------
Five hunks (A..E) of the upstream commit, verbatim. The DSPark
sample_from_anchor semantics flip and the wider speculator rewrites are out of
scope for this task.

Faithful vs adapted
-------------------
FAITHFUL. v0.26.0 text for all five hunks matches the upstream OLD side exactly
(verified against ``git show v0.26.0:.../dflash/speculator.py``); upstream NEW
text copied verbatim.

Container target
----------------
/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
"""

import os
from pathlib import Path
import py_compile  # noqa: E402

TARGET = Path(
    os.environ.get(
        "VLLM_DFLASH_SPEC",
        "/usr/local/lib/python3.12/dist-packages/vllm/"
        "v1/worker/gpu/spec_decode/dflash/speculator.py",
    )
)

# --- Hunk A: sample_idx_mapping init -1 (inert rows under CUDA graph) ------
A_OLD = """        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=device
        )"""

A_NEW = """        # -1 marks an inert sampling row. CUDA graph capture can execute the
        # full buffer before a real batch has populated it, so zero would make
        # every padding row scatter into request slot 0.
        self.sample_idx_mapping = torch.full(
            (max_num_sampled_tokens,), -1, dtype=torch.int32, device=device
        )"""

# --- Hunk B: num_valid_ctx bound -------------------------------------------
B_OLD = """    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
"""

B_NEW = """    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start
"""

# --- Hunk C: is_query / query_off from num_valid_ctx ------------------------
C_OLD = """    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx
"""

C_NEW = """    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
    query_off = j - num_valid_ctx
"""

# --- Hunk D: context loads masked to valid ctx + null-block guard -----------
D_OLD = """    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
"""

D_NEW = """    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block. Old sliding-window context positions can map
    # to it after eviction; rejected suffix rows are invalid context as well.
    # Neither kind of row may write draft KV into physical block 0.
    ctx_resident = is_valid_ctx & (ctx_block_id != 0)
    ctx_slot = tl.where(
        ctx_resident,
        ctx_block_id * block_size + (ctx_pos % block_size),
        PAD_SLOT_ID,
    )
"""

# --- Hunk E: query slot null-block guard ------------------------------------
E_OLD = """    q_slot = q_block_id * block_size + (query_pos % block_size)
"""

E_NEW = """    # A null block is never a writable cache slot. This can occur when a
    # sliding-window block table contains evicted/global padding entries.
    q_resident = is_query & (q_block_id != 0)
    q_slot = tl.where(
        q_resident,
        q_block_id * block_size + (query_pos % block_size),
        PAD_SLOT_ID,
    )
"""

content = TARGET.read_text()
results = []

for label, (old, new) in {
    "A (sampler init -1)": (A_OLD, A_NEW),
    "B (num_valid_ctx)": (B_OLD, B_NEW),
    "C (is_query window)": (C_OLD, C_NEW),
    "D (ctx null-block)": (D_OLD, D_NEW),
    "E (query null-block)": (E_OLD, E_NEW),
}.items():
    if new in content:
        results.append(f"SKIP  {label}: already applied")
        continue
    n = content.count(old)
    if n != 1:
        print(f"FAIL  {label}: expected 1 anchor, found {n}")
        raise SystemExit(1)
    content = content.replace(old, new, 1)
    results.append(f"OK    {label}")

TARGET.write_text(content)
py_compile.compile(str(TARGET), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)