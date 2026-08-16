"""Backport vLLM PR #50365 onto v0.26.0's sparse_utils.py (index-remap atomics drop).

Upstream: commit 9e6be4a72bd29ecf2168a41f659aebfe2cdeda4a
    "[Perf][Sparse MLA] Drop the atomic contention in the index remap (#50365)"
    https://github.com/vllm-project/vllm/pull/50365

What it does
------------
When the top-k column tiling has exactly one Triton program owning an entire
row (BLOCK_N == NUM_TOPK_TOKENS, and NUM_TOPK_TOKENS is a power of two), the
row's valid-slot count is a single in-register reduction. There is no cross-tile
races, so the atomic_add slot allocator / counter is replaced by a plain store,
and the zero-initialized counter buffer becomes torch.empty. This removes atomic
contention on the sparse-MLA index remap hot path.

Scope ported
------------
Hunks 1-4 of the upstream commit (the parts scoped in this task):
  1. Add ``SINGLE_TILE: tl.constexpr`` kernel parameter.
  2. COMPACT_TO_FRONT path: store-vs-atomic base allocation.
  3. COUNT_VALID path: store-vs-atomic accumulation + new ``_remap_tiling`` helper.
  4. ``triton_convert_req_index_to_global_index``: drive tiling via
     ``_remap_tiling``, pick torch.empty vs torch.zeros, and pass
     ``block_n`` / ``single_tile`` / ``num_warps`` to the launch.

Faithful vs adapted
-------------------
FAITHFUL. The v0.26.0 text for all four hunks matches the upstream OLD side
exactly (verified against ``git show v0.26.0:.../sparse_utils.py``); upstream NEW
text is copied verbatim (only the module-level ``_remap_tiling`` docstring is
retained). The DCP-path hunks (5-6) of the upstream commit are intentionally out
of scope for this task and are not applied.

Container target
----------------
/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_utils.py
"""

from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "v1/attention/backends/mla/sparse_utils.py"
)

# ---------------------------------------------------------------------------
# Hunk 1: add SINGLE_TILE constexpr param right after COUNT_VALID
# ---------------------------------------------------------------------------
H1_OLD = """    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
"""
H1_NEW = """    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
    # BLOCK_N == NUM_TOPK_TOKENS: one program owns the row, so the valid count
    # is an in-register reduction and needs no atomic.
    SINGLE_TILE: tl.constexpr,
"""

# ---------------------------------------------------------------------------
# Hunk 2: COMPACT_TO_FRONT -- store base when single-tile, else atomic
# ---------------------------------------------------------------------------
H2_OLD = """        # sum gives each valid lane a distinct local offset; one atomic add of the
        # tile's valid count reserves a contiguous base across racing tiles. The
        # out buffer is pre-filled with -1, so unwritten tail slots stay -1.
        is_valid = (~is_invalid_tok).to(tl.int32)
        local_offset = tl.cumsum(is_valid) - is_valid
        tile_valid_count = tl.sum(is_valid)
        base = tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)
"""
H2_NEW = """        # sum gives each valid lane a distinct local offset; one atomic add of the
        # tile's valid count reserves a contiguous base across racing tiles. The
        # out buffer is pre-filled with -1, so unwritten tail slots stay -1.
        # With no racing tiles the base is 0 and the allocator becomes a store.
        is_valid = (~is_invalid_tok).to(tl.int32)
        local_offset = tl.cumsum(is_valid) - is_valid
        tile_valid_count = tl.sum(is_valid)
        if SINGLE_TILE:
            base = 0
            tl.store(valid_count_ptr + token_id, tile_valid_count)
        else:
            base = tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)
"""

# ---------------------------------------------------------------------------
# Hunk 3: COUNT_VALID store-vs-atomic + insert _remap_tiling helper before the
# triton_convert_req_index_to_global_index function definition
# ---------------------------------------------------------------------------
H3_OLD = """        # Count valid indices in this tile and atomically add to row total
        if COUNT_VALID:
            tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
            tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def triton_convert_req_index_to_global_index(
"""
H3_NEW = """        # Accumulate the tile's valid count into the row total; a single tile's
        # reduction *is* the total.
        if COUNT_VALID:
            tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
            if SINGLE_TILE:
                tl.store(valid_count_ptr + token_id, tile_valid_count)
            else:
                tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def _remap_tiling(
    NUM_TOPK_TOKENS: int, BLOCK_N: int, count_valid: bool
) -> tuple[bool, int, int, int]:
    \"\"\"Pick the column tiling for the index remap kernel.

    Counting the valid slots per row is the only reason the column tiles have to
    talk to each other, so when counting give one program the whole row: the
    count becomes an in-register reduction plus a plain store, needing neither
    atomics nor a zero-initialized counter. The row is one ``tl.arange``, so this
    needs a power-of-two width; other top-k sizes stay tiled and atomic.

    Returns:
        (single_tile, block_n, tiles_per_row, num_warps)
    \"\"\"
    single_tile = (
        count_valid and triton.next_power_of_2(NUM_TOPK_TOKENS) == NUM_TOPK_TOKENS
    )
    if single_tile:
        return True, NUM_TOPK_TOKENS, 1, 8
    return False, BLOCK_N, NUM_TOPK_TOKENS // BLOCK_N, 4


def triton_convert_req_index_to_global_index(
"""

# ---------------------------------------------------------------------------
# Hunk 4a: replace tiles_per_row with _remap_tiling unpacking
# (anchor disambiguated from the DCP function by the following comment line)
# ---------------------------------------------------------------------------
H4A_OLD = """    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
"""
H4A_NEW = """    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]

    single_tile, block_n, tiles_per_row, num_warps = _remap_tiling(
        NUM_TOPK_TOKENS, BLOCK_N, return_valid_counts
    )

    # Ensure contiguous tensors on the same device
"""

# ---------------------------------------------------------------------------
# Hunk 4b: torch.empty vs torch.zeros for the valid-count buffer
# ---------------------------------------------------------------------------
H4B_OLD = """    # Allocate valid count buffer if needed (must be zero-initialized for atomics)
    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )
"""
H4B_NEW = """    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        # Zero-init only matters for the atomic accumulation path.
        alloc = torch.empty if single_tile else torch.zeros
        valid_counts = alloc(num_tokens, dtype=torch.int32, device=token_indices.device)
"""

# ---------------------------------------------------------------------------
# Hunk 4c: kernel launch -- BLOCK_N -> block_n, insert single_tile constexpr
# ---------------------------------------------------------------------------
H4C_OLD = """        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        HAS_PREFILL_WORKSPACE,
        return_valid_counts,
        False,  # COMPACT_TO_FRONT: keep input column == output column
"""
H4C_NEW = """        max_num_blocks_per_req,
        BLOCK_SIZE,
        block_n,
        HAS_PREFILL_WORKSPACE,
        return_valid_counts,
        single_tile,
        False,  # COMPACT_TO_FRONT: keep input column == output column
"""

# ---------------------------------------------------------------------------
# Hunk 4d: pass num_warps to the launch.
# Anchored on the `single_tile,` line inserted by Hunk 4c — that string only
# exists in the `triton_convert_req_index_to_global_index` launch (the DCP
# launch carries `count_valid, compact_valid_to_front` there instead), so this
# span disambiguates the two identical stride-tails.
# ---------------------------------------------------------------------------
H4D_OLD = """        single_tile,
        False,  # COMPACT_TO_FRONT: keep input column == output column
        # DCP disabled (no-op de-interleave)
        1,
        0,
        1,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
"""
H4D_NEW = """        single_tile,
        False,  # COMPACT_TO_FRONT: keep input column == output column
        # DCP disabled (no-op de-interleave)
        1,
        0,
        1,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
        num_warps=num_warps,
    )
"""

HUNKS = [
    ("hunk1  SINGLE_TILE param", H1_OLD, H1_NEW),
    ("hunk2  COMPACT_TO_FRONT store-vs-atomic", H2_OLD, H2_NEW),
    ("hunk3  COUNT_VALID + _remap_tiling helper", H3_OLD, H3_NEW),
    ("hunk4a tiles_per_row -> _remap_tiling", H4A_OLD, H4A_NEW),
    ("hunk4b torch.empty/torch.zeros alloc", H4B_OLD, H4B_NEW),
    ("hunk4c launch block_n + single_tile", H4C_OLD, H4C_NEW),
    ("hunk4d launch num_warps", H4D_OLD, H4D_NEW),
]

with open(TARGET) as f:
    content = f.read()

results = []
fatal = False

for label, old, new in HUNKS:
    if new in content:
        results.append(f"OK    {label}: already applied")
    elif content.count(old) != 1:
        results.append(
            f"FAIL  {label}: expected 1 anchor, found {content.count(old)}"
        )
        fatal = True
    else:
        content = content.replace(old, new, 1)
        results.append(f"OK    {label}: applied")

if not fatal:
    with open(TARGET, "w") as f:
        f.write(content)
    import py_compile  # noqa: E402

    py_compile.compile(str(TARGET), doraise=True)
    results.append("OK    compile")

for r in results:
    print(r)

raise SystemExit(1 if fatal else 0)
