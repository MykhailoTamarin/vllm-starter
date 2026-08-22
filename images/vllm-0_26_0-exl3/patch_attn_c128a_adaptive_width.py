"""Backport vLLM PR #52823 onto v0.26.0: adaptive C128A topk width.

Upstream:   e6f35d3c69 "[DSv4 Perf] Adaptive topk width for dsv4, making
            #50004 back (#52823)"  (current main incarnation of #50004)

What it does
------------
Short-context C128A metadata used to run at the full planned
``c128a_max_compressed`` width (buffer-sized) for every forward. Now the
active width is driven by the batch::

    active_topk_width = min(
        max(next_pow2(max(max_seq_len // compress_ratio, 1)), _C128A_TOPK_ALIGNMENT),
        self.c128a_max_compressed,
    )

so decode/prefill with short sequences run the topk/index kernel at a
128-aligned width close to the sequence's compressed length instead of the
max capacity — fewer inactive columns on the sparse-MLA hot path.
``build_c128a_topk_metadata`` slices the pre-allocated buffers to
``:max_compressed_tokens`` and asserts stride stability so CUDA-graph
addresses stay valid (the buffers keep their full stride(0)).

Ported as two hunks (builder + build fn) with anchors verified VERBATIM
against the real v0.26.0 ``sparse_mla.py`` (``_C128A_TOPK_ALIGNMENT = 128``,
``self.c128a_max_compressed``, buffers, and ``cm.max_seq_len`` all exist in
v0.26.0; ``triton`` is imported via ``vllm.triton_utils``).

Fail-closed: aborts BEFORE writing if any anchor is missing/ambiguous.
"""

import os
import sys
from pathlib import Path

import py_compile  # noqa: E402

TARGET = Path(
    os.environ.get(
        "VLLM_SPARSE_MLA",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/sparse_mla.py",
    )
)


def apply(content: str, old: str, new: str, label: str) -> str:
    if new in content:
        print(f"OK    {label}: already applied")
        return content
    n = content.count(old)
    if n != 1:
        print(f"FAIL  {label}: expected 1 anchor, found {n}")
        raise SystemExit(1)
    print(f"OK    {label}")
    return content.replace(old, new, 1)


# --- Hunk 1: builder — compute active_topk_width, pass to the kernel --------
H1_OLD = """        assert cm.positions is not None, (
            "positions is required for C128A metadata build"
        )
        block_size = self.kv_cache_spec.block_size // self.compress_ratio
        global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
            cm.positions[:num_total],
            self.compress_ratio,
            num_decode_tokens,
            req_id_per_token,
            cm.block_table_tensor[:num_decodes],
            block_size,
            cm.slot_mapping,
            self.c128a_global_decode_buffer,
            self.c128a_decode_lens_buffer,
            self.c128a_prefill_buffer,
            max_compressed_tokens=self.c128a_max_compressed,
        )"""

H1_NEW = """        assert cm.positions is not None, (
            "positions is required for C128A metadata build"
        )
        # Adaptive width (#52823): run the topk/index kernel at an
        # 128-aligned width close to the batch's compressed length instead of
        # the full planned capacity (short contexts pay for fewer inactive
        # columns). Clamped to the pre-allocated buffer width so CUDA-graph
        # addresses stay stable (stride(0) is unchanged by the column slice).
        active_topk_width = min(
            max(
                triton.next_power_of_2(max(cm.max_seq_len // self.compress_ratio, 1)),
                _C128A_TOPK_ALIGNMENT,
            ),
            self.c128a_max_compressed,
        )
        assert active_topk_width >= cm.max_seq_len // self.compress_ratio
        assert active_topk_width % _C128A_TOPK_ALIGNMENT == 0
        block_size = self.kv_cache_spec.block_size // self.compress_ratio
        global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
            cm.positions[:num_total],
            self.compress_ratio,
            num_decode_tokens,
            req_id_per_token,
            cm.block_table_tensor[:num_decodes],
            block_size,
            cm.slot_mapping,
            self.c128a_global_decode_buffer,
            self.c128a_decode_lens_buffer,
            self.c128a_prefill_buffer,
            max_compressed_tokens=active_topk_width,
        )"""

# --- Hunk 2: build_c128a_topk_metadata — width-limited buffer slices --------
H2_OLD = """    num_tokens = positions.shape[0]
    num_prefill_tokens = num_tokens - num_decode_tokens

    global_decode = global_decode_buffer[:num_decode_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens]

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local"""

H2_NEW = """    num_tokens = positions.shape[0]
    num_prefill_tokens = num_tokens - num_decode_tokens
    assert max_compressed_tokens % _C128A_TOPK_ALIGNMENT == 0
    assert (
        0
        < max_compressed_tokens
        <= min(global_decode_buffer.shape[1], prefill_buffer.shape[1])
    )
    assert global_decode_buffer.stride(-1) == prefill_buffer.stride(-1) == 1

    global_decode = global_decode_buffer[:num_decode_tokens, :max_compressed_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens, :max_compressed_tokens]
    assert global_decode.stride(0) == global_decode_buffer.stride(0)
    assert prefill_local.stride(0) == prefill_buffer.stride(0)

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local"""

content = TARGET.read_text()
content = apply(content, H1_OLD, H1_NEW, "hunk1 adaptive width (builder)")
content = apply(content, H2_OLD, H2_NEW, "hunk2 width-limited slices (build fn)")
TARGET.write_text(content)

py_compile.compile(str(TARGET), doraise=True)
print("OK    compile")
print("ALL HUNKS APPLIED AND VERIFIED")