"""Backport vLLM PR #49486 ('[DSv4 Perf] Skip topk and router when not needed') to v0.26.0.

Upstream commit: b0cb1da1bde62a738baba33f1fbb1fcf906d29bc
PR:             vllm-project/vllm#49486  (Wentao Ye, 2026-07-23; ~3.4% E2E TTFT gain on Decode)

It patches ``vllm/models/deepseek_v4/attention.py`` in three hunks:

  Hunk 1 — add ``from vllm.triton_utils import tl, triton`` right after the
           ``DeepseekCompressor`` import.
  Hunk 2 — add the ``@triton.jit _fill_short_context_topk_indices`` kernel
           immediately after ``logger = init_logger(__name__)``.
  Hunk 3 — inside ``DeepseekV4Indexer.forward`` insert a short-context
           early-return block IMMEDIATELY BEFORE ``def wq_b_and_q_quant():``.
           When the number of compressed candidates
           ``max_seq_len // compress_ratio`` fits within ``topk_tokens``, every
           candidate is selected: build the K cache, fill
           ``topk_indices_buffer`` with a tiny Triton kernel selecting all
           rows (-1 otherwise), and return early — skipping the topk/router
           and the indexer op entirely.

v0.26.0 DELTA / adaptation notes (all verified against real v0.26.0 text):
  * ``vllm.triton_utils`` is a PACKAGE in v0.26.0 (vllm/triton_utils/__init__.py)
    — not a module as in later dev heads. It still exports ``tl`` and ``triton``
    (``__all__ = ["HAS_TRITON", "triton", "tl", "tldevice", "LOG2E", "LOGE2"]``),
    so the upstream import is byte-for-byte compatible.``
  * ``cast`` and ``Any`` are already imported in v0.26.0 attention.py
    (``from typing import TYPE_CHECKING, Any, ClassVar, cast``) — no import added.
  * ``get_forward_context`` already imported in v0.26.0.``
  * Forward-context metadata is a dict keyed by the K-cache prefix. v0.26.0's
    own ``compressor.py:370`` already does
    ``k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])``, and in
    attention.py ``self.k_cache.prefix == f"{prefix}.k_cache"`` is exactly the
    key the indexer must read — so ``attn_metadata[self.k_cache.prefix]`` (the
    upstream form) is correct for v0.26.0, no change needed.
  * The metadata object is ``DeepseekV32IndexerMetadata`` (dataclass) with the
    exact fields the patch reads: ``max_seq_len``, ``num_decode_tokens``,
    ``num_prefill_tokens``. Confirmed present in v0.26.0 indexer.py.``
  * All three OLD anchors below are VERBATIM v0.26.0 text (they differ from the
    upstream dev file in nothing at these locations), so no context drift.

Verification (against ``git show v0.26.0:vllm/models/deepseek_v4/attention.py``):
  Hunk 1 OK, Hunk 2 OK, Hunk 3 OK, py_compile OK — see VERIFY block at bottom.
"""

from pathlib import Path

ROUTER = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/attention.py"
)

# ---------------------------------------------------------------------------
# Hunk 1: import after DeepseekCompressor
# ---------------------------------------------------------------------------
H1_OLD = (
    "from vllm.models.deepseek_v4.compressor import DeepseekCompressor\n"
    "from vllm.utils.multi_stream_utils import ("
)
H1_NEW = (
    "from vllm.models.deepseek_v4.compressor import DeepseekCompressor\n"
    "from vllm.triton_utils import tl, triton\n"
    "from vllm.utils.multi_stream_utils import ("
)

# ---------------------------------------------------------------------------
# Hunk 2: Triton kernel after logger init
# ---------------------------------------------------------------------------
H2_OLD = (
    "logger = init_logger(__name__)\n"
    "\n"
    "\n"
    "def _resolve_dsv4_kv_cache_dtype("
)
H2_NEW = (
    "logger = init_logger(__name__)\n"
    "\n"
    "\n"
    "@triton.jit\n"
    "def _fill_short_context_topk_indices(\n"
    "    output,\n"
    "    positions,\n"
    "    TOP_K: tl.constexpr,\n"
    "    COMPRESS_RATIO: tl.constexpr,\n"
    "    PADDED_TOP_K: tl.constexpr,\n"
    "):\n"
    "    # small triton kernel that selects every candidate, -1 otherwise\n"
    "    row = tl.program_id(0)\n"
    "    offsets = tl.arange(0, PADDED_TOP_K)\n"
    "    num_compressed = (tl.load(positions + row) + 1) // COMPRESS_RATIO\n"
    "    tl.store(\n"
    "        output + row * TOP_K + offsets,\n"
    "        tl.where(offsets < num_compressed, offsets, -1),\n"
    "        mask=offsets < TOP_K,\n"
    "    )\n"
    "\n"
    "\n"
    "def _resolve_dsv4_kv_cache_dtype("
)

# ---------------------------------------------------------------------------
# Hunk 3: short-context early return before wq_b_and_q_quant
# ---------------------------------------------------------------------------
H3_OLD = (
    "        compressor = self.compressor\n"
    "\n"
    "        def wq_b_and_q_quant():"
)
H3_NEW = (
    "        compressor = self.compressor\n"
    "\n"
    "        attn_metadata = get_forward_context().attn_metadata\n"
    "        if isinstance(attn_metadata, dict):\n"
    "            indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])\n"
    "            if indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens:\n"
    "                # candidates num smaller than topk, every candidate is selected\n"
    "                # but we still need to build k cache\n"
    "                compressor(compressed_kv_score, positions, rotary_emb)\n"
    "                assert self.topk_indices_buffer is not None\n"
    "                num_tokens = (\n"
    "                    indexer_metadata.num_decode_tokens\n"
    "                    + indexer_metadata.num_prefill_tokens\n"
    "                )\n"
    "                if num_tokens > 0:\n"
    "                    _fill_short_context_topk_indices[(num_tokens,)](\n"
    "                        self.topk_indices_buffer,\n"
    "                        positions,\n"
    "                        TOP_K=self.topk_tokens,\n"
    "                        COMPRESS_RATIO=self.compress_ratio,\n"
    "                        PADDED_TOP_K=triton.next_power_of_2(self.topk_tokens),\n"
    "                        num_warps=8,\n"
    "                    )\n"
    "                return self.topk_indices_buffer\n"
    "\n"
    "        def wq_b_and_q_quant():"
)

HUNKS = [
    ("hunk1-import", H1_OLD, H1_NEW),
    ("hunk2-kernel", H2_OLD, H2_NEW),
    ("hunk3-shortctx", H3_OLD, H3_NEW),
]

with open(ROUTER) as f:
    content = f.read()

results = []
fatal = False

for name, old, new in HUNKS:
    if new in content:
        results.append(f"OK    {name}: already applied")
    elif content.count(old) != 1:
        results.append(
            f"FAIL  {name}: expected 1 anchor, found {content.count(old)}"
        )
        fatal = True
    else:
        content = content.replace(old, new, 1)
        results.append(f"OK    {name}: applied")

if not fatal:
    with open(ROUTER, "w") as f:
        f.write(content)

# Fail-closed: only compile if every hunk succeeded.
if not fatal:
    import py_compile  # noqa: E402

    py_compile.compile(str(ROUTER), doraise=True)
    results.append("OK    compile: py_compile passed")

for r in results:
    print(r)

raise SystemExit(1 if fatal else 0)
