"""SM12x DSpark fix: pad the non-causal SWA index width to a topk that
FlashInfer's SM120/SM121 sparse-MLA decode kernel actually instantiates.

The DSpark draft block runs non-causal decode over
``cdiv(window_size + num_speculative_tokens, 128) * 128`` indices per token
(128 + 5 -> 256 for DeepSeek-V4-Flash).  FlashInfer's
``sparse_mla_sm120_decode_dsv4`` kernel is only instantiated for topk in
{128, 512, 1024}, so a 256-wide draft decode fails the dispatch check in
``flashinfer/mla/_sparse_mla_sm120.py::_paged_attention`` and falls through
to the prefill orchestrator ``sparse_mla_sm120_paged_attention``, which
asserts ``num_tokens > 64`` and aborts the 5-token draft decode:

    tvm.error.InternalError: Check failed: num_tokens > 64 (5 vs. 64) :
    Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2
    or sparse_mla_sm120_decode_dsv4; got num_tokens=5

Pad the width up to the next dispatchable topk (256 -> 512).  The Triton
index kernel already writes -1 into slots beyond the effective length and
the FlashInfer kernel skips -1 slots, so the extra capacity is inert.
"""

from pathlib import Path

SPARSE_SWA = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py"
)

content = SPARSE_SWA.read_text()
results = []

anchor = """        self.is_dspark = spec_config is not None and spec_config.use_dspark()
        self.noncausal_index_width = (
            cdiv(self.window_size + self.num_speculative_tokens, 128) * 128
            if self.is_dspark
            else 0
        )"""

replacement = """        self.is_dspark = spec_config is not None and spec_config.use_dspark()
        # SM12x fix: FlashInfer's sparse_mla_sm120_decode_dsv4 kernel is only
        # instantiated for topk in {128, 512, 1024}.  The raw DSpark width
        # (window 128 + k spec tokens -> 256) would miss the decode dispatch
        # and fall through to the prefill orchestrator, which requires
        # num_tokens > 64 and crashes the small draft decode.  Pad up to the
        # next dispatchable topk; slots beyond the effective length stay -1
        # and are skipped by the kernel.
        _raw_noncausal_width = (
            cdiv(self.window_size + self.num_speculative_tokens, 128) * 128
        )
        _fi_sm12x_dsv4_topks = (128, 512, 1024)
        self.noncausal_index_width = (
            next(
                w for w in _fi_sm12x_dsv4_topks if w >= _raw_noncausal_width
            )
            if self.is_dspark
            else 0
        )"""

if anchor not in content:
    if "_fi_sm12x_dsv4_topks" in content:
        results.append("SKIP  sparse_swa: already applied")
    else:
        print("FAIL  sparse_swa: anchor not found")
        raise SystemExit(1)
else:
    content = content.replace(anchor, replacement, 1)
    SPARSE_SWA.write_text(content)
    results.append("OK    sparse_swa: noncausal index width padded for SM12x")

import py_compile  # noqa: E402

py_compile.compile(str(SPARSE_SWA), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)
