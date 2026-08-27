"""Backport vLLM PR #51733 onto v0.26.0: sparse-MLA prefill workspace sizing.

Upstream: commit 608c12473f ("[Attention] Fix MLA prefill workspace allocation
          size (#51733)")

What it fixes
-------------
``SparseMLACommonMetadataBuilder.determine_chunked_prefill_workspace_size``
over-allocated the prefill workspace: it forced the size up to
``max_num_seqs * cache_config.block_size``, but a single workspace row already
covers one block. The fix caps the floor at ``cache_config.block_size``,
saving GPU memory on the sparse-MLA prefill path.

EXL3 relevance: EXL3 keeps native DeepSeek-V4 sparse-MLA attention with
chunked prefill (--enable-chunked-prefill) and FP8 KV; the workspace lives in
the same GPU memory pool as the 512K-context KV cache at
--gpu-memory-utilization 0.9, so every MiB saved is context headroom.

Note on the upstream companion hunk (``mla_attention.py``,
``align_mla_chunked_context_workspace_size``): that function does NOT exist in
v0.26.0 (it was introduced later), so only the ``sparse_mla_attention.py``
hunk is ported here.

Scope ported
------------
Single hunk of the upstream commit (sparse_mla_attention.py), verbatim.

Faithful vs adapted
-------------------
FAITHFUL. v0.26.0 text matches the upstream OLD side exactly (verified against
``git show v0.26.0:.../sparse_mla_attention.py``); upstream NEW text copied
verbatim (the v0.26.0 form is a ``return max(...)``, so the one-line change is
identical to upstream's post-branch shape).

Container target
----------------
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/sparse_mla_attention.py
"""

import os
from pathlib import Path
import py_compile  # noqa: E402

TARGET = Path(
    os.environ.get(
        "VLLM_SPARSE_MLA_ATTN",
        "/usr/local/lib/python3.12/dist-packages/vllm/"
        "model_executor/layers/attention/sparse_mla_attention.py",
    )
)

H1_OLD = """        return max(
            workspace_size,
            scheduler_config.max_num_seqs * cache_config.block_size,
        )"""

H1_NEW = """        return max(workspace_size, cache_config.block_size)"""

content = TARGET.read_text()
results = []

if H1_NEW in content:
    results.append("SKIP  hunk: already applied")
else:
    n = content.count(H1_OLD)
    if n != 1:
        print(f"FAIL  hunk: expected 1 anchor, found {n}")
        raise SystemExit(1)
    content = content.replace(H1_OLD, H1_NEW, 1)
    results.append("OK    hunk")

TARGET.write_text(content)
py_compile.compile(str(TARGET), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)