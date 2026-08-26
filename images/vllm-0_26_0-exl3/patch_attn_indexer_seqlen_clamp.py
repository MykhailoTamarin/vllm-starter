"""Backport vLLM PR #51538 onto v0.26.0: indexer padded-request seq_len clamp.

Upstream: commit 97388c44f9 ("[Bugfix] Make DSV4 sparse MLA work end-to-end for
          plain decode, MTP, and DSpark (#51538)") — indexer.py sub-hunks only.

What it fixes
-------------
Padding rows in the uniform-decode indexer have seq_len == 0. Without a clamp,
the first token of each padded request computes a NEGATIVE per-token seq len
(e.g. next_n=2: 0 - 2 + 0 + 1 = -1). Downstream kernels read these values as
uint32, turning -1 into ~4e9 and corrupting the decode path.

EXL3 relevance: the stack runs padded multi-seq decode batches
(--max-num-seqs 2, full CUDA graph with fixed capture sizes 1..16), so this
clamp is on the exact hot path a padded batch takes.

Scope ported
------------
Two hunks of the upstream commit:
  1. Triton kernel ``_prepare_uniform_decode_kernel``: wrap the per-token
     seq-len computation in ``tl.maximum(..., 0)``.
  2. Host-side ``seq_lens_buffer[:] = (...)`` in the native next_n>1 path:
     add ``.clamp_(min=0)``.

Faithful vs adapted
-------------------
FAITHFUL. The v0.26.0 text for both hunks matches the upstream OLD side
exactly (verified against ``git show v0.26.0:.../indexer.py``); upstream NEW
text is copied verbatim including comments.

Container target
----------------
/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py
"""

import os
from pathlib import Path
import py_compile  # noqa: E402

TARGET = Path(
    os.environ.get(
        "VLLM_INDEXER",
        "/usr/local/lib/python3.12/dist-packages/vllm/"
        "v1/attention/backends/mla/indexer.py",
    )
)

# ---------------------------------------------------------------------------
# Hunk 1: Triton kernel — clamp per-token seq len at 0
# ---------------------------------------------------------------------------
H1_OLD = """    # Compute number of KVs attended to by this token.
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = seq_len - max_decode_len + local_idx + 1"""

H1_NEW = """    # Compute number of KVs attended to by this token. Padding requests have
    # seq_len == 0, which would otherwise make the first token of each padded
    # request negative (e.g. next_n=2 gives 0-2+0+1 = -1). Downstream kernels
    # read these as uint32, turning -1 into ~4e9.
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = tl.maximum(seq_len - max_decode_len + local_idx + 1, 0)"""

# ---------------------------------------------------------------------------
# Hunk 2: host-side — clamp the seq_lens_buffer assignment at 0
# ---------------------------------------------------------------------------
H2_OLD = """                seq_lens_buffer[:] = (
                    seq_lens.unsqueeze(1)
                    - max_decode_len
                    + 1
                    + self.offsets_buffer[:max_decode_len]
                )"""

H2_NEW = """                # Clamp at 0: padding requests have seq_len == 0, which would
                # otherwise make token 0 negative (next_n=2 gives 0-2+1+0 = -1).
                # Downstream kernels read these as uint32, turning -1 into ~4e9.
                seq_lens_buffer[:] = (
                    seq_lens.unsqueeze(1)
                    - max_decode_len
                    + 1
                    + self.offsets_buffer[:max_decode_len]
                ).clamp_(min=0)"""

content = TARGET.read_text()
results = []

for hunk, (old, new) in enumerate([(H1_OLD, H1_NEW), (H2_OLD, H2_NEW)], 1):
    if new in content:
        results.append(f"SKIP  hunk {hunk}: already applied")
        continue
    n = content.count(old)
    if n != 1:
        print(f"FAIL  hunk {hunk}: expected 1 anchor, found {n}")
        raise SystemExit(1)
    content = content.replace(old, new, 1)
    results.append(f"OK    hunk {hunk}")

TARGET.write_text(content)
py_compile.compile(str(TARGET), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)