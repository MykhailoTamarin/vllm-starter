"""Backport vLLM PR #52288 onto v0.26.0: DSPark draft inherits target backend.

Upstream: commit acb0f1dcdb ("[Bugfix][Spec Decode] DSpark: inherit the
          target's attention backend when the speculative config names none
          (#52288)")

What it fixes
-------------
load_dspark_model passed ``backend=speculative_config.attention_backend`` to
the draft's attention config. When the speculative config omits
``attention_backend`` (None), the draft re-ran backend auto-selection and could
pick a DIFFERENT attention class than the target, silently diverging on the
draft's sparse-MLA path.

EXL3 relevance: models/deepseek-v4-flash-0731-exl3-dspark.yaml pins the target
to --attention-backend FLASHINFER_MLA but the --speculative-config JSON omits
``attention_backend``. In v0.26.0 that None re-runs auto-selection for the
K160 compact draft — exactly the divergence this fix closes.

Scope ported
------------
Single hunk of the upstream commit, verbatim.

Faithful vs adapted
-------------------
FAITHFUL. v0.26.0 text matches the upstream OLD side exactly (verified against
``git show v0.26.0:.../dspark/utils.py``); upstream NEW text copied verbatim.

Container target
----------------
/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/utils.py
"""

import os
from pathlib import Path
import py_compile  # noqa: E402

TARGET = Path(
    os.environ.get(
        "VLLM_DSPARK_UTILS",
        "/usr/local/lib/python3.12/dist-packages/vllm/"
        "v1/worker/gpu/spec_decode/dspark/utils.py",
    )
)

H1_OLD = """    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),"""

H1_NEW = """    # None re-runs backend auto-selection for the draft, which can pick a
    # different attention class than the target; fall back to the target's.
    draft_attention_backend = (
        speculative_config.attention_backend or vllm_config.attention_config.backend
    )

    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),"""

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