"""Placeholder DSPark patch.

The embedded DSPark MTP draft experts in this checkpoint are stored in the
native DeepSeek V4 fp4 format (packed e2m1 I8 + ue8m0 block-32 scales).  The
exl3 overlay routes non-EXL3 MoE layers (the draft) to ``Mxfp4MoEMethod``,
which registers ``w13/w2_weight``/``weight_scale`` as uint8 params matching
that exact layout.  The stock ``dspark.py`` loader handles the ``.scale`` ->
``.weight_scale`` suffix and the int8->uint8 bit-preserving copy_(), so no
source patch is needed here.
"""

from pathlib import Path

DSPARK = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py"
)

content = DSPARK.read_text()
results = ["OK    noop"]


DSPARK.write_text(content)

import py_compile  # noqa: E402

py_compile.compile(str(DSPARK), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)
