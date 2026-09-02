"""#228 backport: keep graphed tiny-decode routes with inactive expert ids in range.

Upstream local-inference-lab/b12x#228. During CUDA-graph capture vLLM pads
decode batches with routes whose expert id sits outside [0, weight_E) (graph
padding rows and non-local EP routes). The tiny-decode FC1/FC2 kernels indexed
weights with ``eid = Int32(topk_ids[rt_idx])`` unconditionally, so a padded/inactive
id could dereference OOB weight/scales/scale-rows during graph replay.

Clamp: compute ``route_active`` (id in [0, weight_E)), read all weights at
``eid=0`` (in range) and only store the result when the route is active --
inactive routes retain the pre-zeroed intermediate and output buffers that the
wrapper guarantees.

Applied to the sparkinfer package installed in the image
(``sparkinfer/moe/_shared/kernels/tiny_decode.py``).
"""

from pathlib import Path

TINY_DECODE = Path(
    "/usr/local/lib/python3.12/dist-packages/sparkinfer/moe/_shared/kernels/tiny_decode.py"
)

content = TINY_DECODE.read_text()
results = []

anchor_eid = "            eid = Int64(Int32(topk_ids[rt_idx]))"

replacement_eid = """            route_eid = Int32(topk_ids[rt_idx])
            route_active = route_eid >= Int32(0) and route_eid < Int32(c["weight_E"])
            # vLLM represents CUDA-graph padding and non-local EP routes with
            # an expert id outside [0, weight_E). Keep every address in range;
            # inactive routes retain the pre-zeroed intermediate and output.
            eid = Int64(0)
            if route_active:
                eid = Int64(route_eid)"""

anchor_guard = "            if cgrp == Int32(0):"
replacement_guard = "            if route_active and cgrp == Int32(0):"

if "route_active" in content:
    results.append("SKIP  tiny_decode: route_active guard already applied")
else:
    n_eid = content.count(anchor_eid)
    n_guard = content.count(anchor_guard)
    if n_eid != 2:
        print(f"FAIL  tiny_decode: expected 2 'eid = Int64(Int32(topk_ids...))' anchors, got {n_eid}")
        raise SystemExit(1)
    if n_guard != 2:
        print(f"FAIL  tiny_decode: expected 2 'if cgrp == Int32(0):' anchors, got {n_guard}")
        raise SystemExit(1)
    content = content.replace(anchor_eid, replacement_eid)
    content = content.replace(anchor_guard, replacement_guard)
    TINY_DECODE.write_text(content)
    results.append("OK    tiny_decode: inactive-expert-id routes clamped in range")

import py_compile  # noqa: E402

py_compile.compile(str(TINY_DECODE), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)