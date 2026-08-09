"""Enable the fused DSv4 Triton router kernel for REAP K216 (216 experts).

v0.26.0's `dsv4_topk` kernel is generic over expert count: it launches with
`BLOCK_N = next_power_of_2(num_experts)` (216 -> 256) and natively masks the
40 padded slots with -inf via `expert_mask`, so no tensor padding is needed.
The only blocker is the `can_use_dsv4_topk()` shape whitelist, which hardcodes
`(256, 384)` and rejects 216. Relaxing it lets the fused Triton kernel replace
the pure-Torch `_topk_softplus_sqrt_torch` fallback on the routing hot path
(~1 kernel per layer instead of ~5-7 torch launches).

The kernel is numerically equivalent to the torch fallback (stable softplus+sqrt
with threshold 20, biased selection, unbiased weight gather, renorm-by-sum,
routed_scaling_factor). Requires: topk == 6, renormalize=True, fp32 gating+bias
-- all enforced by the other can_use_dsv4_topk() conditions.
"""

from pathlib import Path

ROUTER = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/router/dsv4_topk.py"
)

OLD = "        and gating_output.shape[1] in (256, 384)"
NEW = "        and gating_output.shape[1] in (216, 256, 384)"

with open(ROUTER) as f:
    content = f.read()

results = []

if NEW in content:
    results.append("OK    dsv4-topk: gate already relaxed")
elif content.count(OLD) != 1:
    results.append(f"FAIL  dsv4-topk: expected 1 anchor, found {content.count(OLD)}")
else:
    content = content.replace(OLD, NEW, 1)
    with open(ROUTER, "w") as f:
        f.write(content)
    results.append("OK    dsv4-topk: gate relaxed (216, 256, 384)")

import py_compile  # noqa: E402

py_compile.compile(str(ROUTER), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)
