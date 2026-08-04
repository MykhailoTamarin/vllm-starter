"""Add pure-Torch fallback for REAP K216 (216 experts) router.

v0.26.0's `vllm_topk_softplus_sqrt` CUDA op only supports expert counts in
{16,32,64,128,192,256,320,384,512}. The REAP K216 checkpoint uses 216 experts
per MoE scope, so the CUDA launcher raises:

  RuntimeError: topkGatingSoftplusSqrtKernelLauncher, ... Unsupported expert
  number: 216

Add a pure-PyTorch fallback (already present as `_topk_softplus_sqrt_torch`)
for non-supported sqrtsoftplus expert counts. Mirrors the runtime patch used
by the NVFP4 DSPark recipe (patch_vllm_k160_dspark.py), but baked into the
image so no volume-mount patch step is needed.
"""

from pathlib import Path

ROUTER = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/router/fused_topk_bias_router.py"
)

SUPPORTED = "(16, 32, 64, 128, 192, 256, 320, 384, 512)"

with open(ROUTER) as f:
    content = f.read()

results = []


def patch(old: str, new: str, label: str) -> None:
    if old not in content:
        results.append(f"FAIL  {label}: pattern not found")
        return
    n = content.count(old)
    if n > 1:
        results.append(f"FAIL  {label}: {n} ambiguous matches")
        return
    results.append(f"OK    {label}")
    globals()["content"] = content.replace(old, new, 1)


# 1. The main `not rocm_aiter_ops.is_fused_moe_enabled()` gate: also take the
#    pure-Torch path for sqrtsoftplus with a non-standard expert count.
patch(
    """    if not rocm_aiter_ops.is_fused_moe_enabled():""",
    """    if not rocm_aiter_ops.is_fused_moe_enabled() or (
        scoring_func == "sqrtsoftplus"
        and gating_output.shape[-1] not in %s
    ):""" % SUPPORTED,
    "reap-router:gate",
)

# 2. The first (in-gate) sqrtsoftplus call: fall back to Torch for
#    non-supported counts.
patch(
    """        elif scoring_func == "sqrtsoftplus":
            return vllm_topk_softplus_sqrt(""",
    """        elif scoring_func == "sqrtsoftplus":
            if gating_output.shape[-1] not in %s:
                return _topk_softplus_sqrt_torch(
                    topk_weights,
                    topk_ids,
                    token_expert_indices,
                    gating_output,
                    renormalize,
                    e_score_correction_bias,
                    input_tokens,
                    hash_indices_table,
                    routed_scaling_factor,
                )
            return vllm_topk_softplus_sqrt(""" % SUPPORTED,
    "reap-router:sqrt-in-gate",
)

# 3. The fall-through sqrtsoftplus call (after the rocm gate).
patch(
    """    if scoring_func == "sqrtsoftplus":
        M = hidden_states.size(0)
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32 if indices_type is None else indices_type,
            device=hidden_states.device,
        )
        token_expert_indices = torch.empty(
            M, topk, dtype=torch.int32, device=hidden_states.device
        )
        return vllm_topk_softplus_sqrt(""",
    """    if scoring_func == "sqrtsoftplus":
        M = hidden_states.size(0)
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32 if indices_type is None else indices_type,
            device=hidden_states.device,
        )
        token_expert_indices = torch.empty(
            M, topk, dtype=torch.int32, device=hidden_states.device
        )
        if gating_output.shape[-1] not in %s:
            return _topk_softplus_sqrt_torch(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
                input_tokens,
                hash_indices_table,
                routed_scaling_factor,
            )
        return vllm_topk_softplus_sqrt(""" % SUPPORTED,
    "reap-router:sqrt-fallthrough",
)

with open(ROUTER, "w") as f:
    f.write(content)

import py_compile  # noqa: E402

py_compile.compile(str(ROUTER), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)
