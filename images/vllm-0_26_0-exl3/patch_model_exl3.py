"""Apply EXL3-specific deltas onto v0.26.0 native DeepseekV4 model.py.

Keeps the native DSPark/EPLB/Eagle3/TileLang attention paths intact and
only adds the EXL3 rank-sliced checkpoint handling:

1. Mapper: strip `_rankN` segments + map `.attn.compressor.` -> mla_attn.
2. Class: packed_modules_mapping for merged linears.
3. load_weights: normalize EXL3 expert `_rankN.<sub>` names before the
   expert weight loader sees them.
4. skip_weight_name_before_load: drop mtp.* from the target model so the
   embedded DSPark draft weights load into the draft model instead.
"""

import re
from pathlib import Path

MODEL = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py"
)

with open(MODEL) as f:
    content = f.read()

results = []


def patch(old: str, new: str, label: str) -> None:
    if old not in content:
        results.append(f"FAIL  {label}: old text not found")
        return
    n = content.count(old)
    if n > 1:
        results.append(f"FAIL  {label}: {n} ambiguous matches")
        return
    results.append(f"OK    {label}")
    globals()["content"] = content.replace(old, new, 1)


# 1a. Mapper fp4 branch: add _rankN strip to the regex dict. The fp4 branch
# already carries `w[123].scale` -> weight_scale; EXL3 checkpoint names are
# `w1.rankN.trellis`/`w2.rankN.suh` etc, so strip the `_rankN.` segment so the
# per-expert scale/weight regexes match.
patch(
    """        scale_regex = {
            re.compile(r"(\\.experts\\.\\d+\\.w[123])\\.scale$"): r"\\1.weight_scale",
            re.compile(r"\\.scale$"): ".weight_scale_inv",
        }""",
    """        scale_regex = {
            re.compile(r"(\\.experts\\.\\d+\\.w[123])\\.scale$"): r"\\1.weight_scale",
            re.compile(r"_rank\\d+\\."): ".",
            re.compile(r"\\.scale$"): ".weight_scale_inv",
        }""",
    "mapper:rank-strip",
)


# 1b. Substr: (removed for v0.26.0) the old 0xsero base named the compressor
# under `mla_attn.compressor`; v0.26.0 native names it `attn.compressor`
# directly, matching the checkpoint layout. No substr needed.


# 2. packed_modules_mapping on DeepseekV4ForCausalLM.
patch(
    """    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")
""",
    """    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }
""",
    "class:packed-modules-mapping",
)


# 3. Expert weight name normalization in DeepseekV4Model.load_weights.
# v0.26.0 maps `w1`->gate_up_proj; EXL3 names carry `_rankN.` segments
# (e.g. `w1.rank0.mcg`, `w1.rank0.trellis`). Normalize so the Exl3MoEParameter
# weight loader receives the base name.
patch(
    """                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue""",
    """                        name_mapped = name.replace(weight_name, param_name)
                        name_mapped = re.sub(
                            r"_rank\\d+\\.(mcg|trellis|suh|svh|mul1)$",
                            r"_\\1", name_mapped,
                        )
                        if is_pp_missing_parameter(name_mapped, self):
                            continue""",
    "load:expert-rank-normalize",
)


# 4. skip_weight_name_before_load on DeepseekV4ForCausalLM so the target model
# leaves mtp.* weights for the DSPark draft model.
patch(
    """    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        \"\"\"Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step.\"\"\"
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()
        return loaded_params""",
    """    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        \"\"\"Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step.\"\"\"
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def skip_weight_name_before_load(self, name: str) -> bool:
        mapped = self.hf_to_vllm_mapper._map_name(name)
        return mapped is None or "mtp." in mapped

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()
        return loaded_params""",
    "class:skip-weight-name-before-load",
)

with open(MODEL, "w") as f:
    f.write(content)

import py_compile  # noqa: E402

py_compile.compile(str(MODEL), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)
