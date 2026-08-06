"""Compact DSPark draft support: build draft DecoderLayers with the draft's
routed-expert count instead of the target's.

Port of the upstream compact-draft fix (0xSero spark-sparkinfer recipe,
`vllm-dspark-compact-draft.patch` hunk 1) onto stock v0.26.0.

``DeepseekV4DecoderLayer`` reads ``n_routed_experts`` from
``vllm_config.model_config.hf_config`` -- the *target* config.  A compact draft
(e.g. REAP-sliced K64) carries a different expert count in its own config.json,
so without an override the draft layers are built with the target's 216 experts
and the 64-expert draft weights cannot load.  Temporarily override the target
config's expert count for the duration of draft construction, then restore it.

When no separate draft model is configured the draft config is a copy of the
target config (n_routed_experts == 216) and the override is a no-op, so the
embedded-draft flow is unaffected.
"""

from pathlib import Path

DSPARK = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py"
)

content = DSPARK.read_text()
results = []

anchor = """        current_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                )
                for i in range(self.num_dspark_layers)
            ]
        )"""

replacement = """        # Compact DSPark draft: the draft model may carry fewer routed experts
        # than the target (e.g. a REAP-sliced K64 draft). DeepseekV4DecoderLayer
        # reads n_routed_experts from the *target* ModelConfig, so temporarily
        # override it for the duration of construction, then restore. Each
        # instantiated MoE owns its resulting dimensions.
        current_vllm_config = get_current_vllm_config()
        target_hf_config = current_vllm_config.model_config.hf_config
        target_n_routed_experts = target_hf_config.n_routed_experts
        target_hf_config.n_routed_experts = config.n_routed_experts
        try:
            self.layers = nn.ModuleList(
                [
                    DeepseekV4DecoderLayer(
                        current_vllm_config,
                        prefix=maybe_prefix(
                            prefix, f"layers.{self.num_hidden_layers + i}"
                        ),
                    )
                    for i in range(self.num_dspark_layers)
                ]
            )
        finally:
            target_hf_config.n_routed_experts = target_n_routed_experts"""

if "target_n_routed_experts" in content:
    results.append("SKIP  dspark: n_routed_experts override already applied")
elif anchor not in content:
    print("FAIL  dspark: anchor not found")
    raise SystemExit(1)
else:
    n = content.count(anchor)
    if n > 1:
        print(f"FAIL  dspark: {n} ambiguous matches")
        raise SystemExit(1)
    content = content.replace(anchor, replacement, 1)
    DSPARK.write_text(content)
    results.append("OK    dspark: draft DecoderLayers built with draft expert count")

import py_compile  # noqa: E402

py_compile.compile(str(DSPARK), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)
