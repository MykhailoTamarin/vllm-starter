import os
import re

init_path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/__init__.py"

with open(init_path) as f:
    content = f.read()

# Add exl3 to QuantizationMethods literal
if '"exl3"' not in content:
    content = content.replace(
        '"deepseek_v4_fp8",',
        '"exl3",\n    "deepseek_v4_fp8",'
    )

# Add exl3 import
if "from .exl3 import Exl3Config" not in content:
    # Find the last .auto_gptq import and add after it
    content = content.replace(
        "from .auto_gptq import AutoGPTQConfig",
        "from .auto_gptq import AutoGPTQConfig\n    from .exl3 import Exl3Config"
    )

# Add exl3 config mapping  
if '"exl3": Exl3Config,' not in content:
    content = content.replace(
        '"deepseek_v4_fp8": DeepseekV4FP8Config,',
        '"exl3": Exl3Config,\n        "deepseek_v4_fp8": DeepseekV4FP8Config,'
    )

with open(init_path, 'w') as f:
    f.write(content)

print("Patched quantization/__init__.py with exl3 support")
