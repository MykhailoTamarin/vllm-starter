import os

model_path = "/usr/local/lib/python3.12/dist-packages/vllm/config/model.py"

with open(model_path) as f:
    content = f.read()

# Add exl3 to overrides list
old = '"deepseek_v4_fp8",'
new = '"exl3",\n                "deepseek_v4_fp8",'
if old in content and '"exl3"' not in content:
    content = content.replace(old, new)
    with open(model_path, 'w') as f:
        f.write(content)
    print("Added exl3 to overrides list in model.py")
else:
    print("exl3 already in overrides or pattern not found")

# Also add exl3 to QuantizationConfigArgs quantization field if it exists
old2 = '"quantization": "exl3",'
if old2 not in content:
    # Find the exl3 field and check if there are references
    pass

with open(model_path) as f:
    for i, line in enumerate(f, 1):
        if 'exl3' in line.lower():
            print(f"  Line {i}: {line.rstrip()}")
