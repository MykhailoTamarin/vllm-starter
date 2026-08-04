"""Stamp the vLLM version with a dev postfix.

Identifies this image as a custom build (EXL3 backend + DSPark on SM121)
on top of stock v0.26.0 so the startup banner and vllm.__version__ show a
non-release version.
"""

from pathlib import Path

VER = Path("/usr/local/lib/python3.12/dist-packages/vllm/_version.py")

POSTFIX = "0.26.0-exl3.dspark.sm121"

with open(VER) as f:
    content = f.read()

if POSTFIX in content:
    print("OK    version: already stamped")
elif "__version__ = version = '0.26.0'" in content:
    content = content.replace(
        "__version__ = version = '0.26.0'",
        f"__version__ = version = '{POSTFIX}'",
    )
    content = content.replace(
        "__version_tuple__ = version_tuple = (0, 26, 0)",
        f"__version_tuple__ = version_tuple = (0, 26, 0, '{POSTFIX}')",
    )
    with open(VER, "w") as f:
        f.write(content)
    print(f"OK    version: stamped {POSTFIX}")
else:
    print("FAIL  version: pattern not found")
    raise SystemExit(1)
