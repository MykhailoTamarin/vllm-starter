"""Fix v0.26.0's instanttensor_weights_iterator for instanttensor==0.1.5.

1. Stock v0.26.0 passes copy=True to instanttensor.safe_open(), but
   instanttensor 0.1.5 does not accept it (added in a later release). Drop the
   kwarg; instanttensor.patch already clones every tensor in get_tensor(), so
   buffer ownership is handled (matches the validated 0xSero/LIL recipe).

2. InstantTensor yields a .clone() of every tensor, and torch's caching
   allocator keeps those post-load clone blocks reserved (~5 GiB for this
   model). vLLM's KV-cache profiler then counts them as "non-torch memory"
   and shrinks the KV pool. Call torch.cuda.empty_cache() in the iterator's
   finally so the freed blocks return to the driver before profiling.
"""

from pathlib import Path

WT = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
    "model_loader/weight_utils.py"
)

content = WT.read_text()
results = []

# ── 1. Drop copy=True (and its stale comment) from the safe_open call ──
old_call = """    # copy=True yields tensors that own their memory, staying valid after the
    # context exits or InstantTensor reuses its buffer.
    with instanttensor.safe_open(
        hf_weights_files,
        framework="pt",
        device=device,
        process_group=process_group,
        copy=True,
    ) as f:"""

new_call = """    with instanttensor.safe_open(
        hf_weights_files, framework="pt", device=device, process_group=process_group
    ) as f:"""

if "copy=True" not in content:
    results.append("OK    loader: copy=True already removed")
elif content.count(old_call) != 1:
    print(f"FAIL  loader: expected 1 copy=True anchor, found {content.count(old_call)}")
    raise SystemExit(1)
else:
    content = content.replace(old_call, new_call, 1)
    results.append("OK    loader: removed copy=True from safe_open call")

# ── 2. Free cached clone buffers after iteration ─────────────────────────────
old_finally = """        try:
            for name, tensor in f.tensors():
                pbar.update(tensor.numel() * tensor.element_size())
                yield name, tensor
        finally:
            pbar.close()"""

new_finally = """        try:
            for name, tensor in f.tensors():
                pbar.update(tensor.numel() * tensor.element_size())
                yield name, tensor
        finally:
            pbar.close()
            # InstantTensor yields a clone of every tensor; torch's caching
            # allocator keeps those post-load clones reserved and vLLM's KV
            # profiler counts them as non-torch memory, shrinking the KV pool.
            # Free them to the CUDA driver before the KV profiler runs.
            torch.cuda.empty_cache()"""

if "torch.cuda.empty_cache()" in content:
    results.append("OK    loader: empty_cache already present")
elif content.count(old_finally) != 1:
    print(f"FAIL  loader: expected 1 finally anchor, found {content.count(old_finally)}")
    raise SystemExit(1)
else:
    content = content.replace(old_finally, new_finally, 1)
    results.append("OK    loader: added torch.cuda.empty_cache() to finally")

WT.write_text(content)

import py_compile  # noqa: E402

py_compile.compile(str(WT), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)
