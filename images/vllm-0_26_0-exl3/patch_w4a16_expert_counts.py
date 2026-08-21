"""#150 backport: preallocate the W4A16 route histogram for CUDA-graph capture.

Upstream local-inference-lab/b12x#150. ``pack_topk_routes_by_expert`` computes a
per-expert route histogram (``expert_counts``) before the prefix sum that lays
out each expert's GEMM blocks. When the buffer is not provided and the stream is
capturing, route_pack.py falls back to a slow single-CTA count+prefix kernel so
it can avoid allocating during capture -- but the fast parallel-count kernel is
the one we want on the hot path, and it needs the histogram preallocated.

The W4A16 workspace already allocates every other route buffer (packed indices,
block ids, count, offsets). Add ``expert_counts`` to that workspace, thread it
into the binding, and pass it to ``run_w4a16_moe`` so the fast histogram path is
always taken -- including during CUDA-graph capture -- without any
capture-time allocation.

Applied to the sparkinfer package installed in the image
(``sparkinfer/moe/fused_moe/_impl.py``).
"""

from pathlib import Path

IMPL = Path(
    "/usr/local/lib/python3.12/dist-packages/sparkinfer/moe/fused_moe/_impl.py"
)

content = IMPL.read_text()
results = []


def apply(anchor: str, replacement: str, label: str) -> None:
    global content
    if anchor not in content:
        print(f"FAIL  {label}: anchor not found")
        raise SystemExit(1)
    n = content.count(anchor)
    if n > 1:
        print(f"FAIL  {label}: {n} ambiguous matches")
        raise SystemExit(1)
    content = content.replace(anchor, replacement, 1)
    results.append(f"OK    {label}")


if "expert_counts" in content:
    results.append("SKIP  w4a16: expert_counts preallocation already applied")
else:

    # --- 1. TPW4A16Workspace: add expert_counts field -----------------------
    apply(
        """    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    planned_token_counts: frozenset[int] = field(default_factory=frozenset)""",
        """    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor
    planned_token_counts: frozenset[int] = field(default_factory=frozenset)""",
        "w4a16 workspace: expert_counts field",
    )

    # --- 2. TPMoEFP4Binding: add expert_counts field ------------------------
    apply(
        """    packed_route_count: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None
    fused_launch: object | None = None""",
        """    packed_route_count: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None
    expert_counts: torch.Tensor | None = None
    fused_launch: object | None = None""",
        "binding: expert_counts field",
    )

    # --- 3. W4A16 arena tensor specs: allocate expert_counts ----------------
    apply(
        """                _TensorAllocSpec("packed_route_count", (1,), torch.int32),
                _TensorAllocSpec("expert_offsets", (int(weight_E) + 1,), torch.int32),
            ),""",
        """                _TensorAllocSpec("packed_route_count", (1,), torch.int32),
                _TensorAllocSpec("expert_offsets", (int(weight_E) + 1,), torch.int32),
                _TensorAllocSpec("expert_counts", (int(weight_E),), torch.int32),
            ),""",
        "arena specs: expert_counts alloc",
    )

    # --- 4. build_tp_moe_fp4_binding (arena): thread expert_counts ----------
    apply(
        """            packed_route_indices=tensors["packed_route_indices"],
            block_expert_ids=tensors["block_expert_ids"],
            packed_route_count=tensors["packed_route_count"],
            expert_offsets=tensors["expert_offsets"],
        )""",
        """            packed_route_indices=tensors["packed_route_indices"],
            block_expert_ids=tensors["block_expert_ids"],
            packed_route_count=tensors["packed_route_count"],
            expert_offsets=tensors["expert_offsets"],
            expert_counts=tensors["expert_counts"],
        )""",
        "binding build: expert_counts from arena",
    )

    # --- 5. TPW4A16Workspace materialization: thread expert_counts ----------
    apply(
        """            packed_route_count=tensors["packed_route_count"],
            expert_offsets=tensors["expert_offsets"],
            volatile_launch_state=bool(volatile_launch_state),""",
        """            packed_route_count=tensors["packed_route_count"],
            expert_offsets=tensors["expert_offsets"],
            expert_counts=tensors["expert_counts"],
            volatile_launch_state=bool(volatile_launch_state),""",
        "workspace materialize: expert_counts",
    )

    # --- 6. Workspace -> binding (non-arena path): thread expert_counts -----
    apply(
        """            packed_route_indices=workspace.packed_route_indices,
            block_expert_ids=workspace.block_expert_ids,
            packed_route_count=workspace.packed_route_count,
            expert_offsets=workspace.expert_offsets,
            fused_launch=fused_launch,""",
        """            packed_route_indices=workspace.packed_route_indices,
            block_expert_ids=workspace.block_expert_ids,
            packed_route_count=workspace.packed_route_count,
            expert_offsets=workspace.expert_offsets,
            expert_counts=workspace.expert_counts,
            fused_launch=fused_launch,""",
        "workspace->binding: expert_counts",
    )

    # --- 7. run_w4a16_moe: pass expert_counts from binding ------------------
    apply(
        """        packed_route_count = _require_binding_field(binding, "packed_route_count")
        expert_offsets = _require_binding_field(binding, "expert_offsets")
        fused_launch = binding.fused_launch""",
        """        packed_route_count = _require_binding_field(binding, "packed_route_count")
        expert_offsets = _require_binding_field(binding, "expert_offsets")
        expert_counts = _require_binding_field(binding, "expert_counts")
        fused_launch = binding.fused_launch""",
        "run path: require expert_counts",
    )

    apply(
        """            packed_route_indices=packed_route_indices,
            block_expert_ids=block_expert_ids,
            packed_route_count=packed_route_count,
            expert_offsets=expert_offsets,
            activation_amax=activation_amax,""",
        """            packed_route_indices=packed_route_indices,
            block_expert_ids=block_expert_ids,
            packed_route_count=packed_route_count,
            expert_offsets=expert_offsets,
            expert_counts=expert_counts,
            activation_amax=activation_amax,""",
        "run path: pass expert_counts to run_w4a16_moe",
    )

    IMPL.write_text(content)

import py_compile  # noqa: E402

py_compile.compile(str(IMPL), doraise=True)
results.append("OK    compile")

for r in results:
    print(r)