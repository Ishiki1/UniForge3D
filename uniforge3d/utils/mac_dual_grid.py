"""Sparse flexible dual grid extraction for Mac inference.

Derived from trellis2-apple's ``o_voxel.convert.flexible_dual_grid`` at
17347247c91c36c8cdc1896234983e878a457bba, adapted from Microsoft TRELLIS.2.
Copyright (c) Microsoft Corporation. MIT; see THIRD_PARTY_LICENSES.txt.

The original dense int64 lookup requires 8 GiB at a 1024 cubed grid, in addition
to neighbor construction and output memory. This adapter replaces that lookup
with O(N) sparse storage and chunks active-edge neighbor queries. This avoids
grid-volume allocation; it does not enforce a total-memory or RSS limit.

The supported inference boundary is int32/int64 voxel coordinates, boolean
active-edge flags and train=False. Float coordinates and integer flags are
rejected explicitly. Decoder split weights have shape (N, 1); None retains the
original angle heuristic. Integer-coordinate duplicate/wrapping behavior,
active-edge order, quad winding, strict greater-than split decisions, vertex
values and transforms are preserved. In particular, the legacy cross-product
axis when exactly three quads survive and the original second-split arithmetic
are retained, without changing the geometry heuristic.

Topology runs explicitly on CPU through NumPy sorted-key lookups. Floating
vertex/split arithmetic and returned tensors use the original input device.
This is a named CPU topology boundary, not generic accelerator fallback.
Native MPS execution requires its own validation; CPU parity alone does not
establish native support, full decoder execution or production memory use.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch

__all__ = [
    "mac_flexible_dual_grid_to_mesh",
    "SparseVoxelLookup",
    "EDGE_NEIGHBOR_VOXEL_OFFSET",
    "QUAD_SPLIT_1",
    "QUAD_SPLIT_2",
    "DEFAULT_EDGE_CHUNK_SIZE",
]

# ---------------------------------------------------------------------------
# Production topology tables. Tests pin separate literal reference values.
# ---------------------------------------------------------------------------

# (3 axes, 4 corners, 3 xyz); same values as upstream ``edge_neighbor_voxel_offset``.
EDGE_NEIGHBOR_VOXEL_OFFSET = (
    (  # x-axis edge
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 1, 0),
    ),
    (  # y-axis edge
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
        (0, 0, 1),
    ),
    (  # z-axis edge
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 0),
    ),
)

# Same values as upstream ``quad_split_1`` / ``quad_split_2``.
QUAD_SPLIT_1 = (0, 1, 2, 0, 2, 3)
QUAD_SPLIT_2 = (0, 1, 3, 3, 1, 2)

# Limit on active edges per neighbor-query chunk. The corner block alone uses
# 96 bytes/edge; offset gathers, search keys/indices and masks add workspace.
# Input lookup/active-edge arrays and retained output quads remain O(N)/O(L),
# so this parameter is not a total-memory ceiling. Chunking preserves order.
DEFAULT_EDGE_CHUNK_SIZE = 2**18


# ---------------------------------------------------------------------------
# Bounded sparse coordinate lookup
# ---------------------------------------------------------------------------


class SparseVoxelLookup:
    """Sorted-key sparse replacement for the upstream dense voxel hashmap.

    Build memory is ``O(N)`` (keys ``int64`` + rows ``int32`` + one temporary
    ``argsort``) instead of the dense table's ``O(D*H*W)``. Lookup is
    ``O(M log N)`` for ``M`` queries.

    Slot semantics match the upstream dense table exactly; see the module
    docstring's "Faithfully reproduced upstream quirks".
    """

    __slots__ = ("_slots", "_rows", "_dims", "_vol", "n_colocated_drops")

    def __init__(self, dims, slots: np.ndarray, rows: np.ndarray) -> None:
        dims = (int(dims[0]), int(dims[1]), int(dims[2]))
        vol = dims[0] * dims[1] * dims[2]
        slots = np.asarray(slots, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int32)
        assert slots.shape == rows.shape, (slots.shape, rows.shape)

        self._dims = dims
        self._vol = vol

        if slots.size:
            if slots.min() < -vol or slots.max() > vol - 1:
                bad = slots[(slots < -vol) | (slots > vol - 1)][0]
                raise IndexError(
                    f"index {int(bad)} is out of bounds for dimension 0 with size {vol}"
                )
            # The dense table indexed with ``lookup[flat] = row``, which wraps a
            # negative flat index once. Reproduce that before deduplicating.
            slots = np.where(slots < 0, slots + vol, slots)

        order = np.argsort(slots, kind="stable")
        slots = slots[order]
        rows = rows[order]

        # Duplicate slots collapse to the highest row index (insert order is
        # increasing), matching ``lookup[key] = arange(N)``'s last write.
        if slots.size:
            is_last = np.empty(slots.shape, dtype=bool)
            is_last[-1] = True
            np.not_equal(slots[:-1], slots[1:], out=is_last[:-1])
            dropped = int(slots.size - int(is_last.sum()))
            slots = slots[is_last]
            rows = rows[is_last]
        else:
            dropped = 0
        self._slots = np.ascontiguousarray(slots)
        self._rows = np.ascontiguousarray(rows)
        # Observability: how many coordinate rows aliased an earlier key. On a
        # well-formed decode this is 0 (or equals the duplicate count).
        self.n_colocated_drops = dropped

    # -- construction ---------------------------------------------------------

    @property
    def dims(self) -> Tuple[int, int, int]:
        return self._dims

    @property
    def volume(self) -> int:
        """Logical grid volume ``D*H*W`` (the size of the dense table replaced)."""
        return self._vol

    def nbytes(self) -> int:
        """Bytes held by the sparse table -- the quantity the dense table blew up on."""
        return int(self._slots.nbytes + self._rows.nbytes)

    @staticmethod
    def flat_keys(coord_xyz: np.ndarray, dims) -> np.ndarray:
        """Upstream row-major flat key ``x*H*W + y*W + z`` as ``int64``.

        Multiplication order matches upstream ``_flat_key`` (``x * H * W`` is
        evaluated ``(x * H) * W``) and the int64 promotion matches its ``.long()``.
        """
        d, h, w = int(dims[0]), int(dims[1]), int(dims[2])
        c = np.asarray(coord_xyz)
        x = c[:, 0].astype(np.int64, copy=False)
        y = c[:, 1].astype(np.int64, copy=False)
        z = c[:, 2].astype(np.int64, copy=False)
        return x * (h * w) + y * w + z

    @classmethod
    def from_coords(cls, coords_xyz: np.ndarray, dims) -> "SparseVoxelLookup":
        """Build from ``(N, 3)`` integer coordinates; row index is the value."""
        coords_xyz = np.asarray(coords_xyz).reshape(-1, 3)
        keys = cls.flat_keys(coords_xyz, dims)
        rows = np.arange(coords_xyz.shape[0], dtype=np.int32)
        return cls(dims, keys, rows)

    # -- query ----------------------------------------------------------------

    def lookup_coords(self, query_xyz: np.ndarray) -> np.ndarray:
        """Resolve ``(M, 3)`` integer coordinates to row indices (``-1`` = missing).

        Mirrors upstream ``lookup_3d``: a query coordinate outside the grid is
        "missing" regardless of which slot it might otherwise hash to.
        """
        q = np.asarray(query_xyz).reshape(-1, 3)
        m = q.shape[0]
        out = np.full((m,), -1, dtype=np.int32)
        if m == 0 or self._slots.size == 0:
            return out

        d, h, w = self._dims
        qx = q[:, 0].astype(np.int64, copy=False)
        qy = q[:, 1].astype(np.int64, copy=False)
        qz = q[:, 2].astype(np.int64, copy=False)

        valid = (
            (qx >= 0) & (qx < d) & (qy >= 0) & (qy < h) & (qz >= 0) & (qz < w)
        )
        if not valid.any():
            return out

        # A valid coordinate always lands in [0, vol); -1 can never match a
        # stored (non-negative) slot, so masking is equivalent to upstream's
        # clamp-then-mask and needs no per-query branch.
        flat = np.where(valid, qx * (h * w) + qy * w + qz, np.int64(-1))

        idx = np.searchsorted(self._slots, flat, side="left")
        np.clip(idx, 0, self._slots.size - 1, out=idx)
        hit = valid & (self._slots[idx] == flat)
        out[hit] = self._rows[idx[hit]]
        return out


# ---------------------------------------------------------------------------
# Legacy cross-product compatibility
# ---------------------------------------------------------------------------


def _legacy_cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Preserve the original cross-product axis, including exactly three quads.

    The original dim=None call selects the first dimension of length three.
    An explicit last-axis call would change its behavior for shape (3, 3).
    """
    sized3 = [i for i, s in enumerate(a.shape) if s == 3]
    if not sized3:
        raise RuntimeError(
            "no dimension of size 3 in input"
        )
    return torch.linalg.cross(a, b, dim=sized3[0])


# ---------------------------------------------------------------------------
# Argument coercion (upstream precedence and assertions, unchanged)
# ---------------------------------------------------------------------------


def _resolve_aabb(aabb, device):
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    assert isinstance(aabb, torch.Tensor), (
        f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    )
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"
    return aabb


def _resolve_voxel_and_grid_size(aabb, voxel_size, grid_size, device):
    """Resolve ``(voxel_size, grid_size)`` with upstream precedence.

    ``voxel_size`` wins when supplied and ``grid_size`` is derived from it;
    otherwise ``grid_size`` is required and ``voxel_size`` is derived.
    """
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=device)
        assert isinstance(voxel_size, torch.Tensor), (
            f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, "
            f"but got {type(voxel_size)}"
        )
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    assert isinstance(voxel_size, torch.Tensor), (
        f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, "
        f"but got {type(voxel_size)}"
    )
    assert voxel_size.dim() == 1, f"voxel_size must be a 1D tensor, but got {voxel_size.shape}"
    assert voxel_size.size(0) == 3, (
        f"voxel_size must have 3 elements, but got {voxel_size.size(0)}"
    )
    assert isinstance(grid_size, torch.Tensor), (
        f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, "
        f"but got {type(grid_size)}"
    )
    assert grid_size.dim() == 1, f"grid_size must be a 1D tensor, but got {grid_size.shape}"
    assert grid_size.size(0) == 3, (
        f"grid_size must have 3 elements, but got {grid_size.size(0)}"
    )
    return voxel_size, grid_size


def _to_host_coords_int64(coords: torch.Tensor, n: int) -> np.ndarray:
    """Copy validated integer coordinates to contiguous int64 host storage."""
    arr = np.asarray(coords.detach().to("cpu").numpy())
    if arr.shape != (n, 3):
        raise ValueError(
            f"coords must have shape ({n}, 3) to match dual_vertices, got {tuple(arr.shape)}"
        )
    if arr.dtype != np.int64:
        arr = arr.astype(np.int64, copy=False)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@torch.no_grad()
def mac_flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected_flag: torch.Tensor,
    split_weight: Optional[torch.Tensor],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    train: bool = False,
    edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract a triangle mesh from decoded sparse dual-grid features.

    Mac inference replacement for upstream ``flexible_dual_grid_to_mesh`` with
    bounded host memory. Argument order and meaning match upstream, so it is a
    drop-in for the decoder call; ``edge_chunk_size`` is appended (a Mac-only
    memory knob that cannot change results).

    Args:
        coords: ``(N, 3)`` integer voxel coordinates (the decoder's
            ``SparseTensor`` batch column already dropped).
        dual_vertices: ``(N, 3)`` dual vertex offsets within the unit cell.
        intersected_flag: ``(N, 3)`` boolean active-edge flags per axis.
        split_weight: ``(N, 1)`` per-voxel quad-merge weights, or ``None`` to
            use the upstream minimum-angle heuristic.
        aabb: ``(2, 3)`` axis-aligned bounding box.
        voxel_size: Voxel size; one of this and ``grid_size`` is required.
        grid_size: Grid resolution; one of this and ``voxel_size`` is required.
        train: Must be ``False``; ``True`` raises ``ValueError``.
        edge_chunk_size: Maximum active edges materialised per chunk (``>= 1``).

    Returns:
        ``(mesh_vertices, mesh_triangles)`` on ``coords.device``.
        ``mesh_vertices`` is ``(N, 3)`` -- one vertex per input row, input
        order; ``mesh_triangles`` is ``(2 * L, 3)`` ``int32`` where ``L`` is the
        number of valid active edges.

    Raises:
        TypeError: if coordinates are not int32/int64 or flags are not boolean.
        ValueError: if ``train`` is truthy, or ``edge_chunk_size < 1``.
        IndexError: if a coordinate's flat key falls outside ``[-vol, vol - 1]``
            of the logical grid -- the same failure the upstream dense table
            produced. Decoder coordinates are in-grid by construction, so this
            cannot trigger during real inference.
    """
    if train:
        raise ValueError(
            "mac_flexible_dual_grid_to_mesh supports train=False only: the "
            "training path (mid-edge split vertices) is not ported to this Mac "
            "inference adapter. Call with train=False."
        )
    edge_chunk_size = int(edge_chunk_size)
    if edge_chunk_size < 1:
        raise ValueError(f"edge_chunk_size must be >= 1, got {edge_chunk_size}")

    if coords.dtype not in (torch.int32, torch.int64):
        raise TypeError("coords must have dtype torch.int32 or torch.int64")
    if intersected_flag.dtype != torch.bool:
        raise TypeError("intersected_flag must have dtype torch.bool")

    device = coords.device
    aabb = _resolve_aabb(aabb, device)
    voxel_size, grid_size = _resolve_voxel_and_grid_size(
        aabb, voxel_size, grid_size, device
    )

    N = int(dual_vertices.shape[0])
    dims = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))

    # --- CPU topology extraction -------------------------------------------
    # Integer coordinate work only: which edges are active, in what order, and
    # which row each quad corner resolves to. Doing it once on the host avoids a
    # per-neighbour device round-trip and lets the sparse search use int64 keys.
    coords_np = _to_host_coords_int64(coords, N)

    flags_np = np.asarray(intersected_flag.detach().to("cpu").numpy())
    if flags_np.shape != (N, 3):
        raise ValueError(
            "intersected_flag must have shape (N, 3) with N == coords.shape[0]; "
            f"got {tuple(flags_np.shape)} for N={N}"
        )
    flags_bool = flags_np  # dtype validated before the explicit host transfer

    def _finalize(quad_indices: Optional[torch.Tensor]):
        mesh_vertices = (
            (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)
        )
        if quad_indices is None or quad_indices.shape[0] == 0:
            return mesh_vertices, torch.empty((0, 3), dtype=torch.int32, device=device)

        split_1 = torch.tensor(QUAD_SPLIT_1, dtype=torch.long, device=device)
        split_2 = torch.tensor(QUAD_SPLIT_2, dtype=torch.long, device=device)

        if split_weight is None:
            # Minimum-angle heuristic with upstream's exact expression order:
            # float32 subtract, then the *legacy* cross (_legacy_cross, which
            # reproduces upstream's dim=None pick), then abs of the elementwise
            # product sum reduced over dim=1 -- so ties resolve identically under
            # the strict ``>`` rule and the L == 3 legacy quirk is preserved.
            def _align(split_table):
                tris = quad_indices[:, split_table]
                v = mesh_vertices[tris]
                n0 = _legacy_cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
                n1 = _legacy_cross(v[:, 2] - v[:, 1], v[:, 3] - v[:, 1])
                return (n0 * n1).sum(dim=1, keepdim=True).abs()

            mesh_triangles = torch.where(
                _align(split_1) > _align(split_2),
                quad_indices[:, split_1],
                quad_indices[:, split_2],
            ).reshape(-1, 3)
        else:
            ws = split_weight[quad_indices]
            ws_02 = ws[:, 0] * ws[:, 2]
            ws_13 = ws[:, 1] * ws[:, 3]
            mesh_triangles = torch.where(
                ws_02 > ws_13,
                quad_indices[:, split_1],
                quad_indices[:, split_2],
            ).reshape(-1, 3)

        return mesh_vertices, mesh_triangles.int()

    if N == 0:
        return _finalize(None)

    lookup = SparseVoxelLookup.from_coords(coords_np, dims)
    offset = np.asarray(EDGE_NEIGHBOR_VOXEL_OFFSET, dtype=np.int64)  # (3, 4, 3)

    # ``np.nonzero`` on (N, 3) yields row-major order over (row, axis), which is
    # exactly the order of ``edge_neighbor_voxel[intersected_flag]`` upstream.
    active_row, active_axis = np.nonzero(flags_bool)
    n_active = active_row.shape[0]
    if n_active == 0:
        return _finalize(None)

    blocks = []  # (C, 4) int32 index blocks, concatenated in chunk order
    total_quads = 0
    for start in range(0, n_active, edge_chunk_size):
        stop = min(start + edge_chunk_size, n_active)
        ax = active_axis[start:stop]
        rw = active_row[start:stop]
        # (C, 4, 3) corners; the bounded transient this chunking exists to cap.
        corners = coords_np[rw][:, None, :] + offset[ax]
        idx = lookup.lookup_coords(corners.reshape(-1, 3)).reshape(-1, 4)
        valid = idx.min(axis=1) >= 0  # all four corners present
        if not valid.any():
            continue
        block = idx[valid].astype(np.int32, copy=False)
        blocks.append(torch.from_numpy(block))
        total_quads += block.shape[0]

    if total_quads == 0:
        return _finalize(None)

    quad_indices = torch.cat(blocks, dim=0).to(device=device, dtype=torch.int32)
    return _finalize(quad_indices)
