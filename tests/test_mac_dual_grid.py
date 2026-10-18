"""CPU parity and memory tests for the explicit Mac extraction adapter.

The dictionary oracle uses independently fixed literal neighbor and split
geometry. Mutation controls prove that production winding/offset changes are
detected. A required SHA256-pinned upstream fixture provides a separate exact
parity reference, including the original angle-heuristic behavior.
"""

import ast
import builtins
import importlib.util
import hashlib
import os
import subprocess
import sys
import textwrap
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODULE_PATH = os.path.join(ROOT, "uniforge3d", "utils", "mac_dual_grid.py")
UPSTREAM_PATH = os.path.join(HERE, "fixtures", "upstream_flexible_dual_grid.py")
DECODER_PATH = os.path.join(ROOT, "uniforge3d", "models", "vaes", "flexi_dual_grid.py")

AABB_UNIT = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


md = _load("mac_dual_grid", MODULE_PATH)


def _load_upstream_reference():
    """Load the required exact reference; absence or modification fails closed."""
    assert os.path.isfile(UPSTREAM_PATH), "Required pinned upstream reference is missing"
    with open(UPSTREAM_PATH, "rb") as source:
        digest = hashlib.sha256(source.read()).hexdigest()
    assert digest == "deb1c42444de13792255c09d261de2d194b7b61f92860c49d2acbc5efbbc17af", "Upstream reference hash changed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _load("upstream_fdg", UPSTREAM_PATH)


@pytest.fixture(scope="session")
def upstream():
    return _load_upstream_reference()


@pytest.mark.parametrize("reference_path, message", [
    (os.path.join(HERE, "missing_upstream_reference.py"), "reference is missing"),
    (MODULE_PATH, "reference hash changed"),
])
def test_upstream_reference_is_required_and_pinned(monkeypatch, reference_path, message):
    monkeypatch.setattr(sys.modules[__name__], "UPSTREAM_PATH", reference_path)
    with pytest.raises(AssertionError, match=message):
        _load_upstream_reference()


# ---------------------------------------------------------------------------
# Independent oracle
# ---------------------------------------------------------------------------


def _legacy_dim(t):
    """Upstream's ``torch.cross`` (dim=None) picks the first dimension of size 3."""
    for i, s in enumerate(t.shape):
        if s == 3:
            return i
    raise RuntimeError("no dimension of size 3 in input")


def _naive_align(mesh_vertices, quad_indices, split_table):
    """Alignment computed with the 'obvious' ``dim=-1`` cross, for contrast only."""
    tris = quad_indices[:, torch.tensor(list(split_table), dtype=torch.long)]
    v = mesh_vertices[tris]
    n0 = torch.linalg.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1)
    n1 = torch.linalg.cross(v[:, 2] - v[:, 1], v[:, 3] - v[:, 1], dim=-1)
    return (n0 * n1).sum(dim=1).abs()


def _oracle_align(mesh_vertices, quad_indices, split_table):
    """Upstream's alignment for one split, using the legacy-cross dimension rule."""
    tris = quad_indices[:, torch.tensor(list(split_table), dtype=torch.long)]
    v = mesh_vertices[tris]
    d0 = v[:, 2] - v[:, 0]
    d1 = v[:, 3] - v[:, 1]
    n0 = torch.linalg.cross(v[:, 1] - v[:, 0], d0, dim=_legacy_dim(v[:, 1] - v[:, 0]))
    n1 = torch.linalg.cross(v[:, 2] - v[:, 1], d1, dim=_legacy_dim(v[:, 2] - v[:, 1]))
    return (n0 * n1).sum(dim=1).abs()


def _oracle(coords, dual_vertices, flags, split_weight, aabb, grid_size):
    """Dictionary-based reimplementation of the contract.

    Args:
        coords: ``(N, 3)`` int array.
        dual_vertices: ``(N, 3)`` float array.
        flags: ``(N, 3)`` bool array.
        split_weight: ``(N, 1)`` array or ``None``.
        aabb: ``(2, 3)`` sequence.
        grid_size: ``(3,)`` sequence of ints.

    Returns:
        ``(mesh_vertices, mesh_triangles)`` torch tensors, matching the module's
        contract (float32 vertices, int32 triangles, input row order).
    """
    coords = np.asarray(coords)
    dv = np.asarray(dual_vertices, dtype=np.float32)
    gs = tuple(int(v) for v in grid_size)
    n = coords.shape[0]
    d, h, w = gs
    vol = d * h * w

    # Fixed independently of the production module so a shared typo cannot
    # change both implementation and expected geometry.
    offs = np.asarray([
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    ], dtype=np.int64)
    split_1 = [0, 1, 2, 0, 2, 3]
    split_2 = [0, 1, 3, 3, 1, 2]

    # Voxel table: dict, last write wins, negative flat index wraps once.
    table = {}
    for r in range(n):
        flat = int(coords[r, 0]) * h * w + int(coords[r, 1]) * w + int(coords[r, 2])
        if flat < 0:
            flat += vol
        if flat < 0 or flat > vol - 1:
            raise IndexError(f"index {flat} is out of bounds for dimension 0 with size {vol}")
        table[flat] = r

    quad_rows = []
    for r in range(n):
        for axis in range(3):
            if not bool(flags[r, axis]):
                continue
            corners = []
            complete = True
            for c in range(4):
                nx = int(coords[r, 0]) + int(offs[axis, c, 0])
                ny = int(coords[r, 1]) + int(offs[axis, c, 1])
                nz = int(coords[r, 2]) + int(offs[axis, c, 2])
                if not (0 <= nx < d and 0 <= ny < h and 0 <= nz < w):
                    complete = False
                    break
                row = table.get(nx * h * w + ny * w + nz)
                if row is None:
                    complete = False
                    break
                corners.append(row)
            if complete:
                quad_rows.append(corners)

    aabb_t = torch.tensor(aabb, dtype=torch.float32)
    voxel_size = (aabb_t[1] - aabb_t[0]) / torch.tensor(gs, dtype=torch.int32)
    mesh_vertices = (
        (torch.tensor(coords, dtype=torch.float32) + torch.tensor(dv))
        * voxel_size
        + aabb_t[0].reshape(1, 3)
    )

    if not quad_rows:
        return mesh_vertices, torch.empty((0, 3), dtype=torch.int32)

    qi = torch.tensor(np.asarray(quad_rows, dtype=np.int64), dtype=torch.long)
    s1 = torch.tensor(split_1, dtype=torch.long)
    s2 = torch.tensor(split_2, dtype=torch.long)

    if split_weight is None:
        a0 = _oracle_align(mesh_vertices, qi, split_1).unsqueeze(1)
        a1 = _oracle_align(mesh_vertices, qi, split_2).unsqueeze(1)
        tri = torch.where(a0 > a1, qi[:, s1], qi[:, s2]).reshape(-1, 3)
    else:
        sw = torch.tensor(np.asarray(split_weight, dtype=np.float32))[qi]
        tri = torch.where(
            (sw[:, 0] * sw[:, 2]) > (sw[:, 1] * sw[:, 3]), qi[:, s1], qi[:, s2]
        ).reshape(-1, 3)

    return mesh_vertices, tri.int()


def _call_md(coords, dv, flags, split_weight=None, aabb=AABB_UNIT, **kw):
    return md.mac_flexible_dual_grid_to_mesh(
        coords, dv, flags, split_weight, aabb=aabb, **kw
    )


def _t(x, dtype=None):
    a = np.asarray(x)
    return torch.tensor(a, dtype=dtype) if dtype is not None else torch.tensor(a)


def _single_quad_fixture(axis, choose_split_1):
    # Build the input geometry independently of production tables. The full
    # cube supplies every corner; one active axis gives exactly one quad.
    coords = np.array([(x, y, z) for x in range(4) for y in range(4) for z in range(4)])
    flags = np.zeros((len(coords), 3), dtype=bool)
    row = next(i for i, xyz in enumerate(coords.tolist()) if xyz == [1, 1, 1])
    flags[row, axis] = True
    dual = np.full((len(coords), 3), .25, dtype=np.float32)
    weights = np.ones((len(coords), 1), dtype=np.float32)
    if choose_split_1:
        # This row is corner zero for every axis; doubling its weight makes
        # w0*w2 > w1*w3. All-one weights exercise the strict-tie split instead.
        weights[row] = 2.
    return coords, dual, flags, weights


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("split", [1, 2])
def test_dictionary_oracle_detects_reversed_winding(monkeypatch, axis, split):
    coords, dual, flags, weights = _single_quad_fixture(axis, split == 1)
    expected = _oracle(coords, dual, flags, weights, AABB_UNIT, (4, 4, 4))
    baseline = _call_md(_t(coords, torch.int32), _t(dual), _t(flags), _t(weights), grid_size=4)
    assert expected[1].shape == (2, 3) and torch.equal(baseline[1], expected[1])
    reversed_table = (0, 2, 1, 0, 3, 2) if split == 1 else (0, 3, 1, 3, 2, 1)
    monkeypatch.setattr(md, f"QUAD_SPLIT_{split}", reversed_table)
    # Recomputing the oracle after mutation must still return the original
    # winding; otherwise it shares the production table being tested.
    assert torch.equal(_oracle(coords, dual, flags, weights, AABB_UNIT, (4, 4, 4))[1], expected[1])
    mutant = _call_md(_t(coords, torch.int32), _t(dual), _t(flags), _t(weights), grid_size=4)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(mutant[1], expected[1], atol=0, rtol=0)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("mutation", ["missing_corner", "swapped_corners"])
def test_dictionary_oracle_detects_neighbor_offset_mutations(monkeypatch, axis, mutation):
    coords, dual, flags, weights = _single_quad_fixture(axis, False)
    expected = _oracle(coords, dual, flags, weights, AABB_UNIT, (4, 4, 4))
    baseline = _call_md(_t(coords, torch.int32), _t(dual), _t(flags), _t(weights), grid_size=4)
    assert expected[1].shape == (2, 3) and torch.equal(baseline[1], expected[1])
    changed = [[tuple(corner) for corner in plane] for plane in md.EDGE_NEIGHBOR_VOXEL_OFFSET]
    if mutation == "missing_corner":
        changed[axis][1] = (9, 9, 9)
    else:
        changed[axis][1], changed[axis][3] = changed[axis][3], changed[axis][1]
    monkeypatch.setattr(md, "EDGE_NEIGHBOR_VOXEL_OFFSET", tuple(tuple(plane) for plane in changed))
    assert torch.equal(_oracle(coords, dual, flags, weights, AABB_UNIT, (4, 4, 4))[1], expected[1])
    mutant = _call_md(_t(coords, torch.int32), _t(dual), _t(flags), _t(weights), grid_size=4)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(mutant[1], expected[1], atol=0, rtol=0)


def _random_case(rng, max_dim=6, max_n=40):
    gs = tuple(int(v) for v in rng.integers(2, max_dim + 1, size=3))
    n = int(rng.integers(1, max_n))
    dims = np.array(gs)
    coords = np.stack([rng.integers(0, int(dims[i]), size=n) for i in range(3)], axis=1)
    dv = rng.random((n, 3)).astype(np.float32)
    flags = rng.random((n, 3)) < 0.5
    return coords, dv, flags, gs


# ---------------------------------------------------------------------------
# imports / basic contract
# ---------------------------------------------------------------------------


def test_module_imports_and_exports():
    for name in ("mac_flexible_dual_grid_to_mesh", "SparseVoxelLookup",
                 "EDGE_NEIGHBOR_VOXEL_OFFSET", "QUAD_SPLIT_1", "QUAD_SPLIT_2"):
        assert hasattr(md, name), name
    assert tuple(md.QUAD_SPLIT_1) == (0, 1, 2, 0, 2, 3)
    assert tuple(md.QUAD_SPLIT_2) == (0, 1, 3, 3, 1, 2)
    assert np.asarray(md.EDGE_NEIGHBOR_VOXEL_OFFSET).shape == (3, 4, 3)


def test_train_true_rejected():
    coords = _t([[0, 0, 0]], torch.int32)
    dv = torch.zeros(1, 3)
    flags = _t([[True, False, False]])
    for train in (True, 1):
        with pytest.raises(ValueError, match="train=False"):
            _call_md(coords, dv, flags, None, grid_size=(4, 4, 4), train=train)


def test_edge_chunk_size_validated():
    coords = _t([[0, 0, 0]], torch.int32)
    with pytest.raises(ValueError, match="edge_chunk_size"):
        _call_md(coords, torch.zeros(1, 3), _t([[True, False, False]]),
                 None, grid_size=(4, 4, 4), edge_chunk_size=0)


# ---------------------------------------------------------------------------
# upstream dense parity (tiny grids)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("with_split", [False, True])
def test_matches_upstream_dense(upstream, seed, with_split):
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(120):
        coords, dv, flags, gs = _random_case(rng)
        sw = rng.random((coords.shape[0], 1)).astype(np.float32) if with_split else None
        tc = _t(coords, torch.int32)
        td = _t(dv, torch.float32)
        tf = _t(flags)
        tsw = None if sw is None else _t(sw, torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            up_v, up_t = upstream.flexible_dual_grid_to_mesh(
                tc, td, tf, tsw, aabb=AABB_UNIT, grid_size=gs
            )
        my_v, my_t = _call_md(tc, td, tf, tsw, grid_size=gs)
        assert torch.equal(up_v, my_v), "vertex values"
        assert torch.equal(up_t, my_t), "triangle indices/order"
        assert my_t.dtype == torch.int32
        checked += 1
    assert checked >= 100


def test_exhaustive_flag_combinations_match_upstream(upstream):
    """Enumerate every active-edge flag pattern on tiny grids against upstream.

    Boundary and degenerate behaviour lives in the flag patterns: with all cells
    present, each pattern decides which corner quadruples are complete. Tiny
    grids make upstream's dense table affordable, so this is the strongest
    order-and-semantics check available. Capped at 12 flag bits per grid to keep
    runtime bounded.
    """
    import itertools

    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    cases = 0
    for gs in [(2, 2, 2), (3, 1, 1), (1, 1, 3), (2, 2, 1)]:
        voxels = [(x, y, z) for x in range(gs[0]) for y in range(gs[1]) for z in range(gs[2])]
        n = len(voxels)
        dv = _t(np.random.default_rng(n).random((n, 3)).astype(np.float32), torch.float32)
        tc = _t(voxels, torch.int32)
        for bits in itertools.product([0, 1], repeat=min(n * 3, 12)):
            flags = np.zeros((n, 3), dtype=bool)
            for k, b in enumerate(bits):
                if k < n * 3:
                    flags[k // 3, k % 3] = bool(b)
            tf = _t(flags)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                up_v, up_t = upstream.flexible_dual_grid_to_mesh(
                    tc, dv, tf, None, aabb=aabb, grid_size=gs)
            my_v, my_t = _call_md(tc, dv, tf, None, aabb=aabb, grid_size=gs)
            assert torch.equal(my_v, up_v), (gs, bits)
            assert torch.equal(my_t, up_t), (gs, bits)
            cases += 1
    assert cases >= 5000, cases


def test_malformed_input_raises_clearly():
    """Shape errors raise ValueError with a message; upstream raises a cryptic IndexError.

    A declared divergence: upstream's dense path indexes with a mismatched mask
    and surfaces ``IndexError: The shape of the mask [4, 3] at index 0 does not
    match the shape of the indexed tensor``. The port validates first and raises
    ``ValueError`` naming the expected shape. Both refuse the input; only the
    message quality differs.
    """
    coords = _t(np.zeros((8, 3), dtype=int), torch.int32)
    dv = torch.zeros(8, 3)
    with pytest.raises(ValueError, match="intersected_flag must have shape"):
        _call_md(coords, dv, _t(np.zeros((4, 3), dtype=bool)), None, grid_size=(2, 2, 2))


def test_matches_upstream_dense_anisotropic_and_shuffled(upstream):
    rng = np.random.default_rng(99)
    for gs in [(11, 3, 5), (2, 9, 4), (7, 7, 2)]:
        n = 30
        coords = np.stack([rng.integers(0, gs[i], size=n) for i in range(3)], axis=1)
        dv = rng.random((n, 3)).astype(np.float32)
        flags = rng.random((n, 3)) < 0.6
        perm = rng.permutation(n)
        for c, d, f in ((coords, dv, flags), (coords[perm], dv[perm], flags[perm])):
            tc, td, tf = _t(c, torch.int32), _t(d, torch.float32), _t(f)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                up_v, up_t = upstream.flexible_dual_grid_to_mesh(
                    tc, td, tf, None, aabb=AABB_UNIT, grid_size=gs
                )
            my_v, my_t = _call_md(tc, td, tf, None, grid_size=gs)
            assert torch.equal(up_v, my_v)
            assert torch.equal(up_t, my_t)


# ---------------------------------------------------------------------------
# independent dictionary oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [11, 12, 13])
@pytest.mark.parametrize("with_split", [False, True])
def test_matches_dictionary_oracle(seed, with_split):
    rng = np.random.default_rng(seed)
    tri_total = 0
    for _ in range(150):
        coords, dv, flags, gs = _random_case(rng)
        sw = rng.random((coords.shape[0], 1)).astype(np.float32) if with_split else None
        tc, td, tf = _t(coords, torch.int32), _t(dv, torch.float32), _t(flags)
        o_v, o_t = _oracle(coords, dv, flags, sw, AABB_UNIT, gs)
        m_v, m_t = _call_md(tc, td, tf, None if sw is None else _t(sw), grid_size=gs)
        assert torch.equal(o_v, m_v)
        assert torch.equal(o_t, m_t), (gs, coords.tolist())
        tri_total += m_t.shape[0]
    assert tri_total > 0


def test_oracle_agrees_on_anisotropic_voxel_and_grid_derivation():
    rng = np.random.default_rng(77)
    coords = np.stack([rng.integers(0, 6, size=20) for _ in range(3)], axis=1)
    dv = rng.random((20, 3)).astype(np.float32)
    flags = rng.random((20, 3)) < 0.5
    aabb = [[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]]
    gs = (8, 4, 2)
    o_v, o_t = _oracle(coords, dv, flags, None, aabb, gs)
    m_v, m_t = _call_md(_t(coords, torch.int32), _t(dv), _t(flags), None, aabb=aabb, grid_size=gs)
    assert torch.equal(o_v, m_v)
    assert torch.equal(o_t, m_t)
    # voxel_size path derives grid_size from aabb/voxel_size, so compare derived.
    vs = (np.array(aabb[1]) - np.array(aabb[0])) / np.array(gs, dtype=np.float64)
    m_v2, m_t2 = _call_md(_t(coords, torch.int32), _t(dv), _t(flags), None,
                          aabb=aabb, voxel_size=[float(v) for v in vs])
    assert torch.equal(m_t2, m_t)
    assert torch.allclose(m_v2, m_v, atol=1e-6)


# ---------------------------------------------------------------------------
# split directions and ties
# ---------------------------------------------------------------------------


def test_split_weight_both_directions_and_tie():
    coords = _t([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], torch.int32)
    dv = _t([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.1]], torch.float32)
    flags = _t([[True, False, False]] * 4)
    s1 = list(md.QUAD_SPLIT_1)
    s2 = list(md.QUAD_SPLIT_2)

    # ws02 = sw0*sw2 > ws13 = sw1*sw3 -> split 1
    sw = _t([[4.0], [1.0], [4.0], [1.0]], torch.float32)
    _, out = _call_md(coords, dv, flags, sw, grid_size=(8, 8, 8))
    assert out.shape == (2, 3)
    assert out.reshape(-1).tolist() == [0, 1, 2, 0, 2, 3]

    # ws02 < ws13 -> split 2
    sw = _t([[1.0], [4.0], [1.0], [4.0]], torch.float32)
    _, out = _call_md(coords, dv, flags, sw, grid_size=(8, 8, 8))
    assert out.reshape(-1).tolist() == [0, 1, 3, 3, 1, 2]

    # ws02 == ws13 -> strict '>' fails, split 2 wins
    sw = _t([[2.0], [2.0], [2.0], [2.0]], torch.float32)
    _, out = _call_md(coords, dv, flags, sw, grid_size=(8, 8, 8))
    assert out.reshape(-1).tolist() == [0, 1, 3, 3, 1, 2]

    # The decoder supplies split_weight as (N, 1); that is the supported shape.
    sw1 = torch.full((4, 1), 2.0)
    _, out1 = _call_md(coords, dv, flags, sw1, grid_size=(8, 8, 8))
    assert out1.reshape(-1).tolist() == [0, 1, 3, 3, 1, 2]

    # (N, k>1) trailing dims fail identically in upstream and in the port: the
    # gathered (L, k) block cannot be indexed by a (L, 6) split table.
    for k in (2, 4):
        with pytest.raises(RuntimeError):
            _call_md(coords, dv, flags, torch.full((4, k), 0.5), grid_size=(8, 8, 8))


def test_angle_heuristic_tie_falls_through_to_split_2():
    """A fully degenerate quad gives an exact 0 > 0 failure -> split 2.

    ``quad_split_2``'s first four slots are ``(0, 1, 3, 3)``, so the second
    cross product is ``cross(v3 - v1, v3 - v1)`` and its alignment is ~0 for
    every non-degenerate quad. Split 2 therefore only appears when the first
    alignment is *also* zero, i.e. on a degenerate (zero-area) quad. This test
    pins that exact-tie behaviour; the quirk itself is recorded in
    IMPLEMENTATION.md rather than changing the original arithmetic.
    """
    coords = _t([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], torch.int32)
    # dual_vertices = -coords collapses all four corners onto the same point.
    dv = _t(-np.asarray(coords, dtype=np.float32), torch.float32)
    flags = _t([[True, False, False]] * 4)
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    mesh_vertices = (coords.float() + dv) * (torch.tensor([1.0, 1.0, 1.0]) / 8) \
        + torch.tensor(aabb[0])
    a0 = _oracle_align(mesh_vertices, torch.arange(4).reshape(1, 4), md.QUAD_SPLIT_1)
    a1 = _oracle_align(mesh_vertices, torch.arange(4).reshape(1, 4), md.QUAD_SPLIT_2)
    assert torch.equal(a0, a1), "degenerate quad must be an exact tie"
    assert a0.item() == 0.0

    _, out = _call_md(coords, dv, flags, None, aabb=aabb, grid_size=(8, 8, 8))
    assert out.reshape(-1).tolist() == [0, 1, 3, 3, 1, 2]


def test_legacy_cross_dimension_quirk_is_preserved(upstream):
    """``torch.cross`` without ``dim`` crosses the FIRST axis of size 3.

    Upstream's heuristic calls ``torch.cross(n0, n1)`` with the deprecated
    ``dim=None``. Legacy semantics pick the first dimension of size 3 in the
    broadcast result -- the vertex axis for a ``(L, 3)`` pair, *except* when the
    quad count ``L`` equals 3, where dim 0 wins and the cross product is taken
    across quads. That silently changes the alignment, and therefore the split
    decision, for a 3-quad grid, so a naive ``dim=-1`` port diverges from
    upstream exactly there.

    The case below is a real reaching instance: upstream emits triangles
    ``[3, 6, 14], [14, 6, 0]`` while a ``dim=-1`` port emits
    ``[3, 6, 0], [3, 0, 14]`` for the same quad. The port must match upstream.
    """
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    gs = (4, 3, 3)
    coords = _t([[1, 2, 0], [1, 1, 0], [3, 2, 2], [0, 1, 0], [1, 1, 1], [1, 1, 2],
                 [0, 2, 0], [2, 2, 1], [3, 2, 1], [3, 1, 2], [1, 2, 1], [2, 1, 1],
                 [3, 2, 1], [1, 1, 1], [1, 1, 0], [3, 1, 2], [0, 1, 2], [3, 2, 1]],
                torch.int32)
    flags = _t(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0],
                         [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1],
                         [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0],
                         [0, 1, 1], [0, 0, 0], [0, 0, 1]], dtype=bool))
    dv = _t(np.array([0.307404, 0.497917, 0.517319, 0.482623, 0.010487, 0.980052,
                      0.990318, 0.711968, 0.183635, 0.874536, 0.250487, 0.037806,
                      0.18446, 0.788941, 0.870025, 0.963152, 0.80459, 0.306169,
                      0.553679, 0.700233, 0.055527, 0.085554, 0.714023, 0.341832,
                      0.092704, 0.502115, 0.091438, 0.672008, 0.084228, 0.975516,
                      0.822685, 0.203529, 0.15337, 0.710537, 0.23481, 0.867309,
                      0.752654, 0.803331, 0.382695, 0.810189, 0.920507, 0.874113,
                      0.286354, 0.194836, 0.099592, 0.219092, 0.607359, 0.308728,
                      0.475176, 0.897006, 0.58557, 0.426615, 0.094875, 0.642506],
                     dtype=np.float32).reshape(18, 3), torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        up_v, up_t = upstream.flexible_dual_grid_to_mesh(
            coords, dv, flags, None, aabb=aabb, grid_size=gs)
    my_v, my_t = _call_md(coords, dv, flags, None, aabb=aabb, grid_size=gs)

    assert up_t.shape[0] == 6, "this fixture must hit the L == 3 legacy path"
    assert torch.equal(my_v, up_v), "vertex values"
    assert torch.equal(my_t, up_t), "triangles under the legacy-cross quirk"

    # Guard: a dim=-1 port really would differ here, so this test has teeth.
    quad_indices = _t([[14, 13, 10, 0], [3, 6, 0, 14], [11, 8, 6, 1]], torch.int32)
    a0 = _oracle_align(my_v, quad_indices, md.QUAD_SPLIT_1)
    naive0 = _naive_align(my_v, quad_indices, md.QUAD_SPLIT_1)
    assert not torch.equal(a0, naive0), "fixture must distinguish legacy from dim=-1"


def test_legacy_cross_dimension_quirk_across_random_3quad_grids(upstream):
    """Many random 3-quad grids must match upstream bit-for-bit."""
    rng = np.random.default_rng(19)
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    reached = 0
    for _ in range(400):
        gs = tuple(int(v) for v in rng.integers(2, 5, size=3))
        n = int(rng.integers(2, 18))
        coords = np.stack([rng.integers(0, gs[i], size=n) for i in range(3)], axis=1)
        flags = rng.random((n, 3)) < 0.5
        dv = rng.random((n, 3)).astype(np.float32)
        tc = _t(coords, torch.int32)
        td = _t(dv, torch.float32)
        tf = _t(flags)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            up = upstream.flexible_dual_grid_to_mesh(tc, td, tf, None, aabb=aabb, grid_size=gs)
        if up[1].shape[0] != 6:
            continue
        mine = _call_md(tc, td, tf, None, aabb=aabb, grid_size=gs)
        assert torch.equal(mine[1], up[1]), (gs, coords.tolist())
        reached += 1
    assert reached > 0, "no L == 3 case encountered"


def test_legacy_cross_matches_torch_cross_operation():
    """``_legacy_cross`` reproduces ``torch.cross`` (dim=None) exactly, incl. L == 3."""
    rng = np.random.default_rng(5)
    for L in (1, 2, 3, 4, 9):
        a = torch.rand(L, 3, generator=torch.Generator().manual_seed(L))
        b = torch.rand(L, 3, generator=torch.Generator().manual_seed(L + 100))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            legacy = torch.cross(a, b)
        assert torch.equal(md._legacy_cross(a, b), legacy), L


def test_angle_heuristic_split2_requires_exact_tie():
    """Over many random vertex sets the heuristic picks split 1 unless degenerate.

    Guards the documented quirk: ``align1`` is ~0 by construction
    (``quad_split_2`` slots 2 and 3 are both index 3), so split 2 can only win
    when ``align0`` is also ~0. If a future edit "fixes" the cross dimension or
    the table, this test fails and forces a deliberate decision.
    """
    rng = np.random.default_rng(2024)
    coords = _t([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], torch.int32)
    flags = _t([[True, False, False]] * 4)
    picks = set()
    for _ in range(500):
        dv = _t(rng.random((4, 3)).astype(np.float32), torch.float32)
        _, out = _call_md(coords, dv, flags, None, grid_size=(8, 8, 8))
        picks.add(out.reshape(-1)[2].item())
    assert picks == {2}, f"non-degenerate quads must pick split 1, saw {picks}"

    # A degenerate quad is the exact tie that selects split 2 instead.
    dv0 = _t(-np.asarray(coords, dtype=np.float32), torch.float32)
    _, out0 = _call_md(coords, dv0, flags, None, grid_size=(8, 8, 8))
    assert out0.reshape(-1)[2].item() == 3


# ---------------------------------------------------------------------------
# empty / no-face inputs
# ---------------------------------------------------------------------------


def test_empty_coords():
    coords = torch.zeros((0, 3), dtype=torch.int32)
    dv = torch.zeros((0, 3))
    flags = torch.zeros((0, 3), dtype=torch.bool)
    v, t = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert v.shape == (0, 3) and t.shape == (0, 3) and t.dtype == torch.int32
    # and with a split_weight present (upstream accepts it on this path)
    v, t = _call_md(coords, dv, flags, torch.zeros((0, 1)), grid_size=(4, 4, 4))
    assert t.shape == (0, 3)


def test_no_active_flags():
    coords = _t([[1, 1, 1], [2, 2, 2]], torch.int32)
    dv = _t([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], torch.float32)
    flags = torch.zeros((2, 3), dtype=torch.bool)
    v, t = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert v.shape == (2, 3) and t.shape == (0, 3)
    # vertices are still returned for every input row, in order
    expected = (coords.float() + dv) * ((torch.tensor([1.0, 1.0, 1.0])) / 4) \
        + torch.tensor([-0.5, -0.5, -0.5])
    assert torch.equal(v, expected)


def test_empty_upstream_parity(upstream):
    coords = torch.zeros((0, 3), dtype=torch.int32)
    dv = torch.zeros((0, 3))
    flags = torch.zeros((0, 3), dtype=torch.bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        up = upstream.flexible_dual_grid_to_mesh(coords, dv, flags, None, aabb=AABB_UNIT, grid_size=(4, 4, 4))
    mine = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert torch.equal(up[0], mine[0]) and torch.equal(up[1], mine[1])


# ---------------------------------------------------------------------------
# boundary / missing neighbours
# ---------------------------------------------------------------------------


def test_boundary_missing_voxel_creates_no_quad():
    # A quad whose neighbour lies outside the grid must be dropped entirely.
    coords = _t([[1, 1, 3], [1, 1, 2]], torch.int32)
    dv = torch.zeros(2, 3)
    flags = _t([[True, False, False], [False, False, False]])
    v, t = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert t.shape == (0, 3)


def test_boundary_partial_quad_dropped():
    # Three of four corners present -> no quad; all four -> quad.
    gs = (4, 4, 4)
    base = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]
    flags3 = [[True, False, False]] * 3
    dv3 = [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.2, 0.1]]
    v, t = _call_md(_t(base[:3], torch.int32), _t(dv3, torch.float32), _t(flags3), None, grid_size=gs)
    assert t.shape == (0, 3)
    dv4 = dv3 + [[0.1, 0.2, 0.1]]
    v, t = _call_md(_t(base, torch.int32), _t(dv4, torch.float32), _t([[True, False, False]] * 4), None, grid_size=gs)
    assert t.shape == (2, 3)


def test_boundary_parity_with_upstream(upstream):
    rng = np.random.default_rng(5)
    gs = (4, 4, 4)
    for _ in range(30):
        coords, dv, flags, _ = _random_case(rng, max_dim=4, max_n=25)
        tc, td, tf = _t(coords, torch.int32), _t(dv, torch.float32), _t(flags)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            up = upstream.flexible_dual_grid_to_mesh(tc, td, tf, None, aabb=AABB_UNIT, grid_size=gs)
        mine = _call_md(tc, td, tf, None, grid_size=gs)
        assert torch.equal(up[1], mine[1])


# ---------------------------------------------------------------------------
# duplicates and out-of-range coordinates (dense-table quirks reproduced)
# ---------------------------------------------------------------------------


def test_duplicate_coords_last_row_wins():
    # Two rows share the key of (0,0,0); upstream's dense table keeps the LAST.
    coords = _t([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], torch.int32)
    dv = _t([[8.0, 8.0, 8.0], [0.1, 0.1, 0.1], [0.2, 0.1, 0.1],
             [0.2, 0.2, 0.1], [0.1, 0.2, 0.1]], torch.float32)
    flags = _t([[True, False, False]] * 5)
    _, out = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert out[0, 0].item() == 1, "quad corner must resolve to the last duplicate row"


def test_duplicate_coords_upstream_parity(upstream):
    coords = _t([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]], torch.int32)
    dv = _t([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.1], [7.0, 7.0, 7.0]], torch.float32)
    flags = _t([[True, False, False]] * 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        up = upstream.flexible_dual_grid_to_mesh(coords, dv, flags, None, aabb=AABB_UNIT, grid_size=(4, 4, 4))
    mine = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert torch.equal(up[0], mine[0]) and torch.equal(up[1], mine[1])


def test_negative_coord_wraps_like_dense_table(upstream):
    # flat = -64 wraps to slot 0; appending it after the real rows must clobber.
    quad = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]
    dvq = [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.1]]
    t3 = [True, False, False]
    f3 = [False, False, False]

    coords = _t(np.array(quad + [[-4, 0, 0]]), torch.int32)
    dv = _t(np.array(dvq + [[9.0, 9.0, 9.0]]), torch.float32)
    flags = _t(np.array([t3] * 4 + [f3]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        up = upstream.flexible_dual_grid_to_mesh(coords, dv, flags, None, aabb=AABB_UNIT, grid_size=(4, 4, 4))
    mine = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert torch.equal(up[1], mine[1])
    assert mine[1][0, 0].item() == 4, "wrapped row must alias slot 0"

    # Prepended, the legitimate row is inserted later and wins instead.
    coords2 = _t(np.array([[-4, 0, 0]] + quad), torch.int32)
    dv2 = _t(np.array([[9.0, 9.0, 9.0]] + dvq), torch.float32)
    flags2 = _t(np.array([f3] + [t3] * 4))
    mine2 = _call_md(coords2, dv2, flags2, None, grid_size=(4, 4, 4))
    assert torch.equal(mine2[1], mine2[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        up2 = upstream.flexible_dual_grid_to_mesh(coords2, dv2, flags2, None, aabb=AABB_UNIT, grid_size=(4, 4, 4))
    assert torch.equal(up2[1], mine2[1])
    assert mine2[1][0, 0].item() == 1


def test_far_out_of_range_coord_raises_index_error(upstream):
    quad = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]
    dvq = [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.1]]
    coords = _t(np.array(quad + [[-4, 0, -1]]), torch.int32)  # flat = -65
    dv = _t(np.array(dvq + [[1.0, 1.0, 1.0]]), torch.float32)
    flags = _t(np.array([[True, False, False]] * 4 + [[False, False, False]]))
    with pytest.raises(IndexError):
        _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(IndexError):
            upstream.flexible_dual_grid_to_mesh(coords, dv, flags, None, aabb=AABB_UNIT, grid_size=(4, 4, 4))


def test_out_of_range_query_is_missing_not_aliased():
    # A neighbour that leaves the grid is simply absent -- never aliases a voxel.
    coords = _t([[0, 0, 0], [0, 0, 3]], torch.int32)  # second row is in-grid but far
    dv = torch.zeros(2, 3)
    flags = _t([[True, False, False], [False, False, False]])
    _, t = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert t.shape == (0, 3)


# ---------------------------------------------------------------------------
# shuffled input ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_shuffled_coords_permute_outputs_consistently(seed):
    """Shuffling the input rows permutes vertices and relabels identical geometry.

    Row indices are *not* preserved as triangle labels: the table resolves a
    coordinate to the last row that wrote it, so duplicate coordinates can swap
    which row a triangle names when the input order changes. The invariant that
    must hold -- and does -- is that the extracted geometry is identical when
    compared in coordinate space, and that vertex ``i`` tracks input row ``i``.
    """
    rng = np.random.default_rng(seed)
    gs = (4, 4, 4)
    # A full dense block: every cell present, every axis-flag active -> many quads.
    coords = np.array([(x, y, z) for x in range(gs[0]) for y in range(gs[1]) for z in range(gs[2])])
    n = coords.shape[0]
    dv = rng.random((n, 3)).astype(np.float32)
    flags = np.ones((n, 3), dtype=bool)

    base_v, base_t = _call_md(_t(coords, torch.int32), _t(dv, torch.float32), _t(flags), None, grid_size=gs)
    assert base_t.shape[0] >= 50, "dense block should yield many triangles"

    perm = rng.permutation(n)
    sh_v, sh_t = _call_md(_t(coords[perm], torch.int32), _t(dv[perm], torch.float32),
                          _t(flags[perm]), None, grid_size=gs)

    # Vertices follow their own input row.
    assert torch.equal(sh_v, base_v[torch.tensor(perm)])

    def coord_tri_sets(tris, cs):
        return sorted(tuple(sorted(tuple(int(v) for v in cs[i]) for i in row)) for row in tris.tolist())

    assert coord_tri_sets(base_t, coords) == coord_tri_sets(sh_t, coords[perm])
    # Degenerate (repeated-row) triangles must not appear.
    assert (base_t[:, 0] != base_t[:, 1]).all()
    assert (base_t[:, 1] != base_t[:, 2]).all()
    assert (base_t[:, 2] != base_t[:, 0]).all()


def test_shuffled_coords_matches_dictionary_oracle():
    rng = np.random.default_rng(4)
    gs = (5, 4, 5)
    coords = np.array([(x, y, z) for x in range(gs[0]) for y in range(gs[1]) for z in range(gs[2])])
    n = coords.shape[0]
    dv = rng.random((n, 3)).astype(np.float32)
    flags = rng.random((n, 3)) < 0.7
    perm = rng.permutation(n)
    o_v, o_t = _oracle(coords[perm], dv[perm], flags[perm], None, AABB_UNIT, gs)
    m_v, m_t = _call_md(_t(coords[perm], torch.int32), _t(dv[perm], torch.float32),
                        _t(flags[perm]), None, grid_size=gs)
    assert torch.equal(o_v, m_v)
    assert torch.equal(o_t, m_t)
    assert m_t.shape[0] > 0


# ---------------------------------------------------------------------------
# flags shape / dtype validation
# ---------------------------------------------------------------------------


def test_flags_shape_mismatch_rejected():
    coords = _t([[0, 0, 0], [0, 0, 1]], torch.int32)
    dv = torch.zeros(2, 3)
    for bad in (torch.zeros((2, 4), dtype=torch.bool),
                torch.zeros((2,), dtype=torch.bool),
                torch.zeros((3, 3), dtype=torch.bool),
                torch.zeros((2, 1, 3), dtype=torch.bool)):
        with pytest.raises(ValueError, match="intersected_flag"):
            _call_md(coords, dv, bad, None, grid_size=(4, 4, 4))


def test_coords_shape_mismatch_rejected():
    coords = _t(np.zeros((5, 3), dtype=int), torch.int32)
    dv = torch.zeros(3, 3)  # N mismatch
    flags = _t(np.zeros((5, 3), dtype=bool))
    with pytest.raises(ValueError, match="coords must have shape"):
        _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_float_coordinates_rejected(dtype):
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=dtype)
    flags = torch.ones((4, 3), dtype=torch.bool)
    # Both integral and fractional float values are outside the integer API.
    for values in (coords, coords - .5):
        with pytest.raises(TypeError, match="coords must have dtype"):
            _call_md(values, torch.zeros(4, 3), flags, torch.ones(4, 1), grid_size=4)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float32])
def test_non_boolean_flags_rejected(dtype):
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=torch.int32)
    with pytest.raises(TypeError, match="intersected_flag must have dtype"):
        _call_md(coords, torch.zeros(4, 3), torch.ones((4, 3), dtype=dtype), torch.ones(4, 1), grid_size=4)


def test_empty_inputs_still_require_supported_dtypes():
    with pytest.raises(TypeError, match="coords must have dtype"):
        _call_md(torch.empty((0, 3)), torch.empty((0, 3)), torch.empty((0, 3), dtype=torch.bool), grid_size=4)
    with pytest.raises(TypeError, match="intersected_flag must have dtype"):
        _call_md(torch.empty((0, 3), dtype=torch.int32), torch.empty((0, 3)), torch.empty((0, 3), dtype=torch.int32), grid_size=4)


# ---------------------------------------------------------------------------
# device / dtype contract
# ---------------------------------------------------------------------------


def test_original_device_and_dtype_preserved():
    coords = _t([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], torch.int32)
    dv = _t([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.1]], torch.float32)
    flags = _t([[True, False, False]] * 4)
    v, t = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert v.device == coords.device and t.device == coords.device
    assert v.dtype == torch.float32 and t.dtype == torch.int32

    # float64 inputs keep float64 vertices (upstream `coords.float()` contract
    # promotes to float32 only for the coordinate part; dual_vertices dtype wins)
    v64, t64 = _call_md(coords, dv.double(), flags, None, grid_size=(4, 4, 4))
    assert v64.dtype == torch.float64 and t64.dtype == torch.int32


def test_int64_coords_accepted():
    coords = _t([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], torch.int64)
    dv = _t([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.2, 0.1], [0.1, 0.2, 0.1]], torch.float32)
    flags = _t([[True, False, False]] * 4)
    v, t = _call_md(coords, dv, flags, None, grid_size=(4, 4, 4))
    assert t.shape == (2, 3)


# ---------------------------------------------------------------------------
# chunking invariance and the memory bound
# ---------------------------------------------------------------------------


def test_chunk_size_does_not_change_output():
    rng = np.random.default_rng(8)
    coords, dv, flags, gs = _random_case(rng, max_dim=6, max_n=120)
    tc, td, tf = _t(coords, torch.int32), _t(dv, torch.float32), _t(flags)
    base = _call_md(tc, td, tf, None, grid_size=gs)
    assert base[1].shape[0] > 0
    for cs in (1, 2, 3, 5, 7, 64, 10 ** 9):
        out = _call_md(tc, td, tf, None, grid_size=gs, edge_chunk_size=cs)
        assert torch.equal(out[0], base[0]), cs
        assert torch.equal(out[1], base[1]), cs


def test_chunk_size_invariance_with_split_weight():
    rng = np.random.default_rng(9)
    coords, dv, flags, gs = _random_case(rng, max_dim=6, max_n=120)
    sw = rng.random((coords.shape[0], 1)).astype(np.float32)
    tc, td, tf = _t(coords, torch.int32), _t(dv, torch.float32), _t(flags)
    base = _call_md(tc, td, tf, _t(sw), grid_size=gs)
    for cs in (1, 3, 17, 10 ** 9):
        out = _call_md(tc, td, tf, _t(sw), grid_size=gs, edge_chunk_size=cs)
        assert torch.equal(out[1], base[1]), cs


def test_sparse_lookup_is_linear_in_n_not_grid_volume():
    gs = (1024, 1024, 1024)
    coords = np.array([[i % 64, (i // 64) % 64, i // 4096] for i in range(4096)])
    lk = md.SparseVoxelLookup.from_coords(coords, gs)
    assert lk.volume == 1024 ** 3
    dense_bytes = lk.volume * 8
    assert dense_bytes / 2 ** 30 == 8.0, "dense table is 8 GiB at 1024^3"
    assert lk.nbytes() <= 4096 * 16, lk.nbytes()
    assert lk.nbytes() * 100_000 < dense_bytes


def test_no_grid_volume_allocation_at_1024():
    """Peak Python-visible allocation must stay far below the 8 GiB dense table."""
    import tracemalloc

    gs = (1024, 1024, 1024)
    rng = np.random.default_rng(4)
    n = 20000
    # A sparse surface patch: z = 511 plane only.
    coords = np.stack([rng.integers(0, 200, size=n), rng.integers(0, 200, size=n),
                       np.full(n, 511)], axis=1)
    dv = rng.random((n, 3)).astype(np.float32) * 0.5
    flags = rng.random((n, 3)) < 0.5
    tc, td, tf = _t(coords, torch.int32), _t(dv, torch.float32), _t(flags)

    tracemalloc.start()
    tracemalloc.reset_peak()
    v, t = _call_md(tc, td, tf, None, grid_size=gs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert t.shape[0] > 0
    assert peak < 256 * 2 ** 20, f"peak {peak / 2**20:.1f} MiB"
    assert peak < (1024 ** 3 * 8) / 1000, "must not approach the 8 GiB dense table"


def test_peak_rss_bound_in_subprocess():
    """End-to-end RSS check for the 1024^3 threshold case.

    The small sparse case runs in a child and reports its own peak RSS. This
    rejects a grid-volume allocation for this fixture; it is not a universal
    memory limit or evidence for a million-voxel workload.
    """
    script = textwrap.dedent(
        """
        import importlib.util, os, resource, numpy as np, torch
        module_path = %r
        spec = importlib.util.spec_from_file_location("md", module_path)
        md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
        gs = (1024, 1024, 1024)
        rng = np.random.default_rng(4)
        n = 20000
        coords = np.stack([rng.integers(0, 200, size=n), rng.integers(0, 200, size=n),
                           np.full(n, 511)], axis=1)
        dv = (rng.random((n, 3)).astype(np.float32) * 0.5)
        flags = rng.random((n, 3)) < 0.5
        v, t = md.mac_flexible_dual_grid_to_mesh(
            torch.tensor(coords, dtype=torch.int32), torch.tensor(dv), torch.tensor(flags),
            None, aabb=[[-0.5]*3, [0.5]*3], grid_size=gs)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert t.shape[0] > 0, "expected quads"
        print(int(peak - base), int(peak))
        """
    ) % MODULE_PATH
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env=env, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    delta_bytes, peak_bytes = (int(v) for v in out.stdout.strip().splitlines()[-1].split())
    # ru_maxrss is bytes on macOS; the dense table would need 8 GiB, so a
    # generous 512 MiB allowance detects that allocation in this small fixture.
    assert delta_bytes < 512 * 2 ** 20, f"RSS grew {delta_bytes / 2**20:.1f} MiB"
    assert peak_bytes < 4 * 2 ** 30, f"peak RSS {peak_bytes / 2**30:.2f} GiB"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_smoke():
    script = textwrap.dedent(
        """
        import importlib.util, os, numpy as np, torch
        module_path = %r
        spec = importlib.util.spec_from_file_location("md", module_path)
        md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
        coords = torch.tensor([[0,0,0],[0,0,1],[0,1,1],[0,1,0]], dtype=torch.int32)
        dv = torch.tensor([[0.1,0.1,0.1],[0.2,0.1,0.1],[0.2,0.2,0.1],[0.1,0.2,0.1]])
        flags = torch.tensor([[True,False,False]]*4)
        v, t = md.mac_flexible_dual_grid_to_mesh(coords, dv, flags, None,
                                                aabb=[[-0.5]*3,[0.5]*3], grid_size=(4,4,4))
        print("OK", tuple(v.shape), tuple(t.shape), t.tolist())
        """
    ) % MODULE_PATH
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("OK (4, 3) (2, 3)")


# ---------------------------------------------------------------------------
# Decoder wiring: execute only the dispatch helper, with import spies.
# No decoder/model construction, accelerator dependency or device allocation.
# ---------------------------------------------------------------------------


def _decoder_source():
    with open(DECODER_PATH, encoding="utf-8") as source:
        return ast.parse(source.read(), filename=DECODER_PATH)


def _decoder_dispatch():
    tree = _decoder_source()
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_flexible_dual_grid_to_mesh")
    namespace = {"torch": torch, "__package__": "uniforge3d.models.vaes"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), DECODER_PATH, "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_decoder_dispatch_selects_only_requested_backend_and_forwards_arguments(monkeypatch, device_type):
    dispatch = _decoder_dispatch()
    coords = SimpleNamespace(device=SimpleNamespace(type=device_type))
    values = (object(), object(), object())
    options = {"aabb": [[-.5] * 3, [.5] * 3], "grid_size": 1024, "train": False}
    result = object()
    calls = []
    def extract(*args, **kwargs):
        assert args == (coords, *values)
        assert kwargs == options
        return result
    def selected_import(name, globals=None, locals=None, fromlist=(), level=0):
        calls.append((name, tuple(fromlist), level))
        if device_type == "mps":
            assert name == "utils.mac_dual_grid" and level == 3
            return SimpleNamespace(mac_flexible_dual_grid_to_mesh=extract)
        assert name == "o_voxel.convert" and level == 0
        return SimpleNamespace(flexible_dual_grid_to_mesh=extract)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", selected_import)
        assert dispatch(coords, *values, **options) is result
    assert len(calls) == 1


@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_decoder_dispatch_propagates_missing_dependency_without_fallback(monkeypatch, device_type):
    dispatch = _decoder_dispatch()
    coords = SimpleNamespace(device=SimpleNamespace(type=device_type))
    calls = []
    def unavailable(name, globals=None, locals=None, fromlist=(), level=0):
        calls.append(name)
        raise ImportError("selected extraction dependency unavailable")
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", unavailable)
        with pytest.raises(ImportError, match="selected extraction dependency unavailable"):
            dispatch(coords, object(), object(), object(), grid_size=1024, train=False)
    assert calls == ["utils.mac_dual_grid" if device_type == "mps" else "o_voxel.convert"]


def test_decoder_import_is_lazy_and_forward_keeps_existing_extraction_contract():
    tree = _decoder_source()
    # No original extraction dependency may be imported while defining the module.
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("o_voxel")
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("o_voxel") for alias in node.names)
    decoder = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "FlexiDualGridVaeDecoder")
    forward = next(node for node in decoder.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_flexible_dual_grid_to_mesh"]
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args] == ["v.coords[:, 1:]", "v.feats", "i.feats", "q.feats"]
    options = {kw.arg: kw.value for kw in calls[0].keywords}
    assert set(options) == {"aabb", "grid_size", "train"}
    assert ast.literal_eval(options["aabb"]) == [[-.5] * 3, [.5] * 3]
    assert ast.unparse(options["grid_size"]) == "self.resolution"
    assert ast.literal_eval(options["train"]) is False
