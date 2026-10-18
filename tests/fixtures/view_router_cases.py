"""Analytic router fixtures shared by CPU tests and the guarded native probe."""
import itertools

import torch


def integer_case(reduce):
    # Adjacent integers above 2**53 detect both float64 and int32 narrowing.
    low, high = -(2**63), 2**63 - 1
    initial = [2**53 + 3, -(2**53) - 3, 17, -21, high, low, 29]
    indexes = [0, 0, 1, 1, 2, 2, 3, 4, 5, 5]
    values = [2**53 + 2, 2**53 + 4, -(2**53) - 2, -(2**53) - 4,
              17, 17, -22, high - 1, low + 1, low + 2]
    expected = initial.copy()
    operation = min if reduce == 'amin' else max
    for index, value in zip(indexes, values):
        expected[index] = operation(expected[index], value)
    return initial, indexes, values, expected


def router_case(grid):
    """Two batches/four views, exact dyadic sums, independently assigned winners."""
    points = torch.zeros((2, 4, 448, 448, 3), dtype=torch.float32)
    confidence = torch.zeros((2, 4, 448, 448), dtype=torch.float32)
    masks = torch.zeros((2, 4, 448, 448), dtype=torch.bool)
    expected = torch.zeros((2, grid, grid, grid), dtype=torch.int64)
    cursor = {(b, v): grid * 4 for b in range(2) for v in range(4)}
    occupied = set()
    patch_pixels = 448 // grid

    def add(voxel, view, count, conf, *, batch=0, patch=None, point=None, valid=True):
        if patch is None:
            patch = cursor[batch, view]
            cursor[batch, view] += 2  # Disconnected unless explicitly made adjacent.
        py, px = divmod(patch, grid)
        assert 0 <= py < grid and count < patch_pixels * patch_pixels
        for n in range(count):
            y = py * patch_pixels + n // patch_pixels
            x = px * patch_pixels + n % patch_pixels
            assert (batch, view, y, x) not in occupied
            occupied.add((batch, view, y, x))
            points[batch, view, y, x] = torch.tensor(
                point if point is not None else [(axis + .5) / grid - .5 for axis in voxel])
            confidence[batch, view, y, x] = conf
            masks[batch, view, y, x] = valid

    # Direct priority: support, anchor, confidence, then lowest non-anchor view.
    a, b, c, d = (2, 2, 2), (6, 2, 2), (10, 2, 2), (14, 2, 2)
    add(a, 1, 3, .25); add(a, 2, 2, 1.); expected[0, *a] = 1
    add(b, 0, 2, .125); add(b, 3, 2, 1.)  # Anchor wins regardless of confidence.
    add(c, 1, 2, .25); add(c, 2, 2, .75); expected[0, *c] = 2
    add(d, 2, 2, .5); add(d, 3, 2, .5); expected[0, *d] = 2
    # A view's support is its best connected component, not all its pixels.
    e, f = (18, 2, 2), (22, 2, 2)
    add(e, 1, 2, 1.); add(e, 1, 2, 1.); add(e, 2, 3, .25); expected[0, *e] = 2
    # Equal support components must choose the greater mean confidence.
    add(f, 1, 2, .25); add(f, 1, 2, .75); add(f, 2, 2, .5); expected[0, *f] = 1
    # Connected adjacent patches combine support (8-connected, including diagonal).
    j = (18, 10, 10)
    add(j, 1, 1, .25, patch=8); add(j, 1, 1, .25, patch=grid + 9)
    add(j, 2, 1, 1., patch=8); expected[0, *j] = 1
    # A missing target gets direct neighbor support only; no two-hop propagation.
    g = (26, 2, 2)
    add(g, 3, 1, .5); expected[0, *g] = 3; expected[0, 27, 2, 2] = 3
    targets = [(0, 27, 2, 2), (0, 28, 2, 2)]
    # Each neighbor priority gets its own isolated pair of directly supported voxels.
    for center, left_view, right_view, left_count, right_count, left_conf, right_conf, winner in (
        ((2, 10, 10), 1, 2, 3, 2, .25, 1., 1),
        ((8, 10, 10), 0, 3, 2, 2, .125, 1., 0),
        ((14, 10, 10), 1, 2, 2, 2, .25, .75, 2),
        ((24, 10, 10), 2, 3, 2, 2, .5, .5, 2),
    ):
        left = (center[0] - 1, *center[1:]); right = (center[0] + 1, *center[1:])
        add(left, left_view, left_count, left_conf)
        add(right, right_view, right_count, right_conf)
        expected[0, *left] = left_view; expected[0, *right] = right_view
        expected[0, *center] = winner; targets.append((0, *center))
    # Lower domain boundary is included; upper boundary, NaN and masked pixels excluded.
    add((0, 0, 0), 2, 1, .5, point=[-.5, -.5, -.5]); expected[0, 0, 0, 0] = 2
    add((0, 0, 0), 3, 1, .5, point=[.5, 0., 0.])
    add((0, 0, 0), 3, 1, .5, point=[float('nan'), 0., 0.])
    add((0, 0, 0), 3, 1, float('nan'))
    add((0, 0, 0), 3, 1, .5, valid=False)
    # Batch identity and duplicate/unsorted targets must not change the dense result.
    add(a, 3, 2, .5, batch=1); expected[1, *a] = 3
    expected[0, 0, 0, 1] = 2
    targets += [(1, *a), (0, *a), (0, 27, 2, 2), (1, 31, 31, 31), (0, 0, 0, 1)]
    return (points, confidence, masks, torch.tensor(targets[::-1], dtype=torch.int32)), expected


def confidence_case(grid):
    """ceil(21 * .95) = 20; stable equal-confidence sort keeps first 20 pixels."""
    points = torch.zeros((1, 4, 448, 448, 3), dtype=torch.float32)
    confidence = torch.zeros((1, 4, 448, 448), dtype=torch.float32)
    masks = torch.zeros((1, 4, 448, 448), dtype=torch.bool)
    expected = torch.zeros((1, grid, grid, grid), dtype=torch.int64)
    for index in range(21):
        voxel = (index, 20, 20)
        points[0, 2, 0, index] = torch.tensor([(axis + .5) / grid - .5 for axis in voxel])
        confidence[0, 2, 0, index] = .5
        masks[0, 2, 0, index] = True
        if index < 20:
            expected[0, *voxel] = 2
    # Empty target set prevents intentionally dropped direct support being inherited.
    return (points, confidence, masks, torch.empty((0, 4), dtype=torch.int32)), expected


def neighbor_offsets_case(grid):
    """Every one of the 26 offsets independently reaches a missing target."""
    points = torch.zeros((1, 4, 448, 448, 3), dtype=torch.float32)
    confidence = torch.zeros((1, 4, 448, 448), dtype=torch.float32)
    masks = torch.zeros((1, 4, 448, 448), dtype=torch.bool)
    expected = torch.zeros((1, grid, grid, grid), dtype=torch.int64)
    offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]
    centers = list(itertools.product((4, 12, 20), repeat=3))
    targets = []
    for index, (offset, center) in enumerate(zip(offsets, centers)):
        source = tuple(axis + delta for axis, delta in zip(center, offset))
        view = 1 + index % 3
        points[0, view, 0, index] = torch.tensor([(axis + .5) / grid - .5 for axis in source])
        confidence[0, view, 0, index] = .5
        masks[0, view, 0, index] = True
        expected[0, *source] = view
        expected[0, *center] = view
        targets.append((0, *center))
    return (points, confidence, masks, torch.tensor(targets, dtype=torch.int32)), expected
