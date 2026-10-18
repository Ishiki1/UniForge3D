"""CPU exactness tests; the guarded native proof is separate and never implicit."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


router = load('tested_view_router', ROOT / 'uniforge3d/models/view_router.py')
original_path = ROOT / 'tests/fixtures/upstream_view_router.py'
assert hashlib.sha256(original_path.read_bytes()).hexdigest() == 'a98e4824c004362664d78042d5ff12b00564b7c85eb9c2669b6c6e7f10a4a6aa'
original = load('original_view_router', original_path)
fixtures = load('view_router_cases', ROOT / 'tests/fixtures/view_router_cases.py')


@pytest.mark.parametrize('reduce', ['amin', 'amax'])
@pytest.mark.parametrize('strided', [False, True])
def test_integer_boundary_exact_full_width_and_in_place(reduce, strided):
    initial, indexes, values, expected = fixtures.integer_case(reduce)

    def tensor(sequence):
        if not strided:
            return torch.tensor(sequence, dtype=torch.int64)
        backing = torch.full((len(sequence) * 2,), 911, dtype=torch.int64)
        result = backing[::2]
        result.copy_(torch.tensor(sequence, dtype=torch.int64))
        return result

    destination, index, source = map(tensor, (initial, indexes, values))
    before_index, before_source = index.clone(), source.clone()
    assert router._scatter_reduce_int64_cpu_(destination, index, source, reduce=reduce) is destination
    assert destination.dtype == torch.int64 and destination.tolist() == expected
    assert torch.equal(index, before_index) and torch.equal(source, before_source)
    if strided:
        assert destination._base[1::2].tolist() == [911] * len(initial)


@pytest.mark.parametrize('reduce', ['amin', 'amax'])
def test_empty_preserves_initialized_destination(reduce):
    destination = torch.tensor([-(2**63), 2**63 - 1, 17], dtype=torch.int64)
    original = destination.clone()
    empty = torch.empty(0, dtype=torch.int64)
    assert router._scatter_reduce_int64_cpu_(destination, empty, empty, reduce=reduce) is destination
    assert torch.equal(destination, original)


@pytest.mark.parametrize('dtype', [torch.int64, torch.int32, torch.float32])
@pytest.mark.parametrize('reduce', ['amin', 'amax'])
def test_cpu_dispatch_is_original_native_operation(dtype, reduce, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU native dispatch unexpectedly used the MPS boundary')
    monkeypatch.setattr(router, '_scatter_reduce_int64_cpu_', forbidden)
    destination = torch.tensor([9, -11, 29], dtype=dtype)
    source = torch.tensor([10, -12, 10], dtype=dtype)
    index = torch.tensor([0, 1, 0], dtype=torch.int64)
    expected = destination.clone().scatter_reduce_(0, index, source, reduce=reduce, include_self=True)
    assert router._scatter_reduce_extrema_(destination, index, source, reduce=reduce) is destination
    assert torch.equal(destination, expected)


@pytest.mark.parametrize('grid', [32, 64])
@pytest.mark.parametrize('case_name,keep', [('router_case', 1.), ('confidence_case', .95), ('neighbor_offsets_case', 1.)])
def test_full_router_frozen_source_and_independent_winners(grid, case_name, keep, monkeypatch):
    values, expected = getattr(fixtures, case_name)(grid)
    kwargs = dict(confidence_keep_ratio=keep, grid_resolution=grid, patch_grid_size=grid)
    untouched = original.GPUVoxelViewRouter(**kwargs)(*values)['selected_view']
    assert torch.equal(untouched, expected), 'Frozen original must match analytic fixture'
    native_cpu = router.GPUVoxelViewRouter(**kwargs)(*values)['selected_view']
    assert torch.equal(native_cpu, untouched)
    # Exercise every CPU boundary operation, even though the test owns no MPS device.
    calls = []
    def force_boundary(destination, index, source, *, reduce):
        calls.append((reduce, destination.dtype))
        return router._scatter_reduce_int64_cpu_(destination, index, source, reduce=reduce)
    monkeypatch.setattr(router, '_scatter_reduce_extrema_', force_boundary)
    unique_calls = []
    def force_unique(key):
        unique_calls.append(key.clone())
        return router._unique_consecutive_int64_cpu(key)
    monkeypatch.setattr(router, '_unique_consecutive_packed_keys', force_unique)
    boundary = router.GPUVoxelViewRouter(**kwargs)(*values)['selected_view']
    assert torch.equal(boundary, untouched)
    assert calls == [(r, torch.int64) for r in ['amax', 'amin', 'amax', 'amax', 'amin']]
    assert len(unique_calls) == 1 and unique_calls[0].dtype == torch.int64


@pytest.mark.parametrize('grid', [32, 64])
def test_all_empty_and_invalid_target_contract(grid):
    values, _ = fixtures.confidence_case(grid)
    values[2].zero_()
    kwargs = dict(grid_resolution=grid, patch_grid_size=grid)
    expected = original.GPUVoxelViewRouter(**kwargs)(*values)['selected_view']
    actual = router.GPUVoxelViewRouter(**kwargs)(*values)['selected_view']
    assert torch.equal(actual, expected) and not actual.any()
    values, _ = fixtures.confidence_case(grid)
    values = (*values[:3], torch.tensor([[0, grid, 0, 0]], dtype=torch.int32))
    for module in (original, router):
        with pytest.raises(ValueError, match='outside'):
            module.GPUVoxelViewRouter(**kwargs)(*values)


def test_component_root_tie_keeps_lowest_index(monkeypatch):
    # Two disconnected roots with identical support and confidence. The returned
    # summary alone cannot expose this tie, so observe the actual integer result.
    entries = dict(key=torch.tensor([0, 4]), group=torch.tensor([0, 0]),
                   patch=torch.tensor([0, 4]), count=torch.tensor([2, 2]),
                   conf=torch.tensor([1., 1.]))
    outputs = []
    def observe(destination, index, source, *, reduce):
        result = router._scatter_reduce_int64_cpu_(destination, index, source, reduce=reduce)
        outputs.append((reduce, result.tolist()))
        return result
    monkeypatch.setattr(router, '_scatter_reduce_extrema_', observe)
    summary = router.GPUVoxelViewRouter(grid_resolution=32, patch_grid_size=32)._summaries(entries)
    assert outputs == [('amax', [2]), ('amin', [0])]
    assert summary['support'].tolist() == [2] and summary['mean'].tolist() == [.5]


def python_runs(sequence):
    values, inverse, counts = [], [], []
    for item in sequence:
        if not values or item != values[-1]:
            values.append(item)
            counts.append(0)
        inverse.append(len(values) - 1)
        counts[-1] += 1
    return values, inverse, counts


UNIQUE_CASES = [
    [],
    [2**63 - 1],
    [-(2**63), -(2**63), -1, 0, 2**63 - 1, 2**63 - 1],
    [0, 2**32, 2**32, 1, 2**32 + 1, 2**32 + 1],
    [2**53, 2**53 + 1, 2**53 + 1, 2**53, 2**53 + 2],
    [-(2**53) - 1, -(2**53), -(2**31) - 1, -(2**31), -1, -1],
]


@pytest.mark.parametrize('sequence', UNIQUE_CASES)
@pytest.mark.parametrize('strided', [False, True])
def test_unique_int64_boundary_python_run_oracle(sequence, strided):
    initial = torch.tensor(sequence, dtype=torch.int64)
    if strided:
        backing = torch.full((2 * len(sequence),), 177, dtype=torch.int64)
        key = backing[::2]
        key.copy_(initial)
    else:
        key = initial.clone()
    result = router._unique_consecutive_int64_cpu(key)
    assert isinstance(result, tuple) and len(result) == 3
    for actual, expected in zip(result, python_runs(sequence)):
        assert actual.dtype == torch.int64 and actual.device == key.device
        assert actual.tolist() == expected
    assert torch.equal(key, initial)
    if strided:
        assert backing[1::2].tolist() == [177] * len(sequence)


@pytest.mark.parametrize('dtype', [torch.int64, torch.int32, torch.float32])
def test_cpu_unique_dispatch_preserves_original_native(dtype, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU unique unexpectedly used the MPS boundary')
    monkeypatch.setattr(router, '_unique_consecutive_int64_cpu', forbidden)
    key = torch.tensor([1, 1, -2, -2, 0, 1], dtype=dtype)
    expected = torch.unique_consecutive(key, return_inverse=True, return_counts=True)
    actual = router._unique_consecutive_packed_keys(key)
    assert all(torch.equal(a, b) and a.dtype == b.dtype for a, b in zip(actual, expected))
