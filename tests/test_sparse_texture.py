"""CPU oracle for the explicit Mac bake sampler; no native libraries imported."""
from itertools import product
import math

import numpy as np
import pytest
import torch

from uniforge3d.utils.sparse_texture import SparseTextureSampler


def oracle(features, coords, points):
    table = {tuple(c): f for c, f in zip(coords.tolist(), features.float().numpy())}
    result = []
    for point in points.float().tolist():
        base = [math.floor(x-0.5) for x in point]
        total = np.zeros(features.shape[1], dtype=np.float64)
        denominator = 0.0
        for delta in product((0, 1), repeat=3):
            key = tuple(b+d for b, d in zip(base, delta))
            if key not in table:
                continue
            weight = math.prod(1-abs(q-k-0.5) for q,k in zip(point,key))
            total += table[key]*weight
            denominator += weight
        result.append(total/max(denominator, 1e-12))
    return torch.tensor(np.asarray(result), dtype=torch.float32)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('chunk', [1, 7, 65536])
def test_ramps_missing_neighbours_shuffled_coordinates(dtype, chunk):
    coords = torch.tensor(list(product(range(4), range(3), range(5))), dtype=torch.int32)[::2]
    coords = coords[torch.randperm(len(coords), generator=torch.Generator().manual_seed(29))]
    x,y,z = coords.float().T
    features = torch.stack([x/4,y/3,z/5,(x+2*y)/10,z/8+0.1,torch.ones_like(x)],dim=1).to(dtype)
    points = torch.rand((71,3), generator=torch.Generator().manual_seed(93))*torch.tensor([5,4,6])-0.5
    sampler = SparseTextureSampler(features,coords,[4,3,5],chunk_size=chunk)
    actual = sampler(points)
    assert actual.dtype == torch.float32 and actual.device.type == 'cpu'
    torch.testing.assert_close(actual,oracle(features,coords,points),rtol=2e-6,atol=2e-6)


def test_centres_boundaries_zero_support_and_tiny_positive_weight():
    coords = torch.tensor([[0,0,0],[1023,1023,1023]],dtype=torch.int32)
    features = torch.tensor([[0.2,0.4,0.6,0.8,0.1,1],[1,0,0,0,0.5,1]])
    points = torch.tensor([[0.5,0.5,0.5],[-0.49999]*3,[-0.5]*3,[1023.5]*3,[512.0]*3,[1e30]*3])
    sampler = SparseTextureSampler(features,coords,[1024]*3,chunk_size=2)
    actual = sampler(points)
    torch.testing.assert_close(actual,oracle(features,coords,points),rtol=2e-6,atol=1e-8)
    assert 0 < actual[1,0] < features[0,0]  # Native Metal threshold bug would return zero.
    assert torch.count_nonzero(actual[[2,4,5]]) == 0
    assert sampler.keys.nbytes == 16  # Sparse grid, not 1024 cubed.


def test_empty_inputs():
    sampler=SparseTextureSampler(torch.empty(0,6),torch.empty(0,3,dtype=torch.int32),[1024]*3)
    assert sampler(torch.ones(2,3)).shape == (2,6)
    assert sampler(torch.empty(0,3)).shape == (0,6)
    with pytest.raises(ValueError,match='nonfinite'):
        sampler(torch.full((1,3),float('nan')))


@pytest.mark.parametrize('coords', [[[1,1,1],[1,1,1]], [[0,0,0],[4,0,0]], [[0,0,0],[-1,0,0]]])
def test_invalid_coords(coords):
    with pytest.raises(ValueError):
        SparseTextureSampler(torch.ones(2,6),torch.tensor(coords,dtype=torch.int32),[4]*3)


def test_nonfinite_and_chunk_contracts():
    for chunk in [0,-1,1.5,True]:
        with pytest.raises(ValueError,match='chunk_size'):
            SparseTextureSampler(torch.ones(1,6),torch.zeros(1,3,dtype=torch.int32),[4]*3,chunk_size=chunk)
    with pytest.raises(ValueError,match='nonfinite'):
        SparseTextureSampler(torch.full((1,6),float('nan')),torch.zeros(1,3,dtype=torch.int32),[4]*3)
    sampler=SparseTextureSampler(torch.ones(1,6),torch.zeros(1,3,dtype=torch.int32),[4]*3)
    with pytest.raises(ValueError,match='nonfinite'):
        sampler(torch.full((1,3),float('inf')))
