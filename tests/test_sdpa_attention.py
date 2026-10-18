import math
import os
import subprocess
import sys

import pytest
import torch

from uniforge3d.modules import sparse as sp
from uniforge3d.modules.sparse import config
from uniforge3d.modules.sparse.attention.sdpa import (
    packed_sdpa, query_tile_size, validate_cumulative_lengths,
)


def oracle(q, k, v):
    """Independent CPU float64 attention, no SDPA or attention implementation."""
    qh, kh, vh = [t.detach().double().transpose(0, 1) for t in (q, k, v)]
    scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    weights = (scores - scores.amax(-1, keepdim=True)).exp()
    weights = weights / weights.sum(-1, keepdim=True)
    return (weights @ vh).transpose(0, 1)


def ragged(feats, lengths, sparse=False):
    layout = sp.VarLenTensor.layout_from_seqlen(lengths)
    if not sparse:
        return sp.VarLenTensor(feats, layout)
    coords = torch.zeros((len(feats), 4), dtype=torch.int32)
    for b, s in enumerate(layout):
        coords[s, 0] = b
        coords[s, 1] = torch.arange(s.stop - s.start)
    x = sp.SparseTensor(feats, coords, shape=torch.Size([len(lengths), *feats.shape[1:]]))
    x.register_spatial_cache('layout', layout)
    x.register_spatial_cache('sentinel', object())
    return x


def reference_batches(q, k, v, qlens, klens):
    out = []
    qi = ki = 0
    for qn, kn in zip(qlens, klens):
        if qn:
            out.append(oracle(q[qi:qi+qn], k[ki:ki+kn], v[ki:ki+kn]))
        qi += qn
        ki += kn
    return torch.cat(out) if out else torch.empty((0, q.shape[1], v.shape[2]), dtype=torch.float64)


def compare(actual, expected, atol=2e-6, rtol=2e-5):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('style', ['positional', 'keyword'])
def test_public_qkv(sparse, style, monkeypatch):
    torch.manual_seed(101)
    monkeypatch.setattr(config, 'SDPA_QUERY_CHUNK_SIZE', 2)
    qkv = torch.randn(8, 3, 2, 7)
    x = ragged(qkv, [3, 0, 5], sparse)
    out = sp.sparse_scaled_dot_product_attention(x) if style == 'positional' else sp.sparse_scaled_dot_product_attention(qkv=x)
    compare(out.feats, reference_batches(*qkv.unbind(1), [3, 0, 5], [3, 0, 5]))
    assert type(out) is type(x) and out.dtype == x.dtype and out.device == x.device
    assert out.layout == x.layout
    if sparse:
        assert out.coords is x.coords
        assert out._spatial_cache is x._spatial_cache
    else:
        assert out._cache is x._cache


@pytest.mark.parametrize('mode', ['ragged_ragged', 'ragged_dense', 'dense_ragged'])
@pytest.mark.parametrize('arity', [2, 3])
@pytest.mark.parametrize('style', ['positional', 'mixed', 'keyword'])
def test_public_cross_overloads(mode, arity, style, monkeypatch):
    torch.manual_seed(23)
    monkeypatch.setattr(config, 'SDPA_QUERY_CHUNK_SIZE', 2)
    qlens = [2, 2, 2] if mode == 'dense_ragged' else [4, 0, 2]
    klens = [3, 3, 3] if mode == 'ragged_dense' else [1, 3, 6]
    q = torch.randn(sum(qlens), 2, 5)
    k = torch.randn(sum(klens), 2, 5)
    v = torch.randn(sum(klens), 2, 5 if arity == 2 else 3)
    original_q = q.reshape(3, 2, 2, 5) if mode == 'dense_ragged' else ragged(q, qlens, True)
    if arity == 2:
        kv = torch.stack((k, v), 1)
        packed = kv.reshape(3, 3, 2, 2, 5) if mode == 'ragged_dense' else ragged(kv, klens, True)
        values, names = [original_q, packed], ['q', 'kv']
    else:
        packed_k = k.reshape(3, 3, 2, 5) if mode == 'ragged_dense' else ragged(k, klens, True)
        packed_v = v.reshape(3, 3, 2, 3) if mode == 'ragged_dense' else ragged(v, klens, True)
        values, names = [original_q, packed_k, packed_v], ['q', 'k', 'v']
    split = {'positional': arity, 'mixed': 1, 'keyword': 0}[style]
    out = sp.sparse_scaled_dot_product_attention(*values[:split], **dict(zip(names[split:], values[split:])))
    data = out if isinstance(out, torch.Tensor) else out.feats
    compare(data.reshape(sum(qlens), 2, v.shape[-1]), reference_batches(q, k, v, qlens, klens))
    if mode == 'dense_ragged':
        assert tuple(out.shape) == (3, 2, 2, v.shape[-1])
    else:
        assert out.coords is original_q.coords and out._spatial_cache is original_q._spatial_cache


@pytest.mark.parametrize('qlens,klens', [([0, 2, 0], [0, 3, 5]), ([0, 0], [0, 0]), ([], [])])
def test_empty_sequences(qlens, klens):
    q, k, v = torch.randn(sum(qlens), 2, 3), torch.randn(sum(klens), 2, 3), torch.randn(sum(klens), 2, 4)
    out = sp.sparse_scaled_dot_product_attention(ragged(q, qlens), ragged(k, klens), ragged(v, klens))
    assert out.feats.shape == (sum(qlens), 2, 4)
    compare(out.feats, reference_batches(q, k, v, qlens, klens))


def test_empty_keys_rejected():
    with pytest.raises(ValueError, match='no keys'):
        sp.sparse_scaled_dot_product_attention(ragged(torch.randn(1, 2, 3), [1]), ragged(torch.empty(0, 2, 2, 3), [0]))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_noncontiguous_chunk_boundaries_all_keys(dtype, monkeypatch):
    torch.manual_seed(44)
    q = torch.randn(33, 3, 16)[..., ::2].to(dtype)
    k = torch.randn(41, 3, 16)[..., ::2].to(dtype)
    v = torch.randn(41, 3, 10)[..., ::2].to(dtype)
    # dtype conversion can make a view contiguous: create the strided final views.
    q, k, v = [torch.stack((x, x), -1)[..., 0] for x in (q, k, v)]
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    expected = oracle(q, k, v)
    import uniforge3d.modules.sparse.attention.sdpa as helper
    calls = []
    original = helper.F.scaled_dot_product_attention
    def recording(q, k, v, **kwargs):
        calls.append((q.shape[-2], k.shape[-2]))
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(helper.F, 'scaled_dot_product_attention', recording)
    for tile in (1, 15, 16, 17, 31, 32, 33, 100):
        calls.clear()
        monkeypatch.setattr(config, 'SDPA_QUERY_CHUNK_SIZE', tile)
        out = sp.sparse_scaled_dot_product_attention(ragged(q, [33]), ragged(k, [41]), ragged(v, [41])).feats
        tol = {torch.float16: 1e-3, torch.bfloat16: 8e-3, torch.float32: 2e-6, torch.float64: 2e-14}[dtype]
        compare(out, expected, atol=tol, rtol=tol)
        assert out.dtype == dtype
        assert sum(a for a, b in calls) == 33
        assert all(a <= tile and b == 41 for a, b in calls)


def test_exact_attention_invariants():
    torch.manual_seed(45)
    q, k, v = torch.zeros(3, 2, 5), torch.zeros(17, 2, 5), torch.randn(17, 2, 4)
    def call(q, k, v):
        return sp.sparse_scaled_dot_product_attention(ragged(q, [len(q)]), ragged(k, [len(k)]), ragged(v, [len(v)])).feats
    compare(call(q, k, v), v.double().mean(0).expand(3, -1, -1))
    compare(call(q, k[:1], v[:1]), v[:1].double().expand(3, -1, -1))
    q, k = torch.randn_like(q) * 1000, torch.randn_like(k) * 1000
    compare(call(q, k, v), oracle(q, k, v))
    perm = torch.randperm(17)
    compare(call(q, k[perm], v[perm]), oracle(q, k, v))


@pytest.mark.parametrize('bad', [[1, 2], [0, 3], [0, 2, 1, 2], []])
def test_malformed_cumulative_lengths(bad):
    with pytest.raises(ValueError):
        validate_cumulative_lengths(torch.tensor(bad, dtype=torch.int64), 2, name='bad', device=torch.device('cpu'))


@pytest.mark.parametrize('layout', [[slice(1, 3)], [slice(0, 1), slice(2, 3)], [slice(0, 2), slice(1, 3)], [slice(0, 3, 2)], [slice(0, 4)], [slice(0, 2)], [slice(0, 3.0)]])
def test_malformed_public_layout(layout):
    x = sp.VarLenTensor(torch.randn(3, 3, 2, 4), layout)
    with pytest.raises(ValueError, match='layout'):
        sp.sparse_scaled_dot_product_attention(x)


def test_sparse_coordinate_and_kv_layout_validation():
    q = ragged(torch.randn(3, 2, 4), [1, 2], True)
    kv = ragged(torch.randn(3, 2, 2, 4), [1, 2], True)
    q.coords[0, 0] = 1
    with pytest.raises(ValueError, match='batch coordinates'):
        sp.sparse_scaled_dot_product_attention(q, kv)
    q.coords[0, 0] = 0
    k, v = kv.unbind(1)
    v = v.replace(v.feats, v.coords.clone())
    v.coords[0, 1] = 10
    with pytest.raises(ValueError, match='coordinates must match'):
        sp.sparse_scaled_dot_product_attention(q, k, v)
    k, v = ragged(k.feats, [1, 2]), ragged(v.feats, [2, 1])
    with pytest.raises(ValueError, match='identical sequence layouts'):
        sp.sparse_scaled_dot_product_attention(q, k, v)


@pytest.mark.parametrize('kind', ['dtype', 'heads', 'width', 'tokens', 'batch', 'integer', 'fused_axis'])
def test_public_shape_type_errors(kind):
    q = ragged(torch.randn(3, 2, 4), [3])
    kv = ragged(torch.randn(5, 2, 2, 4), [5])
    if kind == 'dtype': kv = kv.double() if hasattr(kv, 'double') else kv.to(torch.float64)
    elif kind == 'heads': kv = ragged(torch.randn(5, 2, 3, 4), [5])
    elif kind == 'width': kv = ragged(torch.randn(5, 2, 2, 3), [5])
    elif kind == 'batch': kv = ragged(kv.feats, [2, 3])
    elif kind == 'integer': q = q.to(torch.int32)
    elif kind == 'fused_axis': kv = ragged(torch.randn(5, 3, 2, 4), [5])
    elif kind == 'tokens':
        with pytest.raises(ValueError):
            sp.sparse_scaled_dot_product_attention(q, torch.randn(1, 5, 2, 4), torch.randn(1, 4, 2, 3))
        return
    with pytest.raises(ValueError):
        sp.sparse_scaled_dot_product_attention(q, kv)


def test_signature_and_dense_only_rejections():
    x = ragged(torch.randn(2, 2, 4), [2])
    for args, kwargs in [((x,), {'q': x}), ((), {'hello': x}), ((x, x, x, x), {}), ((torch.randn(1, 2, 2, 4), torch.randn(1, 3, 2, 2, 4)), {})]:
        with pytest.raises(TypeError):
            sp.sparse_scaled_dot_product_attention(*args, **kwargs)


def test_score_workspace_estimate_forces_tile_and_rejects_one_row(monkeypatch):
    # Deliberately derive the independent budget from the documented formula.
    nk, h, dq, dv = 4101, 12, 128, 128
    fixed = 1024**2 + 2 * h * nk * (dq + dv) * 4
    per_row = h * (3 * nk + 2 * dq + 2 * dv) * 4
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        assert query_tile_size(nk, h, dq, dv, dtype, score_workspace_bytes=fixed+17*per_row, query_chunk_size=256) == 17
        with pytest.raises(ValueError, match='cannot fit one query row'):
            query_tile_size(nk, h, dq, dv, dtype, score_workspace_bytes=fixed+per_row-1, query_chunk_size=256)
    monkeypatch.setattr(config, 'SDPA_SCORE_WORKSPACE_BYTES', 1)
    q = ragged(torch.randn(2, 2, 4), [2])
    with pytest.raises(ValueError, match='not total RSS'):
        sp.sparse_scaled_dot_product_attention(q, q, q)


def test_configuration_and_lazy_import_in_clean_processes():
    env = {k: v for k, v in os.environ.items() if k not in ('ATTN_BACKEND', 'SPARSE_ATTN_BACKEND')}
    script = '''
import sys
import uniforge3d
from uniforge3d.modules.sparse import config
assert config.ATTN == 'flash_attn'
assert 'uniforge3d.pipelines' not in sys.modules
assert not any(k.split('.')[0] in ('flash_attn','flex_gemm','nvdiffrast') for k in sys.modules)
print('default=flash_attn; pipeline and CUDA extensions not imported')
'''
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout.strip())
    script = '''
import sys, importlib.util
from uniforge3d.modules import sparse as sp
from uniforge3d.modules.sparse.attention.routed_full_only_flash import RoutedFullOnlyFlashCrossAttention
assert sp.config.ATTN == 'sdpa'
assert importlib.util.find_spec('flash_attn') is None
layer = RoutedFullOnlyFlashCrossAttention(1536, 4, 12)
assert not any(k.split('.')[0] in ('flash_attn','flex_gemm','nvdiffrast') for k in sys.modules)
print('real public API and routed class imported/constructed with FlashAttention absent')
'''
    result = subprocess.run([sys.executable, '-c', script], env={**env, 'ATTN_BACKEND': 'sdpa'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout.strip())
    for overrides in ({'SPARSE_ATTN_BACKEND': 'typo'}, {'ATTN_BACKEND': 'typo'}, {'SPARSE_SDPA_SCORE_WORKSPACE_BYTES': '0'}, {'SPARSE_SDPA_QUERY_CHUNK_SIZE': 'no'}):
        result = subprocess.run([sys.executable, '-c', 'from uniforge3d.modules.sparse import config'], env={**env, **overrides}, capture_output=True, text=True)
        assert result.returncode != 0 and 'ValueError' in result.stderr
    result = subprocess.run([sys.executable, '-c', "from uniforge3d.modules.sparse import config; assert config.ATTN == 'sdpa'"], env={**env, 'ATTN_BACKEND': 'typo', 'SPARSE_ATTN_BACKEND': 'sdpa'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError): config.set_attn_backend('typo')


def test_budget_drives_actual_public_query_tiles(monkeypatch):
    import uniforge3d.modules.sparse.attention.sdpa as helper
    q, k, v = torch.randn(8, 2, 5), torch.randn(41, 2, 5), torch.randn(41, 2, 3)
    fixed = 1024**2 + 2 * 2 * 41 * (5 + 3) * 4
    per_row = 2 * (3 * 41 + 2 * 5 + 2 * 3) * 4
    monkeypatch.setattr(config, 'SDPA_SCORE_WORKSPACE_BYTES', fixed + 3 * per_row)
    calls = []
    original = helper.F.scaled_dot_product_attention
    def recording(q, k, v, **kwargs):
        calls.append((q.shape[-2], k.shape[-2]))
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(helper.F, 'scaled_dot_product_attention', recording)
    output = sp.sparse_scaled_dot_product_attention(ragged(q, [8]), ragged(k, [41]), ragged(v, [41])).feats
    assert calls == [(3, 41), (3, 41), (2, 41)]
    compare(output, oracle(q, k, v))
    calls.clear()
    monkeypatch.setattr(config, 'SDPA_SCORE_WORKSPACE_BYTES', fixed + per_row - 1)
    with pytest.raises(ValueError, match='cannot fit one query row'):
        sp.sparse_scaled_dot_product_attention(ragged(q, [8]), ragged(k, [41]), ragged(v, [41]))
    assert not calls


def test_public_production_shape_self_attention(monkeypatch):
    torch.manual_seed(55)
    monkeypatch.setattr(config, 'SDPA_QUERY_CHUNK_SIZE', 16)
    qkv = torch.randn(33, 3, 12, 128)
    x = ragged(qkv, [16, 17], True)
    out = sp.sparse_scaled_dot_product_attention(x)
    compare(out.feats, reference_batches(*qkv.unbind(1), [16, 17], [16, 17]))


def test_dense_queries_empty_shape_and_empty_qkv():
    q = torch.empty(2, 0, 2, 5)
    kv = ragged(torch.randn(4, 2, 2, 5), [1, 3])
    assert sp.sparse_scaled_dot_product_attention(q, kv).shape == q.shape
    qkv = ragged(torch.empty(0, 3, 2, 5), [0, 0])
    out = sp.sparse_scaled_dot_product_attention(qkv)
    assert out.feats.shape == (0, 2, 5) and len(out.layout) == 2


def test_cumulative_dtype_batch_and_device_checks():
    q, k, v = torch.randn(2, 2, 3), torch.randn(3, 2, 3), torch.randn(3, 2, 4)
    with pytest.raises(ValueError, match='int32/int64'):
        packed_sdpa(q, k, v, torch.tensor([0., 2.]), torch.tensor([0, 3]))
    with pytest.raises(ValueError, match='same number of sequences'):
        packed_sdpa(q, k, v, torch.tensor([0, 2]), torch.tensor([0, 1, 3]))
    # A meta tensor tests a mismatched device without touching MPS/CUDA.
    with pytest.raises(ValueError, match='same device'):
        sp.sparse_scaled_dot_product_attention(ragged(q, [2]),
            torch.empty(1, 3, 2, 3, device='meta'), torch.empty(1, 3, 2, 4, device='meta'))
