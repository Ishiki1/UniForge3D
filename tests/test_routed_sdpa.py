import math

import pytest
import torch

from uniforge3d.modules import sparse as sp
from uniforge3d.modules.sparse import config
from uniforge3d.modules.sparse.attention.modules import SparseMultiHeadAttention
from uniforge3d.modules.sparse.attention.routed_full_only_flash import (
    PackedRoutedFullOnlyLayoutBuilder, RoutedFullOnlyFlashCrossAttention,
)
from test_sdpa_attention import compare, oracle


def fixture(resolution, lengths=(17, 3, 1, 2)):
    """Two batches with missing view IDs and deliberately unsorted view rows."""
    # All five reserved tokens remain in the bank; each batch/view has a signature.
    torch.manual_seed(200 + resolution)
    views, width = 4, 4
    assignments = ([2] * lengths[0] + [0] * lengths[1],
                   [3] * lengths[2] + [1] * lengths[3])
    coordinates, selected_rows = [], []
    selected = torch.zeros((2, resolution, resolution, resolution), dtype=torch.int64)
    for b, assignment in enumerate(assignments):
        perm = torch.randperm(len(assignment)).tolist()
        for i, index in enumerate(perm):
            view = assignment[index]
            xyz = (i % resolution, i // resolution, 0)
            coordinates.append((b, *xyz))
            selected[b, *xyz] = view
            selected_rows.append((b, view))
    coords = torch.tensor(coordinates, dtype=torch.int32)
    # Noncontiguous raw sparse features.
    feats = torch.randn(len(coords), 1536 * 2)[:, ::2] * 0.3
    x = sp.SparseTensor(feats, coords, shape=torch.Size([2, 1536]))
    x.register_spatial_cache('sentinel', object())
    full = torch.randn(2, views, resolution**2 + 5, width * 2)[..., ::2] * 0.2
    for b in range(2):
        for v in range(views):
            full[b, v] += b * 0.31 + v * 0.13
    layout = PackedRoutedFullOnlyLayoutBuilder(resolution).prepare(
        x=x, selected_view=selected, batch_size=2, views=views)
    return x, full, selected, selected_rows, layout


def linear_reference(x, layer):
    # Compute mathematically in fp64, then round to the production projection dtype.
    result = x.double() @ layer.weight.detach().double().T
    if layer.bias is not None:
        result += layer.bias.detach().double()
    return result.to(x.dtype)


def rms_reference(x, norm):
    # The production contract explicitly performs these operations in fp32,
    # including when incoming activations are fp16/bf16/fp64. Do not pretend
    # that an entirely fp64 RMS norm is the prescribed reference.
    value = x.float()
    denominator = (value.square().sum(-1, keepdim=True)).sqrt().clamp_min(1e-12)
    return ((value / denominator) * norm.gamma.detach().float() * math.sqrt(x.shape[-1])).to(x.dtype)


def routed_reference(layer, x, full, assignments, normalize=True):
    q = linear_reference(x.feats, layer.to_q).reshape(-1, 12, 128)
    if layer.qk_rms_norm and normalize:
        q = rms_reference(q, layer.q_rms_norm)
    grouped = {}
    for row, group in enumerate(assignments):
        grouped.setdefault(group, []).append(row)
    out = torch.empty_like(q, dtype=torch.float64)
    for (batch, view), indexes in grouped.items():
        kv = linear_reference(full[batch, view], layer.to_kv).reshape(-1, 2, 12, 128)
        k, v = kv.unbind(1)
        if layer.qk_rms_norm and normalize:
            k = rms_reference(k, layer.k_rms_norm)
        out[indexes] = oracle(q[indexes], k, v)
    # SDPA returns input dtype before the output projection.
    return linear_reference(out.to(q.dtype).flatten(1), layer.to_out).double()


@pytest.mark.parametrize('resolution', [32, 64])
@pytest.mark.parametrize('norm,bias', [(False, False), (False, True), (True, False), (True, True)])
def test_real_routed_module_and_layout(resolution, norm, bias, monkeypatch):
    monkeypatch.setattr(config, 'SDPA_QUERY_CHUNK_SIZE', 16)
    x, full, selected, assignments, layout = fixture(resolution)
    layer = RoutedFullOnlyFlashCrossAttention(1536, 4, 12, qkv_bias=bias, qk_rms_norm=norm).eval()
    layer.full_tokens = resolution**2 + 5
    if norm:
        with torch.no_grad():
            layer.q_rms_norm.gamma.copy_(torch.linspace(0.6, 1.4, 1536).reshape(12, 128))
            layer.k_rms_norm.gamma.copy_(torch.linspace(1.3, 0.7, 1536).reshape(12, 128))
    context = {'full_bank': full, 'selected_view': selected, **layout}
    order, inverse = layout['routed_query_order'], layout['routed_query_inverse']
    assert torch.equal(order[inverse], torch.arange(len(x.feats)))
    assert torch.equal(inverse[order], torch.arange(len(x.feats)))
    assert layout['routed_active_groups'].tolist() == [0, 2, 5, 7]
    assert layout['routed_cu_seqlens_k'].tolist() == [n * layer.full_tokens for n in range(5)]
    with torch.no_grad():
        output = layer(x, context)
        expected = routed_reference(layer, x, full, assignments)
    compare(output.feats, expected, atol=5e-6, rtol=3e-4)
    delta = (output.feats.double() - expected).abs()
    print(f'routed grid={resolution} keys={layer.full_tokens} heads=12x128 norm={norm} bias={bias}: max_abs={delta.max().item():.9g}, worst_row_l2={delta.norm(dim=1).max().item():.9g}')
    assert output.coords is x.coords and output._spatial_cache is x._spatial_cache
    # A wrong view or inverse scatter is measurably wrong; the oracle detects it.
    wrong = [(b, (v + 1) % 4) for b, v in assignments]
    with torch.no_grad():
        wrong_expected = routed_reference(layer, x, full, wrong)
    assert not torch.allclose(output.feats.double(), wrong_expected, atol=5e-6, rtol=3e-4)
    assert not torch.allclose(output.feats.double()[order], expected, atol=5e-6, rtol=3e-4)
    if norm:
        with torch.no_grad(): unnormalized = routed_reference(layer, x, full, assignments, normalize=False)
        assert not torch.allclose(output.feats.double(), unnormalized, atol=5e-6, rtol=3e-4)


@pytest.mark.parametrize('resolution', [32, 64])
def test_routed_view_batch_isolation_and_cfg_zero_layout(resolution, monkeypatch):
    monkeypatch.setattr(config, 'SDPA_QUERY_CHUNK_SIZE', 1)
    x, full, selected, assignments, layout = fixture(resolution, (2, 1, 1, 1))
    layer = RoutedFullOnlyFlashCrossAttention(1536, 4, 12, qk_rms_norm=True).eval()
    layer.full_tokens = resolution**2 + 5
    with torch.no_grad():
        baseline = layer(x, {'full_bank': full, **layout}).feats
        changed = full.clone()
        changed[0, 1] += 1000  # no query in batch 0 chooses view 1
        compare(layer(x, {'full_bank': changed, **layout}).feats, baseline.double(), atol=0, rtol=0)
        changed[0, 2] += 1
        affected = layer(x, {'full_bank': changed, **layout}).feats
    unaffected = torch.tensor([b != 0 or v != 2 for b, v in assignments])
    compare(affected[unaffected], baseline[unaffected].double(), atol=0, rtol=0)
    assert not torch.allclose(affected[~unaffected], baseline[~unaffected])
    negative = PackedRoutedFullOnlyLayoutBuilder(resolution).prepare(
        x=x, selected_view=torch.zeros_like(selected), batch_size=2, views=4)
    assert negative['routed_active_groups'].tolist() == [0, 4]
    assert not torch.equal(negative['routed_query_order'], layout['routed_query_order'])
    with torch.no_grad():
        output = layer(x, {'full_bank': torch.zeros_like(full), **negative}).feats
        expected = routed_reference(layer, x, torch.zeros_like(full), [(b, 0) for b, v in assignments])
    compare(output, expected, atol=5e-6, rtol=3e-4)
    # Zero features alone hide a stale route. A tagged-bank probe proves that the
    # separately built CFG layout actually selects view zero in both batches.
    with torch.no_grad():
        tagged = layer(x, {'full_bank': full, **negative}).feats
        expected = routed_reference(layer, x, full, [(b, 0) for b, v in assignments])
    compare(tagged, expected, atol=5e-6, rtol=3e-4)


@pytest.mark.parametrize('resolution', [32, 64])
def test_all_view_full_cross_attention_retains_special_tokens(resolution, monkeypatch):
    import uniforge3d.modules.sparse.attention.sdpa as helper

    # Keep the production defaults: grid64 must split these 56 queries 55+1.
    assert config.SDPA_QUERY_CHUNK_SIZE == 256
    assert config.SDPA_SCORE_WORKSPACE_BYTES == 512 * 1024**2
    length = resolution**2 + 5
    total = 4 * length
    q = torch.zeros(56, 12, 128)
    kv = torch.zeros(1, total, 2, 12, 128)
    expected = torch.zeros_like(q, dtype=torch.float64)
    marker_positions = []
    for view in range(4):
        # A distinct channel for EACH of the five leading special tokens,
        # the first patch token and the last patch token of EACH view.
        for slot, offset in enumerate((0, 1, 2, 3, 4, 5, length - 1)):
            channel = view * 7 + slot
            position = view * length + offset
            kv[0, position, 1, :, channel] = total * (channel + 1)
            expected[..., channel] = channel + 1
            marker_positions.append(position)
        # A whole-view signature also detects omission independently of markers.
        kv[0, view * length:(view + 1) * length, 1, :, 28 + view] = 4 * (view + 1)
        expected[..., 28 + view] = view + 1

    calls = []
    original = helper.F.scaled_dot_product_attention
    def recording(q, k, v, **kwargs):
        calls.append((q.shape[-2], k.shape[-2]))
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(helper.F, 'scaled_dot_product_attention', recording)
    out = sp.sparse_scaled_dot_product_attention(sp.VarLenTensor(q), kv).feats
    compare(out, expected)
    assert calls == ([(55, total), (1, total)] if resolution == 64 else [(56, total)])

    # Execute the actual public operator with an intentionally faulty native
    # call. The SAME oracle assertion must reject each removed view, the middle
    # two views together, and every individual special/boundary token. One
    # query per mutant is sufficient; only the baseline needs the 55+1 edge.
    positions = torch.arange(total)
    omitted_sets = [
        positions[(positions // length) == view] for view in range(4)
    ] + [positions[(positions // length == 1) | (positions // length == 2)]]
    omitted_sets += [torch.tensor([position]) for position in marker_positions]
    for omitted in omitted_sets:
        keep_mask = torch.ones(total, dtype=torch.bool)
        keep_mask[omitted] = False
        kept = positions[keep_mask]
        def dropping(q, k, v, **kwargs):
            return original(q, k.index_select(-2, kept), v.index_select(-2, kept), **kwargs)
        monkeypatch.setattr(helper.F, 'scaled_dot_product_attention', dropping)
        mutant = sp.sparse_scaled_dot_product_attention(sp.VarLenTensor(q[:1]), kv).feats
        assert torch.isfinite(mutant).all()
        with pytest.raises(AssertionError):
            compare(mutant, expected[:1])
    print(f'all-view grid={resolution}: keys={total}, chunks={calls}, '
          f'{len(omitted_sets)} key-omission mutants rejected')


@pytest.mark.parametrize('resolution', [32, 64])
@pytest.mark.parametrize('queries', [1, 15, 16, 17, 31, 32, 33])
def test_routed_query_chunk_edges(resolution, queries, monkeypatch):
    monkeypatch.setattr(config, 'SDPA_QUERY_CHUNK_SIZE', 16)
    x, full, selected, assignments, layout = fixture(resolution, (queries, 1, 1, 1))
    layer = RoutedFullOnlyFlashCrossAttention(1536, 4, 12).eval()
    layer.full_tokens = resolution**2 + 5
    with torch.no_grad():
        output = layer(x, {'full_bank': full, 'selected_view': selected, **layout}).feats
        expected = routed_reference(layer, x, full, assignments)
    compare(output, expected, atol=5e-6, rtol=3e-4)


def test_four_view_production_budget_reduces_query_tile():
    from uniforge3d.modules.sparse.attention.sdpa import query_tile_size
    assert config.SDPA_SCORE_WORKSPACE_BYTES == 512 * 1024**2
    rows = query_tile_size(4 * 4101, 12, 128, 128, torch.float32,
                          score_workspace_bytes=config.SDPA_SCORE_WORKSPACE_BYTES,
                          query_chunk_size=config.SDPA_QUERY_CHUNK_SIZE)
    assert rows == 55


def test_stale_layout_is_rejected_when_selected_grid_is_present():
    x, full, selected, assignments, layout = fixture(32, (2, 1, 1, 1))
    layer = RoutedFullOnlyFlashCrossAttention(1536, 4, 12)
    layer.full_tokens = 1029
    with pytest.raises(ValueError, match='rebuild it'):
        layer(x, {'full_bank': full, 'selected_view': torch.zeros_like(selected), **layout})


@pytest.mark.parametrize('norm,bias', [(False, False), (False, True), (True, False), (True, True)])
def test_strict_checkpoint_compatibility_with_existing_cross_attention(norm, bias):
    old = SparseMultiHeadAttention(1536, 12, ctx_channels=4, type='cross', qkv_bias=bias, qk_rms_norm=norm)
    new = RoutedFullOnlyFlashCrossAttention(1536, 4, 12, qkv_bias=bias, qk_rms_norm=norm)
    expected = {'to_q.weight', 'to_kv.weight', 'to_out.weight', 'to_out.bias'}
    if bias: expected |= {'to_q.bias', 'to_kv.bias'}
    if norm: expected |= {'q_rms_norm.gamma', 'k_rms_norm.gamma'}
    assert set(old.state_dict()) == set(new.state_dict()) == expected
    assert {k: v.shape for k, v in old.state_dict().items()} == {k: v.shape for k, v in new.state_dict().items()}
    result = new.load_state_dict(old.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    old.load_state_dict(new.state_dict(), strict=True)
    for key in expected:
        assert torch.equal(old.state_dict()[key], new.state_dict()[key])
    assert not dict(new.named_buffers())


@pytest.mark.parametrize('kind', ['grid_shape', 'grid_float', 'view_range', 'coord_float', 'coord_negative', 'coord_upper', 'batch', 'batch_count', 'empty'])
def test_layout_builder_rejects_invalid_inputs(kind):
    x, full, selected, assignments, layout = fixture(32, (1, 1, 1, 1))
    batch = 2
    if kind == 'grid_shape': selected = selected[:, :-1]
    elif kind == 'grid_float': selected = selected.float()
    elif kind == 'view_range': selected[0, 31, 31, 31] = 4
    elif kind == 'coord_float': x.coords = x.coords.float()
    elif kind == 'coord_negative': x.coords[0, 1] = -1
    elif kind == 'coord_upper': x.coords[0, 2] = 32
    elif kind == 'batch': x.coords[0, 0] = 2
    elif kind == 'batch_count': batch = 1; selected = selected[:1]
    elif kind == 'empty': x = sp.SparseTensor(torch.empty(0, 1536), torch.empty(0, 4, dtype=torch.int32), shape=torch.Size([2, 1536]))
    with pytest.raises(ValueError):
        PackedRoutedFullOnlyLayoutBuilder(32).prepare(x=x, selected_view=selected, batch_size=batch, views=4)


@pytest.mark.parametrize('kind', ['order_duplicate', 'inverse', 'group_duplicate', 'group_range', 'cu_q_total', 'cu_q_decrease', 'cu_k_bank', 'max_q', 'cross_batch', 'index_dtype'])
def test_routed_module_rejects_malformed_layout(kind):
    x, full, selected, assignments, layout = fixture(32, (2, 1, 1, 1))
    layer = RoutedFullOnlyFlashCrossAttention(1536, 4, 12)
    layer.full_tokens = 1029
    layout = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in layout.items()}
    if kind == 'order_duplicate': layout['routed_query_order'][0] = layout['routed_query_order'][1]
    elif kind == 'inverse': layout['routed_query_inverse'][0] = len(x.feats)
    elif kind == 'group_duplicate': layout['routed_active_groups'][1] = layout['routed_active_groups'][0]
    elif kind == 'group_range': layout['routed_active_groups'][-1] = 8
    elif kind == 'cu_q_total': layout['routed_cu_seqlens_q'][-1] += 1
    elif kind == 'cu_q_decrease': layout['routed_cu_seqlens_q'][1] = -1
    elif kind == 'cu_k_bank': layout['routed_cu_seqlens_k'][1] -= 1
    elif kind == 'max_q': layout['routed_max_seqlen_q'] += 1
    elif kind == 'cross_batch':
        # A complete invertible permutation can still mix the two batches.
        order = layout['routed_query_order']
        order[0], order[-1] = order[-1].clone(), order[0].clone()
        layout['routed_query_inverse'][order] = torch.arange(len(order))
    elif kind == 'index_dtype': layout['routed_query_order'] = layout['routed_query_order'].int()
    with pytest.raises(ValueError):
        layer(x, {'full_bank': full, **layout})


def test_routed_production_dimensions_are_preserved():
    with pytest.raises(ValueError, match='12 heads x 128'):
        RoutedFullOnlyFlashCrossAttention(64, 4, 2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rmsnorm_prescribed_fp32_and_return_dtype(dtype):
    from uniforge3d.modules.sparse.attention.modules import SparseMultiHeadRMSNorm
    torch.manual_seed(120)
    norm = SparseMultiHeadRMSNorm(128, 12)
    with torch.no_grad(): norm.gamma.copy_(torch.linspace(0.5, 1.5, 1536).reshape(12, 128))
    values = torch.randn(3, 12, 128).to(dtype)
    values[0] = 0
    values[1] *= 1e-13
    with torch.no_grad():
        out = norm(values)
        expected = rms_reference(values, norm)
    tol = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-6, torch.float64: 1e-6}[dtype]
    compare(out, expected.double(), atol=tol, rtol=tol)
    assert out.dtype == dtype and torch.equal(out[0], torch.zeros_like(out[0]))
