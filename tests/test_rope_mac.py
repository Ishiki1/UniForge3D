"""CPU mathematics and checkpoint/transfer-control tests; no MPS execution."""
import math

import numpy as np
import pytest
import torch

from uniforge3d.modules.attention.rope import RotaryPositionEmbedder
from uniforge3d.modules.sparse.attention.rope import SparseRotaryPositionEmbedder
from uniforge3d.modules.sparse.basic import SparseTensor
from uniforge3d.models.sparse_structure_flow import SparseStructureFlowModel
from uniforge3d.utils.device import move_model_to_device


def scalar_rotation_oracle(values, coords, frequencies, head_dim):
    """Independent scalar trig/rotation on the prescribed fp32 input/frequencies.

    The existing algorithm explicitly rounds features and phase products to
    fp32, even for fp64 inputs. Respect those input roundings, then evaluate
    rotations with Python fp64; no production phase/rotation helper is called.
    """
    flat_coords = coords.reshape(-1, coords.shape[-1]).tolist()
    flat_values = values.float().reshape(-1, values.shape[-2], head_dim).double()
    output = torch.empty_like(flat_values)
    for row, coordinate in enumerate(flat_coords):
        angles = [float(np.float32(c * f)) for c in coordinate for f in frequencies]
        angles += [0.] * (head_dim // 2 - len(angles))
        for head in range(values.shape[-2]):
            for pair, angle in enumerate(angles):
                a, b = flat_values[row, head, 2*pair:2*pair+2].tolist()
                output[row, head, 2*pair] = a * math.cos(angle) - b * math.sin(angle)
                output[row, head, 2*pair+1] = a * math.sin(angle) + b * math.cos(angle)
    return output.reshape(values.shape)


@pytest.mark.parametrize('head_dim', [2, 8, 128])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_dense_real_rope_against_independent_scalar_oracle(head_dim, dtype):
    torch.manual_seed(800 + head_dim)
    coords = torch.tensor([[0, 0, 0], [1, 3, 7], [31, 32, 63], [64, 511, 1023]], dtype=torch.int64)
    heads = 12 if head_dim == 128 else 2
    x = torch.randn(4, heads, head_dim * 2).to(dtype)[..., ::2]
    original = RotaryPositionEmbedder(head_dim)
    real = RotaryPositionEmbedder(head_dim, phase_format='real')
    phases = real(coords)
    assert phases.shape == (4, head_dim // 2, 2) and not phases.is_complex()
    output = real.apply_rotary_embedding(x, phases)
    expected = scalar_rotation_oracle(x, coords, original.freqs.double().tolist(), head_dim)
    tolerance = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float64: 2e-6}[dtype]
    torch.testing.assert_close(output.double(), expected, atol=tolerance, rtol=tolerance)
    # The untouched CPU complex implementation remains a second comparison.
    legacy = original.apply_rotary_embedding(x.contiguous(), original(coords))
    torch.testing.assert_close(output, legacy, atol=tolerance, rtol=tolerance)
    assert output.dtype == dtype and torch.isfinite(output).all()
    padding = head_dim // 2 - 3 * original.freq_dim
    if padding:
        assert torch.equal(output[..., -2*padding:], x.float().to(dtype)[..., -2*padding:])
    assert original.state_dict() == real.state_dict() == {}


def test_dense_batched_phases_broadcast_and_preserve_pair_order():
    coords = torch.tensor([[[0., 1., 2.], [2., 3., 4.]], [[5., 6., 7.], [8., 9., 10.]]])
    values = torch.arange(2 * 2 * 3 * 8, dtype=torch.float32).reshape(2, 2, 3, 8) / 20
    rope = RotaryPositionEmbedder(8, phase_format='real')
    output = rope.apply_rotary_embedding(values, rope(coords))
    expected = scalar_rotation_oracle(values, coords, rope.freqs.double().tolist(), 8)
    torch.testing.assert_close(output.double(), expected, atol=2e-6, rtol=1e-6)
    # Swapped sin/cos pairs are a wrong orientation, detected by the same oracle.
    wrong = rope.apply_rotary_embedding(values, rope(coords).flip(-1))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong.double(), expected, atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize('head_dim', [8, 128])
def test_sparse_real_rope_qk_and_cache_separation(head_dim):
    torch.manual_seed(19)
    coords = torch.tensor([[0, 0, 2, 9], [0, 63, 31, 7], [1, 1, 2, 3]], dtype=torch.int32)
    q = SparseTensor(torch.randn(3, 12 if head_dim == 128 else 2, head_dim), coords)
    k = q.replace(torch.randn_like(q.feats))
    original = SparseRotaryPositionEmbedder(head_dim)
    legacy_q, legacy_k = original(q, k)
    before = dict(q.get_spatial_cache())
    real = SparseRotaryPositionEmbedder(head_dim, phase_format='real')
    output_q, output_k = real(q, k)
    torch.testing.assert_close(output_q.feats, legacy_q.feats, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(output_k.feats, legacy_k.feats, atol=2e-6, rtol=1e-6)
    for output, source in ((output_q, q), (output_k, k)):
        assert output.coords is source.coords and output._spatial_cache is source._spatial_cache
    after = q.get_spatial_cache()
    added = set(after) - set(before)
    assert len(added) == 1
    key = next(iter(added))
    assert key.endswith('_real_cpu') and not after[key].is_complex()
    phase = after[key]
    real(q)
    assert q.get_spatial_cache(key) is phase
    assert all(q.get_spatial_cache(name) is value for name, value in before.items())


def tiny_model():
    return SparseStructureFlowModel(resolution=2, in_channels=2, model_channels=16,
                                    cond_channels=6, out_channels=2, num_blocks=1,
                                    num_heads=2, pe_mode='rope')


def test_dense_runtime_cache_preserves_strict_checkpoint_contract_and_invalidation():
    model = tiny_model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    phases = model.rope_phases
    pairs = model._real_rope_phases(torch.device('cpu'))
    assert torch.equal(torch.view_as_complex(pairs), phases)
    assert model._real_rope_phases(torch.device('cpu')) is pairs
    assert set(model.state_dict()) == set(before)
    assert model.state_dict()['rope_phases'].is_complex()
    assert model.state_dict()['rope_phases'].shape == before['rope_phases'].shape
    assert not any('runtime' in key for key in model.state_dict())
    with torch.no_grad(): phases.mul_(1j)
    updated = model._real_rope_phases(torch.device('cpu'))
    assert updated is not pairs
    assert torch.equal(torch.view_as_complex(updated), model.rope_phases)
    result = model.load_state_dict(before, strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert model._rope_runtime_cache is None
    rebuilt = model._real_rope_phases(torch.device('cpu'))
    assert torch.equal(torch.view_as_complex(rebuilt), before['rope_phases'])
    move_model_to_device(model, 'cpu')
    assert model._rope_runtime_cache is None and model.get_rope_phases() is model.rope_phases
    clone = tiny_model()
    clone.load_state_dict(model.state_dict(), strict=True)
    for name, value in before.items():
        assert torch.equal(clone.state_dict()[name], value)


def test_mps_transfer_control_excludes_only_persistent_complex_phase(monkeypatch):
    model = tiny_model()
    original = model.rope_phases.clone()
    model.prepare_rope_device(torch.device('mps'))
    # Simulate the reported post-transfer target without allocating on MPS.
    # The real _apply traverses real CPU parameters; this is control-path proof.
    monkeypatch.setattr(SparseStructureFlowModel, 'device', property(lambda self: torch.device('mps')))
    transformed = []
    def cpu_identity(tensor):
        assert not tensor.is_complex(), 'complex checkpoint buffer reached MPS transfer'
        transformed.append(tensor)
        return tensor
    model._apply(cpu_identity)
    assert transformed and torch.equal(model.state_dict()['rope_phases'], original)
    assert model.state_dict()['rope_phases'].device.type == 'cpu'
    assert model._rope_runtime_cache is None


def test_failed_prepared_transfer_restores_checkpoint_buffer():
    model = tiny_model()
    original = model.rope_phases
    model.prepare_rope_device(torch.device('mps'))
    def failure(tensor):
        raise RuntimeError('deliberate transfer failure')
    with pytest.raises(RuntimeError, match='deliberate'):
        model._apply(failure)
    assert model.rope_phases is original
    assert torch.equal(model.state_dict()['rope_phases'], original)


def test_real_runtime_phases_match_actual_cpu_model_forward(monkeypatch):
    from uniforge3d.modules.attention import config
    monkeypatch.setattr(config, 'BACKEND', 'sdpa')
    torch.manual_seed(76)
    model = tiny_model().eval()
    with torch.no_grad():
        model.out_layer.weight.normal_(0, 0.2)
        model.blocks[0].adaLN_modulation[-1].weight.normal_(0, 0.2)
        model.blocks[0].adaLN_modulation[-1].bias.normal_(0, 0.2)
    x, time, cond = torch.randn(1, 2, 2, 2, 2), torch.tensor([0.5]), torch.randn(1, 3, 6)
    with torch.no_grad():
        original = model(x, time, cond)
        monkeypatch.setattr(model, 'get_rope_phases', lambda: model._real_rope_phases(torch.device('cpu')))
        real = model(x, time, cond)
    assert original.abs().max() > 0
    torch.testing.assert_close(real, original, atol=3e-6, rtol=1e-5)
