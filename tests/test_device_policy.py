"""CPU-only device-policy tests. Accelerator calls are spies, never executed."""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from uniforge3d.utils import device as policy

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('argument,expected', [
    (0, 'cuda:0'), (2, 'cuda:2'), ('3', 'cuda:3'), ('cuda', 'cuda'),
    ('cuda:4', 'cuda:4'), (torch.device('cuda:1'), 'cuda:1'),
    ('mps', 'mps'), ('mps:0', 'mps:0'), ('cpu', 'cpu'),
])
def test_device_parsing_preserves_cuda_ordinals(argument, expected):
    assert policy.resolve_device(argument) == torch.device(expected)


@pytest.mark.parametrize('argument', [True, False, -1, 'cuda:-1', 'mps:1', 'cpu:2', 'xpu', 'meta', 'typo', 0.5, None])
def test_device_parsing_rejects_ambiguous_or_unsupported_values(argument):
    with pytest.raises(ValueError):
        policy.resolve_device(argument)


def test_mps_selection_configures_both_backends_without_accelerator_calls(monkeypatch):
    from uniforge3d.modules.attention import config as dense
    from uniforge3d.modules.sparse import config as sparse
    monkeypatch.setattr(dense, 'BACKEND', 'flash_attn')
    monkeypatch.setattr(sparse, 'ATTN', 'flash_attn')
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK', raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('configuration must not touch accelerator runtime')
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(torch.mps, 'empty_cache', forbidden)
    monkeypatch.setattr(torch.backends.mps, 'is_available', forbidden)
    assert policy.configure_inference_device('mps') == torch.device('mps')
    assert dense.BACKEND == sparse.ATTN == 'sdpa'
    assert 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ


def test_mps_generic_fallback_rejected_before_backend_mutation(monkeypatch):
    from uniforge3d.modules.attention import config as dense
    from uniforge3d.modules.sparse import config as sparse
    monkeypatch.setattr(dense, 'BACKEND', 'flash_attn')
    monkeypatch.setattr(sparse, 'ATTN', 'flash_attn')
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    with pytest.raises(RuntimeError, match='explicit adapters'):
        policy.configure_inference_device('mps')
    assert dense.BACKEND == sparse.ATTN == 'flash_attn'


@pytest.mark.parametrize('device', ['cpu', 0, 'cuda:2'])
def test_other_devices_preserve_existing_attention_choice(device, monkeypatch):
    from uniforge3d.modules.attention import config as dense
    from uniforge3d.modules.sparse import config as sparse
    monkeypatch.setattr(dense, 'BACKEND', 'flash_attn_3')
    monkeypatch.setattr(sparse, 'ATTN', 'xformers')
    policy.configure_inference_device(device)
    assert dense.BACKEND == 'flash_attn_3' and sparse.ATTN == 'xformers'


def test_current_device_selection_has_no_mps_cpu_cuda_side_effect(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'set_device', calls.append)
    policy.select_current_device('cpu')
    policy.select_current_device('mps')
    assert calls == []
    policy.select_current_device(3)
    assert calls == [torch.device('cuda:3')]


@pytest.mark.parametrize('device,available,expected', [
    ('cpu', True, ['gc']), ('cuda:2', True, ['gc', 'cuda-enter:cuda:2', 'cuda-cache', 'cuda-exit']),
    ('cuda:2', False, ['gc']), ('mps', True, ['gc', 'mps-cache']), ('mps', False, ['gc']),
])
def test_cleanup_touches_only_selected_backend(device, available, expected, monkeypatch):
    calls = []
    monkeypatch.setattr(policy.gc, 'collect', lambda: calls.append('gc'))
    @contextmanager
    def cuda_device(value):
        calls.append(f'cuda-enter:{value}')
        yield
        calls.append('cuda-exit')
    monkeypatch.setattr(torch.cuda, 'device', cuda_device)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('cuda-cache'))
    monkeypatch.setattr(torch.mps, 'empty_cache', lambda: calls.append('mps-cache'))
    def cuda_available():
        assert str(device).startswith('cuda'), 'unrelated CUDA availability probe'
        return available
    def mps_available():
        assert device == 'mps', 'unrelated MPS availability probe'
        return available
    monkeypatch.setattr(torch.cuda, 'is_available', cuda_available)
    monkeypatch.setattr(torch.backends.mps, 'is_available', mps_available)
    policy.release_device(device)
    assert calls == expected


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_actual_cpu_autocast_preserves_requested_dtype_and_fp32_subregion(dtype):
    torch.manual_seed(503)
    layer = torch.nn.Linear(7, 5)
    data = torch.randn(3, 7)
    state = torch.get_rng_state().clone()
    with policy.autocast_for_device('cpu', dtype):
        assert torch.is_autocast_enabled('cpu')
        assert torch.get_autocast_dtype('cpu') == dtype
        assert layer(data).dtype == dtype
        with policy.autocast_for_device('cpu', torch.float32):
            assert not torch.is_autocast_enabled('cpu')
            assert layer(data).dtype == torch.float32
        assert torch.is_autocast_enabled('cpu')
    assert not torch.is_autocast_enabled('cpu')
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize('device', ['cuda:2', 'mps', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_autocast_selects_actual_device_without_precision_substitution(device, dtype, monkeypatch):
    calls = []
    monkeypatch.setattr(torch, 'autocast', lambda **kwargs: calls.append(kwargs))
    policy.autocast_for_device(device, dtype)
    assert calls == [dict(device_type=policy.resolve_device(device).type, dtype=dtype, enabled=dtype != torch.float32)]


def test_cli_help_imports_without_mesh_models_or_weights():
    result = subprocess.run([sys.executable, 'infer.py', '--help'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--device DEVICE' in result.stdout and 'mps' in result.stdout and 'cuda:0' in result.stdout
    result = subprocess.run([sys.executable, 'infer.py', '--device', 'mps:2'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and 'invalid resolve_device value' in result.stderr


def test_pipeline_policy_wiring_before_model_construction_and_cpu_rng_unchanged():
    # Source contract only: importing/running the whole pipeline would require
    # the unproved native mesh stack and weights. Helpers above execute for real.
    source = (ROOT / 'uniforge3d/pipelines/uniforge3d_image_to_3d.py').read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
    configure = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'configure_inference_device')
    load = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'load_models')
    assert configure.lineno < load.lineno
    assert 'torch.cuda.set_device' not in source and 'release_cuda(' not in source
    noise = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'randn']
    assert len(noise) == 1
    assert any(k.arg == 'device' and isinstance(k.value, ast.Constant) and k.value.value == 'cpu' for k in noise[0].keywords)
    assert 'torch.manual_seed(seed)' in source
    for filename in ('vggt_adapter.py', 'vggt_routed_features.py'):
        model_source = (ROOT / 'uniforge3d/models' / filename).read_text()
        assert 'autocast_for_device(self.device, self.dtype)' in model_source
        assert 'torch.autocast(device_type="cuda"' not in model_source
    assert 'autocast_for_device(self.device, torch.float32)' in (ROOT / 'uniforge3d/models/vggt_routed_features.py').read_text()


def test_manual_cast_queries_tensor_device_autocast():
    from uniforge3d.modules.utils import manual_cast
    from uniforge3d.modules.sparse.basic import VarLenTensor
    tensor = torch.randn(3, 5)
    sparse = VarLenTensor(tensor)
    assert manual_cast(tensor, torch.bfloat16).dtype == torch.bfloat16
    with policy.autocast_for_device('cpu', torch.bfloat16):
        assert manual_cast(tensor, torch.float16) is tensor
        assert manual_cast(sparse, torch.float16) is sparse
        with policy.autocast_for_device('cpu', torch.float32):
            assert manual_cast(tensor, torch.bfloat16).dtype == torch.bfloat16
