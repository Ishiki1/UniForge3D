"""Explicit inference device, precision, backend and cache-release policy."""

import gc
import os
from numbers import Integral

import torch


def resolve_device(device: int | str | torch.device = 0) -> torch.device:
    """Retain integer CUDA ordinals while accepting explicit MPS/CPU devices.

    This function parses only. It does not initialize an accelerator, allocate
    tensors, select a current CUDA device, or change attention backends.
    """
    if isinstance(device, bool):
        raise ValueError("device must be a CUDA ordinal or cpu/cuda/mps device")
    if isinstance(device, Integral):
        if device < 0:
            raise ValueError("CUDA device ordinal must be nonnegative")
        return torch.device("cuda", int(device))
    if isinstance(device, str) and device.isdecimal():
        return torch.device("cuda", int(device))
    if not isinstance(device, (str, torch.device)):
        raise ValueError("device must be a CUDA ordinal or cpu/cuda/mps device")
    try:
        selected = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid inference device: {device!r}") from exc
    if selected.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported inference device: {selected}")
    if selected.type in ("cpu", "mps") and selected.index not in (None, 0):
        raise ValueError(f"{selected.type} supports only its default device (index 0)")
    return selected


def configure_inference_device(device: int | str | torch.device = 0) -> torch.device:
    """Select MPS's explicit attention adapters before model construction.

    CUDA/CPU selection leaves the configured attention backends unchanged.
    Backend settings are process-global, as in the original package; concurrent
    pipelines with different backend policies require separate processes.
    """
    selected = resolve_device(device)
    if selected.type == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError(
                "MPS inference requires PYTORCH_ENABLE_MPS_FALLBACK to be disabled; "
                "unsupported operations must be handled by explicit adapters"
            )
        from ..modules.attention import config as dense_attention
        from ..modules.sparse import config as sparse_attention

        dense_attention.set_backend("sdpa")
        sparse_attention.set_attn_backend("sdpa")
    return selected


def select_current_device(device: int | str | torch.device) -> None:
    """CUDA keeps its ordinal selection; MPS and CPU have no CUDA side effect."""
    selected = resolve_device(device)
    if selected.type == "cuda":
        torch.cuda.set_device(selected)


def autocast_for_device(device: int | str | torch.device, dtype: torch.dtype):
    """Use the requested model precision on its actual execution device.

    Float32 explicitly disables that device's outer autocast (needed by DPT).
    This never substitutes a lower precision or a different device.
    """
    selected = resolve_device(device)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported inference autocast dtype: {dtype}")
    return torch.autocast(
        device_type=selected.type, dtype=dtype, enabled=dtype != torch.float32,
    )


def move_model_to_device(model, device: int | str | torch.device):
    """Prepare explicit runtime adapters, then use the model's normal .to().

    Dense RoPE keeps its checkpoint complex buffer on CPU and transfers a
    separate real-pair runtime cache on MPS. Other modules move unchanged.
    """
    selected = resolve_device(device)
    if isinstance(model, torch.nn.Module):
        for module in model.modules():
            prepare = getattr(module, "prepare_rope_device", None)
            if prepare is not None:
                prepare(selected)
    return model.to(selected)


def release_device(device: int | str | torch.device) -> None:
    """Collect Python objects and release only the selected device's cache."""
    selected = resolve_device(device)
    gc.collect()
    if selected.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(selected):
            torch.cuda.empty_cache()
    elif selected.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def release_cuda() -> None:
    """Compatibility entry point for existing CUDA-only callers."""
    release_device("cuda")
