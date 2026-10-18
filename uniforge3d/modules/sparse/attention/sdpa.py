"""Exact full-key, query-tiled SDPA for packed inference attention.

The byte budget estimates score/workspace, not total process/device memory.
Inputs, persistent outputs, projections, model weights and allocator overhead
remain outside it. PyTorch owns kernel selection; native peak memory must be
measured separately. No device fallback or precision change is enabled here.
"""
from __future__ import annotations

from numbers import Integral

import torch
import torch.nn.functional as F

from .. import config
from ..basic import SparseTensor, VarLenTensor


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def query_tile_size(key_tokens, heads, q_width, v_width, dtype, *,
                    score_workspace_bytes, query_chunk_size):
    """Choose rows using three score arrays and conservative fp32 scratch.

    Reserve 1 MiB plus two accumulation-precision K/V copies; each row then
    reserves three score vectors and two Q/output vectors. This deliberately
    does not assume a fused, linear-memory SDPA kernel is selected.
    """
    budget = _positive_int(score_workspace_bytes, "score_workspace_bytes")
    limit = _positive_int(query_chunk_size, "query_chunk_size")
    for value, name in ((key_tokens, "key_tokens"), (heads, "heads"),
                        (q_width, "q_width"), (v_width, "v_width")):
        _positive_int(value, name)
    itemsize = max(4, torch.empty((), dtype=dtype).element_size())
    fixed = 1024 * 1024 + 2 * heads * key_tokens * (q_width + v_width) * itemsize
    per_row = heads * (3 * key_tokens + 2 * q_width + 2 * v_width) * itemsize
    rows = (budget - fixed) // per_row
    if rows < 1:
        raise ValueError(
            f"SDPA score/workspace estimate cannot fit one query row: "
            f"budget={budget} bytes, requires at least {fixed + per_row} bytes; "
            "this estimate is not total RSS or total device memory"
        )
    return min(limit, rows)


def validate_cumulative_lengths(cu, total, *, name, device):
    """Return Python boundaries after checking complete, ordered coverage."""
    if not isinstance(cu, torch.Tensor) or cu.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    if cu.dtype not in (torch.int32, torch.int64) or cu.device != device:
        raise ValueError(f"{name} must be int32/int64 on {device}")
    boundaries = cu.tolist()
    if (not boundaries or boundaries[0] != 0 or boundaries[-1] != total
            or any(a > b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError(f"{name} must start at zero, be monotone, and cover {total} rows")
    return boundaries


def packed_sdpa(q, k, v, cu_q, cu_k, *, score_workspace_bytes=None,
                query_chunk_size=None):
    """Noncausal attention with zero dropout and scale Dq**-0.5.

    Q/K/V have shapes [Tq,H,Dq], [Tk,H,Dq], [Tk,H,Dv]. Empty query
    sequences produce no rows; a nonempty query with no keys is rejected.
    Every query tile sees ALL keys in its sequence. Outputs keep Q packing,
    dtype and device. No approximation, key truncation, or key tiling is used.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [T,H,D]")
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"{name} must use a supported floating dtype")
        if tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
            raise ValueError(f"{name} must have positive head count and width")
    if not (q.device == k.device == v.device and q.dtype == k.dtype == v.dtype):
        raise ValueError("Q/K/V must have the same device and dtype")
    if q.shape[1:] != k.shape[1:] or k.shape[:2] != v.shape[:2]:
        raise ValueError("Q/K head shape and K/V token/head counts must match")
    qs = validate_cumulative_lengths(cu_q, len(q), name="cu_q", device=q.device)
    ks = validate_cumulative_lengths(cu_k, len(k), name="cu_k", device=q.device)
    if len(qs) != len(ks):
        raise ValueError("Q and K must have the same number of sequences")
    budget = config.SDPA_SCORE_WORKSPACE_BYTES if score_workspace_bytes is None else score_workspace_bytes
    limit = config.SDPA_QUERY_CHUNK_SIZE if query_chunk_size is None else query_chunk_size
    _positive_int(budget, "score_workspace_bytes")
    _positive_int(limit, "query_chunk_size")
    tiles = []
    for qa, qb, ka, kb in zip(qs, qs[1:], ks, ks[1:]):
        if qa == qb:
            tiles.append(0)
            continue
        if ka == kb:
            raise ValueError("Nonempty query sequence has no keys")
        tiles.append(query_tile_size(kb - ka, q.shape[1], q.shape[2], v.shape[2], q.dtype,
                                    score_workspace_bytes=budget, query_chunk_size=limit))
    out = q.new_empty((len(q), q.shape[1], v.shape[2]))
    for qa, qb, ka, kb, tile in zip(qs, qs[1:], ks, ks[1:], tiles):
        if not tile:
            continue
        keys = k[ka:kb].transpose(0, 1).unsqueeze(0)
        values = v[ka:kb].transpose(0, 1).unsqueeze(0)
        for start in range(qa, qb, tile):
            stop = min(start + tile, qb)
            queries = q[start:stop].transpose(0, 1).unsqueeze(0)
            result = F.scaled_dot_product_attention(
                queries, keys, values, dropout_p=0.0, is_causal=False,
            )
            out[start:stop] = result.squeeze(0).transpose(0, 1)
    return out


def _pack(tensor, *, name, fused=None):
    """Validate public dense/ragged shape and contiguous batch layout."""
    if not isinstance(tensor, (torch.Tensor, VarLenTensor)):
        raise TypeError(f"{name} must be a tensor or VarLenTensor")
    ragged = isinstance(tensor, VarLenTensor)
    data = tensor.feats if ragged else tensor
    expected = (3 if fused is None else 4) + (not ragged)
    if data.ndim != expected or (fused is not None and data.shape[-3] != fused):
        raise ValueError(f"Invalid shape for {name}: {tuple(data.shape)}")
    if ragged:
        boundaries = [0]
        for entry in tensor.layout:
            if (not isinstance(entry, slice) or entry.step not in (None, 1)
                    or isinstance(entry.start, bool) or isinstance(entry.stop, bool)
                    or not isinstance(entry.start, Integral) or not isinstance(entry.stop, Integral)
                    or entry.start != boundaries[-1] or entry.stop < entry.start):
                raise ValueError(f"{name} layout must be contiguous monotone slices starting at zero")
            boundaries.append(int(entry.stop))
        if boundaries[-1] != len(data) or len(boundaries) != tensor.shape[0] + 1:
            raise ValueError(f"{name} layout must cover all feature rows and batches")
        if isinstance(tensor, SparseTensor):
            coords = tensor.coords
            if (coords.ndim != 2 or len(coords) != len(data) or coords.shape[1] < 1
                    or coords.device != data.device or coords.dtype not in (torch.int32, torch.int64)):
                raise ValueError(f"{name} sparse coordinates must match feature rows/device")
            for batch, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
                if torch.any(coords[a:b, 0] != batch).item():
                    raise ValueError(f"{name} sparse batch coordinates disagree with layout")
    else:
        batch, length = data.shape[:2]
        boundaries = [i * length for i in range(batch + 1)]
        data = data.reshape(batch * length, *data.shape[2:])
    cu = torch.tensor(boundaries, device=data.device, dtype=torch.int64)
    return data, cu


def sparse_sdpa(*args, **kwargs):
    """Adapter for the existing public one/two/three-input sparse API."""
    names = {1: ("qkv",), 2: ("q", "kv"), 3: ("q", "k", "v")}.get(len(args) + len(kwargs))
    if names is None or set(kwargs) != set(names[len(args):]):
        raise TypeError("Expected qkv, q/kv, or q/k/v without duplicate or unknown arguments")
    inputs = args + tuple(kwargs[name] for name in names[len(args):])
    original_q = inputs[0]
    if len(inputs) == 1:
        if not isinstance(original_q, VarLenTensor):
            raise TypeError("Packed qkv must be a VarLenTensor")
        qkv, cu_q = _pack(original_q, name="qkv", fused=3)
        q, k, v = qkv.unbind(1)
        cu_k = cu_q
    else:
        if not isinstance(original_q, VarLenTensor) and not isinstance(inputs[1], VarLenTensor):
            raise TypeError("At least Q or KV must be a VarLenTensor")
        q, cu_q = _pack(original_q, name="q")
        if len(inputs) == 2:
            kv, cu_k = _pack(inputs[1], name="kv", fused=2)
            k, v = kv.unbind(1)
        else:
            if type(inputs[1]) is not type(inputs[2]):
                raise TypeError("K and V must have the same container type")
            k, cu_k = _pack(inputs[1], name="k")
            v, cu_v = _pack(inputs[2], name="v")
            if not (q.device == k.device == v.device):
                raise ValueError("Q/K/V must have the same device")
            if not torch.equal(cu_k, cu_v):
                raise ValueError("K and V must have identical sequence layouts")
            if isinstance(inputs[1], SparseTensor) and not torch.equal(inputs[1].coords, inputs[2].coords):
                raise ValueError("Sparse K and V coordinates must match")
    result = packed_sdpa(q, k, v, cu_q, cu_k)
    if isinstance(original_q, VarLenTensor):
        return original_q.replace(result)
    return result.reshape(*original_q.shape[:2], result.shape[1], result.shape[2])
