"""Checkpoint-compatible Full-only routed cross-attention (Flash or SDPA)."""
from __future__ import annotations

from typing import Dict
from numbers import Integral
from math import isqrt

import torch
import torch.nn as nn

from ..basic import SparseTensor
from .. import config
from .modules import SparseMultiHeadRMSNorm


def _flash_attention():
    # Import only for the existing CUDA backend, never for operator-only SDPA.
    try:
        from flash_attn import flash_attn_varlen_func
    except Exception as exc:  # pragma: no cover - production dependency
        raise RuntimeError(
            "flash_attn_varlen_func is required for routed Full-only attention"
        ) from exc
    return flash_attn_varlen_func


class PackedRoutedFullOnlyLayoutBuilder:
    """Build one packed selected-view query layout shared by all routed layers."""

    def __init__(self, grid_resolution: int = 64) -> None:
        if isinstance(grid_resolution, bool) or not isinstance(grid_resolution, Integral) or grid_resolution <= 0:
            raise ValueError("grid_resolution must be a positive integer")
        self.grid_resolution = int(grid_resolution)

    @staticmethod
    def _gather_grid(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 4:
            raise ValueError(f"Expected routed grid [B,R,R,R], got {tuple(grid.shape)}")
        return grid[
            coords[:, 0].long(),
            coords[:, 1].long(),
            coords[:, 2].long(),
            coords[:, 3].long(),
        ]

    def prepare(
        self,
        *,
        x: SparseTensor,
        selected_view: torch.Tensor,
        batch_size: int,
        views: int,
    ) -> Dict[str, torch.Tensor | int]:
        """Return the packed-query fields consumed directly by routed attention."""
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in (batch_size, views)):
            raise ValueError("batch_size and views must be positive integers")
        if not isinstance(x, SparseTensor):
            raise TypeError("Routed queries must be a SparseTensor")
        coords = x.coords
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"Expected sparse coords [N,4], got {tuple(coords.shape)}")
        if (coords.dtype not in (torch.int32, torch.int64) or coords.device != x.device
                or len(coords) != len(x.feats)):
            raise ValueError("Sparse coords must be integer and match query rows/device")
        if not len(coords):
            raise ValueError("Routed attention received no sparse queries")
        expected_grid = (batch_size,) + (self.grid_resolution,) * 3
        if (not isinstance(selected_view, torch.Tensor) or tuple(selected_view.shape) != expected_grid
                or selected_view.dtype not in (torch.int32, torch.int64)
                or selected_view.device != coords.device):
            raise ValueError(f"selected_view must be an integer grid {expected_grid} on {coords.device}")
        if torch.any((selected_view < 0) | (selected_view >= views)).item():
            raise ValueError("selected_view contains a non-existent view")
        if x.shape[0] != batch_size:
            raise ValueError("Sparse query batch count disagrees with the view bank")
        xyz = coords[:, 1:].long()
        if xyz.numel() and (
            int(xyz.min().item()) < 0 or int(xyz.max().item()) >= self.grid_resolution
        ):
            raise ValueError(
                f"Sparse coords must be in [0,{self.grid_resolution - 1}]"
            )
        batch_id = coords[:, 0].long()
        if batch_id.numel() and (
            int(batch_id.min().item()) < 0 or int(batch_id.max().item()) >= batch_size
        ):
            raise ValueError("Sparse query batch id is out of range")

        selected = self._gather_grid(selected_view, coords).long()
        if selected.numel() and (
            int(selected.min().item()) < 0 or int(selected.max().item()) >= views
        ):
            raise ValueError("selected_view contains a non-existent view")

        group_id = batch_id * int(views) + selected
        group_count = int(batch_size) * int(views)
        counts = torch.bincount(group_id, minlength=group_count)
        active_groups = torch.nonzero(counts > 0, as_tuple=False).flatten()
        if active_groups.numel() == 0:
            raise ValueError("Routed attention received no sparse queries")

        try:
            order = torch.argsort(group_id, stable=True)
        except TypeError:
            order = torch.argsort(group_id)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)

        query_lengths = counts[active_groups].to(torch.int32)
        cu_q = torch.zeros(
            active_groups.numel() + 1, dtype=torch.int32, device=coords.device
        )
        cu_q[1:] = torch.cumsum(query_lengths, dim=0)
        full_tokens = self.grid_resolution ** 2 + 5
        cu_k = (
            torch.arange(
                active_groups.numel() + 1, dtype=torch.int32, device=coords.device
            )
            * full_tokens
        )
        return {
            "routed_query_order": order,
            "routed_query_inverse": inverse,
            "routed_active_groups": active_groups.long(),
            "routed_cu_seqlens_q": cu_q,
            "routed_cu_seqlens_k": cu_k,
            "routed_max_seqlen_q": int(query_lengths.max().item()),
        }


def _validate_sdpa_layout(x, context, batch, views, full_tokens):
    """Reject malformed packing before indexing or projecting the view bank.

    The prepared layout is bound to unchanged sparse rows and selected views;
    the model must rebuild it when either changes (including CFG branches).
    """
    from .sdpa import validate_cumulative_lengths

    count = len(x.feats)
    if not count:
        raise ValueError("Routed attention received no sparse queries")
    if (x.coords.ndim != 2 or tuple(x.coords.shape) != (count, 4)
            or x.coords.dtype not in (torch.int32, torch.int64) or x.coords.device != x.device
            or x.shape[0] != batch):
        raise ValueError("Sparse coordinates must match query rows/device and view-bank batches")
    resolution = isqrt(full_tokens - 5)
    if torch.any((x.coords[:, 1:] < 0) | (x.coords[:, 1:] >= resolution)).item():
        raise ValueError("Sparse query coordinates are outside the routed grid")
    indexes = {}
    for name in ("routed_query_order", "routed_query_inverse", "routed_active_groups"):
        value = context[name]
        if (not isinstance(value, torch.Tensor) or value.ndim != 1
                or value.dtype != torch.int64 or value.device != x.device):
            raise ValueError(f"{name} must be an int64 vector on {x.device}")
        indexes[name] = value
    order, inverse, groups = (indexes[n] for n in (
        "routed_query_order", "routed_query_inverse", "routed_active_groups"))
    expected = torch.arange(count, device=x.device)
    if len(order) != count or len(inverse) != count or not torch.equal(order.sort().values, expected):
        raise ValueError("Routed order must be a complete query permutation")
    if (torch.any((inverse < 0) | (inverse >= count)).item()
            or not torch.equal(order[inverse], expected)):
        raise ValueError("Routed inverse must invert the query order")
    if (not len(groups) or torch.any((groups < 0) | (groups >= batch * views)).item()
            or torch.any(groups[1:] <= groups[:-1]).item()):
        raise ValueError("Active groups must be unique, increasing batch/view IDs in range")
    qs = validate_cumulative_lengths(context["routed_cu_seqlens_q"], count,
                                     name="routed_cu_seqlens_q", device=x.device)
    ks = validate_cumulative_lengths(context["routed_cu_seqlens_k"], len(groups) * full_tokens,
                                     name="routed_cu_seqlens_k", device=x.device)
    if (len(qs) != len(groups) + 1 or len(ks) != len(groups) + 1
            or any(a == b for a, b in zip(qs, qs[1:]))
            or any(b - a != full_tokens for a, b in zip(ks, ks[1:]))):
        raise ValueError("Routed boundaries must cover each active group with a complete view bank")
    maximum = context["routed_max_seqlen_q"]
    if (isinstance(maximum, bool) or not isinstance(maximum, Integral)
            or maximum != max(b - a for a, b in zip(qs, qs[1:]))):
        raise ValueError("routed_max_seqlen_q must match actual query lengths")
    for group, a, b in zip(groups.tolist(), qs, qs[1:]):
        if torch.any(x.coords[order[a:b], 0] != group // views).item():
            raise ValueError("Routed query groups must preserve batch identity")
    # The production prepared condition includes the original selected grid.
    # Check it when present, so stale positive/negative layouts fail loudly.
    if "selected_view" in context:
        selected = context["selected_view"]
        if (not isinstance(selected, torch.Tensor)
                or tuple(selected.shape) != (batch, resolution, resolution, resolution)
                or selected.dtype not in (torch.int32, torch.int64) or selected.device != x.device):
            raise ValueError("selected_view must match the routed integer grid and device")
        chosen = PackedRoutedFullOnlyLayoutBuilder._gather_grid(selected, x.coords)
        for group, a, b in zip(groups.tolist(), qs, qs[1:]):
            if torch.any(chosen[order[a:b]] != group % views).item():
                raise ValueError("Prepared layout disagrees with selected_view; rebuild it after routing changes")


class RoutedFullOnlyFlashCrossAttention(nn.Module):
    """Each query attends to all tokens of exactly one selected real view."""

    full_tokens = 4101

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or channels <= 0 or ctx_channels <= 0 or channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = int(channels)
        self.ctx_channels = int(ctx_channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.qk_rms_norm = bool(qk_rms_norm)
        if self.num_heads != 12 or self.head_dim != 128:
            raise ValueError("production Full-only Flash path requires 12 heads x 128")
        if config.ATTN == 'flash_attn':
            _flash_attention()
        elif config.ATTN != 'sdpa':
            raise ValueError("Routed Full-only attention supports flash_attn or sdpa")

        self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
        self.to_kv = nn.Linear(ctx_channels, channels * 2, bias=qkv_bias)
        self.to_out = nn.Linear(channels, channels)
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)

    def forward(
        self,
        x: SparseTensor,
        context: Dict[str, torch.Tensor],
    ) -> SparseTensor:
        full = context["full_bank"]
        if full.ndim != 4 or int(full.shape[2]) != self.full_tokens:
            raise ValueError(
                f"Expected Full bank [B,V,{self.full_tokens},C], got {tuple(full.shape)}"
            )
        batch, views, _, width = full.shape
        if int(width) != self.ctx_channels:
            raise ValueError(
                f"Expected Full width {self.ctx_channels}, got {int(width)}"
            )
        if config.ATTN == 'sdpa':
            if self.full_tokens not in (1029, 4101):
                raise ValueError("Routed SDPA requires the production 1029 or 4101 token bank")
            if (full.device != x.device or full.dtype != x.dtype or x.feats.ndim != 2
                    or x.feats.shape[1] != self.channels or batch <= 0 or views <= 0):
                raise ValueError("Full bank and sparse queries must have matching device/dtype and valid shapes")
            _validate_sdpa_layout(x, context, int(batch), int(views), self.full_tokens)

        q = self.to_q(x.feats).view(-1, self.num_heads, self.head_dim)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
        order = context["routed_query_order"]
        inverse = context["routed_query_inverse"]
        q = q[order].contiguous()

        active_groups = context["routed_active_groups"]
        flat_full = full.reshape(
            int(batch) * int(views), self.full_tokens, width
        )
        active_full = flat_full.index_select(0, active_groups).contiguous()
        kv = self.to_kv(active_full).view(
            active_groups.numel(),
            self.full_tokens,
            2,
            self.num_heads,
            self.head_dim,
        )
        k, v = kv.unbind(dim=2)
        if self.qk_rms_norm:
            k = self.k_rms_norm(k)
        k = k.contiguous()
        v = v.contiguous()

        k_packed = k.reshape(-1, self.num_heads, self.head_dim)
        v_packed = v.reshape(-1, self.num_heads, self.head_dim)
        if config.ATTN == 'sdpa':
            from .sdpa import packed_sdpa
            out_grouped = packed_sdpa(
                q, k_packed, v_packed,
                context["routed_cu_seqlens_q"], context["routed_cu_seqlens_k"],
            )
        elif config.ATTN == 'flash_attn':
            out_grouped = _flash_attention()(
                q,
                k_packed,
                v_packed,
                context["routed_cu_seqlens_q"],
                context["routed_cu_seqlens_k"],
                int(context["routed_max_seqlen_q"]),
                self.full_tokens,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
        else:
            raise ValueError("Routed Full-only attention supports flash_attn or sdpa")
        out = out_grouped[inverse]
        return x.replace(self.to_out(out.reshape(-1, self.channels)))
