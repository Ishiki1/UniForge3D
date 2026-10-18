from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ...attention.rope import real_rotary_phases, apply_real_rotary_embedding


class SparseRotaryPositionEmbedder(nn.Module):
    def __init__(
        self, 
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        phase_format: Literal["auto", "real"] = "auto",
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        if phase_format not in ("auto", "real"):
            raise ValueError("phase_format must be auto or real")
        self.phase_format = phase_format
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** (self.freqs))
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        if not phases.is_complex():
            return apply_real_rotary_embedding(x, phases)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: SparseTensor, k: Optional[SparseTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (SparseTensor): [..., N, H, D] tensor of queries
            k (SparseTensor): [..., N, H, D] tensor of keys
        """
        assert q.coords.shape[-1] == self.dim + 1, "Last dimension of coords must be equal to dim+1"
        phases_cache_name = f'rope_phase_{self.dim}d_freq{self.rope_freq[0]}-{self.rope_freq[1]}_hd{self.head_dim}'
        real_phases = self.phase_format == "real" or q.device.type == "mps"
        if real_phases:
            # SparseTensor.to preserves the cache; never reuse CPU complex
            # phases or another device's real phases after a device move.
            phases_cache_name += f'_real_{q.device}'
        phases = q.get_spatial_cache(phases_cache_name)
        if phases is None:
            coords = q.coords[..., 1:]
            if real_phases:
                phases = real_rotary_phases(coords, self.freqs, self.head_dim)
            else:
                phases = self._get_phases(coords.reshape(-1)).reshape(*coords.shape[:-1], -1)
            if not real_phases and phases.shape[-1] < self.head_dim // 2:
                padn = self.head_dim // 2 - phases.shape[-1]
                phases = torch.cat([phases, torch.polar(
                    torch.ones(*phases.shape[:-1], padn, device=phases.device),
                    torch.zeros(*phases.shape[:-1], padn, device=phases.device)
                )], dim=-1)
            q.register_spatial_cache(phases_cache_name, phases)
        q_embed = q.replace(self._rotary_embedding(q.feats, phases))
        if k is None:
            return q_embed
        k_embed = k.replace(self._rotary_embedding(k.feats, phases))
        return q_embed, k_embed
