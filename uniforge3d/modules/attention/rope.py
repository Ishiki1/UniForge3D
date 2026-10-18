from typing import *
import torch
import torch.nn as nn


def real_rotary_phases(indices: torch.Tensor, freqs: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Represent the original unit complex phases as real (cos, sin) pairs."""
    angles = torch.outer(indices.reshape(-1), freqs.to(indices.device))
    pairs = torch.stack((angles.cos(), angles.sin()), dim=-1)
    pairs = pairs.reshape(*indices.shape[:-1], indices.shape[-1] * len(freqs), 2)
    padding = head_dim // 2 - pairs.shape[-2]
    if padding:
        identity = pairs.new_zeros((*pairs.shape[:-2], padding, 2))
        identity[..., 0] = 1
        pairs = torch.cat((pairs, identity), dim=-2)
    return pairs


def apply_real_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    """Real arithmetic equivalent of (a+ib)*(cos(theta)+i*sin(theta))."""
    if phases.is_complex() or phases.shape[-2:] != (x.shape[-1] // 2, 2):
        raise ValueError("Real rotary phases must have shape [..., head_dim/2, 2]")
    if phases.device != x.device:
        raise ValueError("Rotary phases and features must share a device")
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cosine = phases[..., 0].unsqueeze(-2)
    sine = phases[..., 1].unsqueeze(-2)
    real = pairs[..., 0] * cosine - pairs[..., 1] * sine
    imaginary = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return torch.stack((real, imaginary), dim=-1).flatten(-2).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
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
    
    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        if not phases.is_complex():
            return apply_real_rotary_embedding(x, phases)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        assert indices.shape[-1] == self.dim, f"Last dim of indices must be {self.dim}"
        if self.phase_format == "real" or indices.device.type == "mps":
            return real_rotary_phases(indices, self.freqs, self.head_dim)
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
                padn = self.head_dim // 2 - phases.shape[-1]
                phases = torch.cat([phases, torch.polar(
                    torch.ones(*phases.shape[:-1], padn, device=phases.device),
                    torch.zeros(*phases.shape[:-1], padn, device=phases.device)
                )], dim=-1)
        return phases
