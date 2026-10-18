"""Bounded inference sampling for the Mac material bake.

Topology lookup explicitly runs on CPU; weighted feature sums run on the input
device in float32. This is not PyTorch's generic MPS fallback. The half-voxel
centres, missing-neighbour renormalization and 1e-12 denominator match the
upstream FlexGEMM CUDA trilinear kernel (MIT), including tiny positive sums.
"""
from itertools import product
import math

import numpy as np
import torch


class SparseTextureSampler:
    def __init__(self, features, coords, grid_size, *, chunk_size=65536):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if features.ndim != 2 or not features.is_floating_point():
            raise ValueError("features must be floating [N,C]")
        if coords.ndim != 2 or coords.shape != (len(features), 3):
            raise ValueError("coords must be [N,3] matching features")
        if coords.dtype not in (torch.int32, torch.int64):
            raise ValueError("coords must be int32 or int64")
        dims = np.asarray(torch.as_tensor(grid_size).cpu(), dtype=np.int64)
        if dims.shape != (3,) or np.any(dims <= 0) or math.prod(map(int, dims)) >= 2**63:
            raise ValueError("grid_size must contain three positive bounded dimensions")
        xyz = coords.detach().cpu().numpy().astype(np.int64, copy=False)
        if np.any(xyz < 0) or np.any(xyz >= dims):
            raise ValueError("coordinates outside grid")
        self.dims = dims
        self.strides = np.array([int(dims[1])*int(dims[2]), int(dims[2]), 1], dtype=np.int64)
        keys = xyz @ self.strides
        self.order = np.argsort(keys)
        self.keys = keys[self.order]
        if len(keys) > 1 and np.any(self.keys[1:] == self.keys[:-1]):
            raise ValueError("duplicate sparse coordinates")
        self.features = features.detach().float().contiguous()
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError("nonfinite material features")
        self.chunk_size = chunk_size

    @torch.no_grad()
    def __call__(self, query):
        if query.ndim != 2 or query.shape[1] != 3 or not query.is_floating_point():
            raise ValueError("query must be floating [M,3] voxel coordinates")
        if query.device != self.features.device:
            raise ValueError("query and features must share a device")
        out = torch.zeros((len(query), self.features.shape[1]), device=query.device, dtype=torch.float32)
        if not len(self.keys):
            if not bool(torch.isfinite(query).all()):
                raise ValueError("nonfinite sample positions")
            return out
        offsets = np.array(list(product((0, 1), repeat=3)), dtype=np.int64)
        for start in range(0, len(query), self.chunk_size):
            q = query[start:start+self.chunk_size].detach().float().cpu().numpy()
            if not np.isfinite(q).all():
                raise ValueError("nonfinite sample positions")
            # Fully outside queries have zero support; clamp only their temporary
            # integer lookup positions so huge finite inputs cannot overflow int64.
            nearby = np.all((q >= -0.5) & (q <= self.dims + 0.5), axis=1)
            safe_q = np.where(nearby[:, None], q, np.float32(0.5))
            base = np.floor(safe_q - np.float32(0.5)).astype(np.int64)
            neighbours = base[:, None, :] + offsets[None, :, :]
            bounded = np.all((neighbours >= 0) & (neighbours < self.dims), axis=-1) & nearby[:, None]
            # Clipping prevents flat-key collisions across out-of-bounds axes.
            keys = np.clip(neighbours, 0, self.dims-1) @ self.strides
            loc = np.searchsorted(self.keys, keys).clip(max=len(self.keys)-1)
            valid = bounded & (self.keys[loc] == keys)
            weights = np.prod(np.float32(1) - np.abs(safe_q[:, None, :] - neighbours.astype(np.float32) - np.float32(0.5)), axis=-1)
            weights = np.where(valid, weights, np.float32(0))
            index = torch.from_numpy(self.order[loc].reshape(-1)).to(query.device)
            w = torch.from_numpy(weights).to(query.device)
            values = self.features.index_select(0, index).reshape(len(q), 8, -1)
            out[start:start+len(q)] = (values * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
        return out
