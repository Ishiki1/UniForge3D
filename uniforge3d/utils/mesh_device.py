"""Explicit device boundary for Metal's CPU-returning remeshing kernels."""
from threading import RLock

import torch

_REMESH_LOCK = RLock()
_TABLES = ('edge_neighbor_voxel_offset', 'quad_split_1_n', 'quad_split_1_p',
           'quad_split_2_n', 'quad_split_2_p')


def remesh_narrow_band_dc(vertices, faces, *, center, **kwargs):
    """Keep CUDA unchanged; orchestrate Metal topology explicitly on the CPU.

    Metal hash/contour bindings return CPU tensors. CPU query tensors also make
    BVH dispatch synchronous and return CPU results. The native kernels still
    use Metal, so callers must hold their normal accelerator resource guard.
    The same BVH and every algorithm parameter are passed through unchanged.
    """
    import cumesh
    operation = cumesh.remeshing.remesh_narrow_band_dc
    if vertices.device.type != 'mps':
        return operation(vertices, faces, center=center, **kwargs)
    device = vertices.device
    with _REMESH_LOCK:
        # The upstream wrapper caches tables on its first call's device. A
        # prior direct MPS call must not contaminate this CPU orchestration.
        for name in _TABLES:
            value = getattr(operation, name, None)
            if value is not None:
                setattr(operation, name, value.cpu())
        out_vertices, out_faces = operation(
            vertices.to(device='cpu', dtype=torch.float32).contiguous(),
            faces.to(device='cpu', dtype=torch.int32).contiguous(),
            center=center.to(device='cpu', dtype=torch.float32).contiguous(),
            **kwargs,
        )
    return (out_vertices.to(device=device, dtype=torch.float32).contiguous(),
            out_faces.to(device=device, dtype=torch.int32).contiguous())
