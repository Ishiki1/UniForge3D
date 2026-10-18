# Apple Silicon mesh and material boundary

The Mac bake uses pinned mtlmesh, mtlbvh and mtldiffrast packages. Native mesh
reads and UV maps may return CPU tensors; positions, triangles, normal indexing,
rasterization and BVH projection are normalized explicitly to MPS. The UV bake
uses one batch, float32 positions and int32 triangles. Geometry cleanup,
remeshing, UV chart settings, material channel packing, axis transform and UV
flip retain the upstream algorithm. This is inference only.

The sparse material sampler uses a sorted CPU coordinate index and bounded
CPU neighbor lookup, then float32 weighted sums on the feature device. This is
an explicit host/device boundary, not generic MPS fallback. It preserves CUDA's
floor(q - 0.5) voxel-centre convention, missing-neighbor normalization and the
max(weight_sum, 1e-12) denominator. The available Metal sampling kernel reads
float32 pointers without validating dtype and returns zero for tiny positive
weight sums; the adapter does not silently use those semantics. Query chunks
are limited to 65,536; storage scales with active voxels and output texels,
not grid volume. No texture channels are removed: base RGB, metallic,
roughness and alpha all survive into RGBA base-color and packed [0,R,M].

CPU analytic tests compare asymmetric ramps, missing neighbors, shuffled
coordinates, half precision inputs, exact centres, borders, empty support and
tiny positive weights against an independent dictionary oracle. Six bounded
native fixtures also pass on Apple Silicon: material sampling, BVH projection,
chunked UV rasterization, PBR GLB export, projected shape remeshing and PBR
export with remeshing enabled. The fixture is an asymmetric box with constant
material channels. Covered texels retain the expected values; inpainted atlas
padding is measured separately. This does not establish spatial texture
fidelity, production-grid performance or full trained-model results.
The upstream model generates no normal or emissive map; those require explicit
asset production steps after preserving the generated master.

The Metal remesher's Python topology orchestration runs explicitly on CPU
tensors because its native hash and contour operations return CPU arrays.
Its BVH and hash kernels still execute on Metal. The adapter normalizes the
five cached topology tables to CPU, preserving prior-call safety, and returns
the mesh on the original MPS device. Band, resolution, projection and topology
formulas are unchanged. CUDA retains its original dispatch. CPU/CUDA numerical
parity and high-resolution performance require their own evidence.

The tested native sources are:

| Package | Repository | Revision |
| --- | --- | --- |
| cumesh | https://github.com/pedronaugusto/mtlmesh | `7de3864f783407201486bf6c900a4bd76a4018a3` |
| mtlbvh | https://github.com/pedronaugusto/mtlbvh | `6b2a0f63b476ad8225695d092902104beafd7f58` |
| mtldiffrast | https://github.com/pedronaugusto/mtldiffrast | `c9499ba2e4ed6849e95ce5da7aae5bf61addb528` |
| flex_gemm | https://github.com/pedronaugusto/mtlgemm | `566c133781ce4992de3f77c3a97dd6feaae9d013` |

Build them inside the isolated environment from `requirements-mac.txt`, with
their pinned submodules and the Xcode Metal toolchain. Before building mtlbvh,
apply `patches/mtlbvh-pytorch-stream.patch` to its source checkout. It ends
PyTorch's active coalesced encoder before beginning BVH commands on the same
MPS stream; without it, a combined raster/BVH bake aborts even when isolated
operators pass. mtldiffrast and flex_gemm already use this stream boundary.
The small fixture passed with the patched library; the patch does not change
BVH geometry or distance calculations.
