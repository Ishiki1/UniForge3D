# Mac flexible dual grid extraction

This inference adapter replaces the dense voxel lookup in the pinned
`trellis2-apple` implementation with a sparse sorted-key lookup and chunks
active-edge neighbor queries. The geometry decoder selects this adapter only
when coordinates are on MPS. CPU/CUDA retain the original `o_voxel.convert`
extractor, imported lazily so its absence does not prevent the MPS path from
being imported. No complete model/pipeline run is established by this integration.

## Supported boundary

`mac_flexible_dual_grid_to_mesh` accepts int32/int64 coordinates `(N,3)`, boolean
intersection flags `(N,3)`, dual offsets `(N,3)`, and the decoder's split weights
`(N,1)`. Float coordinates and non-boolean flags raise `TypeError`, even on empty
inputs. `train=True` raises `ValueError`. A `None` split weight retains the
original angle heuristic. These deliberate input restrictions avoid promising
parity for unsupported coordinate/flag types.

For supported inputs, active-edge order, neighbor winding, strict `>` split
comparisons (ties choose split 2), output row order, vertex values, int32 faces,
and AABB/voxel transforms retain the original operation order. Integer duplicate
coordinates retain last-writer lookup behavior; integer out-of-range coordinates
retain the original flat-index wrapping/error behavior. The original heuristic's
cross-product axis for exactly three quads and its second-split arithmetic are
preserved. No geometry cleanup or resolution reduction is performed.

## Explicit CPU topology boundary

Coordinates and flags are copied to CPU. NumPy constructs sorted sparse keys,
enumerates active edges in row/axis order, and resolves neighbors in chunks.
The resulting int32 indices return to the original device. Vertex arithmetic
and split decisions run in PyTorch on that device. This is an explicit CPU
boundary, not generic MPS fallback. Bounded native MPS evidence is recorded
below; CPU-only float64 parity is not a native MPS dtype claim.

## Memory scope

The original dense int64 lookup allocates `1024**3 * 8 = 8,589,934,592` bytes
(8 GiB) at a logical 1024 cubed grid, before neighbor and output memory. This is
an avoidable allocation, not proof that the original can never run on a Mac.

The new lookup and active-edge lists are O(N) in input voxels; query workspace
is O(edge_chunk_size); retained quads and final geometry are O(L). No array is
allocated in proportion to logical grid volume. The default chunk is 262,144
active edges. The corner array alone costs 96 bytes per edge, with additional
offset gathers, lookup keys, search indices and masks. This is a query-workspace
control, not a process RSS ceiling. Finalization/concatenation still scale with
the full output and may dominate memory for large decodes.

The tests include a 20,000-row synthetic sparse case in a logical 1024 cubed grid
and a fresh-process RSS check. They detect accidental grid-volume allocation for
that case; they do not establish a general memory limit or production capacity.
Larger synthetic measurements and native/full-model evidence must be reported
separately with their exact inputs and metric scope.

## CPU verification

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3.11 -m pytest -q tests/test_mac_dual_grid.py
```

The dictionary oracle fixes neighbor/split geometry independently from the
implementation. Twelve winding/offset mutation controls must reject altered
production tables across all three axes. The original upstream reference is
required and SHA256-pinned; a missing or changed fixture fails rather than skips.
Coverage also includes empty/no-face cases, incomplete boundaries, anisotropic
grids, shuffled/duplicate coordinates, input rejection, split ties, legacy
heuristic behavior and chunk invariance. This suite executes only CPU work.

Candidate author run on 18 October 2026: **77 passed in 3.49 seconds**, exit 0,
using Python 3.11 / PyTorch 2.12 with one intra-op/inter-op CPU thread. Native GPU
package imports were blocked in the test runner. This is CPU evidence only.

## Bounded native verification

On 18 October 2026 the integrated adapter passed **30 MPS cases and 87 exact
chunk comparisons** across float32, float16 and bfloat16 inputs. Returned
vertices are float32 and faces int32, on MPS. The tests exercised the exact
small decoder dispatcher through an isolated AST import boundary, calling the
real adapter with MPS tensors; they did not construct a trained decoder.

A separate literal dictionary oracle was checked against the pinned upstream
CPU implementation on every small case. Fixtures cover both weighted splits,
strict ties, the preserved three-quad and degenerate angle heuristics, empty
and inactive rows, anisotropic grids, shuffled holes/boundaries, voxel-size
precedence and chunk invariance. Dyadic coordinates/weights make exact vertex
comparisons meaningful for these fixtures. The logical 1024 cubed case uses
4,096 boundary-plane rows and 7,938 triangles; it uses the dictionary oracle
without allocating the upstream 8 GiB dense table. This is not a full-scale
decoder or production-memory proof.

The native process used PyTorch 2.12 on Apple Silicon, generic fallback disabled,
one CPU thread and an external 120-second supervisor. It completed in 1.84
seconds including startup, with unchanged source hashes, exit 0 and confirmed
process/lock release. Peak process RSS was 288,309,248 bytes; this is not GPU peak
memory. No full trained geometry, CUDA-equivalence or completed asset claim
follows from this test.

## Attribution

The geometry algorithm and tables derive from Microsoft TRELLIS.2 via
`trellis2-apple` commit `17347247c91c36c8cdc1896234983e878a457bba`.
The sparse lookup/chunked query implementation is adapted around that algorithm.
`tests/fixtures/upstream_flexible_dual_grid.py` is an unchanged test reference (SHA256
`deb1c42444de13792255c09d261de2d194b7b61f92860c49d2acbc5efbbc17af`).
Distribute `THIRD_PARTY_LICENSES.txt`, which includes the upstream Microsoft
copyright and complete MIT permission notice, with the derived source/reference.
