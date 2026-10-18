# Exact integer boundaries for the Mac view router

The view router retains the original 448×448 point maps, all views, confidence
selection, connected components, tie rules and single-hop neighbor selection.
Its 32³ and 64³ grids use the original 32×32 and 64×64 patch grids.

Two specific operations need explicit CPU integer boundaries on MPS:

- Five `int64` extrema reductions use the original `scatter_reduce_` with
  `include_self=True` on a copy of the initialized destination, then copy the
  exact result into the original MPS destination. This preserves aliases,
  sentinels, empty reductions, ties and the full integer range.
- The packed-key `unique_consecutive` runs in `int64` on CPU and returns its
  values, inverse indices and counts to the original MPS device. A four-view
  64³ workload can exceed the signed 32-bit range even at batch size one.

PyTorch 2.12.0 at revision `7661cd9c6b841b62b7f411aa52ec51f05457263b`
narrows MPS `int64` input to `int32` in its
[unique implementation](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/mps/operations/Unique.mm).
A native diagnostic found the first actual routing mismatch at this operation:
the copied keys, integer arithmetic, stable sort and sorted keys matched CPU,
but unique values changed. Inputs separated by 2³² also merged groups, changing
inverse indices and counts. This can produce a negative dense destination.

These are narrow, explicit boundaries. Generic MPS fallback stays disabled.
Floating confidence reductions and accumulation, stable sorting, the later
small group-ID unique, and all original CPU/CUDA operations remain unchanged.
No keys pass through floating point or a narrower integer type. These boundaries
incur synchronization and transfer costs that must be included in measurements.

## Evidence

The portable CPU suite is `tests/test_view_router.py`. Its 36 checks passed on
Python 3.11.15 / PyTorch 2.12.0 in 0.37 seconds. It uses a frozen copy of the
upstream router plus independent analytic dense winners and Python integer
oracles. Cases cover both grids, multiple batches, every neighbor offset,
component and view ties, invalid masks, threshold selection, full integer
extremes, values above 2⁵³, 2³² aliases, strided storage and empty inputs.

The expanded native MPS suite passed all 26 cases on the same runtime and
macOS 26.6.2 arm64, with generic fallback disabled. Both complete dense grids
matched the frozen CPU router and analytic winners exactly. The three unique
outputs and all five integer extrema boundaries were checked on actual MPS
tensors, including full-range and strided cases. Source hashes and CPU/MPS RNG
states were unchanged. The supervised child completed in 1.688 seconds and
released its resource guard; process peak RSS was 582,795,264 bytes, not a GPU
peak or full-pipeline memory estimate.

These operator results do not establish trained conditioning, CUDA equivalence,
dense-workload performance, complete asset generation or final visual fidelity.
