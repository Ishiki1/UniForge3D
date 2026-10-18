# Mac attention adapter: operator proof, not a full pipeline port

This cut adds an explicit PyTorch `sdpa` backend for **full sparse attention**
and `RoutedFullOnlyFlashCrossAttention`. Ordinary sparse attention still defaults
to `flash_attn`; the existing CUDA implementation and checkpoint parameter names
and dimensions remain intact. The routed class retains its production name and
12 heads × 128 channels. No checkpoint weights or private asset inputs accompany
these tests.

Importing `uniforge3d` or its sparse operators no longer eagerly imports the
pipeline and its CUDA mesh/model dependencies. Accessing
`uniforge3d.UniForge3DImageTo3DPipeline` resolves the same public pipeline class
lazily; constructing/running that pipeline still requires the remaining port.

## Selection and scope

Set `SPARSE_ATTN_BACKEND=sdpa` before import, or call
`uniforge3d.modules.sparse.config.set_attn_backend('sdpa')` before constructing
attention. `ATTN_BACKEND=sdpa` also selects sparse SDPA when the sparse override
is absent. Invalid backend names raise `ValueError`, rather than silently
retaining FlashAttention. The sparse setting takes precedence over the shared
setting. Routed attention supports `flash_attn` and `sdpa`; other existing
backends continue to serve their existing full sparse attention paths.

This adapter does **not** support windowed or double-windowed sparse SDPA.
Their original implementation is unchanged. Do not select those modes with
`sdpa`. The adapter does not set PyTorch's MPS CPU fallback, change tensor devices,
change input precision, truncate tokens, average keys, or substitute local
attention. Native unsupported operations must surface for a separate decision.

## Numerical and layout contract

The shared packed helper accepts Q `[Tq,H,Dq]`, K `[Tk,H,Dq]` and V `[Tk,H,Dv]`.
It performs noncausal attention with zero dropout and scale `1/sqrt(Dq)`.
Every query tile attends to every key of its sequence. Sequences are evaluated
independently; outputs retain the original query packing, device and dtype.
There is no giant padded batch, global block-diagonal score matrix, or key
chunking. Native SDPA kernel selection remains PyTorch's responsibility;
floating-point results need not be bit-identical to FlashAttention.

The public `sparse_scaled_dot_product_attention` keeps its one-input packed QKV,
two-input Q/KV, and three-input Q/K/V signatures, including positional,
keyword and mixed calls. At least Q or KV must be ragged; all-dense calls remain
outside this API. Dense Q returns `[B,L,H,Dv]`; ragged Q returns its original
container type through `replace`, preserving coordinates and caches. The
three-input form permits `Dv != Dq`.

Validation rejects wrong dimensions, fused axes, head counts, key/value token
counts, devices, dtypes, batches, non-covering or noncontiguous ragged layouts,
non-integer/nonmonotone/incomplete cumulative boundaries, disagreeing sparse
K/V coordinates, and sparse coordinates inconsistent with batch slices.
Cumulative lengths are int32/int64 on the feature device. Empty query sequences
(including all-empty public inputs) return no rows. A nonempty query sequence
with no keys raises `ValueError`; an undefined empty-key softmax is not replaced
with fabricated values. Routed attention preserves its no-query rejection.

The real routed layout builder validates integer grid/coordinate types,
matching dimensions/devices and index bounds. It groups by **batch × selected
view**, retains stable query packing and inverse scatter, and uses all 1,029
(grid32) or 4,101 (grid64) tokens, including the five special tokens. Routed
execution checks complete permutations, inverse consistency, unique sorted
active groups, exact query/key coverage, actual maximum query length and batch
identity before indexing/projection. When `selected_view` accompanies the
prepared context (as in the production model), its per-query assignments must
also match the layout. If a caller supplies only the prepacked fields without
the source grid, that source-view agreement cannot be verified: the caller must
rebuild the layout whenever rows or selected views change. Positive and negative
CFG conditions require separate layouts. The production Q/K RMS normalization
is unchanged: normalize in fp32, multiply learned gamma and `sqrt(D)`, then
restore activation dtype.

## Score/workspace estimate

Defaults are `SPARSE_SDPA_SCORE_WORKSPACE_BYTES=536870912` (512 MiB) and
`SPARSE_SDPA_QUERY_CHUNK_SIZE=256`. Both require positive integers. The effective
query tile is the smaller of the requested cap and the rows the estimate admits.
If even one query row cannot fit the estimate, the call fails before attention.
Budget rejection never changes precision, resolution, view count or key count.

For key length `K`, heads `H`, widths `Dq,Dv`, and
`A=max(4,input_element_bytes)`, the conservative estimate is:

```text
fixed_bytes   = 1 MiB + 2 * H * K * (Dq + Dv) * A
per_query_row = H * (3 * K + 2 * Dq + 2 * Dv) * A
rows          = min(query_cap, floor((budget - fixed_bytes) / per_query_row))
```

This reserves two accumulation-precision K/V copies, three score arrays and
query/output scratch, even when the selected native kernel uses less memory.
Half and bfloat16 do not get a falsely reduced fp16 score estimate. Float64
uses eight-byte accumulation. The default admits both routed banks and the
actual four-view all-view grid64 bank: **16,404 keys, 12 × 128, 55 query rows**
under this formula. A larger bank may require a larger explicitly chosen budget.

**This is a score/workspace estimate, not a total RSS/device-memory guarantee.**
Inputs, persistent output, projections and projected active banks, model weights,
autograd retention, unrelated processes, allocator overhead and undocumented
native workspace are outside its scope. The routed wrapper projects all active
banks before the helper runs. Production inference must still use inference/no-grad
mode and a separate measured memory/resource policy. Query tiling alone does
not make an arbitrarily large full pipeline safe. Peak native memory and the
actual MPS kernel are not measured by these CPU tests.

## CPU evidence

Run from the repository root with the installed runtime, without installing any
CUDA/Metal extensions or loading models:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3.11 -m pytest tests -q -s
```

The test harness also sets `torch.set_num_threads(1)`. All numerical tensors are
CPU tensors; a `meta` tensor only tests device mismatch rejection. No GPU/MPS,
`flex_gemm` (whose CPU tensor interface can execute Metal), model weights,
engine, server or guard operation is used.

The independent oracle is explicit CPU float64 matrix multiplication and stable
exponential softmax, not SDPA or a second call to this implementation. Routed
reference projections are computed in float64 then rounded to the activation
dtype, and RMS normalization independently follows the prescribed fp32
arithmetic/rounding. Tests assert absolute and relative elementwise tolerance,
finiteness, output dtype/shape, exact routing metadata and preserved caches.
The routed fp32 assertion is `atol=5e-6, rtol=3e-4`; public fp32 is
`atol=2e-6, rtol=2e-5`. Quantized input tests have separate dtype-appropriate
bounds, not a relaxed fp32 gate. Wrong-view, wrong-scatter and omitted-normalization
alternatives are explicitly shown to disagree with the oracle.

Coverage includes all signatures and mixed dense/ragged inputs; unequal and
empty lengths; one-key/mean-value invariants; extreme finite logits; coupled K/V
permutation; noncontiguous inputs; forced query tiles with every key preserved;
impossible budgets; real routed module/layout at both banks and 1/15/16/17/31/32/33
query lengths; multi-batch/view isolation; distinct CFG zero-view routing;
Q/K normalization and projection bias combinations; strict synthetic state-dict
loading against the existing cross-attention class; and fresh-process imports
with FlashAttention genuinely absent. The four-view special-token fixture
uses the actual public sparse cross-attention API.

Author environment on 18 October 2026: Python 3.11.15, PyTorch 2.12.0,
pytest 9.0.3, macOS 26.6.2 arm64. The eight main routed oracle cases measured
maximum absolute output error from `8.94069672e-7` to `1.37090683e-6`; the
largest per-row L2 error was `1.01698202e-5`. Exact test-run result is recorded
with the candidate commit's review handoff: **113 passed in 2.81s**, exit 0.
`git diff --check` also passed. The first draft's 64 MiB default
correctly refused grid64; the final 512 MiB setting also covers the four-view
production all-view case. That refusal was corrected by an explicit budget
choice, with no removed views or precision reduction.

## Remaining proof gates

This does not establish MPS execution, MPS performance/peak memory, CUDA parity,
real-checkpoint loading, full model execution, final geometry/PBR fidelity or
an engine-ready game asset. Independent source review precedes a separately
bounded MPS probe. Device/precision policy, conditioning, sparse convolution,
mesh extraction, texture baking and full stage execution remain subsequent
adapter work. Full geometry and textures remain the destination.
