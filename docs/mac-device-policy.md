# Explicit Mac device and precision policy

This adapter cut makes device selection and conditioning precision explicit and
provides a real-valued RoPE path for MPS. It does not establish that the full
geometry-and-material pipeline runs. Model weights, conditioning outputs on real
images, native RoPE behavior, sparse/mesh dependencies and complete inference
require their own execution evidence.

## Selecting a device

The pipeline constructor and `from_config` / `from_pretrained` accept a CUDA
ordinal, string or `torch.device`. Integers retain their original CUDA meaning:
`device=2` means `cuda:2`. Default `device=0` is unchanged. The command-line entry
adds `--device mps`; `--device cuda:2`, `--device 2` and `--device cpu` are also
accepted. Invalid device types, negative ordinals and nonzero CPU/MPS indices
fail explicitly. There is no automatic CUDA-to-MPS or accelerator-to-CPU switch.

MPS selection activates the dense and sparse `sdpa` attention backends **before
model construction**. CPU/CUDA selection leaves configured backends unchanged.
The settings are process-global, as in the original implementation; separate
processes are required for simultaneous pipelines with conflicting backend
policies. Device parsing and backend configuration do not allocate accelerator
tensors or prove device availability. The CLI resolves its pipeline import only
after argument parsing and device configuration, so `--help` does not import
native mesh/model dependencies.

Start the process with generic `PYTORCH_ENABLE_MPS_FALLBACK` disabled. Selecting
MPS rejects an environment value of `1`; the adapter never enables fallback or
changes that variable. Do not toggle it after importing PyTorch and expect the
already-initialized runtime to change. Unsupported operations must receive
explicitly tested adapters or named CPU boundaries.

## Precision, randomness and memory release

`autocast_for_device` uses the actual selected device and the original requested
float16/bfloat16 precision. Float32 explicitly disables outer autocast on that
device, preserving the VGGT DPT/point head's full-precision region. The frozen
VGGT backbone and shape encoder use their configured dtype on MPS as on CUDA.
CPU uses the same declared autocast policy for reference work. There is no dtype
substitution to make an unsupported operation succeed.

`manual_cast` now queries autocast for the input tensor's device, including sparse
tensor containers. The former default query only observed CUDA and could cast
MPS activations despite active MPS autocast. Existing CUDA semantics are retained.
DINO/structure bfloat16 contexts, sampler parameters, noise generation on CPU,
seed calls, checkpoint loading and image transformations remain unchanged.

`move_model_to_device` prepares the explicit RoPE runtime boundary and then calls
the model's normal `.to`. The pipeline uses it both when loading a stage and
returning it to CPU. Cache release collects Python objects and touches only the
chosen accelerator. CUDA cleanup enters the selected CUDA ordinal; MPS cleanup
uses the MPS allocator. CPU cleanup does not query or empty accelerator caches.
These helpers do not implement GPU scheduling, stage resumability, a memory
ceiling, or full-model eviction when `low_vram=False` retains live references.

## RoPE without changing checkpoint buffers

The existing complex operation is `(a + ib) * (cos(theta) + i sin(theta))`.
The real path evaluates `a*cos - b*sin` and `a*sin + b*cos`, retaining frequency
order, adjacent feature pairing, identity padding, the original explicit fp32
feature conversion and final output dtype. Sparse MPS phases are formed from
real sine/cosine pairs. `phase_format='real'` permits CPU numerical verification;
`auto` retains the original complex CPU/CUDA path and chooses real phases for
MPS. No learned parameter or persistent state is added to either embedder.

Dense structure models precompute a persistent complex `rope_phases` buffer on
CPU. It must not be converted to MPS complex tensors or silently cast to real
values. The transfer helper calls `prepare_rope_device` before moving the model;
a scoped `_apply` override excludes that buffer during MPS transfer and retains
its original complex values on CPU. All learned parameters move normally.
`get_rope_phases()` returns a separate real-pair runtime cache for MPS, while
CPU/CUDA continue to receive the original phase buffer. Both the structure
model's forward and the three joint DiT/VGGT call sites use this accessor.

The persistent buffer's name, shape, dtype and values are unchanged in
`state_dict()`. The runtime cache is an ordinary nonpersistent attribute and is
invalidated on device preparation, tensor transforms, checkpoint load, phase
buffer replacement or phase-buffer version change. Inference tensors without a
version counter are not reused through the cache. Failed transfers restore the
persistent buffer. CPU/CUDA return transfers restore the ordinary buffer path.
Direct initial `.to('mps')` calls that bypass `move_model_to_device` do not prepare
this boundary and remain unsupported; the pipeline uses the prepared path.

Sparse real-phase cache keys include representation and device, preventing
reuse of CPU complex phases when `SparseTensor.to` preserves its spatial cache.
The existing rule that cached spatial metadata requires unchanged coordinate
rows still applies.

## CPU evidence and its limits

Run only ordinary PyTorch CPU tests, without loading weights or invoking native
Metal, MPS, CUDA or engine work:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3.11 -m pytest \
  tests/test_device_policy.py tests/test_rope_mac.py -q
```

The author CPU run on 18 October 2026 included these two suites and the existing
`test_sdpa_attention.py` / `test_routed_sdpa.py` regression suites: **177 passed
in 4.16 seconds**, exit 0, using Python 3.11 and PyTorch 2.12 with one CPU thread.

The tests execute real CPU autocast and nested fp32 regions, input-device-aware
`manual_cast`, CLI argument handling, and real/complex RoPE implementations.
An independent scalar float64 trigonometric oracle accounts for the prescribed
fp32 inputs and phase products. Coverage includes 12 heads × 128 dimensions,
float32/float16/bfloat16/float64 inputs, noncontiguous features, batching, frequency
padding and wrong-orientation rejection. A small real structure model produces
matching nonzero CPU outputs with complex and real runtime phases. Strict
checkpoint round trips retain keys, shapes and complex phase values; cache
invalidation and transfer failure recovery are checked.

Accelerator selection/cache operations are spies in these tests, and the MPS
buffer-transfer control test simulates the reported destination while traversing
real CPU parameters. They are **control-path tests, not native MPS proof**. The
whole pipeline and checkpoint-backed VGGT extractor are not executed. Source
checks verify their helper wiring and unchanged CPU noise generation. The prior
attention-only MPS proof does not cover these new device/RoPE changes. A separate
bounded native probe and independent source review are required before claiming
native compatibility for this cut.
