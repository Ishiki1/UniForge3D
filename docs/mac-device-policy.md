# Explicit Mac device and precision policy

This adapter cut makes device selection and conditioning precision explicit and
provides a real-valued RoPE path for MPS. It does not establish that the full
geometry-and-material pipeline runs. Complete inference and generated asset quality require their own execution
evidence. The component results below separate native operator checks from
trained visual-encoder inference.

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

## Bounded native and pretrained component evidence

On 18 October 2026 a separate supervised MPS probe passed 12 model-operation
cases: dense and sparse RoPE plus an actual small structure-model transfer and
forward in float32/float16/bfloat16, and native sparse-convolution cases. The
RoPE reference uses 12 heads by 128 dimensions; convolution checks include
missing neighbors and reuse with changed features. The small model uses random
weights and preserves its CPU checkpoint through CPU/MPS/CPU transfer. This
is component evidence, not a trained structure-model result.

The unchanged public DINOv3 encoder also loaded Meta's original ViT-L/16
LVD-1689M snapshot `ea8dc2863c51be0a264bab82070e3e8836b02d51`. All 415 tensor
keys/shapes/dtypes and eight exact tensor sentinels matched the SafeTensors
checkpoint. Four-view 128-pixel CPU and MPS fp32 features agreed with maximum
absolute error 0.00009871 and RMS error 0.000003258. Four-dimensional/five-
dimensional reshape paths agreed exactly. Separate bf16-autocast MPS runs at
512 and 1024 retained all four views, all 24 layers, five special tokens and
all patch tokens, returning finite float32 banks of shape `[1,4,1029,1024]`
and `[1,4,4101,1024]`. Learned HF final normalization and HF model.forward were
excluded by observation hooks; the public non-affine normalization remained.
These use synthetic images; full-size CPU/CUDA parity and asset quality were
not tested. Each fresh process ran with generic fallback disabled, offline
weights, source/checkpoint hashes before and after, and a 120-second supervisor.

BRIA RMBG-2.0 revision `5df4c9c76d8170882c34f6986e848ee07fd0ba43` also passed
strict local SafeTensors loading and one real 1024-pixel image inference on
MPS in fp32. Its 20 deformable-convolution modules were observed on MPS; the
installed torchvision 0.27 operator independently matched CPU on two small
fixtures. The original full-resolution, multiscale, non-SDPA model settings
were retained. This establishes execution on this stack, not segmentation
quality across arbitrary images or complete UniForge inference.

The same original BRIA/public preprocessing and DINO paths subsequently prepared
four real prop views. Both DINO banks were saved as exact CPU float32 tensors,
with all views/tokens retained and source/input/weight hashes checked before
and after. Encoder forward did not change either CPU or MPS RNG state.

The full trained Stage1 joint structure model also passed original strict
base/LoRA/trainable-weight loading: **2,460,456,876 FP32 parameters**. A load-only
process completed in 33.05 seconds. Separate real-input conditional and
unconditional predictions at the original seed 42 and first timestep passed
in 38.14 and 28.61 seconds including imports, loading and integrity checks.
The synchronized forwards took 7.33 and 1.45 seconds. Observations retained
all 30 DiT blocks and, for the conditional branch, all 24 frame/global VGGT
blocks, full four-view contexts and native bf16 projections. Outputs were
finite bf16 tensors of shape `[1,8,16,16,16]`.

These are full trained component predictions, **not completed sampler steps,
occupancy, a generated mesh or a finished PBR asset**. Full trained CPU/CUDA
prediction parity was not tested. Each run used the supervised short-resource
lease, offline local weights and disabled generic MPS fallback. Process peak
RSS was about 17.51 GB; it is not a GPU peak measurement. An initial probe-only
optional-library discovery error was corrected before these successful runs;
no model math changed to resolve it.
