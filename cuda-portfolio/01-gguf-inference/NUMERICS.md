# Numerical guarantees

What the kernels in `01-gguf-inference` and the runtime in `19-llm-engine`
promise about their arithmetic, how each promise is tested, and the tolerance
each test allows. Every number below was measured on a GTX 1650 (sm_75,
driver 616.92, CUDA 13.4) unless it is stated as a bound.

## Accumulation precision and determinism

| operation | integer part | floating part | order |
|---|---|---|---|
| Q4 GEMV (`gemv_q4`) | per 32-weight group, exact in `int`: \|s\| <= 32 x 8 x 127 = 32,512 | groups x FP32 scale, summed in FP32 per lane, then across the warp | fixed for a fixed launch |
| FP32 GEMV (`gemv_f32`) | - | FP32, 4 products per lane step, summed across the warp | fixed for a fixed launch |
| RMSNorm | - | FP32 sum of squares per warp, then across 8 warps in one block | fixed for a fixed launch |
| host references | exact `int` | `double` sums (RMSNorm, softmax), `float` elsewhere | sequential |

- **Deterministic run to run.** Same inputs and launch configuration give
  bitwise-identical outputs (`Numerics.KernelsAreDeterministicRunToRun`). No
  kernel here uses a reduction whose order depends on scheduling.
- **Not order-identical to the host.** The device sums in a different order
  from the sequential references, so results match them to a tolerance, not
  bit for bit (tables below).
- **The two Q4 kernels are bitwise identical.** The `__dp4a` kernel and the
  portable fallback compute the same integers and accumulate them in the same
  order (`Capability.PortableKernelIsBitwiseIdenticalToDp4a`).
- **CUDA graphs don't change results.** The runtime's graph replay reproduces
  the stream path's logits bit for bit
  (`EndToEnd.CudaGraphsReproduceTheStreamPathBitForBit`).
- **Batching doesn't change results.** A sequence's logits are bitwise
  identical whether it steps alone or beside others
  (`Isolation.BatchedSequencesMatchTheirSoloRunsExactly`).

## Quantization

**Weights (Q4).** Scale = max|w| / 7 per 32 weights; nibble =
clamp(round-to-nearest-even(w / scale), -8, 7) + 8. Every value is in [-7, 7]
by construction, so -8 is never produced (it is decoded if a file contains
it). An all-zero group gets scale 1e-12 and decodes to exact zeros. Non-finite
weights are rejected. Round-trip error is at most half a step, max|w| / 14,
per element (`Quant.RoundTripIsWithinQuantizationStep`); weights on the grid
round-trip exactly (`Numerics.QuantizationRoundsToNearestAndSaturates`).

**Native GGUF Q4_0.** `q4_from_gguf_q4_0` decodes the FP16 scale per block (a
non-finite scale is rejected) and reorders llama.cpp's split-half nibbles into
the interleaved order the kernel consumes. The internal `Q4Matrix` keeps FP32
scales (5.0 bits per weight); it is not the on-disk format and never stands in
for it.

**Activations (INT8).** Per-tensor symmetric: scale = max|x| / 127;
q = clamp(round-to-nearest-even(x / scale), -127, 127). One outlier sets the
scale for the whole vector; that loss belongs to the format, not the kernel.
Non-finite activations are rejected, because NaN has no INT8 value. An all-zero
vector gives exact zeros.

| test | compares | tolerance (relative L2) | measured |
|---|---|---|---|
| `GemvQ4.KernelMatchesCpuModelExactly` | kernel vs its integer model | 1e-5 | FP32 accumulation order only |
| `GemvQ4.ActivationQuantizationLossIsBounded` | W4A8 vs W4A32, same weights | 2e-2 | ~0.6% at K=1024 |
| `GemvQ4.AgreesWithFp32WithinQuantizationError` | W4A8 vs FP32 weights | 0.15 | a few % |

## RMSNorm epsilon

`x * w / sqrt(mean(x^2) + eps)` with eps finite and >= 0 (checked). eps > 0
keeps a zero vector at zero. **eps = 0 on a zero vector divides by zero and
yields NaN**, on the device and in the reference alike. It is allowed, because
it is well defined for any non-zero input. NaN inputs propagate; nothing masks
them (`Numerics.RmsNormEpsilonBehaviour`). Device vs host: 1e-5 relative
(`RmsNorm.MatchesCpu`).

## RoPE

Supported variant: the original Llama/GPT-NeoX "rotate half" pairing
(i, i + head_dim/2), frequencies theta^(-2i/head_dim), **no frequency scaling**
(linear, NTK, YaRN and Llama-3 scaling are rejected at model load in
`19-llm-engine`). theta must be finite and > 0; positions are 0 to 2^24, the
range where every integer position is exact in FP32.

The angle is pos x freq in FP32. Device `powf` and host `std::pow` may differ
by one ULP in freq, and the position multiplies that difference, so agreement
degrades linearly with position:

| position | relative error vs host |
|---|---|
| 1 | 3.3e-8 |
| 4,095 | 1.4e-7 |
| 65,535 | 2.1e-6 |
| 2^20 | 3.3e-5 |
| 2^24 | 5.3e-4 |

Tested against 1e-6 + 5e-11 x pos, and 1e-5 up to position 65,535
(`Numerics.RopeAgreementDegradesLinearlyWithPosition`). The fused and unfused
device paths agree to 1e-5 (`Rope.FusedMatchesUnfused`,
`SharedMemory.OversizedVectorsTakeTheUnfusedPathWithTheSameResult`).

## Softmax and sampling: infinities and NaN

| input | `softmax_cpu` | runtime sampler (`19`) |
|---|---|---|
| any NaN | every output NaN | `Error(Device)`: NaN logits mean a numerical fault upstream |
| +inf entries | share probability equally; finite entries get 0 | greedy picks the lowest such id; sampling is uniform over them |
| -inf entries | probability 0 | never sampled |
| all -inf | `invalid_argument` | `Error(InvalidArgument)` |
| large finite logits | stable (max subtracted, `double` sums) | same |

Greedy ties go to the lowest token id. Top-k and top-p are computed in
`double`. A fixed seed gives a fixed token sequence (`Sampler.*`,
`EndToEnd.FixedPromptAndSeedGiveTheSameTokensEveryTime`).

## End to end

| comparison | metric | tolerance | measured |
|---|---|---|---|
| float runtime vs host FP32 forward pass | relative L2 of logits | 1e-4 | 2.9e-6 |
| int8 runtime vs host FP32 forward pass | relative L2 / argmax | 0.10 / equal | answer agrees |
| float runtime vs **llama.cpp** (CPU build b10932) | mean KL divergence / same top token | <= 0.01 / >= 95% | **0.00071 / 98.48%** |
| int8 runtime vs llama.cpp | mean KL divergence / same top token | <= 0.01 / >= 95% | **0.00072 / 98.24%** |
| perplexity, same tokens | ours vs llama.cpp | reported | 23.376 (float), 23.379 (int8) vs 23.319 |
| tokenizer, llama.cpp's score order | identical tokens | all | **4,096 of 4,096** |

The llama.cpp comparison uses 8 chunks of 512 tokens from wikitext-2 (2,040
scored positions), llama.cpp's own `--kl-divergence-base` file, and the same
statistics its `--kl-divergence` mode reports (`19-llm-engine/src/compare_llamacpp.cpp`).
The remaining gap is the expected one between two different implementations
of the same quantized model: llama.cpp's CPU path multiplies Q4_0 weights
against Q8_0-quantized activations and sums in a different order.

**Tokenization differs by design, with evidence.** llama.cpp tokenizes a
`tokenizer.ggml.model = "llama"` file by vocabulary score and ignores
`tokenizer.ggml.merges`. This file's 32,000 scores are all zero, so its order
degenerates to "leftmost mergeable pair first" (` Boulter` becomes ` Bou`+...).
The runtime defaults to merge order, the order of the tokenizer the model was
trained with, and keeps score order (`Tokenizer::Algorithm::Scores`) for
parity. The model decides between them: on the same 7,543 bytes it needs
**0.901 bits/byte with merge order and 1.313 with llama.cpp's order**.
