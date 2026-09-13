# Production Readiness Guide

## Current State

`01-gguf-inference` is a CUDA inference prototype. It currently provides:

- A bounds-checked GGUF parser.
- Q4 quantization and GPU GEMV kernels.
- RMSNorm and RoPE kernels, including a fused implementation.
- A paged GPU KV cache.
- Kernel benchmarks and unit tests.

The current benchmark does not load a real model or generate text. It measures isolated operations using synthetic data.

## Target Architecture

A production runtime should connect these components into one complete pipeline:

```text
GGUF model
  -> metadata and tensor validation
  -> tokenizer
  -> device-resident weights
  -> transformer forward pass
  -> paged KV cache
  -> logits
  -> sampling
  -> generated tokens
  -> streamed text
```

## 1. Implement the Complete Inference Path

Add the missing model operations:

- Tokenizer vocabulary and merges.
- Embedding lookup.
- Transformer block execution.
- Q/K/V projections.
- RoPE.
- Attention score computation.
- Device-side KV-cache reads.
- SwiGLU or the model's required MLP.
- Output projection and logits.
- Greedy, temperature, top-k, and top-p sampling.
- Token detokenization and streaming output.

Until these components exist, the project is a kernel and memory-management prototype rather than a complete text-generation engine.

## 2. Build a Validated Model Configuration

Do not scatter GGUF metadata lookups across inference code. Convert metadata into one validated configuration object:

```cpp
struct ModelConfig {
    int layers;
    int hidden_size;
    int intermediate_size;
    int attention_heads;
    int kv_heads;
    int head_dim;
    int vocab_size;
    int context_length;
    float rms_epsilon;
    float rope_theta;
};
```

Validate relationships before allocating GPU memory:

```text
hidden_size == attention_heads * head_dim
kv_heads <= attention_heads
attention_heads % kv_heads == 0
all dimensions > 0
context_length > 0
```

Reject unsupported architectures, tensor types, and missing tensors during model loading.

## 3. Make Model Loading Transactional

The model loader should:

1. Parse and validate the GGUF file.
2. Resolve every required tensor by name.
3. Validate tensor types, shapes, offsets, and byte sizes.
4. Check the available VRAM budget.
5. Allocate device memory.
6. Copy or otherwise prepare weights.
7. Publish the model only after every step succeeds.

If any step fails, all previously allocated resources must be released.

The existing parser in `src/gguf.cpp` is a good foundation, but production loading also needs checked arithmetic for tensor dimensions, byte counts, offsets, and alignment.

## 4. Separate Ownership Boundaries

Use explicit ownership between the major runtime objects:

```text
GgufModel
  owns the mapped file and metadata

DeviceModel
  owns GPU-resident weights

InferenceSession
  owns reusable activation buffers and execution state

KvCache
  owns sequence-specific attention history

Sampler
  owns generation policy
```

Avoid allocating and freeing CUDA buffers inside every kernel wrapper call. Allocate a reusable workspace once and use it for every token.

The current functions in `src/kernels.cu` allocate temporary device buffers on every call, which will add substantial overhead during generation.

## 5. Strengthen the KV Cache

The paged cache should be extended to:

- Enforce layer append ordering.
- Track the next expected layer for every sequence.
- Enforce the maximum context length.
- Make allocation failure transactional.
- Support batching and concurrent sessions.
- Expose device page tables or pointers directly to attention kernels.
- Support prefix sharing and copy-on-write pages where useful.

The current `gather_keys()` and `gather_values()` functions copy cached data back to the host. Production attention should read KV pages directly on the GPU.

## 6. Harden Kernel Wrappers

Before launching CUDA kernels, validate all dimensions and buffer sizes.

For `Q4Matrix`, validate:

```text
rows > 0
cols > 0
cols % 32 == 0
qs.size() == rows * (cols / 2)
scales.size() == rows * (cols / 32)
```

For FP32 GEMV, validate:

```text
rows > 0
cols > 0
cols % 4 == 0
weights.size() == rows * cols
input.size() == cols
```

For RMSNorm and RoPE, validate:

```text
input is not empty
weight.size() == input.size()
n_heads > 0
head_dim > 0
head_dim % 2 == 0
n_heads * head_dim == input.size()
epsilon >= 0 and finite
theta > 0 and finite
position >= 0
```

Use checked `size_t` multiplication before allocating buffers. Never let invalid dimensions reach a CUDA launch.

## 7. Protect Shared-Memory Usage

The fused RMSNorm plus RoPE path allocates shared memory proportional to the input size:

```cpp
k_rmsnorm_rope<<<1, 256, n * sizeof(float)>>>(...);
```

For large vectors this can exceed the device's per-block shared-memory limit. Query the device limit and either reject oversized inputs or use a multi-block/global-memory implementation.

Also validate block and grid dimensions against the selected GPU's capabilities.

## 8. Support Real GGUF Quantization

The internal `Q4Matrix` is useful for testing, but production loading must support the native GGUF tensor layouts required by the target models.

In particular:

- Support all required quantization formats.
- Correctly decode native block layouts.
- Handle FP16 scales where required by the format.
- Avoid silently treating an internal FP32-scale representation as native GGUF Q4.
- Provide a fallback path for unsupported GPU instructions.
- Check GPU compute capability before using `__dp4a`.

Architecture-specific kernels should be added only after profiling identifies a real bottleneck.

## 9. Define Numerical Guarantees

Document and test:

- Accumulation precision.
- Quantization rounding and saturation.
- RMSNorm epsilon behavior.
- RoPE variants and scaling.
- Softmax behavior for infinities and NaNs.
- Deterministic versus nondeterministic reductions.
- Expected error tolerances for each operation.

Kernel tests must be supplemented with end-to-end logit comparisons against a trusted implementation such as llama.cpp.

## 10. Improve Error Reporting

Every public failure should include useful context:

- Operation name.
- Model and tensor name.
- Tensor shape.
- CUDA device.
- CUDA error text.
- Sequence ID and token position when relevant.

Also add:

- VRAM budget checks.
- Context-length checks.
- Cancellation support.
- Clean handling of out-of-memory errors.
- Explicit synchronization and thread-safety rules.
- No exceptions crossing a C API boundary, if one is added later.

## 11. Add Observability

Record at least:

- Prompt-processing tokens per second.
- Decode tokens per second.
- Time to first token.
- Per-kernel latency.
- GPU memory usage.
- KV-cache utilization.
- Batch size and context length.
- Quantization format.
- CUDA and driver versions.

Use Nsight Systems and Nsight Compute to validate the actual bottleneck. Theoretical bandwidth estimates are not enough for production decisions.

## 12. Test at Multiple Levels

### Unit tests

Test parsing, tensor shapes, quantization, sampling, page allocation, overflow handling, and invalid inputs.

### Kernel tests

Compare GPU operations with CPU references, including boundary dimensions and multiple GPU architectures.

### Integration tests

Load a small real GGUF model and compare logits against a trusted runtime.

### End-to-end tests

With a fixed prompt and seed, verify:

- Model loading succeeds.
- Generated token count is correct.
- Output is within the expected numerical tolerance.
- Memory use remains stable.
- KV-cache exhaustion fails cleanly.
- Multiple sequences do not interfere with each other.

Run CUDA Compute Sanitizer with `memcheck`, `racecheck`, and `initcheck`. Run AddressSanitizer and UndefinedBehaviorSanitizer for host-side code where supported.

## Recommended Implementation Order

1. Build a correct CPU reference transformer.
2. Load one supported GGUF architecture completely.
3. Compare CPU logits with llama.cpp.
4. Move one operation at a time to CUDA.
5. Compare every CUDA operation with the CPU reference.
6. Add device-resident KV-cache attention.
7. Add sampling and streaming generation.
8. Add batching and concurrent sessions.
9. Add memory budgeting, cancellation, and observability.
10. Profile and optimize only after correctness is stable.

The most important production step is creating a trusted end-to-end reference path. Individual kernel tests can prove that a kernel matches its CPU model, but only an end-to-end comparison can prove that the runtime performs correct GGUF inference.
