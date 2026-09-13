# Production readiness: status

Each item in the production-readiness guide, where it is implemented, and the
test that holds it. The runtime the guide describes lives in `19-llm-engine`,
built on this project's parser, kernels and KV cache. Test names are
`Suite.Name` in the binaries listed. Device suites need the model
(`scripts/fetch_model.*`) and skip cleanly without it.

| suite | where it runs |
|---|---|
| `test_gguf`, `test_llm_host` | anywhere; CI runs them under ASan + UBSan (`sanitize-host/`) |
| `test_kernels`, `test_llm_engine`, `test_error_paths` | a CUDA device (sm_75 here; COLAB.md for a T4) |
| `compare_llamacpp` | a device, llama.cpp's reference file, and wikitext-2 |

## 1. Complete inference path — done

Tokenizer (`tokenizer.cpp`), embedding lookup, transformer blocks with GQA,
RoPE, device-side paged attention, SwiGLU, Q6_K/Q4_0/F32 output projection
(`runtime.cu`), greedy/temperature/top-k/top-p sampling (`sampler.cpp`),
detokenization and streaming callbacks (`generator.cpp`).
Tests: `EndToEnd.GreedyChatAnswersAFactualQuestion`, `Reference.*`, `Sampler.*`.

## 2. Validated model configuration — done

`ModelConfig` is built once from metadata (`model_config.cpp`), with every
relationship in the guide checked before any allocation. Violations are
`InvalidModel`; valid shapes the kernels don't implement are `Unsupported`.
The tensor manifest reports every missing, mistyped or misshaped tensor at
once. Tests: `ModelConfig.*`, `TensorManifest.*`.

## 3. Transactional loading — done

1. The parser rejects hostile files with checked element counts and byte sizes,
   power-of-two alignment, and misaligned, past-end, overlapping and duplicate
   tensors (`GgufHostile.*`).
2. Required tensors are resolved and validated (`TensorManifest.*`).
3. The memory plan (weights, KV cache, workspace) is checked against free VRAM
   plus headroom (`Failure.AMemoryPlanLargerThanTheDeviceIsRefusedBeforeAllocating`).
4. Allocation and upload happen in an exception-safe constructor. The runtime
   exists only if every step succeeded.

## 4. Ownership boundaries — done

`GgufModel` (mapped file, config, tokenizer; immutable, shared) ->
`Runtime` (device weights, KV page pool, reusable workspace, graphs) ->
`Sequence` (page table and position; forks share pages) -> `SamplingParams` /
generator. The token loop allocates nothing per token. See `runtime.h`.
Tests: `Isolation.SequencesMayOutliveTheirRuntime`,
`EndToEnd.DeviceMemoryIsStableAcrossRepeatedGenerations`.

The per-call wrappers in `kernels.h` still allocate per call. They are
documented as test and benchmark entry points, not a generation path.

## 5. KV cache — done, in both caches

| requirement | runtime cache (`19`) | `PagedKvCache` (`01`) |
|---|---|---|
| layer append ordering | all layers written inside one step | `KvOrderError`, `next_layer()` — `KvCacheRules.LayersMustBeAppendedInOrder` |
| maximum context | `ContextFull` — `Failure.ContextFullIsReportedWithTheSequence` | `KvContextFull` — `KvCacheRules.MaxContextIsEnforcedWithoutSideEffects` |
| transactional allocation | all-or-nothing across the batch — `Failure.KvExhaustionFailsCleanlyAndLeavesSequencesUsable` | all layers reserved at layer 0 — `KvCacheRules.PageAllocationIsAllOrNothingAcrossLayers` |
| batching / concurrent sessions | batched `step()` — `Isolation.BatchedSequencesMatchTheirSoloRunsExactly` | mutex-serialized — `KvCacheRules.ConcurrentSequencesOnSeparateThreads` |
| device page tables to attention | the paged attention kernel reads them | `page_table()`, `device_layout()`, `gather_device()` — `KvCacheRules.DeviceGatherReadsPagesWhereTheyAre` |
| prefix sharing, copy-on-write | `Sequence::fork` — `Isolation.ForkSharesPagesAndCopiesOnWrite` | `fork_sequence` — `KvCacheRules.ForkSharesPagesAndCopiesOnWrite` |

## 6. Hardened kernel wrappers — done

Every condition in the guide's three lists is checked before allocation or
launch, with checked `size_t` arithmetic. Launches are checked against the
device's grid and block limits. Tests: `Validation.*`, `Capability.BoundaryShapesMatchTheIntegerModel`.

## 7. Shared memory — done

The fused RMSNorm+RoPE wrapper asks the runtime how much dynamic shared memory
the device leaves the kernel (`cudaOccupancyAvailableDynamicSMemPerBlock`:
12,256 elements on sm_75). Larger vectors take the global-memory path with the
same result. Test: `SharedMemory.OversizedVectorsTakeTheUnfusedPathWithTheSameResult`.

## 8. Real GGUF quantization — done for what the target model uses

- **Native layouts decoded:** Q4_0 (FP16 scale, split-half nibbles), Q6_K, F32
  (`quant.cpp`, `Quant.*`). `q4_from_gguf_q4_0` imports native blocks into the
  kernel layout (`GgufQ4_0.*`).
- **The internal format is labeled as internal:** `Q4Matrix` (FP32 scales,
  interleaved) is documented as not the on-disk format.
- **Capability-checked `__dp4a` with a fallback:** the runtime refuses int8
  activations below sm_61 and `Auto` falls back to float. The kernel wrapper has
  a portable integer kernel, bitwise identical to `__dp4a`
  (`Capability.PortableKernelIsBitwiseIdenticalToDp4a`).
- **Not implemented:** the other K-quants and Q5/Q8 are parsed (their sizes are
  validated), but a model using them for any tensor the runtime needs is
  refused at load, with each offending tensor and its type listed.

## 9. Numerical guarantees — done

[`NUMERICS.md`](NUMERICS.md) covers accumulation precision, rounding and
saturation, epsilon behavior, the RoPE variant and its position-dependent
error, softmax special values, determinism, and tolerances with measurements.
Tests: `Numerics.*`, plus the end-to-end comparison below.

## 10. Error reporting — done

`llm::Error` (`errors.h`) carries a kind (`InvalidArgument`, `InvalidModel`,
`Unsupported`, `OutOfMemory`, `ContextFull`, `Cancelled`, `Device`) and context
(operation, model, tensor, shape, device, CUDA error, sequence, position).
VRAM budget and context checks: sections 3 and 5. Cancellation: a token or the
streaming callback (`EndToEnd.CancellationStopsGeneration`). Thread-safety
rules are in `runtime.h` and `kv_cache.h`; a concurrent `step()` is refused
(`Failure.ConcurrentStepsAreRefusedNotInterleaved`). There is no C API, so no
exception crosses one.

## 11. Observability — done

`metrics_json`: TTFT, prefill and decode tokens/s, memory plan, KV pages and
utilization, live sequences, batch and context limits, quantization and
activation format, device name, compute capability, driver release (NVML) and
CUDA driver/runtime versions. Per-stage latency comes from
`RuntimeOptions::profile_stages` (`bench_engine`). Test:
`EndToEnd.MetricsRecordCarriesTheOperationalContext`. Nsight Systems and
Compute: `scripts/profile.ps1`. On this driver Nsight Compute needs an elevated
shell.

## 12. Testing at every level — done

| level | coverage |
|---|---|
| unit | parser (hostile files), config, manifest, tokenizer (incl. the merge-order definition), sampler, page allocator, errors, dequantization, llama.cpp file reader, KL math |
| kernel | every kernel vs a CPU reference at boundary shapes; `__dp4a` vs portable; fused vs unfused |
| integration | device runtime vs an independent host FP32 forward pass (2.9e-6); **vs llama.cpp**: mean KL divergence 0.0007, 98.2–98.5% same top token, llama.cpp's tokenization reproduced token for token |
| end to end | fixed prompt and seed, token counts, memory stability, KV exhaustion, context full, sequence isolation, fork/COW, truncation, cancellation, invalid steps, plan refusal |
| sanitizers | ASan + UBSan on host code in CI; compute-sanitizer memcheck / racecheck / initcheck via `scripts/sanitize.*` (see below) |

**compute-sanitizer on this machine: not completed.** On driver 616.92 it
could not attach to the test process even from an elevated shell, so no device
memory, race or initialization results are claimed for this machine. On Linux
and Colab, `scripts/sanitize.sh` needs no special permissions. All measured
results are in [`../TEST_RESULTS.md`](../TEST_RESULTS.md).

## Reproducing the llama.cpp comparison

```
scripts/fetch_model.ps1
# wikitext-2: https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
llama-perplexity -m tinyllama-1.1b-chat-v1.0.Q4_0.gguf -f wiki.test.raw -c 512 -b 512 --chunks 8 \
                 --kl-divergence-base reference.kld
build/bin/compare_llamacpp --reference reference.kld --text wiki.test.raw
```

The exit status is 0 only if llama.cpp's tokens are reproduced (with `--text`), the mean KL
divergence is at most 0.01, and the top token agrees at least 95% of the time.
