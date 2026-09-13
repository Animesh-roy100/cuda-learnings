# Test results: the production inference runtime

Results from testing the runtime (`01-gguf-inference` + `19-llm-engine`) after
it was hardened to `01-gguf-inference/PRODUCTION_READINESS.md`. Everything here
was run on 2026-09-13 against the code at commit `f91017a` (later commits
change documentation only). Nothing is estimated: each number was printed by
the command named beside it.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1650, 4 GB, Turing sm_75, 14 SMs, 192 GB/s |
| Driver | 616.92 (CUDA 13.4 API) |
| Toolkit / compiler | CUDA 13.4, MSVC 19.44, CMake 4.4, Release build |
| OS | Windows 11 |
| Model | TinyLlama-1.1B-Chat v1.0, Q4_0 GGUF, SHA-256 `da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556` |
| Reference | llama.cpp build b10932 (commit `737e0980f`), official Windows CPU x64 release |
| Text | wikitext-2-raw-v1 `wiki.test.raw` |

## Summary

| check | result |
|---|---|
| Full test suite (`ctest`, 22 suites) | **22 / 22 passed** in 104 s |
| Runtime device tests (`test_llm_engine`) | **20 / 20 passed** |
| Host tests (`test_llm_host`) | **36 / 36 passed** |
| Kernel and KV-cache tests (`test_kernels`) | **45 passed, 1 skipped** (the no-`__dp4a` test; this GPU has `__dp4a`) |
| GGUF parser tests (`test_gguf`) | **20 / 20 passed** |
| Error-path tests (`test_error_paths`) | **18 / 18 passed** |
| End to end vs llama.cpp, float | **PASS**: mean KL divergence 0.00071, same top token 98.48% |
| End to end vs llama.cpp, int8 | **PASS**: mean KL divergence 0.00072, same top token 98.24% |
| Tokenizer vs llama.cpp (llama.cpp-compatible mode) | **4,096 / 4,096 tokens identical** |
| GitHub CI (Linux, CUDA 12.8.2 and 13.3.1; host code under ASan + UBSan) | **passed** |
| compute-sanitizer on this machine | **not completed**: could not attach, even elevated (below) |

## 1. Runtime device tests (`test_llm_engine`)

| test | what it proves | result |
|---|---|---|
| `Reference.FloatRuntimeMatchesTheHostForwardPass` | float logits vs an independent host FP32 forward pass, relative error < 1e-4; answer " Paris" | pass (10.4 s) |
| `Reference.Int8RuntimeAgreesOnTheAnswer` | int8 picks the same token, relative error < 0.10 | pass (9.8 s) |
| `Reference.PerplexityIsLowOnEnglishText` | perplexity < 20, float and int8 | pass (8.1 s) |
| `EndToEnd.GreedyChatAnswersAFactualQuestion` | "What is the capital of France?" answer contains "Paris" | pass (2.0 s) |
| `EndToEnd.FixedPromptAndSeedGiveTheSameTokensEveryTime` | sampled generation is reproducible; token count respected | pass (2.2 s) |
| `EndToEnd.CudaGraphsReproduceTheStreamPathBitForBit` | graph replay changes no logit | pass (3.4 s) |
| `EndToEnd.DeviceMemoryIsStableAcrossRepeatedGenerations` | 20 generations lose < 16 MB free VRAM; all KV pages returned | pass (7.3 s) |
| `EndToEnd.CancellationStopsGeneration` | stops via callback and via token; pages released | pass (1.7 s) |
| `EndToEnd.MetricsRecordCarriesTheOperationalContext` | JSON has driver, CUDA, quant, KV, limits, TTFT, tok/s | pass (1.7 s) |
| `Isolation.BatchedSequencesMatchTheirSoloRunsExactly` | batching changes no bit of a sequence's logits | pass (1.7 s) |
| `Isolation.ForkSharesPagesAndCopiesOnWrite` | fork copies nothing; one page copied on write; both branches match unshared controls | pass (1.7 s) |
| `Isolation.TruncateReleasesPagesAndReplaysExactly` | truncate frees pages; replay reproduces logits | pass (2.1 s) |
| `Isolation.SequencesMayOutliveTheirRuntime` | destroying a sequence after its runtime is safe | pass (1.6 s) |
| `Failure.KvExhaustionFailsCleanlyAndLeavesSequencesUsable` | `OutOfMemory`; no position or page changes; recovers | pass (1.9 s) |
| `Failure.ContextFullIsReportedWithTheSequence` | `ContextFull` names the sequence; the batch is not partly applied | pass (1.7 s) |
| `Failure.InvalidStepsAreRejectedBeforeAnyKernelRuns` | empty, duplicate, foreign, out-of-range, mismatched inputs | pass (3.2 s) |
| `Failure.ConcurrentStepsAreRefusedNotInterleaved` | a second thread's `step()` is refused; 30 steps completed, 40 concurrent calls refused, cache intact | pass |
| `Failure.AMemoryPlanLargerThanTheDeviceIsRefusedBeforeAllocating` | ~11 GB KV plan refused as `OutOfMemory`, no VRAM consumed | pass |
| `Failure.OptionsAreValidated` | bad context and batch sizes rejected | pass |
| `Failure.AMissingModelFileIsAnInvalidModelError` | missing file is `InvalidModel` | pass |

## 2. End-to-end comparison against llama.cpp

Reference produced by llama.cpp itself:

```
llama-perplexity -m tinyllama-1.1b-chat-v1.0.Q4_0.gguf -f wiki.test.raw -c 512 -b 512 --chunks 8 -t 8 \
                 --kl-divergence-base tinyllama-q4_0-wikitext2-c512-8chunks.kld
# llama.cpp: Final estimate: PPL = 23.4069 +/- 1.84029
```

Replayed through this runtime with `compare_llamacpp` (8 chunks x 512 tokens,
2,040 scored positions, statistics computed the way llama.cpp's
`--kl-divergence` mode computes them):

| mode | mean KLD | median | p99 | max | same top | PPL ours | PPL llama.cpp | RMS Δp | verdict |
|---|---|---|---|---|---|---|---|---|---|
| float | 0.000710 | 0.000490 | 0.004870 | 0.015192 | 98.48% | 23.3760 | 23.3185 | 0.566% | PASS |
| int8 | 0.000720 | 0.000543 | 0.003880 | 0.010771 | 98.24% | 23.3794 | 23.3185 | 0.592% | PASS |

Thresholds: mean KLD <= 0.01 and same top token >= 95%. llama.cpp prints
23.41 from its full-precision logits. The 23.32 in the table is computed from
the compressed log-probabilities in its reference file, which clamp any value
more than 16 nats below the top token to that floor, so rare true tokens score
slightly better. Both runtimes' perplexities in the table use the same scored
positions.

### Tokenizer

| mode | result on the first 65,646 bytes |
|---|---|
| score order (llama.cpp-compatible) | **all 4,096 tokens identical** to llama.cpp, 23 ms |
| merge order (runtime default) | identical for the first 5 tokens, then `" B"` (350) vs llama.cpp's `" Bou"` (12476); expected |

The two orders differ because llama.cpp ranks merges by vocabulary score, and
this file's 32,000 scores are all zero. The model itself says which fits: the
same 7,543 bytes scored under each tokenization (int8):

| tokenization | tokens | bits per byte |
|---|---|---|
| merge order (default) | 2,041 | **0.9014** |
| score order (llama.cpp) | 2,046 | 1.3128 |

## 3. Host tests (`test_llm_host`, 36 tests)

All passed in 144 ms. Groups: `ModelConfig` (6), `TensorManifest` (2),
`PageAllocator` (5, including 8-thread concurrency), `Sampler` (7), `Errors`
(1), `Quant` (4), `Tokenizer` (7: known Llama ids, Unicode round trip, byte
fallback, control tokens, priority-queue merges vs the literal definition,
llama.cpp score order, 200 KB input), `LlamaCppReference` (2),
`KlDivergence` (2).

## 4. Kernel and KV-cache tests (`test_kernels`, 46 tests)

45 passed, 1 skipped (`Capability.Dp4aIsRefusedWhereTheDeviceLacksIt` runs
only on GPUs without `__dp4a`). Groups: quantization, Q4 GEMV vs its integer
model, RMSNorm, RoPE, KV cache, and the hardening suite: `Validation` (4),
`Capability` (3), `SharedMemory` (1; fused capacity on this device 12,256
elements), `GgufQ4_0` (3), `Numerics` (5), `KvCacheRules` (8).

Measured RoPE agreement with the host reference, by position:

| position | 1 | 4,095 | 65,535 | 2^20 | 2^24 |
|---|---|---|---|---|---|
| relative error | 3.3e-8 | 1.4e-7 | 2.1e-6 | 3.3e-5 | 5.3e-4 |

## 5. Full suite (`ctest`)

22 of 22 suites passed, 104.18 s total, including every other project in the
portfolio (`test_llm_engine` 62.9 s, `test_error_paths` 33.1 s).

## 6. Applications

`bench_engine` (perplexity on the opening of *Alice's Adventures in
Wonderland*; speed from a 96-token greedy generation):

| configuration | perplexity | TTFT ms | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| float (W4A16), no graphs | 5.594 | 2740.4 | 11.3 | 11.2 |
| int8 (W4A8), no graphs | 5.551 | 242.9 | 127.7 | 119.8 |
| int8 (W4A8), CUDA graphs | 5.551 | 231.6 | 133.9 | 125.0 |

| batch | ms / step | tokens/s total | per sequence |
|---|---|---|---|
| 1 | 8.51 | 117.5 | 117.5 |
| 2 | 13.46 | 148.6 | 74.3 |
| 4 | 23.76 | 168.4 | 42.1 |

`chat`: "What is the capital of France?" -> "The capital of France is Paris."
(TTFT 211 ms, decode 127 tok/s); two prompts batched together both answered.

## 7. Continuous integration

GitHub Actions on commit `f91017a`: `portfolio / CUDA 12.8.2`, `portfolio /
CUDA 13.3.1` (build only; hosted runners have no GPU) and `host code / ASan +
UBSan` (builds and runs `test_gguf` and `test_llm_host` under
AddressSanitizer and UndefinedBehaviorSanitizer): all passed. Model-dependent
host tests skip there, since the model is not in the repository.

## 8. compute-sanitizer: not completed on this machine

Run from an elevated PowerShell with `scripts/sanitize.ps1` (memcheck first,
then racecheck and initcheck) on `test_kernels` and a filtered
`test_llm_engine`:

| suite | tool | status |
|---|---|---|
| `test_kernels` | memcheck | NO-ATTACH after the 1,800 s timeout |
| `test_llm_engine` (`Isolation.*`, KV exhaustion, context full, invalid steps) | memcheck | NO-ATTACH |

Both logs show the same failure: host-only tests run, and the first test that
touches the GPU waits for the sanitizer until
`Error: No attachable process found. compute-sanitizer timed-out.` Elevation,
which Nsight Compute needs on this driver, does not fix compute-sanitizer on
driver 616.92. **No device memory, race or initialization findings are
claimed for this machine.** The run belongs on Linux: `scripts/sanitize.sh` on
Colab (see COLAB.md), which needs no special permissions.

## Not tested

- Any GPU other than the GTX 1650 (the Colab T4 steps are in COLAB.md).
- compute-sanitizer on device code (section 8).
- Prompts longer than 1,024 tokens; prefill still runs one token at a time.
- Models other than TinyLlama-1.1B-Chat Q4_0.
