# CUDA on a GTX 1650 — learning workspace

## Your hardware

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 (TU117) |
| Architecture | Turing, **compute capability 7.5** → always compile with `-arch=sm_75` |
| VRAM | 4 GB |
| SMs | 14 |
| Host compiler | MSVC 14.44 (VS 2022 Build Tools) |

Turing is a genuinely good architecture to learn on: it has the full modern
CUDA programming model — independent thread scheduling, warp shuffles,
unified shared memory/L1 — without the tensor-core complications that make
newer cards confusing for a beginner.

The 4 GB limit only bites for large ML models. It is irrelevant for
learning the language.

## Build and run

```bat
build.bat 00-check\driver_check.cu
bin\driver_check.exe
```

## The examples, in order

| Dir | Teaches |
|---|---|
| `00-check` | Driver/runtime sanity check. Run this first, and whenever something breaks. |
| `01-device-query` | Your actual hardware limits. Note the peak bandwidth — it is the ceiling you measure against later. |
| `02-hello` | Grid/block/thread model, `blockIdx * blockDim + threadIdx`, warps, async launches. |
| `03-vector-add` | Host↔device memory, bounds checks, CUDA-event timing, and why simple kernels are *memory bound*. |
| `04-matmul` | Shared-memory tiling and `__syncthreads()`. Same math, several times faster. The core optimization lesson. |

`common/cuda_check.h` has the error-checking macros. Use them everywhere —
CUDA reports failures through return codes, so unchecked errors surface as
wrong numbers rather than crashes.

## Measured on this machine (2026-09-12)

Baseline numbers from a working install — compare against these if something
ever looks slow.

| Example | Result |
|---|---|
| `01-device-query` | 14 SMs, 48 KB shared/block, 1024 KB L2, 128-bit bus, **192.0 GB/s peak bandwidth** |
| `03-vector-add` | 1.142 ms for 16.7M elements → **176.3 GB/s = 92% of peak** |
| `04-matmul` (1024x1024) | naive 186.6 GFLOP/s, tiled 273.1 GFLOP/s → **1.46x** |

Two things to read from that:

**Vector add hitting 92% of peak** is the lesson working exactly as intended.
That kernel is memory bound and essentially optimal — no amount of cleverness
in the arithmetic will improve it, because the bottleneck is moving bytes.

**The matmul speedup is 1.46x, not the 3-4x the textbooks quote.** That is
real, not a bug, and it is worth understanding: Turing has a large unified
L1/shared cache (64 KB per SM) plus 1 MB of L2, so the naive kernel's
redundant reads already hit cache most of the time. Shared-memory tiling wins
less on modern hardware than it did on Fermi/Kepler, where textbook numbers
came from. Good experiments from here:

- Raise `TILE` to 32 (1024 threads/block — check occupancy).
- Raise `N` to 2048 or 4096 so the working set stops fitting in L2.
- Give each thread more than one output element (register tiling) — usually
  a bigger win than shared memory alone on this architecture.

## Roadmap after these four

1. **Reduction** (summing an array) — the hardest "easy" problem. Teaches warp
   divergence, tree reductions, `__shfl_down_sync`, and the fact that the last
   warp needs special handling.
2. **Memory coalescing** — write the same kernel with strided vs contiguous
   access and measure. Often a 10x difference on its own.
3. **Occupancy** — `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, and why
   maximum occupancy is *not* always fastest.
4. **Streams and async copy** — overlap transfers with compute.
5. **Atomics** — `atomicAdd`, histogram kernels, contention.
6. **Profiling** — Nsight Compute on a real kernel. This is where you stop
   guessing and start measuring.

## Reference material

- *Programming Massively Parallel Processors* (Kirk & Hwu) — the standard textbook.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — the spec.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — read after the first few kernels.

## The one habit worth forming early

Before optimizing any kernel, decide whether it is **memory bound** or
**compute bound**. Compute arithmetic intensity (FLOPs per byte moved). Most
kernels you write early on are memory bound, which means cleverer arithmetic
buys you nothing and better memory access patterns buy you everything.
