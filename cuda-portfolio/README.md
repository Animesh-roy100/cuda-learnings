# CUDA systems portfolio — GTX 1650 (Turing, `sm_75`)

Fourteen production-structured CUDA projects. Every one builds, runs, and
self-verifies against an independent reference. **Every number below was
measured on this machine**, not estimated.

**Hardware:** GTX 1650, Turing `sm_75`, 14 SMs, 896 CUDA cores, 4 GB GDDR6,
**192 GB/s peak bandwidth**, PCIe gen3 ×16, no Tensor Cores, FP64 at 1/32 rate.

> The 128 GB/s figure often quoted for the GTX 1650 is the **GDDR5** variant.
> This is the GDDR6 card: measured 176–184 GB/s in practice. Tuning against
> 128 GB/s would mean declaring victory at 67% of the real ceiling.

**Toolchain:** CUDA 13.4, MSVC 19.44, driver 616.92, CMake 4.4, GoogleTest 1.15.

## Build and test

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest --output-on-failure
```

```
100% tests passed out of 15       (185 test cases across 14 projects)
```

| Suite | Cases | | Suite | Cases |
|---|---|---|---|---|
| `test_gguf` | 11 | | `test_video_pipeline` | 12 |
| `test_kernels` | 22 | | `test_audio_dsp` | 11 |
| `test_image_pipeline` | 12 | | `test_mc_pricing` | 10 |
| `test_hash_kv` | 11 | | `test_graph_engine` | 10 |
| `test_spatial_index` | 11 | | | |

Profiling artifacts:

```powershell
.\scripts\profile.ps1              # all targets
.\scripts\profile.ps1 -SkipNcu     # timelines only
```

## Layout

```
CMakeLists.txt              modern CMake, CUDA as a first-class language
cmake/ProjectDefaults.cmake shared target config (sm_75, -lineinfo, C++20)
common/include/cu/          check.hpp, timer.hpp, device.hpp
NN-project/
  include/                  public headers — ZERO CUDA syntax (pimpl)
  src/                      kernels (.cu) + host pipeline (.cpp)
  tests/                    GoogleTest, checked against CPU ground truth
scripts/profile.ps1         nsys timelines + ncu roofline/occupancy
profiles/                   generated artifacts
```

Public headers contain no `__global__`, no `<<<>>>`, no `cuda_runtime.h`.
Test suites and host code compile as plain C++20; only the `.cu` translation
units need nvcc.

---

## 1. GGUF inference engine (`01-gguf-inference`)

mmap'd GGUF parser, Q4_0 `__dp4a` GEMV, paged KV cache, fused RMSNorm+RoPE.

| Shape | fp32 | int4 | speedup |
|---|---|---|---|
| q_proj 2048×2048 | 0.114 ms / 147 GB/s | 0.033 ms | **3.49×** |
| gate/up 8192×2048 | 0.375 ms / 179 GB/s | 0.080 ms | **4.69×** |
| down 2048×8192 | 0.387 ms / 174 GB/s | 0.078 ms | **4.97×** |

Kernel verified against a CPU model of the identical W4A8 integer arithmetic:
**L2 relative error 1.7e-07** (float epsilon).

Llama 3.2 1B (1.24 B params) decode projection: int4 **0.70 GB → ~207 tok/s**,
the only format leaving room for weights *and* KV cache in 4 GB.

**The parser rejects malformed input rather than trusting it** — truncation,
bad magic, implausible counts, tensors running past EOF, element counts that
aren't a multiple of the block size. Tests build malformed GGUF images in
memory to prove it, so no model download is needed.

**Honest result:** fused RMSNorm+RoPE measured **0.95× — a wash**. At 8 KB
neither variant is memory bound; both are launch-overhead bound, and the fused
kernel runs as a single block (1 of 14 SMs) to share its reduction. Fusion pays
when the tensor is bandwidth bound or the launch sits in a hot loop.

## 2. Streaming image embedding pipeline (`02-image-embed`)

Bilinear resize + normalize + HWC→CHW in one kernel, 4-stream overlap, dynamic
batcher with a deadline.

| | |
|---|---|
| serial | 43.7 ms, 733 img/s |
| 4 streams | **37.3 ms, 857 img/s (1.17×)** |
| single-core CPU | 779 img/s |

**Honest result:** the GPU is only **1.10×** the single-core CPU path. The
breakdown says why — for a 32×1080p batch: host staging memcpy ~25 ms, PCIe
upload ~17 ms (these overlap, which is what the streams buy). The resize
arithmetic is negligible; the pipeline is **entirely input-transfer bound**.
1080p RGB8 is 6.2 MB in and 0.6 MB out: 10× more data shipped than used.

GPU preprocessing of full-size images only pays if the copy disappears — decode
into pinned memory, or keep frames on-device. Moving arithmetic to the GPU while
leaving the transfer in place just relocates the stall.

## 3. Lock-free hash table / KV store (`03-hash-kv`)

Open addressing, one 64-bit `atomicCAS` per mutation, key+value packed in a
single word so they land atomically together. Linear and Robin Hood probing,
per-thread and warp-cooperative lookup.

16.7M slots, end-to-end through the public API:

| Load | Insert | Find | Displacement avg | **max** |
|---|---|---|---|---|
| 0.50 linear | 162 M/s | 228 M/s | 0.50 | 42 |
| 0.95 linear | 140 M/s | 174 M/s | 9.51 | **4775** |
| 0.50 robinhood | 184 M/s | 230 M/s | 0.50 | 12 |
| 0.95 robinhood | 97 M/s | 182 M/s | 9.51 | **114** |

Robin Hood leaves the **average displacement identical** and collapses the tail:
**4775 → 114 at load 0.95**, a 42× better worst case. That is the tail latency a
KV store is judged on. The cost is insert throughput at high load.

**Honest result:** warp-cooperative lookup **loses** — 0.52× at load 0.5,
parity only at 0.95. It burns a 256-byte transaction per query when the average
chain is 1.5 slots. Per-thread probing lets 32 lanes resolve 32 *different*
queries from one transaction.

**Bug caught by tests:** the warp lookup originally capped probing at 1024 slots,
silently reporting present keys as missing at load ≥0.90 where clusters run
longer. An empty slot is the only correct terminator.

## 4. Spatial indexing, k-NN, DBSCAN (`04-spatial`)

Morton Z-order codes → sort → cell ranges → 27-cell queries, lock-free
union-find clustering. 2M points.

| | |
|---|---|
| index build | 9.7 ms (206 M points/s) |
| k-NN k=8, 4096 queries | grid 5.8 ms vs brute force 345 ms = **59.8×, identical** |
| DBSCAN | 195 ms (10.2 M points/s), noise 10.4% vs 10% planted |

**Cost driver:** each query scans 27 cells, so cost scales with **points per
cell, not N**. At mean 8.1 pts/cell DBSCAN takes 195 ms; tightening the blobs to
σ=0.004 packs ~6000 points into a cell and the identical code takes **18× longer**.
Grid resolution must be chosen against data density.

**API trap fixed:** `dbscan()` originally returned labels in Morton-sorted order,
so `labels[i]` described a different point than the caller's `points[i]`. Now
scattered back through the permutation.

## 5. Video analytics pipeline (`05-video`)

NV12→RGB, bilateral filter (explicit vs texture-unit clamping), motion history,
transfer hierarchy, CUDA IPC. 1080p.

| | |
|---|---|
| NV12→RGB | 0.183 ms → **5460 fps** |
| motion history | 0.175 ms → 5702 fps |
| bilateral r=4 (81 taps) | 2.12 ms → 403 fps |
| pinned vs pageable H2D | 11.3 vs 3.5 GB/s (**3.2×**) |

**Honest result #1:** texture-unit clamping is **0.85× — slower**, consistently
across radii, the opposite of the usual advice. Hardware address clamping *is*
free, but the `tex2D` fetch carries more latency than a plain L1 load on Turing's
unified L1/texture cache, and the `min/max` it replaces is two cheap ALU ops.
Texture units pay for *filtered* or strided sampling, not for bounds checks.

**Honest result #2:** zero-copy does not automatically win. `cudaMemcpy` streams
one full-width DMA burst; a zero-copy kernel issues fine-grained PCIe reads.
Resident VRAM is ~15× faster than either — keeping frames on-device is what
matters.

**CUDA IPC works cross-process on Windows**, verified with a real child process.
This is widely documented as Linux-only.

**NVDEC/NVENC:** the card has the engines (nvidia-smi reports encoder/decoder
counters), but `nvcuvid.h` / `nvEncodeAPI.h` ship in NVIDIA's separate **Video
Codec SDK**, not the CUDA Toolkit. Every kernel here consumes NV12 directly, the
layout NVDEC produces — dropping in the SDK replaces the frame source, not the
processing path.

## 6. Audio spectrogram + phase vocoder (`06-audio`)

Batched cuFFT STFT, phase vocoder pitch shift, overlap-add resynthesis.
1024-point frames, 75% overlap, 48 kHz.

| Channels | STFT | Pitch shift |
|---|---|---|
| 1 | 0.09 ms (11,121× real time) | 0.27 ms (3,737×) |
| 32 | 1.03 ms (31,213×) | 2.53 ms (12,647×) |
| 512 | 13.73 ms (**37,287×**) | — |

Pitch shift verified by **independent brute-force frequency measurement**: a
440 Hz tone shifted ×2 measures ~880 Hz. Identity resynthesis reconstructs the
input to <2% RMS, which is the strongest single check that windowing, cuFFT
normalisation and overlap-add are all right together.

**Scaling insight:** phase integration is a **recurrence along time** and cannot
be parallelised across frames. One thread owns one bin and walks every frame in
order; parallelism comes from bins × channels. That is why more channels scale
well and a single channel does not.

**Bug caught by tests:** the overlap-add buffer was sized from `max_samples`, but
its length depends on the **stretch ratio** — at ratio 2.0 the intermediate
signal is twice as long, and `cudaMemset` failed with `cudaErrorInvalidValue`.
Now grows on demand.

## 7. Monte Carlo pricing and risk (`07-montecarlo`)

cuRAND Philox, Kahan-compensated FP32 accumulation, European/Asian/barrier
options, pathwise and likelihood-ratio Greeks, portfolio batching.

European call, Black-Scholes exact **8.021352**:

| | Price | Std err | Throughput |
|---|---|---|---|
| fp32, 100M paths | 8.021233 | ±0.0013 | 9,967 M paths/s |
| antithetic | 8.020891 | ±0.0007 | 8,469 M paths/s |

Estimate sits **0.09 standard errors** from exact; antithetic cuts standard
error 1.78×.

Greeks vs closed form — delta −3.9e-05, gamma −1e-06, vega −2e-04.

Portfolio: **10,000 contracts × 100k paths in 67.8 ms**, one launch with one
block per contract. Worst contract 3.88σ from closed form.

**Serious bug caught here.** `block_add()` reuses one `__shared__` array across
five consecutive calls. Without a barrier at entry, warps racing into call *N+1*
overwrote the array while warp 0 was still reading it for call *N*. The
corruption was **timing dependent**: invisible at 4M paths, and **817 standard
errors wrong at 100M**, where longer per-thread loops let warps drift apart.
There is now a regression test at 100M paths.

> Concurrency bugs scale in with occupancy and loop length. A correctness suite
> that only runs small inputs proves very little.

## 8. Graph engine — PageRank / SSSP (`08-graph`)

CSR layout, push and pull PageRank, thread- and warp-per-node balancing,
Bellman-Ford SSSP with `atomicMin` on float bit patterns.

2M nodes, 16M edges, power-law (max in-degree **122,355**):

| | Time | Throughput |
|---|---|---|
| pull / warp | 669 ms | 478 M edges/s |
| pull / thread | 2117 ms | 159 M edges/s |
| push / warp | 638 ms | 527 M edges/s |

**Warp-per-node is 3.2× faster** than thread-per-node — the load-balancing win,
since a hub's neighbour list otherwise stalls 31 idle lanes.

**Honest result:** push is **not** slower than pull here, contradicting the usual
advice. The skew determines it: this generator gives every node ≤15 out-edges
while in-degrees follow a power law, so push scatters from uniformly small lists
while pull gathers 122k edges into one node. The rule is not "pull beats push" —
it is *put the parallelism on the side that is not skewed*.

SSSP verified against a CPU Dijkstra; grid distances verified against Manhattan
distance exactly.

---

## 9. Vector ANN search (`09-vector-ann`)

IVF-Flat: k-means partitioning, warp-cooperative distance kernels,
register-resident top-k. 200k x 768-dim vectors (614 MB).

Recall/latency curve, k=10, 1024 queries:

| nprobe | recall | queries/s | speedup vs exhaustive |
|---|---|---|---|
| 1 | 14.5% | 3610 | 23.9x |
| 16 | 68.4% | 437 | 2.9x |
| 64 | 92.2% | 240 | 1.6x |
| 128 | **100%** | 201 | 1.3x |

**The biggest win was coalescing**, measured both ways: one lane per candidate
gives 43.4 GB/s (23% of peak); having the whole warp cooperate on one row gives
**92.9 GB/s (48% of peak), 2.14x**. At 768 dims a row is 3072 bytes, so
lane-per-candidate puts neighbouring lanes 3072 bytes apart and every load
becomes its own transaction.

**Bug caught:** probe selection used a register top-k capped at 32, so any
`nprobe > 32` was silently truncated. Recall plateaued at 80% with no error
reported anywhere.

## 10. DPI packet matching (`10-dpi-matching`)

Aho-Corasick with a dense 256-way goto table, `atomicOr` bitmask reporting,
`__ldg()` for the transition table. **14 Gbit/s** against 1024 signatures.

**Honest result:** zero-copy is **7x slower** here, contrary to the usual advice
for capture rings. The automaton walks bytes serially with data-dependent
transitions, so mapped memory pays PCIe latency per byte and nothing coalesces.
Zero-copy suits bulk coalesced streaming, not pointer-chasing.

At 1.75 GB/s this sits near 1% of peak, and the benchmark says exactly why:
uncoalesced payload reads (thread *p* reads packet *p*, so byte *i* of adjacent
packets is ~1 MTU apart) and an 8.3 MB goto table that will not fit 1 MB of L2.

## 11. Optical flow / KLT (`11-optical-flow`)

Harris corners with shared-memory halo tiles, Gaussian pyramids, iterative
Lucas-Kanade with an explicit 2x2 inverse.

1080p: Harris **0.43 ms (2346 fps)**, tracking **1.7-4.7 ms**, mean error
**0.008 px**. Comfortably past the 120 fps target.

**Bug caught:** the kernel used one array as both the template anchor in frame A
and the moving estimate in frame B. Once a coarse level refined the position,
finer levels sampled the template at the wrong place — so the pyramid actively
made things **worse** (33 px error at 4 levels vs 12.5 px at one). Separating
the two arrays fixed it.

## 12. SHA-256 / proof of work (`12-crypto-hash`)

Fully unrolled rounds, rolling 16-word message schedule, `__constant__` round
constants, `__funnelshift_r` rotations. Validated against the FIPS 180-4
vectors, and mined nonces are re-verified on the host.

**1.26 GH/s — about 72% of integer peak.** Verified zero spill:

```
ptxas: 45 registers, 0 bytes spill stores, 0 bytes spill loads   (hashrate)
ptxas: 54 registers, 0 bytes spill stores, 0 bytes spill loads   (mine)
ptxas: 46 registers, 0 bytes spill stores, 0 bytes spill loads   (hash_batch)
```

**Reports no mining hashrate, deliberately.** Charging the full budget ignores
the early exit and gave rates *above* the raw kernel — impossible. Estimating
from the winning nonce is also wrong: the launch has far more blocks than fit on
14 SMs, so blocks run in waves and the later ones never execute when an early
block wins. An honest count needs an atomic in the inner loop, which would slow
the thing being measured. Time-to-solution is reported instead.

## 13. SpMV and conjugate gradient (`13-spmv`)

Four formats — CSR scalar, CSR vector, ELLPACK (column-major), and a hybrid
ELL+CSR split at the 90th percentile row length.

| Matrix | Best format | GFLOP/s | A-stream |
|---|---|---|---|
| banded, 4.5M nnz | Hybrid | 35.3 | 149 GB/s (78% peak) |
| power-law, 1.8M nnz | Hybrid | 7.5 | 37 GB/s |
| 2D Laplacian, 2.4M nnz | ELLPACK | 30.4 | 134 GB/s (70% peak) |

**No format wins everywhere, which is the whole point.** CSR vector is the
*slowest* option on banded and Laplacian matrices (31 of 32 lanes idle on rows
of length 5) and the second fastest on the skewed one. Same kernel, opposite
verdict, decided entirely by the row-length distribution.

ELLPACK on the power-law matrix needs **260x the real non-zeros — 3.84 GB**,
more than this card can sensibly allocate, so the benchmark refuses to run it.
Hybrid does the same work in 23 MB.

Dot products accumulate in **double**: CG compares residuals shrinking by orders
of magnitude, and FP32 accumulation destroys exactly the digits the stopping
test depends on. The SpMV stays FP32; only the reduction needs the precision.

## 14. CUDA primitives, measured in isolation (`14-primitives`)

The other thirteen projects solve problems and use whatever primitives those
problems need. This one inverts that: each section isolates ONE mechanism, so
the number is attributable to that mechanism and nothing else.

**CUDA Graphs** — launch overhead on a chain of tiny kernels:

| kernels/iter | stream | graph | speedup |
|---|---|---|---|
| 1 | 1.94 ms | 1.72 ms | 1.13x |
| 10 | 18.22 ms | 2.66 ms | 6.84x |
| 100 | 178.38 ms | 26.06 ms | **6.84x** |
| 200 | 176.98 ms | 29.56 ms | 5.99x |

About **7.6 microseconds saved per launch**, and results bit-identical to the
stream path. The single-kernel row is the control: with nothing to amortise
graphs do essentially nothing, which is the correct behaviour and worth showing.

**Unified Memory** — 16 MB, host produces, GPU transforms, host consumes:

| mode | ms/pass |
|---|---|
| explicit copy | 17.4 |
| managed (naive) | **138.5 — 8x slower** |
| managed + advise | unavailable |
| managed + prefetch | unavailable |

Managed memory is dramatically *worse* here, and both tuning APIs are
unavailable: this device reports `concurrentManagedAccess = 0`, which is the
Windows WDDM driver model, and **both `cudaMemAdvise` and `cudaMemPrefetchAsync`
return `cudaErrorInvalidDevice` without it**. The same GPU on Linux reports 1 and
the rows run — the limit is the driver model, not the hardware.

**Shared memory bank conflicts** — 32 banks, one warp per block:

| stride | conflict | vs stride-1 |
|---|---|---|
| 1 | none | 1.00x |
| 4 | 4-way | 1.24x |
| 16 | 16-way | 2.58x |
| 32 | 32-way | **4.41x** |

**Occupancy** — and why it is not a performance target:

| block | occupancy | measured |
|---|---|---|
| 32 | 50% | 1.276 ms |
| 128 | 100% | **0.824 ms** |
| 256 | 100% | 0.920 ms |
| 1024 | 100% | 1.065 ms |

Block sizes 64–1024 all reach **100% occupancy** and still differ by ~25% in
runtime. `cudaOccupancyMaxPotentialBlockSize` suggests **1024 — the slowest row
here**. It optimises occupancy, which is what it claims; it does not promise
speed. Treat it as a starting point to measure from.

**`__activemask`** — divergence made visible. A warp split by `lane & 1`: even
lanes see `0x55555555`, odd lanes `0xAAAAAAAA`, 16 active each. The hardware
runs the halves in sequence and each half sees only itself.

## Cross-cutting lessons

1. **Decide memory-bound vs compute-bound before optimising.** Projects 1–6 and
   8 are memory bound; only 7 is compute bound. The right move is opposite.
2. **Measure, don't assume.** Four "obvious" optimizations lost when measured:
   warp-cooperative hashing, texture-unit clamping, kernel fusion at small
   sizes, and zero-copy transfer.
3. **Warm up before timing.** An un-warmed first launch made q_proj read 31 GB/s
   instead of 147, and made zero-copy look faster than VRAM — physically
   impossible.
4. **Compare with the right metric.** Per-element relative error falsely flagged
   a correct INT4 kernel; under cancellation only an L2 norm is meaningful.
   Robin Hood's benefit is invisible in insert-loop counts and obvious in
   displacement.
5. **Verify against something independent.** CPU reference models, closed-form
   Black-Scholes, brute-force k-NN, Dijkstra, brute-force frequency estimation.
6. **Test at production scale.** The Monte Carlo race was invisible at 4M paths
   and catastrophic at 100M.

## CUDA 13 / Windows gotchas hit along the way

- `memoryClockRate` and `clockRate` **removed** from `cudaDeviceProp` — use
  `cudaDeviceGetAttribute`.
- `thrust` and `cub` moved to `include/cccl/`.
- Thrust needs `-std=c++17` **and** `-Xcompiler /Zc:preprocessor` on MSVC.
- A non-integral `static const float` at namespace scope is unusable in device
  code — use `constexpr`.
- Math-library DLLs live in `bin/x64`, not `bin`. A shell opened before the
  toolkit was installed dies with `0xC0000135` and no message. CMake stages
  them next to the binaries here.
- `nsys --trace osrt` is Linux-only and rejects the whole invocation on Windows.
- Windows PowerShell turns *any* native-tool stderr into a terminating error
  under `$ErrorActionPreference = "Stop"`, so a benign nsys warning aborts the
  script. Use `Continue` and check `$LASTEXITCODE`.
- `(x86)` in a path closes an `if (` block early in a `.bat` file unless quoted.

## Nsight Compute permissions

`ncu` needs GPU performance counter access, blocked by default on consumer
cards (`ERR_NVGPUCTRPERM`). Applied here, **takes effect after reboot**:

```
reg add "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" /v RmProfilingAdminOnly /t REG_DWORD /d 0 /f
```

`nsys` is unaffected and works without it.
