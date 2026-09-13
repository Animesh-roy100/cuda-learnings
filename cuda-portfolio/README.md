# CUDA systems portfolio — GTX 1650 (Turing, `sm_75`)

Nineteen production-structured CUDA projects. Every one builds, runs, and
self-verifies against an independent reference. **Every number below was
measured on this machine**, not estimated.

**Hardware:** GTX 1650, Turing `sm_75`, 14 SMs, 896 CUDA cores, 4 GB GDDR6,
**192 GB/s peak bandwidth**, PCIe gen3 ×16, FP64 at 1/32 rate, and no Tensor
Cores according to the spec sheet — a claim project 16 measures and partly
contradicts.

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

## 15. Warp and block primitives (`15-warp-primitives`)

Project 14 isolates *subsystems* — graphs, unified memory, occupancy. This one
goes a level down to individual **instructions**: every warp shuffle, vote,
barrier variant, fence, atomic, bit intrinsic and packed dot product, each with
a host reference that fails if the semantics were misunderstood. 27 tests.

**Shuffles** — lane values 1..32, exchanged through registers, no shared memory:

| intrinsic | lane 0 | lane 31 | what it builds |
|---|---|---|---|
| `__shfl_sync(7)` | 8 | 8 | broadcast |
| `__shfl_up_sync` | 1 | **528** | inclusive prefix scan |
| `__shfl_down_sync` | **528** | 1024 | reduction tree |
| `__shfl_xor_sync` | **528** | **528** | butterfly |

528 = 32·33/2 is the warp total. The last two rows are the distinction worth
remembering: `down` leaves the answer in lane 0 only; `xor` leaves it in *every*
lane, for the same instruction count.

**Votes** — predicate true on every third lane:

```
__ballot_sync   0x49249249   01001001001001001001001001001001
__popc          11 lanes true
__all_sync      false      __any_sync   true
__activemask    0xffffffff
```

The trap here cost a debugging session. Every vote must be evaluated with the
**whole warp converged**, before any divergence — calling `__all_sync` or
`__activemask` inside `if (lane == 0)` polls only the lanes still active there,
which is lane 0 alone. It returns `all_true = true` and `activemask = 0x1`, both
wrong, both plausible-looking.

**Block barrier variants** — 256 threads, every fourth true: `__syncthreads_count`
→ 64, `_and` → false, `_or` → true. Each is a barrier *and* a block-wide
reduction in one instruction; by hand it costs a shared array plus two syncs.

**Atomics** — the full set over 1..1000: add 500500, sub −500500, min 1, max
1000, and 0, or 1023, xor 1000, exch 128 (nondeterministic by design), CAS 1
winner of 1000 racers. `atomicInc` → 1000 and `atomicDec` → **4294966296**:
both **wrap**, they do not saturate.

**Packed dot products** — `__dp4a([1,2,3,4],[10,20,30,40])` = 300 in one
instruction. `__dp2a([100,200] i16, [7,8] i8)` = 2300 — and it is **mixed**
precision, int16 against *int8*. Feed it two int16 operands and it compiles,
runs, and silently reads only the low byte of the second: 700 instead of 2300,
no error of any kind. The API here is typed `std::int8_t` so the mistake cannot
be made through it.

**FMA** — `__fmaf_rn(a,b,c)` rounds once; `a*b+c` rounds twice:

| a·b+c | fused | separate | exact |
|---|---|---|---|
| (1+ε)(1−ε) − 1 | **−1.421e−14** | 0.000e+00 | −1.421e−14 |
| 3 · (1/3) − 1 | **2.980e−08** | 0.000e+00 | 2.980e−08 |

The fused result matches the exact double-precision answer; the separate one
rounds the difference away before the add ever happens and reports a confident
zero.

**Fences** reported **0 torn reads both with and without** `__threadfence()`.
That is recorded as-is rather than tidied up: the unfenced path is unsafe *by
construction*, and absence of a visible failure is not evidence of correctness
in a memory model. The test suite asserts only on the fenced path.

## 16. Memory layout and advanced subsystems (`16-layout-advanced`)

Six features, each measured on this card rather than quoted from a guide. One
does not help. One helps considerably more than the spec sheet predicts.

**AoS vs SoA** — 4M particles of 6 floats, kernel reads 3 of them:

| layout | ms | useful GB/s |
|---|---|---|
| AoS `{x,y,z,vx,vy,vz}[]` | 0.679 | 98.8 |
| SoA `x[], y[], z[], ...` | **0.377** | **178.1** |

**1.80× for rearranging the same bytes**, results bit-identical. AoS moves 24
bytes per particle to use 12 — the unread velocities sit inside the same 32-byte
sectors the positions do, so DRAM delivers them regardless. SoA reaches **93% of
peak**; AoS cannot get there at any occupancy.

**Shared-memory bank conflicts** — a 32×32 tile read column-wise:

| declaration | ms |
|---|---|
| `tile[32][32]` | 6.097 |
| `tile[32][33]` | **0.771** |

**7.91× for 128 extra bytes.** At width 32 the whole warp asks for bank `ty` and
the access serialises 32 ways; at width 33 each row shifts one bank and the lanes
spread across all 32. One character in a declaration.

**Cooperative groups** — `cg::reduce` over a 32-lane tile → 32, over a
256-thread block → 256, `grid.size()` → 14336, and `grid.sync()` held across 1M
elements. The grid size is **not a free choice**: a grid-wide barrier deadlocks
unless every block is resident simultaneously, so it comes from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, not from the problem size.

**Dynamic parallelism** — the host issued one launch; 8 more came from device
code. Under CUDA 12+ (CDP2) the device-side `cudaDeviceSynchronize()` **no
longer exists**; the guarantee that remains is that the parent grid is not
complete, as the host observes it, until its children are. Costs `-rdc=true`
and `cudadevrt`, which is why this is the only target in the repo overriding
`CUDA_SEPARABLE_COMPILATION`.

**Tensor Cores (WMMA)** — 512×512. Two kernels could not have settled this,
because switching to WMMA also halves the bytes read. Three kernels, one variable
changed at a time, can:

| kernel | ms | vs above |
|---|---|---|
| fp32 operands, fp32 math | 1.192 | — |
| fp16 operands, fp32 math | 1.190 | 1.00× |
| fp16 operands, `mma_sync` | **0.485** | **2.45×** |

NVIDIA lists the GTX 16-series as shipping **without** Tensor Cores, so the
expectation going in was "correct but not faster". The middle row makes the
answer readable: at this size every byte is reused 512 times, so the kernel is
compute-bound and narrowing the operands buys **nothing** — leaving the entire
2.45× on `mma_sync`. Whatever the spec sheet says, this chip retires HMMA
meaningfully faster than FP32 FMA. Max |wmma − fp32| = 0.0106, which is fp16
input precision, not a bug; a reading of exactly 0 would mean the fp16 path
never ran.

**Asynchronous shared-memory copy** — `cg::memcpy_async` vs load-and-barrier:
0.211 ms vs 0.207 ms, **0.98×**. No speedup, and none was expected.
`cg::memcpy_async` compiles from sm_70, but the `cp.async` instruction that lets
DRAM write straight into shared memory arrived with **Ampere, sm_80**. Here it
falls back to the ordinary load. This is the cleanest example in the repo of the
thing worth internalising: *the API being available is not the same as the
hardware being there, and only the clock can tell the two apart.*

## 17. Tensor-core GEMM, FP16 and INT8, against cuBLAS (`17-tc-gemm`)

A from-scratch `C = A*B` built on WMMA fragments, measured against cuBLAS at
N = 4096. Fourteen paths, arranged so each comparison changes one thing:

| path | ms | GFLOPS | % of cuBLAS (same precision) |
|---|---|---|---|
| fp32 tiled, CUDA cores | 615.9 | 111.6 | 14.7% |
| fp16 WMMA, from global | 241.8 | 284.2 | **156.6%** |
| fp16 WMMA, staged (best band) | 242.1 | 283.8 | 156.4% |
| int8 WMMA, from global | 110.9 | 619.8 | **12.8%** |
| int8 WMMA, staged (best band) | 110.4 | 622.7 | 12.8% |
| int8 `__dp4a`, CUDA cores | 785.7 | 87.5 | 1.8% |
| cuBLAS sgemm | 90.4 | **760.2** | - |
| cuBLAS fp16, 32F compute | 406.4 | 169.1 | - |
| cuBLAS fp16, 16F compute | 378.8 | 181.4 | - |
| cuBLAS int8 | 14.2 | **4854.5** | - |

Every path is checked against an FP64 host product, not against cuBLAS: a
reference computed by one of the things under test cannot catch a bug they share.

**INT8 is where tensor cores work on this card.** Tensor-core INT8 is **7.09x**
`__dp4a` on identical operands - the same instruction project 01's GEMV is built
on, doing the same multiply-adds in CUDA cores. cuBLAS INT8 is 6.4x cuBLAS FP32.
The hand-written INT8 path reaches 12.8% of cuBLAS; what the remaining gap is made
of needs an Nsight Compute roofline, which on this machine needs elevation, so it
is stated as unexplained rather than guessed at.

**FP16 does not pay here at all.** The hand-written FP16 path beats cuBLAS's FP16
by 1.57x - but cuBLAS FP32 beats *every* FP16 path, including cuBLAS's own, by
2.7-4.5x. cuBLAS evidently does not route FP16 to tensor cores on this chip. That
is the single result most likely to change on a T4.

### Staging lost, and the reason was not the one in the comment

The textbook tiled design stages each band of operands into shared memory so
fragments load from L1. The first version lost to loading straight from DRAM -
**0.62x for fp16, 0.46x for int8** - and the kernel's own comment blamed bank
conflicts: fp16 packs two elements per 4-byte bank word, int8 packs four.

A word-aligned control that removes the conflicts won back **1.27-1.34x**. Real,
but it still lost. Asking the occupancy API explained the rest:

| kernel (band 512) | blocks / SM | compute warps / SM |
|---|---|---|
| fp16 WMMA from global | 8 | **32** |
| fp16 WMMA staged | 2 | **2** |

The 32 KB staging arrays let only two blocks fit per SM. The staged kernel had
been running with a sixteenth of the warps - and still reached 0.6-0.8x, so per
warp it was far more efficient. The band width is a compile-time parameter
(it sizes `__shared__` arrays), so every width was compiled and swept:

| band | fp16 by element | int8 by word | compute warps |
|---|---|---|---|
| 16 | 0.49x | 0.31x | 16 |
| 64 | 0.92x | 0.80x | 16 |
| **128** | **0.98x** | **0.99x** | 8-16 |
| 256 | 0.93x | 0.81x | 4-8 |
| 512 | 0.60x | 0.93x | 2-4 |

(Relative to loading from global, N = 2048.) Narrow bands do too little work per
barrier; wide ones evict warps. **At the best band, staging ties loading from
global and never beats it** - and it cannot, because a one-warp-per-block design
caps at 16 resident warps per SM where the global kernel, four warps per block,
gets 32. On this card, WMMA fragment loads straight from DRAM are as good as any
staging scheme.

Software pipelining - a second warp staging the next band while the first
multiplies - reached **0.99x fp16 / 0.83x int8**. It is the sm_75 substitute for
`cp.async`, which on sm_80 lets DRAM write into shared memory while the warp keeps
computing; here the overlap costs a second buffer, and the buffer costs occupancy.

### Measured constraints

- **cuBLAS INT8 refuses odd multiples of 16.** `GemmEx` with int8 operands accepts
  N = 16 and every multiple of 32, and returns `CUBLAS_STATUS_NOT_SUPPORTED` for
  48, 80, 112, ... 496 - although the documented requirement is only multiples of
  4. The hand-written INT8 paths accept every multiple of 16; the refusal is
  surfaced as a typed `tc::Unsupported`, not a generic failure.
- **The shared-memory limit per kernel is 48 KB.** The device linker rejected the
  fp16 pipelined kernel at band 512 ("0xc000 max") - not the 64 KB figure the
  first comment assumed. Instantiations over budget are excluded at compile time.
- **Every launch is kept under the Windows display-driver timeout.** One tiled
  FP32 product at N = 4096 takes 616 ms; a single launch running past ~2 s resets
  the GPU. The hand-written paths launch in bands of 16 tile rows.
- **The experimental 4-bit fragment (`u4`, 8x8x32) is exact** on a host-checked
  matrix - kept as a correctness check on a type deprecated after Turing.

## 18. FlashAttention-style fused attention, from scratch (`18-flash-attention`)

`O = softmax(QK^T / sqrt(d)) V`, three ways, all checked against an FP64
log-sum-exp reference - bidirectional and causal, partial warps and tiles,
logits in the hundreds (where a naive `exp` overflows FP32 many times over),
and the exact identity that a causal first query returns `v_0`:

- **naive** - what frameworks do eagerly: cuBLAS materializes the
  `[heads][seq][seq]` score matrix, a kernel softmaxes each row, a second GEMM
  applies it to V.
- **fused** - one pass over the keys per query with an *online softmax*: a
  running max, normalizer and weighted sum, updated key by key, every exponent
  kept <= 0. No `seq x seq` matrix ever exists.
- **fused, tiled** - the same pass with keys and values staged into shared
  memory a tile at a time. Tile size sizes the `__shared__` arrays, so each size
  is compiled and swept.

**The fused kernels do not beat materialized attention on this card.** They
lose by 3-4x at every length that fits in VRAM:

| seq (8 heads) | naive | fused, tiled | device memory |
|---|---|---|---|
| 1024 | **8.0 ms** | 19.7 ms | 40 MB vs **8 MB** |
| 4096 | **85.7 ms** | 313.5 ms | 544 MB vs **32 MB** |
| 8192 | **358.6 ms** | 1106.0 ms | 2112 MB vs **64 MB** |

They win on two axes. **Memory, 17-33x**, always. And **time, once the score
matrix no longer fits**: at 16 heads x 8192 tokens naive needs 4224 MB on a
4096 MB card. It does not fail - the Windows sysmem fallback moves the matrix
into system RAM - and takes **7118 ms against 2205 ms** for the fused kernel.

Why they lose below that point is measured, not assumed:

| kernel | shared per block | blocks / SM |
|---|---|---|
| fused, from global | 16.6 KB | 3 |
| fused, tile 8 | 20.7 KB | 3 |
| fused, tile 16 | 24.8 KB | 2 |
| fused, tile 32 | 41.0 KB | **1** |

The per-query staging arrays - query and running output, `[32][65]` so that 32
lanes never share a bank - cost the occupancy that project 17 found decisive:
one to three warps per SM. And the online update is compute-bound at about three
times the multiply-adds per (query, key) pair of the GEMM cuBLAS runs. At these
sizes the N x N matrix fits comfortably and moves at 192 GB/s, so avoiding it
buys little. The advantage FlashAttention is known for appears when the score
matrix is the bottleneck - here, only past VRAM - while the memory advantage is
unconditional.

Two more measured details:

- **Causal masking helps the fused kernels and barely helps naive.** At 4096
  tokens, causal cuts fused-from-global from 370 to 155 ms (2.4x), while naive
  goes from 84 to 63 ms: it still multiplies the whole matrix and only zeroes the
  masked half in the softmax. The tiled kernel gains less (326 to 196 ms)
  because it still stages every tile, masked keys included.
- **The smallest tile won the sweep** (tile 8: 1.16x over reading keys from
  global; tile 32: 0.48x), for the same reason: each larger tile costs a block.

## 19. An LLM inference engine, GGUF to text (`19-llm-engine`)

Projects 01, 14 and 18 assembled into a program that runs TinyLlama-1.1B-Chat
(Q4_0 GGUF) end to end: tokenizer, 22 transformer layers with grouped-query
attention, RoPE, SwiGLU, a device KV cache, sampling, and CUDA graphs over the
decode loop.

```
> Explain in three sentences why GPUs are good at matrix multiplication.

GPUs (Graphics Processing Units) are good at matrix multiplication because they
are designed to perform complex calculations quickly and efficiently. ...
```

| configuration | perplexity | prefill tok/s | decode tok/s |
|---|---|---|---|
| float activations (W4A16) | 5.594 | 11.3 | 11.2 |
| int8 activations (W4A8, `__dp4a`) | 5.551 | 127.7 | 119.8 |
| int8 + CUDA graphs | 5.551 | **133.9** | **125.0** |

Perplexity is on the opening of *Alice's Adventures in Wonderland*. Model
weights are not in the repository: `scripts/fetch_model.ps1` / `.sh` download
the 638 MB file and verify its SHA-256, and every model-dependent test skips
cleanly without it.

**Validated against an independent host forward pass**, FP32 from dequantized
weights, sharing no code with the device path beyond the parser and block
dequantizers. The float path matches it to **2.9e-6** relative error in the
logits; int8 agrees on the answer; CUDA graphs reproduce the stream path's
logits bit for bit. The tokenizer reproduces known Llama ids
(`The capital of France is` -> `1 450 7483 310 3444 338`) and round-trips
Unicode through byte fallback.

### The bug that validation could not catch

The first outputs were fluent nonsense, and the device engine matched the host
reference to 3e-6 - both were wrong in the same way, so the bug was in what they
shared. Two things were in doubt: the Q6_K output projection's bit layout, which
had been written from memory, and everything else.

A logit lens and a bit-correlation test were both inconclusive. The decisive
experiment used the fact that the output projection is the *last* operation:
compute the final hidden states once, then score every candidate Q6_K
arrangement by the perplexity of the predictions it produces:

| high 2 bits | best perplexity | worst perplexity |
|---|---|---|
| **quarters per 128 weights** | **4.99** | 5.93 |
| any other arrangement | 377 | 352,000 |

That one run proved the Q6_K layout *and* the 22-layer body at once: nothing
short of a correct transformer reaches perplexity 5. The low-nibble layout that
scored best, split per 128, matches ggml's AVX2 kernel once its shape is known.
The arrangement is now documented in `quant.h` with how it was determined.

### Two tokenizer facts the file forced

- **Every vocabulary score is 0.** All 32,000. A SentencePiece encoder that picks
  merges by score picks arbitrarily. The file ships 61,249 merges instead, so the
  encoder merges by rank.
- **Control tokens arrive as text.** The chat template writes `</s>` literally,
  and it must become token 2, not five characters.

### Decode attention: the latency was hiding in the position

Decode measured slower than prefill although both run the same step. Timing the
step at increasing positions found why:

| position | single warp | warp per head |
|---|---|---|
| 16 | 8.90 ms | **7.76 ms** |
| 128 | 47.61 ms | **8.47 ms** |
| 1008 | 77.17 ms | **15.54 ms** |

The first attention kernel ran all 32 heads on **one warp**, walking every
cached key per head - latency linear in position on a single warp. The fix
launches one warp per head, strides each head's keys across its lanes, and
replaces the key-by-key online softmax with reductions across the warp (max,
normalizer, weighted sum). **5.0x at position 1008**, and prefill and decode
throughput now agree, which is what confirms the diagnosis. The paged runtime
keeps the per-head launch and reads keys through the device page table.

### What CUDA graphs were worth

**6% on decode.** Project 14 measured 6.84x for a chain of tiny kernels; here
each token launches ~270 kernels whose work, not their launch, dominates. The
graphs work - every per-token launch reads its position from device memory, so
they are byte-identical and capturable - they just have little to remove.

### The production runtime

The engine was then rebuilt to the production-readiness guide in
`01-gguf-inference`; [`01-gguf-inference/PRODUCTION_STATUS.md`](01-gguf-inference/PRODUCTION_STATUS.md)
maps every item of it to code and a test. In short:

- **Ownership.** `GgufModel` (immutable, validated) -> `Runtime` (one device:
  weights, KV page pool, workspace, CUDA graphs) -> `Sequence` (a page table
  and a position; may outlive its runtime) -> sampler and generator.
- **Loading is transactional.** Config relationships, every tensor's type and
  shape, and the memory plan against free VRAM are checked before the first
  allocation; `llm::Error` names the model, tensor, shape, sequence and position.
- **Paged, batched device KV cache.** Attention reads device page tables; forks
  share pages copy-on-write; `step()` over a batch is a transaction, so KV
  exhaustion or a full context changes no sequence at all.
- **Observability.** TTFT, prefill/decode tokens/s, per-stage latency, memory
  plan, KV utilization, batch and context limits, quant format, driver release
  and CUDA versions, as a JSON record (`chat --json`).

| batch | ms / step | tokens/s total |
|---|---|---|
| 1 | 8.51 | 117.5 |
| 2 | 13.46 | 148.6 |
| 4 | 23.76 | 168.4 |

### Checked against llama.cpp

`compare_llamacpp` replays llama.cpp's own `--kl-divergence-base` output
(CPU build b10932, 8 x 512-token chunks of wikitext-2) through this runtime:

| mode | mean KL divergence | p99 | same top token | PPL ours | PPL llama.cpp |
|---|---|---|---|---|---|
| float | 0.00071 | 0.0049 | 98.48% | 23.376 | 23.319 |
| int8 | 0.00072 | 0.0039 | 98.24% | 23.379 | 23.319 |

It also found that the two **tokenize this file differently**: llama.cpp
orders merges by vocabulary score, and this GGUF's 32,000 scores are all zero.
The runtime's llama.cpp-compatible mode reproduces all 4,096 of llama.cpp's
tokens; its default follows the merge list, and the model prefers it by a wide
margin - **0.901 bits/byte against 1.313** on the same text.

### Honest limits

- Prefill runs token by token through the decode path. A batched prefill with
  18-flash-attention's fused kernel and 17-tc-gemm's GEMM is the obvious next
  step and is not built.
- The attention kernels are compiled for head_dim 64 and at most 32 heads, which
  covers the Llama-family models of this size but not larger ones; other shapes
  are refused at load as `Unsupported`.
- Only the `llama` architecture with Q4_0 matrices and embeddings (Q6_K, Q4_0 or
  F32 output projection) is loaded; anything else is refused with the list of
  offending tensors.

## 20. PyTorch extension (`20-torch-extension`)

The Q4_0 GEMV from project 19 as a pip-installable PyTorch op with autograd -
`Q4Linear.from_float(nn.Linear)`. Separate from the CMake build; see
[its README](20-torch-extension/README.md). At TinyLlama's FFN shape it is
**2.5x faster than PyTorch's fp32 linear at batch 1** and 1.4x at batch 16, with
6.4x less weight memory; at batch 128 cuBLAS wins by 3.5x, because a batched
GEMM amortizes weight reads that a per-element GEMV repeats. 14 tests, every one
checked against `F.linear` in float64.

## Error paths (`error-paths`)

Every other suite proves the code is right when things go well. This one proves
it fails cleanly when they do not - and it found more real bugs than any
project did.

| finding | before | after |
|---|---|---|
| failed `StftProcessor` construction | **3.4 GB stranded** | released |
| failed `VideoPipeline` construction | free VRAM 3296 MB -> **0 MB** | released |
| `PagedKvCache`, byte count = 2^64 | wraps to 0, constructs with a null slab | `invalid_argument` |
| `GpuHashTable(SIZE_MAX)` | **infinite loop** | `invalid_argument` |
| `GpuHashTable(2^33)` | reaches `cudaMalloc`; mask would truncate | `invalid_argument` |
| `CU_CHECK` on a failed call | next kernel check blames a healthy kernel | error consumed |

The constructor leak was in **eleven classes across ten projects**: resources
acquired in the constructor body, released only in the destructor, which C++
never runs after a constructor throws. Ownership moved into `Impl::~Impl`. Each
leak test was run against the original source first and fails there exactly as
described - one that did not (a 40-byte leak below `cudaMemGetInfo`'s
resolution) was deleted.

**Recoverable, stale, and sticky are three different things**, and the suite
tests each:

- *Recoverable* - out-of-memory, bad launch dimensions, oversized copies. The
  call fails; the context is fine.
- *Stale* - a failed call's error stays recorded until something reads it, and
  the next `cudaGetLastError()` returns it from wherever it is called. This is
  what `14-primitives` originally misdiagnosed as context poisoning.
- *Sticky* - an illegal device address. The context is gone; the death test
  confirms the child cannot launch anything afterwards while the parent can.

**The Windows allocator does not fail where Linux does.** With the driver's
sysmem fallback, 7168 MB was allocatable on this 4 GB card against 3294 MB free,
so the constructor tests size their failures from a measured allocation
capacity rather than from `cudaMemGetInfo`.

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
