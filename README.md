# CUDA on a GTX 1650

Learning CUDA from first principles, then building production-structured systems
with it. Everything here targets **Turing `sm_75`** and runs on a single 4 GB
laptop GPU.

**Every number in this repository was measured on that machine.** Nothing is
estimated, and several results contradict the conventional advice — those are
called out rather than quietly dropped.

## Hardware and toolchain

| | |
|---|---|
| GPU | GTX 1650 (TU117), Turing `sm_75`, 14 SMs, 896 CUDA cores |
| Memory | 4 GB GDDR6, **192 GB/s peak** (176–184 GB/s measured) |
| Notable limits | no Tensor Cores *(but see below)*, FP64 at 1/32 rate, PCIe gen3 ×16 |
| Toolchain | CUDA 13.4, MSVC 19.44, driver 616.92, CMake 4.4, GoogleTest 1.15 |

> The 128 GB/s figure usually quoted for the GTX 1650 is the **GDDR5** variant.
> This is the GDDR6 card. Tuning against 128 GB/s would mean declaring victory
> at 67% of the real ceiling.

> **On "no Tensor Cores":** that is NVIDIA's own description of the GTX
> 16-series, and it predicts that `wmma` should compile, return correct
> results, and be no faster. Project 16 measures **2.45×** against a tuned
> FP32 SGEMM — with a third control kernel proving the gain is the `mma_sync`
> instruction and not the narrower fp16 operands. The spec sheet is kept above
> as written; the measurement is reported as measured.

## No GTX 1650? Run it free on Colab

Google Colab's free tier gives you a **Tesla T4 — also Turing `sm_75`**, the
same architecture this targets. The code runs unmodified; only the machine
around it changes (Linux, CUDA 12, 16 GB instead of 4).

**[COLAB.md](COLAB.md)** has the full walkthrough, including a single cell that
goes from nothing to a passing 251-test suite.

## Layout

```
cuda-learning/     fundamentals — grid/block model, memory hierarchy, tiling
cuda-projects/     five standalone programs, one file each
cuda-portfolio/    sixteen production-structured systems: CMake, tests, profiling
cuda.code-workspace   opens all three in VS Code
```

### `cuda-learning/` — fundamentals

Five short programs, each teaching one thing: device query, the thread/block
model, host↔device transfer and event timing, and shared-memory tiling. Build
with `build.bat 02-hello\hello.cu` (Windows) or `./build.sh 02-hello/hello.cu`
(Linux and Colab).

The tiled matmul is the centrepiece: **1.46× over naive**, not the 3–4× the
textbooks quote, because Turing's 64 KB unified L1/shared cache already absorbs
most of the naive kernel's redundant reads. The textbook figures come from
Fermi/Kepler.

### `cuda-projects/` — standalone programs

Five single-file programs exploring one idea each, without build machinery in
the way. Quantized `__dp4a` GEMV, a lock-free hash table, Morton-code spatial
indexing with DBSCAN, a zero-copy/CUDA-IPC pipeline, and Monte Carlo pricing.

### `cuda-portfolio/` — production structure

Sixteen systems with modern CMake, GoogleTest suites, and profiling scripts.
Public headers contain **no CUDA syntax** (pimpl), so tests and host code
compile as plain C++20 and only `.cu` files need nvcc.

```bat
cd cuda-portfolio
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest --output-on-failure
```

```
100% tests passed out of 18 suites     (251 test cases)
```

| # | Project | Headline result |
|---|---|---|
| 1 | GGUF inference engine | INT4 `__dp4a` GEMV **3.4–5.1×** fp32; kernel matches its CPU model to **1.7e-07** |
| 2 | Image embedding pipeline | 857 img/s, 4-stream overlap 1.17× |
| 3 | Lock-free hash / KV store | Robin Hood cuts worst-case displacement **4775 → 114** |
| 4 | Spatial index, k-NN, DBSCAN | k-NN **59.8× vs brute force, bit-identical** |
| 5 | Video analytics | NV12→RGB **5460 fps**; pinned 3.2× pageable |
| 6 | Audio STFT + phase vocoder | 512 channels at **37,287× real time** |
| 7 | Monte Carlo pricing & risk | 9,967 M paths/s, **0.09σ** from Black-Scholes |
| 8 | Graph engine (PageRank/SSSP) | warp-per-node **3.2×** thread-per-node |
| 9 | Vector ANN search (IVF-Flat) | 100% recall at nprobe=128; coalescing fix worth **2.14×** |
| 10 | DPI packet matching | **14 Gbit/s** against 1024 Aho-Corasick signatures |
| 11 | Optical flow (KLT) | 1080p Harris **2346 fps**, tracking error **0.008 px** |
| 12 | SHA-256 proof of work | **1.26 GH/s**, ~72% of integer peak, zero register spill |
| 13 | SpMV + conjugate gradient | 4 formats; hybrid uses **23 MB where ELLPACK needs 3.8 GB** |
| 14 | CUDA primitives, isolated | Graphs **6.84×**; bank conflicts **4.41×**; managed memory **8× slower** |
| 15 | Warp & block primitives | Every shuffle, vote, barrier, atomic and intrinsic, each checked against a host reference |
| 16 | Layout & advanced subsystems | SoA **1.80×**; tile padding **7.91×** for 128 bytes; WMMA **2.45×** on a card with "no Tensor Cores" |

Each project's README documents its own measurements in detail.

## Seven "obvious" optimizations that lost when measured

Every one of these is standard advice. All seven were slower here:

- **Warp-cooperative hash probing** — 0.52× at load factor 0.5. It spends a
  256-byte transaction per query when the average probe chain is 1.5 slots.
  Per-thread probing lets 32 lanes resolve 32 *different* queries from one
  transaction.
- **Texture-unit boundary clamping** — 0.85×, consistently. Hardware clamping
  really is free, but `tex2D` carries more latency than a plain L1 load on
  Turing's unified L1/texture cache, and the `min/max` it replaces is two cheap
  ALU ops.
- **Fusing RMSNorm + RoPE** — 0.95×. At 8 KB neither variant is memory bound;
  both are launch-bound, and the fused kernel uses 1 of 14 SMs to share its
  reduction.
- **Zero-copy transfer** — loses to `cudaMemcpy`, which streams one full-width
  DMA burst instead of fine-grained PCIe reads.
- **Zero-copy for packet inspection** — 7× slower. The automaton walks bytes
  serially with data-dependent transitions, so mapped memory pays PCIe latency
  per byte and nothing coalesces.
- **Warp-per-row SpMV** — the *slowest* of four formats on matrices with
  uniform short rows, because 31 of 32 lanes sit idle. It is the second
  fastest on a skewed matrix. Same kernel, opposite verdict.

- **Unified Memory** — 8× slower than explicit copies for a host-produces /
  GPU-transforms / host-consumes workload, because every pass migrates the whole
  buffer both ways. And on this driver model `cudaMemAdvise` and
  `cudaMemPrefetchAsync` are not merely unhelpful, they are unavailable.

Also: **push PageRank beat pull**, the opposite of the usual guidance. The skew
decides it — this graph has uniform out-degrees and power-law in-degrees, so
put the parallelism on the side that *isn't* skewed.

## Real bugs, and what they teach

- **Shared-memory race in a block reduction** (Monte Carlo). One `__shared__`
  array reused across five consecutive calls with no barrier at entry. Warps
  racing into call *N+1* overwrote it while warp 0 still read it for call *N*.
  Invisible at 4M paths; **817 standard errors wrong at 100M**. There is now a
  regression test at full scale.
  → *Concurrency bugs scale in with occupancy and loop length. A suite that
  only runs small inputs proves very little.*
- **Capped probe length** (hash table). The warp lookup stopped after 1024
  slots, silently reporting present keys as missing once load factor ≥0.90
  produced longer clusters. Only an empty slot is a correct terminator.
- **Buffer sized from the wrong variable** (phase vocoder). The overlap-add
  length depends on the stretch ratio, not the input length. Hard failure at
  ratio 2.0.
- **Silent truncation** (ANN index). Probe selection used a 32-entry register
  top-k, so any `nprobe > 32` was quietly ignored. Recall plateaued at 80% with
  no error raised anywhere — the worst kind of bug, because nothing looks wrong.
- **Two meanings in one array** (optical flow). The KLT kernel used one array as
  both the template anchor in frame A and the moving estimate in frame B. Once a
  coarse pyramid level refined the position, finer levels sampled the template
  at the wrong place, so the pyramid made tracking **worse**: 33 px error at four
  levels versus 12.5 px at one.
- **Three wrong ways to report a hashrate** (SHA-256) and **two wrong ways to
  count SpMV bandwidth**. Both produced figures that exceeded what the hardware
  can physically do — above the raw kernel's rate, and 115% of peak memory
  bandwidth. An impossible number is the most reliable signal available that the
  measurement, not the code, is broken.

- **A read the compiler deleted** (primitives). The host-side "touch" that was
  supposed to force page migration was written `(void)m[i]`, which the optimiser
  elides outright. No read, no migration — and Unified Memory appeared **21×
  faster** than explicit copies. Accumulating into a value that is observed later
  made the read real, and the true answer is 8× *slower*.
- **An error that was stale, diagnosed as sticky** (primitives, then
  error-paths). Calling `cudaMemAdvise` without `concurrentManagedAccess` returns
  `cudaErrorInvalidDevice`, and three unrelated test suites failed after it. This
  README originally said the error *poisoned the context*. Measured later, it
  does not: the next kernel launches and computes correctly. What actually
  happened is that `CU_CHECK` threw without reading the runtime's last-error
  record, so the next `CU_CHECK_KERNEL` - in whichever test came next - found the
  old error and blamed its own, healthy kernel. The capability gate was the right
  fix for the wrong reason; `CU_CHECK` now consumes what it throws for.
- **A vote taken after the warp had already split** (warp primitives).
  `__all_sync` and `__activemask` called inside `if (lane == 0)` poll only the
  lanes still active *there* — which is lane 0 alone. They returned
  `all_true = true` and `activemask = 0x1`: both wrong, both entirely
  plausible-looking. Every vote has to be evaluated with the warp converged,
  before the branch, and the result carried in.
- **An API that exists without the hardware behind it** (layout & advanced).
  `cg::memcpy_async` compiles from sm_70, so it looks available here. The
  `cp.async` instruction that makes it genuinely asynchronous arrived with
  Ampere, so on Turing it silently lowers to an ordinary load-and-barrier and
  measures **0.98×**. Compiling is not evidence of acceleration.
- **Every constructor leaked on failure** (error-paths). Eleven classes across
  ten projects acquired device memory in the constructor body after
  `impl_(new Impl)` and freed it only in the destructor - which C++ never runs
  for an object whose constructor threw. A `StftProcessor` that failed partway
  through construction stranded **3.4 GB**; a failed `VideoPipeline` took free
  VRAM from 3296 MB to **0**. The same test run against the unfixed source fails
  exactly that way, which is the only reason to believe the fix. Ownership now
  lives in `Impl::~Impl`, so a partial construction releases what it acquired.
- **Sizes that wrapped instead of failing** (error-paths). A `PagedKvCache`
  configuration whose byte count is exactly 2^64 wrapped to zero, and
  `cudaMalloc(&p, 0)` *succeeds* - so the cache constructed with a null slab.
  `GpuHashTable(SIZE_MAX)` looped forever, because doubling toward it overflows
  to zero. Both are now rejected before they reach the allocator.
- **Tests that pass against the bug prove nothing.** A test measuring a 40-byte
  VRAM leak through `cudaMemGetInfo` passed on the unfixed code - the leak was
  below the resolution of the measurement. It was deleted, not kept as a green
  checkmark.

## Method notes

1. **Decide memory-bound vs compute-bound before optimising.** Eleven of the
   thirteen application projects are memory bound; Monte Carlo and SHA-256 are
   compute bound. The right move is opposite in each case. (14–16 are
   instrument projects: they measure one mechanism at a time rather than solving
   a problem.)
2. **Warm up before timing.** An un-warmed first launch made one GEMV read
   31 GB/s instead of 147, and made zero-copy look faster than VRAM — which is
   physically impossible.
3. **Use the right error metric.** Per-element relative error falsely flagged a
   *correct* INT4 kernel: signed dot products nearly cancel, so a tiny absolute
   error looks enormous beside one small output. Only an L2 norm is meaningful
   there.
4. **Verify against something independent.** CPU reference models, closed-form
   Black-Scholes, brute-force k-NN, Dijkstra, brute-force frequency estimation.
5. **Run a regression test against the unfixed code before trusting it.** Every
   error-path test for a leak was confirmed to fail on the original source;
   one that did not was removed.

## CUDA 13 / Windows gotchas

- `memoryClockRate` and `clockRate` were **removed** from `cudaDeviceProp` —
  use `cudaDeviceGetAttribute`.
- `thrust` and `cub` moved to `include/cccl/`.
- Thrust needs `-std=c++17` **and** `-Xcompiler /Zc:preprocessor` on MSVC.
- A non-integral `static const float` at namespace scope is unusable in device
  code — use `constexpr`.
- Math-library DLLs live in `bin/x64`, not `bin`. A shell opened before the
  toolkit was installed dies with `0xC0000135` and no message.
- `cudaMemAdvise` and `cudaMemPrefetchAsync` now take a `cudaMemLocation`
  struct where CUDA 12 took a device ordinal. Code written for either fails to
  compile on the other; `14-primitives` keeps both behind a `CUDART_VERSION`
  check so one source builds on CUDA 12 and 13 alike.
- **`cudaMalloc` can exceed VRAM on Windows.** The driver's CUDA sysmem
  fallback silently continues allocating in system RAM: on this 4 GB card
  **7168 MB** was allocatable against 3294 MB free. Code that sizes work from
  `cudaMemGetInfo` gets a PCIe-speed performance cliff instead of an error, and
  an out-of-memory test gets no failure at all. Linux does not do this.
- **A failed allocation costs time in proportion to its size** on this driver,
  while the fallback tries to page for it: 1.1 s for 8 GB, 7.9 s for 256 GB,
  and under a millisecond for anything at or above 1 TB.
- **CUDA errors are recorded, and read once.** Any failed runtime call also sets
  the thread's last error, and the next `cudaGetLastError()` returns it - even
  from an unrelated kernel launch. A wrapper that throws on a failed call without
  reading that record makes the next kernel check blame a healthy kernel.
- Bad block dimensions (4096 threads) report `cudaErrorInvalidValue` on CUDA
  13.4, not the `cudaErrorInvalidConfiguration` most documentation describes.
- `compute-sanitizer` 2026.3 could not attach to any process unelevated on this
  machine - not even a 20-line program. `scripts/sanitize.ps1` detects that and
  stops; `scripts/sanitize.sh` is for Linux and Colab.
- `nsys --trace osrt` is Linux-only and rejects the whole invocation on Windows.
- Windows PowerShell turns *any* native-tool stderr into a terminating error
  under `$ErrorActionPreference = "Stop"` — a benign nsys warning aborts the
  script.

## Profiling

```powershell
cd cuda-portfolio
.\scripts\profile.ps1            # nsys timelines + ncu metrics
.\scripts\profile.ps1 -SkipNcu   # timelines only, no elevation needed
```

Nsight Systems timelines for the benchmarks are committed under
`cuda-portfolio/profiles/`.

**On Nsight Compute:** the documented fix for `ERR_NVGPUCTRPERM` is to set
`RmProfilingAdminOnly = 0` under
`HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak` and reboot.
Verified on this machine: **that does not work** on driver 616.92 — the key was
set correctly as a DWORD, the machine rebooted, and `ncu` still failed
unelevated while succeeding elevated. The profiling script therefore runs `ncu`
elevated in a single batch (one UAC prompt for all targets).
