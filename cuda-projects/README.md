# Five CUDA projects on a GTX 1650

All five build and run on this machine. Every number below was measured here,
not estimated.

**Hardware:** GTX 1650 (Turing, `sm_75`), 14 SMs, 4 GB VRAM, 192 GB/s peak
memory bandwidth, PCIe gen3 x16, no Tensor Cores, FP64 at 1/32 rate.

**Toolchain:** CUDA 13.4, MSVC 19.44, driver 616.92.

## Build

```bat
cd C:\Users\legion\cuda-projects
build.bat 05-monte-carlo\monte_carlo.cu
bin\monte_carlo.exe
```

`build.bat` sets up MSVC, locates the toolkit, and compiles with
`-arch=sm_75 -O3 -std=c++17`. Run it from this directory.

---

## 1. Quantized LLM inference (`01-quant-llm`)

INT4/INT8 GEMV using `__dp4a`, the 4-way INT8 dot-product instruction. No
Tensor Cores needed — `__dp4a` has existed since `sm_61`.

Layout is llama.cpp Q4_0 style: 32 weights share one FP32 scale, packed as
nibbles. 32 weights arrive in a single 16-byte load and become 8 `__dp4a`
instructions.

Measured at Llama 3.2 1B shapes:

| Shape | fp32 | int8 | int4 |
|---|---|---|---|
| q_proj 2048x2048 | 0.094 ms | 0.029 ms (3.2x) | 0.021 ms (**4.5x**) |
| gate/up 8192x2048 | 0.365 ms | 0.106 ms (3.5x) | 0.068 ms (**5.3x**) |
| down 2048x8192 | 0.367 ms | 0.112 ms (3.3x) | 0.064 ms (**5.7x**) |

Correctness: the INT4 kernel is checked against a CPU model of the exact same
integer math — **L2 relative error 1.7e-07**, i.e. float32 epsilon.

Decode projection for Llama 3.2 1B (1.24 B params):

| | size | fits 4 GB? | tok/s (realistic) |
|---|---|---|---|
| fp32 | 4.94 GB | no | 29 |
| fp16 | 2.47 GB | yes | 58 |
| int8 | 1.24 GB | yes | 117 |
| int4 | 0.70 GB | yes | **207** |

**The lesson:** batch-1 decode is memory bound, so speed is literally
`weight bytes / bandwidth`. Quantization is not an approximation trick, it is
the primary performance lever.

**Worth knowing:** INT4 gets 4.5–5.7x, not the 8x its byte count implies. Below
INT8 the kernel stops being purely bandwidth bound — nibble unpacking costs
real ALU work. Effective bandwidth drops from ~150 GB/s (int8) to ~120 GB/s
(int4) even as total time improves.

## 2. Lock-free hash table (`02-hash-table`)

Open addressing, linear probing, one 64-bit `atomicCAS` per mutation. Key and
value are packed into a single 64-bit word so they land atomically together —
with separate arrays, a reader could see a valid key beside an uninitialised
value.

16.7M slots, measured:

| Load factor | Insert | Find (thread) | Avg probes |
|---|---|---|---|
| 0.50 | 312 M ops/s | 782 M ops/s | 1.50 |
| 0.75 | 309 M ops/s | 687 M ops/s | 2.49 |
| 0.90 | 265 M ops/s | 500 M ops/s | 5.50 |
| 0.95 | 208 M ops/s | 345 M ops/s | 10.51 |

**The honest finding:** warp-cooperative lookup (32 lanes testing 32 slots,
collapsed with `__ballot_sync`) **loses here** — 0.52x at load 0.5, reaching
parity only at 0.95. It burns a full 256-byte transaction per query, so with
an average chain of 1.5 slots it does ~20x the memory work. Per-thread probing
lets 32 lanes resolve 32 *different* queries from one transaction.

The primitives are excellent on this card. Applying them to short probe chains
is not. Warp-cooperative probing pays off only when chains are long enough to
serialise a scalar probe — the regime a well-sized table exists to avoid.

**Bug worth remembering:** the warp version originally capped probing at 1024
slots. That silently returned "not found" for present keys at load factor
≥0.90, where clustering builds chains far longer than that. Terminating on an
empty slot is the only correct bound.

## 3. Spatial indexing, k-NN, DBSCAN (`03-spatial-dbscan`)

Morton (Z-order) codes → `thrust::sort_by_key` → cell ranges → 27-cell
neighbour queries. 2M points in 3D.

| Stage | Time |
|---|---|
| morton + sort + grid | 19.4 ms |
| neighbour count | 68.0 ms |
| union cores (lock-free union-find) | 156–178 ms |
| attach borders | 0.8 ms |
| **total DBSCAN** | **~250 ms (7.5 M points/s)** |

k-NN (k=8, 4096 queries against 2M points):

| | Time |
|---|---|
| brute force | 253.2 ms |
| grid | 4.9 ms |
| **speedup** | **51.4x, results identical** |

DBSCAN recovered 10.2% noise against 10% planted. Cluster count comes back as
16 large clusters from 20 planted blobs — **correct behaviour**: blob centres
are random, some land within `eps` and DBSCAN merges them. A cluster of ~180k
is two merged 90k blobs.

**Where to optimise next:** union-cores is ~65% of runtime. It re-scans the 27
cells *and* pays atomic contention inside dense blobs.

## 4. Zero-copy pipeline + CUDA IPC (`04-zero-copy`)

1080p RGBA → luma → Sobel, with three separate lessons.

**Host memory type vs PCIe:**

| | Bandwidth |
|---|---|
| pageable | 4.75 GB/s |
| pinned | **11.98 GB/s (2.5x)** |

PCIe gen3 x16 tops out near 15.8 GB/s, so pinned is close to the wire.

**Zero-copy (mapped) memory**, summing 2 MB:

| | Time | Effective |
|---|---|---|
| device mem, kernel only | 0.036 ms | 57 GB/s (VRAM) |
| device mem, upload + kernel | 0.272 ms | PCIe-dominated |
| zero-copy, no upload | 0.182 ms | 11.4 GB/s (PCIe) |

Zero-copy beats upload+compute by 1.5x for touch-once data, but is **5x slower
than resident VRAM**. Touch the data twice and you pay PCIe twice.

**Stream overlap** (24 frames, 4 streams, 2 copy engines):
serial 27.7 ms → overlapped 19.5 ms = **1.42x**, output bit-identical.
Sobel verified against a CPU reference: 0 of 2,073,600 pixels differ.

**CUDA IPC: works cross-process on Windows.** The parent exports a device
pointer, a separate child process opens it, writes, and the parent sees the
writes. This is often documented as Linux-only — it worked here.

**NVDEC/NVENC caveat:** this card has the hardware engines (nvidia-smi reports
encoder/decoder counters), but `nvcuvid.h` and `nvEncodeAPI.h` ship in NVIDIA's
separate **Video Codec SDK**, not the CUDA Toolkit. Everything above is the
part that matters for CUDA — NVDEC hands you a `CUdeviceptr`, which is exactly
what these sections move without copying. Download the SDK to wire up the codec
front end.

## 5. Monte Carlo option pricing (`05-monte-carlo`)

The one genuinely **compute-bound** project here. Everything else streams
memory; this turns a handful of parameters into billions of paths.

European call, S=100, K=105, r=5%, σ=20%, T=1. Black-Scholes exact: **8.021352**

| | Price | Std err | Throughput |
|---|---|---|---|
| fp32, 100M paths | 8.021233 | ±0.0013 | 16,159 M paths/s |
| fp64, 1M paths | 8.011871 | ±0.0132 | 528 M paths/s |
| fp32 antithetic | 8.020445 | ±0.0010 | 42,173 M paths/s |

**FP32 is 30.6x the FP64 path rate** — Turing consumer silicon runs double
precision at 1/32 rate, measured. Reaching for `double` by reflex is the most
expensive single mistake available on this card.

The fp32 estimate sits **0.09 standard errors** from exact.

Greeks by pathwise differentiation (unbiased, nearly free — no second
simulation, no subtracting two nearly-equal numbers):

| | MC | Exact | Error |
|---|---|---|---|
| delta | 0.542160 | 0.542228 | -6.8e-05 |
| vega | 39.6667 | 39.6705 | -3.9e-03 |

Asian (arithmetic average, 252 daily fixings, no closed form): 3.517 ± 0.005,
correctly cheaper than the European call since averaging suppresses terminal
variance.

---

## Cross-cutting lessons

1. **Know whether you are memory or compute bound before optimising.** Projects
   1–4 are memory bound; only project 5 is compute bound. The right move is
   opposite in each case.
2. **Measure, do not assume.** Warp-cooperative hashing was supposed to win and
   lost. INT4 was supposed to be 8x and delivered 4.5–5.7x.
3. **Warm up before timing.** An un-warmed first launch made zero-copy look
   faster than VRAM — physically impossible.
4. **Verify against something independent.** CPU reference models, closed-form
   Black-Scholes, brute-force k-NN. Every project here checks itself.
5. **Compare with the right metric.** Per-element relative error falsely flagged
   a correct INT4 kernel; under cancellation only an L2 norm is meaningful.

## CUDA 13 gotchas hit along the way

- `memoryClockRate` / `clockRate` removed from `cudaDeviceProp` — use
  `cudaDeviceGetAttribute`.
- `thrust` and `cub` moved to `include/cccl/` — needs an extra `-I`.
- Thrust requires `-std=c++17` **and** `-Xcompiler /Zc:preprocessor` on MSVC.
- A non-integral `static const float` at namespace scope is not usable in
  device code — use `constexpr`.
