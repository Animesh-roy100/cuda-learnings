# Running this on Google Colab's free T4

Everything here targets **Turing `sm_75`**, and Colab's free tier hands out a
**Tesla T4 — also `sm_75`**. Same architecture, same instruction set, same
`__dp4a`, same warp primitives. Nothing needs recompiling for a different
target and no `#ifdef` changes hardware behaviour.

So the code runs unmodified. What changes is the *machine around it*: Colab is
Linux, ships CUDA 12 rather than 13, and the T4 is a considerably better GPU
than the laptop card these numbers came from.

## The T4 is not the same card

| | GTX 1650 (dev machine) | Tesla T4 (Colab) |
|---|---|---|
| Architecture | Turing `sm_75` | Turing `sm_75` — **identical** |
| SMs / CUDA cores | 14 / 896 | **40 / 2560** |
| VRAM | 4 GB GDDR6 | **16 GB GDDR6** |
| Memory bandwidth | 192 GB/s | **320 GB/s** |
| Tensor Cores | none | **320** (unused by this code) |
| FP32 | ~2.8 TFLOP/s | ~8.1 TFLOP/s |
| FP64 | 1/32 rate | 1/32 rate — same penalty |
| Board power | 75 W | 70 W, passively cooled |

**Expect roughly 1.5–2.5× better numbers**, not identical ones. Memory-bound
kernels scale with the 1.67× bandwidth; compute-bound ones (Monte Carlo,
SHA-256) scale closer to the 2.9× core count. The T4 is also passively cooled
in a datacentre and clocks lower under sustained load, so gains are rarely the
full ratio.

Two consequences worth knowing:

- **The 4 GB ceiling disappears.** Several projects are deliberately sized to
  fit 4 GB. On a T4 you can raise them — see *Turning it up* below.
- **The SpMV ELLPACK skip may not trigger.** That benchmark refuses to build a
  3.8 GB ELLPACK matrix on a 4 GB card. With 16 GB it fits, so you will see the
  measurement the laptop cannot produce. That is a genuine bonus.

## Step 1 — get a GPU runtime

`Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save`

Then verify. If this prints nothing, you are on a CPU runtime and nothing below
will work:

```python
!nvidia-smi
```

Look for `Tesla T4` and note the CUDA version in the top-right corner.

## Step 2 — clone and check the toolchain

```python
!git clone https://github.com/Animesh-roy100/cuda-learnings.git
%cd cuda-learnings
!nvcc --version
!cmake --version
```

Colab usually ships CUDA 12.x and a CMake older than the 3.24 this project
requires. Install a current CMake and Ninja if needed — this takes about 30
seconds:

```python
!pip install -q --upgrade cmake ninja
!cmake --version
```

## Step 3 — the fundamentals (fastest way to see something work)

```python
%cd /content/cuda-learnings/cuda-learning
!chmod +x build.sh
!./build.sh 01-device-query/device_query.cu && ./bin/device_query
```

This prints the T4's real geometry — 40 SMs, 16 GB, and its peak bandwidth.
**Write that bandwidth number down**; it is the ceiling every memory-bound
result below should be judged against, and it is not the one in the README.

```python
!./build.sh 04-matmul/matmul.cu && ./bin/matmul
```

## Step 4 — the standalone projects

```python
%cd /content/cuda-learnings/cuda-projects
!chmod +x build.sh
for f in ["01-quant-llm/quant_gemv.cu", "02-hash-table/hash_table.cu",
          "03-spatial-dbscan/spatial.cu", "05-monte-carlo/monte_carlo.cu"]:
    name = f.split("/")[-1].replace(".cu", "")
    !./build.sh {f} && ./bin/{name}
```

`04-zero-copy` spawns a child process to test CUDA IPC. It works on Linux, but
Colab's sandbox may restrict process spawning — if it fails, that is the
environment, not the code.

## Step 5 — the full portfolio (13 projects, 169 tests)

```python
%cd /content/cuda-learnings/cuda-portfolio
!cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
!cmake --build build -j
```

The first configure downloads GoogleTest, so it needs network — Colab has it.
A full build takes **5–10 minutes** on Colab's CPU allocation.

```python
%cd build
!ctest --output-on-failure
%cd ..
```

Expect `100% tests passed out of 14`. Every test is written against an
independent reference (CPU models, NIST vectors, closed-form Black-Scholes,
brute-force k-NN, Dijkstra), so passing on different hardware is meaningful
rather than tautological.

Then run whichever benchmarks interest you:

```python
!./build/bin/bench_inference   # INT4 __dp4a GEMV, LLM decode projection
!./build/bin/bench_hash_kv     # lock-free hash table, Robin Hood tail latency
!./build/bin/bench_mc          # Monte Carlo, FP32 vs FP64 penalty
!./build/bin/bench_spmv        # four sparse formats compared
!./build/bin/bench_ann         # IVF-Flat recall/latency curve
!./build/bin/bench_sha256      # SHA-256 hashrate + register pressure
```

All thirteen, if you want the full sweep (about 5 minutes):

```python
import subprocess, glob, os
for exe in sorted(glob.glob("build/bin/bench_*")):
    if exe.endswith(".exe"):        # ignore Windows artifacts if any
        continue
    print("=" * 70, "\n", os.path.basename(exe), "\n", "=" * 70)
    print(subprocess.run([exe], capture_output=True, text=True).stdout)
```

`bench_ann` spends about a minute on host-side k-means before it measures
anything. That is expected, not a hang.

## What does *not* work on Colab

| Thing | Why | What to do |
|---|---|---|
| `build.bat` | Windows batch | use `build.sh` |
| `scripts/profile.ps1` | PowerShell | see below |
| `scripts/verify_all.ps1` | PowerShell | run `ctest` and the benchmarks directly |
| **Nsight Compute (`ncu`)** | needs GPU performance counter access, which hosted VMs generally withhold | expect `ERR_NVGPUCTRPERM`; try it, but do not count on it |
| **Nsight Systems (`nsys`)** | usually present | often works; try it |

> I have not verified the profiling rows on Colab myself — everything else in
> this guide follows from the code, but those two depend on how Google
> configures its tenant VMs and may differ from what you find.

Nsight Systems is worth trying, since timelines are the more useful artifact
anyway:

```python
!which nsys || apt-get install -y -q nsight-systems 2>/dev/null | tail -1
!nsys profile --trace cuda --output /content/trace ./build/bin/bench_mc
```

Download `/content/trace.nsys-rep` and open it in Nsight Systems locally.

`ncu` is the likely loss. Roofline and occupancy analysis needs performance
counters that hosted VMs typically withhold, and the same restriction exists on
consumer cards locally (it needs a registry change on Windows, or elevation).
If you need that data reliably, it has to come from a machine you control.

## Turning it up — using the 16 GB

Several benchmarks are sized for a 4 GB card. On a T4 you can push them, which
is the most interesting thing you can do with the extra memory. The values
below are in the benchmark sources; edit and rebuild.

| File | Constant | Laptop | Try on T4 |
|---|---|---|---|
| `09-vector-ann/src/bench_ann.cpp` | `N` (database vectors) | 200000 | 600000 |
| `09-vector-ann/src/bench_ann.cpp` | `L` (IVF lists) | 256 | 1024 |
| `13-spmv/src/bench_spmv.cpp` | power-law `n` | 300000 | 600000 |
| `04-spatial/src/bench_spatial.cpp` | `N` (points) | 2000000 | 8000000 |
| `12-crypto-hash/src/bench_sha256.cpp` | PoW `bits` | 24 | 28 |

Raising `L` in the ANN benchmark also increases host-side k-means time, which
is O(sample × nlist × dim) on one core. Raise the list count and the training
sample together only if you are willing to wait.

## Colab-specific annoyances

- **Sessions die.** Free runtimes disconnect after roughly 90 minutes idle or
  12 hours total, and everything under `/content` is erased. Re-clone and
  rebuild; nothing here stores state between runs.
- **Benchmarks are noisy.** The T4 is shared infrastructure with variable
  thermal headroom. Run each benchmark two or three times and take the best;
  every benchmark in this repo already warms up and takes a minimum internally,
  but that cannot compensate for a noisy neighbour on the host.
- **Build times dominate.** Compilation happens on Colab's modest CPU
  allocation, so the 5–10 minute build will exceed the total runtime of every
  benchmark combined. Build once, then iterate on running.
- **Keep the build out of Drive.** Mounting Drive and building into it is
  extremely slow. Build under `/content`.

## Reading results against the README

The README numbers were measured on the GTX 1650 and are not the target on a
T4. The things that should hold on *both* cards are the conclusions, not the
timings:

- INT4 GEMV beating FP32 by roughly 3.4–5.1× (the ratio is bandwidth-bound, so
  it travels between cards)
- Robin Hood collapsing worst-case displacement while leaving the mean untouched
- k-NN via a spatial grid matching brute force **bit for bit**
- FP64 running at about 1/32 of FP32 — a Turing hardware property, identical on
  a T4
- SHA-256 sitting near 70% of integer peak with zero register spill

If any of those *conclusions* change on a T4, that is genuinely interesting and
worth investigating. If the raw milliseconds differ, that is just a faster card.

## One-cell version

Paste this into a single Colab cell to go from nothing to a passing test suite:

```python
!git clone -q https://github.com/Animesh-roy100/cuda-learnings.git
%cd /content/cuda-learnings
!pip install -q --upgrade cmake ninja
!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

%cd cuda-portfolio
!cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null
!cmake --build build -j 2>&1 | tail -3
%cd build
!ctest 2>&1 | tail -4
%cd ..
!./build/bin/bench_mc
```
