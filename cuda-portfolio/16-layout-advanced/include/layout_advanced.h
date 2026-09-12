#pragma once
//
// Memory layout and the advanced execution subsystems. No CUDA syntax here.
//
// Two groups:
//   * layout   -- AoS vs SoA, and the shared-memory padding trick. Both are
//                 pure data-arrangement changes: identical arithmetic, identical
//                 results, different speed.
//   * advanced -- cooperative groups, grid-wide sync, dynamic parallelism,
//                 Tensor Core WMMA, and asynchronous shared-memory copy.
//
// Several of the advanced features have hardware or driver preconditions, so
// each reports whether it was actually available rather than assuming.
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace la {

// ---------------------------------------------------------------------------
// 1. Array of Structures vs Structure of Arrays
//
// AoS: {x,y,z}{x,y,z}... A thread reading every x touches every third float,
// so a warp spans 3x the bytes it uses and the hardware fetches lines that are
// two-thirds discarded.
// SoA: xxx...yyy...zzz. Consecutive threads read consecutive floats and the
// warp's request collapses into the minimum number of transactions.
// ---------------------------------------------------------------------------
struct LayoutResult {
    float aos_ms = 0.0f;
    float soa_ms = 0.0f;
    double aos_gbps = 0.0;
    double soa_gbps = 0.0;
    bool results_match = false;
    double speedup() const { return soa_ms > 0 ? aos_ms / soa_ms : 0.0; }
};
LayoutResult compare_aos_soa(int particles);

// ---------------------------------------------------------------------------
// 2. Shared-memory padding.
//
// A [32][32] float tile makes column j of every row land in the same bank, so
// a column-wise access serialises 32 ways. Widening to [32][33] shifts each row
// by one bank and the conflict disappears -- at a cost of 128 bytes per tile.
// ---------------------------------------------------------------------------
struct PaddingResult {
    float unpadded_ms = 0.0f;   // [32][32]
    float padded_ms = 0.0f;     // [32][33]
    bool results_match = false;
    std::size_t extra_bytes = 0;
    double speedup() const { return padded_ms > 0 ? unpadded_ms / padded_ms : 0.0; }
};
PaddingResult compare_shared_padding(int iterations);

// ---------------------------------------------------------------------------
// 3. Cooperative groups
// ---------------------------------------------------------------------------
struct CoopGroupsResult {
    bool cooperative_launch_supported = false;
    std::string skip_reason;

    float tiled_reduce = 0.0f;      // cg::reduce over a 32-lane tile
    float block_reduce = 0.0f;      // cg::reduce over the whole block
    int grid_size_seen = 0;         // grid.size() from inside the kernel
    bool grid_sync_ok = false;      // a grid-wide barrier actually held
};
CoopGroupsResult exercise_cooperative_groups(int elements);

// ---------------------------------------------------------------------------
// 4. Dynamic parallelism: a kernel launching a kernel, no host involved.
// ---------------------------------------------------------------------------
struct DynamicParallelismResult {
    std::vector<float> output;
    int child_launches = 0;
    bool supported = false;
    std::string skip_reason;
};
DynamicParallelismResult exercise_dynamic_parallelism(int parents, int child_threads);

// ---------------------------------------------------------------------------
// 5. Tensor Cores via the WMMA API.
//
// Worth measuring rather than assuming on this hardware. The GTX 16-series is
// compute capability 7.5, so the WMMA instructions are part of its ISA, but
// NVIDIA lists these chips as shipping without Tensor Cores -- which predicts
// "correct but not faster".
//
// A two-way comparison could not settle it, because switching to WMMA also
// halves the bytes read and either effect could explain a speedup. So there are
// three kernels, identical except for one variable each:
//
//   fp32_ms       fp32 operands, fp32 math in CUDA cores  -- the baseline
//   fp16in_ms     fp16 operands, fp32 math in CUDA cores  -- isolates bandwidth
//   wmma_ms       fp16 operands, mma_sync                 -- adds the instruction
//
// fp32 -> fp16in is the bandwidth effect alone; fp16in -> wmma is what the MMA
// instruction itself is worth.
// ---------------------------------------------------------------------------
struct WmmaResult {
    bool ran = false;
    std::string skip_reason;
    int matrix_dim = 0;
    float wmma_ms = 0.0f;
    float fp32_ms = 0.0f;
    float fp16in_ms = 0.0f;
    double max_abs_error = 0.0;   // wmma (fp16 inputs) vs fp32 reference
    double speedup() const { return wmma_ms > 0 ? fp32_ms / wmma_ms : 0.0; }
    // How much of that came from narrower operands, and how much from mma_sync.
    double bandwidth_speedup() const { return fp16in_ms > 0 ? fp32_ms / fp16in_ms : 0.0; }
    double instruction_speedup() const { return wmma_ms > 0 ? fp16in_ms / wmma_ms : 0.0; }
};
WmmaResult compare_wmma_vs_fp32(int matrix_dim);

// ---------------------------------------------------------------------------
// 6. Asynchronous global -> shared copy.
//
// cooperative_groups::memcpy_async works from sm_70, but the hardware
// instruction that makes it genuinely asynchronous (cp.async) arrived with
// Ampere, sm_80. On Turing it compiles and is correct while falling back to an
// ordinary copy, so no speedup should be expected here -- and that is the
// result worth recording.
// ---------------------------------------------------------------------------
struct AsyncCopyResult {
    bool hardware_accelerated = false;   // true only on sm_80+
    int compute_capability = 0;
    float sync_ms = 0.0f;
    float async_ms = 0.0f;
    bool results_match = false;
    double speedup() const { return async_ms > 0 ? sync_ms / async_ms : 0.0; }
};
AsyncCopyResult compare_async_copy(int elements);

}  // namespace la
