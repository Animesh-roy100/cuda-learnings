#pragma once
//
// Warp- and block-level primitives, each exercised and checked against a host
// reference. No CUDA syntax in this header.
//
// These are the instructions the other projects reach for when they need them.
// Here each one is isolated with a test that would fail if the semantics were
// misunderstood -- which is the only way to know they are understood.
//
#include <cstdint>
#include <vector>

namespace kp {

// ---------------------------------------------------------------------------
// 1. Warp shuffles: register-to-register exchange, no shared memory involved.
//
// All four take a mask of participating lanes. The _sync suffix is not
// decoration: since Volta, lanes can diverge independently, and the mask is how
// the hardware knows which lanes must arrive before the exchange happens.
// ---------------------------------------------------------------------------

// __shfl_sync: every lane reads src_lane's value.
std::vector<float> warp_broadcast(const std::vector<float>& in, int src_lane);

// __shfl_up_sync: lane L reads lane L-delta. Building block of a prefix scan,
// which is what this returns -- an inclusive scan within each warp.
std::vector<float> warp_inclusive_scan(const std::vector<float>& in);

// __shfl_down_sync: lane L reads lane L+delta. The classic reduction tree;
// lane 0 ends up with the warp total.
std::vector<float> warp_reduce_down(const std::vector<float>& in);

// __shfl_xor_sync: lane L exchanges with lane L^mask. A butterfly, so EVERY
// lane ends up with the total rather than just lane 0.
std::vector<float> warp_reduce_xor(const std::vector<float>& in);

// ---------------------------------------------------------------------------
// 2. Warp vote and mask intrinsics
// ---------------------------------------------------------------------------
struct VoteResult {
    std::uint32_t ballot = 0;      // __ballot_sync: one bit per lane
    bool all_true = false;         // __all_sync
    bool any_true = false;         // __any_sync
    int popcount = 0;              // __popc of the ballot
    std::uint32_t activemask = 0;  // __activemask
};
// predicate must hold exactly 32 entries, one per lane.
VoteResult warp_vote(const std::vector<int>& predicate);

// ---------------------------------------------------------------------------
// 3. Block-wide barrier variants. Each is a __syncthreads() that also reduces
// a predicate across the block, which saves a separate shared-memory pass.
// ---------------------------------------------------------------------------
struct BlockSyncResult {
    int count = 0;          // __syncthreads_count: how many threads were true
    bool all_nonzero = false;  // __syncthreads_and
    bool any_nonzero = false;  // __syncthreads_or
};
BlockSyncResult block_sync_variants(const std::vector<int>& predicate);

// ---------------------------------------------------------------------------
// 4. Memory fences.
//
// A fence is NOT a barrier: it orders one thread's memory operations as seen by
// others, without making anyone wait. The canonical use is a producer writing
// data then a flag -- without a fence between them, another block may observe
// the flag before the data, and read garbage that was never written.
// ---------------------------------------------------------------------------
struct FenceResult {
    int iterations = 0;
    int torn_reads = 0;     // consumers that saw the flag but not the payload
    bool used_fence = false;
};
// with_fence = false runs the same code without __threadfence(), which is
// UNSAFE by construction and exists to show what the fence is preventing.
FenceResult producer_consumer(int iterations, bool with_fence);

// ---------------------------------------------------------------------------
// 5. The full atomic instruction set, each applied to the same input so the
// results can be checked against an obvious host computation.
// ---------------------------------------------------------------------------
struct AtomicResults {
    int add = 0, sub = 0, exch = 0, min = 0, max = 0;
    unsigned inc = 0, dec = 0;
    int and_ = 0, or_ = 0, xor_ = 0;
    int cas_winner = 0;      // how many threads won an atomicCAS race
};
AtomicResults exercise_atomics(const std::vector<int>& values);

// ---------------------------------------------------------------------------
// 6. Bit-manipulation intrinsics. Each maps to a single SASS instruction where
// the portable C++ equivalent is a loop.
// ---------------------------------------------------------------------------
struct BitResults {
    std::vector<int> popc;            // __popc
    std::vector<int> clz;             // __clz
    std::vector<int> ffs;             // __ffs
    std::vector<std::uint32_t> brev;  // __brev
    std::vector<std::uint32_t> byte_perm;   // __byte_perm, reversing byte order
    std::vector<std::uint32_t> funnel;      // __funnelshift_r
};
BitResults exercise_bit_intrinsics(const std::vector<std::uint32_t>& values);

// ---------------------------------------------------------------------------
// 7. Packed SIMD dot products. One instruction, four (or two) products plus an
// accumulate -- the basis of every quantized inference kernel.
// ---------------------------------------------------------------------------
// Each group of 4 int8 values in a and b becomes one __dp4a.
std::vector<int> dp4a_dot(const std::vector<std::int8_t>& a,
                          const std::vector<std::int8_t>& b);

// __dp2a is MIXED precision, which is easy to get wrong: two int16 values from
// one operand against two INT8 values from the other, accumulating into int32.
// It is not an int16 x int16 dot product. Used correctly, as the signature here
// forces, [100,200] against [7,8] returns 2300. Pass two int16 operands instead
// and it compiles, runs, and silently reads only the low byte of the second --
// returning 100*7 + 200*0 = 700 with no error of any kind. The signature is
// typed std::int8_t precisely so that mistake cannot be made through this API.
std::vector<int> dp2a_dot(const std::vector<std::int16_t>& a,
                          const std::vector<std::int8_t>& b);

// ---------------------------------------------------------------------------
// 8. Fused multiply-add.
//
// __fmaf_rn(a,b,c) computes a*b+c with ONE rounding step. Writing a*b+c as two
// operations rounds twice, and the difference is not academic -- it is why a
// carefully written numerical kernel and a naive one disagree.
// ---------------------------------------------------------------------------
struct FmaComparison {
    float fused = 0.0f;     // __fmaf_rn
    float separate = 0.0f;  // (a*b) then +c, two roundings
    double exact = 0.0;     // computed in double on the host
    double fused_error = 0.0;
    double separate_error = 0.0;
};
FmaComparison fma_vs_separate(float a, float b, float c);

}  // namespace kp
