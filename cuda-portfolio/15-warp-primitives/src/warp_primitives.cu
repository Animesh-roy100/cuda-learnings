// Warp- and block-level primitives -- implementation.

#include "warp_primitives.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cu/check.hpp"

namespace kp {
namespace {

constexpr unsigned FULL = 0xffffffffu;

// ---------------------------------------------------------------------------
// 1. Shuffles
// ---------------------------------------------------------------------------
__global__ void k_broadcast(const float* __restrict__ in, float* __restrict__ out,
                            int n, int src_lane) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __shfl_sync(FULL, in[i], src_lane);
}

// Inclusive scan inside a warp. Hillis-Steele: at step d, lane L adds the value
// from lane L-d. Lanes with L < d must NOT add -- they have no predecessor at
// that distance -- which is what the guard is for.
__global__ void k_scan_up(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int lane = threadIdx.x & 31;

    float v = in[i];
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const float up = __shfl_up_sync(FULL, v, d);
        if (lane >= d) v += up;
    }
    out[i] = v;
}

// Reduction tree. Only lane 0 holds the total afterwards, which is the
// difference from the xor version below.
__global__ void k_reduce_down(const float* __restrict__ in, float* __restrict__ out,
                              int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(FULL, v, d);
    out[i] = v;
}

// Butterfly reduction: lane L exchanges with L^mask at every step, so all 32
// lanes finish holding the total. Costs the same shuffles as the tree; the only
// difference is who ends up with the answer.
__global__ void k_reduce_xor(const float* __restrict__ in, float* __restrict__ out,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(FULL, v, m);
    out[i] = v;
}

// ---------------------------------------------------------------------------
// 2. Votes
// ---------------------------------------------------------------------------
__global__ void k_vote(const int* __restrict__ pred, unsigned* __restrict__ ballot,
                       int* __restrict__ all_t, int* __restrict__ any_t,
                       int* __restrict__ popc, unsigned* __restrict__ active) {
    const int lane = threadIdx.x & 31;
    const int p = pred[lane];

    // EVERY vote must be evaluated with the whole warp converged, before any
    // divergence. Calling __all_sync or __activemask inside `if (lane == 0)`
    // polls only the lanes still active there -- which is lane 0 alone. The
    // first version of this kernel did exactly that and reported all_true =
    // true for a predicate that was false on 21 lanes, plus an activemask of
    // 0x00000001. The intrinsics were working; the question being asked was
    // wrong.
    const unsigned b = __ballot_sync(FULL, p);
    const int all_v = __all_sync(FULL, p);
    const int any_v = __any_sync(FULL, p);
    const unsigned act = __activemask();
    const int pc = __popc(b);

    if (lane == 0) {
        *ballot = b;
        *all_t = all_v;
        *any_t = any_v;
        *popc = pc;
        *active = act;
    }
}

// ---------------------------------------------------------------------------
// 3. Block barrier variants
// ---------------------------------------------------------------------------
__global__ void k_block_sync(const int* __restrict__ pred, int n,
                             int* __restrict__ count, int* __restrict__ all_nz,
                             int* __restrict__ any_nz) {
    const int t = threadIdx.x;
    const int p = (t < n) ? pred[t] : 0;

    // Each of these is a full block barrier AND a reduction, in one
    // instruction. Doing it by hand costs a shared array plus two syncs.
    const int c = __syncthreads_count(p);
    const int a = __syncthreads_and((t < n) ? p : 1);   // neutral for padding
    const int o = __syncthreads_or(p);

    if (t == 0) {
        *count = c;
        *all_nz = a;
        *any_nz = o;
    }
}

// ---------------------------------------------------------------------------
// 4. Fences
//
// Block 0 writes a payload then raises a flag. Block 1 spins on the flag and
// reads the payload. Without a fence between the two writes, the compiler and
// the memory system are both free to make the flag visible first.
// ---------------------------------------------------------------------------
// The ACK is not optional. Without it the producer races on to round it+1 and
// overwrites the payload while the consumer is still reading round it -- which
// reports "torn" reads that have nothing to do with the fence being tested.
// Ping-ponging on flag and ack makes the only variable under test the ordering
// of the payload writes against the flag write.
//
// Note this kernel REQUIRES both blocks to be resident simultaneously. A
// spin-wait on another block deadlocks if that block has not been scheduled,
// which is why inter-block handshakes are fragile in general and why grid-wide
// synchronisation properly belongs to cooperative groups.
__global__ void k_producer_consumer(volatile int* __restrict__ payload,
                                    volatile int* __restrict__ flag,
                                    volatile int* __restrict__ ack,
                                    int* __restrict__ torn, int iterations,
                                    int use_fence) {
    const int tid = threadIdx.x;

    for (int it = 0; it < iterations; ++it) {
        if (blockIdx.x == 0) {
            if (tid < 32) payload[tid] = it + 1;
            __syncthreads();
            if (tid == 0) {
                // Orders the payload writes BEFORE the flag write as far as
                // every other block is concerned. Remove it and the flag may
                // become visible first, letting the consumer read stale data.
                if (use_fence) __threadfence();
                *flag = it + 1;
                while (*ack != it + 1) { /* wait for the consumer */ }
            }
            __syncthreads();
        } else if (blockIdx.x == 1) {
            if (tid == 0) {
                while (*flag != it + 1) { /* spin */ }
                if (use_fence) __threadfence();
                int bad = 0;
                for (int i = 0; i < 32; ++i)
                    if (payload[i] != it + 1) ++bad;
                if (bad) atomicAdd(torn, 1);
                if (use_fence) __threadfence();
                *ack = it + 1;
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Atomics
// ---------------------------------------------------------------------------
__global__ void k_atomics(const int* __restrict__ v, int n, int* __restrict__ out,
                          unsigned* __restrict__ uout, int* __restrict__ cas_wins) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int x = v[i];

    atomicAdd(&out[0], x);
    atomicSub(&out[1], x);
    atomicExch(&out[2], x);          // last writer wins; value is nondeterministic
    atomicMin(&out[3], x);
    atomicMax(&out[4], x);
    atomicAnd(&out[5], x);
    atomicOr(&out[6], x);
    atomicXor(&out[7], x);

    // Wrapping counters. atomicInc wraps to 0 above the limit rather than
    // overflowing, which is what distinguishes it from atomicAdd(1).
    atomicInc(&uout[0], 0xffffffffu);
    atomicDec(&uout[1], 0xffffffffu);

    // Exactly one thread should win this race.
    if (atomicCAS(&out[8], 0, i + 1) == 0) atomicAdd(cas_wins, 1);
}

// ---------------------------------------------------------------------------
// 6. Bit intrinsics
// ---------------------------------------------------------------------------
__global__ void k_bits(const unsigned* __restrict__ v, int n,
                       int* __restrict__ popc, int* __restrict__ clz,
                       int* __restrict__ ffs, unsigned* __restrict__ brev,
                       unsigned* __restrict__ bperm, unsigned* __restrict__ funnel) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned x = v[i];

    popc[i] = __popc(x);
    clz[i] = __clz(x);
    ffs[i] = __ffs(x);
    brev[i] = __brev(x);
    // 0x0123 selects bytes 3,2,1,0 of the {lo,hi} pair -- a byte-order reverse.
    bperm[i] = __byte_perm(x, 0u, 0x0123);
    funnel[i] = __funnelshift_r(x, x, 8);   // rotate right by 8
}

// ---------------------------------------------------------------------------
// 7. Packed dot products
// ---------------------------------------------------------------------------
__global__ void k_dp4a(const int* __restrict__ a, const int* __restrict__ b,
                       int* __restrict__ out, int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    out[i] = __dp4a(a[i], b[i], 0);
}

__global__ void k_dp2a(const int* __restrict__ a, const int* __restrict__ b,
                       int* __restrict__ out, int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    // __dp2a_lo takes the low 16-bit halves of each operand pair.
    out[i] = __dp2a_lo(a[i], b[i], 0);
}

// ---------------------------------------------------------------------------
// 8. FMA
// ---------------------------------------------------------------------------
__global__ void k_fma(float a, float b, float c, float* __restrict__ out) {
    if (threadIdx.x != 0) return;
    out[0] = __fmaf_rn(a, b, c);          // one rounding
    // volatile stops the compiler from contracting this back into an FMA,
    // which is exactly what it would otherwise do.
    volatile float prod = a * b;          // rounding one
    out[1] = prod + c;                    // rounding two
}

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    CU_CHECK(cudaMalloc(&d, std::max<std::size_t>(h.size(), 1) * sizeof(T)));
    if (!h.empty())
        CU_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

}  // namespace

// ---------------------------------------------------------------------------
std::vector<float> warp_broadcast(const std::vector<float>& in, int src_lane) {
    if (in.empty() || in.size() % 32 != 0)
        throw std::invalid_argument("input size must be a non-zero multiple of 32");
    if (src_lane < 0 || src_lane > 31)
        throw std::invalid_argument("src_lane must be in [0,31]");

    const int n = (int)in.size();
    float* d_in = upload(in);
    float* d_out = nullptr;
    CU_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    k_broadcast<<<(n + 255) / 256, 256>>>(d_in, d_out, n, src_lane);
    CU_CHECK_KERNEL();

    std::vector<float> out(n);
    CU_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

// The three warp ops share everything except the kernel name. That name cannot
// be hoisted into a parameter: <<<>>> requires it at the call site, and
// launching through a function pointer does not compile. A macro keeps the
// duplication to one line each.
#define KP_WARP_OP(fn, kernel)                                                        std::vector<float> fn(const std::vector<float>& in) {                                 if (in.empty() || in.size() % 32 != 0)                                                throw std::invalid_argument(                                                          "input size must be a non-zero multiple of 32");                          const int n = (int)in.size();                                                     float* d_in = upload(in);                                                         float* d_out = nullptr;                                                           CU_CHECK(cudaMalloc(&d_out, n * sizeof(float)));                                  kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n);                                 CU_CHECK_KERNEL();                                                                std::vector<float> out(n);                                                        CU_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float),                                             cudaMemcpyDeviceToHost));                                     cudaFree(d_in);                                                                   cudaFree(d_out);                                                                  return out;                                                                   }

KP_WARP_OP(warp_inclusive_scan, k_scan_up)
KP_WARP_OP(warp_reduce_down, k_reduce_down)
KP_WARP_OP(warp_reduce_xor, k_reduce_xor)
#undef KP_WARP_OP

// ---------------------------------------------------------------------------
VoteResult warp_vote(const std::vector<int>& predicate) {
    if (predicate.size() != 32)
        throw std::invalid_argument("predicate must have exactly 32 entries");

    int* d_p = upload(predicate);
    unsigned *d_ballot = nullptr, *d_active = nullptr;
    int *d_all = nullptr, *d_any = nullptr, *d_popc = nullptr;
    CU_CHECK(cudaMalloc(&d_ballot, sizeof(unsigned)));
    CU_CHECK(cudaMalloc(&d_active, sizeof(unsigned)));
    CU_CHECK(cudaMalloc(&d_all, sizeof(int)));
    CU_CHECK(cudaMalloc(&d_any, sizeof(int)));
    CU_CHECK(cudaMalloc(&d_popc, sizeof(int)));

    k_vote<<<1, 32>>>(d_p, d_ballot, d_all, d_any, d_popc, d_active);
    CU_CHECK_KERNEL();

    VoteResult r;
    unsigned b = 0, a = 0;
    int all_t = 0, any_t = 0, pc = 0;
    CU_CHECK(cudaMemcpy(&b, d_ballot, sizeof(b), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(&a, d_active, sizeof(a), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(&all_t, d_all, sizeof(all_t), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(&any_t, d_any, sizeof(any_t), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(&pc, d_popc, sizeof(pc), cudaMemcpyDeviceToHost));
    r.ballot = b;
    r.activemask = a;
    r.all_true = all_t != 0;
    r.any_true = any_t != 0;
    r.popcount = pc;

    cudaFree(d_p); cudaFree(d_ballot); cudaFree(d_active);
    cudaFree(d_all); cudaFree(d_any); cudaFree(d_popc);
    return r;
}

// ---------------------------------------------------------------------------
BlockSyncResult block_sync_variants(const std::vector<int>& predicate) {
    const int n = (int)predicate.size();
    if (n <= 0 || n > 1024)
        throw std::invalid_argument("predicate size must be in [1,1024]");

    int* d_p = upload(predicate);
    int *d_c = nullptr, *d_a = nullptr, *d_o = nullptr;
    CU_CHECK(cudaMalloc(&d_c, sizeof(int)));
    CU_CHECK(cudaMalloc(&d_a, sizeof(int)));
    CU_CHECK(cudaMalloc(&d_o, sizeof(int)));

    // Round the block up to a warp multiple; padding threads contribute a
    // neutral predicate.
    const int block = ((n + 31) / 32) * 32;
    k_block_sync<<<1, block>>>(d_p, n, d_c, d_a, d_o);
    CU_CHECK_KERNEL();

    BlockSyncResult r;
    int c = 0, a = 0, o = 0;
    CU_CHECK(cudaMemcpy(&c, d_c, sizeof(c), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(&a, d_a, sizeof(a), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(&o, d_o, sizeof(o), cudaMemcpyDeviceToHost));
    r.count = c;
    r.all_nonzero = a != 0;
    r.any_nonzero = o != 0;

    cudaFree(d_p); cudaFree(d_c); cudaFree(d_a); cudaFree(d_o);
    return r;
}

// ---------------------------------------------------------------------------
FenceResult producer_consumer(int iterations, bool with_fence) {
    if (iterations <= 0) throw std::invalid_argument("iterations must be positive");

    int *d_payload = nullptr, *d_flag = nullptr, *d_ack = nullptr, *d_torn = nullptr;
    CU_CHECK(cudaMalloc(&d_payload, 32 * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CU_CHECK(cudaMalloc(&d_ack, sizeof(int)));
    CU_CHECK(cudaMalloc(&d_torn, sizeof(int)));
    CU_CHECK(cudaMemset(d_payload, 0, 32 * sizeof(int)));
    CU_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
    CU_CHECK(cudaMemset(d_ack, 0, sizeof(int)));
    CU_CHECK(cudaMemset(d_torn, 0, sizeof(int)));

    // Exactly two blocks: a producer and a consumer, both resident at once.
    k_producer_consumer<<<2, 32>>>(d_payload, d_flag, d_ack, d_torn, iterations,
                                   with_fence ? 1 : 0);
    CU_CHECK_KERNEL();

    FenceResult r;
    r.iterations = iterations;
    r.used_fence = with_fence;
    CU_CHECK(cudaMemcpy(&r.torn_reads, d_torn, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_payload); cudaFree(d_flag); cudaFree(d_ack); cudaFree(d_torn);
    return r;
}

// ---------------------------------------------------------------------------
AtomicResults exercise_atomics(const std::vector<int>& values) {
    if (values.empty()) throw std::invalid_argument("values must not be empty");
    const int n = (int)values.size();

    int* d_v = upload(values);
    int* d_out = nullptr;
    unsigned* d_uout = nullptr;
    int* d_cas = nullptr;
    CU_CHECK(cudaMalloc(&d_out, 9 * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_uout, 2 * sizeof(unsigned)));
    CU_CHECK(cudaMalloc(&d_cas, sizeof(int)));

    // Identity elements, so each operation's result is attributable to the data
    // rather than to the seed.
    const int init[9] = {0, 0, 0, INT32_MAX, INT32_MIN, ~0, 0, 0, 0};
    const unsigned uinit[2] = {0u, 0u};
    CU_CHECK(cudaMemcpy(d_out, init, sizeof(init), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_uout, uinit, sizeof(uinit), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(d_cas, 0, sizeof(int)));

    k_atomics<<<(n + 255) / 256, 256>>>(d_v, n, d_out, d_uout, d_cas);
    CU_CHECK_KERNEL();

    int h[9] = {};
    unsigned hu[2] = {};
    AtomicResults r;
    CU_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(hu, d_uout, sizeof(hu), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(&r.cas_winner, d_cas, sizeof(int), cudaMemcpyDeviceToHost));

    r.add = h[0]; r.sub = h[1]; r.exch = h[2]; r.min = h[3]; r.max = h[4];
    r.and_ = h[5]; r.or_ = h[6]; r.xor_ = h[7];
    r.inc = hu[0]; r.dec = hu[1];

    cudaFree(d_v); cudaFree(d_out); cudaFree(d_uout); cudaFree(d_cas);
    return r;
}

// ---------------------------------------------------------------------------
BitResults exercise_bit_intrinsics(const std::vector<std::uint32_t>& values) {
    if (values.empty()) throw std::invalid_argument("values must not be empty");
    const int n = (int)values.size();

    unsigned* d_v = upload(values);
    int *d_popc = nullptr, *d_clz = nullptr, *d_ffs = nullptr;
    unsigned *d_brev = nullptr, *d_bperm = nullptr, *d_fun = nullptr;
    CU_CHECK(cudaMalloc(&d_popc, n * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_clz, n * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_ffs, n * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_brev, n * sizeof(unsigned)));
    CU_CHECK(cudaMalloc(&d_bperm, n * sizeof(unsigned)));
    CU_CHECK(cudaMalloc(&d_fun, n * sizeof(unsigned)));

    k_bits<<<(n + 255) / 256, 256>>>(d_v, n, d_popc, d_clz, d_ffs, d_brev,
                                     d_bperm, d_fun);
    CU_CHECK_KERNEL();

    BitResults r;
    r.popc.resize(n); r.clz.resize(n); r.ffs.resize(n);
    r.brev.resize(n); r.byte_perm.resize(n); r.funnel.resize(n);
    CU_CHECK(cudaMemcpy(r.popc.data(), d_popc, n * sizeof(int), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(r.clz.data(), d_clz, n * sizeof(int), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(r.ffs.data(), d_ffs, n * sizeof(int), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(r.brev.data(), d_brev, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(r.byte_perm.data(), d_bperm, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(r.funnel.data(), d_fun, n * sizeof(unsigned), cudaMemcpyDeviceToHost));

    cudaFree(d_v); cudaFree(d_popc); cudaFree(d_clz); cudaFree(d_ffs);
    cudaFree(d_brev); cudaFree(d_bperm); cudaFree(d_fun);
    return r;
}

// ---------------------------------------------------------------------------
std::vector<int> dp4a_dot(const std::vector<std::int8_t>& a,
                          const std::vector<std::int8_t>& b) {
    if (a.size() != b.size() || a.empty() || a.size() % 4 != 0)
        throw std::invalid_argument("inputs must match and be a non-zero multiple of 4");
    const int groups = (int)(a.size() / 4);

    // Four int8 values are packed into one 32-bit word; that packing IS the
    // instruction's calling convention.
    std::vector<int> pa(groups), pb(groups);
    for (int g = 0; g < groups; ++g) {
        pa[g] = (a[4 * g] & 0xff) | ((a[4 * g + 1] & 0xff) << 8) |
                ((a[4 * g + 2] & 0xff) << 16) | ((a[4 * g + 3] & 0xff) << 24);
        pb[g] = (b[4 * g] & 0xff) | ((b[4 * g + 1] & 0xff) << 8) |
                ((b[4 * g + 2] & 0xff) << 16) | ((b[4 * g + 3] & 0xff) << 24);
    }

    int* d_a = upload(pa);
    int* d_b = upload(pb);
    int* d_o = nullptr;
    CU_CHECK(cudaMalloc(&d_o, groups * sizeof(int)));

    k_dp4a<<<(groups + 255) / 256, 256>>>(d_a, d_b, d_o, groups);
    CU_CHECK_KERNEL();

    std::vector<int> out(groups);
    CU_CHECK(cudaMemcpy(out.data(), d_o, groups * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_o);
    return out;
}

std::vector<int> dp2a_dot(const std::vector<std::int16_t>& a,
                          const std::vector<std::int8_t>& b) {
    if (a.size() != b.size() || a.empty() || a.size() % 2 != 0)
        throw std::invalid_argument("inputs must match and be a non-zero multiple of 2");
    const int groups = (int)(a.size() / 2);

    std::vector<int> pa(groups), pb(groups);
    for (int g = 0; g < groups; ++g) {
        // Two int16 in the first operand...
        pa[g] = (a[2 * g] & 0xffff) | ((a[2 * g + 1] & 0xffff) << 16);
        // ...against two INT8 in the LOW half of the second. _lo selects that
        // half; __dp2a_hi would take bytes 2 and 3 instead.
        pb[g] = (b[2 * g] & 0xff) | ((b[2 * g + 1] & 0xff) << 8);
    }

    int* d_a = upload(pa);
    int* d_b = upload(pb);
    int* d_o = nullptr;
    CU_CHECK(cudaMalloc(&d_o, groups * sizeof(int)));

    k_dp2a<<<(groups + 255) / 256, 256>>>(d_a, d_b, d_o, groups);
    CU_CHECK_KERNEL();

    std::vector<int> out(groups);
    CU_CHECK(cudaMemcpy(out.data(), d_o, groups * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_o);
    return out;
}

// ---------------------------------------------------------------------------
FmaComparison fma_vs_separate(float a, float b, float c) {
    float* d_out = nullptr;
    CU_CHECK(cudaMalloc(&d_out, 2 * sizeof(float)));
    k_fma<<<1, 32>>>(a, b, c, d_out);
    CU_CHECK_KERNEL();

    float h[2] = {};
    CU_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    FmaComparison r;
    r.fused = h[0];
    r.separate = h[1];
    r.exact = (double)a * (double)b + (double)c;
    r.fused_error = std::abs((double)r.fused - r.exact);
    r.separate_error = std::abs((double)r.separate - r.exact);
    return r;
}

}  // namespace kp
