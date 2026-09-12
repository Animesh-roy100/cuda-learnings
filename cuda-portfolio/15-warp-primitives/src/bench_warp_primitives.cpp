// Warp and block primitives, with their results printed.

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cu/device.hpp"
#include "warp_primitives.h"

namespace {

std::vector<float> ramp(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = float(i % 32) + 1.0f;
    return v;
}

void bits32(unsigned v, char* out) {
    for (int i = 31; i >= 0; --i) out[31 - i] = (v & (1u << i)) ? '1' : '0';
    out[32] = '\0';
}

}  // namespace

int main() try {
    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("Every primitive below is checked against a host reference in the\n");
    std::printf("test suite; this prints what they actually produce.\n");

    // ------------------------------------------------------------- shuffles
    std::printf("\n=== 1. Warp shuffles (lane values 1..32) ===\n");
    auto in = ramp(32);
    auto bc = kp::warp_broadcast(in, 7);
    auto sc = kp::warp_inclusive_scan(in);
    auto rd = kp::warp_reduce_down(in);
    auto rx = kp::warp_reduce_xor(in);

    std::printf("  __shfl_sync       broadcast lane 7 to all lanes\n");
    std::printf("    lanes 0,1,31 -> %.0f %.0f %.0f   (all equal in[7] = %.0f)\n",
                bc[0], bc[1], bc[31], in[7]);
    std::printf("  __shfl_up_sync    inclusive prefix scan\n");
    std::printf("    lanes 0,1,2,31 -> %.0f %.0f %.0f %.0f   (last = 32*33/2)\n",
                sc[0], sc[1], sc[2], sc[31]);
    std::printf("  __shfl_down_sync  reduction tree\n");
    std::printf("    lane 0 -> %.0f, lane 31 -> %.0f   (only lane 0 holds the total)\n",
                rd[0], rd[31]);
    std::printf("  __shfl_xor_sync   butterfly reduction\n");
    std::printf("    lane 0 -> %.0f, lane 31 -> %.0f   (EVERY lane holds the total)\n",
                rx[0], rx[31]);

    // ---------------------------------------------------------------- votes
    std::printf("\n=== 2. Warp vote intrinsics ===\n");
    std::vector<int> pred(32, 0);
    for (int i = 0; i < 32; i += 3) pred[i] = 1;
    auto v = kp::warp_vote(pred);
    char buf[33];
    bits32(v.ballot, buf);
    std::printf("  predicate: every third lane true\n");
    std::printf("  __ballot_sync  0x%08x  %s\n", v.ballot, buf);
    std::printf("  __popc         %d lanes true\n", v.popcount);
    std::printf("  __all_sync     %s\n", v.all_true ? "true" : "false");
    std::printf("  __any_sync     %s\n", v.any_true ? "true" : "false");
    std::printf("  __activemask   0x%08x  (no divergence here, so all 32)\n",
                v.activemask);

    // --------------------------------------------------------- block syncs
    std::printf("\n=== 3. Block barrier variants ===\n");
    std::vector<int> bp(256, 0);
    for (int i = 0; i < 256; i += 4) bp[i] = 1;
    auto b = kp::block_sync_variants(bp);
    std::printf("  256 threads, every fourth true\n");
    std::printf("  __syncthreads_count  %d\n", b.count);
    std::printf("  __syncthreads_and    %s\n", b.all_nonzero ? "true" : "false");
    std::printf("  __syncthreads_or     %s\n", b.any_nonzero ? "true" : "false");
    std::printf("  Each is a barrier AND a block-wide reduction in one instruction;\n");
    std::printf("  by hand it costs a shared array plus two syncs.\n");

    // --------------------------------------------------------------- fences
    std::printf("\n=== 4. Memory fences: producer / consumer across blocks ===\n");
    auto with_f = kp::producer_consumer(5000, true);
    auto without = kp::producer_consumer(5000, false);
    std::printf("  with    __threadfence : %d torn reads in %d rounds\n",
                with_f.torn_reads, with_f.iterations);
    std::printf("  without __threadfence : %d torn reads in %d rounds\n",
                without.torn_reads, without.iterations);
    std::printf("\n  A fence is NOT a barrier: it orders one thread's writes as seen\n");
    std::printf("  by others without making anyone wait. The unfenced run is unsafe\n");
    std::printf("  BY CONSTRUCTION even when it reports zero -- absence of a visible\n");
    std::printf("  failure is not evidence of correctness in a memory model, which is\n");
    std::printf("  why the test suite asserts only on the fenced path.\n");

    // -------------------------------------------------------------- atomics
    std::printf("\n=== 5. Atomic instruction set (values 1..1000) ===\n");
    std::vector<int> av(1000);
    std::iota(av.begin(), av.end(), 1);
    auto a = kp::exercise_atomics(av);
    std::printf("  atomicAdd  %9d     atomicSub  %10d\n", a.add, a.sub);
    std::printf("  atomicMin  %9d     atomicMax  %10d\n", a.min, a.max);
    std::printf("  atomicAnd  %9d     atomicOr   %10d\n", a.and_, a.or_);
    std::printf("  atomicXor  %9d     atomicExch %10d  (nondeterministic)\n",
                a.xor_, a.exch);
    std::printf("  atomicInc  %9u     atomicDec  %10u  (both WRAP, not saturate)\n",
                a.inc, a.dec);
    std::printf("  atomicCAS  %d winner out of %zu racing threads\n",
                a.cas_winner, av.size());

    // ------------------------------------------------------------- bit ops
    std::printf("\n=== 6. Bit intrinsics ===\n");
    std::vector<unsigned> bv = {0x12345678u, 0x80000000u, 0x00000001u, 0u};
    auto bits = kp::exercise_bit_intrinsics(bv);
    std::printf("  %-12s %6s %5s %5s %12s %12s\n",
                "value", "popc", "clz", "ffs", "brev", "byte_perm");
    for (std::size_t i = 0; i < bv.size(); ++i)
        std::printf("  0x%08x %6d %5d %5d   0x%08x   0x%08x\n",
                    bv[i], bits.popc[i], bits.clz[i], bits.ffs[i],
                    bits.brev[i], bits.byte_perm[i]);
    std::printf("  Each is ONE instruction where portable C++ needs a loop.\n");

    // --------------------------------------------------------- packed dots
    std::printf("\n=== 7. Packed SIMD dot products ===\n");
    auto d4 = kp::dp4a_dot({1, 2, 3, 4}, {10, 20, 30, 40});
    auto d2 = kp::dp2a_dot(std::vector<std::int16_t>{100, 200}, std::vector<std::int8_t>{7, 8});
    std::printf("  __dp4a([1,2,3,4],[10,20,30,40]) = %d   (4 int8 products, 1 cycle)\n",
                d4[0]);
    std::printf("  __dp2a([100,200] i16, [7,8] i8) = %d   (MIXED precision)\n", d2[0]);
    std::printf("\n  __dp4a is the whole reason INT8 inference is fast on a card with\n");
    std::printf("  no tensor cores -- see 01-gguf-inference.\n");
    std::printf("\n  __dp2a is easy to misuse: it is int16 against INT8, not int16\n");
    std::printf("  squared. Passing two int16 operands compiles, runs, and silently\n");
    std::printf("  reads only the low byte of the second -- giving 100*7 + 200*0 = 700\n");
    std::printf("  where 2300 was intended. No error, just a wrong answer.\n");

    // ----------------------------------------------------------------- FMA
    std::printf("\n=== 8. Fused multiply-add: one rounding vs two ===\n");
    std::printf("  %-24s %14s %14s %14s\n", "a*b+c", "fused", "separate", "exact");
    struct Case { float a, b, c; const char* label; };
    const Case cases[] = {
        {1.0f + 1e-7f, 1.0f - 1e-7f, -1.0f, "(1+e)(1-e) - 1"},
        {16777217.0f, 1.0f, -16777216.0f, "2^24+1 - 2^24"},
        {3.0f, 1.0f / 3.0f, -1.0f, "3 * (1/3) - 1"},
    };
    for (const auto& k : cases) {
        auto r = kp::fma_vs_separate(k.a, k.b, k.c);
        std::printf("  %-24s %14.3e %14.3e %14.3e\n",
                    k.label, r.fused, r.separate, r.exact);
    }
    std::printf("\n  __fmaf_rn keeps the full intermediate product and rounds ONCE.\n");
    std::printf("  Writing a*b+c as two operations rounds twice, and the low bits c\n");
    std::printf("  was meant to cancel are gone before the add ever happens.\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
