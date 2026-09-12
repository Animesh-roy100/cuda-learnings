#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "warp_primitives.h"

namespace {

std::vector<float> ramp(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = float(i % 32) + 1.0f;   // 1..32 per warp
    return v;
}

// Portable host equivalents. Comparing an intrinsic against std:: or a loop is
// the only way to show the intrinsic was understood rather than just invoked.
int host_popc(std::uint32_t x) {
    int c = 0;
    while (x) { c += x & 1u; x >>= 1; }
    return c;
}
int host_clz(std::uint32_t x) {
    if (x == 0) return 32;
    int c = 0;
    while (!(x & 0x80000000u)) { ++c; x <<= 1; }
    return c;
}
int host_ffs(std::uint32_t x) {
    if (x == 0) return 0;
    int c = 1;
    while (!(x & 1u)) { ++c; x >>= 1; }
    return c;
}
std::uint32_t host_brev(std::uint32_t x) {
    std::uint32_t r = 0;
    for (int i = 0; i < 32; ++i) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// Shuffles
// ---------------------------------------------------------------------------
TEST(Shuffle, BroadcastGivesEveryLaneTheSourceValue) {
    auto in = ramp(128);                       // four warps of 1..32
    for (int src : {0, 7, 31}) {
        auto out = kp::warp_broadcast(in, src);
        ASSERT_EQ(out.size(), in.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            const std::size_t warp_base = (i / 32) * 32;
            EXPECT_FLOAT_EQ(out[i], in[warp_base + src])
                << "lane " << i << " src " << src;
        }
    }
}

TEST(Shuffle, BroadcastRejectsBadLane) {
    auto in = ramp(32);
    EXPECT_THROW(kp::warp_broadcast(in, -1), std::invalid_argument);
    EXPECT_THROW(kp::warp_broadcast(in, 32), std::invalid_argument);
    EXPECT_THROW(kp::warp_broadcast({1.0f, 2.0f}, 0), std::invalid_argument);
}

TEST(Shuffle, ShflUpBuildsAnInclusiveScan) {
    auto in = ramp(64);                        // two warps of 1..32
    auto out = kp::warp_inclusive_scan(in);
    ASSERT_EQ(out.size(), in.size());

    for (int w = 0; w < 2; ++w) {
        float running = 0.0f;
        for (int lane = 0; lane < 32; ++lane) {
            running += in[w * 32 + lane];
            EXPECT_FLOAT_EQ(out[w * 32 + lane], running)
                << "warp " << w << " lane " << lane;
        }
    }
    // Last lane of a 1..32 warp holds 32*33/2.
    EXPECT_FLOAT_EQ(out[31], 528.0f);
}

TEST(Shuffle, ShflDownReducesIntoLaneZero) {
    auto in = ramp(64);
    auto out = kp::warp_reduce_down(in);
    // Only lane 0 is guaranteed to hold the total; that IS the distinction
    // from the xor butterfly below.
    EXPECT_FLOAT_EQ(out[0], 528.0f);
    EXPECT_FLOAT_EQ(out[32], 528.0f);
}

TEST(Shuffle, ShflXorGivesEveryLaneTheTotal) {
    auto in = ramp(64);
    auto out = kp::warp_reduce_xor(in);
    for (int i = 0; i < 64; ++i)
        EXPECT_FLOAT_EQ(out[i], 528.0f) << "lane " << i << " missing the total";
}

// ---------------------------------------------------------------------------
// Votes
// ---------------------------------------------------------------------------
TEST(Vote, BallotEncodesOneBitPerLane) {
    std::vector<int> pred(32, 0);
    pred[0] = 1;
    pred[5] = 1;
    pred[31] = 1;

    auto r = kp::warp_vote(pred);
    EXPECT_EQ(r.ballot, (1u << 0) | (1u << 5) | (1u << 31));
    EXPECT_EQ(r.popcount, 3);
    EXPECT_FALSE(r.all_true);
    EXPECT_TRUE(r.any_true);
    EXPECT_EQ(r.activemask, 0xffffffffu) << "no divergence, so all lanes active";
}

TEST(Vote, AllAndAnyBehaveAtTheExtremes) {
    auto none = kp::warp_vote(std::vector<int>(32, 0));
    EXPECT_EQ(none.ballot, 0u);
    EXPECT_FALSE(none.all_true);
    EXPECT_FALSE(none.any_true);
    EXPECT_EQ(none.popcount, 0);

    auto all = kp::warp_vote(std::vector<int>(32, 1));
    EXPECT_EQ(all.ballot, 0xffffffffu);
    EXPECT_TRUE(all.all_true);
    EXPECT_TRUE(all.any_true);
    EXPECT_EQ(all.popcount, 32);
}

TEST(Vote, RejectsWrongWidth) {
    EXPECT_THROW(kp::warp_vote(std::vector<int>(16, 1)), std::invalid_argument);
    EXPECT_THROW(kp::warp_vote(std::vector<int>(64, 1)), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Block barrier variants
// ---------------------------------------------------------------------------
TEST(BlockSync, CountAndOrMatchTheObviousHostAnswer) {
    std::vector<int> pred(256, 0);
    for (int i = 0; i < 256; i += 4) pred[i] = 1;      // 64 true

    auto r = kp::block_sync_variants(pred);
    EXPECT_EQ(r.count, 64);
    EXPECT_FALSE(r.all_nonzero);
    EXPECT_TRUE(r.any_nonzero);
}

TEST(BlockSync, AllNonzeroWhenEveryThreadIsTrue) {
    auto r = kp::block_sync_variants(std::vector<int>(128, 7));
    EXPECT_EQ(r.count, 128);
    EXPECT_TRUE(r.all_nonzero);
    EXPECT_TRUE(r.any_nonzero);
}

TEST(BlockSync, NothingTrueIsReportedCorrectly) {
    auto r = kp::block_sync_variants(std::vector<int>(64, 0));
    EXPECT_EQ(r.count, 0);
    EXPECT_FALSE(r.all_nonzero);
    EXPECT_FALSE(r.any_nonzero);
}

TEST(BlockSync, RejectsBadSize) {
    EXPECT_THROW(kp::block_sync_variants({}), std::invalid_argument);
    EXPECT_THROW(kp::block_sync_variants(std::vector<int>(2048, 1)),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Fences
// ---------------------------------------------------------------------------
// With a fence, the consumer must NEVER observe the flag ahead of the payload.
// This is the assertion that matters.
TEST(Fence, WithFenceThereAreNoTornReads) {
    auto r = kp::producer_consumer(2000, true);
    EXPECT_TRUE(r.used_fence);
    EXPECT_EQ(r.torn_reads, 0) << "a fence must prevent flag-before-data reordering";
}

// Deliberately NOT asserting that the unfenced version fails. Whether the
// reordering is observable depends on timing, the scheduler and the compiler,
// so an assertion either way would be flaky. The unfenced path exists to be
// run and reported, not to be relied upon.
TEST(Fence, WithoutFenceStillCompletes) {
    auto r = kp::producer_consumer(500, false);
    EXPECT_FALSE(r.used_fence);
    EXPECT_GE(r.torn_reads, 0);
    EXPECT_EQ(r.iterations, 500);
}

TEST(Fence, RejectsBadIterations) {
    EXPECT_THROW(kp::producer_consumer(0, true), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Atomics
// ---------------------------------------------------------------------------
TEST(Atomics, EveryOperationMatchesTheHostComputation) {
    std::vector<int> v;
    for (int i = 1; i <= 1000; ++i) v.push_back(i);

    auto r = kp::exercise_atomics(v);

    const int sum = std::accumulate(v.begin(), v.end(), 0);
    EXPECT_EQ(r.add, sum);
    EXPECT_EQ(r.sub, -sum);
    EXPECT_EQ(r.min, *std::min_element(v.begin(), v.end()));
    EXPECT_EQ(r.max, *std::max_element(v.begin(), v.end()));

    int a = ~0, o = 0, x = 0;
    for (int e : v) { a &= e; o |= e; x ^= e; }
    EXPECT_EQ(r.and_, a);
    EXPECT_EQ(r.or_, o);
    EXPECT_EQ(r.xor_, x);

    EXPECT_EQ(r.inc, (unsigned)v.size());
    // atomicDec on a counter starting at 0 wraps to the limit, so it does NOT
    // simply mirror atomicInc. That asymmetry is the point of having both.
    EXPECT_NE(r.dec, 0u);

    // exchange leaves SOME input value; which one is genuinely nondeterministic.
    EXPECT_TRUE(std::find(v.begin(), v.end(), r.exch) != v.end());

    EXPECT_EQ(r.cas_winner, 1) << "exactly one thread must win the CAS race";
}

TEST(Atomics, RejectsEmptyInput) {
    EXPECT_THROW(kp::exercise_atomics({}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Bit intrinsics
// ---------------------------------------------------------------------------
TEST(BitIntrinsics, MatchPortableHostImplementations) {
    std::vector<std::uint32_t> v = {0u, 1u, 2u, 0x80000000u, 0xffffffffu,
                                    0x0000ff00u, 0x12345678u, 0xdeadbeefu};
    std::mt19937 rng(7);
    for (int i = 0; i < 64; ++i) v.push_back(rng());

    auto r = kp::exercise_bit_intrinsics(v);
    ASSERT_EQ(r.popc.size(), v.size());

    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(r.popc[i], host_popc(v[i])) << "popc at " << i;
        EXPECT_EQ(r.clz[i], host_clz(v[i])) << "clz at " << i;
        EXPECT_EQ(r.ffs[i], host_ffs(v[i])) << "ffs at " << i;
        EXPECT_EQ(r.brev[i], host_brev(v[i])) << "brev at " << i;

        // __byte_perm with selector 0x0123 reverses byte order.
        const std::uint32_t x = v[i];
        const std::uint32_t swapped = ((x & 0xffu) << 24) | ((x & 0xff00u) << 8) |
                                      ((x >> 8) & 0xff00u) | ((x >> 24) & 0xffu);
        EXPECT_EQ(r.byte_perm[i], swapped) << "byte_perm at " << i;

        // __funnelshift_r(x, x, 8) is a rotate right by 8.
        const std::uint32_t rot = (x >> 8) | (x << 24);
        EXPECT_EQ(r.funnel[i], rot) << "funnelshift at " << i;
    }
}

TEST(BitIntrinsics, HandleZeroCorrectly) {
    auto r = kp::exercise_bit_intrinsics({0u});
    EXPECT_EQ(r.popc[0], 0);
    EXPECT_EQ(r.clz[0], 32) << "clz(0) is 32 by definition";
    EXPECT_EQ(r.ffs[0], 0) << "ffs(0) is 0 -- there is no set bit";
}

TEST(BitIntrinsics, RejectsEmptyInput) {
    EXPECT_THROW(kp::exercise_bit_intrinsics({}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Packed dot products
// ---------------------------------------------------------------------------
TEST(PackedDot, Dp4aMatchesScalarDotProduct) {
    std::vector<std::int8_t> a = {1, 2, 3, 4, -5, 6, -7, 8};
    std::vector<std::int8_t> b = {10, 20, 30, 40, 50, 60, 70, 80};

    auto out = kp::dp4a_dot(a, b);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 1 * 10 + 2 * 20 + 3 * 30 + 4 * 40);
    EXPECT_EQ(out[1], -5 * 50 + 6 * 60 + -7 * 70 + 8 * 80);
}

TEST(PackedDot, Dp4aHandlesTheSignedExtremes) {
    std::vector<std::int8_t> a = {127, -128, 127, -128};
    std::vector<std::int8_t> b = {127, 127, -128, -128};
    auto out = kp::dp4a_dot(a, b);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 127 * 127 + -128 * 127 + 127 * -128 + -128 * -128);
}

// __dp2a is MIXED precision: int16 against INT8. Writing this test assuming
// int16 x int16 produced 100*7 + 200*0 = 700 instead of 2300, because the
// second operand's upper byte was being read as the second int8 and it was
// zero. The instruction was right; the assumption about it was not.
TEST(PackedDot, Dp2aIsInt16TimesInt8NotInt16Squared) {
    std::vector<std::int16_t> a = {100, 200, -300, 400};
    std::vector<std::int8_t> b = {7, 8, 9, 10};
    auto out = kp::dp2a_dot(a, b);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 100 * 7 + 200 * 8);
    EXPECT_EQ(out[1], -300 * 9 + 400 * 10);
}

TEST(PackedDot, Dp2aHandlesTheFullInt16Range) {
    // Values beyond int8 range on the a side prove it really is 16-bit there.
    std::vector<std::int16_t> a = {32767, -32768};
    std::vector<std::int8_t> b = {2, 3};
    auto out = kp::dp2a_dot(a, b);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 32767 * 2 + -32768 * 3);
}

TEST(PackedDot, RejectMismatchedOrRaggedInput) {
    EXPECT_THROW(kp::dp4a_dot({1, 2, 3, 4}, {1, 2}), std::invalid_argument);
    EXPECT_THROW(kp::dp4a_dot({1, 2, 3}, {1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(kp::dp2a_dot({1}, {1}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// FMA
// ---------------------------------------------------------------------------
// The textbook case: a*b is exactly representable only to 24 bits, so doing the
// add separately throws away the low bits before c can cancel them.
TEST(Fma, FusedIsAtLeastAsAccurateAsSeparateOperations) {
    struct Case { float a, b, c; };
    const Case cases[] = {
        {1.0f + 1e-7f, 1.0f - 1e-7f, -1.0f},
        {3.0f, 1.0f / 3.0f, -1.0f},
        {1e20f, 1e-20f, -1.0f},
        {16777217.0f, 1.0f, -16777216.0f},
    };
    for (const auto& k : cases) {
        auto r = kp::fma_vs_separate(k.a, k.b, k.c);
        EXPECT_LE(r.fused_error, r.separate_error + 1e-30)
            << "a=" << k.a << " b=" << k.b << " c=" << k.c
            << " fused=" << r.fused << " separate=" << r.separate
            << " exact=" << r.exact;
    }
}

TEST(Fma, ProducesAStrictlyBetterAnswerOnACancellingCase) {
    // (1+eps)(1-eps) = 1 - eps^2. With a single rounding the eps^2 term
    // survives; rounding twice destroys it and the result comes out exactly 0.
    auto r = kp::fma_vs_separate(1.0f + 1e-7f, 1.0f - 1e-7f, -1.0f);
    EXPECT_LT(r.fused_error, r.separate_error)
        << "fused " << r.fused << " vs separate " << r.separate
        << " (exact " << r.exact << ")";
}
