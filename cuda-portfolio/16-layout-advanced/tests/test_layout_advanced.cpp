// Layout and advanced subsystems.
//
// The layout tests assert on CORRECTNESS, not on speed: a padded tile and an
// unpadded one must agree to the bit, and SoA must agree with AoS. Speed is
// reported by the benchmark, because a timing assertion turns a laptop under
// thermal load into a red test.
//
// The one exception is AoS vs SoA, where the effect is large enough (a factor
// of two in bytes moved) that a regression would mean the layout was silently
// changed back.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "cu/device.hpp"
#include "layout_advanced.h"

namespace {

int compute_capability() {
    auto d = cu::query_device();
    return d.arch();
}

}  // namespace

// ---------------------------------------------------------------- AoS / SoA
TEST(Layout, AosAndSoaAgreeExactly) {
    auto r = la::compare_aos_soa(1 << 20);
    // Same operations in the same order on the same values: not merely close,
    // identical. Any difference would mean one kernel is not doing what the
    // other does.
    EXPECT_TRUE(r.results_match);
}

TEST(Layout, SoaBeatsAosOnAStridedRead) {
    auto r = la::compare_aos_soa(1 << 22);
    ASSERT_GT(r.aos_ms, 0.0f);
    ASSERT_GT(r.soa_ms, 0.0f);
    // AoS moves 24 bytes per particle to use 12. The ideal is 2.0x; anything
    // above 1.3x confirms the effect is real and not noise, while leaving room
    // for caching and a busy laptop.
    EXPECT_GT(r.speedup(), 1.3) << "AoS " << r.aos_ms << " ms, SoA " << r.soa_ms << " ms";
}

TEST(Layout, SoaReachesAUsefulFractionOfPeak) {
    auto dev = cu::query_device();
    auto r = la::compare_aos_soa(1 << 22);
    const double frac = r.soa_gbps / dev.peak_bandwidth_gbps();
    EXPECT_GT(frac, 0.4) << r.soa_gbps << " GB/s of " << dev.peak_bandwidth_gbps();
    // Above peak would mean the accounting is wrong, which has happened twice
    // in this repo and both times the number was the only clue.
    EXPECT_LT(frac, 1.0) << "reported " << r.soa_gbps << " GB/s exceeds hardware peak";
}

// ----------------------------------------------------------- bank conflicts
TEST(SharedPadding, PaddedAndUnpaddedProduceIdenticalResults) {
    auto r = la::compare_shared_padding(32);
    // The padding changes only addresses, never values. If these differed, the
    // padded kernel would be indexing a different element.
    EXPECT_TRUE(r.results_match);
}

TEST(SharedPadding, PaddingRemovesTheSerialisation) {
    auto r = la::compare_shared_padding(64);
    ASSERT_GT(r.unpadded_ms, 0.0f);
    ASSERT_GT(r.padded_ms, 0.0f);
    // A 32-way conflict costs 32 cycles where one would do, but the kernel also
    // does other work, so the end-to-end effect is smaller than 32x. 1.5x is a
    // floor that noise cannot reach.
    EXPECT_GT(r.speedup(), 1.5) << "unpadded " << r.unpadded_ms
                                << " ms, padded " << r.padded_ms << " ms";
}

TEST(SharedPadding, CostsOneExtraColumn) {
    auto r = la::compare_shared_padding(8);
    EXPECT_EQ(r.extra_bytes, 32u * sizeof(float));
}

// ------------------------------------------------------- cooperative groups
TEST(CooperativeGroups, TileReduceSumsThirtyTwoLanes) {
    auto r = la::exercise_cooperative_groups(1 << 20);
    if (!r.cooperative_launch_supported) GTEST_SKIP() << r.skip_reason;
    EXPECT_FLOAT_EQ(r.tiled_reduce, 32.0f);
}

TEST(CooperativeGroups, BlockReduceSumsTheWholeBlock) {
    auto r = la::exercise_cooperative_groups(1 << 20);
    if (!r.cooperative_launch_supported) GTEST_SKIP() << r.skip_reason;
    // 256 threads, each contributing 1.0.
    EXPECT_FLOAT_EQ(r.block_reduce, 256.0f);
}

TEST(CooperativeGroups, GridSizeIsBlocksTimesThreads) {
    auto r = la::exercise_cooperative_groups(1 << 20);
    if (!r.cooperative_launch_supported) GTEST_SKIP() << r.skip_reason;
    EXPECT_GT(r.grid_size_seen, 0);
    EXPECT_EQ(r.grid_size_seen % 256, 0) << "grid.size() should be a multiple of blockDim";
}

TEST(CooperativeGroups, GridSyncHoldsAcrossBlocks) {
    const int n = 1 << 20;
    auto r = la::exercise_cooperative_groups(n);
    if (!r.cooperative_launch_supported) GTEST_SKIP() << r.skip_reason;
    // The second phase reads every block's partial. Without a working grid-wide
    // barrier it would read zeros from blocks that had not finished, and the
    // total would come up short -- so this assertion IS the barrier test.
    EXPECT_TRUE(r.grid_sync_ok);
}

TEST(CooperativeGroups, RepeatedCallsAreStable) {
    // Two launches in a row catch the shared-memory reuse hazard in
    // block_reduce_sum: it is called twice per kernel, and without the leading
    // barrier the second call would corrupt the first.
    for (int i = 0; i < 3; ++i) {
        auto r = la::exercise_cooperative_groups(1 << 18);
        if (!r.cooperative_launch_supported) GTEST_SKIP() << r.skip_reason;
        EXPECT_FLOAT_EQ(r.tiled_reduce, 32.0f) << "iteration " << i;
        EXPECT_FLOAT_EQ(r.block_reduce, 256.0f) << "iteration " << i;
        EXPECT_TRUE(r.grid_sync_ok) << "iteration " << i;
    }
}

// ------------------------------------------------------ dynamic parallelism
TEST(DynamicParallelism, EveryParentLaunchesOneChild) {
    auto r = la::exercise_dynamic_parallelism(8, 32);
    if (!r.supported) GTEST_SKIP() << r.skip_reason;
    EXPECT_EQ(r.child_launches, 8);
}

TEST(DynamicParallelism, ChildrenWroteTheirSlices) {
    const int parents = 8, child_threads = 32;
    auto r = la::exercise_dynamic_parallelism(parents, child_threads);
    if (!r.supported) GTEST_SKIP() << r.skip_reason;
    ASSERT_EQ(int(r.output.size()), parents * child_threads);
    for (int p = 0; p < parents; ++p)
        for (int i = 0; i < child_threads; ++i)
            EXPECT_FLOAT_EQ(r.output[p * child_threads + i], float(p + 1) * float(i))
                << "parent " << p << " element " << i;
}

TEST(DynamicParallelism, ChildrenCompleteBeforeTheHostSeesTheParentFinish) {
    // CDP2 removed the device-side cudaDeviceSynchronize(). What remains is the
    // guarantee this test depends on: the host's synchronize does not return
    // until the children are done. If it did, the tail of the buffer would
    // still hold the zeros it was memset to.
    auto r = la::exercise_dynamic_parallelism(16, 64);
    if (!r.supported) GTEST_SKIP() << r.skip_reason;
    ASSERT_FALSE(r.output.empty());
    const float last = r.output.back();     // parent 15, element 63 -> 16*63
    EXPECT_FLOAT_EQ(last, 16.0f * 63.0f);
}

// ------------------------------------------------------------ tensor cores
TEST(Wmma, RejectsANonMultipleOfSixteen) {
    auto r = la::compare_wmma_vs_fp32(100);
    EXPECT_FALSE(r.ran);
    EXPECT_FALSE(r.skip_reason.empty());
}

TEST(Wmma, MatchesTiledFp32WithinHalfPrecision) {
    if (compute_capability() < 70) GTEST_SKIP() << "WMMA needs sm_70+";
    auto r = la::compare_wmma_vs_fp32(256);
    ASSERT_TRUE(r.ran) << r.skip_reason;
    // fp16 carries 11 mantissa bits, so each product is good to ~5e-4 relative
    // and 256 of them accumulate. This bound is loose enough for that and tight
    // enough that a wrong fragment layout -- which produces garbage, not slight
    // error -- would fail it.
    EXPECT_LT(r.max_abs_error, 0.5) << "max |wmma - fp32| = " << r.max_abs_error;
    EXPECT_GT(r.max_abs_error, 0.0) << "identical to fp32 would mean the fp16 "
                                       "path never ran";
}

TEST(Wmma, ProducesTimingsForAllThreePaths) {
    if (compute_capability() < 70) GTEST_SKIP() << "WMMA needs sm_70+";
    auto r = la::compare_wmma_vs_fp32(512);
    ASSERT_TRUE(r.ran) << r.skip_reason;
    EXPECT_GT(r.wmma_ms, 0.0f);
    EXPECT_GT(r.fp32_ms, 0.0f);
    EXPECT_GT(r.fp16in_ms, 0.0f);
    // Deliberately NOT asserting any of these is faster than another. The
    // benchmark reports the split between operand width and instruction; which
    // way it falls is a property of whatever card the suite is run on, and
    // asserting a direction here would encode this laptop as a requirement.
}

TEST(Wmma, SpeedupFactorsIntoItsTwoCauses) {
    if (compute_capability() < 70) GTEST_SKIP() << "WMMA needs sm_70+";
    auto r = la::compare_wmma_vs_fp32(512);
    ASSERT_TRUE(r.ran) << r.skip_reason;
    // The three kernels form a chain: fp32 -> fp16-in -> wmma. The end-to-end
    // ratio must be the product of the two steps, or the attribution the
    // benchmark prints is not arithmetic anyone should trust.
    const double product = r.bandwidth_speedup() * r.instruction_speedup();
    EXPECT_NEAR(product, r.speedup(), r.speedup() * 0.01);
}

// -------------------------------------------------------------- async copy
TEST(AsyncCopy, SyncAndAsyncAgreeExactly) {
    auto r = la::compare_async_copy(1 << 20);
    EXPECT_TRUE(r.results_match);
}

TEST(AsyncCopy, ReportsWhetherTheHardwarePathExists) {
    auto r = la::compare_async_copy(1 << 18);
    EXPECT_GT(r.compute_capability, 0);
    EXPECT_EQ(r.hardware_accelerated, r.compute_capability >= 80);
}

TEST(AsyncCopy, BothPathsRun) {
    auto r = la::compare_async_copy(1 << 20);
    EXPECT_GT(r.sync_ms, 0.0f);
    EXPECT_GT(r.async_ms, 0.0f);
    // No speedup assertion: below sm_80 cg::memcpy_async lowers to the same
    // load-and-barrier as the synchronous path, so the two times should be
    // close and neither ordering is a defect.
}
