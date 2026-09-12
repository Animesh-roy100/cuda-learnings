#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "primitives.h"

using prim::BankResult;
using prim::GraphComparison;
using prim::MemoryMode;
using prim::OccupancyReport;

TEST(Primitives, ReportsManagedMemoryCapabilities) {
    auto c = prim::query_managed_caps();
    // Every Turing card supports managed allocation; the finer-grained
    // features are driver-model dependent, so only consistency is asserted.
    EXPECT_TRUE(c.managed_memory);
    EXPECT_EQ(c.prefetch_supported(), c.concurrent_managed_access);
}

// ---------------------------------------------------------------------------
// CUDA Graphs
// ---------------------------------------------------------------------------

// The property that matters first: a graph must compute exactly what the
// stream computed. It changes submission, never semantics.
TEST(Graphs, ProduceIdenticalResults) {
    auto r = prim::compare_graph_vs_stream(20, 50, 4096);
    EXPECT_TRUE(r.results_match)
        << "graph replay diverged from individually launched kernels";
}

TEST(Graphs, ReduceLaunchOverheadOnLongChains) {
    // 100 tiny kernels x 200 iterations = 20000 launches. Long enough that
    // per-launch CPU cost dominates, which is the regime graphs target.
    auto r = prim::compare_graph_vs_stream(100, 200, 4096);
    ASSERT_TRUE(r.results_match);
    EXPECT_GT(r.stream_ms, 0.0f);
    EXPECT_GT(r.graph_ms, 0.0f);
    EXPECT_LT(r.graph_ms, r.stream_ms)
        << "graph " << r.graph_ms << " ms vs stream " << r.stream_ms << " ms";
}

// Conversely: with one big kernel there is nothing to amortise, so graphs
// should NOT help. Asserting a speedup everywhere would be asserting something
// false about how they work.
TEST(Graphs, DoNotHelpWhenThereIsNothingToAmortise) {
    auto r = prim::compare_graph_vs_stream(1, 50, 1 << 20);
    ASSERT_TRUE(r.results_match);
    EXPECT_LT(r.speedup(), 3.0)
        << "a single large kernel should not see a big graph speedup, got "
        << r.speedup() << "x";
}

TEST(Graphs, RejectBadArguments) {
    EXPECT_THROW(prim::compare_graph_vs_stream(0, 10), std::invalid_argument);
    EXPECT_THROW(prim::compare_graph_vs_stream(10, 0), std::invalid_argument);
    EXPECT_THROW(prim::compare_graph_vs_stream(10, 10, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Unified Memory
// ---------------------------------------------------------------------------
TEST(UnifiedMemory, AllSupportedModesComputeTheSameResult) {
    const std::size_t N = 1 << 20;
    auto results = prim::compare_memory_modes(N, 3);
    ASSERT_FALSE(results.empty());

    double reference = 0.0;
    bool have_reference = false;
    for (const auto& r : results) {
        if (!r.supported) continue;
        if (!have_reference) {
            reference = r.checksum;
            have_reference = true;
            continue;
        }
        EXPECT_NEAR(r.checksum, reference, std::fabs(reference) * 1e-9 + 1e-6)
            << prim::to_string(r.mode) << " disagrees with the baseline";
    }
    EXPECT_TRUE(have_reference);
}

TEST(UnifiedMemory, ExplicitCopyIsAlwaysAvailable) {
    auto results = prim::compare_memory_modes(1 << 18, 2);
    auto it = std::find_if(results.begin(), results.end(), [](const auto& r) {
        return r.mode == MemoryMode::ExplicitCopy;
    });
    ASSERT_NE(it, results.end());
    EXPECT_TRUE(it->supported);
    EXPECT_GT(it->ms, 0.0f);
}

// Availability of BOTH tuning modes must track the device attribute exactly.
//
// This matters more than it looks. cudaMemAdvise and cudaMemPrefetchAsync both
// return cudaErrorInvalidDevice without concurrentManagedAccess, and that error
// is sticky: it poisons the CUDA context so every later kernel launch in the
// process fails too. Attempting them "just to see" takes down unrelated tests,
// which is exactly what happened before this gate existed.
TEST(UnifiedMemory, TuningModesSkippedExactlyWhenUnsupported) {
    const auto caps = prim::query_managed_caps();
    auto results = prim::compare_memory_modes(1 << 18, 2);

    for (auto mode : {MemoryMode::ManagedAdvised, MemoryMode::ManagedPrefetched}) {
        auto it = std::find_if(results.begin(), results.end(),
                               [mode](const auto& r) { return r.mode == mode; });
        ASSERT_NE(it, results.end());

        const bool expected = (mode == MemoryMode::ManagedAdvised)
                                  ? caps.advise_supported()
                                  : caps.prefetch_supported();
        EXPECT_EQ(it->supported, expected) << prim::to_string(mode);
        if (!it->supported) {
            EXPECT_FALSE(it->skip_reason.empty())
                << prim::to_string(mode) << ": a skipped mode must say why";
            EXPECT_EQ(it->ms, 0.0f)
                << prim::to_string(mode) << ": a skipped mode must not report a timing";
        }
    }
}

// The context must still be usable after compare_memory_modes() runs, which is
// the real regression guard: a sticky error here would break everything after.
TEST(UnifiedMemory, LeavesTheContextUsable) {
    prim::compare_memory_modes(1 << 18, 2);
    auto r = prim::compare_graph_vs_stream(4, 10, 1024);
    EXPECT_TRUE(r.results_match) << "context was poisoned by the memory-mode sweep";
}

TEST(UnifiedMemory, RejectsBadArguments) {
    EXPECT_THROW(prim::compare_memory_modes(0, 1), std::invalid_argument);
    EXPECT_THROW(prim::compare_memory_modes(1024, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Bank conflicts
// ---------------------------------------------------------------------------
TEST(BankConflicts, StrideSweepIsWellFormed) {
    auto r = prim::measure_bank_conflicts(1000);
    ASSERT_EQ(r.size(), 6u);
    EXPECT_EQ(r[0].stride, 1);
    EXPECT_EQ(r[0].expected_way_conflict, 1);
    EXPECT_DOUBLE_EQ(r[0].slowdown_vs_stride1, 1.0);
    EXPECT_EQ(r.back().stride, 32);
    EXPECT_EQ(r.back().expected_way_conflict, 32);
    for (const auto& x : r) EXPECT_GT(x.ms, 0.0f);
}

// The actual claim: a 32-way conflict serialises the warp and is measurably
// slower than conflict-free access.
TEST(BankConflicts, ThirtyTwoWayConflictIsSlowerThanConflictFree) {
    auto r = prim::measure_bank_conflicts(4000);
    ASSERT_EQ(r.size(), 6u);
    const double worst = r.back().slowdown_vs_stride1;
    EXPECT_GT(worst, 1.2) << "stride-32 was only " << worst
                          << "x stride-1; the conflict did not materialise";
}

TEST(BankConflicts, RejectsBadArguments) {
    EXPECT_THROW(prim::measure_bank_conflicts(0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Occupancy
// ---------------------------------------------------------------------------
TEST(Occupancy, ReportsPlausibleNumbers) {
    auto reports = prim::analyze_occupancy(1 << 22);
    ASSERT_FALSE(reports.empty());
    for (const auto& r : reports) {
        EXPECT_GT(r.active_blocks_per_sm, 0) << "block size " << r.block_size;
        EXPECT_GT(r.occupancy, 0.0);
        EXPECT_LE(r.occupancy, 1.0 + 1e-9)
            << "occupancy cannot exceed 100%, got " << r.occupancy;
        EXPECT_LE(r.active_warps_per_sm, r.max_warps_per_sm);
        EXPECT_GT(r.measured_ms, 0.0f);
    }
}

TEST(Occupancy, SuggestedBlockSizeIsUsable) {
    const int b = prim::suggested_block_size();
    EXPECT_GE(b, 32);
    EXPECT_LE(b, 1024);
    EXPECT_EQ(b % 32, 0) << "a sensible block size is a multiple of the warp size";
}

// ---------------------------------------------------------------------------
// __activemask
// ---------------------------------------------------------------------------
TEST(ActiveMask, ReflectsBranchDivergenceExactly) {
    auto s = prim::sample_activemask();
    ASSERT_EQ(s.size(), 32u);

    // Even lanes take one branch, odd lanes the other. Within a branch, the
    // mask must contain exactly the lanes on that side -- 16 of them, and the
    // bit pattern 0x55555555 or 0xAAAAAAAA.
    for (const auto& x : s) {
        EXPECT_EQ(x.branch, x.lane & 1);
        EXPECT_EQ(x.popcount, 16)
            << "lane " << x.lane << " saw " << x.popcount << " active lanes";
        const unsigned expected = (x.branch == 0) ? 0x55555555u : 0xAAAAAAAAu;
        EXPECT_EQ(x.mask, expected)
            << "lane " << x.lane << " mask 0x" << std::hex << x.mask;
        // The lane must always see itself as active.
        EXPECT_TRUE(x.mask & (1u << x.lane));
    }
}
