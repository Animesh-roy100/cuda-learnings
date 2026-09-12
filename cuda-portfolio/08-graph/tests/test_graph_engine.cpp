#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "graph_engine.h"

using graph::Balance;
using graph::CsrGraph;
using graph::Direction;
using graph::GraphEngine;

namespace {

// 4-node cycle: every node has out-degree 1, so PageRank is uniform by symmetry.
CsrGraph cycle4() {
    CsrGraph g;
    g.row_offsets = {0, 1, 2, 3, 4};
    g.col_indices = {1, 2, 3, 0};
    return g;
}

// A star: every node points at node 0, which points nowhere (dangling).
CsrGraph star(int n) {
    CsrGraph g;
    g.row_offsets.assign(n + 1, 0);
    for (int v = 1; v < n; ++v) g.col_indices.push_back(0);
    g.row_offsets[0] = 0;
    g.row_offsets[1] = 0;                   // node 0 has no out-edges
    for (int v = 1; v < n; ++v) g.row_offsets[v + 1] = g.row_offsets[v] + 1;
    return g;
}

}  // namespace

TEST(Graph, TransposeIsInvolution) {
    auto g = CsrGraph::random_power_law(2000, 6, 42);
    auto tt = g.transpose().transpose();
    ASSERT_EQ(tt.num_nodes(), g.num_nodes());
    ASSERT_EQ(tt.num_edges(), g.num_edges());
    EXPECT_EQ(tt.row_offsets, g.row_offsets);
    EXPECT_EQ(tt.col_indices, g.col_indices);
}

TEST(Graph, PageRankOnCycleIsUniform) {
    auto g = cycle4();
    GraphEngine e(g);
    auto r = e.pagerank(0.85f, 200, 1e-9f);
    ASSERT_EQ(r.rank.size(), 4u);
    for (float v : r.rank) EXPECT_NEAR(v, 0.25f, 1e-4f);
}

TEST(Graph, PageRankSumsToOne) {
    auto g = CsrGraph::random_power_law(20000, 8, 7);
    GraphEngine e(g);
    auto r = e.pagerank(0.85f, 200, 1e-8f);
    double total = std::accumulate(r.rank.begin(), r.rank.end(), 0.0);
    // Dangling-mass redistribution is what keeps this at 1; without it the
    // vector bleeds rank every iteration.
    EXPECT_NEAR(total, 1.0, 1e-3);
}

TEST(Graph, PageRankStarConcentratesOnHub) {
    auto g = star(1000);
    GraphEngine e(g);
    auto r = e.pagerank(0.85f, 200, 1e-9f);
    // Node 0 receives from all 999 others and must dominate.
    for (int v = 1; v < 1000; ++v) EXPECT_GT(r.rank[0], r.rank[v]);
}

TEST(Graph, PageRankMatchesCpuReference) {
    auto g = CsrGraph::random_power_law(5000, 6, 123);
    GraphEngine e(g);
    auto gpu = e.pagerank(0.85f, 100, 1e-7f);
    auto cpu = GraphEngine::pagerank_cpu(g, 0.85f, 100, 1e-7f);

    ASSERT_EQ(gpu.rank.size(), cpu.rank.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < gpu.rank.size(); ++i)
        worst = std::max(worst, std::fabs(double(gpu.rank[i]) - cpu.rank[i]));
    EXPECT_LT(worst, 1e-5) << "max per-node deviation from CPU reference";
}

// Push and pull compute the same fixed point by different means; if they
// disagree, one of them has a bug (usually a missing atomic in push).
TEST(Graph, PushAndPullAgree) {
    auto g = CsrGraph::random_power_law(5000, 6, 99);
    GraphEngine e(g);
    auto pull = e.pagerank(0.85f, 100, 1e-7f, Direction::Pull);
    auto push = e.pagerank(0.85f, 100, 1e-7f, Direction::Push);

    double worst = 0.0;
    for (std::size_t i = 0; i < pull.rank.size(); ++i)
        worst = std::max(worst, std::fabs(double(pull.rank[i]) - push.rank[i]));
    EXPECT_LT(worst, 1e-4) << "push vs pull disagreement";
}

// Balance strategy must not change the answer, only the speed.
TEST(Graph, WarpAndThreadBalanceAgree) {
    auto g = CsrGraph::random_power_law(5000, 6, 55);
    GraphEngine e(g);
    auto a = e.pagerank(0.85f, 100, 1e-7f, Direction::Pull, Balance::WarpPerNode);
    auto b = e.pagerank(0.85f, 100, 1e-7f, Direction::Pull, Balance::ThreadPerNode);

    double worst = 0.0;
    for (std::size_t i = 0; i < a.rank.size(); ++i)
        worst = std::max(worst, std::fabs(double(a.rank[i]) - b.rank[i]));
    EXPECT_LT(worst, 1e-6);
}

TEST(Graph, SsspOnGridMatchesManhattanDistance) {
    const int W = 64, H = 64;
    auto g = CsrGraph::grid_2d(W, H);
    GraphEngine e(g);
    auto d = e.sssp(0);

    // Unit-weight 4-connected grid: distance from the corner is |x| + |y|.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            EXPECT_FLOAT_EQ(d[y * W + x], float(x + y)) << "at (" << x << "," << y << ")";
}

TEST(Graph, SsspMatchesDijkstra) {
    auto g = CsrGraph::random_power_law(20000, 8, 321, /*weighted=*/true);
    GraphEngine e(g);
    int iters = 0;
    auto gpu = e.sssp(0, &iters);
    auto cpu = GraphEngine::sssp_cpu(g, 0);

    ASSERT_EQ(gpu.size(), cpu.size());
    std::size_t reached = 0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        if (cpu[i] >= GraphEngine::kUnreachable) {
            EXPECT_GE(gpu[i], GraphEngine::kUnreachable) << "node " << i;
        } else {
            ++reached;
            EXPECT_NEAR(gpu[i], cpu[i], 1e-3f) << "node " << i;
        }
    }
    EXPECT_GT(reached, 0u) << "test graph must have a reachable component";
    EXPECT_GT(iters, 0);
}

TEST(Graph, SsspSourceIsZeroAndUnreachableStaysUnreachable) {
    CsrGraph g;
    g.row_offsets = {0, 1, 1, 1};       // 0 -> 1; node 2 isolated
    g.col_indices = {1};
    GraphEngine e(g);
    auto d = e.sssp(0);
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    EXPECT_FLOAT_EQ(d[1], 1.0f);
    EXPECT_GE(d[2], GraphEngine::kUnreachable);
}
