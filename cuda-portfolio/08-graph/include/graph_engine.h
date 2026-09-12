#pragma once
//
// Graph processing over CSR -- public interface. No CUDA syntax.
//
#include <cstdint>
#include <vector>

namespace graph {

// Compressed Sparse Row. row_offsets has n+1 entries; the neighbours of node i
// are col_indices[row_offsets[i] .. row_offsets[i+1]).
//
// CSR is the only sane layout for GPU graph work: the neighbour list of a node
// is contiguous, so a warp walking it gets coalesced reads. An adjacency list
// of pointers would scatter every access.
struct CsrGraph {
    std::vector<std::int32_t> row_offsets;
    std::vector<std::int32_t> col_indices;
    std::vector<float> weights;            // parallel to col_indices; may be empty

    std::int32_t num_nodes() const {
        return row_offsets.empty() ? 0 : static_cast<std::int32_t>(row_offsets.size() - 1);
    }
    std::int64_t num_edges() const { return static_cast<std::int64_t>(col_indices.size()); }

    // Transpose (in-edges), needed by pull-style algorithms.
    CsrGraph transpose() const;

    // A scale-free-ish generator: a few hub nodes with very high degree, which
    // is exactly the load-imbalance case a naive one-thread-per-node kernel
    // handles worst.
    static CsrGraph random_power_law(std::int32_t n, int avg_degree, unsigned seed,
                                     bool weighted = false);
    static CsrGraph grid_2d(std::int32_t w, std::int32_t h, bool weighted = false);
};

enum class Direction {
    Push,   // each node scatters to its out-neighbours; needs atomics
    Pull,   // each node gathers from its in-neighbours; atomic-free
};

enum class Balance {
    ThreadPerNode,  // one thread per node: hub nodes stall their whole warp
    WarpPerNode,    // one warp per node: hubs are split across 32 lanes
};

struct PageRankResult {
    std::vector<float> rank;
    int iterations = 0;
    double residual = 0.0;
};

class GraphEngine {
public:
    explicit GraphEngine(const CsrGraph& g);
    ~GraphEngine();
    GraphEngine(const GraphEngine&) = delete;
    GraphEngine& operator=(const GraphEngine&) = delete;

    PageRankResult pagerank(float damping = 0.85f,
                            int max_iter = 100,
                            float tol = 1e-6f,
                            Direction dir = Direction::Pull,
                            Balance balance = Balance::WarpPerNode);

    // Single-source shortest path. Unreachable nodes come back as kUnreachable.
    // Uses weights if present, otherwise unit edge cost.
    std::vector<float> sssp(std::int32_t source, int* iterations = nullptr);

    static constexpr float kUnreachable = 3.4e38f;

    // CPU references, for tests.
    static PageRankResult pagerank_cpu(const CsrGraph& g, float damping = 0.85f,
                                       int max_iter = 100, float tol = 1e-6f);
    static std::vector<float> sssp_cpu(const CsrGraph& g, std::int32_t source);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace graph
