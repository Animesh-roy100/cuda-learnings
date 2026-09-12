#pragma once
//
// GPU spatial indexing: Morton grid, k-NN, DBSCAN. No CUDA syntax.
//
#include <cstdint>
#include <vector>

namespace spatial {

struct Point3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

struct BuildStats {
    float morton_ms = 0.0f;
    float sort_ms = 0.0f;
    float ranges_ms = 0.0f;
    float total_ms() const { return morton_ms + sort_ms + ranges_ms; }
};

struct DbscanResult {
    // Indexed like the ORIGINAL points passed to the constructor, not the
    // internal Morton order. labels[i] describes points[i].
    std::vector<std::int32_t> labels;   // cluster id in [0, num_clusters), or kNoise
    std::int64_t core_points = 0;
    std::int64_t noise_points = 0;
    std::int32_t num_clusters = 0;
    float count_ms = 0.0f;
    float union_ms = 0.0f;
    float border_ms = 0.0f;
};

// Points must lie in [0,1]^3; anything outside is clamped into the edge cell.
class SpatialIndex {
public:
    static constexpr std::int32_t kNoise = -1;
    static constexpr int kGrid = 128;                 // cells per axis -> 2^21 cells
    static constexpr float kCell = 1.0f / kGrid;      // cell edge length

    explicit SpatialIndex(const std::vector<Point3>& points);
    ~SpatialIndex();
    SpatialIndex(const SpatialIndex&) = delete;
    SpatialIndex& operator=(const SpatialIndex&) = delete;

    BuildStats build_stats() const;

    // Points in Morton (Z-order) order. Neighbour queries return indices into
    // THIS array, not the caller's original ordering.
    std::vector<Point3> sorted_points() const;

    // Original index of each sorted point, so results can be mapped back.
    std::vector<std::int32_t> permutation() const;

    // Squared distances to the k nearest neighbours of each query point.
    // Query indices refer to sorted order.
    std::vector<float> knn(const std::vector<std::int32_t>& queries, int k,
                           float* elapsed_ms = nullptr) const;
    std::vector<float> knn_bruteforce(const std::vector<std::int32_t>& queries, int k,
                                      float* elapsed_ms = nullptr) const;

    // eps must be <= kCell: the 3x3x3 cell stencil only covers that radius.
    DbscanResult dbscan(float eps, int min_pts) const;

    std::int32_t size() const;

    // Z-order encode, exposed for tests (bit interleave of three 7-bit coords).
    static std::uint32_t morton3(std::uint32_t x, std::uint32_t y, std::uint32_t z);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace spatial
