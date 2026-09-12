#pragma once
//
// IVF-Flat approximate nearest neighbour index -- public interface.
// No CUDA syntax: host code and tests compile as plain C++20.
//
// The shape of the problem: exact search over N vectors costs N distance
// computations per query. IVF partitions the space into `nlist` Voronoi cells
// via k-means, then a query only scans the `nprobe` cells nearest its own
// centroid. That turns O(N) into roughly O(N * nprobe / nlist) -- at the cost
// of missing neighbours that sit just across a cell boundary.
//
// That trade is the whole point, so the API exposes both paths and the test
// suite measures the recall loss rather than assuming it away.
//
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ann {

enum class Metric {
    L2,      // squared euclidean; smaller is closer
    Cosine,  // 1 - cosine similarity, so smaller is closer here too
};

struct SearchResult {
    // Row-major [n_queries][k]. id == -1 marks a slot with no candidate,
    // which happens when nprobe cells hold fewer than k vectors between them.
    std::vector<std::int32_t> ids;
    std::vector<float> distances;
    int n_queries = 0;
    int k = 0;

    std::int32_t id_at(int q, int j) const { return ids[(std::size_t)q * k + j]; }
    float dist_at(int q, int j) const { return distances[(std::size_t)q * k + j]; }
};

class IvfFlatIndex {
public:
    static constexpr int kMaxK = 32;      // top-k is held in registers

    IvfFlatIndex(int dim, int nlist, Metric metric = Metric::L2);
    ~IvfFlatIndex();
    IvfFlatIndex(const IvfFlatIndex&) = delete;
    IvfFlatIndex& operator=(const IvfFlatIndex&) = delete;

    // k-means on the host, then centroids are uploaded. Must precede add().
    void train(const std::vector<float>& vectors, int n, int iters = 10,
               unsigned seed = 1234);
    bool is_trained() const;

    // Assigns each vector to its nearest centroid and builds the inverted lists.
    void add(const std::vector<float>& vectors, int n);

    // nprobe == nlist degenerates to an exhaustive scan, which is how the tests
    // pin the approximate path against the exact one.
    SearchResult search(const std::vector<float>& queries, int nq, int k,
                        int nprobe, float* elapsed_ms = nullptr) const;

    // Exact scan of every vector. The ground truth recall is measured against.
    SearchResult search_bruteforce(const std::vector<float>& queries, int nq, int k,
                                   float* elapsed_ms = nullptr) const;

    int dim() const;
    int nlist() const;
    int size() const;                       // vectors added
    std::vector<int> list_sizes() const;    // vectors per cell, for balance checks
    std::size_t device_bytes() const;

private:
    struct Impl;
    Impl* impl_;
};

// Fraction of true neighbours the approximate result recovered, averaged over
// queries. The standard quality measure for an ANN index.
double recall_at_k(const SearchResult& approx, const SearchResult& exact);

}  // namespace ann
