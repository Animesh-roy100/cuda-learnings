#pragma once
//
// Scaled dot-product attention, three ways, with no CUDA syntax in this header.
//
//   O = softmax(Q K^T / sqrt(d)) V        Q, K, V, O: [heads][seq][head_dim]
//
//   NaiveCublas   how frameworks do it eagerly: cuBLAS materializes the full
//                 [heads][seq][seq] score matrix, a kernel softmaxes each row in
//                 place, and a second GEMM applies it to V. Memory is O(seq^2).
//   FusedGlobal   one pass over the keys per query with an ONLINE softmax -- a
//                 running max, normalizer and weighted sum, updated key by key --
//                 so no seq x seq matrix ever exists. Keys and values are read
//                 straight from global memory.
//   FusedTiled    the same pass, with keys and values staged into shared memory
//                 a tile at a time, the IO-aware structure FlashAttention is
//                 named for. The tile size sizes the __shared__ arrays, so it is
//                 compiled per size and swept.
//
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fa {

struct Shape {
    int heads = 1;
    int seq = 256;
    int head_dim = 64;
    bool causal = false;   // query i attends only to keys 0..i
};

enum class Kernel { NaiveCublas, FusedGlobal, FusedTiled };
const char* to_string(Kernel k);

// The fused kernels stage vectors into fixed-width shared arrays, so they are
// compiled for one head dimension.
constexpr int kFusedHeadDim = 64;

// Tile sizes (keys per tile) FusedTiled is compiled for.
std::vector<int> tile_sizes();

class Attention {
public:
    explicit Attention(const Shape& s);
    ~Attention();
    Attention(const Attention&) = delete;
    Attention& operator=(const Attention&) = delete;

    const Shape& shape() const;

    // Row-major [heads][seq][head_dim] each.
    void set_qkv(const std::vector<float>& q, const std::vector<float>& k,
                 const std::vector<float>& v);

    // tile is used only by FusedTiled.
    std::vector<float> forward(Kernel k, int tile = 32);

    // Median ms for the forward pass alone: no upload, no download.
    float time(Kernel k, int tile = 32, int iterations = 5, int warmup = 2);

    // Device memory the kernel needs for this shape, in bytes. NaiveCublas is
    // dominated by the score matrix; the fused kernels by Q, K, V and O.
    std::size_t device_bytes(Kernel k) const;

    // Blocks per SM for a fused kernel, from the occupancy API.
    int blocks_per_sm(Kernel k, int tile = 32) const;

private:
    struct Impl;
    Impl* impl_;
};

// FP64 host reference, with the softmax computed via log-sum-exp.
std::vector<float> reference_attention(const Shape& s, const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v);

}  // namespace fa
