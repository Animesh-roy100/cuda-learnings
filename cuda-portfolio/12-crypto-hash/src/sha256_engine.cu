// SHA-256 engine -- implementation.

#include "sha256_engine.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace crypto {
namespace {

// Round constants. __constant__ because every thread reads the same value at
// the same time -- exactly the broadcast pattern constant memory is built for,
// and it keeps them out of the register budget.
__constant__ unsigned int d_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static const unsigned int h_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

// __funnelshift_r maps to a single SASS instruction on Turing; the usual
// (x >> n) | (x << (32 - n)) is two ops plus an or. Over 64 rounds x 6 rotations
// that difference is the hot loop.
__device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n) {
    return __funnelshift_r(x, x, n);
}
__host__ __forceinline__ unsigned int rotr_h(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}

#define S0(x) (rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22))
#define S1(x) (rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25))
#define s0(x) (rotr(x, 7) ^ rotr(x, 18) ^ ((x) >> 3))
#define s1(x) (rotr(x, 17) ^ rotr(x, 19) ^ ((x) >> 10))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

// One SHA-256 block. w[] is the 16-word message schedule, expanded in place.
//
// #pragma unroll on both loops is the point of this kernel: fully unrolled,
// the round index becomes a compile-time constant, K[i] folds into the
// instruction stream, and every branch disappears. The cost is register
// pressure, which is why the message schedule is kept to 16 words and rotated
// rather than expanded to 64.
__device__ __forceinline__ void sha256_block(unsigned int* __restrict__ w,
                                             unsigned int* __restrict__ h) {
    unsigned int a = h[0], b = h[1], c = h[2], d = h[3];
    unsigned int e = h[4], f = h[5], g = h[6], hh = h[7];

#pragma unroll
    for (int i = 0; i < 64; ++i) {
        unsigned int wi;
        if (i < 16) {
            wi = w[i];
        } else {
            // Rolling 16-word window: w[i & 15] is updated in place, so the
            // schedule never costs more than 16 registers.
            wi = w[i & 15] + s0(w[(i + 1) & 15]) + w[(i + 9) & 15] + s1(w[(i + 14) & 15]);
            w[i & 15] = wi;
        }
        const unsigned int t1 = hh + S1(e) + CH(e, f, g) + d_K[i] + wi;
        const unsigned int t2 = S0(a) + MAJ(a, b, c);
        hh = g; g = f; f = e;
        e = d + t1;
        d = c; c = b; b = a;
        a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

__device__ __forceinline__ void sha256_init(unsigned int* h) {
    h[0] = 0x6a09e667u; h[1] = 0xbb67ae85u; h[2] = 0x3c6ef372u; h[3] = 0xa54ff53au;
    h[4] = 0x510e527fu; h[5] = 0x9b05688cu; h[6] = 0x1f83d9abu; h[7] = 0x5be0cd19u;
}

// Build the single padded block for a message of len <= 55 bytes.
__device__ __forceinline__ void pad_block(const unsigned char* msg, int len,
                                          unsigned int* w) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = 0u;
    for (int i = 0; i < len; ++i) w[i >> 2] |= ((unsigned int)msg[i]) << (24 - 8 * (i & 3));
    w[len >> 2] |= 0x80u << (24 - 8 * (len & 3));
    w[15] = (unsigned int)(len * 8);
}

__global__ void k_hash_batch(const unsigned char* __restrict__ msgs, int msg_len,
                             int n, unsigned char* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int w[16], h[8];
    pad_block(msgs + (size_t)i * msg_len, msg_len, w);
    sha256_init(h);
    sha256_block(w, h);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        out[(size_t)i * 32 + j * 4 + 0] = (unsigned char)(h[j] >> 24);
        out[(size_t)i * 32 + j * 4 + 1] = (unsigned char)(h[j] >> 16);
        out[(size_t)i * 32 + j * 4 + 2] = (unsigned char)(h[j] >> 8);
        out[(size_t)i * 32 + j * 4 + 3] = (unsigned char)(h[j]);
    }
}

// Proof of work. Each thread owns a nonce range; the whole state stays in
// registers and nothing touches memory until a winner appears.
__global__ void k_mine(const unsigned char* __restrict__ prefix, int prefix_len,
                       unsigned long long start_nonce, unsigned long long total,
                       unsigned int zero_bits,
                       unsigned long long* __restrict__ found_nonce,
                       unsigned int* __restrict__ found_flag,
                       unsigned char* __restrict__ found_digest) {
    const unsigned long long tid =
        blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    unsigned char msg[55];
#pragma unroll
    for (int i = 0; i < 55; ++i) msg[i] = 0;
    for (int i = 0; i < prefix_len; ++i) msg[i] = prefix[i];

    for (unsigned long long n = tid; n < total; n += stride) {
        // An early exit check each iteration would add a global load per hash.
        // Checking once per stride step keeps the inner loop clean while still
        // stopping promptly.
        if (*found_flag) return;

        const unsigned long long nonce = start_nonce + n;
        // Nonce as 16 hex-ish bytes appended to the prefix.
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned int nib = (unsigned int)((nonce >> (60 - 4 * i)) & 0xfu);
            msg[prefix_len + i] = (unsigned char)(nib < 10 ? ('0' + nib) : ('a' + nib - 10));
        }
        const int len = prefix_len + 16;

        unsigned int w[16], h[8];
        pad_block(msg, len, w);
        sha256_init(h);
        sha256_block(w, h);

        // Leading zero bits, without materialising the digest as bytes.
        unsigned int zeros = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (h[j] == 0u) { zeros += 32; continue; }
            zeros += __clz(h[j]);
            break;
        }

        if (zeros >= zero_bits) {
            // atomicCAS so exactly one thread claims the find; the rest see the
            // flag set and bail out on their next iteration.
            if (atomicCAS(found_flag, 0u, 1u) == 0u) {
                *found_nonce = nonce;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    found_digest[j * 4 + 0] = (unsigned char)(h[j] >> 24);
                    found_digest[j * 4 + 1] = (unsigned char)(h[j] >> 16);
                    found_digest[j * 4 + 2] = (unsigned char)(h[j] >> 8);
                    found_digest[j * 4 + 3] = (unsigned char)(h[j]);
                }
            }
            return;
        }
    }
}

// Throughput only: no early exit, no reporting. The accumulator exists purely
// so the optimiser cannot delete the hashing it never observes.
__global__ void k_hashrate(unsigned long long total, unsigned int* __restrict__ sink) {
    const unsigned long long tid =
        blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    unsigned int acc = 0;
    unsigned char msg[24];
#pragma unroll
    for (int i = 0; i < 24; ++i) msg[i] = (unsigned char)('a' + (i % 26));

    for (unsigned long long n = tid; n < total; n += stride) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            msg[16 + i] = (unsigned char)((n >> (8 * i)) & 0xffu);

        unsigned int w[16], h[8];
        pad_block(msg, 24, w);
        sha256_init(h);
        sha256_block(w, h);
        acc ^= h[0];
    }
    if (acc == 0xdeadbeefu) sink[0] = acc;   // never true; keeps the work alive
}

}  // namespace

// ---------------------------------------------------------------------------
struct Sha256Engine::Impl {
    int blocks = 0;
    int threads = 256;
};

Sha256Engine::Sha256Engine() : impl_(new Impl) {
    cudaDeviceProp p{};
    CU_CHECK(cudaGetDeviceProperties(&p, 0));
    // Enough blocks to saturate every SM several times over.
    impl_->blocks = p.multiProcessorCount * 32;
}

Sha256Engine::~Sha256Engine() { delete impl_; }

std::vector<Digest> Sha256Engine::hash_batch(const std::vector<std::string>& messages,
                                             float* elapsed_ms) {
    const int n = (int)messages.size();
    std::vector<Digest> out(n);
    if (n == 0) return out;

    std::size_t len = messages[0].size();
    for (const auto& m : messages) {
        if (m.size() != len)
            throw std::invalid_argument("hash_batch requires equal-length messages");
        if (m.size() > 55)
            throw std::invalid_argument("messages must be <= 55 bytes (single block)");
    }

    std::vector<unsigned char> flat((std::size_t)n * std::max<std::size_t>(len, 1), 0);
    for (int i = 0; i < n; ++i)
        std::memcpy(&flat[(std::size_t)i * len], messages[i].data(), len);

    unsigned char *d_in = nullptr, *d_out = nullptr;
    CU_CHECK(cudaMalloc(&d_in, flat.size()));
    CU_CHECK(cudaMalloc(&d_out, (std::size_t)n * 32));
    CU_CHECK(cudaMemcpy(d_in, flat.data(), flat.size(), cudaMemcpyHostToDevice));

    const int T = 256;
    cu::EventTimer t;
    t.start();
    k_hash_batch<<<(n + T - 1) / T, T>>>(d_in, (int)len, n, d_out);
    CU_CHECK_KERNEL();
    const float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<unsigned char> raw((std::size_t)n * 32);
    CU_CHECK(cudaMemcpy(raw.data(), d_out, raw.size(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) std::memcpy(out[i].data(), &raw[(std::size_t)i * 32], 32);

    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

MiningResult Sha256Engine::mine(const std::string& prefix, int leading_zero_bits,
                                std::uint64_t start_nonce, std::uint64_t max_tries) {
    if (prefix.size() > 39)
        throw std::invalid_argument("prefix must be <= 39 bytes (leaves room for a "
                                    "16-byte nonce inside one 55-byte block)");
    if (leading_zero_bits < 0 || leading_zero_bits > 255)
        throw std::invalid_argument("leading_zero_bits out of range");

    unsigned char* d_prefix = nullptr;
    unsigned long long* d_nonce = nullptr;
    unsigned int* d_flag = nullptr;
    unsigned char* d_digest = nullptr;
    CU_CHECK(cudaMalloc(&d_prefix, std::max<std::size_t>(prefix.size(), 1)));
    CU_CHECK(cudaMalloc(&d_nonce, sizeof(unsigned long long)));
    CU_CHECK(cudaMalloc(&d_flag, sizeof(unsigned int)));
    CU_CHECK(cudaMalloc(&d_digest, 32));
    if (!prefix.empty())
        CU_CHECK(cudaMemcpy(d_prefix, prefix.data(), prefix.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(d_flag, 0, sizeof(unsigned int)));
    CU_CHECK(cudaMemset(d_nonce, 0, sizeof(unsigned long long)));
    CU_CHECK(cudaMemset(d_digest, 0, 32));

    cu::EventTimer t;
    t.start();
    k_mine<<<impl_->blocks, impl_->threads>>>(d_prefix, (int)prefix.size(), start_nonce,
                                              max_tries, (unsigned)leading_zero_bits,
                                              d_nonce, d_flag, d_digest);
    CU_CHECK_KERNEL();
    const float ms = t.stop();

    MiningResult r;
    unsigned int flag = 0;
    CU_CHECK(cudaMemcpy(&flag, d_flag, sizeof(flag), cudaMemcpyDeviceToHost));
    r.found = flag != 0;
    if (r.found) {
        unsigned long long nonce = 0;
        CU_CHECK(cudaMemcpy(&nonce, d_nonce, sizeof(nonce), cudaMemcpyDeviceToHost));
        r.nonce = nonce;
        CU_CHECK(cudaMemcpy(r.digest.data(), d_digest, 32, cudaMemcpyDeviceToHost));
    }
    r.budget = max_tries;
    r.elapsed_ms = ms;

    cudaFree(d_prefix); cudaFree(d_nonce); cudaFree(d_flag); cudaFree(d_digest);
    return r;
}

double Sha256Engine::benchmark_hashrate(std::uint64_t hashes, float* elapsed_ms) {
    unsigned int* d_sink = nullptr;
    CU_CHECK(cudaMalloc(&d_sink, sizeof(unsigned int)));
    CU_CHECK(cudaMemset(d_sink, 0, sizeof(unsigned int)));

    k_hashrate<<<impl_->blocks, impl_->threads>>>(1 << 20, d_sink);   // warm
    CU_CHECK_KERNEL();

    cu::EventTimer t;
    t.start();
    k_hashrate<<<impl_->blocks, impl_->threads>>>(hashes, d_sink);
    CU_CHECK_KERNEL();
    const float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    cudaFree(d_sink);
    return hashes / (ms / 1e3) / 1e6;   // MH/s
}

// ---------------------------------------------------------------------------
Digest Sha256Engine::hash_cpu(const std::string& message) {
    unsigned int h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                         0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

    std::vector<unsigned char> msg(message.begin(), message.end());
    const std::uint64_t bitlen = (std::uint64_t)msg.size() * 8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    for (int i = 7; i >= 0; --i) msg.push_back((unsigned char)((bitlen >> (8 * i)) & 0xff));

    for (std::size_t off = 0; off < msg.size(); off += 64) {
        unsigned int w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((unsigned int)msg[off + i * 4] << 24) |
                   ((unsigned int)msg[off + i * 4 + 1] << 16) |
                   ((unsigned int)msg[off + i * 4 + 2] << 8) |
                   ((unsigned int)msg[off + i * 4 + 3]);
        for (int i = 16; i < 64; ++i) {
            const unsigned int x = w[i - 15], y = w[i - 2];
            const unsigned int ss0 = rotr_h(x, 7) ^ rotr_h(x, 18) ^ (x >> 3);
            const unsigned int ss1 = rotr_h(y, 17) ^ rotr_h(y, 19) ^ (y >> 10);
            w[i] = w[i - 16] + ss0 + w[i - 7] + ss1;
        }
        unsigned int a = h[0], b = h[1], c = h[2], d = h[3];
        unsigned int e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const unsigned int SS1 = rotr_h(e, 6) ^ rotr_h(e, 11) ^ rotr_h(e, 25);
            const unsigned int ch = (e & f) ^ (~e & g);
            const unsigned int t1 = hh + SS1 + ch + h_K[i] + w[i];
            const unsigned int SS0 = rotr_h(a, 2) ^ rotr_h(a, 13) ^ rotr_h(a, 22);
            const unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
            const unsigned int t2 = SS0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    Digest out{};
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = (unsigned char)(h[i] >> 24);
        out[i * 4 + 1] = (unsigned char)(h[i] >> 16);
        out[i * 4 + 2] = (unsigned char)(h[i] >> 8);
        out[i * 4 + 3] = (unsigned char)(h[i]);
    }
    return out;
}

std::string Sha256Engine::to_hex(const Digest& d) {
    static const char* hex = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (unsigned char b : d) {
        s.push_back(hex[b >> 4]);
        s.push_back(hex[b & 0xf]);
    }
    return s;
}

int Sha256Engine::count_leading_zero_bits(const Digest& d) {
    int z = 0;
    for (unsigned char b : d) {
        if (b == 0) { z += 8; continue; }
        for (int i = 7; i >= 0; --i) {
            if (b & (1u << i)) return z;
            ++z;
        }
        return z;
    }
    return z;
}

}  // namespace crypto
