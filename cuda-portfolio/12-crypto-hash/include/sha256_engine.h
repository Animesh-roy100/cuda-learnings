#pragma once
//
// SHA-256 brute-force / proof-of-work engine. No CUDA syntax in this header.
//
// This is the one workload in the portfolio with essentially ZERO memory
// traffic: a nonce goes in, 32 bytes come out, and everything between lives in
// registers. That makes it a pure test of instruction throughput and register
// pressure -- the opposite end of the spectrum from the streaming kernels.
//
// The single number that decides performance here is registers per thread. Go
// one over the budget and the compiler spills to "local" memory, which is
// actually global memory with an L1 cache in front, and throughput collapses.
// Check with: nvcc --ptxas-options=-v
//
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace crypto {

using Digest = std::array<std::uint8_t, 32>;

struct MiningResult {
    bool found = false;
    std::uint64_t nonce = 0;
    Digest digest{};

    std::uint64_t budget = 0;      // nonces the search was allowed to try
    float elapsed_ms = 0.0f;       // TIME TO SOLUTION -- the metric that matters

    // There is deliberately no hashrate here, because this search cannot
    // cheaply report how many hashes it actually performed.
    //
    // Two wrong answers were tried first. Charging the full budget produced
    // rates far above the raw kernel, which is impossible. Estimating from the
    // winning nonce is also wrong: the launch has far more blocks than fit on
    // the device at once, so blocks run in waves, and when an early block wins
    // the later ones never execute. The nonce value therefore says nothing
    // about total work.
    //
    // A true count needs an atomic counter in the inner loop, which would
    // itself slow the thing being measured. Use benchmark_hashrate() for
    // throughput and this field for time-to-solution; they answer different
    // questions.
};

class Sha256Engine {
public:
    Sha256Engine();
    ~Sha256Engine();
    Sha256Engine(const Sha256Engine&) = delete;
    Sha256Engine& operator=(const Sha256Engine&) = delete;

    // Hash many independent messages of equal length, one thread each.
    // Messages must be at most 55 bytes so the block plus padding plus the
    // 8-byte length field fits one 64-byte SHA-256 block -- which keeps the
    // kernel to a single compression round with no loop over blocks.
    std::vector<Digest> hash_batch(const std::vector<std::string>& messages,
                                   float* elapsed_ms = nullptr);

    // Proof of work: find a nonce such that sha256(prefix || nonce) has at
    // least `leading_zero_bits` zero bits at the front.
    //
    // Reports through atomicCAS so the first finder wins and every other
    // thread can exit early, rather than all of them running to completion.
    MiningResult mine(const std::string& prefix, int leading_zero_bits,
                      std::uint64_t start_nonce, std::uint64_t max_tries);

    // Raw throughput with no early exit, for a clean hashrate number.
    double benchmark_hashrate(std::uint64_t hashes, float* elapsed_ms = nullptr);

    // Host reference, straight from FIPS 180-4.
    static Digest hash_cpu(const std::string& message);
    static std::string to_hex(const Digest& d);
    static int count_leading_zero_bits(const Digest& d);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace crypto
