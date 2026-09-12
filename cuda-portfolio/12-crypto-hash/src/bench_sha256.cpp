// SHA-256 throughput and proof-of-work benchmark.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "cu/device.hpp"
#include "sha256_engine.h"

using crypto::MiningResult;
using crypto::Sha256Engine;

int main() try {
    auto dev = cu::query_device();
    cu::print_banner(dev);

    std::printf("SHA-256: the one kernel here with essentially ZERO memory traffic.\n");
    std::printf("A nonce goes in, 32 bytes come out, everything between is registers.\n\n");

    Sha256Engine e;

    std::printf("=== raw hashrate ===\n");
    double best = 0.0;
    for (int i = 0; i < 3; ++i) {
        float ms = 0.0f;
        const double mhs = e.benchmark_hashrate(200'000'000ULL, &ms);
        best = std::max(best, mhs);
    }
    std::printf("  %.1f MH/s  (%.2f GH/s)\n", best, best / 1000.0);

    // Each SHA-256 block is 64 rounds of ~10 integer ops plus the schedule.
    const double ops_per_hash = 64.0 * 12.0;
    const double giops = best * 1e6 * ops_per_hash / 1e9;
    std::printf("  ~%.0f integer ops per hash -> %.0f GIOP/s\n", ops_per_hash, giops);
    std::printf("  %d SMs x 64 INT32 lanes at ~1.5 GHz is roughly %.0f GIOP/s of\n",
                dev.sm_count, dev.sm_count * 64 * 1.5);
    std::printf("  headroom, so this sits at about %.0f%% of integer peak.\n\n",
                100.0 * giops / (dev.sm_count * 64 * 1.5));

    std::printf("=== proof of work: time vs difficulty ===\n");
    std::printf("  Expected work doubles per bit: 2^bits hashes on average.\n");
    std::printf("  %-6s %14s %12s %16s\n",
                "bits", "expected work", "time (ms)", "nonce");
    for (int bits : {16, 20, 22, 24}) {
        const unsigned long long budget = std::min<unsigned long long>(
            1ULL << (bits + 3), 600'000'000ULL);
        MiningResult r = e.mine("portfolio-block-", bits, 0, budget);
        if (r.found) {
            std::printf("  %-6d %14.0f %12.2f %16llu\n",
                        bits, std::pow(2.0, bits), r.elapsed_ms,
                        (unsigned long long)r.nonce);
        } else {
            std::printf("  %-6d %14.0f %12.2f   not found in %llu\n",
                        bits, std::pow(2.0, bits), r.elapsed_ms, budget);
        }
    }

    std::printf("\n  There is no hashrate column here, deliberately. Two plausible\n");
    std::printf("  ways to produce one are both wrong:\n\n");
    std::printf("   * charging the full budget ignores the early exit and gave rates\n");
    std::printf("     ABOVE the raw kernel -- impossible, and a reliable signal that\n");
    std::printf("     the measurement rather than the code is broken;\n");
    std::printf("   * estimating from the winning nonce is also wrong, because this\n");
    std::printf("     launch has far more blocks than fit on %d SMs at once. Blocks\n", dev.sm_count);
    std::printf("     run in waves, and when an early block wins the later ones never\n");
    std::printf("     execute at all, so the nonce says nothing about total work.\n\n");
    std::printf("  Counting truthfully needs an atomic in the inner loop, which would\n");
    std::printf("  slow the very thing being measured. Time-to-solution is the honest\n");
    std::printf("  metric for a search, and it is what proof-of-work actually cares\n");
    std::printf("  about. Use the raw figure above for throughput.\n");

    std::printf("\n=== what governs this kernel ===\n");
    std::printf("  Register pressure, and nothing else. Every variable lives in\n");
    std::printf("  registers; one spill sends state to local memory (global memory\n");
    std::printf("  behind L1) and throughput collapses. Three decisions follow:\n\n");
    std::printf("  * the message schedule is a ROLLING 16-word window, not 64 words.\n");
    std::printf("    Expanding to 64 would cost 48 extra registers per thread for no\n");
    std::printf("    algorithmic gain.\n");
    std::printf("  * round constants live in __constant__ memory, so they broadcast\n");
    std::printf("    to every thread without consuming registers.\n");
    std::printf("  * rotations use __funnelshift_r, one SASS instruction, rather than\n");
    std::printf("    the shift-or-shift idiom. Over 64 rounds x 6 rotations that is\n");
    std::printf("    the hot loop.\n\n");
    std::printf("  Verify with:  nvcc --ptxas-options=-v -arch=sm_75 -c src/sha256_engine.cu\n");
    std::printf("  Look for \"spill stores\"/\"spill loads\" -- both must read 0 bytes.\n");
    return 0;
} catch (const std::exception& ex) {
    std::fprintf(stderr, "\nFATAL: %s\n", ex.what());
    return 1;
}
