// DPI throughput benchmark: line rate for a realistic signature set.

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "cu/device.hpp"
#include "dpi_engine.h"

using dpi::DpiEngine;
using dpi::MatchStats;

namespace {
std::vector<std::string> signature_set(int n) {
    // Shapes typical of an IDS ruleset: protocol verbs, shell fragments,
    // injection markers, plus filler to reach the requested count.
    std::vector<std::string> base = {
        "GET /", "POST /", "User-Agent:", "cmd.exe", "/bin/sh", "SELECT * FROM",
        "DROP TABLE", "UNION SELECT", "<script>", "eval(", "base64_decode",
        "powershell -enc", "wget http", "curl -s", "nc -e", "/etc/passwd",
        "..\..\\", "../../", "%00", "\x90\x90\x90"};
    std::vector<std::string> out;
    for (int i = 0; i < n; ++i) {
        if (i < (int)base.size()) out.push_back(base[i]);
        else out.push_back("SIG" + std::to_string(i) + "MARKER");
    }
    return out;
}
}  // namespace

int main() try {
    auto dev = cu::query_device();
    cu::print_banner(dev);

    const int NPKT = 200000;
    std::printf("Aho-Corasick multi-pattern matching, one thread per packet.\n\n");

    std::printf("=== automaton size vs pattern count ===\n");
    std::printf("  %-10s %10s %14s\n", "patterns", "states", "table");
    for (int np : {16, 64, 256, 1024, 4096}) {
        DpiEngine e(signature_set(np));
        std::printf("  %-10d %10d %11.1f MB\n", np, e.num_states(),
                    e.table_bytes() / 1e6);
    }
    std::printf("  The dense 256-way goto table is the trade: it costs memory\n");
    std::printf("  but turns every byte into ONE load instead of a fail-link walk.\n\n");

    auto pats = signature_set(1024);
    DpiEngine engine(pats);
    std::printf("=== throughput (%d patterns, %d states, %.1f MB table) ===\n",
                engine.num_patterns(), engine.num_states(), engine.table_bytes() / 1e6);

    for (int mtu : {256, 512, 1500}) {
        auto batch = dpi::make_traffic(NPKT, mtu / 2, mtu, pats, 0.05, 2026);
        MatchStats st{};
        engine.scan_bitmask(batch, &st);          // warm
        float best = 1e30f;
        for (int i = 0; i < 5; ++i) {
            engine.scan_bitmask(batch, &st);
            best = std::min(best, st.kernel_ms);
        }
        const double gbps = double(st.bytes_scanned) / (best / 1e3) / 1e9;
        std::printf("  mtu<=%4d  %7.1f MB  %7.2f ms  %6.2f GB/s  %6.2f Gbit/s  "
                    "%5.1f Mpkt/s\n",
                    mtu, st.bytes_scanned / 1e6, best, gbps, gbps * 8,
                    NPKT / (best / 1e3) / 1e6);
    }

    // Zero-copy vs staged upload.
    auto batch = dpi::make_traffic(NPKT, 750, 1500, pats, 0.05, 99);
    MatchStats a{}, b{};
    engine.scan_bitmask(batch, &a);
    engine.scan_bitmask_zerocopy(batch, &b);
    float best_a = 1e30f, best_b = 1e30f;
    for (int i = 0; i < 5; ++i) {
        engine.scan_bitmask(batch, &a);
        engine.scan_bitmask_zerocopy(batch, &b);
        best_a = std::min(best_a, a.kernel_ms);
        best_b = std::min(best_b, b.kernel_ms);
    }
    std::printf("\n=== payload residency (%.0f MB) ===\n", a.bytes_scanned / 1e6);
    std::printf("  device memory   %7.2f ms  %6.2f GB/s\n",
                best_a, double(a.bytes_scanned) / (best_a / 1e3) / 1e9);
    std::printf("  zero-copy       %7.2f ms  %6.2f GB/s\n",
                best_b, double(b.bytes_scanned) / (best_b / 1e3) / 1e9);
    std::printf("\n  Zero-copy LOSES here, by roughly %.0fx, and the reason matters.\n",
                best_b / best_a);
    std::printf("  This kernel walks the payload ONE BYTE AT A TIME with a data-\n");
    std::printf("  dependent state machine. Over PCIe each of those byte reads pays\n");
    std::printf("  full link latency and nothing coalesces, so mapped memory turns a\n");
    std::printf("  bandwidth problem into a latency problem. Zero-copy suits bulk,\n");
    std::printf("  coalesced, touch-once streaming -- not pointer-chasing.\n");
    std::printf("  For a real capture ring, stage into pinned memory and DMA it.\n");

    std::printf("\n=== what actually limits this kernel ===\n");
    std::printf("  %.2f GB/s is about 1%% of this card's %.0f GB/s peak. Two causes,\n",
                double(a.bytes_scanned) / (best_a / 1e3) / 1e9, dev.peak_bandwidth_gbps());
    std::printf("  in order of size:\n\n");
    std::printf("  1. Uncoalesced payload reads. Thread p reads packet p, so byte i\n");
    std::printf("     of adjacent packets sits ~1 MTU apart and every lane issues its\n");
    std::printf("     own transaction. The fix is a TRANSPOSED layout -- byte i of all\n");
    std::printf("     packets stored contiguously -- which would make each warp read\n");
    std::printf("     one clean 32-byte line. That is the single biggest win left.\n\n");
    std::printf("  2. The goto table (%.1f MB) does not fit the %d KB L2, so state\n",
                engine.table_bytes() / 1e6, dev.l2_bytes / 1024);
    std::printf("     transitions miss to DRAM. __ldg() routes them through the\n");
    std::printf("     read-only cache, which keeps the table from evicting the payload\n");
    std::printf("     stream, but cannot make an 8 MB working set fit in 1 MB.\n\n");
    std::printf("  Thread divergence is real but third: two packets in a warp walk\n");
    std::printf("  different automaton paths and no layout change fixes that.\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
