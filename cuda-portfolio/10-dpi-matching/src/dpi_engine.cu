// Aho-Corasick DPI engine -- implementation.

#include "dpi_engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cstring>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace dpi {
namespace {

constexpr int ALPHABET = 256;

// ---------------------------------------------------------------------------
// One thread per packet.
//
// Divergence is inherent here: two packets in the same warp walk different
// paths through the automaton, and nothing can make them agree. What CAN be
// controlled is memory traffic, so the transition table is read through
// __ldg(). The table is read-only for the whole kernel and shared by every
// thread, which is exactly the case the read-only data cache exists for -- it
// does not pollute L1 with the payload stream, and it broadcasts efficiently
// when lanes do happen to hit the same state.
// ---------------------------------------------------------------------------
__global__ void k_scan_bitmask(const unsigned char* __restrict__ data,
                               const int* __restrict__ offsets, int n_packets,
                               const int* __restrict__ goto_tbl,
                               const unsigned int* __restrict__ out_mask,
                               int words_per_packet,
                               unsigned int* __restrict__ result) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_packets) return;

    const int begin = offsets[p];
    const int end = offsets[p + 1];
    unsigned int* out = result + (size_t)p * words_per_packet;

    int state = 0;
    for (int i = begin; i < end; ++i) {
        const unsigned char c = data[i];
        state = __ldg(&goto_tbl[state * ALPHABET + c]);

        // Output words for this state: a bitmask of which patterns end here.
        // Walking a fail-link output list instead would add a data-dependent
        // loop to every byte; precomputing the union costs table space and
        // removes the branch entirely.
        for (int w = 0; w < words_per_packet; ++w) {
            unsigned int bits = __ldg(&out_mask[(size_t)state * words_per_packet + w]);
            if (bits) atomicOr(&out[w], bits);
        }
    }
}

__global__ void k_scan_detailed(const unsigned char* __restrict__ data,
                                const int* __restrict__ offsets, int n_packets,
                                const int* __restrict__ goto_tbl,
                                const unsigned int* __restrict__ out_mask,
                                int words_per_packet, int n_patterns,
                                Match* __restrict__ matches, int max_matches,
                                int* __restrict__ match_count) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_packets) return;

    const int begin = offsets[p];
    const int end = offsets[p + 1];

    int state = 0;
    for (int i = begin; i < end; ++i) {
        const unsigned char c = data[i];
        state = __ldg(&goto_tbl[state * ALPHABET + c]);

        for (int w = 0; w < words_per_packet; ++w) {
            unsigned int bits = __ldg(&out_mask[(size_t)state * words_per_packet + w]);
            while (bits) {
                const int b = __ffs(bits) - 1;
                bits &= bits - 1;
                const int pat = w * 32 + b;
                if (pat >= n_patterns) continue;
                const int slot = atomicAdd(match_count, 1);
                if (slot < max_matches) {
                    matches[slot].packet = p;
                    matches[slot].pattern = pat;
                    matches[slot].end_offset = i - begin + 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side Aho-Corasick construction.
// ---------------------------------------------------------------------------
struct Automaton {
    std::vector<int> go;                  // n_states * 256, already fail-resolved
    std::vector<unsigned int> out;        // n_states * words
    int n_states = 0;
    int words = 0;
};

Automaton build(const std::vector<std::string>& patterns) {
    const int np = (int)patterns.size();
    const int words = (np + 31) / 32;

    // Trie, one node per distinct prefix.
    std::vector<std::array<int, ALPHABET>> trie;
    trie.push_back({});
    trie[0].fill(-1);
    std::vector<unsigned int> out(words, 0u);

    for (int pi = 0; pi < np; ++pi) {
        const std::string& s = patterns[pi];
        int cur = 0;
        for (unsigned char c : s) {
            if (trie[cur][c] < 0) {
                std::array<int, ALPHABET> row;
                row.fill(-1);
                trie.push_back(row);
                out.insert(out.end(), words, 0u);
                trie[cur][c] = (int)trie.size() - 1;
            }
            cur = trie[cur][c];
        }
        out[(size_t)cur * words + pi / 32] |= (1u << (pi % 32));
    }

    const int ns = (int)trie.size();
    std::vector<int> fail(ns, 0);
    std::vector<int> go((size_t)ns * ALPHABET, 0);

    // BFS: build fail links and flatten to a dense goto table in one pass, so
    // the device never has to follow a fail link at runtime.
    std::queue<int> q;
    for (int c = 0; c < ALPHABET; ++c) {
        int nxt = trie[0][c];
        if (nxt < 0) {
            go[c] = 0;
        } else {
            go[c] = nxt;
            fail[nxt] = 0;
            q.push(nxt);
        }
    }
    while (!q.empty()) {
        const int s = q.front();
        q.pop();
        // A state inherits every output its fail-chain carries, so the device
        // reads one mask per state instead of walking the chain per byte.
        for (int w = 0; w < words; ++w)
            out[(size_t)s * words + w] |= out[(size_t)fail[s] * words + w];

        for (int c = 0; c < ALPHABET; ++c) {
            const int nxt = trie[s][c];
            if (nxt < 0) {
                go[(size_t)s * ALPHABET + c] = go[(size_t)fail[s] * ALPHABET + c];
            } else {
                go[(size_t)s * ALPHABET + c] = nxt;
                fail[nxt] = go[(size_t)fail[s] * ALPHABET + c];
                q.push(nxt);
            }
        }
    }

    Automaton a;
    a.go = std::move(go);
    a.out = std::move(out);
    a.n_states = ns;
    a.words = words;
    return a;
}

}  // namespace

// ---------------------------------------------------------------------------
struct DpiEngine::Impl {
    std::vector<std::string> patterns;
    Automaton au;

    int* d_go = nullptr;
    unsigned int* d_out = nullptr;
    std::size_t table_bytes = 0;

    // Owns every resource, so a constructor that throws partway through
    // releases what it had already acquired. A class destructor never runs
    // for an object whose constructor threw. Every release is null-safe.
    ~Impl() {
        cudaFree(d_go);
        cudaFree(d_out);
    }
};

DpiEngine::DpiEngine(const std::vector<std::string>& patterns) : impl_(new Impl) {
    try {
        if (patterns.empty()) throw std::invalid_argument("need at least one pattern");
        for (const auto& p : patterns) {
            if (p.empty()) throw std::invalid_argument("patterns must be non-empty");
            if (p.size() > 255) throw std::invalid_argument("pattern longer than 255 bytes");
        }
        impl_->patterns = patterns;
        impl_->au = build(patterns);

        CU_CHECK(cudaMalloc(&impl_->d_go, impl_->au.go.size() * sizeof(int)));
        CU_CHECK(cudaMemcpy(impl_->d_go, impl_->au.go.data(), impl_->au.go.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
        CU_CHECK(cudaMalloc(&impl_->d_out, impl_->au.out.size() * sizeof(unsigned int)));
        CU_CHECK(cudaMemcpy(impl_->d_out, impl_->au.out.data(),
                            impl_->au.out.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

        impl_->table_bytes = impl_->au.go.size() * sizeof(int) +
                             impl_->au.out.size() * sizeof(unsigned int);
    } catch (...) {
        delete impl_;   // releases anything acquired before the throw
        impl_ = nullptr;
        throw;
    }
}

DpiEngine::~DpiEngine() { delete impl_; }

int DpiEngine::num_patterns() const { return (int)impl_->patterns.size(); }
int DpiEngine::num_states() const { return impl_->au.n_states; }
int DpiEngine::words_per_packet() const { return impl_->au.words; }
std::size_t DpiEngine::table_bytes() const { return impl_->table_bytes; }

std::vector<std::uint32_t> DpiEngine::scan_bitmask(const PacketBatch& batch,
                                                   MatchStats* stats) {
    const int n = batch.size();
    const int words = impl_->au.words;
    std::vector<std::uint32_t> result((std::size_t)std::max(n, 1) * words, 0u);
    if (n == 0) return result;

    unsigned char* d_data = nullptr;
    int* d_off = nullptr;
    unsigned int* d_res = nullptr;
    CU_CHECK(cudaMalloc(&d_data, batch.data.size()));
    CU_CHECK(cudaMalloc(&d_off, batch.offsets.size() * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_res, result.size() * sizeof(unsigned int)));
    CU_CHECK(cudaMemcpy(d_data, batch.data.data(), batch.data.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_off, batch.offsets.data(), batch.offsets.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(d_res, 0, result.size() * sizeof(unsigned int)));

    const int T = 128;
    cu::EventTimer t;
    t.start();
    k_scan_bitmask<<<(n + T - 1) / T, T>>>(d_data, d_off, n, impl_->d_go, impl_->d_out,
                                           words, d_res);
    CU_CHECK_KERNEL();
    const float ms = t.stop();

    CU_CHECK(cudaMemcpy(result.data(), d_res, result.size() * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
    cudaFree(d_data); cudaFree(d_off); cudaFree(d_res);

    if (stats) {
        stats->kernel_ms = ms;
        stats->packets_scanned = n;
        stats->bytes_scanned = (std::int64_t)batch.data.size();
        stats->total_matches = 0;
        for (auto w : result) stats->total_matches += std::bitset<32>(w).count();
    }
    return result;
}

std::vector<std::uint32_t> DpiEngine::scan_bitmask_zerocopy(const PacketBatch& batch,
                                                            MatchStats* stats) {
    const int n = batch.size();
    const int words = impl_->au.words;
    std::vector<std::uint32_t> result((std::size_t)std::max(n, 1) * words, 0u);
    if (n == 0) return result;

    // Mapped (pinned + device-visible) host memory: the kernel reads the
    // payload straight over PCIe. No staging buffer, no cudaMemcpy -- which is
    // what a live capture ring wants, since the NIC is already writing here.
    unsigned char* h_data = nullptr;
    unsigned char* d_data = nullptr;
    int* h_off = nullptr;
    int* d_off = nullptr;
    CU_CHECK(cudaHostAlloc(&h_data, batch.data.size(), cudaHostAllocMapped));
    CU_CHECK(cudaHostGetDevicePointer((void**)&d_data, h_data, 0));
    CU_CHECK(cudaHostAlloc(&h_off, batch.offsets.size() * sizeof(int), cudaHostAllocMapped));
    CU_CHECK(cudaHostGetDevicePointer((void**)&d_off, h_off, 0));
    std::memcpy(h_data, batch.data.data(), batch.data.size());
    std::memcpy(h_off, batch.offsets.data(), batch.offsets.size() * sizeof(int));

    unsigned int* d_res = nullptr;
    CU_CHECK(cudaMalloc(&d_res, result.size() * sizeof(unsigned int)));
    CU_CHECK(cudaMemset(d_res, 0, result.size() * sizeof(unsigned int)));

    const int T = 128;
    cu::EventTimer t;
    t.start();
    k_scan_bitmask<<<(n + T - 1) / T, T>>>(d_data, d_off, n, impl_->d_go, impl_->d_out,
                                           words, d_res);
    CU_CHECK_KERNEL();
    const float ms = t.stop();

    CU_CHECK(cudaMemcpy(result.data(), d_res, result.size() * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
    cudaFreeHost(h_data); cudaFreeHost(h_off); cudaFree(d_res);

    if (stats) {
        stats->kernel_ms = ms;
        stats->packets_scanned = n;
        stats->bytes_scanned = (std::int64_t)batch.data.size();
        stats->total_matches = 0;
        for (auto w : result) stats->total_matches += std::bitset<32>(w).count();
    }
    return result;
}

std::vector<Match> DpiEngine::scan_detailed(const PacketBatch& batch, int max_matches,
                                            MatchStats* stats) {
    const int n = batch.size();
    if (n == 0 || max_matches <= 0) return {};
    const int words = impl_->au.words;

    unsigned char* d_data = nullptr;
    int* d_off = nullptr;
    Match* d_match = nullptr;
    int* d_count = nullptr;
    CU_CHECK(cudaMalloc(&d_data, batch.data.size()));
    CU_CHECK(cudaMalloc(&d_off, batch.offsets.size() * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_match, (std::size_t)max_matches * sizeof(Match)));
    CU_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CU_CHECK(cudaMemcpy(d_data, batch.data.data(), batch.data.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_off, batch.offsets.data(), batch.offsets.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    const int T = 128;
    cu::EventTimer t;
    t.start();
    k_scan_detailed<<<(n + T - 1) / T, T>>>(d_data, d_off, n, impl_->d_go, impl_->d_out,
                                            words, (int)impl_->patterns.size(),
                                            d_match, max_matches, d_count);
    CU_CHECK_KERNEL();
    const float ms = t.stop();

    int count = 0;
    CU_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    const int kept = std::min(count, max_matches);
    std::vector<Match> out(kept);
    if (kept > 0)
        CU_CHECK(cudaMemcpy(out.data(), d_match, (std::size_t)kept * sizeof(Match),
                            cudaMemcpyDeviceToHost));

    cudaFree(d_data); cudaFree(d_off); cudaFree(d_match); cudaFree(d_count);

    if (stats) {
        stats->kernel_ms = ms;
        stats->packets_scanned = n;
        stats->bytes_scanned = (std::int64_t)batch.data.size();
        stats->total_matches = count;   // the true count, even if truncated
    }
    // Sorted so comparisons against the CPU reference are order-independent:
    // GPU threads append via atomicAdd, so arrival order is nondeterministic.
    std::sort(out.begin(), out.end(), [](const Match& a, const Match& b) {
        if (a.packet != b.packet) return a.packet < b.packet;
        if (a.end_offset != b.end_offset) return a.end_offset < b.end_offset;
        return a.pattern < b.pattern;
    });
    return out;
}

std::vector<Match> DpiEngine::scan_cpu(const std::vector<std::string>& patterns,
                                       const PacketBatch& batch) {
    // Deliberately naive: a direct substring search per pattern. Sharing the
    // Aho-Corasick construction with the code under test would let a bug in
    // the automaton hide from its own reference.
    std::vector<Match> out;
    const int n = batch.size();
    for (int p = 0; p < n; ++p) {
        const int begin = batch.offsets[p];
        const int len = batch.offsets[p + 1] - begin;
        const char* hay = reinterpret_cast<const char*>(batch.data.data()) + begin;
        for (int pi = 0; pi < (int)patterns.size(); ++pi) {
            const std::string& pat = patterns[pi];
            const int plen = (int)pat.size();
            for (int i = 0; i + plen <= len; ++i) {
                if (std::memcmp(hay + i, pat.data(), plen) == 0)
                    out.push_back(Match{p, pi, i + plen});
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const Match& a, const Match& b) {
        if (a.packet != b.packet) return a.packet < b.packet;
        if (a.end_offset != b.end_offset) return a.end_offset < b.end_offset;
        return a.pattern < b.pattern;
    });
    return out;
}

// ---------------------------------------------------------------------------
PacketBatch make_traffic(int n_packets, int min_len, int max_len,
                         const std::vector<std::string>& patterns,
                         double inject_probability, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> len_dist(min_len, max_len);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<int> pat_dist(0, (int)patterns.size() - 1);

    PacketBatch b;
    b.offsets.push_back(0);
    for (int p = 0; p < n_packets; ++p) {
        const int len = len_dist(rng);
        std::vector<std::uint8_t> pkt(len);
        // Bytes 'a'..'z' so random noise cannot accidentally spell a pattern
        // built from a disjoint alphabet, keeping injected ground truth exact.
        for (auto& c : pkt) c = (std::uint8_t)('a' + (rng() % 26));

        if (!patterns.empty() && u01(rng) < inject_probability) {
            const std::string& pat = patterns[pat_dist(rng)];
            if ((int)pat.size() <= len) {
                const int pos = (int)(rng() % (unsigned)(len - pat.size() + 1));
                std::memcpy(pkt.data() + pos, pat.data(), pat.size());
            }
        }
        b.data.insert(b.data.end(), pkt.begin(), pkt.end());
        b.offsets.push_back((std::int32_t)b.data.size());
    }
    return b;
}

}  // namespace dpi
