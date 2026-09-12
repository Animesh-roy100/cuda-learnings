#pragma once
//
// Deep packet inspection: Aho-Corasick multi-pattern matching on the GPU.
// No CUDA syntax in this header.
//
// Aho-Corasick finds ALL occurrences of ALL patterns in one pass over the
// input, in O(len) regardless of how many patterns there are. That property is
// what makes it the right algorithm here: a signature set has thousands of
// strings, and scanning the payload once per pattern would be hopeless.
//
// The automaton is built on the host (a goto/fail/output construction), then
// flattened into a dense state-transition matrix for the device: 256 entries
// per state, so a transition is one load instead of a fail-link walk.
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dpi {

struct Match {
    std::int32_t packet = 0;
    std::int32_t pattern = 0;
    std::int32_t end_offset = 0;   // index just past the last matched byte
};

struct MatchStats {
    std::int64_t total_matches = 0;
    std::int64_t packets_scanned = 0;
    std::int64_t bytes_scanned = 0;
    float kernel_ms = 0.0f;
};

// A batch of packets laid out back to back, with an offset table. Packets have
// wildly different lengths, so a fixed stride would waste most of the buffer.
struct PacketBatch {
    std::vector<std::uint8_t> data;
    std::vector<std::int32_t> offsets;   // size n+1, offsets[i]..offsets[i+1]
    int size() const { return offsets.empty() ? 0 : (int)offsets.size() - 1; }
    std::size_t bytes() const { return data.size(); }
};

class DpiEngine {
public:
    // Patterns must be non-empty and at most 255 bytes. Case sensitive.
    explicit DpiEngine(const std::vector<std::string>& patterns);
    ~DpiEngine();
    DpiEngine(const DpiEngine&) = delete;
    DpiEngine& operator=(const DpiEngine&) = delete;

    // Which patterns matched anywhere in each packet, as a bitmask.
    // words_per_packet() 32-bit words per packet, built with atomicOr.
    //
    // A bitmask rather than a match list because the common case is "did this
    // packet hit any signature", and a list forces an atomic append plus an
    // unbounded output buffer. The mask is fixed size and lock-free.
    std::vector<std::uint32_t> scan_bitmask(const PacketBatch& batch,
                                            MatchStats* stats = nullptr);

    // Same scan, but recording every occurrence with its offset. Slower and
    // bounded by max_matches; used to verify the bitmask path.
    std::vector<Match> scan_detailed(const PacketBatch& batch, int max_matches,
                                     MatchStats* stats = nullptr);

    // Zero-copy variant: the payload stays in mapped host memory and the GPU
    // reads it over PCIe, with no cudaMemcpy at all. This is the shape a live
    // capture ring buffer wants.
    std::vector<std::uint32_t> scan_bitmask_zerocopy(const PacketBatch& batch,
                                                     MatchStats* stats = nullptr);

    int num_patterns() const;
    int num_states() const;
    int words_per_packet() const;
    std::size_t table_bytes() const;

    // Host reference implementation, for tests.
    static std::vector<Match> scan_cpu(const std::vector<std::string>& patterns,
                                       const PacketBatch& batch);

private:
    struct Impl;
    Impl* impl_;
};

// Synthetic traffic: random bytes with patterns injected at known positions.
PacketBatch make_traffic(int n_packets, int min_len, int max_len,
                         const std::vector<std::string>& patterns,
                         double inject_probability, unsigned seed);

}  // namespace dpi
