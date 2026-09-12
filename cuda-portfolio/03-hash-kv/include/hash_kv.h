#pragma once
//
// Lock-free GPU hash table -- public interface.
//
// Deliberately free of CUDA syntax: no __global__, no <<<>>>, no cuda_runtime.h.
// Host code (and the test suite) compiles as plain C++20 against this header,
// and only the .cu translation unit needs nvcc. The pimpl hides every CUDA type.
//
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kv {

// Probe strategy. Both are lock-free and use the same 64-bit atomicCAS; they
// differ only in what happens on collision.
enum class Probe {
    Linear,      // classic: walk forward until an empty slot
    RobinHood,   // displace entries that are richer (closer to home) than us
};

enum class Lookup {
    PerThread,   // one thread walks the probe chain
    WarpCoop,    // 32 lanes test 32 slots at once, collapsed with __ballot_sync
};

// Work done by the insert kernel. For Robin Hood this counts loop iterations
// including carrying displaced entries onward, so it is NOT comparable across
// probe strategies -- use Displacement for that.
struct InsertStats {
    double avg_probes = 0.0;
    std::uint64_t max_probes = 0;
};

// Distance of each stored key from its home slot, measured on the final table.
// This is the metric that governs LOOKUP cost, and the one Robin Hood exists to
// improve: it barely moves the mean, but collapses the tail.
struct Displacement {
    double avg = 0.0;
    std::uint64_t max = 0;
};

class GpuHashTable {
public:
    static constexpr std::uint32_t kEmptyKey = 0xFFFFFFFFu;
    static constexpr std::uint32_t kNotFound = 0xFFFFFFFFu;

    // capacity is rounded up to a power of two so the modulo becomes a mask.
    explicit GpuHashTable(std::size_t capacity, Probe probe = Probe::Linear);
    ~GpuHashTable();

    GpuHashTable(const GpuHashTable&) = delete;
    GpuHashTable& operator=(const GpuHashTable&) = delete;
    GpuHashTable(GpuHashTable&&) noexcept;
    GpuHashTable& operator=(GpuHashTable&&) noexcept;

    void clear();

    // Bulk insert. Duplicate keys resolve to the last writer.
    void insert(const std::vector<std::uint32_t>& keys,
                const std::vector<std::uint32_t>& values);

    // Returns kNotFound for absent keys.
    std::vector<std::uint32_t> find(const std::vector<std::uint32_t>& keys,
                                    Lookup mode = Lookup::PerThread) const;

    InsertStats last_insert_stats() const;
    Displacement displacement() const;
    std::size_t capacity() const;
    std::size_t size_hint() const;   // slots currently occupied

    // The Murmur3 finalizer used internally. Exposed because it is a bijection
    // on uint32, which test code relies on to generate provably distinct keys.
    static std::uint32_t hash(std::uint32_t x);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace kv
