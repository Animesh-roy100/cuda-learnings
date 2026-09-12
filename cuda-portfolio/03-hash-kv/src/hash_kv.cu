// Lock-free GPU hash table -- implementation.
//
// Concurrency model: every mutation is a single 64-bit atomicCAS on one slot.
// Key and value are packed into that one word so they land together or not at
// all; with separate arrays a reader could observe a published key beside an
// uninitialised value.

#include "hash_kv.h"

#include <cuda_runtime.h>

#include <algorithm>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace kv {
namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 EMPTY_KEY = GpuHashTable::kEmptyKey;
constexpr u64 EMPTY_SLOT = static_cast<u64>(EMPTY_KEY) << 32;

__host__ __device__ __forceinline__ u64 make_slot(u32 k, u32 v) {
    return (static_cast<u64>(k) << 32) | v;
}
__host__ __device__ __forceinline__ u32 slot_key(u64 s) { return static_cast<u32>(s >> 32); }
__host__ __device__ __forceinline__ u32 slot_val(u64 s) { return static_cast<u32>(s & 0xFFFFFFFFu); }

// Murmur3 finalizer: a bijection on uint32. A weak hash clusters keys and
// destroys the probe-length guarantees both strategies depend on.
__host__ __device__ __forceinline__ u32 hash32(u32 x) {
    x ^= x >> 16; x *= 0x85ebca6bu;
    x ^= x >> 13; x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

__global__ void k_clear(u64* table, std::size_t cap) {
    std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < cap) table[i] = EMPTY_SLOT;
}

// ---------------------------------------------------------------------------
// Linear probing insert
// ---------------------------------------------------------------------------
__global__ void k_insert_linear(u64* __restrict__ table, u32 mask,
                                const u32* __restrict__ keys,
                                const u32* __restrict__ vals, int n,
                                unsigned long long* probe_sum,
                                unsigned long long* probe_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const u32 key = keys[i];
    const u64 want = make_slot(key, vals[i]);
    u32 slot = hash32(key) & mask;
    unsigned long long probes = 0;

    while (true) {
        ++probes;
        u64 cur = table[slot];

        if (slot_key(cur) == key) {                      // update in place
            if (atomicCAS(reinterpret_cast<unsigned long long*>(&table[slot]),
                          cur, want) == cur) break;
            continue;
        }
        if (cur == EMPTY_SLOT) {
            u64 old = atomicCAS(reinterpret_cast<unsigned long long*>(&table[slot]),
                                EMPTY_SLOT, want);
            if (old == EMPTY_SLOT) break;                // claimed
            if (slot_key(old) == key) continue;          // winner wrote our key
        }
        slot = (slot + 1) & mask;
    }
    if (probe_sum) atomicAdd(probe_sum, probes);
    if (probe_max) atomicMax(probe_max, probes);
}

// ---------------------------------------------------------------------------
// Robin Hood insert
//
// The idea: an entry's "wealth" is how far it sits from its home slot. When we
// meet an entry poorer than us (closer to home), we take the slot and carry
// the displaced entry onward. This equalises probe lengths -- the variance
// collapses, so the WORST lookup gets dramatically better even though the
// average barely moves.
// ---------------------------------------------------------------------------
__global__ void k_insert_robinhood(u64* __restrict__ table, u32 mask,
                                   const u32* __restrict__ keys,
                                   const u32* __restrict__ vals, int n,
                                   unsigned long long* probe_sum,
                                   unsigned long long* probe_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    u32 key = keys[i];
    u64 entry = make_slot(key, vals[i]);
    u32 slot = hash32(key) & mask;
    u32 dist = 0;
    unsigned long long probes = 0;

    while (true) {
        ++probes;
        u64 cur = table[slot];

        if (cur == EMPTY_SLOT) {
            u64 old = atomicCAS(reinterpret_cast<unsigned long long*>(&table[slot]),
                                EMPTY_SLOT, entry);
            if (old == EMPTY_SLOT) break;
            continue;                                    // lost race; re-read
        }
        if (slot_key(cur) == slot_key(entry)) {          // same key: overwrite
            if (atomicCAS(reinterpret_cast<unsigned long long*>(&table[slot]),
                          cur, entry) == cur) break;
            continue;
        }

        // How far is the resident entry from ITS home slot?
        u32 cur_home = hash32(slot_key(cur)) & mask;
        u32 cur_dist = (slot - cur_home) & mask;

        if (cur_dist < dist) {
            // Resident is richer than us: steal the slot, carry it onward.
            u64 old = atomicCAS(reinterpret_cast<unsigned long long*>(&table[slot]),
                                cur, entry);
            if (old == cur) {
                entry = cur;                             // now place the evictee
                dist = cur_dist;
            }
            continue;                                    // retry this slot
        }
        slot = (slot + 1) & mask;
        ++dist;
    }
    if (probe_sum) atomicAdd(probe_sum, probes);
    if (probe_max) atomicMax(probe_max, probes);
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------
__global__ void k_find_thread(const u64* __restrict__ table, u32 mask,
                              const u32* __restrict__ keys,
                              u32* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const u32 key = keys[i];
    u32 slot = hash32(key) & mask;
    u32 result = GpuHashTable::kNotFound;

    while (true) {
        u64 cur = table[slot];
        if (slot_key(cur) == key) { result = slot_val(cur); break; }
        if (cur == EMPTY_SLOT) break;        // empty slot terminates the chain
        slot = (slot + 1) & mask;
    }
    out[i] = result;
}

// One warp per query: 32 lanes read 32 consecutive slots as a single coalesced
// transaction, then __ballot_sync collapses "did anyone match?" into one mask.
//
// NOTE: there is deliberately NO probe-length cap. A fixed bound looks safe and
// is not -- at load factor 0.9+ linear probing forms clusters far longer than
// any constant you would pick, and a capped search silently reports a present
// key as missing. An empty slot is the only correct terminator.
__global__ void k_find_warp(const u64* __restrict__ table, u32 mask,
                            const u32* __restrict__ keys,
                            u32* __restrict__ out, int n) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= n) return;

    const u32 key = keys[warp_id];
    const u32 base = hash32(key) & mask;
    u32 result = GpuHashTable::kNotFound;

    for (int step = 0;; step += 32) {
        u32 slot = (base + step + lane) & mask;
        u64 cur = table[slot];

        unsigned hit = __ballot_sync(0xffffffffu, slot_key(cur) == key);
        unsigned empty = __ballot_sync(0xffffffffu, cur == EMPTY_SLOT);

        if (hit) {
            int src = __ffs(hit) - 1;                    // lowest matching lane
            result = __shfl_sync(0xffffffffu, slot_val(cur), src);
            break;
        }
        if (empty) break;
    }
    if (lane == 0) out[warp_id] = result;
}

__global__ void k_count_occupied(const u64* __restrict__ table, std::size_t cap,
                                 unsigned long long* out) {
    std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    unsigned long long n = 0;
    for (; i < cap; i += stride) n += (table[i] != EMPTY_SLOT) ? 1 : 0;

    for (int off = 16; off > 0; off >>= 1) n += __shfl_down_sync(0xffffffffu, n, off);
    __shared__ unsigned long long w[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) w[wid] = n;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        unsigned long long v = (lane < nw) ? w[lane] : 0ULL;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) atomicAdd(out, v);
    }
}

// Displacement of every stored key from its home slot, on the final table.
// This is the true lookup-cost metric, and unlike insert-loop iteration counts
// it is directly comparable between linear probing and Robin Hood.
__global__ void k_displacement(const u64* __restrict__ table, std::size_t cap, u32 mask,
                               unsigned long long* sum, unsigned long long* maxv,
                               unsigned long long* count) {
    std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    unsigned long long s = 0, c = 0, m = 0;
    for (; i < cap; i += stride) {
        u64 cur = table[i];
        if (cur == EMPTY_SLOT) continue;
        u32 home = hash32(slot_key(cur)) & mask;
        unsigned long long d = (static_cast<u32>(i) - home) & mask;
        s += d;
        ++c;
        if (d > m) m = d;
    }
    atomicAdd(sum, s);
    atomicAdd(count, c);
    atomicMax(maxv, m);
}

std::size_t round_up_pow2(std::size_t n) {
    std::size_t c = 1;
    while (c < n) c <<= 1;
    return c;
}

}  // namespace

// ---------------------------------------------------------------------------
struct GpuHashTable::Impl {
    u64* table = nullptr;
    std::size_t cap = 0;
    u32 mask = 0;
    Probe probe = Probe::Linear;
    InsertStats stats{};

    unsigned long long* d_scratch = nullptr;   // [sum, max, count]
};

GpuHashTable::GpuHashTable(std::size_t capacity, Probe probe) : impl_(new Impl) {
    impl_->cap = round_up_pow2(std::max<std::size_t>(capacity, 32));
    impl_->mask = static_cast<u32>(impl_->cap - 1);
    impl_->probe = probe;
    CU_CHECK(cudaMalloc(&impl_->table, impl_->cap * sizeof(u64)));
    CU_CHECK(cudaMalloc(&impl_->d_scratch, 3 * sizeof(unsigned long long)));
    clear();
}

GpuHashTable::~GpuHashTable() {
    if (impl_) {
        cudaFree(impl_->table);
        cudaFree(impl_->d_scratch);
        delete impl_;
    }
}

GpuHashTable::GpuHashTable(GpuHashTable&& o) noexcept : impl_(o.impl_) { o.impl_ = nullptr; }

GpuHashTable& GpuHashTable::operator=(GpuHashTable&& o) noexcept {
    if (this != &o) {
        if (impl_) {
            cudaFree(impl_->table);
            cudaFree(impl_->d_scratch);
            delete impl_;
        }
        impl_ = o.impl_;
        o.impl_ = nullptr;
    }
    return *this;
}

void GpuHashTable::clear() {
    const int T = 256;
    auto blocks = static_cast<unsigned>((impl_->cap + T - 1) / T);
    k_clear<<<blocks, T>>>(impl_->table, impl_->cap);
    CU_CHECK_KERNEL();
}

void GpuHashTable::insert(const std::vector<u32>& keys, const std::vector<u32>& values) {
    const int n = static_cast<int>(keys.size());
    if (n == 0) return;
    if (values.size() != keys.size()) throw std::runtime_error("keys/values size mismatch");

    u32 *d_k = nullptr, *d_v = nullptr;
    CU_CHECK(cudaMalloc(&d_k, sizeof(u32) * n));
    CU_CHECK(cudaMalloc(&d_v, sizeof(u32) * n));
    CU_CHECK(cudaMemcpy(d_k, keys.data(), sizeof(u32) * n, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_v, values.data(), sizeof(u32) * n, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(impl_->d_scratch, 0, 2 * sizeof(unsigned long long)));

    const int T = 256;
    const int B = (n + T - 1) / T;
    if (impl_->probe == Probe::RobinHood) {
        k_insert_robinhood<<<B, T>>>(impl_->table, impl_->mask, d_k, d_v, n,
                                     impl_->d_scratch, impl_->d_scratch + 1);
    } else {
        k_insert_linear<<<B, T>>>(impl_->table, impl_->mask, d_k, d_v, n,
                                  impl_->d_scratch, impl_->d_scratch + 1);
    }
    CU_CHECK_KERNEL();

    unsigned long long h[2] = {0, 0};
    CU_CHECK(cudaMemcpy(h, impl_->d_scratch, sizeof(h), cudaMemcpyDeviceToHost));
    impl_->stats.avg_probes = static_cast<double>(h[0]) / n;
    impl_->stats.max_probes = h[1];

    cudaFree(d_k);
    cudaFree(d_v);
}

std::vector<u32> GpuHashTable::find(const std::vector<u32>& keys, Lookup mode) const {
    const int n = static_cast<int>(keys.size());
    std::vector<u32> out(n);
    if (n == 0) return out;

    u32 *d_k = nullptr, *d_o = nullptr;
    CU_CHECK(cudaMalloc(&d_k, sizeof(u32) * n));
    CU_CHECK(cudaMalloc(&d_o, sizeof(u32) * n));
    CU_CHECK(cudaMemcpy(d_k, keys.data(), sizeof(u32) * n, cudaMemcpyHostToDevice));

    const int T = 256;
    if (mode == Lookup::WarpCoop) {
        const int warps_per_block = T / 32;
        const int B = (n + warps_per_block - 1) / warps_per_block;
        k_find_warp<<<B, T>>>(impl_->table, impl_->mask, d_k, d_o, n);
    } else {
        const int B = (n + T - 1) / T;
        k_find_thread<<<B, T>>>(impl_->table, impl_->mask, d_k, d_o, n);
    }
    CU_CHECK_KERNEL();

    CU_CHECK(cudaMemcpy(out.data(), d_o, sizeof(u32) * n, cudaMemcpyDeviceToHost));
    cudaFree(d_k);
    cudaFree(d_o);
    return out;
}

InsertStats GpuHashTable::last_insert_stats() const { return impl_->stats; }

Displacement GpuHashTable::displacement() const {
    CU_CHECK(cudaMemset(impl_->d_scratch, 0, 3 * sizeof(unsigned long long)));
    k_displacement<<<256, 256>>>(impl_->table, impl_->cap, impl_->mask,
                                 impl_->d_scratch, impl_->d_scratch + 1,
                                 impl_->d_scratch + 2);
    CU_CHECK_KERNEL();
    unsigned long long h[3] = {0, 0, 0};
    CU_CHECK(cudaMemcpy(h, impl_->d_scratch, sizeof(h), cudaMemcpyDeviceToHost));
    Displacement d;
    d.max = h[1];
    d.avg = (h[2] == 0) ? 0.0 : static_cast<double>(h[0]) / static_cast<double>(h[2]);
    return d;
}

std::size_t GpuHashTable::capacity() const { return impl_->cap; }

std::size_t GpuHashTable::size_hint() const {
    CU_CHECK(cudaMemset(impl_->d_scratch, 0, sizeof(unsigned long long)));
    k_count_occupied<<<256, 256>>>(impl_->table, impl_->cap, impl_->d_scratch);
    CU_CHECK_KERNEL();
    unsigned long long h = 0;
    CU_CHECK(cudaMemcpy(&h, impl_->d_scratch, sizeof(h), cudaMemcpyDeviceToHost));
    return static_cast<std::size_t>(h);
}

u32 GpuHashTable::hash(u32 x) { return hash32(x); }

}  // namespace kv
