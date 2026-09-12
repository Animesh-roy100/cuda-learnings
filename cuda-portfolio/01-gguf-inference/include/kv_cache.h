#pragma once
//
// Paged KV cache.
//
// The problem it solves: a naive cache reserves max_context tokens per sequence
// up front. With 4 GB of VRAM that is ruinous -- you size for the worst case and
// then cannot fit a second sequence, even though most sequences are short.
// Growing a contiguous buffer instead fragments VRAM until an allocation that
// should fit does not.
//
// The fix (the vLLM idea): cut the cache into uniform PAGES and give each
// sequence a page table. Pages are interchangeable, so a freed page is
// immediately reusable by any sequence and external fragmentation cannot
// happen. The cost is one indirection per lookup.
//
// Rules this implementation enforces:
//
//   - Ordering. A token is appended to layer 0, then 1, ... then n_layers-1;
//     anything else is KvOrderError. next_layer() says what is expected.
//   - Context. With max_context set, appending token max_context is
//     KvContextFull, and nothing changes.
//   - Transactions. Every page a token will need, in every layer, is reserved
//     when its layer 0 is appended -- all or none. Exhaustion
//     (KvCacheExhausted) therefore happens before anything is written, never
//     part-way through a token.
//   - Sharing. fork_sequence shares every page with the parent; a page shared
//     by several sequences is copied before either writes into it.
//   - Device reads. Attention reads pages where they are: page_table() plus
//     device_layout() address them, and gather_device() collects a sequence
//     into device memory without a host round trip. gather_keys/values copy
//     to the host and exist for tests and debugging.
//   - Threads. Every member function is safe to call concurrently; calls are
//     serialized by one mutex. Pointers from device_layout() stay valid for the
//     cache's lifetime, but the pages behind a page table may be copied or
//     reused once another call runs.
//
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm {

struct KvCacheConfig {
    int n_layers = 1;
    int n_kv_heads = 1;      // GQA: usually fewer than the attention heads
    int head_dim = 64;
    int page_tokens = 16;    // tokens per page
    int total_pages = 256;   // the whole VRAM budget for the cache
    int max_context = 0;     // tokens per sequence; 0 = limited only by pages
};

class KvCacheExhausted : public std::runtime_error {
public:
    explicit KvCacheExhausted(const std::string& w) : std::runtime_error(w) {}
};
class KvContextFull : public std::runtime_error {
public:
    explicit KvContextFull(const std::string& w) : std::runtime_error(w) {}
};
class KvOrderError : public std::logic_error {
public:
    explicit KvOrderError(const std::string& w) : std::logic_error(w) {}
};

// Where a page lives on the device: page p's key for token slot s starts at
// slab + p*floats_per_page + s*vals_per_token; its value at
// slab + p*floats_per_page + floats_per_page/2 + s*vals_per_token.
struct KvDeviceLayout {
    const float* slab = nullptr;
    std::size_t floats_per_page = 0;
    std::size_t vals_per_token = 0;
    int page_tokens = 0;
};

class PagedKvCache {
public:
    explicit PagedKvCache(const KvCacheConfig& cfg);
    ~PagedKvCache();
    PagedKvCache(const PagedKvCache&) = delete;
    PagedKvCache& operator=(const PagedKvCache&) = delete;

    int create_sequence();
    // A new sequence sharing every token of `seq`. Only between tokens
    // (next_layer(seq) == 0); costs no pages until one side writes.
    int fork_sequence(int seq);
    void free_sequence(int seq);

    // k and v must each be n_kv_heads*head_dim long.
    void append(int seq, int layer, const std::vector<float>& k,
                const std::vector<float>& v);
    int next_layer(int seq) const;

    // All cached keys (or values) for one sequence and layer, in token order:
    // [length][n_kv_heads*head_dim]. Host copies -- for tests and debugging.
    std::vector<float> gather_keys(int seq, int layer) const;
    std::vector<float> gather_values(int seq, int layer) const;

    // The same, into caller-owned DEVICE memory of length(seq)*vals floats
    // each. Either pointer may be null. Runs a device kernel over the page
    // table; no host copy of the cache is made.
    void gather_device(int seq, int layer, float* keys_out, float* values_out) const;

    std::vector<int> page_table(int seq, int layer) const;
    KvDeviceLayout device_layout() const;
    int page_refcount(int page) const;

    int length(int seq) const;              // complete tokens (all layers)
    int pages_in_use() const;
    int free_pages() const;
    int sequence_count() const;

    std::size_t bytes_per_page() const;
    std::size_t total_bytes() const;
    const KvCacheConfig& config() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace llm
