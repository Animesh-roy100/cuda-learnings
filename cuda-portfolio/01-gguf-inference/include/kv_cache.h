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
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm {

struct KvCacheConfig {
    int n_layers = 1;
    int n_kv_heads = 1;      // GQA: usually fewer than the attention heads
    int head_dim = 64;
    int page_tokens = 16;    // tokens per page
    int total_pages = 256;   // the whole VRAM budget for the cache
};

class PagedKvCache {
public:
    explicit PagedKvCache(const KvCacheConfig& cfg);
    ~PagedKvCache();
    PagedKvCache(const PagedKvCache&) = delete;
    PagedKvCache& operator=(const PagedKvCache&) = delete;

    // Returns a sequence id, or throws if the cache is exhausted.
    int create_sequence();
    void free_sequence(int seq);

    // k and v must each be n_kv_heads*head_dim long. Allocates a page on demand.
    void append(int seq, int layer, const std::vector<float>& k,
                const std::vector<float>& v);

    // All cached keys (or values) for one sequence and layer, in token order:
    // [length][n_kv_heads*head_dim].
    std::vector<float> gather_keys(int seq, int layer) const;
    std::vector<float> gather_values(int seq, int layer) const;

    int length(int seq) const;              // tokens appended, per layer
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
