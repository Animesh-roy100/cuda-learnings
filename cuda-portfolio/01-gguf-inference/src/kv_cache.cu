// Paged KV cache -- implementation.
//
// Storage is one flat VRAM slab of `total_pages` uniform pages. Page k holds
// K then V for `page_tokens` tokens. Which sequence owns which page lives in a
// host-side page table; the device never needs to know.

#include "kv_cache.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cu/check.hpp"

namespace llm {

struct PagedKvCache::Impl {
    KvCacheConfig cfg;
    float* slab = nullptr;

    std::size_t vals_per_token = 0;   // n_kv_heads * head_dim
    std::size_t floats_per_page = 0;  // page_tokens * vals_per_token * 2 (K and V)

    std::vector<int> free_list;

    struct Seq {
        bool live = false;
        int length = 0;                          // tokens appended per layer
        std::vector<std::vector<int>> pages;     // [layer][page slot] -> page id
    };
    std::vector<Seq> seqs;

    float* page_ptr(int page) const { return slab + static_cast<std::size_t>(page) * floats_per_page; }
    float* key_ptr(int page, int slot) const {
        return page_ptr(page) + static_cast<std::size_t>(slot) * vals_per_token;
    }
    float* val_ptr(int page, int slot) const {
        return page_ptr(page) + floats_per_page / 2 +
               static_cast<std::size_t>(slot) * vals_per_token;
    }

    int alloc_page() {
        if (free_list.empty())
            throw std::runtime_error("PagedKvCache: out of pages (raise total_pages)");
        int p = free_list.back();
        free_list.pop_back();
        return p;
    }

    void check_seq(int s) const {
        if (s < 0 || s >= static_cast<int>(seqs.size()) || !seqs[s].live)
            throw std::out_of_range("PagedKvCache: invalid sequence id " + std::to_string(s));
    }
    void check_layer(int l) const {
        if (l < 0 || l >= cfg.n_layers)
            throw std::out_of_range("PagedKvCache: invalid layer " + std::to_string(l));
    }

    // Owns every resource, so a constructor that throws partway through
    // releases what it had already acquired. A class destructor never runs
    // for an object whose constructor threw. Every release is null-safe.
    ~Impl() {
        cudaFree(slab);
    }
};

PagedKvCache::PagedKvCache(const KvCacheConfig& cfg) : impl_(new Impl) {
    try {
        if (cfg.n_layers <= 0 || cfg.n_kv_heads <= 0 || cfg.head_dim <= 0 ||
            cfg.page_tokens <= 0 || cfg.total_pages <= 0)
            throw std::invalid_argument("PagedKvCache: all config fields must be positive");

        // Every factor is a positive int, but their product is not bounded by
        // anything. Unchecked, a large enough configuration wraps size_t -- to
        // exactly zero for 65536 heads x 65536 dims x 65536 tokens x 8192
        // pages -- and cudaMalloc(&p, 0) SUCCEEDS with a null slab.
        const std::size_t limit = std::numeric_limits<std::size_t>::max();
        auto mul = [&](std::size_t a, std::size_t b) {
            if (b != 0 && a > limit / b)
                throw std::invalid_argument("PagedKvCache: configuration size overflows");
            return a * b;
        };
        const std::size_t vals = mul(cfg.n_kv_heads, cfg.head_dim);
        const std::size_t fpp = mul(mul(cfg.page_tokens, vals), 2);
        const std::size_t bytes = mul(mul(fpp, cfg.total_pages), sizeof(float));

        impl_->cfg = cfg;
        impl_->vals_per_token = vals;
        impl_->floats_per_page = fpp;

        CU_CHECK(cudaMalloc(&impl_->slab, bytes));

        impl_->free_list.reserve(cfg.total_pages);
        // Hand out low page ids first, so tests see deterministic ids.
        for (int p = cfg.total_pages - 1; p >= 0; --p) impl_->free_list.push_back(p);
    } catch (...) {
        delete impl_;   // releases anything acquired before the throw
        impl_ = nullptr;
        throw;
    }
}

PagedKvCache::~PagedKvCache() { delete impl_; }

int PagedKvCache::create_sequence() {
    Impl::Seq s;
    s.live = true;
    s.pages.resize(impl_->cfg.n_layers);

    // Reuse a dead slot if one exists, so ids stay compact.
    for (std::size_t i = 0; i < impl_->seqs.size(); ++i) {
        if (!impl_->seqs[i].live) {
            impl_->seqs[i] = std::move(s);
            return static_cast<int>(i);
        }
    }
    impl_->seqs.push_back(std::move(s));
    return static_cast<int>(impl_->seqs.size() - 1);
}

void PagedKvCache::free_sequence(int seq) {
    impl_->check_seq(seq);
    auto& s = impl_->seqs[seq];
    // Every page returns to the shared pool. Because pages are uniform, they
    // are immediately usable by any other sequence -- this is exactly why
    // paging removes external fragmentation.
    for (auto& layer_pages : s.pages)
        for (int p : layer_pages) impl_->free_list.push_back(p);
    s = Impl::Seq{};
}

void PagedKvCache::append(int seq, int layer, const std::vector<float>& k,
                          const std::vector<float>& v) {
    impl_->check_seq(seq);
    impl_->check_layer(layer);
    if (k.size() != impl_->vals_per_token || v.size() != impl_->vals_per_token)
        throw std::invalid_argument("PagedKvCache: k/v must be n_kv_heads*head_dim long");

    auto& s = impl_->seqs[seq];
    // Length is tracked once and applies to every layer: a token is appended to
    // all layers as it is processed.
    const int token = s.length;
    const int slot = token % impl_->cfg.page_tokens;
    const int page_index = token / impl_->cfg.page_tokens;

    auto& pages = s.pages[layer];
    if (static_cast<int>(pages.size()) <= page_index) pages.push_back(impl_->alloc_page());
    const int page = pages[page_index];

    CU_CHECK(cudaMemcpy(impl_->key_ptr(page, slot), k.data(),
                        impl_->vals_per_token * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(impl_->val_ptr(page, slot), v.data(),
                        impl_->vals_per_token * sizeof(float), cudaMemcpyHostToDevice));

    // Only the last layer advances the token counter.
    if (layer == impl_->cfg.n_layers - 1) ++s.length;
}

std::vector<float> PagedKvCache::gather_keys(int seq, int layer) const {
    impl_->check_seq(seq);
    impl_->check_layer(layer);
    const auto& s = impl_->seqs[seq];
    std::vector<float> out(static_cast<std::size_t>(s.length) * impl_->vals_per_token);
    for (int t = 0; t < s.length; ++t) {
        int page = s.pages[layer][t / impl_->cfg.page_tokens];
        int slot = t % impl_->cfg.page_tokens;
        CU_CHECK(cudaMemcpy(out.data() + static_cast<std::size_t>(t) * impl_->vals_per_token,
                            impl_->key_ptr(page, slot),
                            impl_->vals_per_token * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return out;
}

std::vector<float> PagedKvCache::gather_values(int seq, int layer) const {
    impl_->check_seq(seq);
    impl_->check_layer(layer);
    const auto& s = impl_->seqs[seq];
    std::vector<float> out(static_cast<std::size_t>(s.length) * impl_->vals_per_token);
    for (int t = 0; t < s.length; ++t) {
        int page = s.pages[layer][t / impl_->cfg.page_tokens];
        int slot = t % impl_->cfg.page_tokens;
        CU_CHECK(cudaMemcpy(out.data() + static_cast<std::size_t>(t) * impl_->vals_per_token,
                            impl_->val_ptr(page, slot),
                            impl_->vals_per_token * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return out;
}

int PagedKvCache::length(int seq) const {
    impl_->check_seq(seq);
    return impl_->seqs[seq].length;
}

int PagedKvCache::free_pages() const { return static_cast<int>(impl_->free_list.size()); }
int PagedKvCache::pages_in_use() const {
    return impl_->cfg.total_pages - static_cast<int>(impl_->free_list.size());
}
int PagedKvCache::sequence_count() const {
    int n = 0;
    for (const auto& s : impl_->seqs) n += s.live ? 1 : 0;
    return n;
}
std::size_t PagedKvCache::bytes_per_page() const {
    return impl_->floats_per_page * sizeof(float);
}
std::size_t PagedKvCache::total_bytes() const {
    return bytes_per_page() * impl_->cfg.total_pages;
}
const KvCacheConfig& PagedKvCache::config() const { return impl_->cfg; }

}  // namespace llm
