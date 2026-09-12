// Paged KV cache -- implementation.
//
// Storage is one flat VRAM slab of `total_pages` uniform pages. Page k holds
// K then V for `page_tokens` tokens. Which sequence owns which page lives in a
// host-side page table with a reference count per page; device kernels read
// the slab through an uploaded copy of that table.

#include "kv_cache.h"

#include <cuda_runtime.h>

#include <climits>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cu/check.hpp"

namespace llm {
namespace {

// dst[t*vals + i] = the i-th value of token t, read through the page table.
__global__ void k_kv_gather(const float* __restrict__ slab, const int* __restrict__ table,
                            float* __restrict__ dst, int total, int vals, int page_tokens,
                            size_t floats_per_page, size_t base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int t = idx / vals, i = idx % vals;
    size_t page = static_cast<size_t>(table[t / page_tokens]);
    dst[idx] = slab[page * floats_per_page + base + static_cast<size_t>(t % page_tokens) * vals + i];
}

}  // namespace

struct PagedKvCache::Impl {
    KvCacheConfig cfg;
    float* slab = nullptr;

    std::size_t vals_per_token = 0;   // n_kv_heads * head_dim
    std::size_t floats_per_page = 0;  // page_tokens * vals_per_token * 2 (K and V)

    mutable std::mutex mu;
    std::vector<int> free_list;
    std::vector<int> refcount;        // sequences holding each page

    struct Seq {
        bool live = false;
        int length = 0;                          // complete tokens
        int next_layer = 0;                      // layer the next append must target
        std::vector<std::vector<int>> pages;     // [layer][page slot] -> page id
        std::vector<int> reserved;               // [layer] page reserved for this token, or -1
    };
    std::vector<Seq> seqs;

    float* page_ptr(int page) const { return slab + static_cast<std::size_t>(page) * floats_per_page; }
    float* key_ptr(int page, int slot) const {
        return page_ptr(page) + static_cast<std::size_t>(slot) * vals_per_token;
    }
    float* val_ptr(int page, int slot) const {
        return page_ptr(page) + floats_per_page / 2 + static_cast<std::size_t>(slot) * vals_per_token;
    }

    void check_seq(int s) const {
        if (s < 0 || s >= static_cast<int>(seqs.size()) || !seqs[s].live)
            throw std::out_of_range("PagedKvCache: invalid sequence id " + std::to_string(s));
    }
    void check_layer(int l) const {
        if (l < 0 || l >= cfg.n_layers)
            throw std::out_of_range("PagedKvCache: invalid layer " + std::to_string(l));
    }

    void release(int page) {
        if (--refcount[page] == 0) free_list.push_back(page);
    }

    void gather(int seq, int layer, float* dst, bool values) const {
        const Seq& s = seqs[seq];
        if (!dst || s.length == 0) return;
        const std::size_t total = static_cast<std::size_t>(s.length) * vals_per_token;
        if (total > static_cast<std::size_t>(INT_MAX))
            throw std::invalid_argument("PagedKvCache::gather_device: sequence too long for one launch");
        const int T = 256;
        const std::size_t blocks = (total + T - 1) / T;
        cudaDeviceProp p{};
        int dev = 0;
        CU_CHECK(cudaGetDevice(&dev));
        CU_CHECK(cudaGetDeviceProperties(&p, dev));
        if (blocks > static_cast<std::size_t>(p.maxGridSize[0]) || T > p.maxThreadsPerBlock)
            throw std::runtime_error("PagedKvCache::gather_device: launch exceeds the device's grid");

        const auto& table = s.pages[layer];
        const std::size_t table_bytes = table.size() * sizeof(int);
        int* d_table = nullptr;
        CU_CHECK(cudaMalloc(&d_table, table_bytes));
        try {
            CU_CHECK(cudaMemcpy(d_table, table.data(), table_bytes, cudaMemcpyHostToDevice));
            k_kv_gather<<<static_cast<int>(blocks), T>>>(slab, d_table, dst, static_cast<int>(total),
                                                          static_cast<int>(vals_per_token), cfg.page_tokens,
                                                          floats_per_page, values ? floats_per_page / 2 : 0);
            CU_CHECK_KERNEL();
        } catch (...) {
            cudaFree(d_table);
            throw;
        }
        cudaFree(d_table);
    }

    std::vector<float> gather_host(int seq, int layer, bool values) const {
        const Seq& s = seqs[seq];
        std::vector<float> out(static_cast<std::size_t>(s.length) * vals_per_token);
        for (int t = 0; t < s.length; ++t) {
            int page = s.pages[layer][t / cfg.page_tokens];
            int slot = t % cfg.page_tokens;
            CU_CHECK(cudaMemcpy(out.data() + static_cast<std::size_t>(t) * vals_per_token,
                                values ? val_ptr(page, slot) : key_ptr(page, slot),
                                vals_per_token * sizeof(float), cudaMemcpyDeviceToHost));
        }
        return out;
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
        if (cfg.max_context < 0) throw std::invalid_argument("PagedKvCache: max_context must be >= 0");

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
        // Token positions are ints: the most tokens one sequence could hold
        // must be one.
        if (mul(cfg.page_tokens, cfg.total_pages) > static_cast<std::size_t>(INT_MAX))
            throw std::invalid_argument("PagedKvCache: page_tokens * total_pages exceeds INT_MAX");

        impl_->cfg = cfg;
        impl_->vals_per_token = vals;
        impl_->floats_per_page = fpp;

        CU_CHECK(cudaMalloc(&impl_->slab, bytes));

        impl_->refcount.assign(cfg.total_pages, 0);
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
    std::lock_guard<std::mutex> lock(impl_->mu);
    Impl::Seq s;
    s.live = true;
    s.pages.resize(impl_->cfg.n_layers);
    s.reserved.assign(impl_->cfg.n_layers, -1);

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

int PagedKvCache::fork_sequence(int seq) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->check_seq(seq);
    if (impl_->seqs[seq].next_layer != 0)
        throw KvOrderError("PagedKvCache: fork of sequence " + std::to_string(seq) +
                           " in the middle of a token (next layer " +
                           std::to_string(impl_->seqs[seq].next_layer) + ")");
    Impl::Seq child;
    child.live = true;
    child.length = impl_->seqs[seq].length;
    child.pages = impl_->seqs[seq].pages;
    child.reserved.assign(impl_->cfg.n_layers, -1);
    for (const auto& layer : child.pages)
        for (int p : layer) ++impl_->refcount[p];

    for (std::size_t i = 0; i < impl_->seqs.size(); ++i)
        if (!impl_->seqs[i].live) {
            impl_->seqs[i] = std::move(child);
            return static_cast<int>(i);
        }
    impl_->seqs.push_back(std::move(child));
    return static_cast<int>(impl_->seqs.size() - 1);
}

void PagedKvCache::free_sequence(int seq) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->check_seq(seq);
    auto& s = impl_->seqs[seq];
    // Every page returns to the shared pool once no sequence holds it. Because
    // pages are uniform, they are immediately usable by any other sequence --
    // this is exactly why paging removes external fragmentation.
    for (auto& layer_pages : s.pages)
        for (int p : layer_pages) impl_->release(p);
    for (int p : s.reserved)
        if (p >= 0) impl_->free_list.push_back(p);
    s = Impl::Seq{};
}

void PagedKvCache::append(int seq, int layer, const std::vector<float>& k, const std::vector<float>& v) {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto& I = *impl_;
    I.check_seq(seq);
    I.check_layer(layer);
    if (k.size() != I.vals_per_token || v.size() != I.vals_per_token)
        throw std::invalid_argument("PagedKvCache: k/v must be n_kv_heads*head_dim long");

    auto& s = I.seqs[seq];
    if (layer != s.next_layer)
        throw KvOrderError("PagedKvCache: sequence " + std::to_string(seq) + " token " +
                           std::to_string(s.length) + ": append to layer " + std::to_string(layer) +
                           ", expected layer " + std::to_string(s.next_layer));

    const int token = s.length;
    const int slot = token % I.cfg.page_tokens;
    const int page_index = token / I.cfg.page_tokens;

    if (layer == 0) {
        if (I.cfg.max_context > 0 && token >= I.cfg.max_context)
            throw KvContextFull("PagedKvCache: sequence " + std::to_string(seq) + " is at its context limit of " +
                                std::to_string(I.cfg.max_context) + " tokens");
        // Reserve this token's pages in every layer now, all or none: a fresh
        // page where the token starts one, a copy where the page is shared.
        int need = 0;
        for (int l = 0; l < I.cfg.n_layers; ++l) {
            const auto& pages = s.pages[l];
            if (static_cast<int>(pages.size()) <= page_index || I.refcount[pages[page_index]] > 1) ++need;
        }
        if (need > static_cast<int>(I.free_list.size()))
            throw KvCacheExhausted("PagedKvCache: out of pages: sequence " + std::to_string(seq) + " token " +
                                   std::to_string(token) + " needs " + std::to_string(need) + ", " +
                                   std::to_string(I.free_list.size()) + " free (raise total_pages)");
        for (int l = 0; l < I.cfg.n_layers; ++l) {
            const auto& pages = s.pages[l];
            if (static_cast<int>(pages.size()) <= page_index || I.refcount[pages[page_index]] > 1) {
                s.reserved[l] = I.free_list.back();
                I.free_list.pop_back();
            }
        }
    }

    auto& pages = s.pages[layer];
    const bool fresh = static_cast<int>(pages.size()) <= page_index;
    const bool shared = !fresh && I.refcount[pages[page_index]] > 1;
    int& r = s.reserved[layer];
    if (fresh || shared) {
        if (r < 0) throw std::logic_error("PagedKvCache: internal error: no page reserved");
        if (shared)   // copy the tokens already in the page, then write the copy
            CU_CHECK(cudaMemcpy(I.page_ptr(r), I.page_ptr(pages[page_index]), I.floats_per_page * sizeof(float),
                                cudaMemcpyDeviceToDevice));
        if (fresh) pages.push_back(r);
        else {
            I.release(pages[page_index]);
            pages[page_index] = r;
        }
        I.refcount[r] = 1;
        r = -1;
    } else if (r >= 0) {
        // Reserved for a copy, but the other holder has since copied away.
        I.free_list.push_back(r);
        r = -1;
    }
    const int page = pages[page_index];

    CU_CHECK(cudaMemcpy(I.key_ptr(page, slot), k.data(), I.vals_per_token * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(I.val_ptr(page, slot), v.data(), I.vals_per_token * sizeof(float), cudaMemcpyHostToDevice));

    if (layer == I.cfg.n_layers - 1) {
        s.next_layer = 0;
        ++s.length;
    } else {
        s.next_layer = layer + 1;
    }
}

int PagedKvCache::next_layer(int seq) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->check_seq(seq);
    return impl_->seqs[seq].next_layer;
}

std::vector<float> PagedKvCache::gather_keys(int seq, int layer) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->check_seq(seq);
    impl_->check_layer(layer);
    return impl_->gather_host(seq, layer, false);
}

std::vector<float> PagedKvCache::gather_values(int seq, int layer) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->check_seq(seq);
    impl_->check_layer(layer);
    return impl_->gather_host(seq, layer, true);
}

void PagedKvCache::gather_device(int seq, int layer, float* keys_out, float* values_out) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->check_seq(seq);
    impl_->check_layer(layer);
    impl_->gather(seq, layer, keys_out, false);
    impl_->gather(seq, layer, values_out, true);
}

std::vector<int> PagedKvCache::page_table(int seq, int layer) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->check_seq(seq);
    impl_->check_layer(layer);
    return impl_->seqs[seq].pages[layer];
}

KvDeviceLayout PagedKvCache::device_layout() const {
    return {impl_->slab, impl_->floats_per_page, impl_->vals_per_token, impl_->cfg.page_tokens};
}

int PagedKvCache::page_refcount(int page) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (page < 0 || page >= impl_->cfg.total_pages)
        throw std::out_of_range("PagedKvCache: invalid page " + std::to_string(page));
    return impl_->refcount[page];
}

int PagedKvCache::length(int seq) const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->check_seq(seq);
    return impl_->seqs[seq].length;
}

int PagedKvCache::free_pages() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return static_cast<int>(impl_->free_list.size());
}
int PagedKvCache::pages_in_use() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->cfg.total_pages - static_cast<int>(impl_->free_list.size());
}
int PagedKvCache::sequence_count() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    int n = 0;
    for (const auto& s : impl_->seqs) n += s.live ? 1 : 0;
    return n;
}
std::size_t PagedKvCache::bytes_per_page() const { return impl_->floats_per_page * sizeof(float); }
std::size_t PagedKvCache::total_bytes() const { return bytes_per_page() * impl_->cfg.total_pages; }
const KvCacheConfig& PagedKvCache::config() const { return impl_->cfg; }

}  // namespace llm
