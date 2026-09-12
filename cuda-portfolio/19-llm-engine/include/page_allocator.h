#pragma once
//
// KV page bookkeeping: which pages are free and how many sequences hold each.
// Host-only and GPU-free, so it is tested (under sanitizers, in CI) without a
// device. The page CONTENTS live in the runtime's device pool; this class only
// hands out and takes back page ids.
//
// Thread-safe: sequences can be created, forked and destroyed from any thread.
//
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace llm {

class PageAllocator {
public:
    explicit PageAllocator(int total_pages);

    int total() const { return total_; }
    int free_count() const;
    int in_use() const { return total_ - free_count(); }

    // A page with reference count 1, or nullopt if none is free.
    std::optional<int> allocate();

    // All-or-nothing: n pages, or none (and nullopt) if fewer than n are free.
    // This is what makes a batched step transactional -- either every sequence
    // gets the page it needs, or nothing changes.
    std::optional<std::vector<int>> allocate_many(int n);

    // Another holder of an existing page (a forked sequence).
    void retain(int page);

    // Drops one reference; the page returns to the free list at zero.
    void release(int page);

    int refcount(int page) const;

private:
    void check(int page) const;

    int total_;
    mutable std::mutex mu_;
    std::vector<int> free_;       // stack of free page ids
    std::vector<int> refs_;
};

}  // namespace llm
