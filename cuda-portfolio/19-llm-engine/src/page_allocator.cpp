#include "page_allocator.h"

#include <string>

#include "errors.h"

namespace llm {

PageAllocator::PageAllocator(int total_pages) : total_(total_pages) {
    if (total_pages <= 0)
        throw Error(ErrorKind::InvalidArgument, "PageAllocator: total_pages must be positive");
    free_.reserve(total_pages);
    // Hand out low ids first, so behaviour is deterministic and tests can
    // predict page ids.
    for (int p = total_pages - 1; p >= 0; --p) free_.push_back(p);
    refs_.assign(total_pages, 0);
}

int PageAllocator::free_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(free_.size());
}

std::optional<int> PageAllocator::allocate() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_.empty()) return std::nullopt;
    const int p = free_.back();
    free_.pop_back();
    refs_[p] = 1;
    return p;
}

std::optional<std::vector<int>> PageAllocator::allocate_many(int n) {
    if (n < 0) throw Error(ErrorKind::InvalidArgument, "PageAllocator: negative page count");
    std::lock_guard<std::mutex> lock(mu_);
    if (static_cast<int>(free_.size()) < n) return std::nullopt;
    std::vector<int> pages(n);
    for (int i = 0; i < n; ++i) {
        pages[i] = free_.back();
        free_.pop_back();
        refs_[pages[i]] = 1;
    }
    return pages;
}

void PageAllocator::check(int page) const {
    if (page < 0 || page >= total_)
        throw Error(ErrorKind::InvalidArgument, "PageAllocator: page id " + std::to_string(page) +
                                                    " out of range");
}

void PageAllocator::retain(int page) {
    std::lock_guard<std::mutex> lock(mu_);
    check(page);
    if (refs_[page] == 0)
        throw Error(ErrorKind::InvalidArgument,
                    "PageAllocator: retain of free page " + std::to_string(page));
    ++refs_[page];
}

void PageAllocator::release(int page) {
    std::lock_guard<std::mutex> lock(mu_);
    check(page);
    if (refs_[page] == 0)
        throw Error(ErrorKind::InvalidArgument,
                    "PageAllocator: double release of page " + std::to_string(page));
    if (--refs_[page] == 0) free_.push_back(page);
}

int PageAllocator::refcount(int page) const {
    std::lock_guard<std::mutex> lock(mu_);
    check(page);
    return refs_[page];
}

}  // namespace llm
