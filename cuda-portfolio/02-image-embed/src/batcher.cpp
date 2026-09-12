// Dynamic request batcher. Plain C++20, no CUDA.
//
// A GPU is efficient on batches and wasteful on single items, but requests
// arrive one at a time. This is the standard serving fix: accumulate until
// either the batch is full or a deadline expires, then fire.
//
// The deadline is the part people leave out and then wonder why latency is
// unbounded under light load: without it, one straggling request waits for a
// batch that may never fill.

#include "image_pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace vision {

struct DynamicBatcher::Impl {
    int max_batch;
    std::chrono::microseconds max_delay;
    Handler handler;

    std::mutex m;
    std::condition_variable cv;
    std::vector<int> pending;
    bool stopping = false;

    std::atomic<int> batches{0};
    std::atomic<int> items{0};
    std::atomic<int> largest{0};

    std::thread worker;

    void run() {
        std::unique_lock<std::mutex> lk(m);
        while (true) {
            // Wake on: a full batch, a stop request, or the deadline.
            cv.wait_for(lk, max_delay, [&] {
                return stopping || static_cast<int>(pending.size()) >= max_batch;
            });
            if (pending.empty()) {
                if (stopping) return;
                continue;
            }

            const int take = std::min<int>(static_cast<int>(pending.size()), max_batch);
            std::vector<int> batch(pending.begin(), pending.begin() + take);
            pending.erase(pending.begin(), pending.begin() + take);

            // Release the lock while the handler runs: submitters must not
            // block behind GPU work.
            lk.unlock();
            if (handler) handler(batch);
            batches.fetch_add(1);
            items.fetch_add(take);
            int prev = largest.load();
            while (take > prev && !largest.compare_exchange_weak(prev, take)) {}
            lk.lock();
        }
    }
};

DynamicBatcher::DynamicBatcher(int max_batch, int max_delay_us, Handler on_batch)
    : impl_(new Impl) {
    try {
        impl_->max_batch = max_batch < 1 ? 1 : max_batch;
        impl_->max_delay = std::chrono::microseconds(max_delay_us < 1 ? 1 : max_delay_us);
        impl_->handler = std::move(on_batch);
        impl_->worker = std::thread([this] { impl_->run(); });
    } catch (...) {
        delete impl_;   // releases anything acquired before the throw
        impl_ = nullptr;
        throw;
    }
}

DynamicBatcher::~DynamicBatcher() {
    if (!impl_) return;
    drain();
    delete impl_;
}

void DynamicBatcher::submit(int request_id) {
    {
        std::lock_guard<std::mutex> lk(impl_->m);
        if (impl_->stopping) return;
        impl_->pending.push_back(request_id);
    }
    impl_->cv.notify_one();
}

void DynamicBatcher::drain() {
    if (!impl_->worker.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(impl_->m);
        impl_->stopping = true;
    }
    impl_->cv.notify_all();
    impl_->worker.join();

    // Anything still queued when the worker exits is flushed here, so drain()
    // really means "everything submitted has been handled".
    std::vector<int> rest;
    {
        std::lock_guard<std::mutex> lk(impl_->m);
        rest.swap(impl_->pending);
    }
    while (!rest.empty()) {
        const int take = std::min<int>(static_cast<int>(rest.size()), impl_->max_batch);
        std::vector<int> batch(rest.begin(), rest.begin() + take);
        rest.erase(rest.begin(), rest.begin() + take);
        if (impl_->handler) impl_->handler(batch);
        impl_->batches.fetch_add(1);
        impl_->items.fetch_add(take);
        int prev = impl_->largest.load();
        while (take > prev && !impl_->largest.compare_exchange_weak(prev, take)) {}
    }
}

int DynamicBatcher::batches_fired() const { return impl_->batches.load(); }
int DynamicBatcher::items_processed() const { return impl_->items.load(); }
int DynamicBatcher::largest_batch() const { return impl_->largest.load(); }

}  // namespace vision
