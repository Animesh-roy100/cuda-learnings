// Error paths.
//
// Every other suite in this repo proves the code is right when things go
// well. This one proves it fails cleanly when they do not: out of memory, bad
// launch configurations, sticky faults, and -- the class that found real bugs --
// constructors that throw halfway through acquiring device resources.
//
// Two distinctions run through all of it.
//
// RECOVERABLE vs STICKY. A recoverable error fails one call and leaves the
// context fine; a sticky one ends the context for the life of the process.
// Treating them the same either restarts processes that were healthy or keeps
// driving one that is dead.
//
// FAILED vs STALE. A failed runtime call also records its error as the thread's
// last error, and cudaGetLastError() returns it once, later, to whoever asks
// first. A kernel-launch check that runs after an unrelated handled failure
// reads that record and blames the kernel. That misreading is what this repo
// previously documented as cudaMemAdvise "poisoning the context".

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_dsp.h"
#include "cu/check.hpp"
#include "error_kernels.h"
#include "hash_kv.h"
#include "kv_cache.h"
#include "mc_pricing.h"
#include "video_pipeline.h"

namespace {

std::size_t free_vram() {
    std::size_t free_b = 0, total_b = 0;
    CU_CHECK(cudaMemGetInfo(&free_b, &total_b));
    return free_b;
}

std::size_t total_vram() {
    std::size_t free_b = 0, total_b = 0;
    CU_CHECK(cudaMemGetInfo(&free_b, &total_b));
    return total_b;
}

// How much this process can actually allocate right now, found by allocating.
//
// Not the same as free VRAM. On Windows, NVIDIA's "CUDA sysmem fallback" lets
// cudaMalloc silently continue into system RAM once VRAM is exhausted: in this
// suite's first run a 1.58x-free-VRAM set of allocations succeeded on the GTX
// 1650. Any test that sizes a failure from cudaMemGetInfo is therefore sizing
// it wrong on Windows and right on Linux.
std::size_t allocatable_bytes() {
    constexpr std::size_t chunk = 256ull << 20;
    constexpr int max_chunks = 64;   // 16 GB is past anything this suite needs
    std::vector<void*> held;
    for (int i = 0; i < max_chunks; ++i) {
        void* p = nullptr;
        if (cudaMalloc(&p, chunk) != cudaSuccess) break;
        held.push_back(p);
    }
    (void)cudaGetLastError();   // the failure that ended the loop is expected
    for (void* p : held) cudaFree(p);
    return held.size() * chunk;
}

// A request no machine can satisfy, sized to be refused INSTANTLY. Measured on
// this driver, a failed cudaMalloc costs time proportional to the request while
// sysmem fallback tries to page for it: 1.1 s at 8 GB, 7.9 s at 256 GB. At 1 TB
// and above it is refused in under a millisecond. "64x the card" put ten
// seconds into every test that used it and 200 s into the loop below.
constexpr std::size_t kImpossible = std::size_t(1) << 40;

// Allocator granularity and driver bookkeeping move free VRAM by a few MB
// between calls with no leak involved. What the constructors below stranded
// was gigabytes, so this is far below the signal.
constexpr std::size_t kVramSlack = 64ull << 20;

std::string mb(std::size_t b) { return std::to_string(b >> 20) + " MB"; }

void expect_no_vram_loss(std::size_t before, const char* what) {
    const std::size_t after = free_vram();
    EXPECT_LE(before - std::min(before, after), kVramSlack)
        << what << ": free VRAM fell from " << mb(before) << " to " << mb(after);
}

}  // namespace

// ===========================================================================
// Recoverable errors: the call fails, the context survives.
// ===========================================================================

TEST(Recoverable, AllocationBeyondCapacityThrowsTypedError) {
    float* p = nullptr;
    try {
        CU_CHECK(cudaMalloc(&p, kImpossible));
        cudaFree(p);
        FAIL() << "allocating 1 TB succeeded";
    } catch (const cu::CudaError& e) {
        EXPECT_EQ(e.code(), cudaErrorMemoryAllocation);
        EXPECT_NE(std::string(e.what()).find("cudaErrorMemoryAllocation"), std::string::npos)
            << "the message should name the error: " << e.what();
    }
}

TEST(Recoverable, CuCheckConsumesTheErrorItThrowsFor) {
    // Regression test. Before the fix, CU_CHECK threw without reading the
    // runtime's last-error record, so the NEXT CU_CHECK_KERNEL on a perfectly
    // good kernel reported "kernel launch failed: cudaErrorMemoryAllocation".
    float* p = nullptr;
    EXPECT_THROW(CU_CHECK(cudaMalloc(&p, kImpossible)), cu::CudaError);
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess)
        << "CU_CHECK left " << cudaGetErrorName(cudaPeekAtLastError()) << " recorded";
    EXPECT_TRUE(ep::launch_and_verify(100000));
}

TEST(Recoverable, AnUnreadErrorIsStaleNotSticky) {
    // The distinction measured directly. A raw call fails and nobody reads the
    // error. The context is FINE -- a kernel launches and computes correctly --
    // but the first cudaGetLastError() afterwards still reports the old failure.
    float* p = nullptr;
    ASSERT_EQ(cudaMalloc(&p, kImpossible), cudaErrorMemoryAllocation);
    EXPECT_EQ(cudaPeekAtLastError(), cudaErrorMemoryAllocation);

    try {
        ep::launch_and_verify(1000);
        ADD_FAILURE() << "expected the stale error to be misreported as a launch failure";
    } catch (const cu::CudaError& e) {
        EXPECT_EQ(e.code(), cudaErrorMemoryAllocation)
            << "the kernel check picked up the earlier allocation failure";
    }
    // Read once, gone. Nothing was wrong with the context at any point.
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
    EXPECT_TRUE(ep::launch_and_verify(1000));
}

TEST(Recoverable, RepeatedAllocationFailuresLeakNothing) {
    const std::size_t before = free_vram();
    for (int i = 0; i < 200; ++i) {
        float* p = nullptr;
        ASSERT_EQ(cudaMalloc(&p, kImpossible), cudaErrorMemoryAllocation);
    }
    (void)cudaGetLastError();
    expect_no_vram_loss(before, "200 failed allocations");
}

TEST(Recoverable, InvalidLaunchConfigurationThrowsAndIsNotSticky) {
    try {
        ep::launch_with_oversized_block();
        FAIL() << "a 4096-thread block launched";
    } catch (const cu::CudaError& e) {
        // Documentation generally describes bad block dimensions as
        // cudaErrorInvalidConfiguration. CUDA 13.4 on this driver reports
        // cudaErrorInvalidValue. Either is a correct refusal; what matters is
        // that it is refused, typed, and recoverable.
        EXPECT_TRUE(e.code() == cudaErrorInvalidValue ||
                    e.code() == cudaErrorInvalidConfiguration)
            << "got " << cudaGetErrorName(e.code());
    }
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
    EXPECT_TRUE(ep::launch_and_verify(1000));
}

TEST(Recoverable, OversizedCopyIsRejectedNotPerformed) {
    int* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, sizeof(int) * 16), cudaSuccess);
    std::vector<int> big(1 << 20, 7);
    // Copying 1M ints into a 16-int allocation must be refused by the runtime,
    // not carried out over whatever lies beyond it.
    EXPECT_NE(cudaMemcpy(d, big.data(), sizeof(int) * big.size(), cudaMemcpyHostToDevice),
              cudaSuccess);
    (void)cudaGetLastError();
    cudaFree(d);
    EXPECT_TRUE(ep::launch_and_verify(1000));
}

// ===========================================================================
// Sticky errors: the context is gone. Must run in a child process.
// ===========================================================================

TEST(StickyDeathTest, IllegalAddressPoisonsTheWholeContext) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // The child faults, then tries something unrelated. Exit code 42 means both
    // halves behaved as documented: the fault surfaced as a typed exception,
    // and the context refused all further work -- unlike the stale errors above.
    EXPECT_EXIT(
        {
            try {
                ep::write_to_illegal_address();
                std::exit(1);   // fault went unreported
            } catch (const cu::CudaError& e) {
                if (e.code() != cudaErrorIllegalAddress) std::exit(2);
            }
            try {
                ep::launch_and_verify(10);
                std::exit(3);   // context still worked: not sticky after all
            } catch (const cu::CudaError&) {
                std::exit(42);
            }
        },
        ::testing::ExitedWithCode(42), "");
    // ...and the parent, which has its own context, is unaffected.
    EXPECT_TRUE(ep::launch_and_verify(1000));
}

// ===========================================================================
// Constructors that throw partway through acquiring device resources.
//
// A C++ destructor does not run for an object whose constructor threw. Every
// class here allocated its pimpl with `impl_(new Impl)` and freed device memory
// only in the destructor, so a throw after the first allocation stranded all of
// it until the process exited. A service that retries a failed construction
// loses VRAM on every attempt.
// ===========================================================================

TEST(ConstructorFailure, VideoPipelineReleasesPartialAllocations) {
    const std::size_t T = allocatable_bytes();
    // Allocation order: NV12 1.5n, RGB 3n, four n-byte planes, pitched ~n.
    // With n = T/7.5 the first four total 0.87T and succeed, and the run fails
    // at the fifth or sixth.
    const std::size_t n = T * 2 / 15;
    const int width = 32768;
    const std::size_t rows = (n / width) & ~std::size_t(1);
    if (rows * width > std::size_t(std::numeric_limits<int>::max()) / 2)
        GTEST_SKIP() << "allocatable memory (" << mb(T) << ") is too large to fail "
                        "a frame whose pixel count fits the pipeline's int indexing";
    const int height = static_cast<int>(rows);

    const std::size_t before = free_vram();
    EXPECT_THROW((video::VideoPipeline{width, height}), cu::CudaError);
    expect_no_vram_loss(before, "failed VideoPipeline construction");
}

TEST(ConstructorFailure, StftProcessorReleasesPartialAllocations) {
    const std::size_t T = allocatable_bytes();
    // frame_size == hop, one channel: pcm, frames, spec, spec2, out, wsum and
    // resampled are each ~4 bytes per sample. With 4S = T/5 the first five fill
    // the allocatable budget and the sixth fails.
    audio::StftConfig cfg;
    cfg.frame_size = 1024;
    cfg.hop = 1024;
    const std::size_t samples = T / 20;
    if (samples > std::size_t(std::numeric_limits<int>::max()) / 2)
        GTEST_SKIP() << "allocatable memory (" << mb(T) << ") exceeds an int sample count";

    const std::size_t before = free_vram();
    EXPECT_THROW((audio::StftProcessor{cfg, 1, static_cast<int>(samples)}), cu::CudaError);
    expect_no_vram_loss(before, "failed StftProcessor construction");
}

TEST(ConstructorFailure, RetryingAFailedConstructionDoesNotAccumulate) {
    // The production shape of the bug: a caller that catches and retries.
    const std::size_t T = allocatable_bytes();
    const int width = 32768;
    const std::size_t rows = ((T * 2 / 15) / width) & ~std::size_t(1);
    if (rows * width > std::size_t(std::numeric_limits<int>::max()) / 2)
        GTEST_SKIP() << "allocatable memory too large for this test's frame sizing";

    const std::size_t before = free_vram();
    int failures = 0;
    for (int attempt = 0; attempt < 5; ++attempt) {
        try {
            video::VideoPipeline p(width, static_cast<int>(rows));
        } catch (const cu::CudaError&) {
            ++failures;
        }
    }
    EXPECT_EQ(failures, 5) << "every attempt should fail the same way";
    expect_no_vram_loss(before, "five failed constructions");
}

TEST(ConstructorFailure, ValidationHappensBeforeAcquisition) {
    // price() allocated its device output buffer before validating `steps`,
    // so every rejected path-dependent call leaked it. The leak is 40 bytes and
    // below what cudaMemGetInfo can resolve -- a VRAM-measuring version of this
    // test passed against the unfixed code, which proved nothing -- so this
    // tests the ordering's observable contract instead.
    mc::Engine engine;
    mc::Option asian;
    asian.style = mc::Style::Asian;
    EXPECT_THROW(engine.price(asian, 1000, 0), std::invalid_argument);
    EXPECT_THROW(engine.price(asian, 0, 252), std::invalid_argument);

    // steps is meaningless for a European option and must stay accepted.
    mc::Option euro;
    euro.style = mc::Style::European;
    EXPECT_NO_THROW(engine.price(euro, 10000, 0));
}

TEST(ConstructorFailure, InvalidArgumentsLeaveTheContextClean) {
    // Validation throws used to fire after `new Impl`, leaking it. They must
    // also leave no CUDA error behind for a later check to misreport.
    EXPECT_THROW((video::VideoPipeline{3, 3}), std::invalid_argument);
    audio::StftConfig bad;
    bad.frame_size = 1000;   // not a power of two
    EXPECT_THROW((audio::StftProcessor{bad, 1, 1024}), std::invalid_argument);
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
    EXPECT_TRUE(ep::launch_and_verify(1000));
}

// ===========================================================================
// Sizes that must be rejected before they reach the allocator.
// ===========================================================================

TEST(SizeValidation, HashTableRejectsCapacityBeyondItsIndexWidth) {
    // Slots are addressed through a 32-bit mask. 2^33 slots used to be passed
    // straight to cudaMalloc, and on a card large enough to allocate them the
    // mask would silently truncate and alias half the table.
    EXPECT_THROW(kv::GpuHashTable{std::size_t(1) << 33}, std::invalid_argument);
}

TEST(SizeValidation, HashTableMaximumCapacityFailsInsteadOfHanging) {
    // round_up_pow2 doubled until c >= n. For n > 2^63 the doubling overflows
    // to zero and the loop never terminates.
    EXPECT_THROW(kv::GpuHashTable{std::numeric_limits<std::size_t>::max()},
                 std::invalid_argument);
}

TEST(SizeValidation, HashTableBoundaryCapacityIsLegalButDoesNotFit) {
    // 2^32 slots is the limit itself: valid, just 32 GB. It must fail as
    // out-of-memory, not as a validation error.
    EXPECT_THROW(kv::GpuHashTable{std::size_t(1) << 32}, cu::CudaError);
}

TEST(SizeValidation, KvCacheRejectsASizeThatOverflows) {
    // 65536 heads * 65536 dims * 65536 tokens * 2 (K and V) * 8192 pages *
    // 4 bytes is exactly 2^64, which wraps a 64-bit size to ZERO -- and
    // cudaMalloc(&p, 0) succeeds, returning a null slab. Without validation
    // this constructed "successfully" and the first append wrote through null.
    llm::KvCacheConfig cfg;
    cfg.n_layers = 1;
    cfg.n_kv_heads = 65536;
    cfg.head_dim = 65536;
    cfg.page_tokens = 65536;
    cfg.total_pages = 8192;
    EXPECT_THROW(llm::PagedKvCache c(cfg), std::invalid_argument);
}

TEST(SizeValidation, ValidObjectsStillConstructAfterAllOfThat) {
    kv::GpuHashTable t(1024);
    EXPECT_EQ(t.capacity(), 1024u);
    llm::KvCacheConfig cfg;
    llm::PagedKvCache c(cfg);
    EXPECT_EQ(c.free_pages(), cfg.total_pages);
    EXPECT_TRUE(ep::launch_and_verify(4096));
}

// ===========================================================================
// The platform fact the constructor tests had to be redesigned around.
// ===========================================================================

TEST(Platform, ReportsAllocatableMemoryAgainstFreeVram) {
    const std::size_t F = free_vram();
    const std::size_t T = allocatable_bytes();
    std::printf("  free VRAM %s, allocatable %s (%.2fx)\n", mb(F).c_str(), mb(T).c_str(),
                F ? double(T) / double(F) : 0.0);
    // Not asserted in either direction: above 1.0 is sysmem fallback (Windows),
    // at or below is the Linux behaviour. Both are correct; code that assumes
    // one of them is not.
    EXPECT_GT(T, 0u);
    ::testing::Test::RecordProperty("free_vram_mb", std::to_string(F >> 20));
    ::testing::Test::RecordProperty("allocatable_mb", std::to_string(T >> 20));
}
