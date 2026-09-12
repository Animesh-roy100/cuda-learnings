// GoogleTest suite for the lock-free GPU hash table.
// Compiles as plain C++20 -- no nvcc, because hash_kv.h exposes no CUDA syntax.

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "hash_kv.h"

using kv::GpuHashTable;
using kv::Lookup;
using kv::Probe;

namespace {

// hash() is the Murmur3 finalizer, a bijection on uint32. Feeding it 0,1,2,...
// therefore yields provably distinct keys, and disjoint index ranges yield
// provably disjoint key sets. Random keys would collide by the birthday bound
// (~16k collisions at 8M keys) and silently corrupt a "value == index" check.
std::vector<std::uint32_t> distinct_keys(int n, int offset = 0) {
    std::vector<std::uint32_t> k(n);
    for (int i = 0; i < n; ++i) {
        std::uint32_t h = GpuHashTable::hash(static_cast<std::uint32_t>(i + offset));
        k[i] = (h == GpuHashTable::kEmptyKey) ? GpuHashTable::hash(0xDEADBEEFu) : h;
    }
    return k;
}

std::vector<std::uint32_t> iota_values(int n) {
    std::vector<std::uint32_t> v(n);
    std::iota(v.begin(), v.end(), 0u);
    return v;
}

}  // namespace

TEST(HashKv, CapacityRoundsUpToPowerOfTwo) {
    GpuHashTable t(1000);
    EXPECT_EQ(t.capacity(), 1024u);
    GpuHashTable t2(1024);
    EXPECT_EQ(t2.capacity(), 1024u);
}

TEST(HashKv, EmptyTableFindsNothing) {
    GpuHashTable t(1024);
    auto keys = distinct_keys(100);
    auto got = t.find(keys);
    for (auto v : got) EXPECT_EQ(v, GpuHashTable::kNotFound);
}

TEST(HashKv, InsertThenFind) {
    const int N = 10000;
    GpuHashTable t(N * 2);
    auto keys = distinct_keys(N);
    auto vals = iota_values(N);
    t.insert(keys, vals);

    auto got = t.find(keys);
    ASSERT_EQ(got.size(), vals.size());
    for (int i = 0; i < N; ++i) EXPECT_EQ(got[i], vals[i]) << "at index " << i;
}

TEST(HashKv, SizeHintMatchesInsertedCount) {
    const int N = 5000;
    GpuHashTable t(N * 2);
    t.insert(distinct_keys(N), iota_values(N));
    EXPECT_EQ(t.size_hint(), static_cast<std::size_t>(N));
}

TEST(HashKv, AbsentKeysReturnNotFound) {
    const int N = 10000;
    GpuHashTable t(N * 2);
    t.insert(distinct_keys(N), iota_values(N));

    // Indices [N, 2N) -> provably disjoint from [0, N) because hash is injective.
    auto absent = distinct_keys(N, N);
    auto got = t.find(absent);
    for (int i = 0; i < N; ++i) EXPECT_EQ(got[i], GpuHashTable::kNotFound) << "at " << i;
}

TEST(HashKv, DuplicateKeysCollapseToOneSlot) {
    GpuHashTable t(1024);
    std::vector<std::uint32_t> keys(500, 12345u);
    auto vals = iota_values(500);
    t.insert(keys, vals);

    EXPECT_EQ(t.size_hint(), 1u);          // one key, one slot
    auto got = t.find({12345u});
    ASSERT_EQ(got.size(), 1u);
    EXPECT_NE(got[0], GpuHashTable::kNotFound);
    EXPECT_LT(got[0], 500u);               // some writer won; any is valid
}

TEST(HashKv, WarpCoopMatchesPerThread) {
    const int N = 50000;
    GpuHashTable t(N * 2);
    auto keys = distinct_keys(N);
    t.insert(keys, iota_values(N));

    auto a = t.find(keys, Lookup::PerThread);
    auto b = t.find(keys, Lookup::WarpCoop);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(a, b);
}

// Regression test. The warp-cooperative lookup originally capped probing at
// 1024 slots, which silently reported present keys as missing once clustering
// produced longer chains. That only shows up at high load factor.
TEST(HashKv, CorrectAtHighLoadFactor) {
    const std::size_t CAP = 1u << 16;
    const int N = static_cast<int>(CAP * 0.95);
    GpuHashTable t(CAP);
    auto keys = distinct_keys(N);
    auto vals = iota_values(N);
    t.insert(keys, vals);

    auto a = t.find(keys, Lookup::PerThread);
    auto b = t.find(keys, Lookup::WarpCoop);
    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(a[i], vals[i]) << "per-thread lookup failed at " << i;
        ASSERT_EQ(b[i], vals[i]) << "warp-coop lookup failed at " << i;
    }
}

TEST(HashKv, RobinHoodFindsEverythingToo) {
    const int N = 50000;
    GpuHashTable t(N * 2, Probe::RobinHood);
    auto keys = distinct_keys(N);
    auto vals = iota_values(N);
    t.insert(keys, vals);

    auto got = t.find(keys);
    for (int i = 0; i < N; ++i) EXPECT_EQ(got[i], vals[i]) << "at " << i;
    EXPECT_EQ(t.size_hint(), static_cast<std::size_t>(N));
}

// The actual point of Robin Hood: it does not shorten the AVERAGE chain much,
// it collapses the VARIANCE, so the worst-case LOOKUP improves.
//
// Measure displacement from the home slot on the final table -- not insert-loop
// iterations. Robin Hood's insert loop also carries displaced entries onward,
// so its iteration count reflects insert work and is not comparable to linear
// probing's. Displacement is what a lookup actually walks.
TEST(HashKv, RobinHoodReducesWorstCaseDisplacement) {
    const std::size_t CAP = 1u << 16;
    const int N = static_cast<int>(CAP * 0.90);
    auto keys = distinct_keys(N);
    auto vals = iota_values(N);

    GpuHashTable lin(CAP, Probe::Linear);
    lin.insert(keys, vals);
    auto d_lin = lin.displacement();

    GpuHashTable rh(CAP, Probe::RobinHood);
    rh.insert(keys, vals);
    auto d_rh = rh.displacement();

    EXPECT_GT(d_lin.max, 0u);
    EXPECT_GT(d_rh.max, 0u);

    // The tail is what Robin Hood fixes.
    EXPECT_LT(d_rh.max, d_lin.max)
        << "robin hood max displacement=" << d_rh.max
        << " linear max displacement=" << d_lin.max;

    // The mean should be essentially unchanged -- both store the same keys in
    // the same number of slots, only the distribution differs.
    EXPECT_NEAR(d_rh.avg, d_lin.avg, d_lin.avg * 0.5 + 1.0);
}

TEST(HashKv, ClearEmptiesTable) {
    const int N = 1000;
    GpuHashTable t(N * 4);
    auto keys = distinct_keys(N);
    t.insert(keys, iota_values(N));
    ASSERT_EQ(t.size_hint(), static_cast<std::size_t>(N));

    t.clear();
    EXPECT_EQ(t.size_hint(), 0u);
    auto got = t.find(keys);
    for (auto v : got) EXPECT_EQ(v, GpuHashTable::kNotFound);
}
