#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "dpi_engine.h"

using dpi::DpiEngine;
using dpi::Match;
using dpi::MatchStats;
using dpi::PacketBatch;

namespace {

PacketBatch from_strings(const std::vector<std::string>& packets) {
    PacketBatch b;
    b.offsets.push_back(0);
    for (const auto& s : packets) {
        b.data.insert(b.data.end(), s.begin(), s.end());
        b.offsets.push_back((std::int32_t)b.data.size());
    }
    return b;
}

bool bit_set(const std::vector<std::uint32_t>& mask, int words, int packet, int pattern) {
    return (mask[(std::size_t)packet * words + pattern / 32] >> (pattern % 32)) & 1u;
}

}  // namespace

TEST(Dpi, RejectsBadPatterns) {
    EXPECT_THROW(DpiEngine({}), std::invalid_argument);
    EXPECT_THROW(DpiEngine({"ok", ""}), std::invalid_argument);
    EXPECT_THROW(DpiEngine({std::string(256, 'x')}), std::invalid_argument);
    EXPECT_NO_THROW(DpiEngine({std::string(255, 'x')}));
}

TEST(Dpi, FindsSimpleSubstring) {
    DpiEngine e({"needle"});
    auto batch = from_strings({"haystack", "hasneedlehere", "nothing"});
    auto mask = e.scan_bitmask(batch);

    const int w = e.words_per_packet();
    EXPECT_FALSE(bit_set(mask, w, 0, 0));
    EXPECT_TRUE(bit_set(mask, w, 1, 0));
    EXPECT_FALSE(bit_set(mask, w, 2, 0));
}

TEST(Dpi, ReportsCorrectEndOffset) {
    DpiEngine e({"abc"});
    auto batch = from_strings({"xxabcxx"});
    auto m = e.scan_detailed(batch, 16);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0].packet, 0);
    EXPECT_EQ(m[0].pattern, 0);
    EXPECT_EQ(m[0].end_offset, 5) << "offset is one past the last matched byte";
}

// The defining property of Aho-Corasick: overlapping patterns are all found in
// a single pass. A naive per-pattern scan finds them too, but this checks the
// fail-link output inheritance is right, which is where the algorithm usually
// goes wrong.
TEST(Dpi, FindsOverlappingAndNestedPatterns) {
    DpiEngine e({"he", "she", "his", "hers"});
    auto batch = from_strings({"ushers"});
    auto m = e.scan_detailed(batch, 64);

    std::set<std::pair<int, int>> found;   // (pattern, end_offset)
    for (const auto& x : m) found.insert({x.pattern, x.end_offset});

    // "ushers": she@1..3, he@2..3, hers@2..5
    EXPECT_TRUE(found.count({1, 4})) << "she";
    EXPECT_TRUE(found.count({0, 4})) << "he";
    EXPECT_TRUE(found.count({3, 6})) << "hers";
    EXPECT_FALSE(found.count({2, 0})) << "his must not match";
}

TEST(Dpi, MatchesCpuReferenceOnSyntheticTraffic) {
    std::vector<std::string> pats = {"ATTACK", "MALWARE", "EXPLOIT", "0day", "DROP TABLE"};
    DpiEngine e(pats);
    auto batch = dpi::make_traffic(2000, 64, 512, pats, 0.30, 7);

    auto gpu = e.scan_detailed(batch, 100000);
    auto cpu = DpiEngine::scan_cpu(pats, batch);

    ASSERT_EQ(gpu.size(), cpu.size()) << "match count differs from reference";
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        EXPECT_EQ(gpu[i].packet, cpu[i].packet) << "at " << i;
        EXPECT_EQ(gpu[i].pattern, cpu[i].pattern) << "at " << i;
        EXPECT_EQ(gpu[i].end_offset, cpu[i].end_offset) << "at " << i;
    }
}

// The bitmask path and the detailed path run different kernels; they must
// agree on which patterns hit which packets.
TEST(Dpi, BitmaskAgreesWithDetailed) {
    std::vector<std::string> pats = {"alpha", "beta", "gamma", "delta"};
    DpiEngine e(pats);
    auto batch = dpi::make_traffic(1000, 32, 256, pats, 0.40, 11);

    auto mask = e.scan_bitmask(batch);
    auto detail = e.scan_detailed(batch, 100000);
    const int w = e.words_per_packet();

    std::set<std::pair<int, int>> from_detail;
    for (const auto& m : detail) from_detail.insert({m.packet, m.pattern});

    for (int p = 0; p < batch.size(); ++p)
        for (int pi = 0; pi < (int)pats.size(); ++pi)
            EXPECT_EQ(bit_set(mask, w, p, pi), from_detail.count({p, pi}) > 0)
                << "packet " << p << " pattern " << pi;
}

// Zero-copy changes WHERE the payload lives, never what is computed.
TEST(Dpi, ZeroCopyMatchesDeviceCopy) {
    std::vector<std::string> pats = {"GET /", "POST /", "User-Agent"};
    DpiEngine e(pats);
    auto batch = dpi::make_traffic(800, 64, 300, pats, 0.5, 13);

    auto a = e.scan_bitmask(batch);
    auto b = e.scan_bitmask_zerocopy(batch);
    EXPECT_EQ(a, b);
}

TEST(Dpi, HandlesManyPatternsAcrossWordBoundary) {
    // 40 patterns forces two 32-bit mask words, which is where an off-by-one
    // in the bit indexing would show up.
    std::vector<std::string> pats;
    for (int i = 0; i < 40; ++i) pats.push_back("pat" + std::to_string(i) + "END");
    DpiEngine e(pats);
    EXPECT_EQ(e.words_per_packet(), 2);

    auto batch = from_strings({"xx" + pats[0] + "yy", "zz" + pats[39] + "ww"});
    auto mask = e.scan_bitmask(batch);
    const int w = e.words_per_packet();

    EXPECT_TRUE(bit_set(mask, w, 0, 0));
    EXPECT_FALSE(bit_set(mask, w, 0, 39));
    EXPECT_TRUE(bit_set(mask, w, 1, 39)) << "pattern 39 lives in the second word";
    EXPECT_FALSE(bit_set(mask, w, 1, 0));
}

TEST(Dpi, EmptyBatchIsSafe) {
    DpiEngine e({"x"});
    PacketBatch empty;
    empty.offsets.push_back(0);
    EXPECT_EQ(empty.size(), 0);
    EXPECT_NO_THROW(e.scan_bitmask(empty));
    EXPECT_TRUE(e.scan_detailed(empty, 10).empty());
}

TEST(Dpi, PatternLongerThanPacketNeverMatches) {
    DpiEngine e({"averylongsignature"});
    auto batch = from_strings({"short", "tiny", "a"});
    auto mask = e.scan_bitmask(batch);
    for (auto word : mask) EXPECT_EQ(word, 0u);
}

TEST(Dpi, StatsAreConsistent) {
    std::vector<std::string> pats = {"aaa", "bbb"};
    DpiEngine e(pats);
    auto batch = dpi::make_traffic(500, 64, 128, pats, 0.6, 17);

    MatchStats st{};
    e.scan_bitmask(batch, &st);
    EXPECT_EQ(st.packets_scanned, batch.size());
    EXPECT_EQ(st.bytes_scanned, (std::int64_t)batch.bytes());
    EXPECT_GT(st.kernel_ms, 0.0f);
    EXPECT_GT(st.total_matches, 0);
}

// max_matches must truncate the OUTPUT without corrupting the count or writing
// out of bounds -- the classic overflow bug in a bounded reporting buffer.
TEST(Dpi, DetailedScanTruncatesSafely) {
    std::vector<std::string> pats = {"ab"};
    DpiEngine e(pats);
    auto batch = from_strings({std::string(500, 'a') + "ab" + std::string(500, 'b')});

    MatchStats st{};
    auto few = e.scan_detailed(batch, 1, &st);
    EXPECT_LE(few.size(), 1u);
    EXPECT_GE(st.total_matches, (std::int64_t)few.size())
        << "stats must report the true count even when the buffer truncates";
}
