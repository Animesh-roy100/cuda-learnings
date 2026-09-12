#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "sha256_engine.h"

using crypto::Digest;
using crypto::MiningResult;
using crypto::Sha256Engine;

// FIPS 180-4 / NIST published vectors. These are the only reason to trust the
// implementation at all -- a hash that agrees with your own reference but not
// with the standard is simply a different hash.
TEST(Sha256, MatchesNistVectors) {
    struct V { const char* msg; const char* want; };
    const V vectors[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"a", "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"},
        {"message digest",
         "f7846f55cf23e14eebeab5b4e1550cad5b509e3348fbc4efa3a1413d393cb650"},
        {"abcdefghijklmnopqrstuvwxyz",
         "71c480df93d6ae2f1efad1447c66c9525e316218cf51fc8d9ed832f2daf18b73"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
    };
    for (const auto& v : vectors) {
        EXPECT_EQ(Sha256Engine::to_hex(Sha256Engine::hash_cpu(v.msg)), v.want)
            << "for message \"" << v.msg << "\"";
    }
}

TEST(Sha256, GpuMatchesCpuReference) {
    Sha256Engine e;
    std::vector<std::string> msgs;
    for (int i = 0; i < 5000; ++i) {
        std::string s = "block_" + std::to_string(i);
        s.resize(32, '.');            // equal length, required by hash_batch
        msgs.push_back(s);
    }

    auto gpu = e.hash_batch(msgs);
    ASSERT_EQ(gpu.size(), msgs.size());
    for (std::size_t i = 0; i < msgs.size(); ++i)
        EXPECT_EQ(gpu[i], Sha256Engine::hash_cpu(msgs[i])) << "at index " << i;
}

TEST(Sha256, GpuMatchesNistVectorDirectly) {
    Sha256Engine e;
    // "abc" padded to a fixed width would change the digest, so hash it alone.
    auto out = e.hash_batch({"abc"});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(Sha256Engine::to_hex(out[0]),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Sha256, AvalancheOneBitChangesEverything) {
    Sha256Engine e;
    auto a = e.hash_batch({"aaaaaaaaaaaaaaaa"});
    auto b = e.hash_batch({"aaaaaaaaaaaaaaab"});
    ASSERT_EQ(a.size(), 1u);
    ASSERT_EQ(b.size(), 1u);
    EXPECT_NE(a[0], b[0]);

    int differing = 0;
    for (int i = 0; i < 32; ++i)
        for (int bit = 0; bit < 8; ++bit)
            if (((a[0][i] >> bit) & 1) != ((b[0][i] >> bit) & 1)) ++differing;
    // A good hash flips about half the 256 output bits.
    EXPECT_GT(differing, 90);
    EXPECT_LT(differing, 166);
}

TEST(Sha256, RejectsOversizedAndRaggedInput) {
    Sha256Engine e;
    EXPECT_THROW(e.hash_batch({std::string(56, 'x')}), std::invalid_argument);
    EXPECT_NO_THROW(e.hash_batch({std::string(55, 'x')}));
    EXPECT_THROW(e.hash_batch({"short", "muchlongerstring"}), std::invalid_argument);
}

TEST(Sha256, EmptyBatchIsSafe) {
    Sha256Engine e;
    EXPECT_TRUE(e.hash_batch({}).empty());
}

TEST(Sha256, LeadingZeroBitCounting) {
    Digest d{};
    EXPECT_EQ(Sha256Engine::count_leading_zero_bits(d), 256);   // all zero

    d[0] = 0x80;
    EXPECT_EQ(Sha256Engine::count_leading_zero_bits(d), 0);
    d[0] = 0x01;
    EXPECT_EQ(Sha256Engine::count_leading_zero_bits(d), 7);
    d[0] = 0x00; d[1] = 0x40;
    EXPECT_EQ(Sha256Engine::count_leading_zero_bits(d), 9);
}

// The critical mining test: the returned nonce must actually satisfy the
// difficulty when re-hashed independently on the host. A miner that reports
// a nonce the CPU cannot verify is worthless.
TEST(Sha256, MinedNonceVerifiesOnCpu) {
    Sha256Engine e;
    const std::string prefix = "portfolio-block-";
    const int bits = 20;

    auto r = e.mine(prefix, bits, 0, 40'000'000ULL);
    ASSERT_TRUE(r.found) << "no nonce found for " << bits << " bits in 40M tries";

    // Rebuild exactly the message the kernel hashed.
    std::string msg = prefix;
    static const char* hex = "0123456789abcdef";
    for (int i = 0; i < 16; ++i)
        msg.push_back(hex[(r.nonce >> (60 - 4 * i)) & 0xf]);

    const Digest cpu = Sha256Engine::hash_cpu(msg);
    EXPECT_EQ(cpu, r.digest) << "reported digest does not match the CPU hash";
    EXPECT_GE(Sha256Engine::count_leading_zero_bits(cpu), bits)
        << "digest " << Sha256Engine::to_hex(cpu) << " has too few leading zeros";
}

TEST(Sha256, ImpossibleDifficultyReportsNotFound) {
    Sha256Engine e;
    // 200 leading zero bits will not happen in a small search.
    auto r = e.mine("x", 200, 0, 200000);
    EXPECT_FALSE(r.found);
    EXPECT_EQ(r.budget, 200000u);
}

TEST(Sha256, MiningRejectsOversizedPrefix) {
    Sha256Engine e;
    EXPECT_THROW(e.mine(std::string(40, 'x'), 8, 0, 1000), std::invalid_argument);
    EXPECT_NO_THROW(e.mine(std::string(39, 'x'), 1, 0, 10000));
}

TEST(Sha256, HashrateIsPlausible) {
    Sha256Engine e;
    float ms = 0.0f;
    const double mhs = e.benchmark_hashrate(20'000'000ULL, &ms);
    EXPECT_GT(ms, 0.0f);
    // Sanity bounds, not a performance assertion: anything outside this range
    // means the work was optimised away or something is badly wrong.
    EXPECT_GT(mhs, 1.0);
    EXPECT_LT(mhs, 100000.0);
}
