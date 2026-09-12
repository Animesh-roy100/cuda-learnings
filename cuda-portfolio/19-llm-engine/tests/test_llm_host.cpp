// Host-side runtime components: no GPU required.
//
// Configuration and tensor validation, page allocation, sampling, errors,
// dequantization, and (with the model file) the tokenizer. CI runs this suite
// under AddressSanitizer and UndefinedBehaviorSanitizer on Linux, where there
// is no GPU but all of this code still executes.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "errors.h"
#include "gguf.h"
#include "model_config.h"
#include "page_allocator.h"
#include "quant.h"
#include "sampler.h"
#include "tokenizer.h"

namespace {

// ------------------------------------------------------ in-memory GGUF model

class Writer {
public:
    void u32(std::uint32_t v) { raw(&v, 4); }
    void u64(std::uint64_t v) { raw(&v, 8); }
    void f32(float v) { raw(&v, 4); }
    void str(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    void raw(const void* p, std::size_t n) {
        auto* b = static_cast<const std::uint8_t*>(p);
        bytes.insert(bytes.end(), b, b + n);
    }
    std::vector<std::uint8_t> bytes;
};

struct Spec {
    std::string name;
    std::vector<std::uint64_t> dims;
    llm::GgmlType type;
};

// A structurally complete llama model with tiny dimensions. Every override is
// one way of making it wrong.
struct TinyModel {
    std::string arch = "llama";
    int layers = 1, dim = 64, ffn = 128, heads = 1, kv_heads = 1, ctx = 256, vocab = 300;
    std::map<std::string, std::int64_t> extra_ints;
    std::vector<std::string> drop;                     // tensors to omit
    std::map<std::string, Spec> replace;               // tensors to alter

    std::vector<std::uint8_t> build() const {
        std::vector<Spec> t;
        const std::uint64_t D = dim, F = ffn, V = vocab, KV = std::uint64_t(dim / heads) * kv_heads;
        auto q4 = llm::GgmlType::Q4_0;
        auto f32 = llm::GgmlType::F32;
        t.push_back({"token_embd.weight", {D, V}, q4});
        t.push_back({"output_norm.weight", {D}, f32});
        t.push_back({"output.weight", {D, V}, q4});
        for (int l = 0; l < layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            t.push_back({p + "attn_norm.weight", {D}, f32});
            t.push_back({p + "attn_q.weight", {D, D}, q4});
            t.push_back({p + "attn_k.weight", {D, KV}, q4});
            t.push_back({p + "attn_v.weight", {D, KV}, q4});
            t.push_back({p + "attn_output.weight", {D, D}, q4});
            t.push_back({p + "ffn_norm.weight", {D}, f32});
            t.push_back({p + "ffn_gate.weight", {D, F}, q4});
            t.push_back({p + "ffn_up.weight", {D, F}, q4});
            t.push_back({p + "ffn_down.weight", {F, D}, q4});
        }
        std::vector<Spec> final;
        for (auto s : t) {
            if (std::find(drop.begin(), drop.end(), s.name) != drop.end()) continue;
            if (auto it = replace.find(s.name); it != replace.end()) s = it->second;
            final.push_back(s);
        }

        Writer w;
        w.raw("GGUF", 4);
        w.u32(3);
        w.u64(final.size());
        std::map<std::string, std::int64_t> ints = {
            {"llama.block_count", layers},        {"llama.embedding_length", dim},
            {"llama.feed_forward_length", ffn},    {"llama.attention.head_count", heads},
            {"llama.attention.head_count_kv", kv_heads}, {"llama.context_length", ctx}};
        for (auto& [k, v] : extra_ints) ints[k] = v;
        w.u64(ints.size() + 3);
        w.str("general.architecture");
        w.u32(8);
        w.str(arch);
        for (auto& [k, v] : ints) {
            w.str(k);
            w.u32(11);   // INT64
            w.u64(std::uint64_t(v));
        }
        w.str("llama.attention.layer_norm_rms_epsilon");
        w.u32(6);
        w.f32(1e-5f);
        w.str("tokenizer.ggml.tokens");
        w.u32(9);
        w.u32(8);
        w.u64(vocab);
        for (int i = 0; i < vocab; ++i) w.str("t" + std::to_string(i));

        std::uint64_t offset = 0;
        std::vector<std::uint64_t> sizes;
        for (const auto& s : final) {
            w.str(s.name);
            w.u32(std::uint32_t(s.dims.size()));
            std::uint64_t n = 1;
            for (auto d : s.dims) {
                w.u64(d);
                n *= d;
            }
            w.u32(std::uint32_t(s.type));
            w.u64(offset);
            const auto tr = llm::type_traits(s.type);
            const std::uint64_t bytes = (n / tr.block_elems) * tr.block_bytes;
            offset += (bytes + 31) / 32 * 32;
            sizes.push_back(bytes);
        }
        while (w.bytes.size() % 32) w.bytes.push_back(0);
        w.bytes.resize(w.bytes.size() + offset, 0);
        return w.bytes;
    }
};

llm::ErrorKind kind_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const llm::Error& e) {
        return e.kind();
    }
    ADD_FAILURE() << "no llm::Error thrown";
    return llm::ErrorKind::Device;
}

std::string message_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

// =============================================================== config

TEST(ModelConfig, LoadsAValidTinyModel) {
    auto f = llm::GgufFile::from_memory(TinyModel{}.build());
    // head_dim 64 and widths that are multiples of 32 are what the kernels need.
    const auto c = llm::load_config(f, "tiny.gguf");
    EXPECT_EQ(c.hidden_size, 64);
    EXPECT_EQ(c.head_dim, 64);
    EXPECT_EQ(c.vocab_size, 300);
    EXPECT_NO_THROW(llm::validate_tensors(f, c));
}

TEST(ModelConfig, RejectsAnUnsupportedArchitecture) {
    TinyModel m;
    m.arch = "mamba";
    auto f = llm::GgufFile::from_memory(m.build());
    EXPECT_EQ(kind_of([&] { llm::load_config(f); }), llm::ErrorKind::Unsupported);
}

TEST(ModelConfig, RejectsHeadsThatDoNotDivideTheHiddenSize) {
    TinyModel m;
    m.dim = 96;
    m.heads = 5;   // 96 / 5 truncates: hidden != heads * head_dim
    auto f = llm::GgufFile::from_memory(m.build());
    const auto msg = message_of([&] { llm::load_config(f, "bad.gguf"); });
    EXPECT_NE(msg.find("hidden_size == attention_heads * head_dim"), std::string::npos) << msg;
    EXPECT_NE(msg.find("model=bad.gguf"), std::string::npos) << "errors must name the model: " << msg;
}

TEST(ModelConfig, RejectsKvHeadsThatDoNotDivideTheHeads) {
    llm::ModelConfig c;
    c.architecture = "llama";
    c.layers = 1;
    c.hidden_size = 256;
    c.intermediate_size = 512;
    c.attention_heads = 4;
    c.kv_heads = 3;
    c.head_dim = 64;
    c.vocab_size = 10;
    c.context_length = 16;
    c.rms_epsilon = 1e-5f;
    c.rope_theta = 10000;
    EXPECT_EQ(kind_of([&] { llm::validate_config(c); }), llm::ErrorKind::InvalidModel);
    c.kv_heads = 8;   // more KV heads than heads
    EXPECT_EQ(kind_of([&] { llm::validate_config(c); }), llm::ErrorKind::InvalidModel);
    c.kv_heads = 2;
    c.rms_epsilon = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(kind_of([&] { llm::validate_config(c); }), llm::ErrorKind::InvalidModel);
}

TEST(ModelConfig, RejectsAMissingRequiredKey) {
    TinyModel m;
    auto bytes = m.build();
    // Rename the key in place: same length, so the file stays well-formed.
    const std::string key = "llama.block_count";
    auto it = std::search(bytes.begin(), bytes.end(), key.begin(), key.end());
    ASSERT_NE(it, bytes.end());
    *(it + 6) = 'X';
    auto f = llm::GgufFile::from_memory(bytes);
    const auto msg = message_of([&] { llm::load_config(f); });
    EXPECT_NE(msg.find("llama.block_count"), std::string::npos) << msg;
}

TEST(ModelConfig, RejectsShapesOutsideTheKernels) {
    TinyModel m;
    m.dim = 128;
    m.heads = 4;   // head_dim 32: valid llama, not what the kernels are compiled for
    auto f = llm::GgufFile::from_memory(m.build());
    EXPECT_EQ(kind_of([&] { llm::load_config(f); }), llm::ErrorKind::Unsupported);
}

TEST(TensorManifest, ReportsEveryProblemAtOnce) {
    TinyModel m;
    m.drop = {"blk.0.ffn_up.weight"};
    m.replace["blk.0.attn_q.weight"] = {"blk.0.attn_q.weight", {64, 32}, llm::GgmlType::Q4_0};
    m.replace["output_norm.weight"] = {"output_norm.weight", {64}, llm::GgmlType::F16};
    auto f = llm::GgufFile::from_memory(m.build());
    const auto c = llm::load_config(f);
    const auto msg = message_of([&] { llm::validate_tensors(f, c); });
    EXPECT_NE(msg.find("3 tensor problem"), std::string::npos) << msg;
    EXPECT_NE(msg.find("blk.0.ffn_up.weight: missing"), std::string::npos) << msg;
    EXPECT_NE(msg.find("blk.0.attn_q.weight: shape"), std::string::npos) << msg;
    EXPECT_NE(msg.find("output_norm.weight: type F16"), std::string::npos) << msg;
}

TEST(TensorManifest, CountsDeviceBytesForWhatIsUploaded) {
    auto f = llm::GgufFile::from_memory(TinyModel{}.build());
    const auto c = llm::load_config(f);
    // Q4_0 matrices: n/2 nibble bytes + n/32 float scales. Norms and the
    // output projection as float32. The embedding stays in the mapped file.
    auto q4 = [](std::uint64_t n) { return n / 2 + n / 32 * 4; };
    const std::uint64_t D = 64, F = 128, V = 300;
    const std::uint64_t expect = D * 4 + D * V * 4 +
                                 2 * D * 4 + q4(D * D) * 4 + q4(D * F) * 2 + q4(F * D);
    EXPECT_EQ(llm::weight_device_bytes(f, c), expect);
}

// ======================================================== page allocator

TEST(PageAllocator, HandsOutLowIdsFirstAndRecyclesThem) {
    llm::PageAllocator a(4);
    EXPECT_EQ(*a.allocate(), 0);
    EXPECT_EQ(*a.allocate(), 1);
    a.release(0);
    EXPECT_EQ(*a.allocate(), 0);
    EXPECT_EQ(a.free_count(), 2);
}

TEST(PageAllocator, AllocateManyIsAllOrNothing) {
    llm::PageAllocator a(3);
    ASSERT_TRUE(a.allocate().has_value());
    EXPECT_FALSE(a.allocate_many(3).has_value());
    EXPECT_EQ(a.free_count(), 2) << "a failed request must not take any pages";
    EXPECT_EQ(a.allocate_many(2)->size(), 2u);
    EXPECT_EQ(a.free_count(), 0);
}

TEST(PageAllocator, SharedPagesReturnOnlyAfterTheLastRelease) {
    llm::PageAllocator a(2);
    const int p = *a.allocate();
    a.retain(p);
    EXPECT_EQ(a.refcount(p), 2);
    a.release(p);
    EXPECT_EQ(a.free_count(), 1);
    a.release(p);
    EXPECT_EQ(a.free_count(), 2);
}

TEST(PageAllocator, RejectsMisuseLoudly) {
    llm::PageAllocator a(2);
    EXPECT_THROW(a.release(0), llm::Error) << "double release (never allocated)";
    EXPECT_THROW(a.retain(1), llm::Error) << "retain of a free page";
    EXPECT_THROW(a.release(7), llm::Error) << "out of range";
    EXPECT_THROW(llm::PageAllocator(0), llm::Error);
}

TEST(PageAllocator, IsSafeUnderConcurrentUse) {
    llm::PageAllocator a(64);
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 2000; ++i) {
                auto p = a.allocate();
                if (!p) continue;
                a.retain(*p);
                a.release(*p);
                if (a.refcount(*p) != 1) ++failures;
                a.release(*p);
            }
        });
    for (auto& th : threads) th.join();
    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(a.free_count(), 64) << "every page must come back";
}

// =============================================================== sampler

TEST(Sampler, GreedyPicksTheMaximumAndBreaksTiesLow) {
    llm::Sampler s({0.0f});
    EXPECT_EQ(s.sample({0.1f, 3.0f, 2.0f}), 1);
    EXPECT_EQ(s.sample({5.0f, 5.0f, 1.0f}), 0);
}

TEST(Sampler, RejectsNaNLogits) {
    llm::Sampler s({0.7f});
    EXPECT_EQ(kind_of([&] { s.sample({1.0f, std::numeric_limits<float>::quiet_NaN()}); }),
              llm::ErrorKind::Device);
}

TEST(Sampler, NegativeInfinityMeansNever) {
    const float ninf = -std::numeric_limits<float>::infinity();
    llm::Sampler s({1.0f, 0, 1.0f, 7});
    for (int i = 0; i < 200; ++i) EXPECT_NE(s.sample({ninf, 0.0f, ninf, 0.5f}) % 2, 0);
    EXPECT_EQ(kind_of([&] { s.sample({ninf, ninf}); }), llm::ErrorKind::InvalidArgument);
}

TEST(Sampler, PositiveInfinityTakesAllTheProbability) {
    const float inf = std::numeric_limits<float>::infinity();
    llm::Sampler s({1.0f, 0, 1.0f, 3});
    for (int i = 0; i < 100; ++i) {
        const int t = s.sample({0.0f, inf, 100.0f, inf});
        EXPECT_TRUE(t == 1 || t == 3);
    }
}

TEST(Sampler, TopKLimitsTheCandidates) {
    llm::Sampler s({1.0f, 2, 1.0f, 11});
    std::set<int> seen;
    for (int i = 0; i < 400; ++i) seen.insert(s.sample({1.0f, 5.0f, 4.9f, 0.0f}));
    EXPECT_EQ(seen, (std::set<int>{1, 2}));
}

TEST(Sampler, SameSeedSameTokens) {
    std::vector<float> l = {0.3f, 0.2f, 0.25f, 0.1f, 0.15f};
    llm::Sampler a({0.9f, 0, 0.95f, 42}), b({0.9f, 0, 0.95f, 42});
    for (int i = 0; i < 50; ++i) ASSERT_EQ(a.sample(l), b.sample(l));
}

TEST(Sampler, RejectsInvalidParameters) {
    EXPECT_THROW(llm::Sampler({-1.0f}), llm::Error);
    EXPECT_THROW(llm::Sampler({1.0f, 40, 0.0f}), llm::Error);
    EXPECT_THROW(llm::Sampler({std::numeric_limits<float>::infinity()}), llm::Error);
}

// ================================================================ errors

TEST(Errors, RenderEveryContextField) {
    llm::ErrorContext c;
    c.operation = "step";
    c.model = "m.gguf";
    c.tensor = "blk.3.attn_q.weight";
    c.shape = {2048, 2048};
    c.device = 0;
    c.cuda_error = "cudaErrorMemoryAllocation";
    c.sequence = 7;
    c.position = 512;
    const llm::Error e(llm::ErrorKind::OutOfMemory, "allocation failed", c);
    const std::string s = e.what();
    for (const char* frag : {"out of memory during step: allocation failed", "model=m.gguf",
                             "tensor=blk.3.attn_q.weight", "shape=2048x2048", "device=0",
                             "cuda=cudaErrorMemoryAllocation", "sequence=7", "position=512"})
        EXPECT_NE(s.find(frag), std::string::npos) << frag << " missing from: " << s;
    EXPECT_EQ(e.kind(), llm::ErrorKind::OutOfMemory);
}

// ========================================================= dequantization

TEST(Quant, Fp16DecodesKnownValues) {
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0x3C00), 1.0f);
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0xC000), -2.0f);
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0x3800), 0.5f);
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0x0001), std::ldexp(1.0f, -24));
    EXPECT_TRUE(std::isinf(llm::fp16_to_float(0x7C00)));
}

TEST(Quant, Q4_0SplitHalfNibbleOrder) {
    std::uint8_t blk[18] = {0x00, 0x3C};
    for (int j = 0; j < 16; ++j) blk[2 + j] = std::uint8_t(j | ((15 - j) << 4));
    float out[32];
    llm::dequantize_q4_0(blk, 32, out);
    for (int j = 0; j < 16; ++j) {
        EXPECT_FLOAT_EQ(out[j], float(j - 8));
        EXPECT_FLOAT_EQ(out[16 + j], float(15 - j - 8));
    }
}

TEST(Quant, Q6_KDecodesTheMeasuredBitLayout) {
    std::uint8_t blk[210] = {};
    std::vector<int> codes(256);
    for (int j = 0; j < 256; ++j) codes[j] = (j * 37 + 11) % 64;
    for (int j = 0; j < 256; ++j) {
        const int half = j / 128, k = j % 128;
        const int lo = codes[j] & 15, hi = codes[j] >> 4;
        blk[64 * half + k % 64] |= std::uint8_t(k < 64 ? lo : lo << 4);
        blk[128 + 32 * half + k % 32] |= std::uint8_t(hi << (2 * (k / 32)));
    }
    for (int s = 0; s < 16; ++s) blk[192 + s] = std::uint8_t(s + 1);
    blk[209] = 0x3C;
    float out[256];
    llm::dequantize_q6_k(blk, 256, out);
    for (int j = 0; j < 256; ++j) ASSERT_FLOAT_EQ(out[j], float((j / 16) + 1) * float(codes[j] - 32));
}

TEST(Quant, RejectsPartialBlocks) {
    std::uint8_t b[256] = {};
    float out[256];
    EXPECT_THROW(llm::dequantize_q4_0(b, 31, out), std::invalid_argument);
    EXPECT_THROW(llm::dequantize_q6_k(b, 255, out), std::invalid_argument);
}

// ====================================================== tokenizer (model)

namespace {
std::string model_path() {
    if (const char* env = std::getenv("CUDA_PORTFOLIO_MODEL")) return env;
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
}
std::unique_ptr<llm::Tokenizer>& tok() {
    static std::unique_ptr<llm::GgufFile> f;
    static std::unique_ptr<llm::Tokenizer> t;
    if (!t && std::filesystem::exists(model_path())) {
        f = std::make_unique<llm::GgufFile>(llm::GgufFile::open(model_path()));
        t = std::make_unique<llm::Tokenizer>(*f);
    }
    return t;
}
#define REQUIRE_TOKENIZER() \
    if (!tok()) GTEST_SKIP() << "model not found at " << model_path()
}  // namespace

TEST(Tokenizer, EncodesKnownLlamaIds) {
    REQUIRE_TOKENIZER();
    EXPECT_EQ(tok()->encode("The capital of France is", true), (std::vector<int>{1, 450, 7483, 310, 3444, 338}));
}

TEST(Tokenizer, RoundTripsAsciiAndUnicode) {
    REQUIRE_TOKENIZER();
    for (const std::string s : {"Hello world", "  two  spaces", "na\xC3\xAFve caf\xC3\xA9",
                                "emoji \xF0\x9F\x9A\x80 rocket", "tabs\tand\nnewlines"})
        EXPECT_EQ(tok()->decode(tok()->encode(s, false)), s) << s;
}

TEST(Tokenizer, FallsBackToBytesForUnknownCharacters) {
    REQUIRE_TOKENIZER();
    const std::string pua = "\xEE\x80\x80";
    const auto ids = tok()->encode(pua, false);
    EXPECT_GE(ids.size(), 3u);
    EXPECT_EQ(tok()->decode(ids), pua);
}

TEST(Tokenizer, RecognizesControlTokensWrittenAsText) {
    REQUIRE_TOKENIZER();
    const auto ids = tok()->encode("<|user|>\nhi</s>\n<|assistant|>\n", true);
    EXPECT_EQ(std::count(ids.begin(), ids.end(), tok()->eos()), 1);
}
