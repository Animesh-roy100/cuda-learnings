// LLM engine.
//
// The model-dependent tests need the TinyLlama GGUF (scripts/fetch_model.*)
// and skip cleanly without it, so CI -- which has neither the file nor a GPU --
// still compiles them. Everything that can be checked without the model is.
//
// The central check is independent of every device kernel: the host reference
// forward pass in reference.cpp, FP32 from dequantized weights.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "gguf.h"
#include "quant.h"
#include "tokenizer.h"

namespace {

bool have_model() { return std::filesystem::exists(llm::default_model_path()); }

#define REQUIRE_MODEL()                                                          \
    if (!have_model()) GTEST_SKIP() << "model not found at " << llm::default_model_path() \
                                    << " (run scripts/fetch_model)"

int argmax(const std::vector<float>& v) {
    return int(std::max_element(v.begin(), v.end()) - v.begin());
}

double rel_err(const std::vector<float>& a, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += double(a[i] - ref[i]) * (a[i] - ref[i]);
        den += double(ref[i]) * ref[i];
    }
    return std::sqrt(num / den);
}

// One GgufFile and Tokenizer for the whole suite: parsing the vocabulary and
// merges is the slow part of opening the file.
struct Shared {
    std::unique_ptr<llm::GgufFile> file;
    std::unique_ptr<llm::Tokenizer> tok;
};
Shared& shared() {
    static Shared s;
    if (!s.file && have_model()) {
        s.file = std::make_unique<llm::GgufFile>(llm::GgufFile::open(llm::default_model_path()));
        s.tok = std::make_unique<llm::Tokenizer>(*s.file);
    }
    return s;
}

const char* kAlice =
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of "
    "having nothing to do: once or twice she had peeped into the book her sister was "
    "reading, but it had no pictures or conversations in it.";

}  // namespace

// =====================================================================
// Without the model
// =====================================================================

TEST(Fp16, DecodesKnownValues) {
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0x3C00), 1.0f);
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0x4000), 2.0f);
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0xC000), -2.0f);
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0x3800), 0.5f);
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0x0000), 0.0f);
    EXPECT_FLOAT_EQ(llm::fp16_to_float(0x0001), std::ldexp(1.0f, -24));   // smallest subnormal
    EXPECT_TRUE(std::isinf(llm::fp16_to_float(0x7C00)));
}

TEST(Q4_0, SplitHalfNibbleOrder) {
    // One block: scale 1.0, weight j = j - 8 for j in 0..15 (low nibbles) and
    // 15 - (j - 16) - 8 for j in 16..31 (high nibbles).
    std::uint8_t blk[18] = {0x00, 0x3C};
    for (int j = 0; j < 16; ++j) blk[2 + j] = std::uint8_t(j | ((15 - j) << 4));
    float out[32];
    llm::dequantize_q4_0(blk, 32, out);
    for (int j = 0; j < 16; ++j) {
        EXPECT_FLOAT_EQ(out[j], float(j - 8)) << j;
        EXPECT_FLOAT_EQ(out[16 + j], float(15 - j - 8)) << 16 + j;
    }
}

TEST(Q6_K, DecodesTheMeasuredBitLayout) {
    // Encode 256 known 6-bit codes with the layout quant.h documents, then
    // decode. This checks the decoder matches its own documentation; that the
    // documentation matches the file is what ProducesLowPerplexity checks.
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
    blk[208] = 0x00;
    blk[209] = 0x3C;   // super-scale 1.0
    float out[256];
    llm::dequantize_q6_k(blk, 256, out);
    for (int j = 0; j < 256; ++j)
        ASSERT_FLOAT_EQ(out[j], float((j / 16) + 1) * float(codes[j] - 32)) << "weight " << j;
}

TEST(Quant, RejectsPartialBlocks) {
    std::uint8_t b[256] = {};
    float out[256];
    EXPECT_THROW(llm::dequantize_q4_0(b, 31, out), std::invalid_argument);
    EXPECT_THROW(llm::dequantize_q6_k(b, 255, out), std::invalid_argument);
}

TEST(ChatTemplate, MatchesTinyLlama) {
    EXPECT_EQ(llm::chat_prompt("hi"), "<|user|>\nhi</s>\n<|assistant|>\n");
}

// =====================================================================
// Tokenizer (model file needed for the vocabulary)
// =====================================================================

TEST(Tokenizer, EncodesKnownLlamaIds) {
    REQUIRE_MODEL();
    const auto ids = shared().tok->encode("The capital of France is", true);
    EXPECT_EQ(ids, (std::vector<int>{1, 450, 7483, 310, 3444, 338}));
}

TEST(Tokenizer, RoundTripsAsciiAndUnicode) {
    REQUIRE_MODEL();
    const auto& tok = *shared().tok;
    for (const std::string s : {"Hello world", "  two  spaces", "naïve café",
                                "日本語のテキスト", "emoji \xF0\x9F\x9A\x80 rocket", "tabs\tand\nnewlines"}) {
        EXPECT_EQ(tok.decode(tok.encode(s, false)), s) << s;
    }
}

TEST(Tokenizer, FallsBackToBytesForUnknownCharacters) {
    REQUIRE_MODEL();
    const auto& tok = *shared().tok;
    // A private-use code point has no vocabulary entry: its three UTF-8 bytes
    // must come out as three <0xXX> tokens and decode back to the character.
    const std::string pua = "\xEE\x80\x80";
    const auto ids = tok.encode(pua, false);
    EXPECT_GE(ids.size(), 3u);
    EXPECT_EQ(tok.decode(ids), pua);
}

TEST(Tokenizer, RecognizesControlTokensWrittenAsText) {
    REQUIRE_MODEL();
    const auto& tok = *shared().tok;
    const auto ids = tok.encode(llm::chat_prompt("hi"), true);
    EXPECT_EQ(std::count(ids.begin(), ids.end(), tok.eos()), 1)
        << "the template's literal </s> must become the end-of-sequence token";
}

// =====================================================================
// Engine against the host reference
// =====================================================================

TEST(Engine, FloatPathMatchesTheHostReference) {
    REQUIRE_MODEL();
    const auto& s = shared();
    const auto ids = s.tok->encode("The capital of France is", true);
    const auto ref = llm::reference_logits(*s.file, ids);

    llm::EngineConfig cfg;
    cfg.activations = llm::Activations::Float;
    cfg.cuda_graphs = false;
    llm::Engine e(llm::default_model_path(), cfg);
    std::vector<float> logits;
    for (int id : ids) logits = e.step(id);

    EXPECT_LT(rel_err(logits, ref), 1e-4);   // measured 2.9e-6
    EXPECT_EQ(argmax(logits), argmax(ref));
    EXPECT_EQ(s.tok->piece(argmax(ref)), " Paris");
}

TEST(Engine, Int8PathAgreesWithTheReferenceOnTheAnswer) {
    REQUIRE_MODEL();
    const auto& s = shared();
    const auto ids = s.tok->encode("The capital of France is", true);
    const auto ref = llm::reference_logits(*s.file, ids);

    llm::EngineConfig cfg;
    cfg.activations = llm::Activations::Int8;
    llm::Engine e(llm::default_model_path(), cfg);
    std::vector<float> logits;
    for (int id : ids) logits = e.step(id);

    // Per-group int8 activations are lossy, so not bit-close -- but they must
    // not change the prediction, and must stay within a few percent.
    EXPECT_EQ(argmax(logits), argmax(ref));
    EXPECT_LT(rel_err(logits, ref), 0.10);
}

TEST(Engine, BothDecodeAttentionKernelsMatchTheReference) {
    REQUIRE_MODEL();
    const auto& s = shared();
    const auto ids = s.tok->encode("The capital of France is", true);
    const auto ref = llm::reference_logits(*s.file, ids);
    for (auto a : {llm::DecodeAttention::SingleWarp, llm::DecodeAttention::WarpPerHead}) {
        llm::EngineConfig cfg;
        cfg.activations = llm::Activations::Float;
        cfg.attention = a;
        llm::Engine e(llm::default_model_path(), cfg);
        std::vector<float> logits;
        for (int id : ids) logits = e.step(id);
        EXPECT_LT(rel_err(logits, ref), 1e-4)
            << (a == llm::DecodeAttention::SingleWarp ? "single warp" : "warp per head");
    }
}

TEST(Engine, WarpPerHeadAttentionHandlesLongerThanOneWarpOfKeys) {
    // Past 32 positions every lane has keys, and past 64 each has several: the
    // striding and all four reductions are exercised. The two kernels compute
    // the same softmax in a different order, so agree to rounding, not bits.
    REQUIRE_MODEL();
    const auto ids = shared().tok->encode(kAlice, true);
    ASSERT_GT(ids.size(), 40u);
    std::vector<float> a, b;
    for (auto att : {llm::DecodeAttention::SingleWarp, llm::DecodeAttention::WarpPerHead}) {
        llm::EngineConfig cfg;
        cfg.activations = llm::Activations::Float;
        cfg.attention = att;
        llm::Engine e(llm::default_model_path(), cfg);
        auto& dst = att == llm::DecodeAttention::SingleWarp ? a : b;
        for (int id : ids) dst = e.step(id);
    }
    EXPECT_LT(rel_err(b, a), 1e-4);
    EXPECT_EQ(argmax(a), argmax(b));
}

TEST(Engine, CudaGraphsReproduceTheStreamPathExactly) {
    REQUIRE_MODEL();
    const auto ids = shared().tok->encode(kAlice, true);
    std::vector<std::vector<float>> with, without;
    for (bool graphs : {false, true}) {
        llm::EngineConfig cfg;
        cfg.cuda_graphs = graphs;
        llm::Engine e(llm::default_model_path(), cfg);
        auto& dst = graphs ? with : without;
        for (int i = 0; i < 12; ++i) dst.push_back(e.step(ids[i]));
    }
    // A graph changes when work is submitted, never what the work is.
    for (std::size_t i = 0; i < with.size(); ++i)
        ASSERT_EQ(with[i], without[i]) << "position " << i;
}

TEST(Engine, ProducesLowPerplexityOnEnglishText) {
    REQUIRE_MODEL();
    // The test that pins the Q6_K layout: the wrong bit arrangements gave
    // perplexities in the hundreds to hundreds of thousands.
    for (auto act : {llm::Activations::Float, llm::Activations::Int8}) {
        llm::EngineConfig cfg;
        cfg.activations = act;
        llm::Engine e(llm::default_model_path(), cfg);
        EXPECT_LT(e.perplexity(kAlice), 20.0)
            << (act == llm::Activations::Float ? "float" : "int8");
    }
}

TEST(Engine, GreedyChatAnswersAFactualQuestion) {
    REQUIRE_MODEL();
    llm::Engine e(llm::default_model_path());
    llm::GenerationStats st;
    const auto reply = e.generate(llm::chat_prompt("What is the capital of France?"), 40, {}, &st);
    EXPECT_NE(reply.find("Paris"), std::string::npos) << reply;
    EXPECT_GT(st.generated_tokens, 0);
}

TEST(Engine, ResetReplaysIdenticalLogits) {
    REQUIRE_MODEL();
    llm::Engine e(llm::default_model_path());
    const auto ids = shared().tok->encode("One two three", true);
    std::vector<float> first, second;
    for (int id : ids) first = e.step(id);
    e.reset();
    EXPECT_EQ(e.position(), 0);
    for (int id : ids) second = e.step(id);
    EXPECT_EQ(first, second) << "the cache is overwritten in place, so replay must be exact";
}

TEST(Engine, RejectsBadTokensAndAFullContext) {
    REQUIRE_MODEL();
    llm::EngineConfig cfg;
    cfg.context = 4;
    llm::Engine e(llm::default_model_path(), cfg);
    EXPECT_THROW(e.step(-1), std::out_of_range);
    EXPECT_THROW(e.step(e.info().vocab), std::out_of_range);
    for (int i = 0; i < 4; ++i) e.step(1);
    EXPECT_THROW(e.step(1), std::length_error);
}

TEST(Engine, ReportsTheModelShape) {
    REQUIRE_MODEL();
    llm::Engine e(llm::default_model_path());
    const auto& m = e.info();
    EXPECT_EQ(m.layers, 22);
    EXPECT_EQ(m.dim, 2048);
    EXPECT_EQ(m.heads, 32);
    EXPECT_EQ(m.kv_heads, 4);
    EXPECT_EQ(m.head_dim, 64);
    EXPECT_EQ(m.vocab, 32000);
}
