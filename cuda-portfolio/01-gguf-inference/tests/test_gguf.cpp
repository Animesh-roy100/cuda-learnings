// GGUF parser tests.
//
// The suite writes its own GGUF images in memory, so it needs no model
// download and can construct malformed files on purpose -- which is the only
// way to test that the parser rejects them instead of reading past the buffer.

#include <gtest/gtest.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.h"

using llm::GgmlType;
using llm::GgufFile;

namespace {

class GgufWriter {
public:
    void u32(std::uint32_t v) { raw(&v, 4); }
    void u64(std::uint64_t v) { raw(&v, 8); }
    void f32(float v) { raw(&v, 4); }
    void str(const std::string& s) {
        u64(s.size());
        raw(s.data(), s.size());
    }
    void raw(const void* p, std::size_t n) {
        auto* b = static_cast<const std::uint8_t*>(p);
        bytes.insert(bytes.end(), b, b + n);
    }
    void pad_to(std::size_t align) {
        while (bytes.size() % align) bytes.push_back(0);
    }
    std::vector<std::uint8_t> bytes;
};

// One F32 tensor plus a couple of metadata keys.
std::vector<std::uint8_t> make_simple_gguf(std::uint32_t version = 3) {
    GgufWriter w;
    w.raw("GGUF", 4);
    w.u32(version);
    w.u64(1);                      // tensor count
    w.u64(3);                      // metadata count

    w.str("general.architecture");
    w.u32(8);                      // STRING
    w.str("llama");

    w.str("llama.block_count");
    w.u32(4);                      // UINT32
    w.u32(16);

    w.str("llama.attention.layer_norm_rms_epsilon");
    w.u32(6);                      // FLOAT32
    w.f32(1e-5f);

    w.str("token_embd.weight");
    w.u32(2);                      // rank
    w.u64(64);                     // dim0
    w.u64(2);                      // dim1
    w.u32(static_cast<std::uint32_t>(GgmlType::F32));
    w.u64(0);                      // offset

    w.pad_to(32);
    for (int i = 0; i < 128; ++i) w.f32(static_cast<float>(i));
    return w.bytes;
}

}  // namespace

TEST(Gguf, ParsesHeaderAndMetadata) {
    auto f = GgufFile::from_memory(make_simple_gguf());
    EXPECT_EQ(f.version(), 3u);
    ASSERT_EQ(f.tensors().size(), 1u);

    EXPECT_EQ(f.meta_string("general.architecture").value_or(""), "llama");
    EXPECT_EQ(f.meta_int("llama.block_count").value_or(-1), 16);
    ASSERT_TRUE(f.meta_float("llama.attention.layer_norm_rms_epsilon").has_value());
    EXPECT_NEAR(*f.meta_float("llama.attention.layer_norm_rms_epsilon"), 1e-5, 1e-12);

    EXPECT_FALSE(f.meta_int("does.not.exist").has_value());
    EXPECT_FALSE(f.meta_string("llama.block_count").has_value());   // wrong type
}

TEST(Gguf, ParsesTensorShapeAndData) {
    auto f = GgufFile::from_memory(make_simple_gguf());
    const auto* t = f.find("token_embd.weight");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->type, GgmlType::F32);
    ASSERT_EQ(t->dims.size(), 2u);
    EXPECT_EQ(t->dims[0], 64u);
    EXPECT_EQ(t->dims[1], 2u);
    EXPECT_EQ(t->num_elements(), 128u);
    EXPECT_EQ(t->num_bytes(), 512u);

    const auto* data = static_cast<const float*>(f.tensor_data(*t));
    for (int i = 0; i < 128; ++i) EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
}

TEST(Gguf, FindReturnsNullForMissingTensor) {
    auto f = GgufFile::from_memory(make_simple_gguf());
    EXPECT_EQ(f.find("nope.weight"), nullptr);
}

TEST(Gguf, RejectsBadMagic) {
    auto bytes = make_simple_gguf();
    bytes[0] = 'X';
    EXPECT_THROW(GgufFile::from_memory(bytes), std::runtime_error);
}

TEST(Gguf, RejectsUnsupportedVersion) {
    EXPECT_THROW(GgufFile::from_memory(make_simple_gguf(99)), std::runtime_error);
}

// The important one: a truncated file must produce a clean error, not a read
// past the end of the mapping (which is an access violation, not a bad value).
TEST(Gguf, RejectsTruncatedFile) {
    auto bytes = make_simple_gguf();
    for (std::size_t cut : {8u, 24u, 40u, 80u, 120u}) {
        if (cut >= bytes.size()) continue;
        std::vector<std::uint8_t> t(bytes.begin(), bytes.begin() + cut);
        EXPECT_THROW(GgufFile::from_memory(t), std::runtime_error) << "cut at " << cut;
    }
}

TEST(Gguf, RejectsTensorRunningPastEndOfFile) {
    GgufWriter w;
    w.raw("GGUF", 4);
    w.u32(3);
    w.u64(1);
    w.u64(0);
    w.str("huge.weight");
    w.u32(1);
    w.u64(1 << 20);                 // claims 1M elements
    w.u32(static_cast<std::uint32_t>(GgmlType::F32));
    w.u64(0);
    w.pad_to(32);
    w.f32(1.0f);                    // but only 4 bytes of data follow
    EXPECT_THROW(GgufFile::from_memory(w.bytes), std::runtime_error);
}

TEST(Gguf, RejectsElementCountNotMultipleOfBlockSize) {
    GgufWriter w;
    w.raw("GGUF", 4);
    w.u32(3);
    w.u64(1);
    w.u64(0);
    w.str("bad.q4");
    w.u32(1);
    w.u64(33);                      // Q4_0 needs a multiple of 32
    w.u32(static_cast<std::uint32_t>(GgmlType::Q4_0));
    w.u64(0);
    w.pad_to(32);
    for (int i = 0; i < 64; ++i) w.f32(0.0f);
    EXPECT_THROW(GgufFile::from_memory(w.bytes), std::runtime_error);
}

TEST(Gguf, KeepsArrayMetadataAndParsesPastIt) {
    // A tokenizer lives in array metadata: vocabulary strings, merge scores and
    // token types. They must be kept, and parsed exactly, or every key after
    // them is read from the wrong offset.
    GgufWriter w;
    w.raw("GGUF", 4);
    w.u32(3);
    w.u64(0);
    w.u64(4);

    w.str("tokenizer.ggml.tokens");
    w.u32(9);                       // ARRAY
    w.u32(8);                       // of STRING
    w.u64(3);
    w.str("a");
    w.str("bb");
    w.str("ccc");

    w.str("tokenizer.ggml.scores");
    w.u32(9);                       // ARRAY
    w.u32(6);                       // of FLOAT32
    w.u64(2);
    w.f32(-1.5f);
    w.f32(2.25f);

    w.str("tokenizer.ggml.token_type");
    w.u32(9);                       // ARRAY
    w.u32(5);                       // of INT32
    w.u64(2);
    w.u32(1);
    w.u32(6);

    w.str("after.array");
    w.u32(5);                       // INT32
    w.u32(4242);

    w.pad_to(32);
    auto f = GgufFile::from_memory(w.bytes);
    EXPECT_EQ(f.meta_int("after.array").value_or(-1), 4242);

    const auto* tokens = f.meta_string_array("tokenizer.ggml.tokens");
    ASSERT_NE(tokens, nullptr);
    EXPECT_EQ(*tokens, (std::vector<std::string>{"a", "bb", "ccc"}));

    const auto* scores = f.meta_float_array("tokenizer.ggml.scores");
    ASSERT_NE(scores, nullptr);
    ASSERT_EQ(scores->size(), 2u);
    EXPECT_DOUBLE_EQ((*scores)[0], -1.5);
    EXPECT_DOUBLE_EQ((*scores)[1], 2.25);

    const auto* types = f.meta_int_array("tokenizer.ggml.token_type");
    ASSERT_NE(types, nullptr);
    EXPECT_EQ(*types, (std::vector<std::int64_t>{1, 6}));

    // Wrong element type or absent key: null, not a throw.
    EXPECT_EQ(f.meta_float_array("tokenizer.ggml.tokens"), nullptr);
    EXPECT_EQ(f.meta_string_array("no.such.key"), nullptr);
}

TEST(Gguf, KQuantSuperBlockSizesMatchGgml) {
    // From the ggml block_q6_K struct: 128 bytes of low 4 bits, 64 bytes of high
    // 2 bits, 16 int8 sub-block scales and one fp16 super-scale = 210 bytes per
    // 256 weights.
    EXPECT_EQ(llm::type_traits(GgmlType::Q6_K).block_elems, 256);
    EXPECT_EQ(llm::type_traits(GgmlType::Q6_K).block_bytes, 210);
    EXPECT_EQ(llm::type_traits(GgmlType::Q4_K).block_bytes, 144);
    EXPECT_EQ(llm::type_traits(GgmlType::Q8_K).block_bytes, 292);
}

TEST(Gguf, TypeTraitsMatchSpec) {
    // Q4_0: 32 weights + one fp16 scale = 18 bytes = 4.5 bits/weight.
    auto q4 = llm::type_traits(GgmlType::Q4_0);
    EXPECT_EQ(q4.block_elems, 32);
    EXPECT_EQ(q4.block_bytes, 18);
    EXPECT_DOUBLE_EQ(8.0 * q4.block_bytes / q4.block_elems, 4.5);

    EXPECT_EQ(llm::type_traits(GgmlType::F32).block_bytes, 4);
    EXPECT_EQ(llm::type_traits(GgmlType::F16).block_bytes, 2);
}

// ---------------------------------------------------------------------------
// Hostile headers. Each builds a file that is well-formed except for one lie.
// ---------------------------------------------------------------------------
namespace {

struct TensorDesc {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type = 0;   // F32
    std::uint64_t offset = 0;
};

std::vector<std::uint8_t> make_gguf(const std::vector<TensorDesc>& tensors, std::size_t data_bytes,
                                    long long alignment = -1) {
    GgufWriter w;
    w.raw("GGUF", 4);
    w.u32(3);
    w.u64(tensors.size());
    w.u64(alignment >= 0 ? 1 : 0);
    if (alignment >= 0) {
        w.str("general.alignment");
        w.u32(4);   // UINT32
        w.u32(static_cast<std::uint32_t>(alignment));
    }
    for (const auto& t : tensors) {
        w.str(t.name);
        w.u32(static_cast<std::uint32_t>(t.dims.size()));
        for (auto d : t.dims) w.u64(d);
        w.u32(t.type);
        w.u64(t.offset);
    }
    w.pad_to(alignment > 0 ? std::size_t(alignment) : 32);
    w.bytes.resize(w.bytes.size() + data_bytes, 0);
    return w.bytes;
}

void expect_rejected(const std::vector<std::uint8_t>& bytes, const std::string& fragment) {
    try {
        GgufFile::from_memory(bytes);
        FAIL() << "accepted a file that should be rejected for: " << fragment;
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos)
            << "wrong reason: " << e.what();
    }
}

}  // namespace

TEST(GgufHostile, RejectsAZeroDimension) {
    expect_rejected(make_gguf({{"t", {4, 0}}}, 64), "zero dimension");
}

TEST(GgufHostile, RejectsDimensionsWhoseProductOverflows) {
    // 2^40 x 2^40 wraps a 64-bit count many times over.
    expect_rejected(make_gguf({{"t", {std::uint64_t(1) << 40, std::uint64_t(1) << 40}}}, 64),
                    "overflows");
}

TEST(GgufHostile, RejectsAMisalignedOffset) {
    expect_rejected(make_gguf({{"t", {4}, 0, 3}}, 64), "not aligned");
}

TEST(GgufHostile, RejectsOverlappingTensors) {
    // Two 32-byte F32 tensors, the second starting inside the first.
    expect_rejected(make_gguf({{"a", {8}, 0, 0}, {"b", {8}, 0, 16}}, 96, 16), "overlap");
}

TEST(GgufHostile, RejectsDuplicateTensorNames) {
    expect_rejected(make_gguf({{"a", {4}, 0, 0}, {"a", {4}, 0, 32}}, 64), "duplicate");
}

TEST(GgufHostile, RejectsANonPowerOfTwoAlignment) {
    expect_rejected(make_gguf({{"t", {4}, 0, 0}}, 64, 24), "power of two");
}

TEST(GgufHostile, RejectsAnOffsetPastTheEndWithoutWrapping) {
    // offset + size would wrap to a small number if added naively.
    expect_rejected(make_gguf({{"t", {4}, 0, 0xFFFFFFFFFFFFFFE0ull}}, 64), "past end");
}

TEST(GgufHostile, AcceptsAdjacentTensorsThatDoNotOverlap) {
    auto f = GgufFile::from_memory(make_gguf({{"a", {8}, 0, 0}, {"b", {8}, 0, 32}}, 64, 16));
    EXPECT_EQ(f.tensors().size(), 2u);
}

TEST(Gguf, OpenMissingFileThrows) {
    EXPECT_THROW(GgufFile::open("this_file_does_not_exist_12345.gguf"), std::runtime_error);
}
