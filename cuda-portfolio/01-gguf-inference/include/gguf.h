#pragma once
//
// GGUF file parsing -- the on-disk format llama.cpp uses.
//
// Layout (little-endian throughout):
//   magic "GGUF" | version u32 | tensor_count u64 | metadata_count u64
//   metadata_count x { key:string, type:u32, value }
//   tensor_count   x { name:string, n_dims:u32, dims:u64[], type:u32, offset:u64 }
//   padding to `general.alignment` (default 32)
//   tensor data blob
//
// The file is memory-mapped, not read: weights for a 1B model are ~700 MB at
// INT4, and mapping lets the OS page them in on demand and share them between
// processes, with no copy and no doubling of resident memory.
//
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace llm {

// ggml tensor types, values fixed by the format.
enum class GgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    // K-quants: 256 elements per super-block. Quantized model files commonly
    // keep the output projection in Q6_K even when every other matrix is Q4_0.
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    Unknown = 0xFFFFFFFFu,
};

// Bytes per block and elements per block, for the quantized layouts.
struct TypeTraits {
    int block_elems = 1;   // elements encoded per block
    int block_bytes = 4;   // bytes that block occupies
    const char* name = "?";
};
TypeTraits type_traits(GgmlType t);

using MetaValue = std::variant<std::monostate, std::int64_t, double, std::string, bool>;

struct GgufTensor {
    std::string name;
    std::vector<std::uint64_t> dims;   // ggml order: fastest-varying first
    GgmlType type = GgmlType::Unknown;
    std::uint64_t offset = 0;          // relative to the data blob
    std::uint64_t num_elements() const;
    std::uint64_t num_bytes() const;
};

class GgufFile {
public:
    GgufFile();
    ~GgufFile();
    GgufFile(GgufFile&&) noexcept;
    GgufFile& operator=(GgufFile&&) noexcept;
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    // Throws std::runtime_error with a specific reason on malformed input.
    static GgufFile open(const std::string& path);

    // Parse from an in-memory image. Used by tests so they need no real model.
    static GgufFile from_memory(std::vector<std::uint8_t> bytes);

    std::uint32_t version() const;
    const std::vector<GgufTensor>& tensors() const;
    const GgufTensor* find(const std::string& name) const;

    std::optional<std::int64_t> meta_int(const std::string& key) const;
    std::optional<double> meta_float(const std::string& key) const;
    std::optional<std::string> meta_string(const std::string& key) const;

    // Array metadata -- a tokenizer's vocabulary, scores and token types. Null
    // when the key is absent or holds a different element type. Arrays of
    // arrays are parsed past but not kept.
    const std::vector<std::string>* meta_string_array(const std::string& key) const;
    const std::vector<double>* meta_float_array(const std::string& key) const;
    const std::vector<std::int64_t>* meta_int_array(const std::string& key) const;

    // Pointer into the mapped blob. Valid while this object lives.
    const void* tensor_data(const GgufTensor& t) const;
    std::uint64_t data_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace llm
