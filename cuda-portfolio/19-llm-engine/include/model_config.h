#pragma once
//
// One validated description of the model, built from GGUF metadata before any
// device memory is touched. No CUDA syntax.
//
// Inference code reads this struct, never the metadata: a key looked up in
// twenty places is twenty places to spell it differently, and twenty places
// that never agreed on what to do when it is missing.
//
#include <cstdint>
#include <string>
#include <vector>

#include "errors.h"
#include "gguf.h"

namespace llm {

struct ModelConfig {
    std::string architecture;
    int layers = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int attention_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int context_length = 0;
    float rms_epsilon = 0.0f;
    float rope_theta = 0.0f;

    int kv_dim() const { return kv_heads * head_dim; }
};

// Reads and validates. Throws Error(InvalidModel) naming the offending key, or
// Error(Unsupported) for an architecture or shape this runtime cannot run.
// vocab_size comes from the tokenizer array, which must be present.
ModelConfig load_config(const GgufFile& f, const std::string& path = {});

// The relationships the guide requires, and the ones this runtime's kernels
// add (head_dim 64, at most 32 heads, widths that are multiples of 32).
void validate_config(const ModelConfig& c, const std::string& path = {});

// One tensor the model needs: its name, required shape (ggml order) and the
// types this runtime can load it as.
struct TensorRequirement {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::vector<GgmlType> types;
};
std::vector<TensorRequirement> required_tensors(const ModelConfig& c);

// Checks every required tensor at once and reports ALL problems in one error
// -- a model missing three tensors should say so once, not on three runs.
void validate_tensors(const GgufFile& f, const ModelConfig& c, const std::string& path = {});

// Device bytes the weights will occupy once loaded: Q4_0 as packed nibbles
// plus float32 scales, everything else as float32.
std::uint64_t weight_device_bytes(const GgufFile& f, const ModelConfig& c);

}  // namespace llm
