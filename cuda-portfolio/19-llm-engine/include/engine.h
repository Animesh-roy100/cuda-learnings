#pragma once
//
// A single-GPU Llama inference engine: GGUF weights in, text out. No CUDA
// syntax in this header.
//
// Built for TinyLlama-1.1B-Chat in Q4_0 (22 layers, dim 2048, 32 query heads
// over 4 key/value heads, head_dim 64), and validated against a host FP32
// forward pass of the same weights rather than against another engine.
//
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "quant.h"
#include "tokenizer.h"

namespace llm {

class GgufFile;

enum class Activations {
    // Float activations against dequantized Q4 weights: exact up to rounding.
    // The validation path.
    Float,
    // Activations quantized to int8 in groups of 32 aligned with the weight
    // groups, multiplied with __dp4a -- the W4A8 GEMV of 01-gguf-inference.
    // The fast path, with a measured accuracy cost.
    Int8,
};

enum class DecodeAttention {
    // One warp, one lane per query head, one pass over the cached keys with an
    // online softmax. Every head's keys are walked by the same 32 lanes, so the
    // cost per token grows linearly with position on a single warp: measured
    // 8.7 ms at position 16 and 76.6 ms at position 1008.
    SingleWarp,
    // One block -- one warp -- per query head, its lanes striding across the
    // keys. Scores go to a small [heads][context] buffer, then max, normalizer
    // and weighted sum are each reduced across the warp's lanes. 32 heads run
    // on 32 warps instead of 1.
    WarpPerHead,
};

struct EngineConfig {
    int context = 1024;                     // KV cache length
    Activations activations = Activations::Int8;
    bool cuda_graphs = true;                // replay the per-token launches
    DecodeAttention attention = DecodeAttention::WarpPerHead;
};

struct ModelInfo {
    int layers = 0, dim = 0, heads = 0, kv_heads = 0, head_dim = 0, ffn = 0, vocab = 0;
    int trained_context = 0;
    float rope_base = 10000.0f, rms_eps = 1e-5f;
};

struct Sampling {
    float temperature = 0.0f;   // 0 = greedy
    int top_k = 40;
    float top_p = 0.95f;
    std::uint64_t seed = 1;
};

struct GenerationStats {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    bool stopped_on_eos = false;
    double prefill_tokens_per_s() const { return prompt_tokens / (prefill_ms / 1000.0); }
    double decode_tokens_per_s() const { return generated_tokens / (decode_ms / 1000.0); }
};

class Engine {
public:
    Engine(const std::string& gguf_path, const EngineConfig& cfg = {});
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const ModelInfo& info() const;
    const EngineConfig& config() const;
    const Tokenizer& tokenizer() const;
    std::size_t device_bytes() const;

    // Clears the sequence. The KV cache contents are left in place and
    // overwritten as positions are reused.
    void reset();
    int position() const;

    // Appends one token at the current position and returns the next-token
    // logits.
    std::vector<float> step(int token);

    std::string generate(const std::string& prompt, int max_new_tokens, const Sampling& s,
                         GenerationStats* stats = nullptr,
                         const std::function<void(const std::string&)>& on_text = {});

    // Teacher-forced perplexity over the text's tokens (BOS prepended), capped
    // at max_tokens and at the context length.
    double perplexity(const std::string& text, int max_tokens = 512);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// TinyLlama-Chat's template: "<|user|>\n{message}</s>\n<|assistant|>\n".
std::string chat_prompt(const std::string& user_message);

// $CUDA_PORTFOLIO_MODEL if set, else ~/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf.
std::string default_model_path();

// Next-token logits after the given tokens, computed on the host in FP32 from
// fully dequantized weights. Slow -- seconds per token -- and independent of
// every device kernel, which is the point.
std::vector<float> reference_logits(const GgufFile& f, const std::vector<int>& tokens);

}  // namespace llm
