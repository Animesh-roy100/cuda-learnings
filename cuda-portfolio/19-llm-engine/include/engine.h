#pragma once
//
// The one-sequence convenience interface: open a model, step tokens, generate.
// A thin facade over the production runtime (runtime.h, generator.h), which is
// what to use for batching, several sequences, forking, cancellation and
// metrics. No CUDA syntax.
//
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "generator.h"
#include "reference.h"
#include "runtime.h"
#include "sampler.h"
#include "tokenizer.h"

namespace llm {

class GgufFile;

struct EngineConfig {
    int context = 1024;
    Activations activations = Activations::Auto;
    bool cuda_graphs = true;
};

struct ModelInfo {
    int layers = 0, dim = 0, heads = 0, kv_heads = 0, head_dim = 0, ffn = 0, vocab = 0;
    int trained_context = 0;
    float rope_base = 10000.0f, rms_eps = 1e-5f;
};

class Engine {
public:
    explicit Engine(const std::string& gguf_path, const EngineConfig& cfg = {});
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const ModelInfo& info() const { return info_; }
    const Tokenizer& tokenizer() const { return model_->tokenizer(); }
    Runtime& runtime() { return *runtime_; }
    std::size_t device_bytes() const { return std::size_t(runtime_->memory().total()); }

    void reset();                        // start a new sequence
    int position() const { return seq_->position(); }
    std::vector<float> step(int token);  // next-token logits

    GenerationResult generate(const std::string& prompt, int max_new_tokens,
                              const SamplingParams& s = {},
                              const std::function<void(const std::string&)>& on_text = {});
    double perplexity(const std::string& text, int max_tokens = 512);

private:
    std::shared_ptr<const GgufModel> model_;
    std::unique_ptr<Runtime> runtime_;
    std::unique_ptr<Sequence> seq_;
    ModelInfo info_;
};

// TinyLlama-Chat's template: "<|user|>\n{message}</s>\n<|assistant|>\n".
std::string chat_prompt(const std::string& user_message);

// $CUDA_PORTFOLIO_MODEL if set, else ~/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf.
std::string default_model_path();

}  // namespace llm
