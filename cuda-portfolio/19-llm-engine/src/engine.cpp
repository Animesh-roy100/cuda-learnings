#include "engine.h"

#include <cstdlib>

namespace llm {

Engine::Engine(const std::string& gguf_path, const EngineConfig& cfg) {
    model_ = GgufModel::open(gguf_path);
    RuntimeOptions o;
    o.context = cfg.context;
    o.activations = cfg.activations;
    o.cuda_graphs = cfg.cuda_graphs;
    o.max_batch = 1;
    runtime_ = std::make_unique<Runtime>(model_, o);
    seq_ = runtime_->new_sequence();

    const auto& c = model_->config();
    info_.layers = c.layers;
    info_.dim = c.hidden_size;
    info_.heads = c.attention_heads;
    info_.kv_heads = c.kv_heads;
    info_.head_dim = c.head_dim;
    info_.ffn = c.intermediate_size;
    info_.vocab = c.vocab_size;
    info_.trained_context = c.context_length;
    info_.rope_base = c.rope_theta;
    info_.rms_eps = c.rms_epsilon;
}

Engine::~Engine() {
    seq_.reset();   // pages back before the runtime goes
}

void Engine::reset() {
    seq_.reset();
    seq_ = runtime_->new_sequence();
}

std::vector<float> Engine::step(int token) {
    return runtime_->step({seq_.get()}, {token}).front();
}

GenerationResult Engine::generate(const std::string& prompt, int max_new_tokens, const SamplingParams& s,
                                  const std::function<void(const std::string&)>& on_text) {
    GenerationRequest req;
    req.prompt = prompt;
    req.max_new_tokens = max_new_tokens;
    req.sampling = s;
    TokenCallback cb;
    if (on_text)
        cb = [&](int, const std::string& piece) {
            on_text(piece);
            return true;
        };
    return llm::generate(*runtime_, req, cb);
}

double Engine::perplexity(const std::string& text, int max_tokens) {
    return llm::perplexity(*runtime_, text, max_tokens);
}

std::string chat_prompt(const std::string& user_message) {
    return "<|user|>\n" + user_message + "</s>\n<|assistant|>\n";
}

std::string default_model_path() {
    if (const char* env = std::getenv("CUDA_PORTFOLIO_MODEL")) return env;
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
}

}  // namespace llm
