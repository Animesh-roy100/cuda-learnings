// Generate a reply with TinyLlama-Chat.
//
//   chat "Explain what a GPU kernel is in two sentences."
//   chat --temp 0.7 --max 200 "Write a haiku about memory bandwidth."
//   chat --model path/to/model.gguf --float --no-graphs "..."

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "cu/device.hpp"
#include "engine.h"

int main(int argc, char** argv) try {
    llm::EngineConfig cfg;
    llm::Sampling sampling;
    std::string model = llm::default_model_path();
    std::string prompt = "What is a GPU, in two sentences?";
    int max_new = 128;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--max") max_new = std::stoi(next());
        else if (a == "--temp") sampling.temperature = std::stof(next());
        else if (a == "--seed") sampling.seed = std::stoull(next());
        else if (a == "--float") cfg.activations = llm::Activations::Float;
        else if (a == "--no-graphs") cfg.cuda_graphs = false;
        else prompt = a;
    }

    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("model: %s\n", model.c_str());
    llm::Engine engine(model, cfg);
    const auto& m = engine.info();
    std::printf("%d layers, dim %d, %d heads / %d kv, ffn %d, vocab %d -- %.0f MB on device\n",
                m.layers, m.dim, m.heads, m.kv_heads, m.ffn, m.vocab,
                engine.device_bytes() / (1024.0 * 1024.0));
    std::printf("activations: %s, CUDA graphs: %s\n\n",
                cfg.activations == llm::Activations::Int8 ? "int8 (W4A8)" : "float (W4A16)",
                cfg.cuda_graphs ? "on" : "off");

    std::printf("> %s\n\n", prompt.c_str());
    llm::GenerationStats st;
    bool first = true;
    engine.generate(llm::chat_prompt(prompt), max_new, sampling, &st, [&](const std::string& piece) {
        std::fputs(first && !piece.empty() && piece[0] == ' ' ? piece.c_str() + 1 : piece.c_str(), stdout);
        first = false;
        std::fflush(stdout);
    });
    std::printf("\n\n[prefill %d tokens: %.1f tok/s | decode %d tokens: %.1f tok/s%s]\n",
                st.prompt_tokens, st.prefill_tokens_per_s(), st.generated_tokens,
                st.decode_tokens_per_s(), st.stopped_on_eos ? " | stopped at </s>" : "");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
