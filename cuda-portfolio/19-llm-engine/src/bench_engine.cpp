// End-to-end engine benchmark: quality (perplexity) and speed (tokens/s) for
// each activation path, with and without CUDA graphs.

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "cu/device.hpp"
#include "engine.h"

namespace {

// Public domain: the opening of Lewis Carroll's "Alice's Adventures in
// Wonderland" (1865).
const char* kPassage =
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of "
    "having nothing to do: once or twice she had peeped into the book her sister was "
    "reading, but it had no pictures or conversations in it, \"and what is the use of a "
    "book,\" thought Alice \"without pictures or conversations?\" So she was considering "
    "in her own mind (as well as she could, for the hot day made her feel very sleepy and "
    "stupid), whether the pleasure of making a daisy-chain would be worth the trouble of "
    "getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran "
    "close by her.";

const char* kPrompt = "Explain in three sentences why GPUs are good at matrix multiplication.";

struct Row {
    const char* name;
    llm::Activations act;
    bool graphs;
};

}  // namespace

int main(int argc, char** argv) try {
    const std::string model = argc > 1 ? argv[1] : llm::default_model_path();
    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("model: %s\n", model.c_str());

    const Row rows[] = {
        {"float (W4A16), no graphs", llm::Activations::Float, false},
        {"int8  (W4A8),  no graphs", llm::Activations::Int8, false},
        {"int8  (W4A8),  CUDA graphs", llm::Activations::Int8, true},
    };

    std::printf("\n%-28s %11s %13s %13s %9s\n", "configuration", "perplexity", "prefill tok/s",
                "decode tok/s", "MB");
    std::string sample;
    for (const auto& r : rows) {
        llm::EngineConfig cfg;
        cfg.activations = r.act;
        cfg.cuda_graphs = r.graphs;
        llm::Engine engine(model, cfg);

        const double ppl = engine.perplexity(kPassage);

        // Warm-up generation, then the measured one.
        llm::Sampling greedy;
        llm::GenerationStats st;
        engine.generate(llm::chat_prompt(kPrompt), 8, greedy, &st);
        const std::string text = engine.generate(llm::chat_prompt(kPrompt), 96, greedy, &st);
        if (r.graphs) sample = text;

        std::printf("%-28s %11.3f %13.1f %13.1f %9.0f\n", r.name, ppl, st.prefill_tokens_per_s(),
                    st.decode_tokens_per_s(), engine.device_bytes() / (1024.0 * 1024.0));
    }

    // Per-token latency as the sequence grows. Decode measured slower than
    // prefill even though both run the same single-token step, so time the
    // step itself at increasing positions rather than guess why.
    {
        std::printf("\nStep latency by position (int8, CUDA graphs), mean of 16 steps each:\n");
        std::printf("  %9s %14s %15s\n", "position", "single warp", "warp per head");
        const int checkpoints[] = {16, 64, 128, 256, 512, 1008};
        double lat[2][6] = {};
        for (int which = 0; which < 2; ++which) {
            llm::EngineConfig cfg;
            cfg.attention = which == 0 ? llm::DecodeAttention::SingleWarp
                                       : llm::DecodeAttention::WarpPerHead;
            llm::Engine engine(model, cfg);
            int pos = 0;
            for (int i = 0; i < 6; ++i) {
                while (pos < checkpoints[i]) {
                    engine.step(1);
                    ++pos;
                }
                const auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < 16; ++k) engine.step(1);
                pos += 16;
                lat[which][i] = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0).count() / 16;
            }
        }
        for (int i = 0; i < 6; ++i)
            std::printf("  %9d %11.2f ms %12.2f ms\n", checkpoints[i], lat[0][i], lat[1][i]);
    }

    std::printf("\nGreedy sample (int8, CUDA graphs):\n> %s\n\n%s\n", kPrompt, sample.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
