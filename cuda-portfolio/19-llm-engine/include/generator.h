#pragma once
//
// Text generation on top of the runtime: prompts in, tokens out, with
// cancellation, finish reasons and metrics. No CUDA syntax.
//
#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "runtime.h"
#include "sampler.h"

namespace llm {

enum class FinishReason {
    Length,          // max_new_tokens reached
    EndOfSequence,   // the model emitted </s>
    Cancelled,       // callback returned false or the token was cancelled
    ContextFull,     // the sequence reached the runtime's context length
};
const char* to_string(FinishReason r);

class CancellationToken {
public:
    void cancel() noexcept { flag_.store(true); }
    bool cancelled() const noexcept { return flag_.load(); }

private:
    std::atomic<bool> flag_{false};
};

struct GenerationRequest {
    std::string prompt;
    int max_new_tokens = 128;
    SamplingParams sampling;
    bool add_bos = true;
};

struct GenerationMetrics {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double time_to_first_token_ms = 0.0;   // request start to first sampled token
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    FinishReason finish = FinishReason::Length;
    double prefill_tokens_per_s() const { return prefill_ms > 0 ? prompt_tokens / (prefill_ms / 1000.0) : 0.0; }
    double decode_tokens_per_s() const { return decode_ms > 0 ? generated_tokens / (decode_ms / 1000.0) : 0.0; }
};

struct GenerationResult {
    std::string text;
    std::vector<int> tokens;
    GenerationMetrics metrics;
};

// Called with each generated token and its text. Return false to stop.
using TokenCallback = std::function<bool(int token, const std::string& piece)>;

// One request on a fresh sequence.
GenerationResult generate(Runtime& rt, const GenerationRequest& req, const TokenCallback& on_token = {},
                          const CancellationToken* cancel = nullptr);

// Several requests stepped together: every step advances every unfinished
// request by one token -- a prompt token while prefilling, a sampled token
// after -- so short prompts start generating while long ones still prefill.
// At most runtime.options().max_batch requests may run at once.
std::vector<GenerationResult> generate_batch(Runtime& rt, const std::vector<GenerationRequest>& reqs,
                                             const CancellationToken* cancel = nullptr);

// Teacher-forced perplexity over the text's tokens (BOS prepended).
double perplexity(Runtime& rt, const std::string& text, int max_tokens = 512);

// One JSON object with everything an operator needs to interpret a number:
// device, driver and CUDA versions, quantization and activation format,
// memory plan, KV utilization, batch and context limits, and the metrics.
std::string metrics_json(const Runtime& rt, const GenerationMetrics& m);

}  // namespace llm
