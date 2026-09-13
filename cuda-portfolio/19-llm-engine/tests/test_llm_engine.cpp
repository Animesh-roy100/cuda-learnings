// Production runtime on the device.
//
// Needs the TinyLlama GGUF (scripts/fetch_model.*); every test skips cleanly
// without it. Organized by the levels the production-readiness guide asks for:
// integration against a reference, end-to-end behaviour, isolation between
// sequences, and failure handling.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include <thread>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine.h"
#include "generator.h"
#include "reference.h"
#include "runtime.h"

namespace {

bool have_model() { return std::filesystem::exists(llm::default_model_path()); }
#define REQUIRE_MODEL()                                                                     \
    if (!have_model()) GTEST_SKIP() << "model not found at " << llm::default_model_path() \
                                    << " (run scripts/fetch_model)"

std::shared_ptr<const llm::GgufModel>& model() {
    static std::shared_ptr<const llm::GgufModel> m;
    if (!m && have_model()) m = llm::GgufModel::open(llm::default_model_path());
    return m;
}

llm::RuntimeOptions opts(int batch = 4, llm::Activations act = llm::Activations::Int8, bool graphs = true) {
    llm::RuntimeOptions o;
    o.max_batch = batch;
    o.context = 512;
    o.activations = act;
    o.cuda_graphs = graphs;
    return o;
}

int argmax(const std::vector<float>& v) { return int(std::max_element(v.begin(), v.end()) - v.begin()); }

double rel_err(const std::vector<float>& a, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += double(a[i] - ref[i]) * (a[i] - ref[i]);
        den += double(ref[i]) * ref[i];
    }
    return std::sqrt(num / den);
}

std::vector<float> feed(llm::Runtime& rt, llm::Sequence& s, const std::vector<int>& ids) {
    std::vector<float> l;
    for (int id : ids) l = rt.step({&s}, {id}).front();
    return l;
}

std::size_t free_vram() {
    std::size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

const char* kAlice =
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of "
    "having nothing to do: once or twice she had peeped into the book her sister was "
    "reading, but it had no pictures or conversations in it.";

llm::ErrorKind kind_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const llm::Error& e) {
        return e.kind();
    }
    ADD_FAILURE() << "no llm::Error thrown";
    return llm::ErrorKind::Device;
}

}  // namespace

// ================================================== integration: reference

TEST(Reference, FloatRuntimeMatchesTheHostForwardPass) {
    REQUIRE_MODEL();
    const auto ids = model()->tokenizer().encode("The capital of France is", true);
    const auto ref = llm::reference_logits(model()->file(), ids);
    llm::Runtime rt(model(), opts(1, llm::Activations::Float, false));
    auto s = rt.new_sequence();
    const auto logits = feed(rt, *s, ids);
    EXPECT_LT(rel_err(logits, ref), 1e-4);
    EXPECT_EQ(argmax(logits), argmax(ref));
    EXPECT_EQ(model()->tokenizer().piece(argmax(ref)), " Paris");
}

TEST(Reference, Int8RuntimeAgreesOnTheAnswer) {
    REQUIRE_MODEL();
    const auto ids = model()->tokenizer().encode("The capital of France is", true);
    const auto ref = llm::reference_logits(model()->file(), ids);
    llm::Runtime rt(model(), opts(1));
    auto s = rt.new_sequence();
    const auto logits = feed(rt, *s, ids);
    EXPECT_EQ(argmax(logits), argmax(ref));
    EXPECT_LT(rel_err(logits, ref), 0.10);
}

TEST(Reference, PerplexityIsLowOnEnglishText) {
    REQUIRE_MODEL();
    for (auto act : {llm::Activations::Float, llm::Activations::Int8}) {
        llm::Runtime rt(model(), opts(1, act));
        EXPECT_LT(llm::perplexity(rt, kAlice), 20.0);
    }
}

// ============================================================ end to end

TEST(EndToEnd, GreedyChatAnswersAFactualQuestion) {
    REQUIRE_MODEL();
    llm::Engine e(llm::default_model_path());
    const auto r = e.generate(llm::chat_prompt("What is the capital of France?"), 40);
    EXPECT_NE(r.text.find("Paris"), std::string::npos) << r.text;
    EXPECT_EQ(r.metrics.finish, llm::FinishReason::EndOfSequence);
}

TEST(EndToEnd, FixedPromptAndSeedGiveTheSameTokensEveryTime) {
    REQUIRE_MODEL();
    llm::Runtime rt(model(), opts(1));
    llm::GenerationRequest req{llm::chat_prompt("Name three colors."), 24, {0.8f, 40, 0.95f, 1234}};
    const auto a = llm::generate(rt, req);
    const auto b = llm::generate(rt, req);
    EXPECT_EQ(a.tokens, b.tokens);
    EXPECT_LE(a.metrics.generated_tokens, 24);
    EXPECT_EQ(int(a.tokens.size()), a.metrics.generated_tokens);
}

TEST(EndToEnd, CudaGraphsReproduceTheStreamPathBitForBit) {
    REQUIRE_MODEL();
    const auto ids = model()->tokenizer().encode(kAlice, true);
    std::vector<std::vector<float>> runs[2];
    for (int g = 0; g < 2; ++g) {
        llm::Runtime rt(model(), opts(1, llm::Activations::Int8, g == 1));
        auto s = rt.new_sequence();
        for (int i = 0; i < 20; ++i) runs[g].push_back(rt.step({s.get()}, {ids[i]}).front());
    }
    for (std::size_t i = 0; i < runs[0].size(); ++i) ASSERT_EQ(runs[0][i], runs[1][i]) << "position " << i;
}

TEST(EndToEnd, DeviceMemoryIsStableAcrossRepeatedGenerations) {
    REQUIRE_MODEL();
    llm::Runtime rt(model(), opts(2));
    llm::GenerationRequest req{llm::chat_prompt("Count to five."), 16, {}};
    llm::generate(rt, req);   // first run captures graphs and warms cuBLAS
    const std::size_t before = free_vram();
    for (int i = 0; i < 20; ++i) llm::generate(rt, req);
    const std::size_t after = free_vram();
    EXPECT_LE(before - std::min(before, after), std::size_t(16) << 20)
        << "free VRAM fell from " << (before >> 20) << " MB to " << (after >> 20) << " MB";
    EXPECT_EQ(rt.kv_stats().pages_in_use, 0) << "finished requests must return their pages";
}

TEST(EndToEnd, CancellationStopsGeneration) {
    REQUIRE_MODEL();
    llm::Runtime rt(model(), opts(1));
    int seen = 0;
    const auto r = llm::generate(rt, {llm::chat_prompt("Write a long story."), 64, {}},
                                 [&](int, const std::string&) { return ++seen < 3; });
    EXPECT_EQ(r.metrics.finish, llm::FinishReason::Cancelled);
    EXPECT_EQ(r.metrics.generated_tokens, 3);

    llm::CancellationToken token;
    token.cancel();
    const auto c = llm::generate_batch(rt, {{llm::chat_prompt("hi"), 32, {}}}, &token);
    EXPECT_EQ(c[0].metrics.finish, llm::FinishReason::Cancelled);
    EXPECT_EQ(rt.kv_stats().pages_in_use, 0);
}

TEST(EndToEnd, MetricsRecordCarriesTheOperationalContext) {
    REQUIRE_MODEL();
    llm::Runtime rt(model(), opts(1));
    const auto r = llm::generate(rt, {llm::chat_prompt("hi"), 4, {}});
    const auto j = llm::metrics_json(rt, r.metrics);
    for (const char* key : {"\"driver_release\"", "\"driver_version\"", "\"cuda_runtime_version\"", "\"compute_capability\"",
                            "\"weights\":\"Q4_0\"", "\"activations\":\"int8\"", "\"kv_cache\"",
                            "\"utilization\"", "\"max_batch\"", "\"time_to_first_token_ms\"",
                            "\"decode_tokens_per_s\""})
        EXPECT_NE(j.find(key), std::string::npos) << key << " missing from " << j;
    EXPECT_GT(r.metrics.time_to_first_token_ms, 0.0);
}

// ============================================================= isolation

TEST(Isolation, BatchedSequencesMatchTheirSoloRunsExactly) {
    REQUIRE_MODEL();
    const auto& tok = model()->tokenizer();
    const auto a_ids = tok.encode("The capital of France is", true);
    const auto b_ids = tok.encode("One two three four five six", true);

    llm::Runtime rt(model(), opts(2));
    auto solo = rt.new_sequence();
    std::vector<std::vector<float>> solo_logits;
    for (int id : a_ids) solo_logits.push_back(rt.step({solo.get()}, {id}).front());

    auto a = rt.new_sequence();
    auto b = rt.new_sequence();
    for (std::size_t i = 0; i < a_ids.size(); ++i) {
        const auto out = rt.step({a.get(), b.get()}, {a_ids[i], b_ids[i]});
        // Different positions, different pages, shared launches -- and not a
        // bit of difference in A's logits for having B beside it.
        ASSERT_EQ(out[0], solo_logits[i]) << "position " << i;
    }
}

TEST(Isolation, ForkSharesPagesAndCopiesOnWrite) {
    REQUIRE_MODEL();
    const auto& tok = model()->tokenizer();
    const auto prefix = tok.encode("The capital of France is", true);   // 6 tokens: page 0 partly filled
    const int paris = tok.encode("Paris", false).back();
    const int london = tok.encode("London", false).back();

    llm::Runtime rt(model(), opts(2));
    auto base = rt.new_sequence();
    feed(rt, *base, prefix);
    const int pages_before = rt.kv_stats().pages_in_use;
    auto fork = base->fork();
    EXPECT_EQ(rt.kv_stats().pages_in_use, pages_before) << "a fork shares pages, it does not copy them";

    const auto from_base = rt.step({base.get()}, {paris}).front();     // copy-on-write here
    const auto from_fork = rt.step({fork.get()}, {london}).front();
    EXPECT_EQ(rt.kv_stats().pages_in_use, pages_before + 1) << "exactly one page copied";

    // Each branch must match a sequence that never shared anything.
    auto control_a = rt.new_sequence();
    auto control_b = rt.new_sequence();
    auto ids_a = prefix, ids_b = prefix;
    ids_a.push_back(paris);
    ids_b.push_back(london);
    EXPECT_EQ(feed(rt, *control_a, ids_a), from_base);
    EXPECT_EQ(feed(rt, *control_b, ids_b), from_fork);
}

TEST(Isolation, TruncateReleasesPagesAndReplaysExactly) {
    REQUIRE_MODEL();
    const auto ids = model()->tokenizer().encode(kAlice, true);
    ASSERT_GT(ids.size(), 40u);
    llm::Runtime rt(model(), opts(1));
    auto s = rt.new_sequence();
    std::vector<std::vector<float>> first;
    for (std::size_t i = 0; i < 40; ++i) first.push_back(rt.step({s.get()}, {ids[i]}).front());
    EXPECT_EQ(s->pages_held(), 3);
    s->truncate(10);
    EXPECT_EQ(s->pages_held(), 1);
    for (std::size_t i = 10; i < 40; ++i) ASSERT_EQ(rt.step({s.get()}, {ids[i]}).front(), first[i]) << i;
}

TEST(Isolation, SequencesMayOutliveTheirRuntime) {
    REQUIRE_MODEL();
    std::unique_ptr<llm::Sequence> s;
    {
        llm::Runtime rt(model(), opts(1));
        s = rt.new_sequence();
        rt.step({s.get()}, {1});
    }
    s.reset();   // returns pages to bookkeeping the sequence co-owns: no crash
    SUCCEED();
}

// ======================================================== failure handling

TEST(Failure, KvExhaustionFailsCleanlyAndLeavesSequencesUsable) {
    REQUIRE_MODEL();
    auto o = opts(2);
    o.kv_pages = 2;   // 32 tokens in total
    llm::Runtime rt(model(), o);
    auto a = rt.new_sequence();
    auto b = rt.new_sequence();
    for (int i = 0; i < 16; ++i) rt.step({a.get()}, {1});   // a holds page 0 (full)
    for (int i = 0; i < 16; ++i) rt.step({b.get()}, {1});   // b holds page 1 (full)
    const int pa = a->position(), pb = b->position();

    EXPECT_EQ(kind_of([&] { rt.step({a.get(), b.get()}, {1, 1}); }), llm::ErrorKind::OutOfMemory);
    EXPECT_EQ(a->position(), pa) << "a failed step must not advance any sequence";
    EXPECT_EQ(b->position(), pb);
    EXPECT_EQ(rt.kv_stats().pages_in_use, 2) << "and must not keep pages it could not use";

    b.reset();   // free a page, and the other sequence continues
    EXPECT_NO_THROW(rt.step({a.get()}, {1}));
}

TEST(Failure, ContextFullIsReportedWithTheSequence) {
    REQUIRE_MODEL();
    auto o = opts(2);
    o.context = 16;
    o.page_tokens = 8;
    llm::Runtime rt(model(), o);
    auto full = rt.new_sequence();
    auto ok = rt.new_sequence();
    for (int i = 0; i < 16; ++i) rt.step({full.get()}, {1});
    try {
        rt.step({ok.get(), full.get()}, {1, 1});
        FAIL() << "stepped past the context length";
    } catch (const llm::Error& e) {
        EXPECT_EQ(e.kind(), llm::ErrorKind::ContextFull);
        ASSERT_TRUE(e.context().sequence.has_value());
        EXPECT_EQ(*e.context().sequence, full->id());
    }
    EXPECT_EQ(ok->position(), 0) << "the transaction includes the sequence that could have stepped";
}

TEST(Failure, InvalidStepsAreRejectedBeforeAnyKernelRuns) {
    REQUIRE_MODEL();
    llm::Runtime rt(model(), opts(2));
    llm::Runtime other(model(), opts(1));
    auto s = rt.new_sequence();
    auto foreign = other.new_sequence();
    EXPECT_EQ(kind_of([&] { rt.step({}, {}); }), llm::ErrorKind::InvalidArgument);
    EXPECT_EQ(kind_of([&] { rt.step({s.get(), s.get()}, {1, 1}); }), llm::ErrorKind::InvalidArgument);
    EXPECT_EQ(kind_of([&] { rt.step({foreign.get()}, {1}); }), llm::ErrorKind::InvalidArgument);
    EXPECT_EQ(kind_of([&] { rt.step({s.get()}, {-1}); }), llm::ErrorKind::InvalidArgument);
    EXPECT_EQ(kind_of([&] { rt.step({s.get()}, {1, 2}); }), llm::ErrorKind::InvalidArgument);
    EXPECT_EQ(s->position(), 0);
    EXPECT_NO_THROW(rt.step({s.get()}, {1})) << "and the runtime is still healthy";
}

TEST(Failure, AMemoryPlanLargerThanTheDeviceIsRefusedBeforeAllocating) {
    REQUIRE_MODEL();
    auto o = opts(64);
    o.context = 2048;                       // 64 full-context sequences: ~11 GB of KV cache
    const std::size_t before = free_vram();
    try {
        llm::Runtime rt(model(), o);
        FAIL() << "loaded a runtime that cannot fit";
    } catch (const llm::Error& e) {
        EXPECT_EQ(e.kind(), llm::ErrorKind::OutOfMemory);
        EXPECT_NE(std::string(e.what()).find("KV cache"), std::string::npos) << e.what();
    }
    EXPECT_LE(before - std::min(before, free_vram()), std::size_t(16) << 20)
        << "the refusal must come before any allocation";
}

// Two threads stepping one runtime: each call either completes or is refused
// with InvalidArgument -- never interleaves -- and every sequence ends where its
// successful steps put it, with the logits a solo run produces.
TEST(Failure, ConcurrentStepsAreRefusedNotInterleaved) {
    REQUIRE_MODEL();
    llm::Runtime rt(model(), opts(2));
    auto a = rt.new_sequence();
    auto b = rt.new_sequence();
    std::atomic<int> refused{0}, other{0};
    std::vector<int> done(2, 0);
    auto worker = [&](llm::Sequence* s, int index) {
        for (int i = 0; i < 40 && s->position() < 30; ++i) {
            try {
                rt.step({s}, {1});
                ++done[index];
            } catch (const llm::Error& e) {
                (e.kind() == llm::ErrorKind::InvalidArgument ? refused : other)++;
            }
        }
    };
    std::thread ta(worker, a.get(), 0), tb(worker, b.get(), 1);
    ta.join();
    tb.join();
    std::printf("  completed %d + %d steps, refused %d concurrent calls\n", done[0], done[1], refused.load());
    EXPECT_EQ(other.load(), 0);
    EXPECT_EQ(a->position(), done[0]);
    EXPECT_EQ(b->position(), done[1]);

    auto control = rt.new_sequence();
    std::vector<float> expected;
    for (int i = 0; i <= a->position(); ++i) expected = rt.step({control.get()}, {1}).front();
    EXPECT_EQ(rt.step({a.get()}, {1}).front(), expected) << "a's cache must hold exactly its own steps";
}

TEST(Failure, OptionsAreValidated) {
    REQUIRE_MODEL();
    auto bad_ctx = opts();
    bad_ctx.context = 100000;
    EXPECT_EQ(kind_of([&] { llm::Runtime(model(), bad_ctx); }), llm::ErrorKind::InvalidArgument);
    auto bad_batch = opts();
    bad_batch.max_batch = 0;
    EXPECT_EQ(kind_of([&] { llm::Runtime(model(), bad_batch); }), llm::ErrorKind::InvalidArgument);
}

TEST(Failure, AMissingModelFileIsAnInvalidModelError) {
    EXPECT_EQ(kind_of([] { llm::GgufModel::open("no/such/model.gguf"); }), llm::ErrorKind::InvalidModel);
}
