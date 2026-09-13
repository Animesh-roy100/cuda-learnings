#pragma once
//
// The production inference runtime. No CUDA syntax.
//
// Ownership, one responsibility per object:
//
//   GgufModel   the mapped file, its validated ModelConfig and tokenizer.
//               Immutable; shared by every runtime built from it.
//   Runtime     one device: resident weights, the KV page pool, a workspace
//               sized for the largest batch, CUDA graphs per batch size.
//   Sequence    one conversation's KV state: a page table into the pool and
//               a position. Forks share pages copy-on-write.
//   Sampler     generation policy (sampler.h).
//
// Threading rules:
//   * A GgufModel may be shared freely across threads.
//   * A Runtime is driven by ONE thread at a time: step() is not re-entrant,
//     and a concurrent second call fails with Error(InvalidArgument) instead
//     of sharing the workspace. Serve several callers by batching their
//     sequences into one step(), not by calling step() from several threads.
//   * A Runtime's CUDA work runs on its own stream; step() returns only after
//     that stream is synchronized, so returned logits are final.
//   * Sequences may be created, forked and destroyed from any thread; their
//     page bookkeeping is synchronized. A sequence must not be destroyed while
//     a step() that includes it is running.
//
// Every public failure is an llm::Error (errors.h) with context.
//
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "errors.h"
#include "model_config.h"
#include "tokenizer.h"

namespace llm {

class GgufModel {
public:
    // Parses, validates the configuration and every tensor, and loads the
    // tokenizer. Nothing touches a GPU.
    static std::shared_ptr<const GgufModel> open(const std::string& path);

    const std::string& path() const { return path_; }
    const GgufFile& file() const { return *file_; }
    const ModelConfig& config() const { return config_; }
    const Tokenizer& tokenizer() const { return *tokenizer_; }

    GgufModel(const GgufModel&) = delete;
    GgufModel& operator=(const GgufModel&) = delete;
    ~GgufModel();

private:
    GgufModel() = default;
    std::string path_;
    std::unique_ptr<GgufFile> file_;
    ModelConfig config_;
    std::unique_ptr<Tokenizer> tokenizer_;
};

struct DeviceInfo {
    int ordinal = 0;
    std::string name;
    int compute_capability = 0;    // 75 for sm_75
    std::size_t total_bytes = 0;
    std::size_t free_bytes = 0;
    std::string driver_release;    // NVML, e.g. "616.92"; "unknown" without NVML
    int driver_version = 0;        // cudaDriverGetVersion: newest CUDA API the driver supports, e.g. 13040
    int runtime_version = 0;       // cudaRuntimeGetVersion
    bool dp4a = false;             // __dp4a needs sm_61+
};
DeviceInfo query_device_info(int ordinal = 0);

enum class Activations {
    Auto,    // Int8 where __dp4a exists, else Float
    Float,   // W4A16: exact against dequantized weights
    Int8,    // W4A8 via __dp4a; rejected on devices without it
};

struct RuntimeOptions {
    int device = 0;
    int context = 1024;           // maximum tokens per sequence
    int max_batch = 8;            // sequences per step
    int page_tokens = 16;         // tokens per KV page
    int kv_pages = 0;             // 0 = enough for max_batch full-context sequences
    Activations activations = Activations::Auto;
    bool cuda_graphs = true;
    // Refuse to load when the plan would leave less than this fraction of the
    // device's total memory free. Budgeted against FREE VRAM, not against what
    // cudaMalloc would allow: the Windows driver's sysmem fallback lets
    // allocations spill into system RAM, which does not fail -- it silently
    // turns every memory-bound kernel into a PCIe-bound one.
    double vram_headroom = 0.05;
    // Per-stage latency via CUDA events. Disables graphs and adds
    // synchronization, so it is for diagnosis, not for serving.
    bool profile_stages = false;
};

struct MemoryPlan {
    std::uint64_t weights = 0;
    std::uint64_t kv_cache = 0;
    std::uint64_t workspace = 0;
    std::uint64_t total() const { return weights + kv_cache + workspace; }
    std::uint64_t free_at_load = 0;
};

struct KvStats {
    int total_pages = 0;
    int pages_in_use = 0;
    int page_tokens = 0;
    int live_sequences = 0;
    double utilization() const { return total_pages ? double(pages_in_use) / total_pages : 0.0; }
};

struct StageTimes {                 // accumulated milliseconds, profile_stages only
    double embed_upload = 0, norms = 0, projections = 0, rope_and_cache = 0,
           attention = 0, activation = 0, logits = 0, download = 0;
    int steps = 0;
};

class Runtime;
namespace detail {
struct KvPoolState;   // page allocator and pool geometry, shared with sequences
}

class Sequence {
public:
    // Returns every page to the pool. Safe even after the Runtime is gone:
    // a sequence shares ownership of the host-side page bookkeeping, never of
    // the device memory.
    ~Sequence();
    Sequence(const Sequence&) = delete;
    Sequence& operator=(const Sequence&) = delete;

    int id() const { return id_; }
    int position() const { return position_; }   // tokens held
    int capacity() const;                        // the runtime's context length
    int pages_held() const { return static_cast<int>(pages_.size()); }

    // A new sequence holding the same tokens. Pages are shared, not copied;
    // the first write into a shared, partly filled page copies that page.
    std::unique_ptr<Sequence> fork() const;

    // Drops tokens past new_position and returns pages no longer needed.
    void truncate(int new_position);

private:
    friend class Runtime;
    Sequence(std::shared_ptr<detail::KvPoolState> pool, int id) : pool_(std::move(pool)), id_(id) {}
    std::shared_ptr<detail::KvPoolState> pool_;
    int id_;
    int position_ = 0;
    std::vector<int> pages_;
};

class Runtime {
public:
    // Loads weights onto the device transactionally: the memory plan is
    // checked against free VRAM before any allocation, and a failure at any
    // point releases everything acquired so far.
    Runtime(std::shared_ptr<const GgufModel> model, const RuntimeOptions& options = {});
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    const GgufModel& model() const { return *model_; }
    const ModelConfig& config() const { return model_->config(); }
    const RuntimeOptions& options() const { return options_; }
    Activations activations() const;          // resolved: never Auto
    const DeviceInfo& device() const;
    const MemoryPlan& memory() const;
    KvStats kv_stats() const;
    StageTimes stage_times() const;
    void reset_stage_times();

    std::unique_ptr<Sequence> new_sequence();

    // Appends tokens[i] to seqs[i] and returns next-token logits, one row per
    // sequence. Transactional: if any sequence cannot take its token (context
    // full, no free page), no sequence changes and an Error names which one.
    std::vector<std::vector<float>> step(const std::vector<Sequence*>& seqs,
                                         const std::vector<int>& tokens);

    // The plan the constructor checks, exposed so callers can size options.
    static MemoryPlan plan_memory(const GgufModel& model, const RuntimeOptions& options);

private:
    friend class Sequence;
    struct Impl;
    std::shared_ptr<const GgufModel> model_;
    RuntimeOptions options_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace llm
