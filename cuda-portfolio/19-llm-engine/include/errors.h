#pragma once
//
// Errors with context. No CUDA syntax.
//
// A bare "cudaErrorMemoryAllocation" tells an operator nothing about which
// model, which tensor, which sequence. Every public failure in the runtime is
// an llm::Error carrying whatever of that context applies, and its what()
// renders all of it on one line.
//
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm {

enum class ErrorKind {
    InvalidArgument,    // caller error: bad token, bad shape, bad option
    InvalidModel,       // the file is unusable: missing tensor, bad shape, bad metadata
    Unsupported,        // valid, but not something this runtime implements
    OutOfMemory,        // device budget or allocation failure
    ContextFull,        // a sequence reached its context length
    Cancelled,          // the caller asked generation to stop
    Device,             // a CUDA runtime or kernel failure
};

const char* to_string(ErrorKind k);

struct ErrorContext {
    std::string operation;                   // "load", "step", "sample", ...
    std::string model;                       // file path
    std::string tensor;                      // tensor name
    std::vector<std::uint64_t> shape;
    std::optional<int> device;
    std::string cuda_error;                  // cudaGetErrorName text
    std::optional<int> sequence;
    std::optional<int> position;
};

class Error : public std::runtime_error {
public:
    Error(ErrorKind kind, std::string message, ErrorContext ctx = {});
    ErrorKind kind() const noexcept { return kind_; }
    const std::string& message() const noexcept { return message_; }
    const ErrorContext& context() const noexcept { return ctx_; }

private:
    ErrorKind kind_;
    std::string message_;
    ErrorContext ctx_;
};

}  // namespace llm
