#pragma once
//
// Error handling. CUDA reports failures through return codes, never exceptions,
// so an unchecked call surfaces later as wrong numbers rather than a crash.
// Every call site in this repo goes through CU_CHECK.
//
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace cu {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const char* file, int line, const char* expr)
        : std::runtime_error(build(code, file, line, expr)), code_(code) {}
    cudaError_t code() const noexcept { return code_; }

private:
    static std::string build(cudaError_t code, const char* file, int line, const char* expr) {
        return std::string(file) + ":" + std::to_string(line) + ": " + expr + " failed: " +
               cudaGetErrorName(code) + " (" + cudaGetErrorString(code) + ")";
    }
    cudaError_t code_;
};

inline void check(cudaError_t code, const char* file, int line, const char* expr) {
    if (code != cudaSuccess) throw CudaError(code, file, line, expr);
}

// Kernel launches do not return a status. Two separate questions must be
// asked: did the launch configuration take (getLastError), and did the kernel
// itself fault (synchronize).
inline void check_kernel(const char* file, int line) {
    check(cudaGetLastError(), file, line, "kernel launch");
    check(cudaDeviceSynchronize(), file, line, "kernel execution");
}

}  // namespace cu

#define CU_CHECK(expr) ::cu::check((expr), __FILE__, __LINE__, #expr)
#define CU_CHECK_KERNEL() ::cu::check_kernel(__FILE__, __LINE__)
