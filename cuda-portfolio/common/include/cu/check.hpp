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
    if (code != cudaSuccess) {
        // A failed runtime call ALSO records its error as the thread's "last
        // error", and that record survives until something reads it. Throwing
        // without consuming it meant the next CU_CHECK_KERNEL -- anywhere, on a
        // perfectly good kernel -- read cudaGetLastError(), found the old error,
        // and reported "kernel launch failed" for a call it never made.
        //
        // Measured on this driver: after a failed cudaMalloc or cudaMemAdvise,
        // the next kernel launches, runs and produces correct output, while
        // cudaGetLastError() still returns the earlier failure exactly once.
        // Neither error is sticky. For errors that ARE sticky
        // (cudaErrorIllegalAddress and friends) this call does not clear them,
        // so nothing is hidden.
        (void)cudaGetLastError();
        throw CudaError(code, file, line, expr);
    }
}

// Kernel launches do not return a status. Two separate questions must be
// asked: did the launch configuration take (getLastError), and did the kernel
// itself fault (synchronize).
//
// getLastError cannot tell a failed launch from an earlier failed call whose
// error nobody read. check() consumes errors it throws for; a raw runtime call
// whose failure is handled without CU_CHECK must call cudaGetLastError() itself,
// or the next check_kernel will misattribute it.
inline void check_kernel(const char* file, int line) {
    check(cudaGetLastError(), file, line, "kernel launch");
    check(cudaDeviceSynchronize(), file, line, "kernel execution");
}

}  // namespace cu

#define CU_CHECK(expr) ::cu::check((expr), __FILE__, __LINE__, #expr)
#define CU_CHECK_KERNEL() ::cu::check_kernel(__FILE__, __LINE__)
