#pragma once
//
// Deliberately failing device operations for the error-path suite.
// No CUDA syntax in this header.
//
#include <vector>

namespace ep {

// Launches a small kernel and checks its output. Throws cu::CudaError if the
// context is unusable -- which is how the tests tell a recoverable error from a
// sticky one.
bool launch_and_verify(int n);

// Launches with 4096 threads per block. Throws cudaErrorInvalidConfiguration.
void launch_with_oversized_block();

// Dereferences an unmapped device address. Throws cudaErrorIllegalAddress and
// leaves the context permanently unusable for this process.
void write_to_illegal_address();

}  // namespace ep
