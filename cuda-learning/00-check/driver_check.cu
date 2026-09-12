// 00 - Run this first, and any time something stops working.
//
// It reports the driver/runtime versions WITHOUT aborting, so it still
// tells you something useful when the driver is too old -- unlike a normal
// program that just dies on the first CUDA call.

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int drv = 0, rt = 0;
    cudaError_t e = cudaDriverGetVersion(&drv);

    if (e == cudaErrorInsufficientDriver) {
        std::printf("FAIL: cudaErrorInsufficientDriver (35)\n\n"
                    "Your NVIDIA display driver is older than the CUDA toolkit\n"
                    "you compiled against. The toolkit alone is not enough --\n"
                    "the driver ships the actual GPU-side CUDA implementation.\n\n"
                    "Fix: install a driver new enough for your toolkit, reboot.\n");
        return 1;
    }
    if (e != cudaSuccess) {
        std::printf("FAIL: %s (%s)\n", cudaGetErrorName(e), cudaGetErrorString(e));
        return 1;
    }

    cudaRuntimeGetVersion(&rt);
    std::printf("Driver  API version : %d.%d\n", drv / 1000, (drv % 1000) / 10);
    std::printf("Runtime API version : %d.%d\n", rt  / 1000, (rt  % 1000) / 10);

    if (drv < rt) {
        std::printf("\nWARNING: driver is older than the runtime. Expect failures.\n");
        return 1;
    }

    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("\nFAIL: no CUDA devices visible.\n");
        return 1;
    }

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    std::printf("\nOK: %s, compute capability %d.%d\n", p.name, p.major, p.minor);
    std::printf("Compile with -arch=sm_%d%d\n", p.major, p.minor);
    return 0;
}
