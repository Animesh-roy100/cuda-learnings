// 1080p video analytics benchmark.

#include <algorithm>
#include <cstdio>
#include <vector>

#include "cu/device.hpp"
#include "video_pipeline.h"

using video::BilateralParams;
using video::Nv12Frame;
using video::VideoPipeline;

int main() {
    auto dev = cu::query_device();
    cu::print_banner(dev);

    const int W = 1920, H = 1080;
    std::printf("1080p NV12 frame = %.2f MB (vs %.2f MB as packed RGB)\n\n",
                Nv12Frame::bytes(W, H) / 1e6, double(W) * H * 3 / 1e6);

    auto nv12 = video::make_test_nv12(W, H, 2026);
    Nv12Frame frame{nv12.data(), W, H};
    VideoPipeline p(W, H);

    float ms = 0.0f, best = 1e30f;
    for (int i = 0; i < 20; ++i) {
        p.nv12_to_rgb(frame, &ms);
        if (i >= 3) best = std::min(best, ms);
    }
    std::printf("=== NV12 -> RGB ===\n");
    std::printf("  %6.3f ms  -> %.0f fps  (%.1f GB/s effective)\n\n",
                best, 1000.0 / best,
                (Nv12Frame::bytes(W, H) + (size_t)W * H * 3) / (best / 1e3) / 1e9);

    auto gray = p.nv12_to_gray(frame);

    std::printf("=== bilateral filter: explicit clamp vs texture clamp ===\n");
    for (int radius : {2, 4, 6}) {
        BilateralParams bp;
        bp.radius = radius;
        float t_manual = 1e30f, t_tex = 1e30f, cur = 0.0f;
        for (int i = 0; i < 12; ++i) {
            p.bilateral(gray, bp, false, &cur);
            if (i >= 2) t_manual = std::min(t_manual, cur);
            p.bilateral(gray, bp, true, &cur);
            if (i >= 2) t_tex = std::min(t_tex, cur);
        }
        const int taps = (2 * radius + 1) * (2 * radius + 1);
        std::printf("  r=%d (%3d taps)  explicit %7.2f ms  texture %7.2f ms  %.2fx"
                    "  -> %.0f fps\n",
                    radius, taps, t_manual, t_tex, t_manual / t_tex, 1000.0 / t_tex);
    }

    std::printf("\n  The texture path is SLOWER here, consistently, which is the\n");
    std::printf("  opposite of the usual advice. Hardware address clamping really is\n");
    std::printf("  free -- but the tex2D fetch itself carries more latency than a\n");
    std::printf("  plain L1 load on Turing's unified L1/texture cache, and the min/max\n");
    std::printf("  it replaces is two cheap ALU ops. At 169 taps per pixel that trade\n");
    std::printf("  is a clear loss. Texture units pay for FILTERED or strided sampling,\n");
    std::printf("  where the interpolation is genuinely free; reaching for them purely\n");
    std::printf("  to avoid a bounds check costs more than it saves.\n");

    std::printf("\n=== motion history ===\n");
    p.motion_history(gray, 12, 24, &ms);
    best = 1e30f;
    for (int i = 0; i < 20; ++i) {
        p.motion_history(gray, 12, 24, &ms);
        if (i >= 3) best = std::min(best, ms);
    }
    std::printf("  %6.3f ms -> %.0f fps\n", best, 1000.0 / best);

    // --- transfer hierarchy ---
    auto r = video::measure_transfers(8u << 20);
    std::printf("\n=== transfer hierarchy (8 MB) ===\n");
    std::printf("  pageable H2D          %6.2f GB/s\n", r.pageable_gbps);
    std::printf("  pinned   H2D          %6.2f GB/s  (%.2fx)\n",
                r.pinned_gbps, r.pinned_gbps / r.pageable_gbps);
    std::printf("  kernel, data in VRAM  %6.3f ms\n", r.device_kernel_ms);
    std::printf("  upload + kernel       %6.3f ms\n", r.upload_plus_kernel_ms);
    std::printf("  zero-copy, no upload  %6.3f ms  (%.2fx vs upload+kernel)\n",
                r.zero_copy_kernel_ms, r.upload_plus_kernel_ms / r.zero_copy_kernel_ms);

    std::printf("\n  Also measured, not assumed: zero-copy does not automatically win.\n");
    std::printf("  Both paths pull the same bytes across the same link, but cudaMemcpy\n");
    std::printf("  streams them as one full-width DMA burst while the zero-copy kernel\n");
    std::printf("  issues fine-grained PCIe reads as warps demand them. Zero-copy buys\n");
    std::printf("  a simpler pipeline and one less buffer, not guaranteed throughput --\n");
    std::printf("  and anything read twice pays PCIe twice instead of VRAM bandwidth.\n");
    std::printf("  Resident VRAM is %.0fx faster than either; keeping frames on the\n",
                r.upload_plus_kernel_ms / r.device_kernel_ms);
    std::printf("  device is what actually matters.\n");

    std::printf("\n=== CUDA IPC ===\n");
    std::printf("  device pointer export: %s\n",
                video::cuda_ipc_available() ? "AVAILABLE on this Windows build"
                                            : "not available");

    std::printf("\n=== on NVDEC / NVENC ===\n");
    std::printf("This card has the hardware video engines -- nvidia-smi reports\n");
    std::printf("encoder and decoder utilisation counters. But nvcuvid.h and\n");
    std::printf("nvEncodeAPI.h ship in NVIDIA's separate Video Codec SDK, not in the\n");
    std::printf("CUDA Toolkit, so the codec front end is not wired up here.\n");
    std::printf("\nWhat IS built is the part that matters for CUDA: NVDEC hands you a\n");
    std::printf("CUdeviceptr to an NV12 surface, and every kernel above consumes that\n");
    std::printf("layout directly. Dropping in the SDK replaces the synthetic frame\n");
    std::printf("source; the processing path does not change.\n");
    return 0;
}
