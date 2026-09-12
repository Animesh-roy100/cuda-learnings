// ============================================================================
// Project 4 - Zero-copy image pipeline, stream overlap, and CUDA IPC
//
// Three separate ideas, all about NOT paying for data movement:
//
//   1. HOW you allocate host memory decides transfer speed.
//      pageable -> the driver must stage through an internal pinned buffer
//      pinned   -> DMA straight from your pages, ~2x faster
//      mapped   -> the kernel reads host RAM over PCIe with no copy at all
//                  (a win only when each byte is touched about once)
//
//   2. Stream overlap: while frame N computes, frame N+1 uploads and frame
//      N-1 downloads. This card reports 2 async copy engines, so both
//      directions can run concurrently with compute.
//
//   3. CUDA IPC: export a device pointer to ANOTHER PROCESS so a decoder
//      and a renderer can share a frame buffer with zero copies.
//
// Run with no arguments. It re-launches itself as an IPC child.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(e_));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

static const int W = 1920, H = 1080;
static const int NPIX = W * H;

// ---------------------------------------------------------------------------
// Image kernels: RGBA -> luma, then Sobel edge magnitude.
// ---------------------------------------------------------------------------
__global__ void k_rgba_to_gray(const uchar4* __restrict__ src,
                               unsigned char* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uchar4 p = src[i];
    // Rec.601 luma. Integer math on purpose: this is a bandwidth-bound kernel,
    // so spending float ALU here would be free but pointless.
    dst[i] = (unsigned char)((77 * p.x + 150 * p.y + 29 * p.z) >> 8);
}

__global__ void k_sobel(const unsigned char* __restrict__ src,
                        unsigned char* __restrict__ dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) { dst[y * w + x] = 0; return; }

    int i = y * w + x;
    int tl = src[i - w - 1], t = src[i - w], tr = src[i - w + 1];
    int l  = src[i - 1],                     r  = src[i + 1];
    int bl = src[i + w - 1], b = src[i + w], br = src[i + w + 1];

    int gx = (tr + 2 * r + br) - (tl + 2 * l + bl);
    int gy = (bl + 2 * b + br) - (tl + 2 * t + tr);
    int m  = (int)sqrtf((float)(gx * gx + gy * gy));
    dst[i] = (unsigned char)(m > 255 ? 255 : m);
}

// Touch-once kernel, used to show where zero-copy is and is not sensible.
//
// The per-block reduction matters for the measurement, not just for style: with
// one atomicAdd per THREAD, 262144 atomics serialise on a single address and
// that contention -- not memory bandwidth -- sets the runtime. The kernel then
// takes the same time from VRAM as from host memory over PCIe, which makes the
// comparison meaningless. Reducing to one atomic per block puts the bottleneck
// back where we want to measure it.
__global__ void k_sum_bytes(const unsigned char* __restrict__ src, int n,
                            unsigned long long* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    unsigned long long s = 0;
    for (; i < n; i += stride) s += src[i];

    // warp reduce, then one value per warp into shared, then one atomic total
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    __shared__ unsigned long long wsum[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) wsum[wid] = s;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        unsigned long long v = (lane < nw) ? wsum[lane] : 0ULL;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---------------------------------------------------------------------------
// IPC child: open the handle the parent exported, verify and mutate.
// ---------------------------------------------------------------------------
static int ipc_child(const char* handle_file) {
    cudaIpcMemHandle_t h;
    FILE* f = std::fopen(handle_file, "rb");
    if (!f) { std::printf("CHILD: cannot open %s\n", handle_file); return 2; }
    size_t got = std::fread(&h, 1, sizeof(h), f);
    std::fclose(f);
    if (got != sizeof(h)) { std::printf("CHILD: short read\n"); return 2; }

    void* p = nullptr;
    cudaError_t e = cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
    if (e != cudaSuccess) {
        std::printf("CHILD: cudaIpcOpenMemHandle failed: %s\n", cudaGetErrorName(e));
        return 3;
    }

    // Read what the parent wrote, then write our own marker back.
    std::vector<int> buf(256);
    CUDA_CHECK(cudaMemcpy(buf.data(), p, buf.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int ok = 1;
    for (int i = 0; i < 256; ++i) if (buf[i] != i * 7 + 1) ok = 0;
    std::printf("CHILD: parent data %s\n", ok ? "verified" : "WRONG");

    for (int i = 0; i < 256; ++i) buf[i] = i * 13 + 5;
    CUDA_CHECK(cudaMemcpy(p, buf.data(), buf.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaIpcCloseMemHandle(p));
    std::printf("CHILD: wrote marker back, closed handle\n");
    return ok ? 0 : 4;
}

// ---------------------------------------------------------------------------
static float time_it(cudaEvent_t a, cudaEvent_t b) {
    float m = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&m, a, b)); return m;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "ipc-child") == 0) return ipc_child(argv[2]);

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("%s  sm_%d%d\n", p.name, p.major, p.minor);
    std::printf("  canMapHostMemory   : %s\n", p.canMapHostMemory ? "yes" : "no");
    std::printf("  asyncEngineCount   : %d  (copy engines that overlap with compute)\n",
                p.asyncEngineCount);
    std::printf("  unifiedAddressing  : %s\n", p.unifiedAddressing ? "yes" : "no");
    std::printf("  frame: %dx%d RGBA = %.2f MB\n\n", W, H, NPIX * 4 / 1e6);

    const size_t rgba_bytes = (size_t)NPIX * 4;
    const size_t gray_bytes = (size_t)NPIX;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // =====================================================================
    // 1. Transfer: pageable vs pinned
    // =====================================================================
    std::printf("=== 1. host memory type vs PCIe transfer speed ===\n");
    unsigned char* pageable = (unsigned char*)std::malloc(rgba_bytes);
    unsigned char* pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&pinned, rgba_bytes, cudaHostAllocDefault));
    for (size_t i = 0; i < rgba_bytes; ++i) { pageable[i] = (unsigned char)i; pinned[i] = (unsigned char)i; }

    void* dbuf;
    CUDA_CHECK(cudaMalloc(&dbuf, rgba_bytes));
    const int REP = 30;

    CUDA_CHECK(cudaMemcpy(dbuf, pageable, rgba_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < REP; ++i) CUDA_CHECK(cudaMemcpy(dbuf, pageable, rgba_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    double t_page = time_it(e0, e1) / REP;

    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < REP; ++i) CUDA_CHECK(cudaMemcpy(dbuf, pinned, rgba_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    double t_pin = time_it(e0, e1) / REP;

    std::printf("  pageable H2D  %6.3f ms  %6.2f GB/s\n", t_page, rgba_bytes / (t_page / 1e3) / 1e9);
    std::printf("  pinned   H2D  %6.3f ms  %6.2f GB/s  (%.2fx)\n",
                t_pin, rgba_bytes / (t_pin / 1e3) / 1e9, t_page / t_pin);
    std::printf("  (PCIe gen3 x16 tops out near 15.8 GB/s, so pinned is close to the wire)\n\n");

    // =====================================================================
    // 2. Zero-copy (mapped) memory: kernel reads host RAM directly
    // =====================================================================
    std::printf("=== 2. zero-copy mapped memory ===\n");
    unsigned char* mapped = nullptr;
    unsigned char* mapped_dev = nullptr;
    CUDA_CHECK(cudaHostAlloc(&mapped, gray_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&mapped_dev, mapped, 0));
    for (size_t i = 0; i < gray_bytes; ++i) mapped[i] = (unsigned char)(i & 0xff);

    unsigned char* dgray;
    CUDA_CHECK(cudaMalloc(&dgray, gray_bytes));
    CUDA_CHECK(cudaMemcpy(dgray, mapped, gray_bytes, cudaMemcpyHostToDevice));
    unsigned long long* dsum;
    CUDA_CHECK(cudaMalloc(&dsum, sizeof(unsigned long long)));

    int T = 256, B = 1024;
    const int R2 = 20;

    // WARM UP FIRST. The first launch of any kernel carries one-time setup
    // cost; timing it against an already-resident kernel makes whichever ran
    // second look artificially good. (Measured before adding this: zero-copy
    // appeared to beat device memory, which is physically impossible here --
    // PCIe is ~12 GB/s, VRAM is ~192 GB/s.)
    k_sum_bytes<<<B, T>>>(dgray, NPIX, dsum);
    k_sum_bytes<<<B, T>>>(mapped_dev, NPIX, dsum);
    CUDA_CHECK(cudaDeviceSynchronize());

    // (a) device memory, kernel only (data already resident in VRAM)
    CUDA_CHECK(cudaMemset(dsum, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < R2; ++i) k_sum_bytes<<<B, T>>>(dgray, NPIX, dsum);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    double t_dev = time_it(e0, e1) / R2;
    unsigned long long s_dev = 0;
    CUDA_CHECK(cudaMemcpy(&s_dev, dsum, sizeof(s_dev), cudaMemcpyDeviceToHost));
    s_dev /= R2;

    // (b) device memory, upload + kernel -- the honest cost if the data starts
    //     on the host. Measured, not extrapolated from the 8 MB transfer above.
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < R2; ++i) {
        CUDA_CHECK(cudaMemcpy(dgray, mapped, gray_bytes, cudaMemcpyHostToDevice));
        k_sum_bytes<<<B, T>>>(dgray, NPIX, dsum);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    double t_upload = time_it(e0, e1) / R2;

    // (c) zero-copy: no upload at all, kernel pulls straight over PCIe
    CUDA_CHECK(cudaMemset(dsum, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < R2; ++i) k_sum_bytes<<<B, T>>>(mapped_dev, NPIX, dsum);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    double t_zc = time_it(e0, e1) / R2;
    unsigned long long s_zc = 0;
    CUDA_CHECK(cudaMemcpy(&s_zc, dsum, sizeof(s_zc), cudaMemcpyDeviceToHost));
    s_zc /= R2;

    std::printf("  device mem, kernel only      %6.3f ms  (data already in VRAM)\n", t_dev);
    std::printf("  device mem, upload + kernel  %6.3f ms  (data starts on host)\n", t_upload);
    std::printf("  zero-copy, no upload at all  %6.3f ms\n", t_zc);
    std::printf("  checksums %s\n", s_dev == s_zc ? "match" : "DIFFER");
    std::printf("  -> zero-copy wins only when data is touched ~once. Touch it\n"
                "     twice and you pay PCIe twice instead of VRAM bandwidth.\n\n");

    // =====================================================================
    // 3. Stream overlap on a multi-frame pipeline
    // =====================================================================
    std::printf("=== 3. pipeline: serial vs overlapped streams ===\n");
    const int NFRAMES = 24, NSTREAM = 4;

    uchar4* h_frames = nullptr;
    unsigned char* h_out = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_frames, (size_t)NFRAMES * rgba_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out, (size_t)NFRAMES * gray_bytes, cudaHostAllocDefault));
    unsigned char* raw = (unsigned char*)h_frames;
    for (size_t i = 0; i < (size_t)NFRAMES * rgba_bytes; ++i) raw[i] = (unsigned char)(i * 31 + (i >> 9));

    uchar4* d_in[NSTREAM];
    unsigned char *d_g[NSTREAM], *d_e[NSTREAM];
    cudaStream_t st[NSTREAM];
    for (int i = 0; i < NSTREAM; ++i) {
        CUDA_CHECK(cudaMalloc(&d_in[i], rgba_bytes));
        CUDA_CHECK(cudaMalloc(&d_g[i], gray_bytes));
        CUDA_CHECK(cudaMalloc(&d_e[i], gray_bytes));
        CUDA_CHECK(cudaStreamCreate(&st[i]));
    }

    dim3 blk2(32, 8), grd2((W + 31) / 32, (H + 7) / 8);
    int Bg = (NPIX + T - 1) / T;

    // --- serial: everything on the default stream, fully ordered ---
    CUDA_CHECK(cudaEventRecord(e0));
    for (int f = 0; f < NFRAMES; ++f) {
        CUDA_CHECK(cudaMemcpy(d_in[0], h_frames + (size_t)f * NPIX, rgba_bytes, cudaMemcpyHostToDevice));
        k_rgba_to_gray<<<Bg, T>>>(d_in[0], d_g[0], NPIX);
        k_sobel<<<grd2, blk2>>>(d_g[0], d_e[0], W, H);
        CUDA_CHECK(cudaMemcpy(h_out + (size_t)f * NPIX, d_e[0], gray_bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    double t_serial = time_it(e0, e1);

    std::vector<unsigned char> ref(h_out, h_out + (size_t)NFRAMES * NPIX);
    std::memset(h_out, 0, (size_t)NFRAMES * gray_bytes);

    // --- overlapped: round-robin over streams ---
    CUDA_CHECK(cudaEventRecord(e0));
    for (int f = 0; f < NFRAMES; ++f) {
        int s = f % NSTREAM;
        CUDA_CHECK(cudaMemcpyAsync(d_in[s], h_frames + (size_t)f * NPIX, rgba_bytes,
                                   cudaMemcpyHostToDevice, st[s]));
        k_rgba_to_gray<<<Bg, T, 0, st[s]>>>(d_in[s], d_g[s], NPIX);
        k_sobel<<<grd2, blk2, 0, st[s]>>>(d_g[s], d_e[s], W, H);
        CUDA_CHECK(cudaMemcpyAsync(h_out + (size_t)f * NPIX, d_e[s], gray_bytes,
                                   cudaMemcpyDeviceToHost, st[s]));
    }
    for (int i = 0; i < NSTREAM; ++i) CUDA_CHECK(cudaStreamSynchronize(st[i]));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    double t_pipe = time_it(e0, e1);

    long long diff = 0;
    for (size_t i = 0; i < (size_t)NFRAMES * NPIX; ++i) if (ref[i] != h_out[i]) ++diff;

    std::printf("  serial      %7.2f ms  %5.1f fps\n", t_serial, NFRAMES / (t_serial / 1e3));
    std::printf("  overlapped  %7.2f ms  %5.1f fps   %.2fx   output %s\n",
                t_pipe, NFRAMES / (t_pipe / 1e3), t_serial / t_pipe,
                diff == 0 ? "identical" : "DIFFERS");

    // Verify Sobel against a CPU reference on frame 0.
    {
        std::vector<unsigned char> gray(NPIX), sob(NPIX, 0);
        const unsigned char* src = (const unsigned char*)h_frames;
        for (int i = 0; i < NPIX; ++i) {
            gray[i] = (unsigned char)((77 * src[4 * i] + 150 * src[4 * i + 1] + 29 * src[4 * i + 2]) >> 8);
        }
        long long bad = 0;
        for (int y = 1; y < H - 1; ++y)
        for (int x = 1; x < W - 1; ++x) {
            int i = y * W + x;
            int gx = (gray[i - W + 1] + 2 * gray[i + 1] + gray[i + W + 1])
                   - (gray[i - W - 1] + 2 * gray[i - 1] + gray[i + W - 1]);
            int gy = (gray[i + W - 1] + 2 * gray[i + W] + gray[i + W + 1])
                   - (gray[i - W - 1] + 2 * gray[i - W] + gray[i - W + 1]);
            int m = (int)std::sqrt((double)(gx * gx + gy * gy));
            if (m > 255) m = 255;
            if (std::abs(m - (int)ref[i]) > 1) ++bad;
        }
        std::printf("  sobel vs CPU reference: %lld/%d pixels differ by >1  -> %s\n\n",
                    bad, NPIX, bad == 0 ? "CORRECT" : "CHECK");
    }

    // =====================================================================
    // 4. CUDA IPC across processes
    // =====================================================================
    std::printf("=== 4. CUDA IPC: sharing device memory with another process ===\n");
    int* d_shared;
    CUDA_CHECK(cudaMalloc(&d_shared, 256 * sizeof(int)));
    std::vector<int> seed(256);
    for (int i = 0; i < 256; ++i) seed[i] = i * 7 + 1;
    CUDA_CHECK(cudaMemcpy(d_shared, seed.data(), 256 * sizeof(int), cudaMemcpyHostToDevice));

    cudaIpcMemHandle_t handle;
    cudaError_t ipc_e = cudaIpcGetMemHandle(&handle, d_shared);
    if (ipc_e != cudaSuccess) {
        std::printf("  cudaIpcGetMemHandle failed: %s\n", cudaGetErrorName(ipc_e));
        std::printf("  -> IPC unavailable on this platform.\n");
    } else {
        std::string hf = "cuda_ipc_handle.bin";
        FILE* f = std::fopen(hf.c_str(), "wb");
        std::fwrite(&handle, 1, sizeof(handle), f);
        std::fclose(f);

        std::string cmd = std::string("\"") + argv[0] + "\" ipc-child " + hf;
        std::printf("  parent exported handle, launching child process...\n");
        int rc = std::system(("\"" + cmd + "\"").c_str());
        std::printf("  child exit code %d\n", rc);

        std::vector<int> back(256);
        CUDA_CHECK(cudaMemcpy(back.data(), d_shared, 256 * sizeof(int), cudaMemcpyDeviceToHost));
        int ok = 1;
        for (int i = 0; i < 256; ++i) if (back[i] != i * 13 + 5) ok = 0;
        std::printf("  parent sees child's writes: %s\n",
                    ok ? "YES - genuine cross-process zero-copy sharing" : "no");
        std::remove(hf.c_str());
    }

    std::printf("\n=== note on NVDEC / NVENC ===\n");
    std::printf("This card HAS hardware video engines (nvidia-smi reports encoder and\n"
                "decoder utilisation counters), but nvcuvid.h / nvEncodeAPI.h do NOT\n"
                "ship with the CUDA Toolkit -- they are in NVIDIA's separate Video\n"
                "Codec SDK. Everything above is the part that matters for CUDA:\n"
                "the decoded-frame handoff. NVDEC hands you a CUdeviceptr, which is\n"
                "exactly what sections 2-4 move around without copying.\n");

    std::free(pageable);
    CUDA_CHECK(cudaFreeHost(pinned));
    CUDA_CHECK(cudaFreeHost(mapped));
    CUDA_CHECK(cudaFreeHost(h_frames));
    CUDA_CHECK(cudaFreeHost(h_out));
    cudaFree(dbuf); cudaFree(dgray); cudaFree(dsum); cudaFree(d_shared);
    for (int i = 0; i < NSTREAM; ++i) {
        cudaFree(d_in[i]); cudaFree(d_g[i]); cudaFree(d_e[i]);
        cudaStreamDestroy(st[i]);
    }
    return 0;
}
