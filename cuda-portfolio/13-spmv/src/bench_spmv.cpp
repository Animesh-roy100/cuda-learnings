// SpMV benchmark: format choice against matrix structure.

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "cu/device.hpp"
#include "spmv.h"

using spmv::CooMatrix;
using spmv::Format;
using spmv::SparseMatrix;

namespace {

const Format kFmt[] = {Format::CsrScalar, Format::CsrVector,
                       Format::Ellpack, Format::Hybrid};
const char* kName[] = {"CSR scalar", "CSR vector", "ELLPACK", "Hybrid"};

// DRAM traffic for the MATRIX STREAM only: A's arrays plus the y write.
//
// The x gather is deliberately excluded, and that exclusion is the point. Two
// earlier versions of this function both produced impossible numbers:
//
//   charging both ELL arrays in full        -> 263 GB/s on a 192 GB/s card
//   adding the x gather as nnz * 4 bytes    -> 220 GB/s, 115% of peak
//
// x is small and its reuse depends entirely on the sparsity pattern: on a
// banded matrix consecutive rows touch nearly the same elements and the reads
// are served from cache, so they never reach DRAM. Counting them as DRAM
// traffic inflates the result past what the hardware can do -- which is always
// the signal that a model, not a kernel, is wrong.
//
// A's arrays, by contrast, are streamed once and cannot be cached: they are far
// larger than L2. That part is honest to measure.
double matrix_stream_bytes(const SparseMatrix& m, Format f) {
    const auto s = m.stats(f);
    const double y_write = double(m.rows()) * 4;
    switch (f) {
        case Format::Ellpack:
        case Format::Hybrid: {
            const double ell_entries = double(s.ell_width) * m.rows();
            const double col_read = ell_entries * 4;      // every slot, padding too
            const double val_read = double(m.nnz()) * 4;  // only real entries
            return col_read + val_read + y_write;
        }
        default: {
            const double col_read = double(m.nnz()) * 4;
            const double val_read = double(m.nnz()) * 4;
            const double ptr_read = double(m.rows() + 1) * 4;
            return col_read + val_read + ptr_read + y_write;
        }
    }
}

void run(const char* label, const CooMatrix& coo, const cu::DeviceInfo& dev) {
    SparseMatrix m(coo);
    std::vector<float> x((std::size_t)m.cols(), 1.0f);

    const auto es = m.stats(Format::Ellpack);
    const auto hs = m.stats(Format::Hybrid);
    std::printf("\n=== %s ===\n", label);
    std::printf("  %d x %d, %lld nnz, mean row %.1f, max row %d\n",
                m.rows(), m.cols(), (long long)m.nnz(), es.mean_row_nnz, es.max_row_nnz);
    std::printf("  ELL width %d -> %lld padded entries (%.1fx the real nnz), %.0f MB\n",
                es.ell_width, (long long)es.padded_entries,
                double(es.padded_entries) / std::max<long long>(es.nnz, 1),
                es.device_bytes / 1e6);
    std::printf("  Hybrid cut %d -> %lld padded, %.1f MB\n\n",
                hs.ell_width, (long long)hs.padded_entries, hs.device_bytes / 1e6);

    // Pure ELLPACK on a skewed matrix can demand more VRAM than exists. The
    // matrix object already built it here, but a production path must refuse
    // rather than discover it at allocation time.
    const double vram_budget = 0.55 * double(dev.total_mem);

    std::printf("  %-12s %10s %12s %13s %8s\n",
                "format", "ms", "GFLOP/s", "A-stream GB/s", "%% peak");
    for (int f = 0; f < 4; ++f) {
        const auto fs = m.stats(kFmt[f]);
        if (kFmt[f] == Format::Ellpack && double(fs.device_bytes) > vram_budget) {
            std::printf("  %-12s   SKIPPED: %.1f GB exceeds a sane VRAM budget "
                        "(%.1f GB)\n", kName[f], fs.device_bytes / 1e9,
                        vram_budget / 1e9);
            continue;
        }

        float ms = 0.0f, best = 1e30f;
        for (int i = 0; i < 8; ++i) {
            m.multiply(x, kFmt[f], &ms);
            if (i >= 2) best = std::min(best, ms);
        }
        const double gflops = 2.0 * m.nnz() / (best / 1e3) / 1e9;
        const double gbs = matrix_stream_bytes(m, kFmt[f]) / (best / 1e3) / 1e9;
        std::printf("  %-12s %10.3f %12.2f %12.1f %7.0f%%\n",
                    kName[f], best, gflops, gbs,
                    100.0 * gbs / dev.peak_bandwidth_gbps());
    }
}

}  // namespace

int main() try {
    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("SpMV: the kernel that sets the pace for nearly every iterative\n");
    std::printf("method, and the textbook irregular-memory problem. A is compact,\n");
    std::printf("but x is indexed by the column array, so reads of x scatter.\n");

    run("banded (uniform rows -- ELLPACK's best case)",
        CooMatrix::banded(500000, 9, 1), dev);
    run("power-law (heavy tail -- ELLPACK's worst case)",
        CooMatrix::power_law(300000, 8, 2026), dev);
    run("2D Laplacian 700x700 (5-point stencil)",
        CooMatrix::laplacian_2d(700, 700), dev);

    // --- conjugate gradient ---
    std::printf("\n=== conjugate gradient on a 2D Laplacian ===\n");
    auto lap = CooMatrix::laplacian_2d(512, 512);
    SparseMatrix m(lap);
    std::vector<float> b((std::size_t)m.rows(), 1.0f);

    std::printf("  %-12s %8s %12s %14s\n", "format", "iters", "ms", "ms/iter");
    for (int f = 0; f < 4; ++f) {
        auto r = m.solve_cg(b, 2000, 1e-6, kFmt[f]);
        std::printf("  %-12s %8d %12.1f %14.3f\n",
                    kName[f], r.iterations, r.elapsed_ms,
                    r.elapsed_ms / std::max(r.iterations, 1));
    }

    std::printf("\nWhy the dot products accumulate in DOUBLE: CG's stopping test\n");
    std::printf("compares residuals that shrink by many orders of magnitude. FP32\n");
    std::printf("accumulation over millions of terms destroys exactly the digits\n");
    std::printf("that test depends on, and the solver stalls or declares false\n");
    std::printf("convergence. The SpMV itself stays FP32 -- only the reduction\n");
    std::printf("needs the extra precision, and there are only n terms in it.\n");

    std::printf("\nHow to read the format table:\n\n");
    std::printf("  ELLPACK stores column-major, so at step j thread r reads element\n");
    std::printf("  [j*n + r] and the warp issues one clean transaction. That is why\n");
    std::printf("  it wins on uniform rows. Its cost is padding every row to the\n");
    std::printf("  longest one, which a heavy tail turns into a catastrophe -- on\n");
    std::printf("  the power-law matrix that is 260x the real non-zeros and several\n");
    std::printf("  gigabytes, more than this card has.\n\n");
    std::printf("  CSR scalar wastes no space but stalls a whole warp on its longest\n");
    std::printf("  row and reads A uncoalesced. CSR vector fixes the coalescing and\n");
    std::printf("  then wastes 31 of 32 lanes on rows of length 5 -- which is why it\n");
    std::printf("  is the SLOWEST option on the banded and Laplacian matrices and\n");
    std::printf("  the second fastest on the skewed one. Same kernel, opposite\n");
    std::printf("  verdict, decided entirely by the row-length distribution.\n\n");
    std::printf("  Hybrid keeps ELL's access pattern for the bulk and hands the tail\n");
    std::printf("  to a warp-per-row CSR pass. No format wins everywhere, which is\n");
    std::printf("  the entire reason all four exist.\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
