#!/usr/bin/env bash
# Nsight Compute roofline and occupancy metrics.
#
#   ./scripts/run_ncu.sh                  # all targets
#   ./scripts/run_ncu.sh bench_mc         # just one
#
# PERMISSIONS. ncu needs access to GPU performance counters, which is denied by
# default on consumer cards and on most hosted VMs (Google Colab included). The
# symptom is ERR_NVGPUCTRPERM.
#
#   Linux, per boot:   sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
#   Linux, permanent:  add that option to /etc/modprobe.d/nvidia.conf, then reboot
#   Or simply:         sudo ./scripts/run_ncu.sh
#
# Windows uses a registry key instead; see scripts/profile.ps1. Note that on the
# development machine for this repo the documented registry fix did NOT take
# effect and ncu had to be run elevated -- so verify rather than assume.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/bin"
OUT="$ROOT/profiles"

if [ ! -d "$BIN" ]; then
    echo "ERROR: no build at $BIN"
    exit 1
fi
command -v ncu >/dev/null || { echo "ERROR: ncu not on PATH"; exit 1; }
mkdir -p "$OUT"

# Achieved compute and DRAM throughput give the roofline position; occupancy
# and the launch limiters explain a kernel sitting below the roof.
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem"

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    mapfile -t targets < <(cd "$BIN" && ls bench_* 2>/dev/null)
fi
[ ${#targets[@]} -gt 0 ] || { echo "no benchmark binaries found in $BIN"; exit 1; }

blocked=0
for t in "${targets[@]}"; do
    exe="$BIN/$t"
    [ -x "$exe" ] || { echo "skip $t (not built)"; continue; }
    echo "=== $t ==="
    csv="$OUT/${t}_ncu.csv"

    # --launch-count bounds the work: these benchmarks launch the same kernels
    # thousands of times and ncu REPLAYS each one it profiles.
    raw=$(ncu --csv --page raw --target-processes all \
              --launch-count 8 --metrics "$METRICS" "$exe" 2>&1)

    if grep -q "ERR_NVGPUCTRPERM" <<< "$raw"; then
        echo "  BLOCKED: ERR_NVGPUCTRPERM (see the permissions note at the top)"
        blocked=1
    else
        printf '%s\n' "$raw" > "$csv"
        echo "  wrote profiles/${t}_ncu.csv ($(wc -l < "$csv") rows)"
    fi
done

if [ "$blocked" -eq 1 ]; then
    echo
    echo "At least one target was blocked. Try: sudo $0 $*"
fi
