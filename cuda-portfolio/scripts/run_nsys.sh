#!/usr/bin/env bash
# Nsight Systems timelines for every benchmark (or one named target).
#
#   ./scripts/run_nsys.sh                 # all targets
#   ./scripts/run_nsys.sh bench_mc        # just one
#
# nsys needs no elevation and no special permissions, unlike ncu.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/bin"
OUT="$ROOT/profiles"

if [ ! -d "$BIN" ]; then
    echo "ERROR: no build at $BIN"
    echo "  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build"
    exit 1
fi
command -v nsys >/dev/null || { echo "ERROR: nsys not on PATH"; exit 1; }
mkdir -p "$OUT"

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    mapfile -t targets < <(cd "$BIN" && ls bench_* 2>/dev/null)
fi
[ ${#targets[@]} -gt 0 ] || { echo "no benchmark binaries found in $BIN"; exit 1; }

for t in "${targets[@]}"; do
    exe="$BIN/$t"
    [ -x "$exe" ] || { echo "skip $t (not built)"; continue; }
    echo "=== $t ==="
    # cuda,nvtx only: 'osrt' adds host-thread sampling that needs elevated
    # permissions on most systems and is not what these timelines are for.
    nsys profile --force-overwrite true --output "$OUT/$t" \
         --trace cuda,nvtx --sample none "$exe" > /dev/null 2>&1
    if [ -f "$OUT/$t.nsys-rep" ]; then
        echo "  wrote profiles/$t.nsys-rep ($(du -h "$OUT/$t.nsys-rep" | cut -f1))"
    else
        echo "  WARNING: nsys produced nothing (exit $?)"
    fi
done

echo
echo "Open with: nsys-ui profiles/<target>.nsys-rep"
