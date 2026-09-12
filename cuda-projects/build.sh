#!/usr/bin/env bash
# Linux / Google Colab equivalent of build.bat.
#   ./build.sh 05-monte-carlo/monte_carlo.cu
#   ./bin/monte_carlo
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: ./build.sh <path-to-.cu>"
    exit 1
fi
[ -f "$1" ] || { echo "ERROR: no such file: $1"; exit 1; }
command -v nvcc >/dev/null || { echo "ERROR: nvcc not on PATH"; exit 1; }

mkdir -p bin
name=$(basename "${1%.*}")

# CUDA 13 moved thrust/cub under include/cccl; CUDA 12 (what Colab ships) keeps
# them on the default include path. Add the directory only if it exists.
CCCL=""
CUDA_ROOT="$(dirname "$(dirname "$(command -v nvcc)")")"
[ -d "$CUDA_ROOT/include/cccl" ] && CCCL="-I$CUDA_ROOT/include/cccl"

nvcc -arch=sm_75 -O3 -std=c++17 -lineinfo --extended-lambda \
     $CCCL -lcurand -lcublas \
     "$1" -o "bin/$name"
echo "Built bin/$name"
