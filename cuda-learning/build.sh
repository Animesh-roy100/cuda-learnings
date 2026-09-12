#!/usr/bin/env bash
# Linux / Google Colab equivalent of build.bat.
#   ./build.sh 02-hello/hello.cu
#   ./bin/hello
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: ./build.sh <path-to-.cu>"
    echo "  e.g. ./build.sh 00-check/driver_check.cu"
    exit 1
fi
[ -f "$1" ] || { echo "ERROR: no such file: $1"; exit 1; }
command -v nvcc >/dev/null || { echo "ERROR: nvcc not on PATH"; exit 1; }

mkdir -p bin
name=$(basename "${1%.*}")

# sm_75 is Turing: both the GTX 1650 this was written on and the Tesla T4 that
# Colab hands out. Same architecture, so no flag change is needed between them.
nvcc -arch=sm_75 -O3 -lineinfo "$1" -o "bin/$name"
echo "Built bin/$name"
