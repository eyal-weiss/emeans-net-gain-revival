#!/bin/bash
# Optional C++ kernels of net_gain_revival (per-point margin/projection sums and the whole per-check decision). Build
# in the environment that has FAISS from PR #5915, whose NumPy and OpenBLAS the decision uses. Without them the package
# runs the same computations in NumPy (identical decisions, slower). -ffp-contract=off: no fused multiply-adds.
set -e
cd "$(dirname "$0")"
PREFIX=${CONDA_PREFIX:?activate the environment with the PR build first}
g++ -O3 -std=c++17 -ffp-contract=off -fopenmp -fPIC -shared -o libnetgain.so ngp_decide.cpp \
    -L"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" -l:libcblas.so.3 -l:liblapack.so.3
echo "built net_gain_revival/native/libnetgain.so"
