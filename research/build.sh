#!/bin/bash
# Build the two C++ kernels next to this script, in the Python environment that has FAISS from PR #5915 (98deaca).
# ngp_kernels.so: per-point margin/projection sums and the greedy matching; ngp_decide.so: the whole per-check decision
# (it calls the OpenBLAS/LAPACK that NumPy uses, for the Fay-Herriot variant). -ffp-contract=off: no fused multiply-adds,
# so every operation rounds as in the NumPy reference.
set -e
cd "$(dirname "$0")"
PREFIX=${CONDA_PREFIX:?activate the environment with the PR build first}
g++ -O3 -ffp-contract=off -fopenmp -fPIC -shared -o ngp_kernels.so ngp_kernels.cpp
g++ -O3 -std=c++17 -ffp-contract=off -fopenmp -fPIC -shared -o ngp_decide.so ngp_decide.cpp \
    -L"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" -l:libcblas.so.3 -l:liblapack.so.3
echo "built ngp_kernels.so and ngp_decide.so"
