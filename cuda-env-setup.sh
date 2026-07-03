#!/usr/bin/env bash

# CUDA Toolkit 13.2
export CUDA_HOME=/scratch/public/nvidia/cuda/cuda-13.2
export CUDA_PATH="${CUDA_HOME}"

# Nsight tools
export NSIGHT_COMPUTE_HOME=/scratch/public/nvidia/nsight/nsight-compute/2026.1.0
export NSIGHT_SYSTEMS_HOME=/scratch/public/nvidia/nsight/nsight-systems/2025.6.3

# Locate real ncu binary directory.
if [ -x "${NSIGHT_COMPUTE_HOME}/ncu" ]; then
  export NSIGHT_COMPUTE_BIN="${NSIGHT_COMPUTE_HOME}"
else
  export NSIGHT_COMPUTE_BIN="$(
    dirname "$(find "${NSIGHT_COMPUTE_HOME}" -name ncu -type f -perm -111 | head -n 1)"
  )"
fi

# Locate real nsys binary directory.
if [ -x "${NSIGHT_SYSTEMS_HOME}/bin/nsys" ]; then
  export NSIGHT_SYSTEMS_BIN="${NSIGHT_SYSTEMS_HOME}/bin"
elif [ -x "${NSIGHT_SYSTEMS_HOME}/nsys" ]; then
  export NSIGHT_SYSTEMS_BIN="${NSIGHT_SYSTEMS_HOME}"
else
  export NSIGHT_SYSTEMS_BIN="$(
    dirname "$(find "${NSIGHT_SYSTEMS_HOME}" -name nsys -type f -perm -111 | head -n 1)"
  )"
fi

# Put Nsight dirs before CUDA bin, because CUDA bin may contain wrapper ncu/nsys.
export PATH="${NSIGHT_COMPUTE_BIN}:${NSIGHT_SYSTEMS_BIN}:${CUDA_HOME}/bin:${PATH}"

# CUDA runtime libraries.
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_PATH=${CUDA_PATH}"
echo "NSIGHT_COMPUTE_HOME=${NSIGHT_COMPUTE_HOME}"
echo "NSIGHT_COMPUTE_BIN=${NSIGHT_COMPUTE_BIN}"
echo "NSIGHT_SYSTEMS_HOME=${NSIGHT_SYSTEMS_HOME}"
echo "NSIGHT_SYSTEMS_BIN=${NSIGHT_SYSTEMS_BIN}"

echo
echo "Tool versions:"
echo "nvcc: $(command -v nvcc || true)"
nvcc --version 2>/dev/null || true

echo
echo "ncu:  $(command -v ncu || true)"
ncu --version 2>/dev/null || true

echo
echo "nsys: $(command -v nsys || true)"
nsys --version 2>/dev/null || true
