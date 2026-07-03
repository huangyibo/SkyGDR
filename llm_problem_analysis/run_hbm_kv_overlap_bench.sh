#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# Config
###############################################################################

# GPU assignment:
#   SRC_GPU: P / prefill / KV source GPU
#   DST_GPU: D / decode / KV destination GPU
SRC_GPU="${SRC_GPU:-0}"
DST_GPU="${DST_GPU:-1}"

# decode_hbm_stress parameters:
#   working set in MiB
#   iterations
HBM_WORKING_SET_MIB="${HBM_WORKING_SET_MIB:-8192}"
HBM_ITERS="${HBM_ITERS:-1000}"

# kv_nvlink_transfer parameters:
#   transfer size in MiB
#   iterations
KV_TRANSFER_MIB="${KV_TRANSFER_MIB:-16384}"
KV_ITERS="${KV_ITERS:-1000}"

# Output directory.
OUT_DIR="${OUT_DIR:-bench_results_$(date +%Y%m%d_%H%M%S)}"

# Optional CUDA env script.
CUDA_ENV_SCRIPT="${CUDA_ENV_SCRIPT:-./cuda-env-setup.sh}"

###############################################################################
# Helpers
###############################################################################

log() {
  echo "[$(date '+%F %T')] $*"
}

run_cmd() {
  log "RUN: $*"
  "$@"
}

require_bin() {
  local bin="$1"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "ERROR: cannot find command: $bin" >&2
    exit 1
  fi
}

extract_summary() {
  local name="$1"
  local logfile="$2"

  echo
  echo "================ ${name} ================"
  if [[ ! -f "$logfile" ]]; then
    echo "missing logfile: $logfile"
    return
  fi

  grep -E \
    "HBM stress done|KV transfer done|src_gpu=|dst_gpu=|gpu=|working_set=|size=|iters=|GiB/s|time|bw" \
    "$logfile" || true
}

###############################################################################
# Setup
###############################################################################

mkdir -p "$OUT_DIR"

if [[ -f "$CUDA_ENV_SCRIPT" ]]; then
  log "Sourcing CUDA environment: $CUDA_ENV_SCRIPT"
  # shellcheck source=/dev/null
  source "$CUDA_ENV_SCRIPT"
fi

require_bin nvidia-smi

if [[ ! -x ./decode_hbm_stress ]]; then
  echo "ERROR: ./decode_hbm_stress not found or not executable." >&2
  echo "Build first with: make CUDA_ARCH=sm_90" >&2
  exit 1
fi

if [[ ! -x ./kv_nvlink_transfer ]]; then
  echo "ERROR: ./kv_nvlink_transfer not found or not executable." >&2
  echo "Build first with: make CUDA_ARCH=sm_90" >&2
  exit 1
fi

log "Output directory: $OUT_DIR"

{
  echo "Benchmark configuration"
  echo "======================="
  echo "SRC_GPU=${SRC_GPU}"
  echo "DST_GPU=${DST_GPU}"
  echo "HBM_WORKING_SET_MIB=${HBM_WORKING_SET_MIB}"
  echo "HBM_ITERS=${HBM_ITERS}"
  echo "KV_TRANSFER_MIB=${KV_TRANSFER_MIB}"
  echo "KV_ITERS=${KV_ITERS}"
  echo
  echo "CUDA / GPU information"
  echo "======================"
  command -v nvcc || true
  nvcc --version || true
  nvidia-smi
  echo
  echo "Topology"
  echo "========"
  nvidia-smi topo -m || true
} > "${OUT_DIR}/env_info.txt" 2>&1

###############################################################################
# 1. decode_hbm_stress alone
###############################################################################

log "Running decode_hbm_stress alone on GPU ${DST_GPU}"

./decode_hbm_stress \
  "${DST_GPU}" \
  "${HBM_WORKING_SET_MIB}" \
  "${HBM_ITERS}" \
  > "${OUT_DIR}/decode_hbm_alone.log" 2>&1

###############################################################################
# 2. kv_nvlink_transfer alone
###############################################################################

log "Running kv_nvlink_transfer alone: GPU ${SRC_GPU} -> GPU ${DST_GPU}"

./kv_nvlink_transfer \
  "${SRC_GPU}" \
  "${DST_GPU}" \
  "${KV_TRANSFER_MIB}" \
  "${KV_ITERS}" \
  > "${OUT_DIR}/kv_transfer_alone.log" 2>&1

###############################################################################
# 3. colocated / overlapped execution
###############################################################################

log "Running colocated overlap: decode_hbm_stress on GPU ${DST_GPU}, KV ${SRC_GPU}->${DST_GPU}"

./decode_hbm_stress \
  "${DST_GPU}" \
  "${HBM_WORKING_SET_MIB}" \
  "${HBM_ITERS}" \
  > "${OUT_DIR}/decode_hbm_overlap.log" 2>&1 &

DECODE_PID=$!

# Give the decode kernel a short head start so it is already active.
sleep 1

./kv_nvlink_transfer \
  "${SRC_GPU}" \
  "${DST_GPU}" \
  "${KV_TRANSFER_MIB}" \
  "${KV_ITERS}" \
  > "${OUT_DIR}/kv_transfer_overlap.log" 2>&1

wait "${DECODE_PID}"

###############################################################################
# 4. Summary
###############################################################################

SUMMARY="${OUT_DIR}/summary.txt"

{
  echo "Benchmark summary"
  echo "================="
  echo
  echo "SRC_GPU=${SRC_GPU}"
  echo "DST_GPU=${DST_GPU}"
  echo "HBM_WORKING_SET_MIB=${HBM_WORKING_SET_MIB}"
  echo "HBM_ITERS=${HBM_ITERS}"
  echo "KV_TRANSFER_MIB=${KV_TRANSFER_MIB}"
  echo "KV_ITERS=${KV_ITERS}"

  extract_summary "decode_hbm_stress alone" "${OUT_DIR}/decode_hbm_alone.log"
  extract_summary "kv_nvlink_transfer alone" "${OUT_DIR}/kv_transfer_alone.log"
  extract_summary "decode_hbm_stress overlap" "${OUT_DIR}/decode_hbm_overlap.log"
  extract_summary "kv_nvlink_transfer overlap" "${OUT_DIR}/kv_transfer_overlap.log"
} | tee "$SUMMARY"

log "Done. Results saved to: $OUT_DIR"
log "Summary: $SUMMARY"
