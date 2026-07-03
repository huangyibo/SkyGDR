#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# Trace-shaped KV transfer + transformer-style decode colocation benchmark
#
# Purpose:
#   Emulate realistic P/D disaggregation:
#
#     P GPU transfers KV cache to D GPU
#     +
#     D GPU runs transformer-style decode over resident KV cache
#
# Programs used:
#   ./kv_nvlink_transfer_trace
#   ./decode_transformer_kvcache_stress
#
###############################################################################

###############################################################################
# User-configurable parameters
###############################################################################

# GPU assignment:
#   SRC_GPU = P / prefill / KV source GPU
#   DST_GPU = D / decode / KV destination GPU
SRC_GPU="${SRC_GPU:-0}"
DST_GPU="${DST_GPU:-1}"

# Model geometry.
#
# Use the same geometry for both the decode emulator and KV-transfer emulator.
# A common GQA-style large-model geometry:
#   layers   = 32 or 80
#   kv_heads = 8
#   head_dim = 128
#
LAYERS="${LAYERS:-32}"
KV_HEADS="${KV_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
ELEM_BYTES="${ELEM_BYTES:-2}"

# Decode-side workload.
#
# CONTEXT controls how much existing KV cache decode reads.
# GEN controls how long decode runs.
# FFN_DIM adds extra FFN-like HBM pressure.
#
DECODE_CONTEXT="${DECODE_CONTEXT:-16384}"
DECODE_GEN="${DECODE_GEN:-512}"
FFN_DIM="${FFN_DIM:-8192}"

# KV-transfer workload.
#
# DIST can be:
#   computed: newly computed prefill KV, usually smaller
#   cached:   cached-prefix KV, usually much larger
#   input:    full input-context KV, upper-bound mode
#
# For realistic P/D KV transfer, start with computed.
# For stronger HBM-write pressure, use cached.
#
KV_DISTS="${KV_DISTS:-computed cached}"

# Iterations per transfer mode.
#
# Make cached/input iterations smaller because each transfer can be large.
# Make computed iterations larger because transfers are often smaller.
#
KV_ITERS_COMPUTED="${KV_ITERS_COMPUTED:-10000}"
KV_ITERS_CACHED="${KV_ITERS_CACHED:-5000}"
KV_ITERS_INPUT="${KV_ITERS_INPUT:-5000}"

# Start KV transfer after decode starts, so decode is active.
START_DELAY_SEC="${START_DELAY_SEC:-1}"

# Optional environment setup.
CUDA_ENV_SCRIPT="${CUDA_ENV_SCRIPT:-./cuda-env-setup.sh}"

# Output directory.
OUT_DIR="${OUT_DIR:-trace_decode_results_$(date +%Y%m%d_%H%M%S)}"

###############################################################################
# Helpers
###############################################################################

log() {
  echo "[$(date '+%F %T')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

kv_iters_for_dist() {
  local dist="$1"

  case "$dist" in
    computed)
      echo "${KV_ITERS_COMPUTED}"
      ;;
    cached)
      echo "${KV_ITERS_CACHED}"
      ;;
    input)
      echo "${KV_ITERS_INPUT}"
      ;;
    *)
      die "Unknown KV dist: ${dist}. Use computed, cached, or input."
      ;;
  esac
}

extract_decode_summary() {
  local name="$1"
  local logfile="$2"

  echo
  echo "================ ${name} ================"
  if [[ ! -f "$logfile" ]]; then
    echo "missing logfile: $logfile"
    return
  fi

  grep -E \
    "GPU:|layers=|context=|KV cache:|Time:|Generated tokens:|Throughput:|Approx KV read traffic:|Approx KV write traffic:|Approx FFN traffic:|Approx total traffic:|Approx bandwidth:" \
    "$logfile" || true
}

extract_kv_summary() {
  local name="$1"
  local logfile="$2"

  echo
  echo "================ ${name} ================"
  if [[ ! -f "$logfile" ]]; then
    echo "missing logfile: $logfile"
    return
  fi

  grep -E \
    "src_gpu=|peer_access|KV geometry:|Trace quantiles:|sampled tokens:|sampled bytes:|max transfer allocation:|Trace-shaped KV transfer done:|transfers:|total:|time:|bw:" \
    "$logfile" || true
}

###############################################################################
# Setup
###############################################################################

mkdir -p "${OUT_DIR}"

if [[ -f "${CUDA_ENV_SCRIPT}" ]]; then
  log "Sourcing CUDA environment: ${CUDA_ENV_SCRIPT}"
  # shellcheck source=/dev/null
  source "${CUDA_ENV_SCRIPT}"
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"

[[ -x ./decode_transformer_kvcache_stress ]] || \
  die "./decode_transformer_kvcache_stress not found. Build with: make CUDA_ARCH=sm_90"

[[ -x ./kv_nvlink_transfer_trace ]] || \
  die "./kv_nvlink_transfer_trace not found. Build with: make CUDA_ARCH=sm_90"

log "Output directory: ${OUT_DIR}"

###############################################################################
# Save environment information
###############################################################################

{
  echo "Benchmark configuration"
  echo "======================="
  echo "SRC_GPU=${SRC_GPU}"
  echo "DST_GPU=${DST_GPU}"
  echo "LAYERS=${LAYERS}"
  echo "KV_HEADS=${KV_HEADS}"
  echo "HEAD_DIM=${HEAD_DIM}"
  echo "ELEM_BYTES=${ELEM_BYTES}"
  echo "DECODE_CONTEXT=${DECODE_CONTEXT}"
  echo "DECODE_GEN=${DECODE_GEN}"
  echo "FFN_DIM=${FFN_DIM}"
  echo "KV_DISTS=${KV_DISTS}"
  echo "KV_ITERS_COMPUTED=${KV_ITERS_COMPUTED}"
  echo "KV_ITERS_CACHED=${KV_ITERS_CACHED}"
  echo "KV_ITERS_INPUT=${KV_ITERS_INPUT}"
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
# Optional warmup
###############################################################################

log "Warmup: short decode run"

./decode_transformer_kvcache_stress \
  --gpu "${DST_GPU}" \
  --layers "${LAYERS}" \
  --kv-heads "${KV_HEADS}" \
  --head-dim "${HEAD_DIM}" \
  --context 1024 \
  --gen 8 \
  --ffn-dim 1024 \
  > "${OUT_DIR}/warmup_decode.log" 2>&1 || true

log "Warmup: short KV transfer run"

./kv_nvlink_transfer_trace \
  --src "${SRC_GPU}" \
  --dst "${DST_GPU}" \
  --dist computed \
  --layers "${LAYERS}" \
  --kv-heads "${KV_HEADS}" \
  --head-dim "${HEAD_DIM}" \
  --elem-bytes "${ELEM_BYTES}" \
  --iters 8 \
  > "${OUT_DIR}/warmup_kv.log" 2>&1 || true

###############################################################################
# 1. Decode alone
###############################################################################

log "Running decode alone on GPU ${DST_GPU}"

./decode_transformer_kvcache_stress \
  --gpu "${DST_GPU}" \
  --layers "${LAYERS}" \
  --kv-heads "${KV_HEADS}" \
  --head-dim "${HEAD_DIM}" \
  --context "${DECODE_CONTEXT}" \
  --gen "${DECODE_GEN}" \
  --ffn-dim "${FFN_DIM}" \
  > "${OUT_DIR}/decode_alone.log" 2>&1

###############################################################################
# 2. KV transfer alone for each distribution
###############################################################################

for DIST in ${KV_DISTS}; do
  KV_ITERS="$(kv_iters_for_dist "${DIST}")"

  log "Running KV transfer alone: dist=${DIST}, iters=${KV_ITERS}"

  ./kv_nvlink_transfer_trace \
    --src "${SRC_GPU}" \
    --dst "${DST_GPU}" \
    --dist "${DIST}" \
    --layers "${LAYERS}" \
    --kv-heads "${KV_HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --elem-bytes "${ELEM_BYTES}" \
    --iters "${KV_ITERS}" \
    > "${OUT_DIR}/kv_${DIST}_alone.log" 2>&1
done

###############################################################################
# 3. Colocated runs: decode + KV transfer
###############################################################################

for DIST in ${KV_DISTS}; do
  KV_ITERS="$(kv_iters_for_dist "${DIST}")"

  log "Running colocated decode + KV transfer: dist=${DIST}, iters=${KV_ITERS}"

  ./decode_transformer_kvcache_stress \
    --gpu "${DST_GPU}" \
    --layers "${LAYERS}" \
    --kv-heads "${KV_HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --context "${DECODE_CONTEXT}" \
    --gen "${DECODE_GEN}" \
    --ffn-dim "${FFN_DIM}" \
    > "${OUT_DIR}/decode_overlap_${DIST}.log" 2>&1 &

  DECODE_PID=$!

  sleep "${START_DELAY_SEC}"

  ./kv_nvlink_transfer_trace \
    --src "${SRC_GPU}" \
    --dst "${DST_GPU}" \
    --dist "${DIST}" \
    --layers "${LAYERS}" \
    --kv-heads "${KV_HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --elem-bytes "${ELEM_BYTES}" \
    --iters "${KV_ITERS}" \
    > "${OUT_DIR}/kv_${DIST}_overlap.log" 2>&1

  wait "${DECODE_PID}"
done

###############################################################################
# 4. Generate summary
###############################################################################

SUMMARY="${OUT_DIR}/summary.txt"

{
  echo "Trace-shaped KV transfer + transformer decode colocation benchmark"
  echo "=================================================================="
  echo
  echo "Configuration"
  echo "-------------"
  echo "SRC_GPU=${SRC_GPU}"
  echo "DST_GPU=${DST_GPU}"
  echo "LAYERS=${LAYERS}"
  echo "KV_HEADS=${KV_HEADS}"
  echo "HEAD_DIM=${HEAD_DIM}"
  echo "ELEM_BYTES=${ELEM_BYTES}"
  echo "DECODE_CONTEXT=${DECODE_CONTEXT}"
  echo "DECODE_GEN=${DECODE_GEN}"
  echo "FFN_DIM=${FFN_DIM}"
  echo "KV_DISTS=${KV_DISTS}"
  echo "KV_ITERS_COMPUTED=${KV_ITERS_COMPUTED}"
  echo "KV_ITERS_CACHED=${KV_ITERS_CACHED}"
  echo "KV_ITERS_INPUT=${KV_ITERS_INPUT}"
  echo

  extract_decode_summary \
    "decode_transformer_kvcache_stress alone" \
    "${OUT_DIR}/decode_alone.log"

  for DIST in ${KV_DISTS}; do
    extract_kv_summary \
      "kv_nvlink_transfer_trace alone, dist=${DIST}" \
      "${OUT_DIR}/kv_${DIST}_alone.log"
  done

  for DIST in ${KV_DISTS}; do
    extract_decode_summary \
      "decode_transformer_kvcache_stress overlap, dist=${DIST}" \
      "${OUT_DIR}/decode_overlap_${DIST}.log"

    extract_kv_summary \
      "kv_nvlink_transfer_trace overlap, dist=${DIST}" \
      "${OUT_DIR}/kv_${DIST}_overlap.log"
  done
} | tee "${SUMMARY}"

log "Done."
log "Results directory: ${OUT_DIR}"
log "Summary: ${SUMMARY}"
