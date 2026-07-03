#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# vLLM-like decode + trace-shaped P/D KV-transfer colocation benchmark
#
# Programs:
#   ./decode_vllm_kvcache_hbm_stress
#   ./kv_nvlink_transfer_trace
#
# Scenario:
#   GPU0/P/prefill transfers KV cache to GPU1/D/decode.
#   GPU1/D concurrently runs optimized vLLM-like decode that saturates HBM.
###############################################################################

###############################################################################
# Config
###############################################################################

# GPU assignment.
SRC_GPU="${SRC_GPU:-0}"   # P / prefill / KV source GPU
DST_GPU="${DST_GPU:-1}"   # D / decode / KV destination GPU

# Model/KV geometry.
LAYERS="${LAYERS:-80}"
KV_HEADS="${KV_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
ELEM_BYTES="${ELEM_BYTES:-2}"

# Optimized vLLM-like decode workload.
DECODE_CONTEXT="${DECODE_CONTEXT:-65536}"
DECODE_GEN="${DECODE_GEN:-128}"
DECODE_BATCH="${DECODE_BATCH:-16}"
TILE_TOKENS="${TILE_TOKENS:-128}"

# KV-transfer modes to run.
#
# computed: newly computed prefill KV; smaller but realistic.
# cached:   cached-prefix KV; larger and stronger D-HBM write pressure.
# input:    full input-context KV; upper-bound mode.
KV_DISTS="${KV_DISTS:-computed cached}"

# Iterations for each KV mode.
KV_ITERS_COMPUTED="${KV_ITERS_COMPUTED:-10000}"
KV_ITERS_CACHED="${KV_ITERS_CACHED:-1000}"
KV_ITERS_INPUT="${KV_ITERS_INPUT:-1000}"

# Delay before starting KV transfer after decode starts.
START_DELAY_SEC="${START_DELAY_SEC:-1}"

# Repeat count for each experiment.
REPEATS="${REPEATS:-1}"

# Optional CUDA/Nsight environment script.
CUDA_ENV_SCRIPT="${CUDA_ENV_SCRIPT:-./cuda-env-setup.sh}"

# Enable lightweight monitoring with nvidia-smi dmon.
ENABLE_DMON="${ENABLE_DMON:-1}"

# Output directory.
OUT_DIR="${OUT_DIR:-final_vllm_decode_kv_results_$(date +%Y%m%d_%H%M%S)}"

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
      die "Unknown KV dist=${dist}. Use computed, cached, or input."
      ;;
  esac
}

start_dmon() {
  local tag="$1"
  local out_file="${OUT_DIR}/dmon_${tag}.log"

  if [[ "${ENABLE_DMON}" != "1" ]]; then
    echo ""
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return
  fi

  nvidia-smi dmon -i "${SRC_GPU},${DST_GPU}" -s pucvmt \
    > "${out_file}" 2>&1 &

  echo "$!"
}

stop_dmon() {
  local pid="$1"

  if [[ -n "${pid}" ]]; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

extract_decode_summary() {
  local name="$1"
  local logfile="$2"

  echo
  echo "================ ${name} ================"

  if [[ ! -f "${logfile}" ]]; then
    echo "missing logfile: ${logfile}"
    return
  fi

  grep -E \
    "GPU:|layers=|context=|batch=|KV cache:|Time:|Generated tokens:|Active decode batch:|Throughput:|Effective sequence-throughput:|Approx KV read traffic:|Approx KV append write:|Approx partial write traffic:|Approx total traffic:|Approx bandwidth:" \
    "${logfile}" || true
}

extract_kv_summary() {
  local name="$1"
  local logfile="$2"

  echo
  echo "================ ${name} ================"

  if [[ ! -f "${logfile}" ]]; then
    echo "missing logfile: ${logfile}"
    return
  fi

  grep -E \
    "src_gpu=|peer_access|KV geometry:|Trace quantiles:|sampled tokens:|sampled bytes:|max transfer allocation:|Trace-shaped KV transfer done:|transfers:|total:|time:|bw:" \
    "${logfile}" || true
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

[[ -x ./decode_vllm_kvcache_hbm_stress ]] || \
  die "./decode_vllm_kvcache_hbm_stress not found. Build it first."

[[ -x ./kv_nvlink_transfer_trace ]] || \
  die "./kv_nvlink_transfer_trace not found. Build it first."

log "Output directory: ${OUT_DIR}"

###############################################################################
# Save environment and topology
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
  echo "DECODE_BATCH=${DECODE_BATCH}"
  echo "TILE_TOKENS=${TILE_TOKENS}"
  echo "KV_DISTS=${KV_DISTS}"
  echo "KV_ITERS_COMPUTED=${KV_ITERS_COMPUTED}"
  echo "KV_ITERS_CACHED=${KV_ITERS_CACHED}"
  echo "KV_ITERS_INPUT=${KV_ITERS_INPUT}"
  echo "REPEATS=${REPEATS}"
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
# Warmup
###############################################################################

log "Warmup: optimized decode"

./decode_vllm_kvcache_hbm_stress \
  --gpu "${DST_GPU}" \
  --layers 8 \
  --kv-heads "${KV_HEADS}" \
  --head-dim "${HEAD_DIM}" \
  --context 1024 \
  --gen 8 \
  --batch 4 \
  --tile-tokens "${TILE_TOKENS}" \
  > "${OUT_DIR}/warmup_decode.log" 2>&1 || true

log "Warmup: trace-shaped KV transfer"

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
# Main experiments
###############################################################################

for REP in $(seq 1 "${REPEATS}"); do
  log "Starting repeat ${REP}/${REPEATS}"

  ###########################################################################
  # 1. Optimized vLLM-like decode alone
  ###########################################################################

  log "Repeat ${REP}: decode_vllm alone"

  DMON_PID="$(start_dmon "r${REP}_decode_alone")"

  ./decode_vllm_kvcache_hbm_stress \
    --gpu "${DST_GPU}" \
    --layers "${LAYERS}" \
    --kv-heads "${KV_HEADS}" \
    --head-dim "${HEAD_DIM}" \
    --context "${DECODE_CONTEXT}" \
    --gen "${DECODE_GEN}" \
    --batch "${DECODE_BATCH}" \
    --tile-tokens "${TILE_TOKENS}" \
    > "${OUT_DIR}/decode_vllm_alone_r${REP}.log" 2>&1

  stop_dmon "${DMON_PID}"

  ###########################################################################
  # 2. KV transfer alone for each distribution
  ###########################################################################

  for DIST in ${KV_DISTS}; do
    KV_ITERS="$(kv_iters_for_dist "${DIST}")"

    log "Repeat ${REP}: KV transfer alone, dist=${DIST}, iters=${KV_ITERS}"

    DMON_PID="$(start_dmon "r${REP}_kv_${DIST}_alone")"

    ./kv_nvlink_transfer_trace \
      --src "${SRC_GPU}" \
      --dst "${DST_GPU}" \
      --dist "${DIST}" \
      --layers "${LAYERS}" \
      --kv-heads "${KV_HEADS}" \
      --head-dim "${HEAD_DIM}" \
      --elem-bytes "${ELEM_BYTES}" \
      --iters "${KV_ITERS}" \
      > "${OUT_DIR}/kv_${DIST}_alone_r${REP}.log" 2>&1

    stop_dmon "${DMON_PID}"
  done

  ###########################################################################
  # 3. Colocated decode + KV transfer for each distribution
  ###########################################################################

  for DIST in ${KV_DISTS}; do
    KV_ITERS="$(kv_iters_for_dist "${DIST}")"

    log "Repeat ${REP}: colocated decode + KV transfer, dist=${DIST}"

    DMON_PID="$(start_dmon "r${REP}_overlap_${DIST}")"

    ./decode_vllm_kvcache_hbm_stress \
      --gpu "${DST_GPU}" \
      --layers "${LAYERS}" \
      --kv-heads "${KV_HEADS}" \
      --head-dim "${HEAD_DIM}" \
      --context "${DECODE_CONTEXT}" \
      --gen "${DECODE_GEN}" \
      --batch "${DECODE_BATCH}" \
      --tile-tokens "${TILE_TOKENS}" \
      > "${OUT_DIR}/decode_vllm_overlap_${DIST}_r${REP}.log" 2>&1 &

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
      > "${OUT_DIR}/kv_${DIST}_overlap_r${REP}.log" 2>&1

    wait "${DECODE_PID}"

    stop_dmon "${DMON_PID}"
  done
done

###############################################################################
# Summary
###############################################################################

SUMMARY="${OUT_DIR}/summary.txt"

{
  echo "Optimized vLLM-like decode + trace-shaped KV transfer benchmark"
  echo "================================================================"
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
  echo "DECODE_BATCH=${DECODE_BATCH}"
  echo "TILE_TOKENS=${TILE_TOKENS}"
  echo "KV_DISTS=${KV_DISTS}"
  echo "KV_ITERS_COMPUTED=${KV_ITERS_COMPUTED}"
  echo "KV_ITERS_CACHED=${KV_ITERS_CACHED}"
  echo "KV_ITERS_INPUT=${KV_ITERS_INPUT}"
  echo "REPEATS=${REPEATS}"
  echo

  for REP in $(seq 1 "${REPEATS}"); do
    echo
    echo "################################################################"
    echo "Repeat ${REP}"
    echo "################################################################"

    extract_decode_summary \
      "decode_vllm_kvcache_hbm_stress alone, repeat=${REP}" \
      "${OUT_DIR}/decode_vllm_alone_r${REP}.log"

    for DIST in ${KV_DISTS}; do
      extract_kv_summary \
        "kv_nvlink_transfer_trace alone, dist=${DIST}, repeat=${REP}" \
        "${OUT_DIR}/kv_${DIST}_alone_r${REP}.log"
    done

    for DIST in ${KV_DISTS}; do
      extract_decode_summary \
        "decode_vllm overlap, dist=${DIST}, repeat=${REP}" \
        "${OUT_DIR}/decode_vllm_overlap_${DIST}_r${REP}.log"

      extract_kv_summary \
        "kv transfer overlap, dist=${DIST}, repeat=${REP}" \
        "${OUT_DIR}/kv_${DIST}_overlap_r${REP}.log"
    done
  done
} | tee "${SUMMARY}"

###############################################################################
# Compact grep summary
###############################################################################

COMPACT="${OUT_DIR}/compact_summary.txt"

{
  echo "Compact decode summary"
  echo "======================"
  grep -H -E \
    "Time:|Throughput:|Effective sequence-throughput:|Approx bandwidth:" \
    "${OUT_DIR}"/decode_vllm*.log || true

  echo
  echo "Compact KV transfer summary"
  echo "==========================="
  grep -H -E \
    "total:|time:|bw:" \
    "${OUT_DIR}"/kv_*.log || true
} | tee "${COMPACT}"

log "Done."
log "Results directory: ${OUT_DIR}"
log "Summary: ${SUMMARY}"
log "Compact summary: ${COMPACT}"
