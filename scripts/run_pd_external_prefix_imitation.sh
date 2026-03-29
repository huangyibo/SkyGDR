#!/usr/bin/env bash
# Single entrypoint for the external prefix-cache imitation workflow.
set -euo pipefail

LOG="${1:-$HOME/SkyGDR/results/pd_external_prefix_imitation_run.log}"
exec > >(tee -a "$LOG") 2>&1

export ROOT="${ROOT:-$HOME/SkyGDR}"
export RUN_ROOT="${RUN_ROOT:-$ROOT/results/pd_external_prefix_terminalbench_qwen3_8b}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export SERVED_MODEL="${SERVED_MODEL:-Qwen3-8B-Instruct}"
export API_BASE="${API_BASE:-http://127.0.0.1:8000}"
export PORT="${PORT:-8000}"
export DATA_ROOT="${DATA_ROOT:-/data/danyang}"
export VENV_PATH="${VENV_PATH:-$DATA_ROOT/venvs/vllm}"
export GPU_INDEX="${GPU_INDEX:-0}"
export GPU_METRICS_INTERVAL_MS="${GPU_METRICS_INTERVAL_MS:-20}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

export TERMINALBENCH_DATASET_NAME="${TERMINALBENCH_DATASET_NAME:-yoonholee/terminalbench-trajectories}"
export TERMINALBENCH_SPLIT="${TERMINALBENCH_SPLIT:-train}"
export TERMINALBENCH_CACHE_DIR="${TERMINALBENCH_CACHE_DIR:-$DATA_ROOT/hf-cache/datasets}"
export TERMINALBENCH_STREAMING="${TERMINALBENCH_STREAMING:-0}"
export MAX_ROWS_TO_SCAN="${MAX_ROWS_TO_SCAN:-400}"
export NUM_SESSIONS="${NUM_SESSIONS:-4}"
export REUSE_TURNS_PER_SESSION="${REUSE_TURNS_PER_SESSION:-5}"
export SEED_PROMPT_TOKENS="${SEED_PROMPT_TOKENS:-24576}"
export APPEND_TOKENS="${APPEND_TOKENS:-256}"
export DECODE_TOKENS="${DECODE_TOKENS:-16}"
export POST_REQUEST_SETTLE_MS="${POST_REQUEST_SETTLE_MS:-1200}"
export SLEEP_BETWEEN_GROUPS_MS="${SLEEP_BETWEEN_GROUPS_MS:-0}"
export GROUP_CONCURRENCY="${GROUP_CONCURRENCY:-$NUM_SESSIONS}"

export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-false}"
# Official mock example uses max_local_cpu_size: 10 even with local_cpu: false; 0 can leave
# no LocalCPUBackend and break RemoteBackendHealthCheck (LMCache stays "unhealthy").
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-10}"
export LMCACHE_REMOTE_URL="${LMCACHE_REMOTE_URL:-}"
export MOCK_STORAGE_GB="${MOCK_STORAGE_GB:-256}"
export MOCK_PEEKING_LATENCY_MS="${MOCK_PEEKING_LATENCY_MS:-1}"
export MOCK_READ_GBPS="${MOCK_READ_GBPS:-40}"
export MOCK_WRITE_GBPS="${MOCK_WRITE_GBPS:-8}"

mkdir -p "$RUN_ROOT"/{logs,data,summary,tmp}
mkdir -p "$RUN_ROOT/tmp/prometheus"
cd "$ROOT"
source "$VENV_PATH/bin/activate"

stop_all_vllm() {
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  pkill -f "vllm serve" >/dev/null 2>&1 || true
  sleep 5
  pkill -9 -f "VLLM::EngineCore" >/dev/null 2>&1 || true
  sleep 20
}

wait_health() {
  local n=0
  while ! curl -sf "$API_BASE/health" >/dev/null; do
    n=$((n + 1))
    if [[ $n -gt 600 ]]; then
      echo "health check timeout"
      return 1
    fi
    sleep 1
  done
  echo "health ok"
}

METRICS_PID=""

start_metrics_logger() {
  python3 src/tools/gpu_metrics_logger.py \
    --gpu "$GPU_INDEX" \
    --interval_ms "$GPU_METRICS_INTERVAL_MS" \
    --out "$RUN_ROOT/logs/gpu_metrics.csv" \
    >/dev/null 2>&1 &
  METRICS_PID=$!
  sleep 1
}

stop_metrics_logger() {
  if [[ -n "${METRICS_PID:-}" ]]; then
    kill "$METRICS_PID" >/dev/null 2>&1 || true
    wait "$METRICS_PID" >/dev/null 2>&1 || true
    METRICS_PID=""
  fi
}

if [[ -z "$LMCACHE_REMOTE_URL" ]]; then
  export LMCACHE_REMOTE_URL="mock://${MOCK_STORAGE_GB}/?peeking_latency=${MOCK_PEEKING_LATENCY_MS}&read_throughput=${MOCK_READ_GBPS}&write_throughput=${MOCK_WRITE_GBPS}"
fi

cat > "$RUN_ROOT/data/lmcache_config.yaml" <<EOF
chunk_size: ${LMCACHE_CHUNK_SIZE}
local_cpu: ${LMCACHE_LOCAL_CPU}
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_SIZE}
remote_url: "${LMCACHE_REMOTE_URL}"
remote_serde: "naive"
save_decode_cache: false
save_unfull_chunk: false
EOF

echo "========== CLEAN GPU =========="
stop_all_vllm
rm -rf "$RUN_ROOT/tmp/prometheus"
mkdir -p "$RUN_ROOT/tmp/prometheus"

echo "========== START VLLM + LMCACHE =========="
PYTHONHASHSEED=0 \
PROMETHEUS_MULTIPROC_DIR="$RUN_ROOT/tmp/prometheus" \
LMCACHE_CONFIG_FILE="$RUN_ROOT/data/lmcache_config.yaml" \
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --generation-config vllm \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  2>&1 | tee "$RUN_ROOT/logs/vllm_external_prefix.log" &
wait_health
trap 'stop_metrics_logger; stop_all_vllm' EXIT

echo "========== START METRICS LOGGER =========="
start_metrics_logger

echo "========== BUILD WORKLOAD =========="
python3 src/tools/pd_build_external_prefix_workload.py \
  --model_or_tokenizer "$MODEL_PATH" \
  --dataset_name "$TERMINALBENCH_DATASET_NAME" \
  --split "$TERMINALBENCH_SPLIT" \
  --dataset_cache_dir "$TERMINALBENCH_CACHE_DIR" \
  $( [[ "$TERMINALBENCH_STREAMING" == "1" ]] && echo --streaming ) \
  --max_rows_to_scan "$MAX_ROWS_TO_SCAN" \
  --num_sessions "$NUM_SESSIONS" \
  --reuse_turns_per_session "$REUSE_TURNS_PER_SESSION" \
  --seed_prompt_tokens "$SEED_PROMPT_TOKENS" \
  --append_tokens "$APPEND_TOKENS" \
  --decode_tokens "$DECODE_TOKENS" \
  --max_prompt_tokens 32768 \
  --chunk_size_tokens "$LMCACHE_CHUNK_SIZE" \
  --selected_rows_out "$RUN_ROOT/data/selected_terminalbench_rows.jsonl" \
  --out "$RUN_ROOT/data/trajectory_workload.jsonl"

echo "========== RUN WORKLOAD =========="
python3 src/tools/pd_run_external_prefix_workload.py \
  --api_base "$API_BASE" \
  --model "$SERVED_MODEL" \
  --input_jsonl "$RUN_ROOT/data/trajectory_workload.jsonl" \
  --post_request_settle_ms "$POST_REQUEST_SETTLE_MS" \
  --sleep_between_groups_ms "$SLEEP_BETWEEN_GROUPS_MS" \
  --group_concurrency "$GROUP_CONCURRENCY" \
  --ignore_eos \
  --out_csv "$RUN_ROOT/data/trajectory_samples.csv"

echo "========== STOP METRICS LOGGER =========="
stop_metrics_logger

echo "========== ANALYZE PCIe =========="
python3 src/tools/pd_pcie_offload_analyze.py \
  --metrics_csv "$RUN_ROOT/logs/gpu_metrics.csv" \
  --request_csv "$RUN_ROOT/data/trajectory_samples.csv" \
  --run_label "$(basename "$RUN_ROOT")" \
  --out_svg "$RUN_ROOT/summary/pcie_timeline.svg" \
  --out_md "$RUN_ROOT/summary/pcie_timeline_report.md" \
  --out_csv "$RUN_ROOT/summary/pcie_timeline_window.csv" \
  --out_json "$RUN_ROOT/summary/pcie_timeline_summary.json"

echo "========== BUILD REPORT =========="
python3 src/tools/pd_imitation_report.py \
  --results_dir "$RUN_ROOT"

echo "========== DONE =========="
echo "results dir: $RUN_ROOT"
echo "log file: $LOG"
