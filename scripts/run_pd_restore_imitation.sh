#!/usr/bin/env bash
# Single entrypoint for the restore-focused imitation workflow.
set -euo pipefail

LOG="${1:-$HOME/SkyGDR/results/pd_restore_imitation_run.log}"
exec > >(tee -a "$LOG") 2>&1

export ROOT="${ROOT:-$HOME/SkyGDR}"
export RUN_ROOT="${RUN_ROOT:-$ROOT/results/pd_restore_imitation_qwen3_8b}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export SERVED_MODEL="${SERVED_MODEL:-Qwen3-8B-Instruct}"
export API_BASE="${API_BASE:-http://127.0.0.1:8000}"
export PORT="${PORT:-8000}"
export DATA_ROOT="${DATA_ROOT:-/data/danyang}"
export VENV_PATH="${VENV_PATH:-$DATA_ROOT/venvs/vllm}"
export GPU_INDEX="${GPU_INDEX:-0}"
export GPU_METRICS_INTERVAL_MS="${GPU_METRICS_INTERVAL_MS:-20}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export OFFLOAD_SIZE_GIB="${OFFLOAD_SIZE_GIB:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"

export BASE_PREFIX_TOKENS="${BASE_PREFIX_TOKENS:-24576}"
export APPEND_TOKENS="${APPEND_TOKENS:-256,256,256,256,256,256}"
export NUM_TURNS="${NUM_TURNS:-6}"
export MAIN_DECODE_TOKENS="${MAIN_DECODE_TOKENS:-32}"
export PRESSURE_PROMPT_TOKENS="${PRESSURE_PROMPT_TOKENS:-28672}"
export PRESSURE_BURST_SIZE="${PRESSURE_BURST_SIZE:-8}"
export PRESSURE_ROUNDS_PER_TURN="${PRESSURE_ROUNDS_PER_TURN:-2}"
export PRESSURE_DECODE_TOKENS="${PRESSURE_DECODE_TOKENS:-1}"
export SLEEP_BETWEEN_GROUPS_MS="${SLEEP_BETWEEN_GROUPS_MS:-250}"

mkdir -p "$RUN_ROOT"/{logs,data,summary}
cd "$ROOT"
source "$VENV_PATH/bin/activate"

stop_all_vllm() {
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  pkill -f "vllm serve" >/dev/null 2>&1 || true
  sleep 5
  pkill -9 -f "VLLM::EngineCore" >/dev/null 2>&1 || true
  sleep 25
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

cat > "$RUN_ROOT/data/session_prefix.txt" <<'EOF'
System: You are helping a long-running coding agent that repeatedly inspects files, executes commands, and asks for clarification while working through a complex systems task.

User: Please keep all previous context because the next tool outputs will depend on it.
Assistant: Understood. I will preserve the full history and only add the new observations from each turn.

User: The session will be long and most of the context will repeat. Only a small amount of new information will be appended each round.
Assistant: Then the natural target is a high-reuse workload where most prefix KV can be reused across turns.
EOF

cat > "$RUN_ROOT/data/pressure_prefix.txt" <<'EOF'
System: You are processing a large unrelated debugging transcript with extensive terminal output, code snippets, and stack traces.

User: This request is intentionally long and independent of the main session so that it competes for prefix-cache residency and creates eviction pressure.
Assistant:
EOF

echo "========== CLEAN GPU =========="
stop_all_vllm

echo "========== START VLLM =========="
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --generation-config vllm \
  --enable-prefix-caching \
  --kv-offloading-size "$OFFLOAD_SIZE_GIB" \
  --kv-offloading-backend native \
  --disable-hybrid-kv-cache-manager \
  2>&1 | tee "$RUN_ROOT/logs/vllm_restore_imitation.log" &
wait_health
trap 'stop_metrics_logger; stop_all_vllm' EXIT

echo "========== START METRICS LOGGER =========="
start_metrics_logger

echo "========== BUILD WORKLOAD =========="
python3 src/tools/pd_build_trajectory_workload.py \
  --model_or_tokenizer "$MODEL_PATH" \
  --base_prefix_tokens "$BASE_PREFIX_TOKENS" \
  --append_tokens "$APPEND_TOKENS" \
  --num_turns "$NUM_TURNS" \
  --main_decode_tokens "$MAIN_DECODE_TOKENS" \
  --pressure_prompt_tokens "$PRESSURE_PROMPT_TOKENS" \
  --pressure_burst_size "$PRESSURE_BURST_SIZE" \
  --pressure_rounds_per_turn "$PRESSURE_ROUNDS_PER_TURN" \
  --pressure_decode_tokens "$PRESSURE_DECODE_TOKENS" \
  --max_prompt_tokens 32768 \
  --session_prefix_file "$RUN_ROOT/data/session_prefix.txt" \
  --pressure_prefix_file "$RUN_ROOT/data/pressure_prefix.txt" \
  --out "$RUN_ROOT/data/trajectory_workload.jsonl"

echo "========== RUN WORKLOAD =========="
python3 src/tools/pd_run_restore_workload.py \
  --api_base "$API_BASE" \
  --model "$SERVED_MODEL" \
  --input_jsonl "$RUN_ROOT/data/trajectory_workload.jsonl" \
  --sleep_between_groups_ms "$SLEEP_BETWEEN_GROUPS_MS" \
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
