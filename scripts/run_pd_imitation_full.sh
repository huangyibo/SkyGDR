#!/usr/bin/env bash
# Full baseline + native offload + compare (see docs/pd_imitation_runbook.md)
set -euo pipefail
LOG="${1:-/home/danyang/SkyGDR/results/pd_imitation_full_run.log}"
exec > >(tee -a "$LOG") 2>&1

export ROOT="${ROOT:-$HOME/SkyGDR}"
export BASELINE_RUN_ROOT="$ROOT/results/pd_imitation_qwen3_8b_instruct"
export OFFLOAD_RUN_ROOT="$ROOT/results/pd_imitation_qwen3_8b_instruct_native_offload"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export SERVED_MODEL="${SERVED_MODEL:-Qwen3-8B-Instruct}"
export API_BASE="${API_BASE:-http://127.0.0.1:8000}"
export PORT="${PORT:-8000}"
export DATA_ROOT="${DATA_ROOT:-/data/danyang}"
export VENV_PATH="${VENV_PATH:-$DATA_ROOT/venvs/vllm}"
export GPU_INDEX="${GPU_INDEX:-0}"
export GPU_METRICS_INTERVAL_MS="${GPU_METRICS_INTERVAL_MS:-100}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-4}"
export OFFLOAD_SIZE_GIB="${OFFLOAD_SIZE_GIB:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

mkdir -p "$BASELINE_RUN_ROOT"/{logs,data,summary,fig}
mkdir -p "$OFFLOAD_RUN_ROOT"/{logs,data,summary}
cd "$ROOT"
source "$VENV_PATH/bin/activate"

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

stop_all_vllm() {
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  pkill -f "vllm serve" >/dev/null 2>&1 || true
  sleep 5
  pkill -9 -f "VLLM::EngineCore" >/dev/null 2>&1 || true
  sleep 25
}

stop_port() {
  stop_all_vllm
}

METRICS_PID=""

start_metrics_logger() {
  local run_root="$1"
  python3 src/tools/gpu_metrics_logger.py \
    --gpu "$GPU_INDEX" \
    --interval_ms "$GPU_METRICS_INTERVAL_MS" \
    --out "$run_root/logs/gpu_metrics.csv" \
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

analyze_metrics_for_run_root() {
  local run_root="$1"
  python3 src/tools/pd_pcie_offload_analyze.py \
    --metrics_csv "$run_root/logs/gpu_metrics.csv" \
    --prefill_csv "$run_root/data/prefill_samples.csv" \
    --decode_csv "$run_root/data/decode_samples.csv" \
    --run_label "$(basename "$run_root")" \
    --out_svg "$run_root/summary/pcie_timeline.svg" \
    --out_md "$run_root/summary/pcie_timeline_report.md" \
    --out_csv "$run_root/summary/pcie_timeline_window.csv" \
    --out_json "$run_root/summary/pcie_timeline_summary.json"
}

run_baseline_server() {
  export RUN_ROOT="$BASELINE_RUN_ROOT"
  stop_port
  vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --generation-config vllm \
    2>&1 | tee "$RUN_ROOT/logs/vllm_qwen3_8b_instruct.log" &
  wait_health
}

run_offload_server() {
  export RUN_ROOT="$OFFLOAD_RUN_ROOT"
  stop_port
  vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --generation-config vllm \
    --kv-offloading-size "$OFFLOAD_SIZE_GIB" \
    --kv-offloading-backend native \
    --disable-hybrid-kv-cache-manager \
    2>&1 | tee "$RUN_ROOT/logs/vllm_qwen3_8b_instruct_native_kv_offload.log" &
  wait_health
}

stop_server() {
  stop_all_vllm
}

pipeline_for_run_root() {
  local R="$1"
  export RUN_ROOT="$R"
  echo "=== pipeline RUN_ROOT=$RUN_ROOT ==="

  cat > "$RUN_ROOT/data/long_dialogue_prefix.txt" <<'EOF'
System: You are a careful and concise assistant helping with code, systems, and ML infrastructure questions.

User: I am profiling a distributed inference runtime and want to understand memory movement.
Assistant: Sure. We should separate prefill cost, decode cost, and cache movement so we can reason about contention clearly.

User: I also care about long-context serving and cache offloading.
Assistant: Then we should use longer dialogue-style prompts, preserve multi-turn history, and compare a baseline path with an offloading path under the same request mix.

User: Please keep track of token count, decode length, throughput, and KV size.
Assistant: Understood. We will focus on prompt length, generation length, and the implied KV footprint under the target model.

User: I want the workload to resemble a realistic debugging conversation with repeated context carry-over.
Assistant:
EOF

  python3 src/tools/pd_build_bucket_prompts.py \
    --model_or_tokenizer "$MODEL_PATH" \
    --target_tokens 2048,4096,8192,16384,24576,28672 \
    --samples_per_bucket 12 \
    --prefix_file "$RUN_ROOT/data/long_dialogue_prefix.txt" \
    --out "$RUN_ROOT/data/prefill_prompts.jsonl"
  wc -l "$RUN_ROOT/data/prefill_prompts.jsonl"

  python3 src/tools/pd_collect_openai_samples.py \
    --api_base "$API_BASE" \
    --model "$SERVED_MODEL" \
    --mode prefill \
    --input_jsonl "$RUN_ROOT/data/prefill_prompts.jsonl" \
    --endpoint completion \
    --parallel_requests "$PARALLEL_REQUESTS" \
    --out_csv "$RUN_ROOT/data/prefill_samples.csv"
  wc -l "$RUN_ROOT/data/prefill_samples.csv"

  python3 src/tools/pd_build_bucket_prompts.py \
    --model_or_tokenizer "$MODEL_PATH" \
    --target_tokens 2048,4096,8192,16384,24576,28672 \
    --samples_per_bucket 12 \
    --prefix_file "$RUN_ROOT/data/long_dialogue_prefix.txt" \
    --out "$RUN_ROOT/data/decode_prompts.jsonl"
  wc -l "$RUN_ROOT/data/decode_prompts.jsonl"

  python3 src/tools/pd_collect_openai_samples.py \
    --api_base "$API_BASE" \
    --model "$SERVED_MODEL" \
    --mode decode \
    --input_jsonl "$RUN_ROOT/data/decode_prompts.jsonl" \
    --generated_tokens 128,256,512 \
    --endpoint completion \
    --parallel_requests "$PARALLEL_REQUESTS" \
    --out_csv "$RUN_ROOT/data/decode_samples.csv"
  wc -l "$RUN_ROOT/data/decode_samples.csv"

  python3 src/tools/pd_imitation_trace.py \
    --prefill_csv "$RUN_ROOT/data/prefill_samples.csv" \
    --decode_csv "$RUN_ROOT/data/decode_samples.csv" \
    --num_layers 36 \
    --num_kv_heads 8 \
    --head_dim 128 \
    --dtype_bytes 2 \
    --chunk_size_tokens 256 \
    --out_csv "$RUN_ROOT/summary/pd_imitation_trace.csv" \
    --summary_out "$RUN_ROOT/summary/pd_imitation_summary.json"
  wc -l "$RUN_ROOT/summary/pd_imitation_trace.csv"
}

echo "========== BASELINE SERVER + PIPELINE =========="
run_baseline_server
trap 'stop_metrics_logger; stop_server' EXIT
start_metrics_logger "$BASELINE_RUN_ROOT"
pipeline_for_run_root "$BASELINE_RUN_ROOT"
stop_metrics_logger
analyze_metrics_for_run_root "$BASELINE_RUN_ROOT"
stop_server
trap - EXIT

echo "========== OFFLOAD SERVER + PIPELINE =========="
run_offload_server
trap 'stop_metrics_logger; stop_server' EXIT
start_metrics_logger "$OFFLOAD_RUN_ROOT"
pipeline_for_run_root "$OFFLOAD_RUN_ROOT"
stop_metrics_logger
analyze_metrics_for_run_root "$OFFLOAD_RUN_ROOT"
stop_server
trap - EXIT

echo "========== COMPARE REPORT =========="
python3 src/tools/pd_imitation_report.py \
  --results_dir results/pd_imitation_qwen3_8b_instruct \
  --compare_results_dir results/pd_imitation_qwen3_8b_instruct_native_offload \
  --base_label baseline \
  --compare_label native_offload

echo "DONE. Log: $LOG"
