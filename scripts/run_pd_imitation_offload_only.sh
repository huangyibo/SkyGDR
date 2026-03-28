#!/usr/bin/env bash
# Resume: native offload pipeline + compare (baseline already done). See run_pd_imitation_full.sh
set -euo pipefail
LOG="${1:-$HOME/SkyGDR/results/pd_imitation_offload_only.log}"
exec > >(tee -a "$LOG") 2>&1

ROOT="${ROOT:-$HOME/SkyGDR}"
DATA_ROOT="${DATA_ROOT:-/data/danyang}"
cd "$ROOT"
source "$DATA_ROOT/venvs/vllm/bin/activate"

export BASELINE_RUN_ROOT="$ROOT/results/pd_imitation_qwen3_8b_instruct"
export OFFLOAD_RUN_ROOT="$ROOT/results/pd_imitation_qwen3_8b_instruct_native_offload"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export SERVED_MODEL="${SERVED_MODEL:-Qwen3-8B-Instruct}"
export API_BASE="${API_BASE:-http://127.0.0.1:8000}"
export PORT="${PORT:-8000}"
export RUN_ROOT="$OFFLOAD_RUN_ROOT"

mkdir -p "$OFFLOAD_RUN_ROOT"/{logs,data,summary}

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

echo "========== OFFLOAD ONLY: clean GPU =========="
stop_all_vllm

echo "========== OFFLOAD SERVER =========="
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --generation-config vllm \
  --kv-offloading-size 32 \
  --kv-offloading-backend native \
  --disable-hybrid-kv-cache-manager \
  2>&1 | tee "$RUN_ROOT/logs/vllm_qwen3_8b_instruct_native_kv_offload.log" &
wait_health
trap 'stop_all_vllm' EXIT

echo "========== OFFLOAD PIPELINE =========="
# shellcheck source=run_pd_imitation_full.sh
# Inline same steps as pipeline_for_run_root in run_pd_imitation_full.sh
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

python3 src/tools/pd_collect_openai_samples.py \
  --api_base "$API_BASE" \
  --model "$SERVED_MODEL" \
  --mode prefill \
  --input_jsonl "$RUN_ROOT/data/prefill_prompts.jsonl" \
  --endpoint completion \
  --parallel_requests 4 \
  --out_csv "$RUN_ROOT/data/prefill_samples.csv"

python3 src/tools/pd_build_bucket_prompts.py \
  --model_or_tokenizer "$MODEL_PATH" \
  --target_tokens 2048,4096,8192,16384,24576,28672 \
  --samples_per_bucket 12 \
  --prefix_file "$RUN_ROOT/data/long_dialogue_prefix.txt" \
  --out "$RUN_ROOT/data/decode_prompts.jsonl"

python3 src/tools/pd_collect_openai_samples.py \
  --api_base "$API_BASE" \
  --model "$SERVED_MODEL" \
  --mode decode \
  --input_jsonl "$RUN_ROOT/data/decode_prompts.jsonl" \
  --generated_tokens 128,256,512 \
  --endpoint completion \
  --parallel_requests 4 \
  --out_csv "$RUN_ROOT/data/decode_samples.csv"

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

stop_all_vllm
trap - EXIT

echo "========== COMPARE REPORT =========="
python3 src/tools/pd_imitation_report.py \
  --results_dir results/pd_imitation_qwen3_8b_instruct \
  --compare_results_dir results/pd_imitation_qwen3_8b_instruct_native_offload \
  --base_label baseline \
  --compare_label native_offload

echo "DONE offload-only. Log: $LOG"
