#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/pd_dualnode_rdma_common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pd_dualnode_rdma.sh <amd0|amd1> <action>

Actions:
  probe
  prepare
  start-lmcache-server
  start-proxy
  start-prefiller
  start-decoder
  start-local-stack
  smoke
  reuse
  status
  stop
  destroy-container

Examples:
  bash scripts/run_pd_dualnode_rdma.sh amd0 probe
  bash scripts/run_pd_dualnode_rdma.sh amd0 prepare
  bash scripts/run_pd_dualnode_rdma.sh amd1 prepare
  bash scripts/run_pd_dualnode_rdma.sh amd1 start-decoder
  bash scripts/run_pd_dualnode_rdma.sh amd0 start-local-stack
  bash scripts/run_pd_dualnode_rdma.sh amd0 smoke
  bash scripts/run_pd_dualnode_rdma.sh amd0 reuse
EOF
}

require_local_role() {
  local node="$1"
  if [[ "${HOST_NODE_ROLE}" != "${node}" ]]; then
    echo "action ${ACTION} must run on ${node}, current role is ${HOST_NODE_ROLE}" >&2
    exit 1
  fi
}

resolve_host_role() {
  local requested="$1"
  case "${requested}" in
    amd0|prefiller)
      HOST_NODE_ROLE="amd0"
      NODE_HOST_IP="${LOCAL_HOST_IP}"
      GPU_INDEX="${PREFILL_GPU_INDEX}"
      RDMA_DEVICE="$(resolve_rdma_device "${PREFILL_GPU_INDEX}" "${PREFILL_RDMA_DEVICE}")"
      ;;
    amd1|decoder)
      HOST_NODE_ROLE="amd1"
      NODE_HOST_IP="${REMOTE_HOST_IP}"
      GPU_INDEX="${DECODER_GPU_INDEX}"
      RDMA_DEVICE="$(resolve_rdma_device "${DECODER_GPU_INDEX}" "${DECODER_RDMA_DEVICE}")"
      ;;
    *)
      echo "unknown node role: ${requested}" >&2
      usage
      exit 1
      ;;
  esac

  RDMA_NETDEV="$(netdev_for_rdma_device "${RDMA_DEVICE}")"
  if [[ -z "${RDMA_NETDEV}" ]]; then
    echo "failed to resolve netdev for ${RDMA_DEVICE}" >&2
    exit 1
  fi
  UCX_NET_DEVICES_VALUE="${UCX_NET_DEVICES_VALUE:-${RDMA_DEVICE}:1}"
}

prepare_runtime() {
  ensure_container_python_runtime
  log "runtime versions:"
  print_runtime_versions
}

write_prefiller_launch() {
  local role="prefiller"
  local launch_path config_path
  launch_path="$(role_launch_path "${role}")"
  config_path="$(role_config_path "${role}")"
  write_prefiller_config "${config_path}"
  write_launch_script "${launch_path}" "export PYTHONHASHSEED=${PYTHONHASHSEED_VALUE}
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_VISIBLE_DEVICES=${GPU_INDEX}
export ROCR_VISIBLE_DEVICES=${GPU_INDEX}
export HSA_VISIBLE_DEVICES=${GPU_INDEX}
export UCX_TLS=${UCX_TLS_RDMA}
export UCX_NET_DEVICES=${UCX_NET_DEVICES_VALUE}
export UCX_SOCKADDR_TLS_PRIORITY=${UCX_SOCKADDR_TLS_PRIORITY}
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL}
export LMCACHE_CONFIG_FILE=${config_path}
export LMCACHE_LOG_LEVEL=${LMCACHE_LOG_LEVEL}
export NIXL_LOG_LEVEL=${NIXL_LOG_LEVEL}
cd \"${ROOT}\"
exec vllm serve \"${MODEL_PATH}\" \\
  --served-model-name \"${SERVED_MODEL}\" \\
  --host 0.0.0.0 \\
  --port \"${PREFILL_PORT}\" \\
  --dtype \"${VLLM_DTYPE}\" \\
  --max-model-len \"${MAX_MODEL_LEN}\" \\
  --gpu-memory-utilization \"${GPU_MEMORY_UTILIZATION}\" \\
  --max-num-seqs \"${MAX_NUM_SEQS}\" \\
  --generation-config vllm \\
  --disable-log-requests \\
  --enforce-eager \\
  --no-enable-prefix-caching \\
  --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":false,\"lmcache_rpc_port\":\"producer1\"}}'"
}

write_decoder_launch() {
  local role="decoder"
  local launch_path config_path
  launch_path="$(role_launch_path "${role}")"
  config_path="$(role_config_path "${role}")"
  write_decoder_config "${config_path}"
  write_launch_script "${launch_path}" "export PYTHONHASHSEED=${PYTHONHASHSEED_VALUE}
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_VISIBLE_DEVICES=${GPU_INDEX}
export ROCR_VISIBLE_DEVICES=${GPU_INDEX}
export HSA_VISIBLE_DEVICES=${GPU_INDEX}
export UCX_TLS=${UCX_TLS_RDMA}
export UCX_NET_DEVICES=${UCX_NET_DEVICES_VALUE}
export UCX_SOCKADDR_TLS_PRIORITY=${UCX_SOCKADDR_TLS_PRIORITY}
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL}
export LMCACHE_CONFIG_FILE=${config_path}
export LMCACHE_LOG_LEVEL=${LMCACHE_LOG_LEVEL}
export NIXL_LOG_LEVEL=${NIXL_LOG_LEVEL}
cd \"${ROOT}\"
exec vllm serve \"${MODEL_PATH}\" \\
  --served-model-name \"${SERVED_MODEL}\" \\
  --host 0.0.0.0 \\
  --port \"${DECODER_PORT}\" \\
  --dtype \"${VLLM_DTYPE}\" \\
  --max-model-len \"${MAX_MODEL_LEN}\" \\
  --gpu-memory-utilization \"${GPU_MEMORY_UTILIZATION}\" \\
  --max-num-seqs \"${MAX_NUM_SEQS}\" \\
  --generation-config vllm \\
  --disable-log-requests \\
  --enforce-eager \\
  --no-enable-prefix-caching \\
  --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"discard_partial_chunks\":false,\"lmcache_rpc_port\":\"consumer1\",\"skip_last_n_tokens\":1}}'"
}

write_proxy_launch() {
  local role="proxy"
  local launch_path
  launch_path="$(role_launch_path "${role}")"
  write_launch_script "${launch_path}" "export PYTHONHASHSEED=${PYTHONHASHSEED_VALUE}
cd \"${ROOT}\"
exec python3 \"${ROOT}/LMCache/examples/disagg_prefill/disagg_proxy_server.py\" \\
  --host 0.0.0.0 \\
  --port \"${PROXY_PORT}\" \\
  --prefiller-host \"${LOCAL_HOST_IP}\" \\
  --prefiller-port \"${PREFILL_PORT}\" \\
  --num-prefillers 1 \\
  --decoder-host \"${REMOTE_HOST_IP}\" \\
  --decoder-port \"${DECODER_PORT}\" \\
  --decoder-init-port \"${DECODER_INIT_PORT}\" \\
  --decoder-alloc-port \"${DECODER_ALLOC_PORT}\" \\
  --proxy-host \"${LOCAL_HOST_IP}\" \\
  --proxy-port \"${PD_PROXY_PORT}\" \\
  --num-decoders 1"
}

write_lmcache_server_launch() {
  local role="lmcache-server"
  local launch_path
  launch_path="$(role_launch_path "${role}")"
  write_launch_script "${launch_path}" "export PYTHONHASHSEED=${PYTHONHASHSEED_VALUE}
cd \"${ROOT}\"
exec lmcache server \\
  --host 0.0.0.0 \\
  --port \"${LMCACHE_SERVER_PORT}\" \\
  --http-host 0.0.0.0 \\
  --http-port \"${LMCACHE_HTTP_PORT}\" \\
  --l1-size-gb \"${LMCACHE_SERVER_L1_SIZE_GB}\" \\
  --max-workers \"${LMCACHE_SERVER_MAX_WORKERS}\" \\
  --eviction-policy LRU"
}

show_probe() {
  log "host=$(hostname -f 2>/dev/null || hostname)"
  log "gpu_index=${GPU_INDEX} rdma_device=${RDMA_DEVICE} netdev=${RDMA_NETDEV}"
  printf '\n== rdma link ==\n'
  rdma link 2>/dev/null || true
  printf '\n== ibv_devices ==\n'
  ibv_devices 2>/dev/null || true
  printf '\n== ibstat %s ==\n' "${RDMA_DEVICE}"
  ibstat "${RDMA_DEVICE}" 2>/dev/null || true
  printf '\n== recommended exports ==\n'
  printf 'export %s_GPU_INDEX=%s\n' "$(tr '[:lower:]' '[:upper:]' <<<"${HOST_NODE_ROLE}")" "${GPU_INDEX}"
  printf 'export %s_RDMA_DEVICE=%s\n' "$(tr '[:lower:]' '[:upper:]' <<<"${HOST_NODE_ROLE}")" "${RDMA_DEVICE}"
  printf 'export UCX_NET_DEVICES=%s\n' "${UCX_NET_DEVICES_VALUE}"
}

show_status() {
  log "container: ${CONTAINER_NAME}"
  docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
  printf '\n== pids ==\n'
  ls -1 "${RUN_ROOT}/pids" 2>/dev/null || true
  printf '\n== health ==\n'
  if [[ "${HOST_NODE_ROLE}" == "amd0" ]]; then
    curl -fsS "http://127.0.0.1:${LMCACHE_HTTP_PORT}/api/healthcheck" || true
    printf '\n'
    curl -fsS "http://127.0.0.1:${PREFILL_PORT}/health" || true
    printf '\n'
    curl -fsS "http://127.0.0.1:${PROXY_PORT}/openapi.json" >/dev/null && echo "proxy: healthy" || true
  else
    curl -fsS "http://127.0.0.1:${DECODER_PORT}/health" || true
  fi
  printf '\n== tail logs ==\n'
  for name in lmcache-server prefiller proxy decoder; do
    if [[ -f "$(role_log_path "${name}")" ]]; then
      echo "-- ${name} --"
      tail -n 10 "$(role_log_path "${name}")" || true
    fi
  done
}

stop_all_services() {
  for role in prefiller decoder proxy lmcache-server; do
    stop_role_process "${role}"
  done
  if docker_container_running; then
    docker_exec bash -lc "pkill -f 'vllm serve' >/dev/null 2>&1 || true; pkill -f 'disagg_proxy_server.py' >/dev/null 2>&1 || true; pkill -f 'lmcache server' >/dev/null 2>&1 || true"
  fi
}

destroy_container() {
  stop_all_services
  if docker_container_exists; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
}

start_lmcache_server() {
  require_local_role "amd0"
  prepare_runtime
  write_lmcache_server_launch
  start_launch_script_in_container "lmcache-server" "$(role_launch_path "lmcache-server")"
  wait_http "http://127.0.0.1:${LMCACHE_HTTP_PORT}/api/healthcheck" 300
  log "lmcache server is ready on ${LOCAL_HOST_IP}:${LMCACHE_SERVER_PORT}"
}

start_proxy() {
  require_local_role "amd0"
  prepare_runtime
  write_proxy_launch
  start_launch_script_in_container "proxy" "$(role_launch_path "proxy")"
  wait_http "http://127.0.0.1:${PROXY_PORT}/openapi.json" 300
  log "proxy is ready on ${LOCAL_HOST_IP}:${PROXY_PORT}"
}

start_prefiller() {
  require_local_role "amd0"
  prepare_runtime
  write_prefiller_launch
  start_launch_script_in_container "prefiller" "$(role_launch_path "prefiller")"
  wait_http "http://127.0.0.1:${PREFILL_PORT}/health" 900
  log "prefiller is ready on ${LOCAL_HOST_IP}:${PREFILL_PORT}"
}

start_decoder() {
  require_local_role "amd1"
  prepare_runtime
  write_decoder_launch
  start_launch_script_in_container "decoder" "$(role_launch_path "decoder")"
  wait_http "http://127.0.0.1:${DECODER_PORT}/health" 900
  log "decoder is ready on ${REMOTE_HOST_IP}:${DECODER_PORT}"
}

start_local_stack() {
  require_local_role "amd0"
  start_lmcache_server
  start_proxy
  start_prefiller
}

run_smoke() {
  require_local_role "amd0"
  python3 "${ROOT}/src/tools/pd_dualnode_proxy_workload.py" \
    --api_base "http://127.0.0.1:${PROXY_PORT}" \
    --model "${SERVED_MODEL}" \
    --mode smoke \
    --prompt_repetitions "${SMOKE_PROMPT_REPETITIONS}" \
    --max_tokens "${SMOKE_MAX_TOKENS}" \
    --timeout_secs "${REQUEST_TIMEOUT_SECS}" \
    --out_csv "${RUN_ROOT}/data/smoke_samples.csv"
}

run_reuse() {
  require_local_role "amd0"
  python3 "${ROOT}/src/tools/pd_dualnode_proxy_workload.py" \
    --api_base "http://127.0.0.1:${PROXY_PORT}" \
    --model "${SERVED_MODEL}" \
    --mode reuse \
    --prompt_repetitions "${REUSE_PROMPT_REPETITIONS}" \
    --append_turns "${REUSE_TURNS}" \
    --append_repetitions "${REUSE_APPEND_REPETITIONS}" \
    --max_tokens "${REUSE_MAX_TOKENS}" \
    --timeout_secs "${REQUEST_TIMEOUT_SECS}" \
    --out_csv "${RUN_ROOT}/data/reuse_samples.csv"
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

ROLE="$1"
ACTION="$2"
resolve_host_role "${ROLE}"

case "${ACTION}" in
  probe)
    show_probe
    ;;
  prepare)
    prepare_runtime
    ;;
  start-lmcache-server)
    start_lmcache_server
    ;;
  start-proxy)
    start_proxy
    ;;
  start-prefiller)
    start_prefiller
    ;;
  start-decoder)
    start_decoder
    ;;
  start-local-stack)
    start_local_stack
    ;;
  smoke)
    run_smoke
    ;;
  reuse)
    run_reuse
    ;;
  status)
    show_status
    ;;
  stop)
    stop_all_services
    ;;
  destroy-container)
    destroy_container
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    usage
    exit 1
    ;;
esac
