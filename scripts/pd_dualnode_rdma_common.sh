#!/usr/bin/env bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DEFAULT="$(cd "${COMMON_DIR}/.." && pwd)"

if [[ -f "${COMMON_DIR}/pd_dualnode_rdma.env" ]]; then
  # shellcheck disable=SC1091
  source "${COMMON_DIR}/pd_dualnode_rdma.env"
fi

ROOT="${ROOT:-$ROOT_DEFAULT}"
RUN_NAME="${RUN_NAME:-pd_dualnode_rdma}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/$RUN_NAME}"

LOCAL_HOST_ALIAS="${LOCAL_HOST_ALIAS:-amd0}"
REMOTE_HOST_ALIAS="${REMOTE_HOST_ALIAS:-amd1}"
LOCAL_HOST_IP="${LOCAL_HOST_IP:-45.76.230.97}"
REMOTE_HOST_IP="${REMOTE_HOST_IP:-144.202.52.73}"

DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/vllm-dev:nightly}"
CONTAINER_NAME="${CONTAINER_NAME:-pd-lmcache}"
HF_CACHE_HOST_DIR="${HF_CACHE_HOST_DIR:-$HOME/.cache/huggingface}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
SERVED_MODEL="${SERVED_MODEL:-Qwen3-8B-Instruct}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

LMCACHE_SERVER_PORT="${LMCACHE_SERVER_PORT:-65432}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-18080}"
LMCACHE_SERVER_L1_SIZE_GB="${LMCACHE_SERVER_L1_SIZE_GB:-64}"
LMCACHE_SERVER_MAX_WORKERS="${LMCACHE_SERVER_MAX_WORKERS:-4}"
REMOTE_URL="${REMOTE_URL:-lm://$LOCAL_HOST_IP:$LMCACHE_SERVER_PORT}"
REMOTE_SERDE="${REMOTE_SERDE:-naive}"
LOCAL_CPU_SIZE_GB="${LOCAL_CPU_SIZE_GB:-32}"

PREFILL_PORT="${PREFILL_PORT:-7100}"
DECODER_PORT="${DECODER_PORT:-7200}"
DECODER_INIT_PORT="${DECODER_INIT_PORT:-7300}"
DECODER_ALLOC_PORT="${DECODER_ALLOC_PORT:-7400}"
PD_PROXY_PORT="${PD_PROXY_PORT:-7500}"
PROXY_PORT="${PROXY_PORT:-9100}"

PREFILL_GPU_INDEX="${PREFILL_GPU_INDEX:-3}"
DECODER_GPU_INDEX="${DECODER_GPU_INDEX:-3}"
PREFILL_RDMA_DEVICE="${PREFILL_RDMA_DEVICE:-}"
DECODER_RDMA_DEVICE="${DECODER_RDMA_DEVICE:-}"

PREFILL_PD_BUFFER_SIZE="${PREFILL_PD_BUFFER_SIZE:-1073741824}"
DECODER_PD_BUFFER_SIZE="${DECODER_PD_BUFFER_SIZE:-2147483648}"
PD_BUFFER_DEVICE="${PD_BUFFER_DEVICE:-cuda}"

LMCACHE_PIP_SPEC="${LMCACHE_PIP_SPEC:-lmcache==0.4.2}"
NIXL_PIP_SPEC="${NIXL_PIP_SPEC:-nixl==1.0.0}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-fastapi uvicorn httpx}"

PYTHONHASHSEED_VALUE="${PYTHONHASHSEED_VALUE:-0}"
LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"
NIXL_LOG_LEVEL="${NIXL_LOG_LEVEL:-INFO}"
UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-warn}"
UCX_TLS_RDMA="${UCX_TLS_RDMA:-rc,cuda_copy,cuda_ipc,self,sm}"
UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-rdmacm}"

SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-32}"
SMOKE_PROMPT_REPETITIONS="${SMOKE_PROMPT_REPETITIONS:-384}"
REUSE_TURNS="${REUSE_TURNS:-3}"
REUSE_MAX_TOKENS="${REUSE_MAX_TOKENS:-32}"
REUSE_PROMPT_REPETITIONS="${REUSE_PROMPT_REPETITIONS:-384}"
REUSE_APPEND_REPETITIONS="${REUSE_APPEND_REPETITIONS:-8}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-600}"

mkdir -p "${RUN_ROOT}"/{config,logs,pids,runtime,data,probe}

log() {
  printf '[pd-dualnode-rdma] %s\n' "$*"
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "missing required command: $cmd" >&2
    exit 1
  }
}

docker_container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

docker_container_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

docker_exec() {
  docker exec "${CONTAINER_NAME}" "$@"
}

role_log_path() {
  printf '%s/logs/%s.log' "${RUN_ROOT}" "$1"
}

role_pid_path() {
  printf '%s/pids/%s.pid' "${RUN_ROOT}" "$1"
}

role_launch_path() {
  printf '%s/runtime/%s.launch.sh' "${RUN_ROOT}" "$1"
}

role_config_path() {
  printf '%s/config/%s.yaml' "${RUN_ROOT}" "$1"
}

known_bnxt_for_gpu() {
  case "$1" in
    0) echo "bnxt_re1" ;;
    1) echo "bnxt_re3" ;;
    2) echo "bnxt_re2" ;;
    3) echo "bnxt_re0" ;;
    4) echo "bnxt_re5" ;;
    5) echo "bnxt_re8" ;;
    6) echo "bnxt_re7" ;;
    7) echo "bnxt_re4" ;;
    *) return 1 ;;
  esac
}

active_bnxt_devices() {
  rdma link 2>/dev/null | awk '$1 == "link" && $2 ~ /^bnxt_re/ && $4 == "ACTIVE" {split($2, a, "/"); print a[1]}'
}

netdev_for_rdma_device() {
  local dev="$1"
  rdma link 2>/dev/null | awk -v dev="${dev}" '$1 == "link" && $2 ~ ("^" dev "/") {for (i = 1; i <= NF; ++i) if ($i == "netdev") {print $(i+1); exit}}'
}

resolve_rdma_device() {
  local gpu_index="$1"
  local preferred="${2:-}"
  if [[ -n "${preferred}" ]]; then
    echo "${preferred}"
    return 0
  fi

  local host
  host="$(hostname -f 2>/dev/null || hostname)"
  if [[ "${host}" == chi-mi325x-pod2-* || "${host}" == chi-mi325x-* ]]; then
    local known
    known="$(known_bnxt_for_gpu "${gpu_index}" || true)"
    if [[ -n "${known}" ]] && active_bnxt_devices | grep -Fxq "${known}"; then
      echo "${known}"
      return 0
    fi
  fi

  local fallback
  fallback="$(active_bnxt_devices | head -n 1)"
  if [[ -n "${fallback}" ]]; then
    echo "${fallback}"
    return 0
  fi

  echo "unable to resolve an ACTIVE bnxt_re device for GPU index ${gpu_index}" >&2
  return 1
}

wait_http() {
  local url="$1"
  local timeout="${2:-300}"
  local start
  start="$(date +%s)"
  while ! curl -fsS "${url}" >/dev/null 2>&1; do
    if (( "$(date +%s)" - start >= timeout )); then
      echo "timed out waiting for ${url}" >&2
      return 1
    fi
    sleep 1
  done
}

ensure_container_running() {
  require_cmd docker

  if docker_container_running; then
    log "container ${CONTAINER_NAME} already running"
    return 0
  fi

  if docker_container_exists; then
    log "starting existing container ${CONTAINER_NAME}"
    docker start "${CONTAINER_NAME}" >/dev/null
    return 0
  fi

  local docker_args=(
    run -d
    --name "${CONTAINER_NAME}"
    --network host
    --ipc host
    --device /dev/kfd
    --device /dev/dri
    --device /dev/infiniband
    -v "${ROOT}:${ROOT}"
    -w "${ROOT}"
  )

  if [[ -d "${HF_CACHE_HOST_DIR}" ]]; then
    docker_args+=(-v "${HF_CACHE_HOST_DIR}:/root/.cache/huggingface")
  fi
  if [[ -d /data ]]; then
    docker_args+=(-v /data:/data)
  fi
  if [[ -d /datasets ]]; then
    docker_args+=(-v /datasets:/datasets)
  fi
  if [[ -d /mnt ]]; then
    docker_args+=(-v /mnt:/mnt)
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    docker_args+=(-e "HF_TOKEN=${HF_TOKEN}")
  fi

  docker_args+=("${DOCKER_IMAGE}" sleep infinity)

  log "creating container ${CONTAINER_NAME} from ${DOCKER_IMAGE}"
  docker "${docker_args[@]}" >/dev/null
}

ensure_container_python_runtime() {
  ensure_container_running

  if docker_exec bash -lc 'python3 -c "import vllm, lmcache, nixl, httpx, fastapi, uvicorn"' >/dev/null 2>&1; then
    log "python runtime already ready inside ${CONTAINER_NAME}"
    return 0
  fi

  log "installing python runtime inside ${CONTAINER_NAME}"
  docker_exec bash -lc "
    set -euo pipefail
    python3 -m pip install --upgrade pip
    if [[ -d '${ROOT}/LMCache' ]]; then
      python3 -m pip install -e '${ROOT}/LMCache'
    else
      python3 -m pip install '${LMCACHE_PIP_SPEC}'
    fi
    python3 -m pip install '${NIXL_PIP_SPEC}' ${EXTRA_PIP_PACKAGES}
  "
}

write_prefiller_config() {
  local path="$1"
  cat >"${path}" <<EOF
local_cpu: true
max_local_cpu_size: ${LOCAL_CPU_SIZE_GB}

remote_url: "${REMOTE_URL}"
remote_serde: "${REMOTE_SERDE}"

retrieve_locations: ["LocalCPUBackend", "RemoteBackend"]

enable_pd: true
transfer_channel: "nixl"
pd_role: "sender"
pd_proxy_host: "${LOCAL_HOST_IP}"
pd_proxy_port: ${PD_PROXY_PORT}
pd_buffer_size: ${PREFILL_PD_BUFFER_SIZE}
pd_buffer_device: "${PD_BUFFER_DEVICE}"
nixl_backends: [UCX]

save_unfull_chunk: true
EOF
}

write_decoder_config() {
  local path="$1"
  cat >"${path}" <<EOF
local_cpu: true
max_local_cpu_size: ${LOCAL_CPU_SIZE_GB}

remote_url: "${REMOTE_URL}"
remote_serde: "${REMOTE_SERDE}"

retrieve_locations: ["PDBackend"]
store_location: "RemoteBackend"

enable_pd: true
transfer_channel: "nixl"
pd_role: "receiver"
pd_peer_host: "${REMOTE_HOST_IP}"
pd_peer_init_port: ${DECODER_INIT_PORT}
pd_peer_alloc_port: ${DECODER_ALLOC_PORT}
pd_buffer_size: ${DECODER_PD_BUFFER_SIZE}
pd_buffer_device: "${PD_BUFFER_DEVICE}"
nixl_backends: [UCX]

save_decode_cache: true
save_unfull_chunk: true
EOF
}

write_launch_script() {
  local path="$1"
  local body="$2"
  cat >"${path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
${body}
EOF
  chmod +x "${path}"
}

start_launch_script_in_container() {
  local role="$1"
  local launch_path="$2"
  local log_path
  local pid_path
  log_path="$(role_log_path "${role}")"
  pid_path="$(role_pid_path "${role}")"

  docker_exec bash -lc "
    set -euo pipefail
    mkdir -p '${RUN_ROOT}/logs' '${RUN_ROOT}/pids'
    if [[ -f '${pid_path}' ]] && kill -0 \$(cat '${pid_path}') 2>/dev/null; then
      kill \$(cat '${pid_path}') >/dev/null 2>&1 || true
      sleep 2
    fi
    nohup bash '${launch_path}' >'${log_path}' 2>&1 </dev/null &
    echo \$! >'${pid_path}'
  "
}

stop_role_process() {
  local role="$1"
  local pid_path
  pid_path="$(role_pid_path "${role}")"
  if docker_container_running; then
    docker_exec bash -lc "
      set -euo pipefail
      if [[ -f '${pid_path}' ]] && kill -0 \$(cat '${pid_path}') 2>/dev/null; then
        kill \$(cat '${pid_path}') >/dev/null 2>&1 || true
        sleep 2
        kill -9 \$(cat '${pid_path}') >/dev/null 2>&1 || true
      fi
      rm -f '${pid_path}'
    " >/dev/null 2>&1 || true
  fi
}

print_runtime_versions() {
  docker_exec bash -lc '
    python3 - <<'"'"'PY'"'"'
import importlib
for name in ("vllm", "lmcache", "nixl"):
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '\''__version__'\'', '\''unknown'\'')}")
PY
  '
}
