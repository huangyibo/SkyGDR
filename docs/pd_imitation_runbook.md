# 单 GPU 条件下的 Prefill-Restore Imitation 执行手册

这份手册已经**不再支持**旧的：

- `prefill-only`
- `decode-only`
- `logical pd_imitation_trace.csv`

主流程。

当前唯一支持的目标是：

- 用单 GPU + `vLLM native KV offloading` + `prefix caching`
- 构造一个**多轮 trajectory / 高 prefix reuse / 中间插 pressure burst**
- 观察下一轮 `reuse turn` 的 prefill 是否出现明显的 **RX/H2D restore**

换句话说，这份手册要 imitation 的是这种场景：

- 一个 agent/session 连续跑很多轮
- 每一轮只追加很少的新 token
- 大部分历史上下文都复用
- 中间因为别的请求制造了 eviction 压力
- 所以下一轮 prefill 需要把旧 KV 从 CPU 侧重新拉回 GPU

你真正关心的结果是：

- `pressure` 阶段：
  - TX / D2H eviction 是否真的被打起来
- `reuse` 阶段：
  - RX / H2D restore 是否真的被打起来

## 1. 这份流程和旧版有什么本质区别

旧版更像：

- 一批彼此独立的长请求
- 分别测 prefill 和 decode
- 再做逻辑 trace

新版更像：

- 同一个主 session 的多轮请求
- 新一轮 prompt 以前一轮 prompt 为严格前缀
- 中间插入无关长请求 burst
- 目的是把主 session 的 prefix KV 从 GPU 挤走
- 再观察下一轮高复用 prefill 的 restore 行为

所以新版关注的是：

- **prefix reuse**
- **eviction pressure**
- **reuse-turn prefill RX**

而不是旧版的：

- `prompt_tokens -> prefill_time`
- `(context, gen) -> decode_time`

## 2. 这一版的默认配置

### 2.1 模型

- 实际模型：`Qwen/Qwen3-8B`
- 服务名：`Qwen3-8B-Instruct`

### 2.2 服务参数

默认服务参数是：

- `dtype=bfloat16`
- `max-model-len=32768`
- `gpu-memory-utilization=0.70`
- `max-num-seqs=8`
- `--enable-prefix-caching`
- `--kv-offloading-backend native`
- `--kv-offloading-size 32`
- `--disable-hybrid-kv-cache-manager`

这里的设计目标很明确：

- 让 GPU 侧 KV 空间更紧
- 让 prefix cache 打开
- 让 CPU offloading 真正参与
- 让 pressure burst 更容易触发 eviction

### 2.3 Workload 参数

默认 workload 是：

- `base_prefix_tokens = 24576`
- `append_tokens = 256,256,256,256,256,256`
- `num_turns = 6`
- `main_decode_tokens = 32`
- `pressure_prompt_tokens = 28672`
- `pressure_burst_size = 8`
- `pressure_rounds_per_turn = 2`
- `pressure_decode_tokens = 1`

这表示：

1. 先跑一个主 session 的 `warmup turn`
2. 每个后续 turn 之前，插入两轮 pressure burst
3. 每轮 pressure burst 都发 `8` 个超长独立请求
4. 然后再发一个高复用的 `reuse turn`

这样做的目的就是：

- 让主 session 的旧 prefix KV 更有机会被挤到 CPU
- 下一轮再用它时，prefill 需要 restore

### 2.4 为什么 decode 要故意短

这一版故意把：

- `main_decode_tokens = 32`
- `pressure_decode_tokens = 1`

设得比较短。

因为当前目标不是测 steady-state decode，而是：

- 尽量让 `reuse turn` 的 prefill restore 更显眼
- 避免长 decode 自己又制造大量新的 eviction / D2H

## 3. 输出目录

统一使用：

```bash
export ROOT=$HOME/SkyGDR
export RUN_ROOT=$ROOT/results/pd_restore_imitation_qwen3_8b
```

最终主要产物包括：

- `$RUN_ROOT/data/trajectory_workload.jsonl`
- `$RUN_ROOT/data/trajectory_samples.csv`
- `$RUN_ROOT/logs/gpu_metrics.csv`
- `$RUN_ROOT/summary/pcie_timeline.svg`
- `$RUN_ROOT/summary/pcie_tx_timeline.svg`
- `$RUN_ROOT/summary/pcie_rx_timeline.svg`
- `$RUN_ROOT/summary/request_pcie_summary.csv`
- `$RUN_ROOT/summary/pcie_timeline_report.md`
- `$RUN_ROOT/summary/pd_imitation_report.md`

## 4. 一次性环境准备

下面默认在 GPU server 上执行。

```bash
export ROOT=$HOME/SkyGDR
export DATA_ROOT=/data/danyang
export VENV_PATH=$DATA_ROOT/venvs/vllm
export UV_CACHE_DIR=$DATA_ROOT/uv-cache
export TMPDIR=$DATA_ROOT/tmp
export HF_HOME=$DATA_ROOT/hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

mkdir -p $DATA_ROOT/{uv-cache,tmp,venvs,hf-cache}
cd $ROOT
```

如果还没有环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 --seed $VENV_PATH
source $VENV_PATH/bin/activate

uv pip install vllm --torch-backend=auto
uv pip install "transformers>=4.51.0"
```

## 5. 直接运行

新的主流程只有一个入口：

- [run_pd_restore_imitation.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_restore_imitation.sh)

直接跑：

```bash
cd $ROOT
bash scripts/run_pd_restore_imitation.sh
```

如果你想调压力，可以先设环境变量再跑：

```bash
export MAX_NUM_SEQS=8
export GPU_MEMORY_UTILIZATION=0.70
export PRESSURE_BURST_SIZE=8
export PRESSURE_ROUNDS_PER_TURN=2
export GPU_METRICS_INTERVAL_MS=20

bash scripts/run_pd_restore_imitation.sh
```

## 6. 新主流程到底做了什么

这个脚本会依次做下面几件事：

1. 清理旧的 `vllm` 进程和残留显存
2. 启动带 `prefix caching + native offloading` 的 `vllm serve`
3. 启动 [gpu_metrics_logger.py](/Users/daniel/Documents/code/SkyGDR/src/tools/gpu_metrics_logger.py)
4. 用 [pd_build_trajectory_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_trajectory_workload.py) 生成 trajectory workload
5. 用 [pd_run_restore_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_run_restore_workload.py) 执行 workload
6. 用 [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py) 生成 TX/RX/总带宽图和请求级统计
7. 用 [pd_imitation_report.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_report.py) 写最终摘要

## 7. 怎么看结果

### 7.1 先看 RX 图

你当前最重要的图是：

- [pcie_rx_timeline.svg](/Users/daniel/Documents/code/SkyGDR/results/pd_restore_imitation_qwen3_8b/summary/pcie_rx_timeline.svg)

如果你想 imitate “prefill engines must load large volumes of KV-Cache from remote storage”，
那最值得看的就是：

- `reuse` 阶段里，RX/H2D 有没有明显抬高

### 7.2 再看 TX 图

- [pcie_tx_timeline.svg](/Users/daniel/Documents/code/SkyGDR/results/pd_restore_imitation_qwen3_8b/summary/pcie_tx_timeline.svg)

这张图主要用来判断：

- `pressure` 阶段到底有没有把 eviction 压力打起来

### 7.3 请求级 CSV

最有用的数据表是：

- [request_pcie_summary.csv](/Users/daniel/Documents/code/SkyGDR/results/pd_restore_imitation_qwen3_8b/summary/request_pcie_summary.csv)

重点看：

- `phase=pressure`
  - `peak_tx_GiB_s`
  - `tx_total_GiB`
- `phase=reuse`
  - `peak_rx_GiB_s`
  - `rx_total_GiB`

## 8. 如果 reuse 阶段 RX 还是不高，通常说明什么

最常见的解释有三个：

1. 主 session 的旧 prefix KV 其实还留在 GPU 上
2. restore 确实发生了，但被更细粒度地摊平了，没有形成大尖峰
3. pressure burst 还不够强，没能把主 session 的 prefix KV 真正挤走

这时优先调这几个参数：

- `MAX_NUM_SEQS`
- `GPU_MEMORY_UTILIZATION`
- `PRESSURE_BURST_SIZE`
- `PRESSURE_ROUNDS_PER_TURN`
- `BASE_PREFIX_TOKENS`

## 9. 这一版不再做什么

这份流程不再支持：

- 旧的 baseline/offload 双结果目录对照
- 旧的 `pd_imitation_trace.csv`
- 旧的 `prefill-only / decode-only` 两阶段采样
- 旧的 `timing-first` 主线

如果你现在要回答的问题是：

- “在高 prefix reuse 的长 session 里，prefill restore 的 H2D 到底有没有被打出来？”

那新的这条主线才是当前仓库里的正确入口。
