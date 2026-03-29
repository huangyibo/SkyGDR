# Terminal-Bench 驱动的 External Prefix-Cache Imitation 手册

这份手册现在只服务一个目标：

- 使用 **真实 Terminal-Bench 2.0 agent trajectories**
- 在单 GPU 上通过 `vLLM + LMCache` 做 **external/shared prefix-cache imitation**
- 把 workload 改成 **多 session + 分轮并发 reuse**
- 尽量把 **prefill 侧 external read / PCIe RX** 打高

当前主线已经不再推荐：

- synthetic prefix 文本
- 单 session 串行 `reuse`
- 旧的 `native offloading + pressure burst`

## 1. 这版到底在 imitation 什么

要 imitation 的不是“单个长请求”，而是更接近 agentic workload 的场景：

- 同时存在多条 agent session
- 每条 session 都已经有很长的历史上下文
- 新一轮只追加很少的新 token
- 下一轮 prefill 主要工作是：
  - 从 external/shared prefix cache 读回历史 KV
  - 只对新增 suffix 做新的 prefill

为了尽量把 prefill 侧打满，这版做了两件重要的事：

1. **真实数据源**
   - 直接使用 `Terminal-Bench 2.0 Trajectories`
2. **多 session 并发 reuse**
   - 不是单条 session 一轮一轮串行跑
   - 而是多条 session 在同一个 `reuse_round_*` 里并发发出

## 2. 数据源

默认数据集：

- `yoonholee/terminalbench-trajectories`

参考：

- 数据集页：https://huggingface.co/datasets/yoonholee/terminalbench-trajectories
- Terminal-Bench 仓库：https://github.com/harbor-framework/terminal-bench

当前接入方式不是简单把整条消息直接喂进去，而是：

- 从 dataset row 中提取 `steps / trajectory / messages` 这类轨迹字段
- 渲染成连续 transcript
- 选择足够长的真实轨迹
- 再按 chunk-aligned token 长度切出：
  - `seed`
  - 后续多个 `reuse` turn

所以当前 workload 仍然保留了“高复用、chunk 对齐”的实验控制性，但内容来自真实 agent 轨迹。

## 3. 默认实验配置

### 3.1 模型与服务

- 实际模型：`Qwen/Qwen3-8B`
- 服务名：`Qwen3-8B-Instruct`

默认 `vllm serve` 关键参数：

- `dtype=bfloat16`
- `max-model-len=32768`
- `gpu-memory-utilization=0.85`
- `max-num-seqs=4`
- `--no-enable-prefix-caching`
- `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`

这里最关键的两点：

1. `--no-enable-prefix-caching`
   - 关闭 vLLM 自己的 GPU prefix cache
   - 避免 GPU 本地命中把 external/shared 命中遮掉

2. `max-num-seqs=4`
   - 允许多个 `reuse` 请求在同一轮并发
   - 让 aggregate prefill load 更容易抬高

### 3.2 LMCache 默认配置

默认会生成：

- `chunk_size: 256`
- `local_cpu: false`
- `max_local_cpu_size: 10`
- `remote_serde: "naive"`
- `save_decode_cache: false`
- `save_unfull_chunk: false`

默认 external backend 仍然是 LMCache 官方的 `mock://` remote backend，主要是为了单机快速 imitation。

默认相关参数：

- `MOCK_STORAGE_GB=256`
- `MOCK_PEEKING_LATENCY_MS=1`
- `MOCK_READ_GBPS=40`
- `MOCK_WRITE_GBPS=8`

### 3.3 Terminal-Bench workload 默认参数

默认参数已经改成更偏“把 prefill aggregate load 打高”的配置：

- `NUM_SESSIONS=4`
- `REUSE_TURNS_PER_SESSION=5`
- `SEED_PROMPT_TOKENS=24576`
- `APPEND_TOKENS=256`
- `DECODE_TOKENS=16`
- `MAX_ROWS_TO_SCAN=400`
- `GROUP_CONCURRENCY=4`
- `SLEEP_BETWEEN_GROUPS_MS=0`

这些参数的含义是：

- 先挑 `4` 条足够长的真实 Terminal-Bench 轨迹
- 每条先做一个 `seed` request
- 然后进入 `5` 个 `reuse_round_*`
- 每个 `reuse_round_*` 都会把这 `4` 条 session 同时发出去

这比单条 session 串行更接近：

- prefill engines 持续从外部存储加载历史 KV

## 4. 输出目录

统一使用：

```bash
export ROOT=$HOME/SkyGDR
export RUN_ROOT=$ROOT/results/pd_external_prefix_terminalbench_qwen3_8b
```

最关键的输出包括：

- `$RUN_ROOT/data/selected_terminalbench_rows.jsonl`
- `$RUN_ROOT/data/trajectory_workload.jsonl`
- `$RUN_ROOT/data/trajectory_samples.csv`
- `$RUN_ROOT/logs/gpu_metrics.csv`
- `$RUN_ROOT/logs/vllm_external_prefix.log`
- `$RUN_ROOT/summary/pcie_timeline.svg`
- `$RUN_ROOT/summary/pcie_tx_timeline.svg`
- `$RUN_ROOT/summary/pcie_rx_timeline.svg`
- `$RUN_ROOT/summary/request_pcie_summary.csv`
- `$RUN_ROOT/summary/pcie_timeline_report.md`
- `$RUN_ROOT/summary/pd_imitation_report.md`

## 5. 环境准备

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
uv pip install lmcache datasets "transformers>=4.51.0"
```

## 6. 一键运行

主入口只有一个：

- [run_pd_external_prefix_imitation.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_external_prefix_imitation.sh)

直接跑：

```bash
cd $ROOT
bash scripts/run_pd_external_prefix_imitation.sh
```

## 7. 推荐的“尽量把 prefill 打满”参数

如果你目标很明确，就是尽量把 prefill 侧 aggregate RX 拉高，我建议先从这组开始：

```bash
export NUM_SESSIONS=4
export GROUP_CONCURRENCY=4
export MAX_NUM_SEQS=4
export SLEEP_BETWEEN_GROUPS_MS=0
export SEED_PROMPT_TOKENS=24576
export APPEND_TOKENS=256
export DECODE_TOKENS=16
export GPU_METRICS_INTERVAL_MS=20

bash scripts/run_pd_external_prefix_imitation.sh
```

如果这组还不够，再往上推：

```bash
export NUM_SESSIONS=8
export GROUP_CONCURRENCY=8
export MAX_NUM_SEQS=8
export SLEEP_BETWEEN_GROUPS_MS=0
export MAX_ROWS_TO_SCAN=1000

bash scripts/run_pd_external_prefix_imitation.sh
```

这组更激进，目的就是：

- 让更多真实 session 在同一轮一起做 reuse prefill
- 让 aggregate external read 更接近系统瓶颈

## 8. 主流程到底做了什么

脚本会依次做这些事：

1. 清理旧的 `vllm` 进程和残留显存
2. 写出 `lmcache_config.yaml`
3. 启动带 `LMCacheConnectorV1` 的 `vllm serve`
4. 启动 [gpu_metrics_logger.py](/Users/daniel/Documents/code/SkyGDR/src/tools/gpu_metrics_logger.py)
5. 用 [pd_build_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_external_prefix_workload.py) 下载/扫描 Terminal-Bench，并生成真实 workload
6. 用 [pd_run_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_run_external_prefix_workload.py) 按 `dispatch_group` 执行请求
7. 用 [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py) 生成 PCIe TX/RX/总图
8. 用 [pd_imitation_report.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_report.py) 写摘要报告

## 9. 如何看结果

### 9.1 先看 workload 选了哪些真实轨迹

- `$RUN_ROOT/data/selected_terminalbench_rows.jsonl`

这个文件告诉你：

- 选中了哪些 Terminal-Bench row
- 每条轨迹一共有多少 step
- 总 token 大概有多少

### 9.2 再看 `trajectory_samples.csv`

- `$RUN_ROOT/data/trajectory_samples.csv`

现在最重要的不是旧的请求级 `lmcache_*` 字段，而是：

- `dispatch_group`
- `dispatch_group_size`
- `group_lmcache_remote_read_GiB`
- `group_lmcache_hit_ratio`
- `metrics_attribution_scope`

因为现在是并发模式，LMCache metrics 是按整个 `dispatch_group` 归因的。

### 9.3 最重要的是 RX 图

最重要的图是：

- [pcie_rx_timeline.svg](/Users/daniel/Documents/code/SkyGDR/results/pd_external_prefix_terminalbench_qwen3_8b/summary/pcie_rx_timeline.svg)

你要重点看：

- `reuse` 阶段的 aggregate RX 是否比 `seed` 更高
- 每个 `reuse_round_*` 是否形成连续高平台，而不是只有单个小 burst

### 9.4 再看总图和 request zoom

- `pcie_timeline.svg`
- `pcie_request_zooms.svg`
- `summary/request_focus/*.svg`

这三组图分别回答：

- 全局是不是被打高了
- 哪些 round 在打
- 单个请求的 burst 长什么样

## 10. 结果如何才算“更接近目标”

如果你要 mimic 的是“prefill side 持续 load 历史 KV”，那比起单请求峰值，更值得看这三件事：

1. `reuse` 阶段的 **平均 RX**
2. `reuse_round_*` 的 **group remote read GiB**
3. `reuse` 阶段是不是出现了更长、更连续的 RX 高平台

如果下一轮你发现：

- 单请求 burst 还是短
- 但 `reuse` 阶段整体 RX 平台明显变长
- 而且 group read 总量持续变大

那说明方向是对的。

## 11. 参考来源

- Terminal-Bench dataset: https://huggingface.co/datasets/yoonholee/terminalbench-trajectories
- Terminal-Bench repo: https://github.com/harbor-framework/terminal-bench
- LMCache mock backend: https://docs.lmcache.ai/kv_cache/storage_backends/mock.html
