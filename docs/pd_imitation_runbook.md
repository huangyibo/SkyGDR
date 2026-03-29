# 单 GPU 条件下的 External Prefix-Cache Imitation 执行手册

这份手册已经**不再支持**旧的：

- `prefill-only / decode-only`
- `native offloading + pressure burst`
- `prefill-restore` 主流程

当前唯一支持的目标是：

- 用 `vLLM + LMCache`
- 显式关闭 vLLM 自己的 GPU prefix caching
- 把 LMCache 远端后端当作 **external/shared prefix cache**
- 构造一个**长 session / 高 prefix reuse / 每轮只追加少量 suffix**
- 直接观察：
  - `LMCache remote read bytes`
  - `LMCache hit tokens`
  - `PCIe RX/TX` 随时间变化

换句话说，这份流程要 imitation 的是这种场景：

- 一个 agent/session 连续跑很多轮
- 每一轮只追加很少的新 token
- 大部分历史上下文都复用
- 下一轮 prefill 主要不是重算历史 prefix
- 而是从 external prefix cache 把历史 KV 拉回来，再只算新增 suffix

你真正关心的结果是：

- `seed` 阶段：
  - 长前缀是否真的被写入 external cache
- `reuse` 阶段：
  - `LMCache remote read` 是否很大
  - `PCIe RX / H2D` 是否被显著抬高

## 1. 这一版和上一版的本质区别

上一版更像：

- 依赖 `native CPU offloading`
- 想用 eviction pressure 把旧 KV 从 GPU 挤走
- 再观察下一轮 restore

这一版更像：

- 直接把 LMCache 远端后端作为 **external/shared prefix cache**
- 不再依赖 GPU 内部 prefix cache
- 不再依赖 pressure burst
- 让每个 reuse turn 都走：
  - `external cache hit`
  - `remote read`
  - `GPU RX`

所以新版关注的是：

- **LMCache hit ratio**
- **LMCache remote read/write bytes**
- **reuse turn 的 PCIe RX**

而不是旧版的：

- `pressure` 有没有把 eviction 打起来
- `native offload` 有没有逼出 restore

## 2. 这一版的默认技术路线

### 2.1 模型

- 实际模型：`Qwen/Qwen3-8B`
- 服务名：`Qwen3-8B-Instruct`

### 2.2 服务参数

默认服务参数是：

- `dtype=bfloat16`
- `max-model-len=32768`
- `gpu-memory-utilization=0.85`
- `max-num-seqs=1`
- `--no-enable-prefix-caching`
- `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`

这里最重要的设计点有两个：

1. `--no-enable-prefix-caching`
   - 关闭 vLLM 自己的 GPU prefix cache
   - 避免 GPU 本地命中把 external cache 命中遮住

2. `LMCacheConnectorV1`
   - 把共享前缀的存取交给 LMCache
   - 让后续 reuse turn 真正产生 `remote read`

### 2.3 LMCache 默认配置

默认会写一份 `lmcache_config.yaml`，关键配置是：

- `chunk_size: 256`
- `local_cpu: false`
- `max_local_cpu_size: 10`（与 [LMCache Mock 官方示例](https://docs.lmcache.ai/kv_cache/storage_backends/mock.html) 一致；设为 `0` 可能导致无 Local CPU backend，远程后端健康检查失败，整站表现为 `LMCache is unhealthy`）
- `remote_serde: "naive"`
- `save_decode_cache: false`
- `save_unfull_chunk: false`

这样做的目的很明确：

- `chunk_size=256`
  - 让每轮追加也按 `256` token 对齐
  - 最大化可复用 chunk 的比例
- `local_cpu=false`
  - 关闭「本地 CPU 热缓存」语义上的优先使用；`max_local_cpu_size` 仍保留小容量以符合官方 mock 配置并让健康检查能分配测试缓冲区
- `save_decode_cache=false`
  - 不把重点放在 decode 生成出来的 KV
  - 让实验更聚焦于 prompt/prefix 复用
- `save_unfull_chunk=false`
  - 只缓存完整 chunk
  - 配合 chunk-aligned prompt，让行为更稳定、更容易解释

### 2.4 默认 external backend

默认 backend 是：

- `mock://...`

也就是 LMCache 官方提供的 **mock remote backend**。

这条默认路线的意义是：

- 不需要额外起 Valkey / centralized server
- 仍然是 LMCache 的 remote backend 语义
- 可以直接控制：
  - `peeking_latency`
  - `read_throughput`
  - `write_throughput`

默认参数是：

- `MOCK_STORAGE_GB=256`
- `MOCK_PEEKING_LATENCY_MS=1`
- `MOCK_READ_GBPS=40`
- `MOCK_WRITE_GBPS=8`

这组默认值的意图是：

- 让 `reuse` 阶段的 external read 足够强
- 同时让 `seed` 阶段的 external write 也能被观测到

如果你后面想换成更“真的”后端，可以覆盖：

```bash
export LMCACHE_REMOTE_URL='lm://127.0.0.1:65432'
```

或者别的官方支持后端 URL。主流程和分析脚本不用改。

## 3. Workload 设计

默认 workload 不再有 `pressure`。

它只包含：

- `seed`
- `reuse`

默认参数是：

- `seed_prompt_tokens = 24832`
- `append_tokens = 256,256,256,256,256`
- `num_turns = 6`
- `decode_tokens = 16`
- `chunk_size_tokens = 256`

这表示：

1. `turn_000_seed`
   - 先提交一个约 `24.8K` token 的长 prompt
   - 目的是把首轮长前缀写入 external cache

2. `turn_001_reuse` 以后
   - 每轮只追加 `256` 个新 token
   - 让前一轮的大部分 prefix 都可以从 external cache 复用

这里故意让：

- 每轮新增都是 `256` token
- 总 prompt 长度尽量保持 `256` 对齐

因为我们当前就是在追：

- **chunk-aligned external prefix hit**

### 为什么 decode 要故意短

这一版故意把：

- `decode_tokens = 16`

设得比较短。

因为当前目标不是测 steady-state decode，而是：

- 尽量让 prefill 阶段的 external read 更显眼
- 避免长 decode 自己又制造很多额外干扰

## 4. 输出目录

统一使用：

```bash
export ROOT=$HOME/SkyGDR
export RUN_ROOT=$ROOT/results/pd_external_prefix_imitation_qwen3_8b
```

最终主要产物包括：

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

## 5. 一次性环境准备

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
uv pip install lmcache "transformers>=4.51.0"
```

## 6. 直接运行

新的主流程只有一个入口：

- [run_pd_external_prefix_imitation.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_external_prefix_imitation.sh)

直接跑：

```bash
cd $ROOT
bash scripts/run_pd_external_prefix_imitation.sh
```

如果你想调外部后端强度，可以先设环境变量再跑：

```bash
export MOCK_READ_GBPS=60
export MOCK_WRITE_GBPS=8
export GPU_METRICS_INTERVAL_MS=20

bash scripts/run_pd_external_prefix_imitation.sh
```

如果你想换成真正的 centralized/shared backend：

```bash
export LMCACHE_REMOTE_URL='lm://127.0.0.1:65432'
bash scripts/run_pd_external_prefix_imitation.sh
```

## 7. 新主流程到底做了什么

这个脚本会依次做下面几件事：

1. 清理旧的 `vllm` 进程和残留显存
2. 写出 `lmcache_config.yaml`
3. 启动带 `LMCacheConnectorV1` 的 `vllm serve`
4. 启动 [gpu_metrics_logger.py](/Users/daniel/Documents/code/SkyGDR/src/tools/gpu_metrics_logger.py)
5. 用 [pd_build_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_external_prefix_workload.py) 生成多轮高复用 workload
6. 用 [pd_run_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_run_external_prefix_workload.py) 顺序执行请求，并在每轮前后抓 `/metrics`
7. 用 [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py) 生成 TX/RX/总带宽图和请求级统计
8. 用 [pd_imitation_report.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_report.py) 写最终摘要

## 8. 怎么看结果

### 8.1 先看 `trajectory_samples.csv`

最关键的请求级结果表是：

- `$RUN_ROOT/data/trajectory_samples.csv`

这里最值得看的是：

- `lmcache_requested_tokens`
- `lmcache_hit_tokens`
- `lmcache_hit_ratio`
- `lmcache_remote_read_GiB`
- `lmcache_remote_write_GiB`

如果你想 imitate：

- “prefill engines must load large volumes of KV-Cache from remote storage”

那最直接的成功信号就是：

- `reuse` 行里的 `lmcache_hit_ratio` 很高
- 同时 `lmcache_remote_read_GiB` 很大

### 8.2 再看 RX 图

你当前最重要的图是：

- [pcie_rx_timeline.svg](/Users/daniel/Documents/code/SkyGDR/results/pd_external_prefix_imitation_qwen3_8b/summary/pcie_rx_timeline.svg)

如果实验打对了，最值得期待的是：

- `reuse` 请求附近的 RX/H2D 明显抬高

### 8.3 再看 TX 图

- [pcie_tx_timeline.svg](/Users/daniel/Documents/code/SkyGDR/results/pd_external_prefix_imitation_qwen3_8b/summary/pcie_tx_timeline.svg)

这张图主要用来判断：

- `seed` 请求把长前缀写入 external cache 时，TX/D2H 有没有起来

### 8.4 请求级汇总 CSV

最有用的数据表是：

- [request_pcie_summary.csv](/Users/daniel/Documents/code/SkyGDR/results/pd_external_prefix_imitation_qwen3_8b/summary/request_pcie_summary.csv)

它已经把：

- LMCache 远端读写
- 请求级 RX/TX 总量
- 请求级 RX/TX 峰值

都合在一张表里了。

## 9. 如果 reuse 阶段效果还是不够强，优先调什么

最优先调的是这几个：

1. `seed_prompt_tokens`
   - 让首轮长前缀更长
2. `append_tokens`
   - 继续保持 `256` 对齐，但可以把 turn 数变多
3. `MOCK_READ_GBPS`
   - 把外部后端读吞吐设得更高
4. `GPU_METRICS_INTERVAL_MS`
   - 调到 `10-20ms`，更容易抓到瞬时 RX 峰值

如果你已经看到：

- `lmcache_hit_ratio` 很高
- `lmcache_remote_read_GiB` 很大

但 `RX` 图还不够尖，
那更像是：

- 外部读回发生了
- 只是被摊平在更长的 prefill 窗口里

这时不要先怀疑“没命中”，而是先看：

- 请求级 `lmcache_remote_read_GiB`
- `pcie_rx_timeline.svg`
- `request_pcie_summary.csv`

三者是否一致。
