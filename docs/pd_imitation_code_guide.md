# External Prefix-Cache Imitation 代码说明

这份文档解释当前仓库里与 `PD imitation` 相关的**新主线代码**。

当前已经不再维护旧的：

- `prefill-only`
- `decode-only`
- `native offloading + pressure burst`
- `prefill-restore` 主流程

现在的唯一目标是：

- 模拟 **长 session / 高 prefix reuse / 外部 prefix cache 命中**
- 让 `reuse` turn 主要从 external/shared cache 读回历史 KV
- 同时观测：
  - `LMCache hit / remote read / remote write`
  - `PCIe RX / TX`

## 1. 新主线的整体结构

新的链路分成 5 个模块：

1. workload 生成
2. workload 执行与 LMCache 指标采样
3. GPU/PCIe 指标采样
4. PCIe 分析
5. 汇总报告

对应文件是：

- [pd_build_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_external_prefix_workload.py)
- [pd_run_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_run_external_prefix_workload.py)
- [gpu_metrics_logger.py](/Users/daniel/Documents/code/SkyGDR/src/tools/gpu_metrics_logger.py)
- [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py)
- [pd_imitation_report.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_report.py)
- [run_pd_external_prefix_imitation.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_external_prefix_imitation.sh)

## 2. 端到端数据流

新链路的数据流是：

```text
session_prefix.txt
  -> pd_build_external_prefix_workload.py
  -> trajectory_workload.jsonl
  -> pd_run_external_prefix_workload.py
  -> trajectory_samples.csv
  + gpu_metrics_logger.py
  -> gpu_metrics.csv
  -> pd_pcie_offload_analyze.py
  -> pcie_timeline.svg / pcie_tx_timeline.svg / pcie_rx_timeline.svg
  -> request_pcie_summary.csv / pcie_timeline_report.md
  -> pd_imitation_report.py
  -> pd_imitation_report.md
```

## 3. 各文件职责

## 3.1 `pd_build_external_prefix_workload.py`

路径：

- [pd_build_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_external_prefix_workload.py)

职责：

- 生成一个只有 `seed + reuse` 的多轮长 session workload

它不再生成：

- 独立 token bucket
- pressure burst
- eviction-heavy 干扰请求

### 它输出什么

输出文件：

- `trajectory_workload.jsonl`

每条记录至少包含：

- `request_id`
- `phase`
- `session_id`
- `turn_id`
- `prompt_tokens`
- `reused_prefix_tokens_est`
- `appended_tokens_est`
- `reuse_ratio_est`
- `expected_external_hit`
- `max_tokens`
- `prompt_text`

### 它怎么保证高复用

这份脚本不再拼接一堆独立 block。

它现在是：

- 先生成一段足够长的 token corpus
- 再取这个 corpus 的不同长度前缀

所以：

- `turn_0` 是长前缀
- `turn_1` 是 `turn_0` 的严格扩展
- `turn_2` 是 `turn_1` 的严格扩展

这样天然满足：

- 后一轮 prompt 以前一轮 prompt 为前缀

### 为什么要做 chunk 对齐

脚本会要求：

- 每轮总 prompt token 数对齐到 `chunk_size_tokens`

默认是 `256`。

这样做是为了：

- 让 LMCache 命中尽量落在完整 chunk 上
- 最大化 `reuse turn` 的 external cache hit

## 3.2 `pd_run_external_prefix_workload.py`

路径：

- [pd_run_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_run_external_prefix_workload.py)

职责：

- 顺序执行 workload
- 在每轮请求前后抓一次 `/metrics`
- 用 counter 差分把 LMCache 的远端读写归因到这轮请求

### 它和旧 runner 的本质区别

旧 runner 只负责：

- 发请求
- 记延迟

新 runner 额外负责：

- 抓 `LMCache` 指标
- 把每轮请求对应的：
  - `requested tokens`
  - `hit tokens`
  - `remote read bytes`
  - `remote write bytes`
  记下来

### 它输出什么

输出文件：

- `trajectory_samples.csv`

关键字段包括：

- `submit_ts_unix_ms`
- `response_finish_ts_unix_ms`
- `post_metrics_ts_unix_ms`
- `elapsed_ms`
- `usage_prompt_tokens`
- `usage_completion_tokens`
- `lmcache_requested_tokens`
- `lmcache_hit_tokens`
- `lmcache_hit_ratio`
- `lmcache_remote_read_GiB`
- `lmcache_remote_write_GiB`

### 为什么有 `post_metrics_ts_unix_ms`

这一步很关键。

因为 LMCache 的写回不一定在 response 返回的那个瞬间就完全结束。

所以脚本会：

1. 请求返回
2. 再等一个 `post_request_settle_ms`
3. 再抓一次 `/metrics`

这样：

- `seed` 阶段的远端写
- `reuse` 阶段的小量 suffix 写

都会更完整地计入当前请求。

## 3.3 `gpu_metrics_logger.py`

路径：

- [gpu_metrics_logger.py](/Users/daniel/Documents/code/SkyGDR/src/tools/gpu_metrics_logger.py)

职责：

- 用 NVML 周期采样 GPU / PCIe / CPU 指标

新主线里最关键的字段是：

- `pcie_tx_GiB_s`
- `pcie_rx_GiB_s`
- `pcie_total_GiB_s`
- `pcie_tx_cum_GiB`
- `pcie_rx_cum_GiB`
- `pcie_total_cum_GiB`

这里：

- `RX` 更接近 external cache 读回到 GPU
- `TX` 更接近写出/落盘到 external cache

所以现在的主解释方向是：

- `seed` 看 `TX`
- `reuse` 看 `RX`

## 3.4 `pd_pcie_offload_analyze.py`

路径：

- [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py)

职责：

- 把 GPU metrics 和请求时间轴对齐
- 输出总图、TX 图、RX 图
- 输出请求级 PCIe 汇总

### 它现在的输入

- `metrics_csv`
- `request_csv`

这里的 `request_csv` 现在已经包含 LMCache 请求级指标。

### 它现在的输出

- `pcie_timeline.svg`
- `pcie_tx_timeline.svg`
- `pcie_rx_timeline.svg`
- `pcie_timeline_window.csv`
- `request_pcie_summary.csv`
- `pcie_timeline_summary.json`
- `pcie_timeline_report.md`

### 它为什么更适合新主线

因为它现在围绕的是：

- `seed`
- `reuse`

而且 `request_pcie_summary.csv` 已经把：

- LMCache 远端读写
- PCIe 请求级读写

合并到一张表里。

这让我们可以直接回答：

- 这轮 reuse 到底从 external cache 读了多少
- 同一轮里 GPU RX/TX 又是什么样子

## 3.5 `pd_imitation_report.py`

路径：

- [pd_imitation_report.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_report.py)

职责：

- 把请求级结果和 PCIe 汇总压成一份简洁报告

它现在关注的不是：

- bucket latency
- logical trace
- pressure/offload 对照

而是：

- `mean LMCache hit ratio`
- `total reuse remote read GiB`
- `mean reuse peak RX`
- `seed remote write`

所以它回答的问题是：

- external prefix-cache hit 有没有真的发生
- 它是不是伴随着明显的 H2D / RX

## 3.6 `run_pd_external_prefix_imitation.sh`

路径：

- [run_pd_external_prefix_imitation.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_external_prefix_imitation.sh)

职责：

- 把整个 external prefix-cache imitation 从头串起来

它大致会做这些事：

1. 清理旧 `vllm`
2. 生成 `lmcache_config.yaml`
3. 启动 `vllm + LMCacheConnectorV1`
4. 关闭 vLLM 自己的 prefix caching
5. 启动 `gpu_metrics_logger.py`
6. 生成高复用 workload
7. 顺序执行每轮请求并采集 LMCache 计数器差分
8. 生成 PCIe 图和最终摘要

### 为什么主脚本默认用 mock backend

默认使用：

- `mock://...`

是因为这条线最适合当前单机 imitation：

- 不需要额外服务
- 仍然是 external backend 语义
- 还能直接调读写吞吐

如果以后要换成真正 centralized/shared backend，
主脚本只需要改：

- `LMCACHE_REMOTE_URL`

其余代码不用动。

## 4. 当前链路最关键的实验假设

当前这条线建立在这些假设上：

1. 如果关闭 vLLM GPU prefix cache，
   那么高复用 turn 的主命中路径应该来自 LMCache。

2. 如果每轮追加很小且 chunk 对齐，
   那么 `reuse turn` 的大部分历史前缀都应是 external cache hit。

3. 如果 LMCache hit ratio 高且 remote read GiB 大，
   那么 `reuse` 的 PCIe RX 就应当更值得重点观察。

## 5. 为什么这条线比旧版更对口

旧版主要在追：

- `native offloading`
- `evict-heavy`
- “能不能把旧 KV 挤出去再拉回来”

但现在这条线直接追的是：

- `external/shared prefix cache`
- `prefill loads historical KV from external storage`

所以它更接近你真正想 imitation 的论文语义。
