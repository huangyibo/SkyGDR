# Prefill-Restore Imitation 代码说明

这份文档解释当前仓库里与 `PD imitation` 相关的**新主线代码**。

当前已经不再维护旧的：

- `prefill-only`
- `decode-only`
- `logical trace`

流程。

现在的唯一目标是：

- 模拟 **长 session / 高 prefix reuse / 中间有 eviction pressure**
- 观察下一轮 `reuse turn` 的 prefill 是否出现明显的 **RX/H2D restore**

## 1. 新主线的整体结构

新的链路分成 5 个模块：

1. workload 生成
2. workload 执行
3. GPU/PCIe 指标采样
4. PCIe 分析
5. 汇总报告

对应文件是：

- [pd_build_trajectory_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_trajectory_workload.py)
- [pd_run_restore_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_run_restore_workload.py)
- [gpu_metrics_logger.py](/Users/daniel/Documents/code/SkyGDR/src/tools/gpu_metrics_logger.py)
- [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py)
- [pd_imitation_report.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_report.py)
- [run_pd_restore_imitation.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_restore_imitation.sh)

## 2. 端到端数据流

新链路的数据流是：

```text
session_prefix.txt + pressure_prefix.txt
  -> pd_build_trajectory_workload.py
  -> trajectory_workload.jsonl
  -> pd_run_restore_workload.py
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

## 3.1 `pd_build_trajectory_workload.py`

路径：

- [pd_build_trajectory_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_trajectory_workload.py)

职责：

- 生成“主 session + pressure burst”的完整请求日程

它不再生成独立 token bucket，而是生成按顺序执行的 workload：

- `warmup`
- `pressure`
- `reuse`

### 它输出什么

输出文件：

- `trajectory_workload.jsonl`

每条记录至少包含：

- `request_id`
- `launch_group`
- `phase`
- `session_id`
- `turn_id`
- `prompt_tokens`
- `reused_prefix_tokens_est`
- `appended_tokens_est`
- `reuse_ratio_est`
- `expected_restore`
- `max_tokens`
- `prompt_text`

### 它怎么构造主 session

主 session 的 prompt 是严格递增的：

- `turn_0 = base_prefix + append_0`
- `turn_1 = turn_0 + append_1`
- `turn_2 = turn_1 + append_2`

这样保证：

- 下一轮 prompt 以前一轮 prompt 为严格前缀
- prefix caching 可以直接复用旧前缀

### 它怎么构造 pressure burst

每个后续主 turn 之前，脚本会先插入若干个无关长请求：

- `pressure_rounds_per_turn`
- `pressure_burst_size`

这些请求的目标不是“算业务结果”，而是：

- 占据 prefix cache 和 KV 空间
- 强制制造 eviction 压力

## 3.2 `pd_run_restore_workload.py`

路径：

- [pd_run_restore_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_run_restore_workload.py)

职责：

- 按 workload JSONL 的顺序执行请求

它的执行规则是：

- 相同 `launch_group` 的请求并发发出
- 不同 `launch_group` 按顺序推进

所以：

- `pressure` burst 内部可以并发
- `warmup -> pressure -> reuse` 之间保持顺序

### 它输出什么

输出文件：

- `trajectory_samples.csv`

关键字段包括：

- `request_id`
- `launch_group`
- `phase`
- `session_id`
- `turn_id`
- `prompt_tokens`
- `reused_prefix_tokens_est`
- `reuse_ratio_est`
- `submit_ts_unix_ms`
- `finish_ts_unix_ms`
- `elapsed_ms`
- `usage_prompt_tokens`
- `usage_completion_tokens`
- `http_status`
- `error`

这个 CSV 是后面 PCIe 对齐的主时间轴。

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
- `pcie_link_ref_GiB_s`

这几个字段里：

- `TX` 更接近 eviction / 写出压力
- `RX` 更接近 restore / 拉回压力
- `total = tx + rx` 只适合看总体流量，不适合替代方向分析

## 3.4 `pd_pcie_offload_analyze.py`

路径：

- [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py)

职责：

- 把 GPU metrics 和请求时间轴对齐
- 输出总图、TX 图、RX 图
- 输出请求级的 PCIe 汇总

### 它现在的输入

- `metrics_csv`
- `request_csv`

这里已经不再接受旧的：

- `prefill_csv`
- `decode_csv`

### 它现在的输出

- `pcie_timeline.svg`
- `pcie_tx_timeline.svg`
- `pcie_rx_timeline.svg`
- `pcie_timeline_window.csv`
- `request_pcie_summary.csv`
- `pcie_timeline_summary.json`
- `pcie_timeline_report.md`

### 它为什么更适合新主线

因为它现在是围绕 phase 来分析：

- `warmup`
- `pressure`
- `reuse`

而不是围绕旧的：

- `prefill`
- `decode`

所以它更直接地回答：

- `pressure` 阶段有没有把 TX 打起来
- `reuse` 阶段有没有把 RX 打起来

## 3.5 `pd_imitation_report.py`

路径：

- [pd_imitation_report.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_report.py)

职责：

- 把请求级结果和 PCIe 汇总压成一份简洁报告

它现在关注的不是：

- bucket latency
- logical trace

而是：

- `reuse ratio`
- `reuse peak RX`
- `pressure peak TX`
- phase-level PCIe 统计

### 当前最重要的输出

- `pd_imitation_report.md`

这份报告的重点是帮助你快速判断：

- 这轮是不是已经接近你想要的“prefill restore 主导”形态

## 3.6 `run_pd_restore_imitation.sh`

路径：

- [run_pd_restore_imitation.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_restore_imitation.sh)

职责：

- 新主线的唯一 shell 入口

它负责：

1. 清理旧 `vllm`
2. 启 `vllm serve`
3. 开 `prefix caching + native offloading`
4. 启 `gpu_metrics_logger.py`
5. 生成 workload
6. 执行 workload
7. 分析 PCIe
8. 生成最终报告

它现在已经替代了旧的：

- `run_pd_imitation_full.sh`
- `run_pd_imitation_offload_only.sh`

## 4. 为什么旧文件被删掉了

下面这些文件已经不再属于新主线，所以被移除了：

- `src/tools/pd_imitation_trace.py`
- `scripts/run_pd_imitation_full.sh`
- `scripts/run_pd_imitation_offload_only.sh`
- 旧版语义的 `pd_build_bucket_prompts.py`
- 旧版语义的 `pd_collect_openai_samples.py`

原因很简单：

- 它们的中心问题是“时间建模”和“逻辑 trace”
- 而你现在的问题是“prefill restore 的 RX/H2D 能不能被打出来”

继续保留旧主线只会增加歧义。

## 5. 现在最应该怎么看结果

如果你要判断实验是否成功，优先级是：

1. 看 `pcie_rx_timeline.svg`
2. 看 `request_pcie_summary.csv` 里 `phase=reuse`
3. 看 `pcie_tx_timeline.svg`
4. 看 `request_pcie_summary.csv` 里 `phase=pressure`

最关键的问题是：

- `pressure` 是否足够强到把主 session prefix 挤走
- `reuse` 是否真的触发了 H2D restore

## 6. 当前代码的假设

这套新主线的隐含假设是：

1. `vLLM prefix caching` 会让新 turn 复用旧前缀
2. `native KV offloading` 会在显存压力下把部分 KV 放到 CPU
3. pressure burst 能够提高旧 prefix 被挤走的概率
4. 如果旧 prefix 已经不在 GPU，那么下一轮高复用 prefill 更可能拉高 RX

需要注意的是：

- 这仍然是单机近似，不是真双机 PD
- 所以我们观测到的是：
  - host memory <-> GPU
- 而不是：
  - remote prefill engine <-> decode engine

## 7. 一句话总结

如果把现在这套代码压成一句话，它做的是：

- 用单 GPU、prefix caching、native offloading 和人为 pressure burst，
- 去构造一个更接近“多轮 agent session” 的 restore-focused imitation，
- 然后用 TX/RX 分开的 PCIe 观测去判断：
- `reuse turn` 的 prefill 到底有没有明显的 H2D restore。
