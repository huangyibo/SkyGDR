# Terminal-Bench External Prefix-Cache 主线代码说明

这份文档解释当前仓库里与 `PD imitation` 相关的**最新主线代码**。

当前已经不再以 synthetic prefix 或单 session 串行 workload 为主。现在的目标是：

- 用 **真实 Terminal-Bench 2.0 trajectories**
- 构造 **多 session / 分轮并发 reuse**
- 通过 `vLLM + LMCache` 模拟 external/shared prefix-cache 命中
- 尽量把 **prefill 侧 aggregate external read / PCIe RX** 打高

## 1. 整体结构

当前主线分成 5 层：

1. 真实 workload 生成
2. 分组并发执行 workload
3. GPU/PCIe 采样
4. PCIe 分析
5. 汇总报告

对应文件：

- [pd_build_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_external_prefix_workload.py)
- [pd_run_external_prefix_workload.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_run_external_prefix_workload.py)
- [gpu_metrics_logger.py](/Users/daniel/Documents/code/SkyGDR/src/tools/gpu_metrics_logger.py)
- [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py)
- [pd_imitation_report.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_report.py)
- [run_pd_external_prefix_imitation.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_external_prefix_imitation.sh)

## 2. 端到端数据流

```text
Terminal-Bench trajectories
  -> pd_build_external_prefix_workload.py
  -> selected_terminalbench_rows.jsonl + trajectory_workload.jsonl
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

## 3. 各模块职责

## 3.1 `pd_build_external_prefix_workload.py`

职责：

- 下载或读取 `Terminal-Bench 2.0 Trajectories`
- 从 row 中抽 `steps / trajectory / messages` 等轨迹字段
- 渲染成真实 transcript
- 选出足够长的真实 session
- 生成：
  - `seed`
  - 多个 `reuse`
  - 多 session 并发调度用的 `dispatch_group`

关键点：

- `seed` 是每条 session 的首轮长前缀
- `reuse` 是后续只追加小 suffix 的 turn
- 每个 `reuse_round_*` 会把多条 session 放到同一个 `dispatch_group`
- 输出顺序明确是：
  - 先所有 session 的 `seed`
  - 再 `reuse_round_001`
  - 再 `reuse_round_002`
  - 依次类推
- token 长度保持 `chunk_size_tokens` 对齐，默认 `256`

输出：

- `selected_terminalbench_rows.jsonl`
- `trajectory_workload.jsonl`

`trajectory_workload.jsonl` 里最关键的字段：

- `request_id`
- `phase`
- `dispatch_group`
- `dispatch_group_size`
- `session_id`
- `turn_id`
- `prompt_tokens`
- `reused_prefix_tokens_est`
- `appended_tokens_est`
- `reuse_ratio_est`
- `source_row_id`
- `source_total_tokens`
- `prompt_text`

## 3.2 `pd_run_external_prefix_workload.py`

职责：

- 按 `dispatch_group` 执行 workload
- 一个 group 内可并发发请求
- group 前后抓一次 `/metrics`
- 把 LMCache metrics 明确记录为 **group 级归因**

这和旧版本最大的区别是：

- 旧版是单请求串行，尝试做 request-level diff
- 新版是并发 workload，主真值是 group-level remote read/write

输出：

- `trajectory_samples.csv`

当前最重要的字段：

- `dispatch_group`
- `dispatch_group_size`
- `schedule_index`
- `group_submit_ts_unix_ms`
- `group_response_finish_ts_unix_ms`
- `group_lmcache_remote_read_GiB`
- `group_lmcache_remote_write_GiB`
- `group_lmcache_hit_ratio`
- `metrics_attribution_scope`

说明：

- 现在 `lmcache_*` 的请求级字段默认不再作为主真值
- 主真值是 `group_lmcache_*`

## 3.3 `gpu_metrics_logger.py`

职责：

- 用 NVML 周期采样 GPU / PCIe / CPU 指标

这条主线最关键的字段：

- `pcie_tx_GiB_s`
- `pcie_rx_GiB_s`
- `pcie_total_GiB_s`
- `pcie_tx_cum_GiB`
- `pcie_rx_cum_GiB`
- `pcie_total_cum_GiB`

理解口径：

- `RX` 更接近 external/shared cache 读回到 GPU
- `TX` 更接近写入外部 cache 或其它 GPU->host 方向流量

## 3.4 `pd_pcie_offload_analyze.py`

职责：

- 把 `gpu_metrics.csv` 和 `trajectory_samples.csv` 对齐
- 生成总图、分方向图、request zoom 和摘要

这版特别处理了一个新问题：

- `reuse_round_*` 内请求可能并发重叠
- `seed` 和 `reuse` 需要严格按阶段顺序执行

所以这份脚本会：

- 继续保留 request-level 可视化
- 但 phase 统计改成按 **dispatch_group interval** 聚合
- 避免把并发重叠窗口重复累计

主要输出：

- `pcie_timeline.svg`
- `pcie_tx_timeline.svg`
- `pcie_rx_timeline.svg`
- `pcie_request_zooms.svg`
- `summary/request_focus/*.svg`
- `request_pcie_summary.csv`
- `pcie_timeline_report.md`

## 3.5 `pd_imitation_report.py`

职责：

- 从 `request_pcie_summary.csv` 和 `pcie_timeline_summary.json` 生成最终摘要

新版本重点是：

- 面向真实 Terminal-Bench workload
- 强调 group-level LMCache remote read
- 强调 `reuse` 阶段 aggregate RX，而不是单请求微小 burst

## 3.6 `run_pd_external_prefix_imitation.sh`

职责：

- 一键串起整条主线

当前默认行为：

1. 清理旧 `vllm`
2. 启动 `vllm + LMCache`
3. 扫描 Terminal-Bench
4. 选择足够长的真实轨迹
5. 生成多 session workload
6. 并发执行 `reuse_round_*`
7. 采样 PCIe
8. 生成图和报告

当前脚本还内置了 3 个实验 profile：

- `prefill_max`
- `balanced_dual_pressure`
- `decode_heavy`

默认是：

- `balanced_dual_pressure`

这意味着当前默认主线不是“只把 prefill 打满”，而是：

- 保留大历史前缀和并发 reuse
- 同时把 decode 拉到足够长，避免把 decode 压得过轻

## 4. 现在最该改哪些参数

如果你要把 prefill 侧 aggregate load 打高，优先改这些：

- `NUM_SESSIONS`
- `GROUP_CONCURRENCY`
- `MAX_NUM_SEQS`
- `SLEEP_BETWEEN_GROUPS_MS`
- `SEED_PROMPT_TOKENS`
- `APPEND_TOKENS`
- `MOCK_READ_GBPS`

大方向是：

- 增加并发 session 数
- 保持很高 reuse
- 保持 append 很小
- 减少组间空隙

## 5. 当前代码口径

当前主线已经默认接受这几个事实：

- 单请求 burst 再高，也不一定能持续很久
- 真正接近系统瓶颈，靠的是 **多 session 聚合 prefill load**
- 因此现在最重要的结果，不再是：
  - “某个单请求持续了多久”
- 而是：
  - `reuse` 阶段整体 RX 平台是否抬高
  - `group_lmcache_remote_read_GiB` 是否稳定变大
