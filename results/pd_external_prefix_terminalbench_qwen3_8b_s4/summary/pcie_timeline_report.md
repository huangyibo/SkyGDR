# External Prefix-Cache PCIe Report: pd_external_prefix_terminalbench_qwen3_8b_s4

## 1. 观测范围

- full window start: `1775015653631`
- full window end: `1775015688774`
- requests: `24`

本报告针对的是：

- `seed`：首轮长 prompt，把完整前缀写入 external/shared prefix cache
- `reuse`：后续各轮只追加少量新 token，观察 prefill 是否从外部 cache 读回大部分历史 KV

## 2. 时序图

![pcie timeline](pcie_timeline.svg)

### TX

![pcie tx timeline](pcie_tx_timeline.svg)

### RX

![pcie rx timeline](pcie_rx_timeline.svg)

### Per-Request Zoom

![pcie request zooms](pcie_request_zooms.svg)

## 3. 全窗口统计

- duration: `35.143 s`
- total TX volume: `23.841 GiB`
- total RX volume: `78.645 GiB`
- total bidirectional volume: `102.486 GiB`
- avg TX bandwidth: `0.678 GiB/s`
- avg RX bandwidth: `2.238 GiB/s`
- avg bidirectional bandwidth: `2.916 GiB/s`
- peak TX bandwidth: `15.472 GiB/s`
- peak RX bandwidth: `38.931 GiB/s`
- peak total bandwidth: `40.562 GiB/s`

## 4. 分阶段统计

| phase | duration (s) | tx total (GiB) | rx total (GiB) | total (GiB) | avg tx GiB/s | avg rx GiB/s | peak tx GiB/s | peak rx GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed | 10.136 | 4.916 | 11.442 | 16.358 | 0.485 | 1.129 | 15.456 | 28.319 |
| reuse | 24.828 | 18.925 | 67.202 | 86.128 | 0.762 | 2.707 | 15.472 | 38.931 |

## 5. 请求级汇总

- request summary csv: `request_pcie_summary.csv`

重点建议：

- 看 `reuse` 请求的 `lmcache_remote_read_GiB` 和 `peak_rx_GiB_s`，这是最直接的 external prefix-cache prefill load 信号。
- 看 `seed` 请求的 `lmcache_remote_write_GiB` 和 `peak_tx_GiB_s`，它反映首轮长前缀是怎么被写入外部 cache 的。
- 如果 `lmcache_hit_ratio` 很高但 `peak_rx_GiB_s` 不高，通常意味着外部读回被更平滑地摊开了，而不是没有命中。

