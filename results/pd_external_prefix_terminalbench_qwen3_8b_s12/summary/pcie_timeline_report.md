# External Prefix-Cache PCIe Report: pd_external_prefix_terminalbench_qwen3_8b_s12

## 1. 观测范围

- full window start: `1775016670750`
- full window end: `1775016760939`
- requests: `72`

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

- duration: `90.189 s`
- total TX volume: `74.224 GiB`
- total RX volume: `237.338 GiB`
- total bidirectional volume: `311.562 GiB`
- avg TX bandwidth: `0.823 GiB/s`
- avg RX bandwidth: `2.632 GiB/s`
- avg bidirectional bandwidth: `3.455 GiB/s`
- peak TX bandwidth: `15.461 GiB/s`
- peak RX bandwidth: `42.881 GiB/s`
- peak total bandwidth: `42.886 GiB/s`

## 4. 分阶段统计

| phase | duration (s) | tx total (GiB) | rx total (GiB) | total (GiB) | avg tx GiB/s | avg rx GiB/s | peak tx GiB/s | peak rx GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed | 24.891 | 8.979 | 41.325 | 50.303 | 0.361 | 1.660 | 15.453 | 39.149 |
| reuse | 64.934 | 65.244 | 196.012 | 261.256 | 1.005 | 3.019 | 15.461 | 42.881 |

## 5. 请求级汇总

- request summary csv: `request_pcie_summary.csv`

重点建议：

- 看 `reuse` 请求的 `lmcache_remote_read_GiB` 和 `peak_rx_GiB_s`，这是最直接的 external prefix-cache prefill load 信号。
- 看 `seed` 请求的 `lmcache_remote_write_GiB` 和 `peak_tx_GiB_s`，它反映首轮长前缀是怎么被写入外部 cache 的。
- 如果 `lmcache_hit_ratio` 很高但 `peak_rx_GiB_s` 不高，通常意味着外部读回被更平滑地摊开了，而不是没有命中。

