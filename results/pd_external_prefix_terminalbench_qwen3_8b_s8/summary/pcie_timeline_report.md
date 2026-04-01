# External Prefix-Cache PCIe Report: pd_external_prefix_terminalbench_qwen3_8b_s8

## 1. 观测范围

- full window start: `1775015850371`
- full window end: `1775015912592`
- requests: `48`

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

- duration: `62.221 s`
- total TX volume: `45.264 GiB`
- total RX volume: `159.248 GiB`
- total bidirectional volume: `204.512 GiB`
- avg TX bandwidth: `0.727 GiB/s`
- avg RX bandwidth: `2.559 GiB/s`
- avg bidirectional bandwidth: `3.287 GiB/s`
- peak TX bandwidth: `15.468 GiB/s`
- peak RX bandwidth: `43.618 GiB/s`
- peak total bandwidth: `46.792 GiB/s`

## 4. 分阶段统计

| phase | duration (s) | tx total (GiB) | rx total (GiB) | total (GiB) | avg tx GiB/s | avg rx GiB/s | peak tx GiB/s | peak rx GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed | 17.501 | 7.789 | 26.569 | 34.359 | 0.445 | 1.518 | 15.468 | 36.641 |
| reuse | 44.457 | 37.474 | 132.678 | 170.152 | 0.843 | 2.984 | 15.452 | 43.618 |

## 5. 请求级汇总

- request summary csv: `request_pcie_summary.csv`

重点建议：

- 看 `reuse` 请求的 `lmcache_remote_read_GiB` 和 `peak_rx_GiB_s`，这是最直接的 external prefix-cache prefill load 信号。
- 看 `seed` 请求的 `lmcache_remote_write_GiB` 和 `peak_tx_GiB_s`，它反映首轮长前缀是怎么被写入外部 cache 的。
- 如果 `lmcache_hit_ratio` 很高但 `peak_rx_GiB_s` 不高，通常意味着外部读回被更平滑地摊开了，而不是没有命中。

