# PCIe Offload Observation Report: pd_imitation_qwen3_8b_instruct

## 1. 观测范围

- full window start: `1774764313940`
- full window end: `1774764893228`
- prefill window: `1774764313940 -> 1774764429114`
- decode window: `1774764560570 -> 1774764893228`

这里记录的是 GPU 侧 NVML 提供的 PCIe TX/RX moving-average throughput。

- 最稳妥的主指标是 `pcie_total_GiB_s = tx + rx`。
- 在很多平台上，host->GPU 的 restore/offload-return 往往更容易体现在 GPU PCIe RX 上，但方向语义最好结合实测一起判断。

## 2. 时序图

![pcie timeline](pcie_timeline.svg)

## 3. 全窗口统计

- duration: `579.288 s`
- total TX volume: `3.252 GiB`
- total RX volume: `9.994 GiB`
- total bidirectional volume: `13.246 GiB`
- avg TX bandwidth: `0.006 GiB/s`
- avg RX bandwidth: `0.017 GiB/s`
- avg bidirectional bandwidth: `0.023 GiB/s`
- peak TX bandwidth: `0.022 GiB/s` at `374.327 s`
- peak RX bandwidth: `0.056 GiB/s` at `272.727 s`
- peak bidirectional bandwidth: `0.069 GiB/s` at `251.827 s`

## 4. 分阶段统计

| phase | duration (s) | tx total (GiB) | rx total (GiB) | total (GiB) | avg total GiB/s | peak total GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | 115.174 | 0.375 | 0.983 | 1.359 | 0.012 | 0.022 |
| decode | 332.658 | 2.778 | 8.926 | 11.704 | 0.035 | 0.069 |
| full | 579.288 | 3.252 | 9.994 | 13.246 | 0.023 | 0.069 |

## 5. 解读建议

- 如果你要抓“offloading 开始到最后 restore”的总体量，优先看 `full` 和 `decode` 的 `total_transfer_GiB`。
- 如果你要抓最容易体现 restore 的瞬时冲击，优先看 `decode` 窗口内的 `peak RX` 和 `peak total`。
- 如果 `total_transfer_GiB` 已经不小，但时间指标仍几乎不变，说明当前 offloading 流量可能存在，但还不足以成为端到端瓶颈。

