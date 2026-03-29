# External Prefix-Cache Imitation Report

## 1. 实验目标

这份结果专门针对：

- 单 GPU 条件下的 external/shared prefix-cache imitation
- 首轮长 prompt 先写入 LMCache 外部后端
- 后续每轮只追加少量 chunk-aligned token
- 观察 prefill 是否从外部 prefix cache 读回大部分历史 KV

当前主流程显式关闭了 vLLM 自己的 GPU prefix caching，所以重复前缀不会被 GPU 本地缓存遮住；
reuse turn 的主命中路径应该来自 LMCache 外部后端。

## 2. 关键结果

- seed requests: `1`
- reuse requests: `5`
- mean estimated text-side reuse ratio: `99.00%`
- mean LMCache hit ratio: `20.00%`
- total reuse remote read volume: `10.336 GiB`
- total reuse remote write volume: `3.516 GiB`
- mean reuse remote read volume: `2.067 GiB/request`
- mean reuse peak RX: `31.981 GiB/s`
- mean reuse peak TX: `3.458 GiB/s`
- seed remote write volume: `0.000 GiB`
- mean seed peak TX: `15.452 GiB/s`

最值得先看的 reuse turn：

- request `turn_003_reuse`: remote read `10.336 GiB`, hit ratio `100.00%`, peak RX `32.067 GiB/s`, peak TX `3.221 GiB/s`

## 3. 全局 PCIe 图

![total timeline](pcie_timeline.svg)

![tx timeline](pcie_tx_timeline.svg)

![rx timeline](pcie_rx_timeline.svg)

## 4. 分阶段统计

| phase | duration (s) | total transfer (GiB) | avg TX GiB/s | avg RX GiB/s | peak TX GiB/s | peak RX GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed | 4.621 | 5.079 | 1.091 | 0.008 | 15.452 | 0.049 |
| reuse | 9.566 | 23.305 | 0.262 | 2.174 | 4.123 | 35.759 |

## 5. turn 级摘要

| request_id | phase | turn | prompt tokens | reuse ratio | LMCache hit ratio | remote read GiB | remote write GiB | peak RX GiB/s | peak TX GiB/s | elapsed ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| turn_000_seed | seed | 0 | 24832 | 0.00% | 0.00% | 0.000 | 0.000 | 0.049 | 15.452 | 3413.00 |
| turn_001_reuse | reuse | 1 | 25088 | 98.98% | 0.00% | 0.000 | 0.000 | 35.530 | 3.189 | 677.00 |
| turn_002_reuse | reuse | 2 | 25344 | 98.99% | 0.00% | 0.000 | 0.000 | 27.852 | 3.299 | 690.00 |
| turn_003_reuse | reuse | 3 | 25600 | 99.00% | 100.00% | 10.336 | 3.516 | 32.067 | 3.221 | 697.00 |
| turn_004_reuse | reuse | 4 | 25856 | 99.01% | 0.00% | 0.000 | 0.000 | 28.699 | 4.123 | 700.00 |
| turn_005_reuse | reuse | 5 | 26112 | 99.02% | 0.00% | 0.000 | 0.000 | 35.759 | 3.457 | 699.00 |

## 6. 如何解读

- `seed` 阶段的关键动作是把首轮长 prompt 的 KV 写入外部 prefix cache，所以它通常更偏向 `remote write + TX`。
- `reuse` 阶段的关键动作是从外部 prefix cache 读回历史 KV，只对新增 suffix 做新的 prefill，所以它应该更值得看 `remote read + RX`。
- 如果 `LMCache hit ratio` 已经很高，但 `peak RX` 仍不高，通常说明外部读回被平滑摊开了，而不是命中没发生。
- 如果 `LMCache remote write GiB` 在 reuse turn 仍然明显偏大，往往表示每轮新增 suffix 太长，或者还在写入很多非复用块。

