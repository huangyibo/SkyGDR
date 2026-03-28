# Qwen3-8B-Instruct PD Imitation Results Report

## 1. 数据范围

本报告基于 `results/pd_imitation_qwen3_8b_instruct` 的一轮完整 phase-1 采样结果。

- 模型：`Qwen/Qwen3-8B`，服务名：`Qwen3-8B-Instruct`
- 采样对象：`prefill-only`、`decode-only`、逻辑 `pd_imitation_trace.csv`
- `prefill` 成功 bucket：`2048 ~ 28672` tokens
- `decode` 成功 bucket：context `2048 ~ 28672`，generation `128/256/512`
- 逻辑 trace 行数：`18`

## 2. 关键结论

1. `prefill` 延迟随 prompt 长度单调上升，从 `2048` tokens 的 `590.92 ms` 增长到 `28672` tokens 的 `14518.67 ms`。
2. `prefill throughput` 在中等长度区间最高，峰值出现在 `2048` tokens，约为 `3465.80 tokens/s`；到 `28672` tokens 时回落到 `1974.84 tokens/s`。
3. 当前结果没有采 `g=32`，因此更适合直接把最大 generation bucket 视为 steady-state proxy。本轮 `g=512` 的 decode `ms/token` 区间约为 `12.53 ~ 34.33 ms/token`。
4. 逻辑 KV footprint 与 context 线性相关，本轮最大点是 `28672` tokens，对应 `3.94 GiB` 的 decode-side KV。

## 3. Prefill 结果

![Prefill latency](../fig/prefill_latency.svg)

![Prefill throughput](../fig/prefill_throughput.svg)

聚合结果：

| prompt tokens | samples | mean latency (ms) | std (ms) | throughput (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 12 | 590.92 | 114.22 | 3465.80 |
| 4096 | 12 | 1182.17 | 236.63 | 3464.82 |
| 8192 | 12 | 2654.33 | 456.47 | 3086.27 |
| 16384 | 12 | 6310.08 | 1045.12 | 2596.48 |
| 24576 | 12 | 11278.25 | 1341.89 | 2179.06 |
| 28672 | 12 | 14518.67 | 782.74 | 1974.84 |

解读：

- `prefill latency` 基本随 token 数增加而近似线性上升，但在 `8K -> 16K` 区间已经出现更明显的超线性拉长。
- `prefill throughput` 不是单调增加的：它在 `2K ~ 4K` 左右最好，之后随着上下文变长开始回落。
- 这意味着如果你后面要做 PD imitation，prefill cost 不能只按“每 token 固定时间”处理，长上下文区间最好单独建桶。

## 4. Decode 结果

![Decode ms/token](../fig/decode_ms_per_token.svg)

| context tokens | gen tokens | samples | mean total latency (ms) | ms/token | tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | 128 | 12 | 1716.17 | 13.41 | 74.58 |
| 2048 | 256 | 12 | 3224.58 | 12.60 | 79.39 |
| 2048 | 512 | 12 | 6417.75 | 12.53 | 79.78 |
| 4096 | 128 | 12 | 1936.92 | 15.13 | 66.08 |
| 4096 | 256 | 12 | 3572.83 | 13.96 | 71.65 |
| 4096 | 512 | 12 | 7168.00 | 14.00 | 71.43 |
| 8192 | 128 | 12 | 2550.58 | 19.93 | 50.18 |
| 8192 | 256 | 12 | 4398.00 | 17.18 | 58.21 |
| 8192 | 512 | 12 | 8893.33 | 17.37 | 57.57 |
| 16384 | 128 | 12 | 4026.58 | 31.46 | 31.79 |
| 16384 | 256 | 12 | 6190.50 | 24.18 | 41.35 |
| 16384 | 512 | 12 | 12505.17 | 24.42 | 40.94 |
| 24576 | 128 | 12 | 5884.08 | 45.97 | 21.75 |
| 24576 | 256 | 12 | 8292.33 | 32.39 | 30.87 |
| 24576 | 512 | 12 | 16616.33 | 32.45 | 30.81 |
| 28672 | 128 | 12 | 6965.67 | 54.42 | 18.38 |
| 28672 | 256 | 12 | 9073.58 | 35.44 | 28.21 |
| 28672 | 512 | 12 | 17578.92 | 34.33 | 29.13 |

解读：

- 当前 `decode-only` 口径本质上是“长 context + 指定 generation length 的整段 elapsed time”，不是纯 kernel 级 decode 时间。
- 更大的 generation bucket 更接近稳定区间。本轮里，`g=512` 的 decode 吞吐区间约为 `29.13 ~ 79.78 tokens/s`。
- 对后续 case study，如果你需要一个更稳的 decode proxy，建议优先使用最大的 generation bucket；当前结果就是 `g=512`。

## 5. 逻辑 KV Footprint

![KV footprint](../fig/kv_footprint_gib.svg)

本轮 trace 使用固定模型参数计算出：`KV_bytes_per_token = 147456`，也就是每 token `144 KiB`。

对应关系非常直接：

- `2048` context: `0.28 GiB`
- `4096` context: `0.56 GiB`
- `8192` context: `1.12 GiB`
- `16384` context: `2.25 GiB`
- `24576` context: `3.38 GiB`
- `28672` context: `3.94 GiB`

这部分结论对 PD imitation 很关键：

- prefill 端产出的逻辑 KV 量与 prompt/context 长度线性相关。
- 即使不跑真实 PD，本轮也已经足够给后续 offloading / replay 提供一个量级可信的 KV 大小映射。

## 6. 对当前 trace 的使用建议

如果你现在要把这批结果送入后续 case study，我建议直接采用下面的口径：

1. `prefill_time_ms` 直接取当前 trace 里的桶均值。
2. `decode_time_ms` 如果是做粗粒度 phase-1 模拟，可以保留当前值。
3. 如果你更关心 steady-state decode，优先采信最大的 generation bucket；当前结果里建议使用 `g=512`。
4. 如果你要构造接近上限的长上下文 workload，建议把最大 bucket 保持在略低于上限的位置，并在相同并发下验证成功率。

## 7. 当前局限

- 这仍然是 `single-GPU` 的 phase-1 imitation，不是完整 PD serving。
- `decode-only` 当前口径包含固定开销，因此短 generation 桶会被高估。
- 当前 trace 还没有引入真实请求到达分布，也没有引入跨机传输带宽限制。

## 8. 输出位置

- trace: `results/pd_imitation_qwen3_8b_instruct/summary/pd_imitation_trace.csv`
- summary: `results/pd_imitation_qwen3_8b_instruct/summary/pd_imitation_summary.json`
- figures: `results/pd_imitation_qwen3_8b_instruct/fig`

