# PD Imitation Compare Report

## 1. 对照范围

- baseline results: `results/pd_imitation_qwen3_8b_instruct`
- compare results: `results/pd_imitation_qwen3_8b_instruct_native_offload`
- baseline label: `baseline`
- compare label: `native_offload`

## 2. 关键结论

1. 在最大共享 prefill bucket `prompt=28672` 上，`native_offload` 相比 `baseline` 的 prefill 平均延迟变化为 `+0.19%`。
2. 在最大共享 decode bucket `context=28672, gen=512` 上，`native_offload` 相比 `baseline` 的 decode `ms/token` 变化为 `-0.38%`。
3. 如果 `compare_label` 是 native CPU offloading，这两个量就是最值得先看的主指标：prefill 会不会被拉长，steady-state decode 会不会变差。

## 3. Prefill 对照

![compare prefill](../fig/compare_prefill_latency.svg)

| prompt tokens | baseline mean ms | compare mean ms | delta % |
| --- | ---: | ---: | ---: |
| 2048 | 590.92 | 598.42 | +1.27% |
| 4096 | 1182.17 | 1180.42 | -0.15% |
| 8192 | 2654.33 | 2660.33 | +0.23% |
| 16384 | 6310.08 | 6313.08 | +0.05% |
| 24576 | 11278.25 | 11287.00 | +0.08% |
| 28672 | 14518.67 | 14546.00 | +0.19% |

## 4. Decode 对照（gen=512）

![compare decode](../fig/compare_decode_g512_mspt.svg)

| context tokens | baseline ms/token | compare ms/token | delta % |
| --- | ---: | ---: | ---: |
| 2048 | 12.53 | 12.56 | +0.22% |
| 4096 | 14.00 | 14.03 | +0.22% |
| 8192 | 17.37 | 17.37 | -0.01% |
| 16384 | 24.42 | 24.40 | -0.11% |
| 24576 | 32.45 | 32.37 | -0.27% |
| 28672 | 34.33 | 34.20 | -0.38% |

## 5. 使用建议

- 如果 offloading 主要拉长的是大 context 下的 prefill，说明 CPU 侧 KV 搬运已经开始影响长 prompt 请求。
- 如果 offloading 主要拉长的是 `gen=512` 的 decode `ms/token`，说明它已经影响 steady-state decode，而不仅仅是固定开销。
- 如果只有小 generation bucket 变差而 `g=512` 变化不大，优先把它解释为固定开销或短序列效应，而不是 steady-state decode 退化。

