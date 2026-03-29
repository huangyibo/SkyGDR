# PD Imitation 代码说明

这份文档解释当前仓库里所有和 `PD imitation` 相关的代码，重点回答四件事：

1. 这套代码整体想做什么
2. 各个脚本分别负责哪一段
3. 它们之间如何串起来
4. 如果后面要继续扩展，应该改哪一层

这份说明覆盖的代码主要包括：

- `src/tools/pd_build_bucket_prompts.py`
- `src/tools/pd_collect_openai_samples.py`
- `src/tools/pd_imitation_trace.py`
- `src/tools/pd_imitation_report.py`
- `src/tools/gpu_metrics_logger.py`
- `src/tools/pd_pcie_offload_analyze.py`
- `scripts/run_pd_imitation_full.sh`
- `scripts/run_pd_imitation_offload_only.sh`
- `docs/pd_imitation_runbook.md`

## 1. 整体目标

这套代码的目标不是直接实现一个真实的 `prefill/decode disaggregation` 系统，而是在单 GPU 条件下构造一套可落地的 `PD imitation` 流程。

它的核心思路是：

- 用 `vLLM` 跑真实模型服务
- 分别测量：
  - `prefill-only`
  - `decode-only`
- 用模型结构参数推导 `KV bytes per token`
- 再把这三部分合成为一份逻辑 trace：
  - `pd_imitation_trace.csv`

后面又在这个基础上扩了一层：

- 让 baseline 和 native CPU offloading 跑同一套 workload
- 用 `gpu_metrics_logger.py` 记录 GPU/PCIe/CPU 侧观测
- 用 `pd_pcie_offload_analyze.py` 把时序图和统计结果整理出来

所以现在这套代码其实分成两条线：

- 主线 A：`timing -> logical trace -> report`
- 主线 B：`GPU metrics -> PCIe timeline -> offload observation report`

## 2. 代码分层

从职责上看，可以把这套代码分成 6 层。

### 2.1 Prompt 构造层

文件：

- `src/tools/pd_build_bucket_prompts.py`

职责：

- 根据目标 token bucket 生成长度严格受控的 prompt
- 输出为 JSONL，供后续采样脚本直接读取

它解决的问题是：

- 如果 prompt 长度不可控，那么 `prefill` 和 `decode` 的时间数据就不干净
- 所以这里先用 tokenizer 把输入严格对齐到指定 token 数

### 2.2 请求采样层

文件：

- `src/tools/pd_collect_openai_samples.py`

职责：

- 调用 `vLLM` 的 OpenAI-compatible endpoint
- 发 `prefill` 或 `decode` 请求
- 记录请求开始/结束时间、响应状态、usage 字段等
- 输出 `prefill_samples.csv` 和 `decode_samples.csv`

### 2.3 逻辑 trace 合成层

文件：

- `src/tools/pd_imitation_trace.py`

职责：

- 聚合 prefill/decode 样本
- 计算平均时间和标准差
- 根据模型参数推导 `KV bytes per token`
- 生成逻辑上的 `pd_imitation_trace.csv`

### 2.4 报告生成层

文件：

- `src/tools/pd_imitation_report.py`

职责：

- 从样本 CSV 和 trace CSV 里做聚合
- 生成基础图
- 生成单轮结果报告
- 在给定第二个结果目录时生成 baseline/offload 对照报告

### 2.5 GPU/PCIe 观测层

文件：

- `src/tools/gpu_metrics_logger.py`
- `src/tools/pd_pcie_offload_analyze.py`

职责：

- 用 NVML 周期采样 GPU、PCIe、部分 CPU 指标
- 把采样结果和请求时间窗对齐
- 输出 PCIe 带宽随时间变化图、窗口统计和文字说明

### 2.6 实验编排层

文件：

- `scripts/run_pd_imitation_full.sh`
- `scripts/run_pd_imitation_offload_only.sh`
- `docs/pd_imitation_runbook.md`

职责：

- 启服务
- 跑 baseline
- 跑 native CPU offloading
- 启停 GPU metrics logger
- 调各个 Python 工具把产物串起来

## 3. 端到端数据流

当前完整链路可以概括成下面这条流程：

```text
long_dialogue_prefix.txt
  -> pd_build_bucket_prompts.py
  -> prefill_prompts.jsonl / decode_prompts.jsonl
  -> pd_collect_openai_samples.py
  -> prefill_samples.csv / decode_samples.csv
  -> pd_imitation_trace.py
  -> pd_imitation_trace.csv + pd_imitation_summary.json
  -> pd_imitation_report.py
  -> aggregate csv + svg + markdown report
```

如果是 offloading 观测链路，则还会并行多一条：

```text
vllm serve
  + gpu_metrics_logger.py
  -> logs/gpu_metrics.csv
  + prefill/decode samples
  -> pd_pcie_offload_analyze.py
  -> pcie_timeline.svg + pcie_timeline_report.md + summary json/csv
```

## 4. 各脚本详细说明

## 4.1 `pd_build_bucket_prompts.py`

路径：

- [pd_build_bucket_prompts.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_bucket_prompts.py)

它的输入是：

- `--model_or_tokenizer`
- `--target_tokens`
- `--samples_per_bucket`
- `--prefix` 或 `--prefix_file`
- `--out`

它的核心逻辑分三步：

1. 用 `transformers.AutoTokenizer` 加载 tokenizer
2. 先生成一大段 seed text
3. 反复尝试切分 token 序列，直到找到一个“编码后 token 数恰好等于目标 bucket”的文本

这里最关键的函数是：

- `find_exact_prompt()`

这个函数不是简单“按字符数截断”，而是：

- 先编码成 token
- 再做 token 级搜索
- 最后 decode 回文本
- 再次 encode 验证长度是否严格匹配

这样做的价值是：

- 采样 bucket 更干净
- 不会因为 tokenizer 的边界效应导致 `target_tokens=8192`，实际却变成 `8187/8201`

当前输出字段有：

- `sample_id`
- `target_tokens`
- `prompt_tokens`
- `prompt_text`

你后来把它扩成支持：

- `--prefix_file`

这一步很关键，因为现在 workload 不再是简单短前缀，而是“更像长对话”的多轮模板。

## 4.2 `pd_collect_openai_samples.py`

路径：

- [pd_collect_openai_samples.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_collect_openai_samples.py)

这是整套流程里最核心的采样器。

它做的事情是：

- 读取上一步的 JSONL prompt 列表
- 组装请求
- 发送到 `vLLM` 的 OpenAI-compatible endpoint
- 记录一次请求的时间和返回状态

它有两种模式：

- `--mode prefill`
- `--mode decode`

### prefill 模式

在 `prefill` 模式下：

- 每个请求只生成 1 token
- `effective_prefill_ms` 直接等于整段 `elapsed_ms`
- `prefill_tps = prompt_tokens / elapsed`

这本质上是用“`max_tokens` 很小”的方式近似只测 `prefill`。

### decode 模式

在 `decode` 模式下：

- prompt 先被构造成指定 context 长度
- 再请求生成 `128/256/512` 等长度
- 最后记录整段 elapsed time

当前的 `decode_ms_per_token` 计算方式是：

- `elapsed_ms / generated_tokens`

这不是纯 kernel 级 decode 时间，而是“长 context 下整段 decode 请求的平均每 token 开销”。这点在报告里也反复强调过。

### 为什么默认推荐 `completion` endpoint

脚本支持：

- `--endpoint completion`
- `--endpoint chat`

但当前 runbook 统一推荐 `completion`，原因是：

- 避免 chat template 在服务端再包一层
- 这样 `prompt_tokens` 和本地构造的 bucket 更一致

### 并发能力

你后来补了：

- `--parallel_requests`

实现方式是：

- `ThreadPoolExecutor`

这一步对 offloading 观测很重要，因为如果完全串行，请求重叠度太低，很难把 CPU-GPU KV 传输打出来。

### 当前输出字段

它会写出：

- `sample_id`
- `mode`
- `prompt_tokens`
- `context_tokens`
- `generated_tokens`
- `submit_ts_unix_ms`
- `finish_ts_unix_ms`
- `elapsed_ms`
- `effective_prefill_ms`
- `decode_ms_per_token`
- `decode_tps`
- `prefill_tps`
- `response_id`
- `usage_prompt_tokens`
- `usage_completion_tokens`
- `http_status`
- `error`

这套字段设计后面被两个地方复用：

- `pd_imitation_trace.py` 读时间和 token 信息
- `pd_pcie_offload_analyze.py` 用 `submit/finish` 时间对齐 GPU metrics

## 4.3 `pd_imitation_trace.py`

路径：

- [pd_imitation_trace.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_trace.py)

这个脚本负责把“采样数据”变成“逻辑 PD trace”。

它的输入有两类：

1. 采样结果
   - `--prefill_csv`
   - `--decode_csv`
2. 模型 KV 参数
   - `--model_config`
   - 或 `--num_layers --num_kv_heads --head_dim --dtype_bytes`

当前 runbook 里直接把 Qwen3-8B 的参数写死了：

- `num_layers=36`
- `num_kv_heads=8`
- `head_dim=128`
- `dtype_bytes=2`

### 它实际做了什么

第一步，聚合 prefill：

- 按 `prompt_tokens` 分桶
- 取 `effective_prefill_ms` 的均值和标准差

第二步，聚合 decode：

- 按 `(context_tokens, generated_tokens)` 分桶
- 取 `elapsed_ms` 的均值和标准差

第三步，算 KV 大小：

```text
KV_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
```

第四步，生成每个逻辑请求的字段：

- `prefill_time_ms`
- `decode_time_ms`
- `prefill_kv_bytes`
- `decode_required_kv_bytes`
- `chunked_prefill_kv_bytes`
- `chunked_decode_kv_bytes`

这里有一个很重要的设计点：

- 这份 trace 不是直接从网络抓出来的
- 它是“真实时间测量 + 理论 KV 计算”的合成结果

所以它适合作为：

- offloading replay 的输入
- case study 里的逻辑负载描述

但不等于：

- 真正 PD 部署时某个链路上的精确抓包

### 为什么还有 chunked 字段

脚本里保留了：

- `chunk_size_tokens`
- `chunked_prefill_kv_bytes`
- `chunked_decode_kv_bytes`

这是因为后续如果你要把逻辑 trace 映射到更接近 cache/offloading 的系统行为，通常不会按任意 token 粒度搬运，而是按 chunk 粒度近似。

## 4.4 `pd_imitation_report.py`

路径：

- [pd_imitation_report.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_report.py)

这个脚本负责把结果整理成你现在看到的图和报告。

它做了四件事：

1. 聚合 `prefill_samples.csv`
2. 聚合 `decode_samples.csv`
3. 生成 SVG 图
4. 写 Markdown 报告

### 设计特点

这个脚本没有依赖 matplotlib，而是自己写了简单的 SVG 生成函数：

- `svg_line_chart()`
- `svg_bar_chart()`

这样做的好处是：

- 环境依赖少
- 在实验机上更容易直接跑

### 单轮报告

默认情况下，它会生成：

- `prefill_aggregate.csv`
- `decode_aggregate.csv`
- `prefill_latency.svg`
- `prefill_throughput.svg`
- `decode_ms_per_token.svg`
- `kv_footprint_gib.svg`
- `pd_imitation_report.md`

其中：

- `aggregate_prefill()` 会按 prompt bucket 聚合
- `aggregate_decode()` 会按 `(context, gen)` 聚合
- `build_report()` 负责把关键结论写成可读的 Markdown

### 对照报告

当传入：

- `--compare_results_dir`

时，它还会额外生成：

- `compare_prefill_latency.svg`
- `compare_decode_g<maxgen>_mspt.svg`
- `pd_imitation_compare_report.md`

这里有一个小设计很实用：

- 对照图不是写死看 `g=256`
- 而是自动选择双方共有的最大 generation bucket

所以在你后来把 decode bucket 扩到 `512` 后，图名自然变成了：

- `compare_decode_g512_mspt.svg`

## 4.5 `gpu_metrics_logger.py`

路径：

- [gpu_metrics_logger.py](/Users/daniel/Documents/code/SkyGDR/src/tools/gpu_metrics_logger.py)

这个脚本本来就是一个 GPU 指标记录器，后来你这条 offloading 观测链路把它变成了更直接的 PCIe 侧采样入口。

### 它采什么

主要来源是 NVML：

- PCIe TX throughput
- PCIe RX throughput
- PCIe link generation / width
- GPU util
- GPU memory util
- GPU clocks
- power

如果环境里有 `psutil`，还会补：

- CPU util
- CPU 内存占用

### 它为什么对 offloading 有用

因为 native CPU KV offloading 最想看的不是只有请求 latency，还包括：

- offload / restore 期间 CPU-GPU 之间到底搬了多少
- 峰值带宽出现在什么时间
- 是 prefill 更重，还是 decode/restore 更重

你后来加进去的关键字段包括：

- `pcie_tx_GiB_s`
- `pcie_rx_GiB_s`
- `pcie_total_GiB_s`
- `pcie_tx_util_pct`
- `pcie_rx_util_pct`
- `pcie_total_util_pct`
- `pcie_tx_cum_GiB`
- `pcie_rx_cum_GiB`
- `pcie_total_cum_GiB`
- `gpu_mem_used_GiB`
- `cpu_util_pct`

这里最有价值的两个口径通常是：

- 瞬时：`pcie_total_GiB_s`
- 累计：`pcie_total_cum_GiB`

### 它和 `pd_imitation` 的关系

严格说，这个脚本不是只为 `pd imitation` 写的，但在当前链路里它承担的是：

- offloading 期间的直接硬件观测

也就是说：

- `pd_collect_openai_samples.py` 告诉我们“请求花了多久”
- `gpu_metrics_logger.py` 告诉我们“同一时间 GPU/PCIe 在发生什么”

## 4.6 `pd_pcie_offload_analyze.py`

路径：

- [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py)

这是把“硬件采样”翻译成“可读实验结果”的脚本。

它的输入是三份 CSV：

- `gpu_metrics.csv`
- `prefill_samples.csv`
- `decode_samples.csv`

### 它的关键设计

它不是简单画整段 GPU metrics，而是先根据请求样本推导三个时间窗：

- `prefill_start -> prefill_end`
- `decode_start -> decode_end`
- `full window`

然后再把 GPU metrics 对齐到这些窗口上。

所以它回答的问题更具体：

- 从第一条请求开始到最后一条 decode 结束，PCIe 总共搬了多少
- prefill 窗口内平均/峰值带宽是多少
- decode 窗口内平均/峰值带宽是多少

### 它输出什么

默认会写：

- `pcie_timeline.svg`
- `pcie_timeline_report.md`
- `pcie_timeline_window.csv`
- `pcie_timeline_summary.json`

其中：

- `build_svg()` 画 3 个 panel
  - 瞬时带宽
  - 累计传输量
  - GPU 显存占用
- `summarize_window()` 计算阶段统计
- `write_summary_md()` 把结论写成 Markdown

如果你现在最关心：

- “offloading 开始到最后 restore 的带宽随时间变化”

那真正最对口的就是这一层，而不是 `pd_imitation_trace.py`。

## 5. 两个 Bash 脚本怎么分工

## 5.1 `run_pd_imitation_full.sh`

路径：

- [run_pd_imitation_full.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_imitation_full.sh)

这是全流程入口。

它做的事情按顺序是：

1. 设环境变量
2. 启 baseline server
3. 启 `gpu_metrics_logger.py`
4. 跑一整套 baseline pipeline
5. 分析 baseline 的 PCIe 指标
6. 停服务
7. 启 offload server
8. 再跑同一套 pipeline
9. 分析 offload 的 PCIe 指标
10. 生成 baseline vs offload 对照报告

这个脚本里最重要的函数有三个：

- `stop_all_vllm()`
- `pipeline_for_run_root()`
- `analyze_metrics_for_run_root()`

### `stop_all_vllm()`

这是后来修过的关键点。

它不是只杀端口监听，而是同时处理：

- `fuser -k PORT/tcp`
- `pkill -f "vllm serve"`
- `pkill -9 -f "VLLM::EngineCore"`

原因是之前只关端口不够，EngineCore 还占着显存，下一轮 offload 起不来。

### `pipeline_for_run_root()`

这个函数把整条 Python 工具链串起来：

- 写长对话模板
- 构造 prefill prompts
- 采 prefill
- 构造 decode prompts
- 采 decode
- 生成逻辑 trace

这里 baseline 和 offload 最大的优点是：

- workload 完全一致
- 只有服务启动参数不同

这保证了对照是干净的。

## 5.2 `run_pd_imitation_offload_only.sh`

路径：

- [run_pd_imitation_offload_only.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_imitation_offload_only.sh)

这个脚本是恢复型入口。

适用场景是：

- baseline 已经跑完
- full 脚本在 offload 段失败了
- 你只想补跑 offload，然后重新生成 compare report

它基本复用了 full 脚本里的 offload 半段，只是把 baseline 部分拿掉了。

## 6. baseline 和 offload 的真正差别在哪里

从代码上看，两轮最核心的区别只在 `vllm serve` 参数。

baseline：

- 不开 KV offloading

offload：

- `--kv-offloading-size "$OFFLOAD_SIZE_GIB"`
- `--kv-offloading-backend native`
- `--disable-hybrid-kv-cache-manager`

其中最后这一项是后来补进去的兼容性修复：

- 否则 vLLM 0.18 上会遇到 `OffloadingConnector does not support HMA`

也正因为 baseline/offload 的 workload 和采样逻辑保持相同，所以：

- `pd_imitation_compare_report.md`
- `pcie_timeline_report.md`

这两类报告都能被解释成“只改变 offloading 配置后的差异”。

## 7. 当前结果目录里各文件怎么来的

以 baseline 目录为例：

- [results/pd_imitation_qwen3_8b_instruct](/Users/daniel/Documents/code/SkyGDR/results/pd_imitation_qwen3_8b_instruct)

里面的主要结构是：

- `data/`
  - prompt 输入和原始样本
- `logs/`
  - vLLM 日志和 GPU metrics 原始日志
- `summary/`
  - trace、summary json、markdown 报告、PCIe 报告
- `fig/`
  - SVG 图

其中最关键的对应关系是：

- `data/prefill_prompts.jsonl`
  - 来自 `pd_build_bucket_prompts.py`
- `data/prefill_samples.csv`
  - 来自 `pd_collect_openai_samples.py --mode prefill`
- `data/decode_samples.csv`
  - 来自 `pd_collect_openai_samples.py --mode decode`
- `summary/pd_imitation_trace.csv`
  - 来自 `pd_imitation_trace.py`
- `summary/pd_imitation_report.md`
  - 来自 `pd_imitation_report.py`
- `logs/gpu_metrics.csv`
  - 来自 `gpu_metrics_logger.py`
- `summary/pcie_timeline_report.md`
  - 来自 `pd_pcie_offload_analyze.py`

## 8. 现在这套代码最值得注意的设计取舍

### 8.1 它强调“可运行”而不是“最纯理论”

这套代码的优点是：

- 你可以直接在实验机上跑
- 不需要先搭完整 PD 系统
- 也不需要先引入 LMCache

代价是：

- `prefill-only` 和 `decode-only` 都是近似口径
- `pd_imitation_trace.csv` 是逻辑 trace，不是抓包

### 8.2 它把“时间测量”和“硬件观测”分开了

这其实是个好设计。

因为：

- 时间测量脚本尽量简单稳定
- 硬件观测脚本可以独立增强

所以你后来要补 PCIe 时间序列，不需要推翻前面的 trace 逻辑，只要把：

- `gpu_metrics_logger.py`
- `pd_pcie_offload_analyze.py`

接进来就行。

### 8.3 它已经开始向 case study 过渡

从当前代码结构看，这套 `pd imitation` 已经不只是“采 timing”，而是在为后续 case study 准备三个东西：

- 请求长度分布
- 逻辑 KV 大小映射
- offloading 期间的 GPU/PCIe 观测

所以它其实已经是 case study 的前置数据层了。

## 9. 如果后面要改，优先改哪里

如果你后面继续扩展，我建议按下面这个分层去动。

### 改 workload 形状

优先改：

- [pd_build_bucket_prompts.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_build_bucket_prompts.py)
- [run_pd_imitation_full.sh](/Users/daniel/Documents/code/SkyGDR/scripts/run_pd_imitation_full.sh)

例如：

- 更长对话模板
- 更大的 token buckets
- 不同风格的 prefix

### 改采样方式

优先改：

- [pd_collect_openai_samples.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_collect_openai_samples.py)

例如：

- 更复杂的并发策略
- request pacing
- 更详细的响应字段记录

### 改逻辑 trace 定义

优先改：

- [pd_imitation_trace.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_imitation_trace.py)

例如：

- 不再只取均值，改成保留分位数
- 引入更真实的 chunk 规则
- 引入 request arrival pattern

### 改硬件观测

优先改：

- [gpu_metrics_logger.py](/Users/daniel/Documents/code/SkyGDR/src/tools/gpu_metrics_logger.py)
- [pd_pcie_offload_analyze.py](/Users/daniel/Documents/code/SkyGDR/src/tools/pd_pcie_offload_analyze.py)

例如：

- 采样更快
- 增加 CPU 内存带宽或 NUMA 侧指标
- 单独标注 offload / restore 的局部尖峰

## 10. 一句话总结

如果把这套代码压成一句话，它做的是：

- 用 `vLLM` 构造一套单 GPU 的 `PD imitation` 采样流程，
- 再把请求时间、逻辑 KV 大小和 GPU/PCIe 观测串起来，
- 形成一套可对照 baseline/native offload 的实验与报告链路。

如果你后面还想继续收紧，我建议下一步可以再补一份更“面向维护”的文档，专门列：

- 每个 CSV 的字段定义
- 每份报告是由哪条命令生成的
- 哪些地方是当前实验假设，哪些地方是硬编码
