# External Prefix-Cache Imitation 结果报告

这份报告总结 [pd_external_prefix_imitation_qwen3_8b](/Users/daniel/Documents/code/SkyGDR/results/pd_external_prefix_imitation_qwen3_8b) 这一轮实验结果，并回答一个核心问题：

- 当前这套 `single-GPU + vLLM + LMCache` 工作流，是否已经成功 imitation 了
  - “长 session / 高 prefix reuse”
  - “prefill 从 external/shared prefix cache 拉取大量历史 KV”
  - “reuse 阶段以 RX/H2D 为主，seed 阶段以 TX/D2H 为主”

结论先写在前面：

- **方向上已经明显符合预期，而且比上一版 `native offloading` 更对口。**
- 但当前请求级 `LMCache` counter 差分归因还有 bug，所以：
  - **阶段级 PCIe 结论可信**
  - **vLLM / LMCache 日志里的 per-turn retrieval 结论可信**
  - **`trajectory_samples.csv` 里 per-request `lmcache_remote_read_*` 不能直接当最终真值**

## 1. 目标回顾

这轮实验真正想 imitate 的语义是：

- 第 1 轮：把长前缀写入 external/shared KV cache
- 后续每一轮：绝大多数历史 token 直接命中 external prefix cache
- prefill 不再重算整段历史 prefix
- 而是把历史 KV 从外部存储读回 GPU，再只计算新增 suffix

因此我们真正关心的是：

1. `seed` 阶段是不是以 `TX` 为主
2. `reuse` 阶段是不是以 `RX` 为主
3. `reuse` 阶段是否真的发生了大规模 external cache hit
4. `reuse` 请求的端到端时间是否明显低于首轮长 prompt

## 2. 实验配置

本轮结果对应的主配置为：

- 模型：`Qwen/Qwen3-8B`
- 服务名：`Qwen3-8B-Instruct`
- `vLLM`：
  - 关闭 GPU prefix caching：`--no-enable-prefix-caching`
  - 启用 `LMCacheConnectorV1`
  - `max-num-seqs=1`
  - `gpu-memory-utilization=0.85`
- `LMCache`：
  - `chunk_size=256`
  - `save_decode_cache=false`
  - `save_unfull_chunk=false`
  - `local_cpu=false`
  - `remote_url=mock://256/?peeking_latency=1&read_throughput=40&write_throughput=8`

workload 配置为：

- `seed_prompt_tokens = 24832`
- `append_tokens = 256,256,256,256,256`
- `num_turns = 6`
- `decode_tokens = 16`

所以整个 session 是：

- `turn_000_seed`
- `turn_001_reuse`
- `turn_002_reuse`
- `turn_003_reuse`
- `turn_004_reuse`
- `turn_005_reuse`

而且每个 `reuse` turn 都满足：

- 新 prompt 以前一轮为严格前缀
- 只新增 `256` token
- 文本级复用率约 `99%`

## 3. 结果是否符合预期

### 3.1 结论

**是，方向上已经符合预期。**

这里“符合预期”不是说所有报表字段都已经完美，而是说：

- 系统行为已经清楚地长成了
  - `seed -> 写入 external cache`
  - `reuse -> 从 external cache 读取历史 KV`
- 并且这条行为已经在：
  - `vLLM / LMCache` 日志
  - PCIe `TX/RX` 相位统计
  - 端到端 latency 变化

三条证据链上同时出现了。

### 3.2 为什么说它已经比上一版更对口

上一版 `native offloading` 的问题是：

- `TX` 很高
- `RX` 很弱
- 更像 “evict-heavy”
- 不像 “prefill 从 external storage 读回历史 KV”

而这一版结果已经变成：

- `seed` 阶段：
  - 以 `TX` 为主
  - 几乎没有 `RX`
- `reuse` 阶段：
  - 以 `RX` 为主
  - `LMCache` 日志明确记录了历史前缀被 retrieve

这正是我们想要的形状。

## 4. 最可信的证据

## 4.1 阶段级 PCIe 统计已经很像目标形状

来自 [pcie_timeline_report.md](/Users/daniel/Documents/code/SkyGDR/results/pd_external_prefix_imitation_qwen3_8b/summary/pcie_timeline_report.md)：

- `seed`
  - `duration = 4.621 s`
  - `avg TX = 1.091 GiB/s`
  - `avg RX = 0.008 GiB/s`
  - `peak TX = 15.452 GiB/s`
  - `peak RX = 0.049 GiB/s`
- `reuse`
  - `duration = 9.566 s`
  - `avg TX = 0.262 GiB/s`
  - `avg RX = 2.174 GiB/s`
  - `peak TX = 4.123 GiB/s`
  - `peak RX = 35.759 GiB/s`

这组数据最重要的意义是：

- `seed` 明显是 **TX-dominant**
- `reuse` 明显是 **RX-dominant**

这已经和目标语义高度一致：

- 首轮长前缀先写入外部 cache
- 后续高复用 turn 从外部 cache 读回历史 KV

## 4.2 vLLM / LMCache 日志明确记录了所有 reuse turn 的 external hit

来自 [vllm_external_prefix.log](/Users/daniel/Documents/code/SkyGDR/results/pd_external_prefix_imitation_qwen3_8b/logs/vllm_external_prefix.log)：

- `turn_001_reuse`
  - `LMCache hit tokens: 24832`
  - `need to load: 24832`
  - `Retrieved ... size: 3.4102 GB`
- `turn_002_reuse`
  - `LMCache hit tokens: 25088`
  - `need to load: 25088`
  - `Retrieved ... size: 3.4453 GB`
- `turn_003_reuse`
  - `LMCache hit tokens: 25344`
  - `need to load: 25344`
  - `Retrieved ... size: 3.4805 GB`
- `turn_004_reuse`
  - `LMCache hit tokens: 25600`
  - `need to load: 25600`
  - `Retrieved ... size: 3.5156 GB`
- `turn_005_reuse`
  - `LMCache hit tokens: 25856`
  - `need to load: 25856`
  - `Retrieved ... size: 3.5508 GB`

这组日志非常关键，因为它直接回答了最核心的问题：

- **是的，reuse turn 的大部分 prefix KV 确实是从 external cache 读回来的。**

而且这不是偶发现象，而是：

- 五个 reuse turn 全都发生了

把这五次 retrieval 加总，大约是：

- `3.4102 + 3.4453 + 3.4805 + 3.5156 + 3.5508`
- 约 `17.40 GB`

这已经足够支撑：

- “prefill loads large volumes of KV-Cache from external storage”

这句话在我们当前这条 imitation 流程里已经成立。

## 4.3 端到端时间也出现了“seed 重、reuse 轻”的变化

来自 [trajectory_samples.csv](/Users/daniel/Documents/code/SkyGDR/results/pd_external_prefix_imitation_qwen3_8b/data/trajectory_samples.csv)：

- `turn_000_seed`
  - `prompt_tokens = 24832`
  - `elapsed_ms = 3413`
- `turn_001_reuse`
  - `prompt_tokens = 25088`
  - `elapsed_ms = 677`
- `turn_002_reuse`
  - `elapsed_ms = 690`
- `turn_003_reuse`
  - `elapsed_ms = 697`
- `turn_004_reuse`
  - `elapsed_ms = 700`
- `turn_005_reuse`
  - `elapsed_ms = 699`

这个趋势也很有说服力：

- 首轮长 prompt 花了约 `3.4s`
- 后续虽然 prompt 长度继续增加到 `26K+`
- 但每轮只要约 `0.68-0.70s`

这和 “整段重算长 prefix” 的行为明显不一样，
更像：

- 大部分历史 prefix 已命中外部 cache
- 当前只需把历史 KV 读回，再计算新增 `256` token 的 suffix

## 5. 图像解读

## 5.1 总图

![PCIe Total Timeline](../results/pd_external_prefix_imitation_qwen3_8b/summary/pcie_timeline.svg)

这张图适合看：

- 整个实验窗口内是否真的分成两个明显相位
  - `seed`
  - `reuse`

可以看到：

- `seed` 段主要是写出
- `reuse` 段主要是读回

## 5.2 TX 图

![PCIe TX Timeline](../results/pd_external_prefix_imitation_qwen3_8b/summary/pcie_tx_timeline.svg)

这张图最值得看的点是：

- `seed` 阶段有非常清楚的 TX 高峰
- `peak TX = 15.452 GiB/s`

这和首轮长前缀被存入 external cache 的行为是一致的。

## 5.3 RX 图

![PCIe RX Timeline](../results/pd_external_prefix_imitation_qwen3_8b/summary/pcie_rx_timeline.svg)

这张图是当前最关键的一张。

这里最重要的不是单个点有多高，而是：

- `reuse` 整段都明显比 `seed` 更偏向 RX
- `seed peak RX` 几乎为零
- `reuse peak RX` 被明显打高

这正是我们之前一直想看到、但在 `native offloading` 版本里没打出来的东西。

## 6. 当前结果里最值得警惕的地方

虽然方向已经对了，但当前还有一个**明显的统计归因 bug**：

- `trajectory_samples.csv` 里的 `lmcache_requested_tokens`
- `lmcache_hit_tokens`
- `lmcache_remote_read_GiB`
- `lmcache_remote_write_GiB`

只在 `turn_003_reuse` 上出现了非零值，
而日志明明显示：

- `turn_001` 到 `turn_005` 都发生了 retrieval

也就是说：

- 当前 request-level scrape-delta 逻辑**漏记了前后几轮**

所以：

### 可信的

- `vLLM / LMCache` 日志里的 per-turn retrieval
- 阶段级 PCIe 统计
- seed vs reuse 的 latency 变化

### 暂时不能直接当最终真值的

- `trajectory_samples.csv` 中每个请求的 `lmcache_remote_read_GiB`
- `request_pcie_summary.csv` 中依赖这些字段生成的 request-level LMCache 汇总

另外还有一个较小但需要说明的点：

- `reuse peak RX = 35.759 GiB/s`
- 当前 logger 里的 `pcie_link_ref_GiB_s = 29.340 GiB/s`

也就是说，`NVML` 采到的瞬时 `RX` 峰值超过了单向理论链路参考。

这通常不应该被解释成“链路真的超过物理上限”，更合理的解释是：

- `nvmlDeviceGetPcieThroughput()` 本身是 moving-average 风格采样
- 我们当前对 `KB/s -> GiB/s` 的换算是近似值
- 所以**绝对峰值**更适合拿来看：
  - 哪个相位更强
  - RX/TX 是否分离

而不是直接当成严格物理带宽上限

因此，这轮图最可信的用途是：

- 看 `seed` vs `reuse` 的形状差异
- 看 `RX` 是否明显抬高

而不是拿峰值数本身去做过细的链路饱和度结论

## 7. 这意味着什么

如果只问一句：

- **“新的 external prefix-cache imitation 版本是不是已经比旧版更符合目标？”**

答案是：

- **是，已经明显更符合。**

如果再问得严格一点：

- **“能不能现在就把 request-level LMCache counter 当成完全正确的数据源写进最终 paper 图表？”**

答案是：

- **还不行。**

因为当前 scrape-delta 归因还有 bug，
需要修一轮之后才能把 request-level 统计完全坐实。

## 8. 当前阶段最稳的结论

这轮实验已经足够支持下面这些结论：

1. 单 GPU 条件下，`vLLM + LMCache` 可以比 `native offloading` 更准确地 imitation
   - external/shared prefix-cache
   - prefill-side historical KV loading

2. 当前 workload 已经成功构造出：
   - `~99%` 的文本级 prefix reuse
   - 首轮写入 external cache
   - 后续多轮从 external cache retrieve 历史 KV

3. 系统级流量形状已经变成：
   - `seed`: TX-dominant
   - `reuse`: RX-dominant

4. 端到端时间也出现了预期变化：
   - `seed` 明显更重
   - `reuse` 明显更轻

## 9. 下一步建议

最自然的下一步不是重做 workload，而是：

1. 修 `pd_run_external_prefix_workload.py` 的 request-level LMCache counter 归因
2. 重跑这一组 workload
3. 再生成一版“无统计归因歧义”的最终报告

如果这一步修好，后面我们就可以更稳地回答：

- 每个 reuse turn 到底读回了多少 GiB
- 每个 turn 的 `remote read` 和 `PCIe RX` 对应关系是否线性
- 这条 imitation 路线能否直接作为后续 case study 的输入分布来源
