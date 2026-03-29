# Contention 2 v2 当前结论与文档入口

这份文档是 `contention2_paper_v2` 的当前结果摘要。

如果你要重跑实验，主入口看：

- `contention2_experiment_readme.md`

如果你要看 A1 消融，单独看：

- `contention2_A1_host_vs_gdr_guide.md`

如果你看到旧的 `contention2_fourfig_experiment_report.md`，请把它当成早期 `low` 档数据归档，而不是当前 paper 主结果。

## 1. 数据范围

本版覆盖两部分：

- Part A：`10MB RDMA` 对 host-device memcpy 的影响
- Part B：`256B ~ 100MB` 的 RDMA throughput sweep

当前结论对应的标准产物路径仍然是：

- `paper/figures/partA_memcpy_impact_bar.pdf`
- `paper/figures/partB_fourcases_throughput.pdf`
- `results/contention2_paper_v2/raw/*.csv`
- `results/contention2_paper_v2/summary/partB_size_impact.csv`
- `results/contention2_paper_v2/fig/*.png`

补充说明：

- 当前仓库快照里没有把 `results/` 和 `paper/` 一起带上来
- 这些路径表示完整实验环境中的标准输出位置

## 2. Part A 结论

根据 `partA_memcpy_impact_bar.pdf`：

- H2D isolated: `25.82 GiB/s`
- H2D + write: `5.23 GiB/s`
- D2H isolated: `25.12 GiB/s`
- D2H + read: `11.57 GiB/s`

对应降幅：

- H2D：`79.74%`
- D2H：`53.94%`

结论：

1. 大消息 RDMA 会显著侵占 host-device copy 带宽。
2. 在当前平台和参数下，`H2D + write` 的冲突更重。
3. Part A 已经足够支撑“memcpy 带宽会被大流量 RDMA 明显压制”这个主结论。

## 3. Part B 结论

### 3.1 全局统计

- size 点数：`20`
- size 范围：`256B` 到 `100MB`

write（`write_h2d / write_none`）：

- 平均 ratio：`0.9936`
- 最低 ratio：`0.9368`，出现在 `2KB`
- `>=128KB` 基本恢复到 `1.0`

read（`read_d2h / read_none`）：

- `>=8KB` 区间吞吐损失稳定在 `~41.5%`

### 3.2 关键点位

- `256B`: write ratio `0.9745`
- `2KB`: write ratio `0.9368`
- `16KB`: read degradation `41.5%`
- `1MB`: read degradation `41.5%`
- `100MB`: read degradation `41.5%`

结论：

1. `write + H2D` 只在小包区间出现轻微退化。
2. `read + D2H` 在较大消息区间出现持续而稳定的吞吐损失。
3. 当前平台上呈现出明显方向不对称：D2H 背景对 RDMA read 的压制远强于 H2D 背景对 RDMA write。

## 4. 写 paper 时可直接用的句子

- Part A：在 `10MB RDMA` 条件下，背景 H2D 带宽从 `25.82` 降至 `5.23 GiB/s`，背景 D2H 带宽从 `25.12` 降至 `11.57 GiB/s`，说明大流量 RDMA 会显著侵占 host-device copy 带宽。
- Part B：在 `256B ~ 100MB` 范围内，`write + H2D` 对 RDMA write 吞吐影响很小，而 `read + D2H` 在 `>=8KB` 区间出现稳定的 `~41.5%` 吞吐损失，表现出明显的方向不对称竞争。

## 5. 当前文档关系

- `contention2_experiment_readme.md`
  - 当前执行手册
  - 包含 v2 / read hotfix / 分步命令
- `contention2_paper_v2_report.md`
  - 当前结果摘要
  - 适合 paper 写作和快速回顾
- `contention2_fourfig_experiment_report.md`
  - 旧版 `results/contention2_paper` low 档归档
  - 不再作为当前 paper 主结论依据
