# Agent Context

这份文件给后续协作 agent 使用，目的是快速建立正确上下文，避免把本地开发机、文档快照和真实实验服务器混为一谈。

## 1. 当前访问边界

- 当前 agent 能直接访问的是本地电脑上的仓库副本：
  - `/Users/daniel/Documents/code/SkyGDR`
- 当前 agent 不能直接访问真实实验服务器，除非用户额外提供远程登录方式。
- 文档中出现的服务器路径、结果目录和 paper 输出目录，默认表示实验环境中的标准路径，不等于当前本地机器实际存在这些目录。

## 2. 本地开发机信息

- 当前工作目录：`/Users/daniel/Documents/code/SkyGDR`
- 当前 shell：`zsh`
- 当前本地环境可以安装额外工具。
- 用户明确允许：如果缺少本地分析工具，可以直接用 `brew` 安装。
- 已确认本地有 `brew`：
  - `/opt/homebrew/bin/brew`

## 3. 实验服务器信息

以下信息来自 `docs/` 中的实验文档，默认视为“最近一次记录的实验环境配置”，后续真正执行前仍应和用户确认是否有变动。

### 3.1 GPU 机器

- 角色：
  - 运行 `gpu_server`
  - 运行 `gpu_pcie_memcpy`
  - 运行 `gpu_metrics_logger.py`
  - 在 case study 中运行 `pcie_case_study_controller.py`
- 文档中的 IP：
  - `10.10.10.11`

### 3.2 CPU 机器

- 角色：
  - 运行 `cpu_client`
  - 在部分实验中运行 `rdma_msgsize_sweep.py`
- 文档中的 IP：
  - `10.10.10.10`
- 文档中的用户：
  - `enine`

### 3.3 双机共享配置

- 仓库根目录：
  - `/home/enine/danyang/SkyGDR`
- IB 设备：
  - `mlx5_0`
- RDMA 服务端 TCP 端口：
  - `33333`
- case study control port：
  - `44444`
- `PORT=1`
- `GID_IDX=3`
- `MTU=1024`

## 4. 机器间协作约束

- 当前实验方案默认：
  - 没有 passwordless SSH
  - 没有 shared filesystem
- 因此：
  - 两台机器上的实验往往需要手动双机执行
  - CPU 机器生成的时间序列 CSV 需要手动拷回 GPU 机器
  - case study 的跨机控制依赖 `cpu_client` 暴露的 TCP control port，而不是 SSH

## 5. 当前代码主线

这个仓库当前主要有两条相关主线：

### 5.1 `contention2`

- 用于验证 `PCIe fabric contention` 的主要实验线。
- 当前文档主入口：
  - `docs/contention2_experiment_readme.md`
  - `docs/contention2_paper_v2_report.md`
- 当前 `v2` 结果的核心结论：
  - Part A：大消息 RDMA 会显著压低 H2D / D2H memcpy 带宽
  - Part B：存在明显方向不对称，尤其是 `read + D2H`

### 5.2 `PCIe case study`

- 当前是一个最小闭环原型，不是完整生产调度器。
- 当前文档主入口：
  - `docs/pcie_case_study_report.md`
- 当前 prototype 的结构：
  - foreground：有限 `H2D restore window`
  - background：长时间运行的 GPUDirect RDMA `write`
  - signal：`restore progress` + `GPU-side PCIe RX`
  - control knob：RDMA pacing

## 6. 当前 case study 的真实实现边界

以下内容已经在代码里落地：

- `src/cpu_client.cc`
  - 支持 TCP control plane
  - 支持 `HIGH` / `LOW1..LOW4` / `STOP` / `SLEEP <us>`
- `src/gpu_be_pcie_memcpy_task.cu`
  - 支持有限 restore window
  - 支持 progress CSV 输出
- `src/tools/pcie_case_study_controller.py`
  - 在 GPU 机器本地读 signal
  - 远程控制 CPU 侧 background pacing
- `src/tools/pcie_case_study_analyze.py`
  - 合并 controller timeline 与 background TS

当前 controller 的重要事实：

- `LOW` 等价于 `LOW1`
- 当前状态机是“只升级、不在 restore window 内自动退回”
- restore 完成后会回 `HIGH`，保留短 tail，然后发 `STOP`

## 7. 论文对应关系

用户明确说明：

- `docs/APNet26___FabricContention.pdf` 是当前代码对应的 paper
- 但它目前只对应 `pcie contention` 这条主线
- 后续准备在这个基础上继续扩展 `case study`

从 PDF 元信息与首页摘要可见：

- 标题：`Unlocking Software-defined GPU Fabric Scheduling in the LLM Era`
- 文件路径：`docs/APNet26___FabricContention.pdf`
- paper 的整体叙事同时覆盖：
  - `HBM contention`
  - `PCIe contention`
- paper 中还明确提到一个 `case study` 方向，用来展示如何把任务级目标翻译成硬件层调度策略

这意味着“paper 范围”和“当前代码覆盖范围”不是完全重合的：

- paper 视角更大，讲的是软件定义 GPU fabric scheduling 的整体愿景
- 当前仓库里最扎实、最直接落地的是 `PCIe contention`
- 当前 `pcie case study` 更像是在往 paper 中的 runtime / scheduler 叙事继续靠拢，但还处于扩展阶段

## 8. 对后续 agent 的工作建议

后续 agent 在继续工作时，默认应先遵守这些约束：

- 不要假设自己能登录真实 GPU/CPU 实验服务器。
- 如果任务需要真实实验结果、远程进程状态或服务器文件，应先让用户确认或手动提供。
- 如果任务只涉及本地代码、文档、脚本整理，可以直接在当前仓库完成。
- 如果缺少本地分析工具，可以优先考虑：
  - `brew install <tool>`

## 9. 当前已知扩展点

- `case study` 现在已经有最小闭环原型，但还不是完整 runtime。
- 当前新增了一条更贴合硬件现实的 `PD imitation` 路线：
  - phase 1 不尝试跑真实 PD 部署
  - phase 1 只采 `prefill-only` 与 `decode-only`
  - phase 1 通过模型结构计算 `KV bytes per token`
  - phase 1 最终生成逻辑上的 `pd_imitation_trace.csv`
  - phase 1 当前主线是 `vLLM native / single-GPU / prefill-decode timing extraction`
  - LMCache 不再是 phase 1 必需项，只保留给 phase 2 的 remote KV / replay 扩展
- 对应文档：
  - `docs/pd_imitation_runbook.md`
  - 当前文档已进一步收敛为 `Qwen3-8B-Instruct` 的实操版
- 对应脚本：
  - `src/tools/pd_build_bucket_prompts.py`
  - `src/tools/pd_collect_openai_samples.py`
  - `src/tools/pd_imitation_trace.py`
- 在这条 phase-1 路线中，CPU-only server 不参与采集，只保留给后续 replay 或真实跨机流量实验。
- 如果后续要把 case study 更正式纳入 paper，建议持续保持以下口径一致：
  - foreground 是 `critical restore window`
  - signal 是 `restore progress` + `GPU-side PCIe pressure`
  - knob 是 `RDMA pacing`

## 10. 维护建议

这份 `agent.md` 应优先记录：

- 机器信息
- 访问边界
- 文档主入口
- 论文与代码的对应关系

不要在这里堆太多实验命令细节；命令细节应继续放在 `docs/` 里的各个实验说明中。
