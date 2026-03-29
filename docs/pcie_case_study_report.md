# PCIe Case Study 总览、实验说明与结果

这份文档是当前 `PCIe case study` 的主文档，已经合并了原来的：

- `pcie_case_study_report.md`
- `pcie_case_study_experiment_readme.md`
- `pcie_case_study_results_report.md`

目标是只保留一份长期可维护的说明，直接对应现在仓库里的代码实现。

## 1. 文档范围

这个 case study 对应一个最小可运行的闭环原型：

- foreground：有限大小的 `H2D restore window`
- background：长时间运行的 GPUDirect RDMA `write`
- controller：在 GPU 机器本地读信号，并通过 `cpu_client` 暴露的 TCP control port 远程调 background pacing

闭环链路是：

`restore progress + PCIe RX pressure -> contention inference -> RDMA pacing -> restore behavior changes`

这份文档覆盖四件事：

1. 当前代码到底实现了什么
2. 怎么跑实验
3. 会产出哪些文件
4. 当前结果快照能支持哪些结论

## 2. 代码对应关系

### 2.1 `src/cpu_client.cc`

`cpu_client` 现在除了原来的 RDMA 压测功能，还带了一个轻量运行时控制面。

环境变量：

- `SKYGDR_CTRL_PORT`
- `SKYGDR_CTRL_BIND_IP`
- `SKYGDR_CTRL_HIGH_SLEEP_US`
- `SKYGDR_CTRL_LOW_SLEEP_US`
- `SKYGDR_CTRL_LOW2_SLEEP_US`
- `SKYGDR_CTRL_LOW3_SLEEP_US`
- `SKYGDR_CTRL_LOW4_SLEEP_US`

支持的控制命令：

- `STATUS`
- `HIGH`
- `LOW` / `LOW1`
- `LOW2`
- `LOW3`
- `LOW4`
- `STOP`
- `SLEEP <us>`

控制方式是动态修改 `pace_sleep_us`，让 RDMA 循环在成功发出一轮 work requests 之后休眠指定时间，从而降低注入速率。

CPU 侧时间序列 CSV 现在额外带：

- `control_mode`
- `pace_sleep_us`
- `control_commands`

这意味着 controller 不需要 SSH，也不需要共享文件系统，只要能连到 CPU 机器上的一个 TCP 端口就能切换背景负载档位。

### 2.2 `src/gpu_be_pcie_memcpy_task.cu`

`gpu_pcie_memcpy` 现在同时支持两种模式：

- 开环压力模式：`--total_bytes=0`
- 有限 restore window 模式：`--total_bytes>0`

为 case study 新增/强化的关键参数：

- `--total_bytes`
- `--max_outstanding_bytes`
- `--progress_ms`
- `--progress_smooth_windows`
- `--progress_out`

`--progress_out` 会输出 restore 进度 CSV，核心字段包括：

- `ts_unix_ms`
- `elapsed_ms`
- `issued_bytes`
- `completed_bytes`
- `remaining_bytes`
- `inst_bw_gib_s`
- `smooth_bw_gib_s`
- `avg_bw_gib_s`
- `done`

这里的 `smooth_bw_gib_s` 已经在程序里做了滑动窗口平滑，controller 也优先用它做 guard 带宽判断。

### 2.3 `src/tools/cpu_client_control.py`

这是一个手动调试小工具，用来直连 CPU 侧 control port：

- 连通性检查：`STATUS`
- 手动切换：`HIGH` / `LOW` / `LOW2` / `LOW3` / `LOW4`
- 手动停止：`STOP`

### 2.4 `src/tools/pcie_case_study_controller.py`

这是 GPU 机器本地运行的闭环控制器，职责是：

- 确认 CPU 侧 control port 可达
- 先把 background RDMA 置为 `HIGH`
- 启动 `gpu_metrics_logger.py`
- 启动有限 `H2D restore`
- 周期性读取：
  - restore progress CSV
  - GPU metrics CSV
- 根据阈值策略发控制命令
- 输出 controller 时间线 CSV

当前真正生效的判定信号：

- `restore_guard_bw_gib_s`
  - 优先取 `smooth_bw_gib_s`
  - 如果没有，则回退到 `inst_bw_gib_s`
- `restore_remaining_bytes`
- `pcie_rx_util_pct`

当前真正生效的控制策略是：

- 初始强制 `HIGH`
- 若 `restore_guard_bw_gib_s < baseline * enter_ratio` 且 `pcie_rx_util_pct >= rx_threshold_pct`
  - 进入保护
  - 先切 `LOW1`
  - 后续如果仍然持续 bad window，就继续升级到 `LOW2`、`LOW3`、`LOW4`
- restore 完成后
  - 立即切回 `HIGH`
  - 保留一个短暂 `HIGH tail`
  - 最后发 `STOP`

这里有两个实现层面的重要细节，和旧文档相比需要明确：

1. 当前 controller 是“只升级、不在窗口内回退”的状态机。
   - `--exit_ratio` 和 `--exit_windows` 目前会写进 CSV / 传进参数
   - 但当前代码没有在 restore window 中基于这两个参数做降级或退出保护
2. `LOW` 在控制面上等价于 `LOW1`。

也就是说，当前版本不是一个会反复进出保护态的复杂 scheduler，而是一个更容易解释的“发现冲突后逐级加重限速，直到 restore 结束”的闭环。

### 2.5 `src/tools/pcie_case_study_analyze.py`

这个脚本负责把 GPU 侧 controller 时间线与 CPU 侧 RDMA 时间序列合并，并导出：

- merged timeline CSV
- 调试用 PNG
- 可选的 paper 风格 PDF

它还能根据 controller 和 background TS 中的 mode transition 自动估计时钟偏移，用来对齐双机日志。

## 3. 机器角色

典型部署和旧文档保持一致：

- GPU 机器
  - `gpu_server`
  - `gpu_pcie_memcpy`
  - `gpu_metrics_logger.py`
  - `pcie_case_study_controller.py`
- CPU 机器
  - `cpu_client`

该方案默认：

- 不依赖 passwordless SSH
- 不依赖 shared filesystem

## 4. 推荐实验流程

下面保留的是当前最需要的最小流程。路径依然沿用实验环境里的标准写法；当前仓库快照中未包含 `results/` 目录本身。

### 4.1 统一参数

两台机器建议先统一：

```bash
ROOT=/home/enine/danyang/SkyGDR
RESULT_ROOT=$ROOT/results/pcie_case_study

SERVER_IP=10.10.10.11
CPU_IP=10.10.10.10
CPU_USER=enine

TCP_PORT=33333
CTRL_PORT=44444
IB_DEV=mlx5_0
PORT=1
GID_IDX=3
MTU=1024

RESTORE_TOTAL=64G
RESTORE_MAX_OUTSTANDING=4G
RESTORE_CHUNK_MB=128
RESTORE_STREAMS=8
RESTORE_BATCH=1
RESTORE_INFLIGHT=4
RESTORE_PROGRESS_MS=100
RESTORE_PROGRESS_SMOOTH_WINDOWS=5

GPU_METRICS_INTERVAL_MS=100
CTRL_POLL_MS=100
CTRL_ENTER_WINDOWS=1

BG_ITERS=1000000000
BG_MSG=10485760
BG_QD=1
BG_QPS=1
BG_SPAN=1G
BG_PATTERN=random
BG_ALIGN=256
BG_TS_MS=100

CTRL_HIGH_SLEEP_US=0
CTRL_LOW_SLEEP_US=800
CTRL_LOW2_SLEEP_US=1000
CTRL_LOW3_SLEEP_US=1200
CTRL_LOW4_SLEEP_US=1400
```

建目录：

```bash
mkdir -p $RESULT_ROOT/{baseline,gpu,cpu,merged,fig}
```

编译：

```bash
cd $ROOT/src
make
```

### 4.2 启动 GPU 侧常驻服务

GPU 机器：

```bash
cd $ROOT/src
./bin/gpu_server $IB_DEV 1G $TCP_PORT $PORT $GID_IDX $MTU
```

### 4.3 先跑 baseline restore

这一步的目的只有一个：拿到 `baseline_restore_gib_s`。

GPU 机器：

```bash
cd $ROOT/src
./bin/gpu_pcie_memcpy \
  --dir=h2d \
  --seconds=0 \
  --total_bytes=$RESTORE_TOTAL \
  --chunk_mb=$RESTORE_CHUNK_MB \
  --streams=$RESTORE_STREAMS \
  --batch=$RESTORE_BATCH \
  --inflight=$RESTORE_INFLIGHT \
  --pinned=1 \
  --report_ms=$RESTORE_PROGRESS_MS \
  --progress_ms=$RESTORE_PROGRESS_MS \
  --progress_smooth_windows=$RESTORE_PROGRESS_SMOOTH_WINDOWS \
  --max_outstanding_bytes=$RESTORE_MAX_OUTSTANDING \
  --progress_out=$RESULT_ROOT/baseline/baseline_restore_progress.csv \
  | tee $RESULT_ROOT/baseline/baseline_restore.log
```

然后读最后一行的 `avg_bw_gib_s`：

```bash
uv run python3 - <<'PY'
import csv
rows = list(csv.DictReader(open("/home/enine/danyang/SkyGDR/results/pcie_case_study/baseline/baseline_restore_progress.csv")))
print(rows[-1]["avg_bw_gib_s"])
PY
```

### 4.4 启动 CPU 侧 background RDMA writer

CPU 机器：

```bash
TAG=controlled_r1

cd $ROOT/src
export SKYGDR_CTRL_PORT=$CTRL_PORT
export SKYGDR_CTRL_HIGH_SLEEP_US=$CTRL_HIGH_SLEEP_US
export SKYGDR_CTRL_LOW_SLEEP_US=$CTRL_LOW_SLEEP_US
export SKYGDR_CTRL_LOW2_SLEEP_US=$CTRL_LOW2_SLEEP_US
export SKYGDR_CTRL_LOW3_SLEEP_US=$CTRL_LOW3_SLEEP_US
export SKYGDR_CTRL_LOW4_SLEEP_US=$CTRL_LOW4_SLEEP_US
export SKYGDR_CTRL_BIND_IP=0.0.0.0

./cpu_client \
  $SERVER_IP $TCP_PORT $IB_DEV \
  $BG_ITERS $BG_MSG write $PORT $GID_IDX \
  $BG_QD $BG_SPAN $BG_PATTERN $BG_ALIGN $MTU \
  1 0 \
  $BG_TS_MS $RESULT_ROOT/cpu/${TAG}_bg_ts.csv \
  0 16 16 $BG_QPS \
  | tee $RESULT_ROOT/cpu/${TAG}_bg.log
```

### 4.5 先测 control port

GPU 机器：

```bash
cd $ROOT/src
uv run python3 tools/cpu_client_control.py --host $CPU_IP --port $CTRL_PORT STATUS
uv run python3 tools/cpu_client_control.py --host $CPU_IP --port $CTRL_PORT LOW
uv run python3 tools/cpu_client_control.py --host $CPU_IP --port $CTRL_PORT HIGH
```

如果这里不通，就不要继续跑 controller。

### 4.6 跑 closed-loop controlled case

GPU 机器：

```bash
TAG=controlled_r1
BASELINE_RESTORE_GIB_S=25.113492

cd $ROOT/src
uv run python3 tools/pcie_case_study_controller.py \
  --cpu_control_host $CPU_IP \
  --cpu_control_port $CTRL_PORT \
  --baseline_restore_gib_s $BASELINE_RESTORE_GIB_S \
  --restore_total_bytes $RESTORE_TOTAL \
  --restore_max_outstanding_bytes $RESTORE_MAX_OUTSTANDING \
  --chunk_mb $RESTORE_CHUNK_MB \
  --streams $RESTORE_STREAMS \
  --batch $RESTORE_BATCH \
  --inflight $RESTORE_INFLIGHT \
  --progress_ms $RESTORE_PROGRESS_MS \
  --progress_smooth_windows $RESTORE_PROGRESS_SMOOTH_WINDOWS \
  --poll_ms $CTRL_POLL_MS \
  --gpu_metrics_interval_ms $GPU_METRICS_INTERVAL_MS \
  --warmup_ms 1000 \
  --post_restore_ms 2000 \
  --enter_ratio 0.65 \
  --exit_ratio 0.80 \
  --rx_threshold_pct 85 \
  --enter_windows $CTRL_ENTER_WINDOWS \
  --exit_windows 2 \
  --log_dir $RESULT_ROOT/gpu \
  --tag $TAG
```

### 4.7 跑 open-loop 对照组

建议至少保留两组：

- `always_high_r1`
- `always_low_r1`

做法和 controlled 基本相同，只是不用 controller，而是手动固定 background 档位：

- `always_high_r1`：先发 `HIGH`
- `always_low_r1`：先发 `LOW`，也就是 `LOW1`

如果你想拿“最强固定保护”当对照，就不要用 `LOW`，而是显式发 `LOW4`。这一点也是旧文档里最容易混淆的地方之一。

### 4.8 拷回 CPU 侧时间序列并做合并分析

GPU 机器：

```bash
scp ${CPU_USER}@${CPU_IP}:$RESULT_ROOT/cpu/controlled_r1_bg_ts.csv $RESULT_ROOT/gpu/
```

然后生成 merged timeline：

```bash
cd $ROOT
uv run python3 src/tools/pcie_case_study_analyze.py \
  --controller_csv $RESULT_ROOT/gpu/controlled_r1_controller.csv \
  --bg_ts_csv $RESULT_ROOT/gpu/controlled_r1_bg_ts.csv \
  --out_csv $RESULT_ROOT/merged/controlled_r1_timeline.csv \
  --out_png $RESULT_ROOT/fig/controlled_r1_timeline.png
```

如果要直接导 paper 风格 PDF，可以再加：

```bash
--paper_out_pdf paper/figures/pcie_case_study_control.pdf
```

## 5. 主要输出文件

### 5.1 Baseline

- `baseline/baseline_restore.log`
- `baseline/baseline_restore_progress.csv`

### 5.2 Controlled run

GPU 机器：

- `gpu/controlled_r1_controller.csv`
- `gpu/controlled_r1_gpu_metrics.csv`
- `gpu/controlled_r1_restore_progress.csv`
- `gpu/controlled_r1_restore.log`
- `gpu/controlled_r1_gpu_metrics.log`

CPU 机器：

- `cpu/controlled_r1_bg_ts.csv`
- `cpu/controlled_r1_bg.log`

合并后：

- `merged/controlled_r1_timeline.csv`
- `fig/controlled_r1_timeline.png`

### 5.3 Open-loop references

- `gpu/always_high_r1_restore_progress.csv`
- `gpu/always_low_r1_restore_progress.csv`
- `cpu/always_high_r1_bg_ts.csv`
- `cpu/always_low_r1_bg_ts.csv`

## 6. 当前结果快照

下面这组数字来自原先 `results report` 中整理过的当前快照，仍可作为 paper 讨论时的摘要口径。

### 6.1 Baseline

- isolated restore completion time: `2548 ms`
- isolated restore average bandwidth: `25.11 GiB/s`

### 6.2 Restore side comparison

| Run | Restore time | Restore avg BW |
| --- | ---: | ---: |
| `always_high` | `13077 ms` | `4.89 GiB/s` |
| `always_low` | `4466 ms` | `14.33 GiB/s` |
| `controlled` | `3651 ms` | `17.53 GiB/s` |

对应结论：

- `controlled` 相比 `always_high` 快 `72.1%`
- `controlled` 相比 `always_low` 快 `18.2%`
- `controlled` 达到 isolated baseline 的 `69.8%`

### 6.3 Closed-loop 行为

代表性的触发点：

- `178 ms`: `restore_guard_bw = 0.70 GiB/s`, `pcie_rx = 95.17%`
- `346 ms`: `restore_guard_bw = 2.89 GiB/s`, `pcie_rx = 93.95%`
- `550 ms`: `restore_guard_bw = 7.29 GiB/s`, `pcie_rx = 93.82%`

当前 run 的进入保护阈值：

- `16.32 GiB/s`

对应的控制动作：

- `SWITCH_LOW1` at `178 ms`
- `ESCALATE_LOW2` at `346 ms`
- `ESCALATE_LOW3` at `346 ms`
- `ESCALATE_LOW4` at `550 ms`
- `RESTORE_DONE_HIGH` at `3651 ms`
- `STOP` after post-restore tail

代表性的恢复窗口：

- `685 ms`: `21.46 GiB/s`
- `824 ms`: `22.48 GiB/s`
- `3179 ms`: `15.98 GiB/s`

结论很直接：

- controller 能快速识别“restore 下滑且 PCIe RX 很高”的有害重叠
- background write 在保护窗口内被明显压低
- restore 在 `LOW4` 之后大部分时间能回到或接近保护目标
- restore 结束后，background write 又恢复到高带宽

### 6.4 Background throughput

在合并后的 controlled timeline 中：

- 保护前平均 background throughput：`21.42 GiB/s`
- 保护窗口内平均 background throughput：`7.05 GiB/s`
- restore 完成后的 `HIGH tail`：`21.17 GiB/s`

这说明控制面不是“只改 mode label”，而是真的把 competing RDMA 注入速率压下去了。

## 7. 结果解释时的注意点

### 7.1 `always_low` 的口径要说清楚

当前 open-loop 对照如果是手动发 `LOW`，那它实际对应的是 `LOW1`，不是最强的固定低档。

因此：

- `always_low` 这个名字本身容易让人误解成“最强固定保护”
- 如果你后面要写论文，最好明确写成 `always_low1`
- 如果你要做更严格的固定低档对照，建议单独补一个 `always_low4`

### 7.2 末尾短时尖峰不要按稳态解释

restore 最后一两个窗口可能出现偏高瞬时带宽，这是收尾阶段的记账效应，不应该被解释成可持续 PCIe 带宽。

对外建议优先使用：

- restore completion time
- restore average bandwidth
- 合并时间线的整体形状

### 7.3 当前 controller 不是完整预测式 scheduler

现在这个原型故意保持克制：

- 不是 TTFT-aware scheduler
- 不用 NIC counters 做在线控制
- 不用 CUDA stream priority 做主要手段
- 当前只覆盖 `H2D restore` 对 `RDMA write` 这一组方向

## 8. 推荐的 paper 表述

当前实现和数据最稳妥的描述是：

- foreground 是一个 `critical restore window`
- controller 观察 `restore progress` 和 `GPU-side PCIe RX pressure`
- knob 是 `RDMA pacing`
- 当 restore 关键窗口受到 PCIe 竞争伤害时，系统临时降低 background RDMA 注入速率
- 关键窗口结束后，background 带宽恢复

不建议在这一版原型上额外声称：

- 完整 deadline-aware runtime
- NIC-side counters 已进入控制闭环
- stream priority 是主要控制手段

## 9. 兼容旧入口

以下两份文档现在只保留为跳转页：

- `pcie_case_study_experiment_readme.md`
- `pcie_case_study_results_report.md`

后续如果 case study 逻辑再变化，优先只更新本文件。
