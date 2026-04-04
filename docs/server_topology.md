# 服务器 PCIe / GPU / RDMA 拓扑说明

本文根据当前服务器上的实际探测结果整理，目标是回答两个问题：

1. 这台 GPU 服务器的 PCIe / GPU / RNIC 结构是什么样子？
2. `gpu_server` / `cpu_client` 里的 RDMA 流量，是否会经过 “PCIe switch -> GPU” 这条路径？

## 1. 机器概况

- 主机名：`chi-mi325x-pod2-098.ord.vultr.cpe.ice.amd.com`
- 机型：`Supermicro AS -8126GS-TNMR`
- CPU 平台：AMD Turin
- 操作系统：Ubuntu 24.04.3 LTS
- GPU：8 x `AMD Instinct MI325X`
- RDMA 设备：
  - 2 x `Mellanox ConnectX-6 Dx`（`mlx5_0` / `mlx5_1`）
  - 9 x `Broadcom BCM57608` RoCE 设备（`bnxt_re0` ... `bnxt_re8`）

## 2. 结论先说

### 2.1 如果你用的是 `mlx5_0` / `mlx5_1`

这两个 Mellanox 设备挂在 **单独的 root complex (`0000:30`)** 下：

- `mlx5_0` -> `0000:31:00.0`
- `mlx5_1` -> `0000:31:00.1`

而 8 张 MI325X 分别挂在：

- `0000:05:00.0`
- `0000:15:00.0`
- `0000:65:00.0`
- `0000:75:00.0`
- `0000:85:00.0`
- `0000:95:00.0`
- `0000:e5:00.0`
- `0000:f5:00.0`

所以：

- **默认示例里如果你传的是 `mlx5_*`，RDMA 不会走 GPU 旁边那条共享 PCIe 分支。**
- 它也**不会经过那些出现在 `00/70/80/f0` 域里的外置 `PEX890xx` switch 分支**。
- 从拓扑上看，`mlx5_*` 和 GPU 只是在更上层 CPU / I/O fabric 处汇合，不属于“同一条 GPU-side PCIe 分支”。

### 2.2 如果你用的是 `bnxt_re*`

大部分 `bnxt_re*` 都和某一张 GPU 位于同一个 PCIe 子树下，典型模式是：

- 同一个 AMD root port
- 同一个上游 PCIe bridge
- 然后 GPU 和 Broadcom RNIC 分别挂在两个不同的下游口

这意味着：

- **RDMA 到 GPU 会进入同一个 PCIe 子树。**
- 但它**不是经过外置 Broadcom / LSI `PEX890xx` 那条支路**到 GPU。
- 更准确地说，它是经过 **AMD root complex 下的共享上游桥/交换层**，然后分别走向 RNIC 分支和 GPU 分支。

## 3. 代码里默认更像哪一种？

仓库里的默认示例和注释更偏向 Mellanox：

- `src/gpu_server.cu` 示例：`./gpu_server mlx5_1 ...`
- `src/cpu_client.cc` 示例：`./cpu_client ... mlx5_1 ...`
- `src/tools/rdma_msgsize_sweep.py` 示例参数：`--ib_dev mlx5_0`

所以如果你直接照仓库里的默认示例跑，**大概率走的是 `mlx5_*` 这条“独立 RNIC 分支”，而不是 GPU 旁边的 Broadcom 配对网卡分支。**

## 4. 整机拓扑概览

这台机器不是“1 张 GPU + 1 张 RNIC + 1 个共享 switch”的简单结构，而是：

- 8 张 MI325X
- 其中 8 个 Broadcom 400G RoCE 网卡分别贴近 8 张 GPU
- 另有 1 个额外 Broadcom 网卡 `bnxt_re6`
- 另有 1 张双口 Mellanox ConnectX-6 Dx（`mlx5_0` / `mlx5_1`），挂在完全不同的 root complex 上

### 4.1 整机示意图

```text
                              CPU / AMD Turin Root Complexes
   ==================================================================================
   || 0000:00  0000:10  0000:30  0000:60  0000:70  0000:80  0000:90  0000:e0  0000:f0 ||
   ==================================================================================

      |         |         |         |         |         |         |         |         |
      |         |         |         |         |         |         |         |         |
   GPU岛0    GPU岛1    Mellanox   GPU岛2    GPU岛3    GPU岛4    GPU岛5    GPU岛6    GPU岛7
   +BNXT     +BNXT      独立岛    +BNXT     +BNXT     +BNXT     +BNXT     +BNXT     +BNXT

   00:01.1   10:01.1    30:01.1   60:01.1   70:01.1   80:01.1   90:01.1   e0:01.1   f0:01.1
      |         |          |         |         |         |         |         |         |
    GPU 05    GPU 15    mlx5_0/1   GPU 65    GPU 75    GPU 85    GPU 95    GPU e5    GPU f5
    BNXT 06   BNXT 16    @31:00    BNXT 66   BNXT 76   BNXT 86   BNXT 96   BNXT e6   BNXT f6

   另外：
   - bnxt_re6 = 0000:c1:00.0，单独挂在 0000:c0 域，没有看到对应 GPU 配对
   - 00 / 70 / 80 / f0 域里还能看到一个外置 PEX890xx switch，但它是另一条支路，不在上面这些 GPU/RNIC 主配对路径里
```

## 5. GPU 与 RNIC 的对应关系

### 5.1 GPU 列表

| GPU编号 | PCI BDF | NUMA |
| --- | --- | --- |
| GPU0 | `0000:05:00.0` | 0 |
| GPU1 | `0000:15:00.0` | 0 |
| GPU2 | `0000:65:00.0` | 0 |
| GPU3 | `0000:75:00.0` | 0 |
| GPU4 | `0000:85:00.0` | 1 |
| GPU5 | `0000:95:00.0` | 1 |
| GPU6 | `0000:e5:00.0` | 1 |
| GPU7 | `0000:f5:00.0` | 1 |

### 5.2 RDMA 设备列表

| IB设备名 | PCI BDF | NUMA | 备注 |
| --- | --- | --- | --- |
| `mlx5_0` | `0000:31:00.0` | 0 | Mellanox ConnectX-6 Dx |
| `mlx5_1` | `0000:31:00.1` | 0 | Mellanox ConnectX-6 Dx |
| `bnxt_re1` | `0000:06:00.0` | 0 | 与 GPU0 同岛 |
| `bnxt_re3` | `0000:16:00.0` | 0 | 与 GPU1 同岛 |
| `bnxt_re2` | `0000:66:00.0` | 0 | 与 GPU2 同岛 |
| `bnxt_re0` | `0000:76:00.0` | 0 | 与 GPU3 同岛 |
| `bnxt_re5` | `0000:86:00.0` | 1 | 与 GPU4 同岛 |
| `bnxt_re8` | `0000:96:00.0` | 1 | 与 GPU5 同岛 |
| `bnxt_re7` | `0000:e6:00.0` | 1 | 与 GPU6 同岛 |
| `bnxt_re4` | `0000:f6:00.0` | 1 | 与 GPU7 同岛 |
| `bnxt_re6` | `0000:c1:00.0` | 1 | 单独存在，未看到同岛 GPU |

## 6. 代表性拓扑图

下面选两个最有代表性的情况：

1. `GPU3 (75:00.0)` + `bnxt_re0 (76:00.0)`：共享同一个 PCIe 子树
2. `GPU3 (75:00.0)` + `mlx5_0 (31:00.0)`：完全不在同一个 PCIe 子树

### 6.1 配对 Broadcom RNIC + GPU：会进入同一 PCIe 子树

```text
                         CPU / Root Complex (0000:70)
                                      ^
                                      |
                             0000:70:01.1  Root Port
                                      ^
                                      |
                             0000:71:00.0  Shared upstream bridge
                               /                       \
                              /                         \
                             /                           \
                0000:72:00.0 GPU branch         0000:72:01.0 RNIC branch
                         ^                                 ^
                         |                                 |
                0000:73:00.0                      0000:76:00.0  Broadcom RNIC
                         ^
                         |
                0000:74:00.0
                         ^
                         |
                0000:75:00.0  GPU (MI325X)

说明：
- RDMA 若绑定到这个 Broadcom RNIC，会进入和 GPU 相同的 PCIe 子树
- 但 GPU 与 RNIC 在 `71:00.0` 之后就分叉，不是共用同一个 endpoint link
- 这个路径里没有经过单独那颗 `PEX890xx` 外置 switch
```

### 6.2 Mellanox `mlx5_*`：不走 GPU 同岛路径

```text
                         CPU / AMD I/O Fabric
               _________________________________________________
              /                                                 \
             /                                                   \
            v                                                     v
   Root Complex 0000:70                                   Root Complex 0000:30
            |                                                     |
        70:01.1                                               30:01.1
            |                                                     |
        71:00.0                                               31:00.0/.1
            |                                                     |
       GPU branch                                               Mellanox
            |
       75:00.0  GPU

说明：
- `mlx5_0` / `mlx5_1` 与 GPU 不在同一个 root complex 下
- 因此不会走“GPU 同岛 PCIe 分支”
- 也不会经过 `00/70/80/f0` 这些 GPU 域里出现的 PEX890xx switch 分支
```

## 7. `PEX890xx` switch 到底在不在 RDMA->GPU 主路径上？

从 `lspci -Dtv` 看，`00` / `70` / `80` / `f0` 域里确实都能看到：

- `Broadcom / LSI PEX890xx PCIe Gen 5 Switch`

但它们挂法是这样的：

- GPU 所在分支：`... 71:00.0 -> 72:00.0 -> 73:00.0 -> 74:00.0 -> 75:00.0`
- Broadcom RNIC 所在分支：`... 71:00.0 -> 72:01.0 -> 76:00.0`
- `PEX890xx` 所在分支：`... 71:00.0 -> 72:1e.0 -> 79:00.0`

也就是说：

- `PEX890xx` 是 **和 GPU 分支、RNIC 分支并列的另一条支路**
- **不是** GPU 与 Broadcom RNIC 之间的必经之路

所以，针对“RDMA 会不会经过 PCIe switch 到 GPU 那条路径”这个问题，建议分两层回答：

### 7.1 物理拓扑层面

- 用 `mlx5_*`：**不会走 GPU 那条 PCIe 子树**
- 用配对 `bnxt_re*`：**会进入 GPU 所在的同一 PCIe 子树**
- 但即使是配对 `bnxt_re*`，**也不是经过那颗单独的 `PEX890xx` 外置 switch**

### 7.2 真正数据路径层面

仅凭 `lspci` 拓扑，我们可以判断“设备之间是怎么挂的”，但不能 100% 证明每一次 GPU 内存 DMA 都是某一种特定的 P2P 实现方式。

实际是否是：

- 纯粹的 PCIe P2P
- 经过 CPU I/O die / root complex
- 或者受限于 peer-direct / DMA-BUF / IOMMU 配置

还要看：

- ROCm GPUDirect / peer-direct 支持是否打开
- RNIC 驱动是否支持 GPU memory registration
- IOMMU / ACS / 平台固件策略

也就是说：

- **拓扑上可以判断“可能走哪条硬件路径”**
- **但最终是否形成理想的 GPU-direct 数据平面，还要看驱动和平台能力**

## 8. 对当前项目的直接建议

你的项目里 `gpu_server` / `cpu_client` 是通过命令行传 `ib_dev` 的，所以建议明确区分两种实验模式：

### 8.1 如果你想测“独立 RNIC -> GPU”

用：

- `mlx5_0`
- `mlx5_1`

这更接近“RNIC 与 GPU 不共用同一 PCIe 岛”的情况。

### 8.2 如果你想测“贴近 GPU 的配对 RNIC -> GPU”

优先用这些配对关系：

- `bnxt_re1` <-> `GPU0` (`05:00.0`)
- `bnxt_re3` <-> `GPU1` (`15:00.0`)
- `bnxt_re2` <-> `GPU2` (`65:00.0`)
- `bnxt_re0` <-> `GPU3` (`75:00.0`)
- `bnxt_re5` <-> `GPU4` (`85:00.0`)
- `bnxt_re8` <-> `GPU5` (`95:00.0`)
- `bnxt_re7` <-> `GPU6` (`e5:00.0`)
- `bnxt_re4` <-> `GPU7` (`f5:00.0`)

这样更接近“RNIC 与 GPU 在同一 PCIe 子树内”的情况。

## 9. 本文依据的探测命令

本文主要基于以下命令结果整理：

```bash
hostnamectl
lspci -Dtv
lspci -Dnn
ibv_devices
ibstat
rocm-smi --showbus
rocm-smi --showtopo
readlink -f /sys/class/infiniband/*/device
readlink -f /sys/bus/pci/devices/<GPU_BDF>
```

## 10. 最终一句话结论

**如果你按当前仓库默认示例使用 `mlx5_*`，RDMA 不会经过 GPU 那条同岛 PCIe 路径；如果你改用和 GPU 配对的 `bnxt_re*`，RDMA 会进入同一 PCIe 子树，但也不是走那颗单独的 `PEX890xx` switch 支路。**
