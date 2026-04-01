# LMCache 双机 PD 分离与 CPU KV Cache Offloading 代码说明

这份文档专门回答一个很容易混淆的问题：

- `LMCache` 里的 **双机 PD 分离**
- `LMCache` 里的 **CPU KV cache offloading**
- `LMCache` 里的 **remote backend**
- `LMCache` 里的 **pd_buffer_device=cpu**

它们到底是什么关系，代码里又是怎么接起来的。

这份文档主要基于你本地仓库里的 LMCache 源码，而不是只基于官网 quickstart：

- [LMCache README](/Users/daniel/Documents/code/SkyGDR/LMCache/README.md)
- [config.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/config.py)
- [cache_engine.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py)
- [storage_backend/__init__.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/__init__.py)
- [storage_manager.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py)
- [pd_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py)
- [remote_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/remote_backend.py)
- [disaggregated prefill quickstart](/Users/daniel/Documents/code/SkyGDR/LMCache/docs/source/getting_started/quickstart/disaggregated_prefill.rst)
- [configurations.rst](/Users/daniel/Documents/code/SkyGDR/LMCache/docs/source/api_reference/configurations.rst)
- [1p1d example configs](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml)
- [1p1d example configs](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml)

## 1. 先说结论

如果只看代码，结论可以拆成 4 条：

1. **LMCache 的 PD 主线是存在的**
   - 对应 `enable_pd=True`
   - 运行时核心 backend 是 `PDBackend`
   - 语义上是 `prefiller(sender) -> decoder(receiver)` 的 KV 传输

2. **LMCache 的 CPU offloading 也是独立存在的**
   - 对应 `local_cpu=True`
   - 运行时核心 backend 是 `LocalCPUBackend`
   - 语义上是把 KV cache 存在本机 CPU 内存

3. **LMCache 的 remote backend 又是另一层**
   - 对应 `remote_url != null`
   - 运行时核心 backend 是 `RemoteBackend`
   - 语义上是把 KV 再存到远端共享/集中式后端

4. **代码层面，`PD + LocalCPU + RemoteBackend` 是可以共存的**
   - 这点和一部分文档口径并不完全一致
   - 仓库里甚至已经有 `pd-with-remote-config.yaml` 示例

但如果再加一句更保守的系统结论：

- **“双机 PD 分离 + 一边只有 CPU、没有 GPU” 并不属于当前 LMCache 明确稳定支持的标准形态。**

原因不是 CPU backend 不存在，而是：

- PD receiver 这边本质上仍然是 decode vLLM 实例
- 官方 quickstart 和 example 主线默认都要求至少 2 GPUs

## 2. 这几个词不要混着理解

这里最容易混掉的，是下面 4 个概念。

### 2.1 `PDBackend`

这是 **Prefill-Decode Disaggregation** 的核心 backend。

代码入口：

- [storage_backend/__init__.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/__init__.py)
- [pd_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py)

当：

- `enable_pd=True`

时，LMCache 会创建：

- `PDBackend`

它的角色分成两种：

- `pd_role="sender"`：prefiller
- `pd_role="receiver"`：decoder

在 [pd_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py) 里，`PDBackend` 会：

- 根据 `pd_role` 建 sender / receiver 逻辑
- 根据 `pd_buffer_device` 建 CPU 或 GPU buffer allocator
- 调 `CreateTransferChannel(...)`
- 通过 `transfer_channel`（当前主线是 `nixl`）做真正的 PD 传输

### 2.2 `LocalCPUBackend`

这是 **本机 CPU 内存 KV cache backend**。

控制配置：

- `local_cpu`
- `max_local_cpu_size`

语义是：

- 把 KV 存到**本机** CPU 内存
- 既可作为真实 cache tier
- 也可作为其它 backend 的 staging/buffer

最关键的一点是：

- `LocalCPUBackend` 不等于 PD
- 它也不等于 remote backend

### 2.3 `RemoteBackend`

这是 **远端共享存储 backend**。

控制配置：

- `remote_url`
- `remote_serde`

实现入口：

- [remote_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/remote_backend.py)

它依赖一个很关键的前提：

- 必须先有 `LocalCPUBackend`

在 [storage_backend/__init__.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/__init__.py) 里，这段逻辑非常明确：

- 如果 `config.remote_url is not None`
- 会先要求 `local_cpu_backend is not None`
- 然后才创建 `RemoteBackend`

也就是说：

- `RemoteBackend` 不是“直接从 GPU 对远端”
- 而是依赖本地 CPU backend 当中间层 / buffer

### 2.4 `pd_buffer_device`

这是最容易被误解的一个配置。

控制配置：

- `pd_buffer_device: "cpu"` 或 `"cuda"`

它的含义不是：

- “整个 decode 节点可以只用 CPU”

它真正表示的是：

- **PD 传输 buffer 分配在 CPU 还是 GPU**

在 [pd_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py) 里，这个配置会决定：

- `PagedCpuGpuMemoryAllocator` 初始化 CPU allocator 还是 GPU allocator
- `CreateTransferChannel(...)` 的 `device` 传什么

所以：

- `pd_buffer_device=cpu` 是 **PD buffer 在 CPU**
- 不是“PD receiver 可以没有 GPU”

## 3. 配置校验：文档口径和代码口径有差异

这是目前最值得你记住的一点。

### 3.1 文档口径

在 [configurations.rst](/Users/daniel/Documents/code/SkyGDR/LMCache/docs/source/api_reference/configurations.rst) 的 PD 配置节里，文档明确写了：

- 当 `enable_pd` 时，当前有这些限制：
  - `remote_url must be null`
  - `save_decode_cache must be false`
  - `enable_p2p must be false`

如果只看这段文档，你会得到一个结论：

- PD 模式和 remote backend 是互斥的

### 3.2 代码口径

但在 [config.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/config.py) 的 `_validate_config()` 里，我没有看到：

- `enable_pd=True` 时强制 `remote_url is None`

这段配置校验对 `enable_pd` 真正检查的是：

- `pd_role` 必须有
- `pd_buffer_size` 必须有
- `pd_buffer_device` 必须有
- `enable_p2p=False`
- 自动把 `save_unfull_chunk=True`
- 如果是 receiver：
  - `store_location != "PDBackend"`
  - `retrieve_locations in (None, ["PDBackend"])`

也就是说：

- **代码没有把 `remote_url` 禁掉**

### 3.3 示例口径

更关键的是，仓库里还有这些 example config：

- [lmcache-prefiller-pd-with-remote-config.yaml](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml)
- [lmcache-decoder-pd-with-remote-config.yaml](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml)

它们都明确写了：

- `enable_pd: True`
- `remote_url: "lm://localhost:6800"`
- `local_cpu: True`

这说明什么？

说明至少在当前仓库快照里：

- **官方主文档口径**
和
- **代码/示例口径**

是存在漂移的。

更准确的说法应该是：

- `PD + remote backend` 不是文档里最稳定、最推荐的主线
- 但**代码和示例已经在支持/尝试这种 hybrid 组合**

## 4. backend 初始化路径到底怎么走

最核心的初始化逻辑在：

- [storage_backend/__init__.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/__init__.py)

这里的顺序很关键。

### 4.1 第一步：先决定 `PDBackend`

如果：

- `config.enable_pd`

就先创建：

- `PDBackend`

这意味着：

- PD 是一种一等 backend

### 4.2 第二步：决定要不要创建 `LocalCPUBackend`

相关逻辑是：

- `elif not config.enable_pd or config.local_cpu:`

这句话非常重要。

它表示：

- 即使 `enable_pd=True`
- **只要 `local_cpu=True`**
- 仍然会创建 `LocalCPUBackend`

所以：

- `PD` 并不天然排斥 `LocalCPUBackend`

### 4.3 第三步：决定要不要创建 `RemoteBackend`

当：

- `config.remote_url is not None`

时，会要求：

- `local_cpu_backend is not None`

然后创建：

- `RemoteBackend`

因此：

- `PDBackend + LocalCPUBackend + RemoteBackend`
在代码层面是可以同时存在的

## 5. cache engine 里，store / retrieve 怎么路由

这一层要看：

- [cache_engine.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py)

LMCache 这里有两个很关键的概念：

- `store_location`
- `retrieve_locations`

这两个配置决定了：

- 存 KV 往哪儿走
- 读 KV 从哪儿找

### 5.1 store 路径

在 `store()` / `store_layer()` 里，最终会调用：

- `self.storage_manager.batched_put(..., location=self.store_location)`

也就是说：

- `store_location=None`
  - 往所有活跃 backend 发
- `store_location="RemoteBackend"`
  - 只往 `RemoteBackend` 发
- `store_location="LocalCPUBackend"`
  - 只存本地 CPU

### 5.2 retrieve 路径

在 `retrieve()` / `retrieve_layer()` 里，最终会走：

- `self.retrieve_locations`

并通过：

- `storage_manager.contains(...)`
- `storage_manager.get(...)`

去按顺序找 backend。

所以这两个配置才是 hybrid 语义的关键。

## 6. `pd-with-remote-config` 这两个例子到底在表达什么

这是你现在最值得关注的部分。

### 6.1 prefiller 侧

看：

- [lmcache-prefiller-pd-with-remote-config.yaml](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml)

关键配置：

- `local_cpu: True`
- `remote_url: "lm://localhost:6800"`
- `retrieve_locations: ["LocalCPUBackend", "RemoteBackend"]`
- `enable_pd: True`
- `pd_role: "sender"`

这说明 prefiller 这一侧的语义更像：

- 既能从本地 CPU / 远端共享 backend 查已有 KV
- 又能通过 `PDBackend` 作为 sender，把本轮 prefill 结果发给 decoder

### 6.2 decoder 侧

看：

- [lmcache-decoder-pd-with-remote-config.yaml](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml)

关键配置：

- `local_cpu: True`
- `remote_url: "lm://localhost:6800"`
- `retrieve_locations: ["PDBackend"]`
- `store_location: "RemoteBackend"`
- `enable_pd: True`
- `pd_role: "receiver"`
- `save_decode_cache: true`

这个配置表达的语义其实很清楚：

- decoder 的输入 KV 只从 `PDBackend` 取
- 也就是 prefill -> decode 的主数据路径仍然是 PD
- 但 decoder 自己生成出来的 KV，可以继续往 `RemoteBackend` 存

这意味着：

- **PD 和 remote backend 在这个例子里不是替代关系**
- 而是：
  - `PDBackend` 负责 prefill 到 decode 的跨节点传输
  - `RemoteBackend` 负责额外的共享/持久 tier

## 7. `RemoteBackend` 的真实语义

看：

- [remote_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/remote_backend.py)

你会发现它并不是一个“GPU 直接打远端”的 backend。

它的运行模型是：

1. 依赖 `LocalCPUBackend`
2. 用 serializer 压缩/序列化 `MemoryObj`
3. 通过 `CreateConnector(remote_url, ...)` 接远端
4. 异步 `put`
5. `get` 之后，如果来源不是 `LocalCPUBackend` / `PDBackend`，还会自动写回 `LocalCPUBackend`

因此它的更准确理解是：

- **一个以本地 CPU 为 staging tier 的远端 KV backend**

## 8. `PDBackend` 的真实语义

看：

- [pd_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py)

它不是 remote KV store，而是：

- **prefiller 和 decoder 之间的传输 backend**

最关键的特点：

1. sender / receiver 是不对称的
2. receiver 才真正保留接收到的数据
3. sender 更像“分配 + 发送”
4. buffer 可以在 CPU 或 GPU
5. 真正走哪种链路由 `transfer_channel` 决定

在当前主线里：

- `transfer_channel` 通常是 `nixl`

## 9. 代码层面可以支持哪些组合

从当前代码看，下面这些组合是能成立的。

### 9.1 纯 CPU offloading

- `local_cpu=True`
- `enable_pd=False`
- `remote_url=None`

语义：

- KV 只落在本机 CPU

### 9.2 CPU + remote backend

- `local_cpu=True`
- `enable_pd=False`
- `remote_url!=None`

语义：

- 本机 CPU + 远端共享 backend

### 9.3 纯 PD

- `enable_pd=True`
- `pd_role=sender/receiver`
- `remote_url=None`

语义：

- prefill -> decode 通过 PD 直接传

### 9.4 PD + LocalCPU + RemoteBackend

- `enable_pd=True`
- `local_cpu=True`
- `remote_url!=None`

语义：

- PD 负责 prefill -> decode
- LocalCPUBackend 作为本地 CPU tier / buffer
- RemoteBackend 作为远端 tier

这是代码里“看起来已经支持”的 hybrid 形态。

## 10. 但什么东西仍然不能轻易下结论

最重要的就是这一点：

- **“双机里只有一边有 GPU，另一边只有 CPU，然后还要跑真实 PD decode”**

我不建议把它当作“LMCache 已明确支持”的结论。

原因有三层：

1. 官方 PD quickstart 仍然要求：
   - 至少 2 GPUs
   - [disaggregated_prefill.rst](/Users/daniel/Documents/code/SkyGDR/LMCache/docs/source/getting_started/quickstart/disaggregated_prefill.rst)

2. 官方 1p1d example 仍然默认：
   - prefiller GPU 0
   - decoder GPU 1
   - [1p1d README](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/README.md)

3. `pd_buffer_device=cpu`
   - 只能说明 PD 传输 buffer 可以在 CPU
   - 不能直接推出 “decoder 端可以没有 GPU”

所以更稳妥的说法是：

- **代码里已经有“PD + CPU/Remote tier” 的 hybrid 能力**
- 但**CPU-only decoder node 不是当前文档清晰支持的标准部署形态**

## 11. 如果你想读代码，最推荐的顺序

我建议按下面顺序看：

1. 配置定义与校验
   - [config.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/config.py)

2. backend 是怎么被实例化的
   - [storage_backend/__init__.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/__init__.py)

3. PD 本身怎么工作
   - [pd_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py)

4. remote backend 怎么工作
   - [remote_backend.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/remote_backend.py)

5. store / retrieve 路径如何选择 backend
   - [cache_engine.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py)
   - [storage_manager.py](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py)

6. 最后对照 example config
   - [lmcache-prefiller-pd-with-remote-config.yaml](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml)
   - [lmcache-decoder-pd-with-remote-config.yaml](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml)

## 12. 对你当前项目最有用的判断

如果你的目标是：

- 理解 LMCache 有没有“PD + CPU/remote tier” 这条线

那答案是：

- **有，代码里已经出现了**

如果你的目标是：

- 在你现在 `1 A100 + 1 CPU-only server` 的硬件上直接跑成“真双机 PD”

那我的判断仍然是：

- **不适合作为近期主线**

因为这里卡的不是 “CPU offloading” 四个字，而是：

- decode 节点本身是不是一个 GPU-serving 节点

这才是 PD 主路径真正依赖的前提。

## 13. 一句话总结

LMCache 当前代码里，“CPU KV cache offloading” 至少有三层不同含义：

1. **本机 CPU tier**：`LocalCPUBackend`
2. **远端共享 tier**：`RemoteBackend`
3. **PD 传输 buffer 在 CPU**：`pd_buffer_device=cpu`

它们都和 “CPU” 有关，但**不是同一件事**。

而 `PDBackend` 则是：

- prefill -> decode 的跨节点传输 backend

当前代码显示：

- `PDBackend + LocalCPUBackend + RemoteBackend`

是可能共存的；

但“CPU-only decoder 节点”仍然不能从这些配置直接推出已经被官方稳定支持。
