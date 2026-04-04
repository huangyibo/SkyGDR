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

## 0. 一眼看懂：hybrid 版本里到底有哪些“通道”

如果把你关心的双机 `PD + CPU KV offloading hybrid` 只抽成最核心的数据面，它更像下面这个结构。

```text
Server A: Prefiller                                         Server B: Decoder
+-------------------------------+                           +-------------------------------+
| vLLM Prefill GPU              |                           | vLLM Decode GPU               |
|   - new suffix prefill        |                           |   - decode tokens             |
|   - may reuse old prefix KV   |                           |   - retrieve prefiller KV     |
+---------------+---------------+                           +---------------+---------------+
                |                                                           ^
                | (1) from_gpu()                                            | (4) to_gpu()
                v                                                           |
      +---------+----------+                                   +------------+-----------+
      | LMCache sender     |                                   | LMCache receiver        |
      | PDBackend          |==== (3) NIXL / RDMA / TCP =======>| PDBackend               |
      | LocalCPUBackend    |                                   | LocalCPUBackend         |
      | RemoteBackend      |                                   | RemoteBackend           |
      +----+----------+----+                                   +------------+------+-----+
           |          |                                                        |      |
   (2a)    |          | (2b)                                                   |      | (5)
 retrieve  |          | retrieve                                               |      +--> store decode KV
 old prefix|          | old prefix                                             |           to RemoteBackend
 from CPU  |          | from remote                                            |
           v          v                                                        |
        CPU RAM   remote kv store <======================== optional ===========
```

这里最容易混的点，是这 3 条路径其实不是一回事：

1. `prefiller -> decoder` 的主跨机通道，是 `PDBackend`
2. `CPU KV offloading / remote tier`，是 `LocalCPUBackend + RemoteBackend`
3. `pd_buffer_device=cpu` 只是把 **PD transport buffer** 放到 CPU，不等于“decoder 机器可以没有 GPU”

如果对应到 example config：

- prefiller config：[lmcache-prefiller-pd-with-remote-config.yaml](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml)
  - `local_cpu: True` [#L1](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml#L1)
  - `remote_url: "lm://localhost:6800"` [#L4](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml#L4)
  - `retrieve_locations: ["LocalCPUBackend", "RemoteBackend"]` [#L7](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml#L7)
  - `enable_pd: True` [#L9](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml#L9)
  - `pd_role: "sender"` [#L11](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml#L11)

- decoder config：[lmcache-decoder-pd-with-remote-config.yaml](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml)
  - `retrieve_locations: ["PDBackend"]` [#L7](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml#L7)
  - `store_location: "RemoteBackend"` [#L8](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml#L8)
  - `enable_pd: True` [#L10](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml#L10)
  - `pd_role: "receiver"` [#L12](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml#L12)

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

## 5.3 `cache_engine -> storage_manager -> backend` 的主调用链

如果只盯最关键的几行，数据路径实际上很直白：

```text
store path:
GPU KV
  -> gpu_connector.batched_from_gpu(...)
  -> storage_manager.batched_put(..., location=store_location)
  -> backend.batched_submit_put_task(...)

retrieve path:
backend.get_blocking(...) / batched_get_blocking(...)
  -> cache_engine 收到 MemoryObj
  -> gpu_connector.batched_to_gpu(...)
  -> 如果是 PD receiver，可 remove_after_retrieve
```

对应代码：

- `cache_engine` 保存 `store_location` / `retrieve_locations`
  - [cache_engine.py:181-186](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L181)
- store 时先从 GPU 导出，再交给 `storage_manager.batched_put(...)`
  - [cache_engine.py:532-544](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L532)
- retrieve 时再 `batched_to_gpu(...)`
  - [cache_engine.py:851-856](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L851)
- PD receiver retrieve 后会 `remove(...)`
  - [cache_engine.py:860-863](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L860)

而 `storage_manager` 真正做的事情是：

- `batched_put(...)` 按 backend allocator 复制对象，再把对象交给各 backend
  - [storage_manager.py:379-425](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L379)
- `get(...)` / `batched_get(...)` 从 active backend 顺序查找
  - [storage_manager.py:430-452](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L430)
  - [storage_manager.py:475-505](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L475)
- 如果命中来源不是 `LocalCPUBackend` / `PDBackend`，会自动写回 `LocalCPUBackend`
  - [storage_manager.py:445-451](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L445)
  - [storage_manager.py:489-505](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L489)

## 5.4 真正的数据流：3 条关键路径

你要的 hybrid 语义，最值得单独拆成 3 张“脑内流程图”。

### 路径 A：prefiller 侧复用旧 prefix

这条路回答的是：

- prefiller 本轮 prefill 之前，如果想复用旧 KV，会先从哪儿拿？

在 prefiller example config 里：

- `retrieve_locations = ["LocalCPUBackend", "RemoteBackend"]`
  - [lmcache-prefiller-pd-with-remote-config.yaml:7](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml#L7)

所以它的优先级是：

```text
Prefiller GPU wants old prefix KV
    |
    v
cache_engine.retrieve(...)
    |
    v
storage_manager.get(location=["LocalCPUBackend", "RemoteBackend"])
    |
    +--> LocalCPUBackend.get_blocking(...)
    |      [local hit]
    |
    `--> RemoteBackend.get_blocking(...)
           [remote hit]
             |
             `--> auto write-back to LocalCPUBackend
```

关键代码：

- `LocalCPUBackend.get_blocking(...)`
  - [local_cpu_backend.py:202-214](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py#L202)
- `RemoteBackend.get_blocking(...)`
  - [remote_backend.py:317-363](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/remote_backend.py#L317)
- remote 命中后自动写回本地 CPU
  - [storage_manager.py:445-451](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L445)

### 路径 B：prefiller 把本轮 prefill KV 送到 decoder

这条路才是 PD 主干。

```text
Server A / Prefiller
GPU KV
  |
  | 1. cache_engine 从 GPU 导出 chunk
  v
PDBackend(sender allocator domain)
  |
  | 2. sender 向 receiver 请求远端 buffer 地址
  v
receiver alloc socket
  |
  | 3. receiver 先 allocate + put(key, mem_obj)
  v
PDBackend(receiver local data)
  ^
  | 4. sender batched_write(...) 真正传输
  |
NIXL / RDMA / TCP transfer channel
```

关键代码：

- `cache_engine` 从 GPU 导出并进入 `batched_put(...)`
  - [cache_engine.py:532-544](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L532)
- `storage_manager.batched_put(...)` 最后调用 backend 的 `batched_submit_put_task(...)`
  - [storage_manager.py:405-425](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L405)
- sender 建连、准备 alloc socket
  - [pd_backend.py:302-340](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py#L302)
- sender 请求远端分配
  - [pd_backend.py:342-380](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py#L342)
- sender 真正 `batched_write(...)`
  - [pd_backend.py:383-462](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py#L383)
- receiver 侧 allocation loop
  - [pd_backend.py:471-555](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py#L471)
- receiver 侧把预分配的 `mem_obj` 放进本地 `self.data`
  - [pd_backend.py:488-529](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py#L488)
  - [pd_backend.py:557-563](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py#L557)

### 路径 C：decoder 从 PDBackend 取 prefiller KV，再把 decode KV 存去 remote

这条路是最像“hybrid”而不是“纯 PD”的地方。

decoder example config 写的是：

- `retrieve_locations = ["PDBackend"]`
  - [lmcache-decoder-pd-with-remote-config.yaml:7](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml#L7)
- `store_location = "RemoteBackend"`
  - [lmcache-decoder-pd-with-remote-config.yaml:8](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-decoder-pd-with-remote-config.yaml#L8)

它表达的就是：

```text
decode input KV:
PDBackend(receiver local data)
    -> cache_engine.retrieve(...)
    -> gpu_connector.batched_to_gpu(...)
    -> Decode GPU starts decoding

decode output KV:
Decode GPU
    -> cache_engine.store(...)
    -> storage_manager.batched_put(location="RemoteBackend")
    -> RemoteBackend.batched_submit_put_task(...)
    -> remote kv store
```

关键代码：

- receiver 只能从 `PDBackend` retrieve，这是 config 校验直接限制的
  - [config.py:568-579](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/config.py#L568)
- `PDBackend.get_blocking(...)`
  - [pd_backend.py:565-571](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py#L565)
- retrieve 后 `batched_to_gpu(...)`
  - [cache_engine.py:851-856](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L851)
- decoder decode 后如果 `store_location="RemoteBackend"`，就只往 remote 存
  - [cache_engine.py:183-186](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L183)
  - [cache_engine.py:539-544](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L539)
- `RemoteBackend.batched_submit_put_task(...)`
  - [remote_backend.py:258-315](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/remote_backend.py#L258)

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

## 8.1 如果把 `pd_buffer_device` 改成 `cpu`，图会怎么变

这一点值得单独说，因为它最容易被误读成“decoder 可以只用 CPU”。

先看配置定义：

- `pd_buffer_device` 可取 `"cpu"` 或 `"cuda"`
  - [configurations.rst:248-250](/Users/daniel/Documents/code/SkyGDR/LMCache/docs/source/api_reference/configurations.rst#L248)
- 代码校验也要求 `enable_pd=true` 时必须提供它
  - [config.py:545-549](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/config.py#L545)

再看 `PDBackend.allocate(...)`，它走的是自己的 allocator domain：

- [pd_backend.py:251-260](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py#L251)

结合 `cache_engine` 的 `from_gpu(...)` / `to_gpu(...)`，更准确的理解应该是：

```text
pd_buffer_device = "cuda"
  sender GPU KV ----> GPU PD buffer ==== RDMA/NIXL ====> receiver GPU/PD buffer ----> decoder GPU

pd_buffer_device = "cpu"
  sender GPU KV ----> CPU PD buffer ==== RDMA/NIXL ====> receiver CPU PD buffer ----> decoder GPU
```

这里第二行是基于 allocator 选择和 `from_gpu/to_gpu` 调用链做的代码推断。它说明的是：

- `pd_buffer_device=cpu` 更像是在 PD 通道两端引入 CPU staging/buffer
- 不是把 decoder 本身变成 CPU-only

所以如果你问：

- “双机 PD 分离 + CPU KV cache offloading hybrid 里，GPU / CPU / RDMA 之间的传输图是什么样？”

那更接近的答案是：

```text
Prefiller GPU
  -> (optional) LocalCPU / RemoteBackend retrieve old prefix
  -> prefill new suffix on GPU
  -> PD buffer (GPU or CPU, controlled by pd_buffer_device)
  -> RDMA/NIXL
  -> PD buffer on decoder side
  -> Decoder GPU retrieve and decode
  -> decode KV can be offloaded to LocalCPU / RemoteBackend
```

也就是说：

- `PDBackend` 负责 **prefiller -> decoder**
- `LocalCPUBackend / RemoteBackend` 负责 **cache tiering / offloading**
- `pd_buffer_device` 只决定 **PD 通道内部的 buffer 在 CPU 还是 GPU**

## 8.2 prefiller 这一侧，CPU 和 GPU 之间到底有哪些传输

如果只看 prefiller 这一侧，`CPU <-> GPU` 传输其实主要来自两件事：

1. **复用旧 prefix KV**
2. **把本轮新产生的 prefill KV 导出到 PD / CPU / remote tier**

先看最短版图。

```text
Prefiller side

A. 复用旧 prefix
LocalCPU / RemoteBackend  ----H2D---->  Prefill GPU
                               ^
                               |
                    batched_to_gpu(...)

B. 导出本轮新 prefill KV
Prefill GPU  ----D2H or GPU->GPU---->  PD buffer / LocalCPU / RemoteBackend
                  ^
                  |
         batched_from_gpu(...)
```

### 8.2.1 旧 prefix 命中时，是 `CPU -> GPU`

在 prefiller 的 hybrid example 里：

- `retrieve_locations: ["LocalCPUBackend", "RemoteBackend"]`
  - [lmcache-prefiller-pd-with-remote-config.yaml:7](/Users/daniel/Documents/code/SkyGDR/LMCache/examples/disagg_prefill/1p1d/configs/lmcache-prefiller-pd-with-remote-config.yaml#L7)

所以 prefiller 要复用旧 KV 时，会优先在：

- `LocalCPUBackend`
- `RemoteBackend`

里查找，然后再放回 GPU。

对应调用链：

- `storage_manager.get(...)` / `batched_get(...)`
  - [storage_manager.py:430-452](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L430)
  - [storage_manager.py:475-505](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L475)
- `cache_engine` 命中后再 `batched_to_gpu(...)`
  - [cache_engine.py:851-856](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L851)

所以这条路径的本质就是：

```text
LocalCPUBackend / RemoteBackend -> Prefiller GPU KV cache
```

也就是 **H2D**。

如果命中的是 `RemoteBackend`，还有一个额外动作：

- 命中的对象会自动写回 `LocalCPUBackend`
  - [storage_manager.py:445-451](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L445)
  - [storage_manager.py:489-505](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L489)

但这一步是：

```text
RemoteBackend -> LocalCPUBackend
```

属于 **CPU -> CPU**，不是 GPU 传输。

### 8.2.2 新 prefill KV 导出时，会先执行 `from_gpu(...)`

本轮 prefill 结束后，`cache_engine.store(...)` 会先把 GPU 上的 KV 导出：

- [cache_engine.py:532-544](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/cache_engine.py#L532)

这里最关键的一句是：

- `self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)`

`GPUConnectorInterface` 对它的语义定义也很直接：

- `from_gpu(...)`: “Load the data from a GPU buffer into the memory object.”
  - [gpu_connectors.py:55-69](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py#L55)
- `batched_from_gpu(...)`
  - [gpu_connectors.py:72-90](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py#L72)

所以，不管后面要往哪儿存，第一步都是：

```text
Prefill GPU KV -> 某个 MemoryObj
```

但这个 `MemoryObj` 在 CPU 还是 GPU，要看 allocator backend。

### 8.2.3 在 PD 模式下，prefiller 的 allocator backend 默认是 `PDBackend`

`StorageManager` 在 `enable_pd=True` 时，会把 allocator backend 设成 `PDBackend`：

- [storage_manager.py:312-320](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L312)

也就是说，在 PD 模式下，prefiller 先把 GPU KV 导出到：

- `PDBackend` 自己的 buffer domain

而这个 domain 是 CPU 还是 GPU，又取决于：

- `pd_buffer_device`
  - [pd_backend.py:170-179](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/pd_backend.py#L170)

于是要分两种情况。

#### 情况 A：`pd_buffer_device="cuda"`

这时更像：

```text
Prefill GPU --from_gpu--> PD GPU buffer --RDMA WRITE--> Decoder
```

也就是说：

- 主 PD 路径本身 **不一定经过 prefiller CPU**
- 更像是本地 **GPU -> GPU buffer**，然后 sender 发起 RDMA/NIXL 写

#### 情况 B：`pd_buffer_device="cpu"`

这时更像：

```text
Prefill GPU --D2H--> PD CPU buffer --RDMA WRITE--> Decoder
```

所以在这种配置下，prefiller 侧会明确存在一条：

- **GPU -> CPU**

### 8.2.4 如果 prefiller 还同时往 LocalCPU / RemoteBackend 存，会不会再有 CPU-GPU 传输

会，但要看当前起点对象在哪。

`storage_manager.batched_put(...)` 会根据目标 backend 的 allocator domain 复制对象：

- [storage_manager.py:405-425](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L405)

其中关键逻辑是：

- `allocate_and_copy_objects(...)`
  - [storage_manager.py:416-418](/Users/daniel/Documents/code/SkyGDR/LMCache/lmcache/v1/storage_backend/storage_manager.py#L416)

所以：

#### 如果 `pd_buffer_device="cuda"`

此时起始对象先在 PD 的 GPU buffer 里。  
如果还要再往 `LocalCPUBackend` / `RemoteBackend` 存，通常会出现：

```text
PD GPU buffer -> LocalCPU / RemoteBackend staging
```

也就是又一条 **GPU -> CPU**。

#### 如果 `pd_buffer_device="cpu"`

此时起始对象已经在 CPU buffer 里。  
再往 `LocalCPUBackend` / `RemoteBackend` 走时，更像：

```text
PD CPU buffer -> LocalCPU / RemoteBackend
```

这时主要是 **CPU -> CPU**，不是额外的 GPU 传输。

### 8.2.5 一句话总结 prefiller 侧的 CPU/GPU 传输

如果你只关心 prefiller 这一侧：

1. **复用旧 prefix 时**
   - 主要是 **CPU -> GPU**
   - 即 `LocalCPU/Remote -> Prefiller GPU`

2. **把新 prefill KV 送去 PD 时**
   - `pd_buffer_device="cpu"`：有明确 **GPU -> CPU**
   - `pd_buffer_device="cuda"`：主路径不一定经过 CPU

3. **如果同时还要把新 KV 落到 CPU / remote tier**
   - `pd_buffer_device="cuda"` 时，通常会再出现一条 **GPU -> CPU**
   - `pd_buffer_device="cpu"` 时，更多是 **CPU -> CPU**

所以真正决定 prefiller 侧 CPU-GPU 传输图长什么样的，是这两个问题：

- 这轮是在 **retrieve old prefix**，还是在 **store new prefix**
- `pd_buffer_device` 选的是 `"cpu"` 还是 `"cuda"`

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
