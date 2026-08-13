# PD 分离兼容性与 Mooncake 方案清单

> 核对日期：2026-08-13
> 核对对象：FusionOne AI 23.6.1 ARM/X86 模型兼容性表、`wings_control/utils/model_utils.py`、`wings_control/config/defaults/pd_config.json`、当前 Wings PD 配置生成逻辑、NVIDIA Dynamo Recipe、vLLM 与 vLLM-Ascend 官方文档。
> 记录原则：本文只归档来源、版本、模型、权重、硬件、P/D 拓扑、Connector、Proxy、组网、本地差异和验证状态，不输出统一的“能用/不能用”答案。

## 1. 口径清单

每个 PD 组合按以下字段单独登记：

1. 精确模型名、architecture、权重和量化格式；
2. vLLM 或 vLLM-Ascend 版本；
3. GPU/NPU 型号、单卡显存、每节点卡数和节点数；
4. P/D 两侧的全局 DP、TP、EP，本地 DP 和 rank start；
5. Connector、KV 参数、Proxy/Metaserver 请求流程；
6. RDMA/HCCS、网卡、地址和端口要求；
7. Wings 精确 profile、最终命令和镜像物料；
8. 启动、单请求、连续请求、并发、长上下文和长稳状态。

架构相同但任一字段不同，登记为不同组合，不自动继承其他组合的状态。

### 1.1 状态等级

| 等级 | 已有证据 | 登记状态 | 尚未覆盖的范围 |
|---|---|---|---|
| S0 | 引擎能够识别模型架构 | 普通推理识别 | PD 未登记 |
| S1 | Connector 在代码中注册 | Connector 存在 | 镜像运行库、模型和拓扑未登记 |
| S2 | 社区提供架构级 PD 能力矩阵 | 架构级候选 | 精确模型、权重和硬件未验证 |
| S3 | 社区提供具体模型 PD 命令 | 社区模型级配方 | Wings profile、镜像版本和真机状态未登记 |
| S4 | Wings 有精确模型/权重/硬件 profile，最终命令与选定官方版本等价 | 已适配待验收 | 连续请求、并发和稳定性未通过时不得升为 S5 |
| S5 | 真机完成启动、单请求、连续请求、并发、长上下文、精度和稳定性验收 | 产品验证组合 | 只覆盖记录中的版本、权重、硬件和拓扑 |

Embedding/Rerank 模型没有生成式推理中的独立 Prefill 和逐 token Decode 阶段，不进入本文生成式 PD 状态矩阵。

## 2. Mooncake 共性能力与不可共用项

“通用 Mooncake 方案”分三个层次记录：

| 层次 | 已存在的共性能力 | 不能由该层推导的内容 |
|---|---|---|
| Connector/API | 上游 vLLM 提供通用 `MooncakeConnector` producer/consumer 接口；vLLM-Ascend 提供 Mooncake 系列 Connector | 不能推导任意模型、任意硬件和任意 P/D 拓扑可直接运行 |
| 部署骨架 | vLLM-Ascend 提供单节点/多节点脚本、DP 启动器、Proxy、Mooncake 端口和 AscendDirectTransport 组网模板 | 不能推导 V1、Layerwise、Hybrid、MultiConnector 可以互换 |
| 模型级配方 | 部分模型页提供精确权重、A2/A3 资源、P/D DP/TP、Connector 和参数 | 当前没有一套覆盖所有 Ascend 模型、权重、缓存形态和拓扑的统一可执行配方 |

因此，本文中“没有通用方案”的固定含义是：

> 没有跨模型、跨权重、跨硬件、跨缓存布局和跨 P/D 拓扑的一套 Ascend Mooncake 可执行配方。

它不表示 Mooncake 没有通用 Connector/API，也不表示 vLLM-Ascend 没有通用部署教程。

### 2.1 Mooncake 策略清单

| 策略 | 典型用途 | KV/请求语义 | 必须配套记录的内容 |
|---|---|---|---|
| `MooncakeConnector` | vLLM 上游常规 P2P；vLLM-Ascend `kv_p2p` 新路径 | producer/consumer，整组 KV 传输 | 包版本、模块路径、role 级 engine ID、bootstrap port、普通 Proxy |
| `MooncakeConnectorV1` | vLLM-Ascend 常规模型 PD | producer/consumer，常规 Mooncake KV 传输 | P/D DP/TP 四元组、每 rank engine ID、KV port、普通 Proxy |
| `MooncakeLayerwiseConnector` | 大模型逐层传输 | P 逐层产生并推送，D 分层消费 | Layerwise Proxy、Metaserver、D 先接收请求并反向触发 P 的顺序 |
| `MooncakeHybridConnector` | 多 KV cache group / Hybrid KV 模型 | 同时处理不同 cache spec | Hybrid KV manager、block size、MTP/DSpark、调度和 Prefix Cache |
| `MultiConnector` | Mooncake 与 AscendStore/KV Pool 等组合 | 多个子 Connector 共同装载或回退 | 子 Connector 顺序、KV Pool、失败策略和附加服务 |
| `MooncakeStoreConnector` / StoreV1 | 共享缓存池、CPU/磁盘 offload、跨实例缓存 | 通过 Mooncake store 存取 KV | master/client、存储配置、哈希一致性、容量和淘汰策略 |

### 2.2 不能只改 Connector 名称的配套项

| 变更 | 同时变化的组件 |
|---|---|
| V1 → Layerwise | Proxy 请求顺序、Metaserver、KV 完成通知、端口和实例发现 |
| Layerwise → V1 | 普通 P→D Proxy、producer/consumer 角色、请求元数据和完成通知 |
| V1/Layerwise → Hybrid | Hybrid KV manager、cache group/spec、block layout、调度、MTP/DSpark |
| V1 → MultiConnector | 子 Connector 配置、AscendStore/KV Pool 服务、失败回退策略 |
| 上游 `MooncakeConnector` → vLLM-Ascend V1 | Connector 模块、engine ID 语义、extra config 和镜像中的原生库组合 |

## 3. 当前 Wings 和产品清单

### 3.1 三份兼容性来源

| 来源 | 规模 | 表示的范围 | 不表示的范围 |
|---|---:|---|---|
| `_LLM_MODELS` | 18 个 architecture、79 个具体模型名 | Wings 可识别的模型名和 architecture | 不表示 PD 已适配或已验收 |
| FusionOne AI 23.6.1 ARM 表 | 45 个文本生成数据行、35 个去重模型 | 指定模型、精度、卡型和 ARM 镜像的普通推理记录 | 不表示 vLLM-Ascend PD 状态 |
| FusionOne AI 23.6.1 X86 表 | 38 个文本生成数据行、36 个去重模型 | 指定模型、精度、卡型和 X86 镜像的普通推理记录 | 不表示 NIXL/Mooncake PD 状态 |
| `pd_config.json` | `default` + 5 个精确 architecture profile | Ascend external-lb large-EP 配方 | 不等于 79 个模型均有精确配方 |

产品表路径：

- [ARM 兼容性表](../../target/FusionOne%20AI%20模型兼容性-23.6.1%20-ARM-0730.xlsx)
- [X86 兼容性表](../../target/FusionOne%20AI%20模型兼容性-23.6.1%20-X86-0730.xlsx)

ARM 表以 `wings-vllm-ascend:v0.21.0rc1*` 为主，另有特殊镜像和 MindIE 条目；X86 表以 `wings-vllm:v0.23.0-cu130*` 为主，另有特殊镜像和 xtrtllm 条目。MindIE/xtrtllm 行不继承 vLLM/vLLM-Ascend PD 记录。

### 3.2 Wings PD 路径

| 平台/条件 | 当前 Connector 或 profile 选择 | 未满足时的行为 |
|---|---|---|
| NVIDIA/vLLM PD | `NixlConnector` | 由 vLLM/NIXL 运行时校验 |
| Ascend，当前角色 `DP=1` | `PD_CONNECTOR_TYPE`，默认 `MooncakeConnectorV1`；生成 P/D 全局 TP/DP 配置 | 保留 standalone PD，不读取 large-EP 模型 recipe |
| Ascend，当前角色 `DP>1` | 必须按模型 architecture 精确命中 `pd_config.json` | 抛出 `PD large-EP has no registered profile` |
| profile 有 A2/A3 分支 | 必须从 `hardware_info.json` 识别平台 | 无法识别时 fail-fast |

`pd_config.json` 中的 `default` 是模板数据，不是 large-EP 通用回退。当前 large-EP 代码没有“未命中精确 profile 后使用 `default`”的路径。

### 3.3 当前 5 个 large-EP profile

| 本地 architecture | 本地 Connector | profile 来源 | 机械命中的模型数 | 精确差异记录 |
|---|---|---|---:|---|
| `Qwen3MoeForCausalLM` | `MooncakeLayerwiseConnector` | 用户自定义 Qwen3-30B-A3B 3P1D | 2 | 会同时命中 Qwen3-235B；235B 当前官方方案是 V1、P DP2 TP8、D DP8 TP4 |
| `DeepseekV32ForCausalLM` | `MooncakeLayerwiseConnector` | DeepSeek-V3.2 教程 | 4 | 当前官方页面命令使用 V1；Exp 权重没有单独登记 |
| `GlmMoeDsaForCausalLM` | `MooncakeConnectorV1` | 固定 v0.23.0 GLM-5/5.1 A3 2P4D | 9 | 会同时命中 GLM-5.2；5.2 的 A3/A2 拓扑和 A2 Connector 组合不同 |
| `Qwen3_5MoeForConditionalGeneration` | `MooncakeLayerwiseConnector` | Qwen3.5-397B | 8 | 当前官方 397B 命令使用 V1；同条目还命中 35B/122B、Qwen3.6-35B、AgentWorld 和 NVIDIA NVFP4 |
| `DeepseekV4ForCausalLM` | `MooncakeHybridConnector` | 固定 v0.23.0 V4-Flash | 6 | Connector 方向与 Flash/Pro 一致；Flash、Pro、基础 V4 的硬件和参数没有拆成独立权重级 profile |

5 个 architecture profile 机械命中 29 个模型名，另外 50 个模型名没有 large-EP 精确 profile。机械命中数量只表示查表结果，不表示模型级方案等价。

## 4. vLLM / NVIDIA PD 能力清单

### 4.1 当前 Wings 使用的 NIXL 路径

| 架构类型 | 基础 PD | 推测解码 | P/D 异构 TP | Cross-layer blocks | 异构 block size |
|---|---:|---:|---:|---:|---:|
| Dense Transformer | 支持 | 有条件 | 支持 | 有条件 | 部分支持 |
| MLA | 支持 | 有条件 | 部分支持 | 有条件 | 部分支持 |
| Sparse MLA | 支持 | 有条件 | 部分支持 | 有条件 | 部分支持 |
| Hybrid SSM/Mamba | 支持 | 未确认 | 开发中；当前 P TP = D TP | 不支持 | 不支持 |
| MoE | 支持 | 有条件 | 支持 | 有条件 | 部分支持 |
| Multimodal | 未验证 | 未验证 | 未验证 | 未验证 | 未验证 |
| Encoder-Decoder | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |

NIXL 握手至少要求 P/D 对齐：vLLM 版本、NIXL Connector 版本、模型结构、dtype、KV head/head size/layer 数、Attention backend、KV cache dtype、EAGLE/MTP 和 draft-model 配置。

NIXL 运行记录还需包含：

- `UCX_TLS`、`UCX_NET_DEVICES` 或所选 LIBFABRIC 插件；
- 每个本机 worker 唯一的 `VLLM_NIXL_SIDE_CHANNEL_PORT`；
- 跨机可路由的 `VLLM_NIXL_SIDE_CHANNEL_HOST`；
- Hybrid/Mamba 的 `VLLM_SSM_CONV_STATE_LAYOUT`；
- 多轮双向 KV 所需的有状态 Proxy 和上一轮 `kv_transfer_params`。

### 4.2 NVIDIA 官方方案是否基于 Mooncake

结论是：**vLLM 上游同时提供 `NixlConnector` 和 `MooncakeConnector`，但 NVIDIA Dynamo 当前公开的 vLLM 模型级 Recipe 主线使用 `NixlConnector`，不是 `MooncakeConnector`。** Dynamo 由 Frontend/Router 编排 P、D worker，通过 NIXL 将 KV 从 P 端 GPU VRAM 直接传到 D 端 GPU VRAM；跨机传输通常使用 UCX + InfiniBand/RoCE，GB 系列还可使用 MNNVL。vLLM 上游仍将通用 Disaggregated Prefilling 标记为 experimental，Dynamo Recipe 则通过固定模型、镜像、硬件和拓扑给出可复现边界。

NVIDIA 的 Qwen3-32B Recipe 中出现的 **Mooncake conversation trace** 只是压测数据集来源。该 Recipe 的 `deploy.yaml` 明确配置：

```json
{"kv_connector":"NixlConnector","kv_role":"kv_both"}
```

因此，不能根据压测数据名称把 NVIDIA 的 PD 实现归类为 Mooncake。

### 4.3 NVIDIA Dynamo 官方模型级 PD Recipe（按 Backend 分层）

审计基线固定为 2026-08-13 的 [`ai-dynamo/dynamo@22e80d2`](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes)。按 `recipes` 路径包含 `disagg`/`disaggregated` 的显式部署清单统计：

| Backend | PD `deploy*.yaml` 数 | 说明 |
|---|---:|---|
| vLLM | 28 | 另有 Qwen3-32B 的 5 个云厂商 overlay；当前 Wings X86 精确模型统计使用此 backend |
| SGLang | 11 | 包含 4 个 GLM manifest；GLM-5/5.2 不能因不在 vLLM 表中而被记为“官方无方案” |
| TensorRT-LLM | 12 | 包含 DeepSeek-V3.2、Kimi-K2.5、Qwen3-235B/32B 等；不等价于 Wings vLLM 支持 |

以上是 manifest 库存，不是去重模型数或产品支持数；同一模型的硬件、网络和云厂商变体分别计数。

#### 4.3.1 vLLM：当前兼容模型及必要对照项

下表按 Ascend 场景的同一粒度记录当前精确模型、容易误继承的同族权重，以及已单独要求核查的 Kimi-K3，包含“精确权重、官方配置、成熟度/证据、物理资源、P/D 拓扑、Connector 和 Router”。其他 backend 不混入 vLLM 的 4/36 分母。

模型名链接到实际权重页；Recipe 链接直接落到对应 vLLM Disagg 目录或部署清单。P/D 数量表示 worker 副本数，`TP/DP/EP` 表示单个 worker 的并行配置。Recipe 单元格中的 `Production-ready`、`Experimental`、`Functional` 和 Feature 分类沿用官方标记；“有 deploy”与“有 perf/benchmark”分别记录，不能把能部署自动等同于已完成性能验证。

| 模型/示例权重 | 官方 Recipe / 配置 | 官方硬件与物理形态 | P 全局拓扑 | D 全局拓扑 | KV Connector / 数据面 | Router | 关键边界 |
|---|---|---|---|---|---|---|---|
| [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) | [Disagg KV Router](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b/vllm/disagg-kv-router)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/vllm/disagg-kv-router/deploy.yaml)，[perf](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/vllm/disagg-kv-router/perf.yaml)）；已压测 Feature Recipe | 16×H200，2 节点 | 6P，每个 TP2 | 2D，每个 TP2 | `NixlConnector`；NIXL + RDMA | KV-aware | `max_model_len=131072`；Mooncake 仅为 conversation trace 数据集名 |
| [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) | [Cloud Provider Overlays](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b/vllm/cloud-providers)；Functional、未压测 | 8 GPU，1P1D；AKS IB、AWS EFA、GKE RoCE、Nebius IB、Nscale IB | 1P，TP4 | 1D，TP4 | `NixlConnector`；UCX 或 EFA/libfabric | 官方清单未显式启用 KV Router | 只能作为功能基线，不能替代性能验收；具体 GPU、RDMA 资源、镜像和环境变量以 provider overlay 为准 |
| [`Qwen/Qwen3-32B-FP8`](https://huggingface.co/Qwen/Qwen3-32B-FP8) | [vLLM Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b-fp8/vllm/disagg)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/vllm/disagg/deploy.yaml)，[perf](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/vllm/disagg/perf.yaml)）；Production-ready | 8×A100，单节点；文档也要求 H100/H200/A100 集群 | 2P，每个 TP2 | 1D，TP4 | `NixlConnector`；NIXL + UCX/RDMA | 官方清单未显式启用 KV Router | 全部 worker 同节点；FP8 KV；A100 40GB 配置限制 `max_model_len=8192` |
| [`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`](https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic) | [单节点 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/llama-3-70b/vllm/disagg-single-node)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/llama-3-70b/vllm/disagg-single-node/deploy.yaml)，[perf](https://github.com/ai-dynamo/dynamo/blob/main/recipes/llama-3-70b/vllm/disagg-single-node/perf.yaml)）；Production-ready | 8×H100/H200，单节点 | 2P，每个 TP2 | 1D，TP4 | `NixlConnector`；单节点 NIXL | 官方清单未显式启用 KV Router | Prefix Cache 关闭；block size 128；不是当前兼容表中的 Llama-3-8B |
| [`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`](https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic) | [多节点 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/llama-3-70b/vllm/disagg-multi-node)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/llama-3-70b/vllm/disagg-multi-node/deploy.yaml)，[perf](https://github.com/ai-dynamo/dynamo/blob/main/recipes/llama-3-70b/vllm/disagg-multi-node/perf.yaml)）；Production-ready | 16×H100/H200，2 节点 | 1P，TP8 | 1D，TP8 | `NixlConnector`；NIXL + 跨节点高速网络 | 官方清单未显式启用 KV Router | 每个 worker 独占 8 GPU；不能从 70B FP8 推导 8B BF16 已验证 |
| [`Qwen/Qwen3.5-122B-A10B-FP8`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) | [H200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3.5-122b/fp8/vllm/disagg-h200-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3.5-122b/fp8/vllm/disagg-h200-agentic/deploy.yaml)，[perf](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3.5-122b/fp8/perf/perf.yaml)）；Production-ready | 3×H200 | 1P，TP1 | 2D，每个 TP1 | `NixlConnector`；NIXL + IB/RDMA | KV-aware | FP8 KV；`VLLM_SSM_CONV_STATE_LAYOUT=DS`；PD 明确不启用 MTP；D 关闭 Async Scheduling |
| [`nvidia/Qwen3.5-122B-A10B-NVFP4`](https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4) | [B200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3.5-122b/nvfp4/vllm/disagg-b200-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3.5-122b/nvfp4/vllm/disagg-b200-agentic/deploy.yaml)，[perf](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3.5-122b/nvfp4/perf/perf.yaml)）；Production-ready | 3×B200 | 1P，TP1 | 2D，每个 TP1 | `NixlConnector`；NIXL + IB/RDMA | KV-aware | NVFP4 权重、FP8 KV；同样要求 DS conv state、无 MTP、D 关闭 Async Scheduling |
| [`deepseek-ai/DeepSeek-R1`](https://huggingface.co/deepseek-ai/DeepSeek-R1) | [vLLM Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-r1/vllm/disagg)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-r1/vllm/disagg/deploy_hopper_16gpu.yaml)）；Production-ready、部署完整、无 perf | 32×H100/H200，4 节点 | 1P，跨 2 节点×8 GPU，DP16+EP16+TP1 | 1D，跨 2 节点×8 GPU，DP16+EP16+TP1 | `NixlConnector`；NIXL + RDMA | 官方清单未显式启用 KV Router | DeepEP/NVSHMEM 的跨节点 DEP 另要求 IBGDA；IBGDA 不是 P→D KV Connector；该资源规模不能套当前 8 卡单机产品记录 |
| [`nvidia/DeepSeek-V4-Flash-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4) | [B200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-flash/vllm/disagg-b200-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4/deepseek-v4-flash/vllm/disagg-b200-agentic/deploy.yaml)）；Production-ready | 12×B200 | 2P，每个 TP4 | 1D，TP4 | `NixlConnector`；NIXL + UCX/GDR | KV-aware | FP8 KV、block 256、1M context；P/D 均无 MTP |
| [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | [H200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-flash/vllm/disagg-h200-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4/deepseek-v4-flash/vllm/disagg-h200-agentic/deploy.yaml)）；Production-ready | 28×H200 | 4P，每个 DP4+TP1+EP | 3D，每个 DP4+TP1+EP | `NixlConnector`；NIXL + UCX/GDR | KV-aware | 公共 FP8 权重、FP8 KV、1M context；P 无 MTP，D 使用 MTP-1 |
| [`nvidia/DeepSeek-V4-Pro-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4-Pro-NVFP4) | [B200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg-b200-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg-b200-agentic/deploy.yaml)）；Production-ready | 16×B200 | 1P，TP8+EP | 1D，TP8+EP | `NixlConnector`；NIXL + UCX/GDR | KV-aware | 1M context；P 无 MTP，D 使用 MTP-2；官方推荐 B200 使用该 Disagg 方案 |
| [`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | [H200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg-h200-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg-h200-agentic/deploy.yaml)）；Production-ready、best-effort lane | 32×H200 | 1P，TP8+EP | 3D，每个 TP8+EP | `NixlConnector`；NIXL + UCX/GDR | KV-aware | 公共 FP8 权重；`max_model_len=86016`；官方性能结果中 H200 聚合部署优于该 1P3D 方案 |
| [`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | [GB200 Experimental Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg/gb200)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg/gb200/deploy.yaml)）；Experimental、部署完整、无 perf | 16×GB200，4 个 NVL4 节点 | 1P，跨 2 节点，每节点 4 GPU，DP8+TP1+EP | 1D，跨 2 节点，每节点 4 GPU，DP8+TP1+EP | `NixlConnector`；KV bulk 走 MNNVL/`cuda_ipc`，UCX TCP 仅作 active-message 控制面 | 官方清单未显式启用 KV Router | `kv_role=kv_both`；DRA `ComputeDomain`；自定义镜像；`max_model_len=9280`；实验方案不能按 Production-ready 对外承诺 |
| [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) | [GB300 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/kimi-k3/vllm/disagg-gb300-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3/vllm/disagg-gb300-agentic/deploy.yaml)）；Production-ready、部署完整、无 perf | 24×GB300，6 节点 | 1P，TP8，跨 2 节点 | 2D，每个 TP8，跨 2 节点 | `NixlConnector`；NIXL over MNNVL | KV-aware | MXFP4 expert + BF16 dense + FP8 KV；DRA `ComputeDomain`；DS conv state |
| [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) | [GB200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/kimi-k3/vllm/disagg-gb200-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3/vllm/disagg-gb200-agentic/deploy.yaml)）；Production-ready、部署完整、无 perf | 32×GB200，8 节点 | 1P，TP16，跨 4 节点 | 1D，TP16，跨 4 节点 | `NixlConnector`；NIXL over UCX/RDMA | KV-aware | DRA `ComputeDomain`；MNNVL/NVLS 用于并行通信，KV 走 RDMA；`UCX_TLS=^cuda_ipc`；DS conv state |

“官方清单未显式启用 KV Router”只表示对应 manifest 没有设置 `--router-mode kv`，不表示该模型不能使用 KV-aware Router，也不影响固定 1P1D/普通 Frontend 路由下的基础 PD 数据传输。

#### 4.3.2 不纳入当前兼容模型分母的官方 vLLM PD 清单

下列 **13 个 deploy manifest** 属于 Dynamo 官方 vLLM PD 库存，但其模型既不在当前 Wings X86 36 个去重文本模型中，也未被选作当前同族/Kimi 对照项，因此不进入 4/36 的分母。它们用于说明官方全量边界，不能反向证明当前产品已经兼容。

| 官方模型族 | deploy manifest 数 | 官方 P/D 与硬件组合 | Connector / 组网重点 | 未纳入当前兼容分母的原因 |
|---|---:|---|---|---|
| [`GPT-OSS-120B`](https://github.com/ai-dynamo/dynamo/tree/main/recipes/gpt-oss-120b/vllm) | 2 | B200：2P6D、各 TP1；H200：4P4D、各 TP1；同节点 NVLink | `NixlConnector`；KV-aware Router；EAGLE3 | 当前 Wings 兼容表没有该模型 ID |
| [`Nemotron-3-Ultra`](https://github.com/ai-dynamo/dynamo/tree/main/recipes/nemotron-3-ultra/vllm) | 1 | B200：1P1D、各 TP4 | `NixlConnector`；IB/RDMA；KV-aware Router | 当前 Wings 兼容表没有该模型 ID |
| [`Nemotron-3.5-Lightning`](https://github.com/ai-dynamo/dynamo/tree/main/recipes/nemotron-3.5-lightning/vllm) | 9 | H100/H200 的 MTP、DFlash、DSpark；B200 的 DSpark；GB200 的 DFlash、DSpark；各变体均为 1P1D | H100 EFA/libfabric；H200 UCX IB/RDMA；B200/GB200 依具体 manifest | 当前 Wings 兼容表没有该模型 ID；该族已进入仓库 tree，但未列入根 `recipes/README.md` 汇总表 |
| [`Qwen3-VL-32B-Instruct-FP8`](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-vl-32b-fp8/vllm/hetero_hardware_disagg) | 1 | 1 个 Intel XPU Encode + 1 个 NVIDIA GPU Decode | NIXL/RDMA 传 embedding；Kubernetes DRA resource claim | 当前兼容表没有该多模态权重；异构 Encode/Decode 也不是文本模型 KV-only PD 的等价拓扑 |

另有 Llama-3.3-70B 的 1 个 GAIE 部署 manifest；它是同一模型、同一 1P1D TP4 资源形态的 GAIE 集成变体，不作为新的模型/拓扑 Recipe 重复计入上表。由此可核对：当前兼容相关 14 个 + 兼容表外 13 个 + GAIE 1 个 = 固定审计版本中的 28 个 vLLM PD deploy manifest。

#### 4.3.3 GLM：官方 SGLang PD 细粒度 Recipe

GLM 在 NVIDIA Dynamo 中不是 vLLM Recipe，而是 SGLang Recipe。以下 4 个 manifest 均使用 SGLang 的 `--disaggregation-transfer-backend nixl`；它与 vLLM 的 `NixlConnector` 属于同一 NIXL 数据传输方向，但配置入口、worker 生命周期、bootstrap 和 Router 接口不同，不能互换启动参数。

| 模型/权重 | 官方 Recipe / 证据 | 官方硬件与物理形态 | P 全局拓扑 | D 全局拓扑 | NIXL 数据面 / 组网 | Router / 缓存 | 对当前 Wings X86 表的结论 |
|---|---|---|---|---|---|---|---|
| [`nvidia/GLM-5-NVFP4`](https://huggingface.co/nvidia/GLM-5-NVFP4) | [GB200 UCX Disagg](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5-nvfp4/sglang/disagg)（[deploy](https://github.com/ai-dynamo/dynamo/blob/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5-nvfp4/sglang/disagg/deploy.yaml)，[perf](https://github.com/ai-dynamo/dynamo/blob/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5-nvfp4/sglang/disagg/perf.yaml)）；`validated`、recommended | 20×GB200，5 个 4-GPU 节点，处于 NVL36/NVL72 domain | 1P×4 GPU；TP4/DP1/EP1 | 1D rank group 跨 4 节点×4 GPU；TP16/DP16/EP16 | NIXL + UCX；`cuda_copy,cuda_ipc,tcp`；MNNVL；DRA `ComputeDomain` 5 节点；共享 RWX PVC | manifest 未设置 `DYN_ROUTER_MODE=kv`；EAGLE；FP8 KV | 与 `ZhipuAI/GLM-5-FP8` 仅同模型族；namespace、量化、backend、GPU 和拓扑均不同 |
| [`nvidia/GLM-5-NVFP4`](https://huggingface.co/nvidia/GLM-5-NVFP4) | [GB200 AWS EFA Disagg](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5-nvfp4/sglang/disagg/efa)（[deploy](https://github.com/ai-dynamo/dynamo/blob/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5-nvfp4/sglang/disagg/efa/deploy.yaml)，[perf](https://github.com/ai-dynamo/dynamo/blob/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5-nvfp4/sglang/disagg/efa/perf.yaml)）；`validated`、非 recommended | 20×GB200；5×`p6e-gb200.36xlarge`；每节点 4 GPU+4 EFA NIC | 同上 | 同上 | NIXL `LIBFABRIC` + EFA RDMA；每 Pod 的 EFA 请求数与 GPU 数相同；privileged；DRA `ComputeDomain` | manifest 未设置 KV-aware Router；必须自建含 patched libfabric 的镜像，否则 UCX 可能回退 TCP | 同上；它是网络变体，不是第二个可泛化模型方案 |
| [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | [B200 agentic Disagg](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5.2/sglang/disagg-b200-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5.2/sglang/disagg-b200-agentic/deploy.yaml)，[benchmark](https://github.com/ai-dynamo/dynamo/blob/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5.2/perf/perf.yaml)）；`validated`、非 recommended | manifest 实际 20×B200：3 个 4-GPU P Pod + 1 个 8-GPU D Pod | 3P；每个 TP4/DP4/EP4、启用 DP attention | 1D×8 GPU；TP8/DP8；未显式设置 `ep-size` | NIXL + UCX `cuda_ipc,cuda_copy,rc`；动态选择 GPU-local IB；每 Pod `rdma/shared_ib:1` | KV-aware Router；P 发布 KV events；P 开 200 GiB HiCache；EAGLE；500K context | 当前 X86 表没有 GLM-5.2；不能给 GLM-5/5.1 继承 |
| [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) | [H200 agentic Disagg](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5.2/sglang/disagg-h200-agentic)（[deploy](https://github.com/ai-dynamo/dynamo/blob/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5.2/sglang/disagg-h200-agentic/deploy.yaml)，[benchmark](https://github.com/ai-dynamo/dynamo/blob/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5.2/perf/perf.yaml)）；`validated`、非 recommended | 16×H200；1 个 8-GPU P Pod + 1 个 8-GPU D Pod | 1P×8 GPU；TP8/DP1/EP8 | 1D×8 GPU；TP8/DP8/EP1 | NIXL + UCX `cuda_ipc,cuda_copy,rc`；动态选择 GPU-local IB；每 Pod `rdma/ib:8` | KV-aware Router；P 发布 KV events；EAGLE；250K context | 当前 X86 表没有 GLM-5.2；它也不是 GLM-5.1 Recipe |

GLM-5.2 的官方元数据存在一处需显式保留的差异：catalog YAML 把 B200 disagg 的 `hardware.count` 写成 12，但根 Recipe 表写 20×B200，实际 manifest 是 `3P×4 + 1D×8 = 20`，且 benchmark 摘要也写明 3P1D。本文以可执行 manifest 的 20 卡拓扑为准，不把 catalog 的 12 当成实际集群总卡数。

### 4.4 NVIDIA Recipe 与当前 X86 兼容表的逐模型结论

当前 X86 表有 38 个文本生成部署行、36 个去重模型。按“模型 ID → 权重/精度 → 产品卡型与数量 → P/D 拓扑”逐层比对：

| 当前产品模型或模型组 | NVIDIA Dynamo 官方 PD Recipe | 匹配等级 | 不能直接宣称支持的具体差异 |
|---|---|---|---|
| `Qwen/Qwen3-32B` | 有，16×H200 6P2D TP2；另有 8 GPU 1P1D Functional Overlay | 模型 ID 精确；产品组合不精确 | 产品记录为 BF16、4×NL02、单机；官方 PD 的 GPU 数、P/D 拓扑和网络均不同 |
| `deepseek-ai/DeepSeek-R1` | 有，32×H100/H200、4 节点、P/D 各 16 GPU、DEP16 | 模型 ID 精确；产品组合不精确 | 产品记录为 FP8、8×NH02、单机；资源规模和组网不等价 |
| `deepseek-ai/DeepSeek-V4-Flash` | 有，28×H200、4P3D、每 worker DP4+TP1+EP | 模型 ID 精确；产品组合不精确 | 产品记录为 FP4、8×NH02、单机；官方公共权重为 FP8，资源和拓扑不同 |
| `deepseek-ai/DeepSeek-V4-Pro` | 有，H200 Production-ready 为 32 卡 1P3D TP8+EP；另有 GB200 Experimental 16 卡 1P1D DP8+EP | 模型 ID 精确；产品组合不精确 | 产品记录为 FP4、8×NH02、单机；两套官方方案的卡型、卡数、节点和成熟度均不匹配，且官方更推荐 H200 聚合方案 |
| `nv-community/DeepSeek-V4-Flash-NVFP4` | 有同模型族 `nvidia/DeepSeek-V4-Flash-NVFP4`，12×B200、2P1D、TP4 | 近似权重，不是精确 ID/产品组合 | namespace、4×NRP0500 与 12×B200 均不同；需先证明权重等价，再重做拓扑和性能验证 |
| DeepSeek-Coder、R1-0528、R1 Distill、V3/V3.1/V3.2/Exp/0324 | 没有相同模型 ID 的 NVIDIA Dynamo vLLM Disagg Recipe；V3.2 官方 Dynamo 方案为 TensorRT-LLM | 无精确 vLLM Recipe | 不能把 R1、V4 或 TensorRT-LLM Recipe 横向继承到这些 vLLM 模型 |
| `Qwen/Qwen3-235B-A22B` | Dynamo 有 Qwen3-235B-A22B-FP8 Disagg，但 backend 是 TensorRT-LLM | vLLM 路径无精确 Recipe | 当前产品是 BF16、vLLM；backend、精度、硬件和拓扑均不同 |
| `nv-community/Qwen3.5-397B-A17B-NVFP4` | NVIDIA vLLM Recipe 为 122B-A10B FP8/NVFP4，不是 397B | 无精确 Recipe | 不能按同架构复用 122B 的 1P2D TP1、DS state 和无 MTP 结论 |
| `ZhipuAI/GLM-5-FP8` | 有同模型族 `nvidia/GLM-5-NVFP4` SGLang Recipe：20×GB200，P TP4、D TP16；UCX/MNNVL 和 EFA/LIBFABRIC 两种网络变体 | 同模型族；无精确权重/产品组合 | namespace、FP8/NVFP4、vLLM/SGLang、8×NH02/20×GB200 均不同；不能只替换模型路径 |
| `ZhipuAI/GLM-4.7-FP8`、`ZhipuAI/GLM-5.1-FP8`、`ZhipuAI/GLM-4-9B-0414` | 未找到相同模型 ID 的 NVIDIA Dynamo PD Recipe；GLM-5.2 的两套 SGLang Recipe 也不是这些模型 | 无精确 Recipe | 不能在 GLM-4.7/5/5.1 之间按架构继承权重、EAGLE、TP/DP/EP 和缓存参数 |
| Kimi-K2.6 | NVIDIA 当前 vLLM Recipe 为聚合部署；Kimi-K3 才有 vLLM Disagg | 无精确 PD Recipe | K2.6 与 K3 的模型结构、TP 和 Hybrid state 方案不同 |
| MiniMax-M2.5/M2.7/M3 | 未找到 NVIDIA Dynamo vLLM Disagg Recipe | 无精确 Recipe | 仅有普通推理兼容性或其他平台方案，不构成 NVIDIA vLLM PD 证据 |
| `LLM-Research/Meta-Llama-3-8B`、R1-Distill-Llama-8B/70B | NVIDIA Recipe 是 Llama-3.3-70B-Instruct-FP8-dynamic | 无精确权重 Recipe | 模型版本、参数量和精度不同；不能由 Dense 架构或 70B Recipe 推导 |
| Qwen2.5-32B、QwQ-32B、Qwen3-8B/14B/30B/Next/AgentWorld | 未找到相同模型 ID 的 NVIDIA Dynamo vLLM Disagg Recipe | 无精确 Recipe | 上游 Connector 通用示例或同族 Recipe 只能作为预研入口 |

量化后，36 个 X86 去重文本生成模型中：**4 个命中相同模型 ID 的 NVIDIA vLLM PD Recipe，32 个没有相同模型 ID；0 个与产品表中的“模型+精度+卡型/卡数+P/D 拓扑”完整一致。** GLM 新增的是 SGLang 同模型族证据：精确模型 ID 命中仍为 0，因此不改变 vLLM 的 4/36。这里的 0 表示没有可直接照抄的完整 Recipe，不表示 36 个模型都被社区明确禁止。

### 4.5 NVIDIA NIXL Recipe 的共同组网要求

| 层面 | 官方 Recipe 共同要求 | 缺失后的典型结果 |
|---|---|---|
| Connector | vLLM worker 使用 `NixlConnector`；SGLang 使用 `--disaggregation-transfer-backend nixl` 并通过环境变量选择 UCX/LIBFABRIC；两者都要求 P/D 模型、KV dtype 和 layout 对齐，但配置接口不能互换 | 参数被 backend 拒绝、bootstrap/握手失败、KV load 失败或静默错误输出 |
| 版本 | Recipe 使用其固定 Dynamo/vLLM/NIXL 镜像；当前 vLLM 已将 NIXL 的 `kv_role=kv_both` 标记为 deprecated，而部分 Dynamo Recipe 仍使用该值 | 把新旧参数直接混用会出现启动参数、握手或生命周期行为差异 |
| 数据面 | NIXL；UCX 为默认 backend，也可显式选择 LIBFABRIC | NIXL 插件缺失、只走 TCP 或传输性能不足 |
| 跨机网络 | InfiniBand、RoCE 或等价高速网络；Kubernetes 安装 RDMA device plugin | KV 传输成为 TTFT/吞吐瓶颈，或 worker 无法获得 RDMA 设备 |
| Pod 资源 | 根据 TP 申请 `rdma/ib`；容器增加 `IPC_LOCK`，部分 Recipe 还需要 `SYS_RESOURCE` | 内存注册失败、UCX/NIXL 初始化失败 |
| UCX | 配置 `UCX_TLS`、`UCX_NET_DEVICES`、`UCX_RNDV_SCHEME=get_zcopy`；不能只配 NCCL 网卡变量 | 选错 NIC、回退 host staging/TCP，或带宽明显不足 |
| 控制面 | Dynamo Frontend/PrefillRouter、服务发现；Kubernetes 生产方案依赖 ETCD/NATS | P/D worker 无法发现、路由或传递 `kv_transfer_params` |
| Side channel | 跨机地址可路由；同一主机的 worker side-channel 端口唯一 | 地址不可达或端口冲突 |
| GB200/GB300 | MNNVL 场景需要 VMM KV 注册、`UCX_CUDA_IPC_ENABLE_MNNVL=y`，多节点 Recipe 还依赖 DRA `ComputeDomain` | 无法使用跨节点 NVLink，回退 RDMA/TCP 或调度失败 |
| Hybrid/Mamba/GDN | Recipe 明确要求时设置 `VLLM_SSM_CONV_STATE_LAYOUT=DS`；Qwen3.5-122B PD 禁用 MTP，D 关闭 Async Scheduling | `3-read Mamba conv transfer requires DS conv state layout`、EngineCore 崩溃或静默错误答案 |

### 4.6 上游 vLLM Mooncake 路径

上游 vLLM 另有通用 `MooncakeConnector` 使用指南，官方示例为 Qwen2.5-7B-Instruct 的一个 P 实例和一个 D 实例，通过 Mooncake transfer engine 和 Proxy 传输 KV。该示例证明 Connector/API 和部署骨架存在，不是 Wings NVIDIA 路径的默认 Connector，也不是 79 个模型的逐模型生产配方。

官方来源：

- [NIXL Compatibility Matrix](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/)
- [NIXL Usage Guide](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [MooncakeConnector Usage Guide](https://docs.vllm.ai/en/stable/features/mooncake_connector_usage/)
- [NVIDIA Dynamo Disaggregated Serving Design](https://docs.nvidia.com/dynamo/latest/design-docs/disaggregated-serving)
- [NVIDIA Dynamo Disaggregated Serving Guide](https://docs.nvidia.com/dynamo/latest/user-guides/disaggregated-serving)
- [NVIDIA Dynamo Production-Ready Recipes](https://github.com/ai-dynamo/dynamo/blob/main/recipes/README.md)

具体模型权重、vLLM Disagg Recipe、`deploy.yaml` 和 `perf.yaml` 已在 4.3 表中逐行链接，不再在本节重复列举模型族根目录。

## 5. vLLM-Ascend 官方模型级组合

本节只记录官方页面中同时出现精确模型/权重、硬件或资源规模、P/D 拓扑和 Connector 的组合。模型页只有普通单机/多机命令但没有 PD 章节时，不登记为模型级 PD 配方。

| 模型/示例权重 | 官方硬件与物理形态 | P 全局拓扑 | D 全局拓扑 | Connector | 官方页面状态 | Wings 精确 profile |
|---|---|---|---|---|---|---|
| DeepSeek-R1-w8a8 | 4 台 Atlas 800T A3；通用多节点示例为 2 个 P 节点 + 2 个 D 节点 | DP2 TP8 | DP32 TP1 | Layerwise 和 V1 两套分支 | 通用 Mooncake 多节点教程 | 无 `DeepseekV3ForCausalLM` |
| DeepSeek-V3.1-w8a8-mtp-QuaRot | 4 台 Atlas 800 A3 64GB×16；页面标记 2P1D | DP2 TP8 | DP32 TP1 | 页面文字写 Layerwise，实际启动命令为 V1 | 官方页面内部存在标记/命令差异，原样记录 | 无 `DeepseekV3ForCausalLM` |
| DeepSeek-V3.2-w8a8-mtp-QuaRot | A3 多节点 | DP2 TP16 | DP8 TP4 | `MooncakeConnectorV1` | 当前模型页命令 | 有，当前本地为 Layerwise |
| DeepSeek-V4-Flash-w8a8-mtp，A3 | 2 台 A3 128GB×8；1P+1D | DP4 TP4 | DP16 TP1 | `MooncakeHybridConnector` | 模型专项 | 有架构 profile |
| DeepSeek-V4-Flash-w8a8-mtp，A2 | 8 台 A2 64GB×8；4P+4D | DP8 TP1 | DP32 TP1 | `MooncakeHybridConnector` | 模型专项 | 有 A2 overlay |
| DeepSeek-V4-Pro-w4a8-mtp，A3 | 4 台 A3 128GB×8；2P+2D | DP2 TP16 | DP16 TP2 | `MooncakeHybridConnector` | 模型专项 | 被 Flash 来源的架构 profile 命中 |
| DeepSeek-V4-Pro-w4a8-mtp，A2 | 8 台 A2 64GB×8；4P+4D | DP4 TP8 | DP8 TP4 | `MooncakeHybridConnector` | 模型专项 | 被 Flash 来源的架构 profile 命中 |
| GLM-4.7-W8A8-floatmtp | 4 台 Atlas 800 A3；2P+2D | DP2 TP8 | DP8 TP4 | `MooncakeConnectorV1` | main / 0.14.0rc1 后模型页 | 无 `Glm4MoeForCausalLM` |
| GLM-5/5.1-w8a8 | 6 台 A3；2P+4D | DP2 TP16 | DP16 TP4 | `MooncakeConnectorV1` | 模型专项 | 有 `GlmMoeDsaForCausalLM` |
| GLM-5.2-w4a8c8，A3 | 4 台 A3；2P+2D | DP4 TP8 | DP32 TP1 | `MooncakeConnectorV1` | 模型专项 | 无 5.2 独立 profile |
| GLM-5.2-w4a8c8，A2 | 8 台 A2；4P+4D | DP4 TP8 | DP8 TP4 | `MultiConnector`：V1 + AscendStore | 模型专项 | 无 5.2 独立 profile |
| Qwen3-235B-A22B-w8a8-rot | 3 台 A3；1P+2D | DP2 TP8 | DP8 TP4 | `MooncakeConnectorV1` | 模型专项 | 被 Qwen3-30 Layerwise profile 命中 |
| Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | 2 台 A3；1P+1D | DP8 TP2 | DP8 TP2 | `MooncakeConnectorV1` | 模型专项；Atlas 300I DUO 不适用该多节点方案 | 无 `Qwen3_5ForConditionalGeneration` |
| Qwen3.5-397B-A17B-w8a8 | 3 台 A3 64GB×16；1P+2D | DP8 TP2 | DP16 TP2 | `MooncakeConnectorV1`；D 设置 `kv_buffer_device=npu` | 模型专项；D Prefix Cache 有页面已知限制 | 有，当前本地为 Layerwise |
| Kimi-K2.5-w4a8 | 4 台 A3；页面推荐 2P1D 资源形态 | DP2 TP8 | DP32 TP1 | `MooncakeConnectorV1` | 模型专项 | 无 `KimiK25ForConditionalGeneration` |
| Kimi-K2.6-w4a8 | 4 台 A3；页面推荐 2P1D 资源形态 | DP4 TP4 | DP8 TP4 | `MooncakeConnectorV1` | 模型专项 | 无 `KimiK25ForConditionalGeneration` |
| MiniMax-M2.7-w8a8-QuaRot | 2 台 A3 64GB×16；1P+1D | DP2 TP8 | DP2 TP8 | `MooncakeConnectorV1` | 模型专项 | 无 `MiniMaxM2ForCausalLM` |

官方来源：

- [PD Disaggregation (Mooncake, Multi Node)](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)
- [DeepSeek-V3.1](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.1.html)
- [DeepSeek-V3.2](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.2.html)
- [DeepSeek-V4-Flash](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/DeepSeek-V4-Flash.html)
- [DeepSeek-V4-Pro](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Pro.html)
- [GLM-4.x](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM4.x.html)
- [GLM-5/5.1](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html)
- [GLM-5.2](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html)
- [Qwen3-235B-A22B](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-235B-A22B.html)
- [Qwen3.5-27B/Qwen3.6-27B](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html)
- [Qwen3.5-397B-A17B](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-397B-A17B.html)
- [Kimi-K2.5](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Kimi-K2.5.html)
- [Kimi-K2.6](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K2.6.html)
- [MiniMax-M2](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/MiniMax-M2.html)

## 6. 79 个 Wings LLM 模型覆盖清单

“vLLM/NIXL”列优先登记 NVIDIA Dynamo 精确模型 Recipe；没有精确 Recipe 时才登记架构级能力。“Ascend 模型级配方”只登记当前找到的精确教程或同模型族教程；“本地 large-EP”只登记 profile 查表结果。

### 6.1 DeepSeek

| Architecture | Wings 模型名 | vLLM/NIXL | Ascend 模型级配方 | 本地 large-EP | 差异/缺项 |
|---|---|---|---|---|---|
| `DeepseekV2ForCausalLM` | DeepSeek-Coder-V2-Instruct；DeepSeek-Coder-V2-Instruct-w8a8 | MLA 架构级 | 未找到精确配方 | 无 | 精确模型 recipe、profile、物料和真机记录缺失 |
| `DeepseekV3ForCausalLM` | DeepSeek-R1；DeepSeek-R1-0528；DeepSeek-V3；DeepSeek-V3-0324；DeepSeek-V3.1；以及对应 5 个 w8a8 名称 | NVIDIA 对 `deepseek-ai/DeepSeek-R1` 有 32×Hopper、P/D 各 16 GPU 的 vLLM NIXL Recipe；其余仅架构级 | R1 有通用多节点示例；V3.1 有模型页配方；其余精确权重未逐一登记 | 无 | R1 的 NVIDIA Recipe 与当前 8 卡单机产品组合不等价；V3/V3.1/0528 仍需独立核对 |
| `DeepseekV32ForCausalLM` | DeepSeek-V3.2；DeepSeek-V3.2-w8a8；DeepSeek-V3.2-Exp；DeepSeek-V3.2-Exp-W8A8 | Sparse MLA 架构级 | V3.2-w8a8 有模型配方；Exp 未找到独立配方 | 有，Layerwise | 当前官方 V3.2 命令为 V1；Exp 不能自动继承 |
| `DeepseekV4ForCausalLM` | DeepSeek-V4；DeepSeek-V4-w8a8；DeepSeek-V4-Flash；DeepSeek-V4-Flash-w8a8-mtp；DeepSeek-V4-Pro；DeepSeek-V4-Pro-w4a8-mtp | NVIDIA 对 Flash/Pro 的公共 FP8 和 NVFP4 权重均有 NIXL Recipe；基础 V4 仍是架构级 | Flash/Pro 精确量化权重有配方；基础 V4 未找到同等配方 | 有一个 Flash 来源的架构 profile | NVIDIA/Ascend 均要求按 Flash/Pro、权重和硬件拆分；不能用一个架构 profile 覆盖 |

### 6.2 Kimi

| Architecture | Wings 模型名 | vLLM/NIXL | Ascend 模型级配方 | 本地 large-EP | 差异/缺项 |
|---|---|---|---|---|---|
| `KimiK25ForConditionalGeneration` | Kimi-K2.5；Kimi-K2.5-w4a8；Kimi-K2.6-w4a8；Kimi-K2.7；Kimi-K2.7-w4a8；Kimi-K2.7-Code | 完整多模态在 NIXL 矩阵中未验证 | K2.5-w4a8、K2.6-w4a8 有；K2.7/Code 未找到 | 无 | K2.5/K2.6 profile 缺失；K2.7/Code 模型级配方缺失 |
| `KimiK3ForConditionalGeneration` | Kimi-K3 | NVIDIA 有 GB300 1P2D TP8 和 GB200 1P1D TP16 的 vLLM NIXL Recipe | 未找到 | 无 | NVIDIA Recipe 依赖 MNNVL/DRA ComputeDomain、RDMA 和 DS conv state；当前产品未登记同等组合 |

### 6.3 GLM

| Architecture | Wings 模型名 | vLLM/NIXL | Ascend 模型级配方 | 本地 large-EP | 差异/缺项 |
|---|---|---|---|---|---|
| `Glm4ForCausalLM` | GLM-4-9B-0414 | Dense 架构级 | 未找到精确配方 | 无 | 精确 recipe 和 profile 缺失 |
| `Glm4MoeForCausalLM` | GLM-4.7；GLM-4.7-FP8；GLM4.7；GLM-4.7-w8a8 | MoE 架构级 | `Eco-Tech/GLM-4.7-W8A8-floatmtp` 有当前 main 配方 | 无 | 官方精确权重名与 Wings 名称需映射；profile 缺失 |
| `GlmMoeDsaForCausalLM` | GLM-5；GLM-5-FP8；GLM-5-w4a8；GLM-5.1；GLM5.1；GLM-5.1-w8a8；GLM-5.1-FP8 | MLA/MoE 架构级 | GLM-5/5.1 有，主要为 w8a8 | 有 GLM-5/5.1 profile | FP8/w4a8 与 w8a8 分项记录 |
| `GlmMoeDsaForCausalLM` | GLM-5.2；GLM-5.2-w8a8 | MLA/MoE 架构级 | GLM-5.2 w4a8c8 有 A3/A2 两套 | 会命中 GLM-5/5.1 profile | 精度、拓扑和 A2 MultiConnector 均不等价 |

### 6.4 Qwen

| Architecture | Wings 模型名 | vLLM/NIXL | Ascend 模型级配方 | 本地 large-EP | 差异/缺项 |
|---|---|---|---|---|---|
| `Qwen2ForCausalLM` | DeepSeek-R1-Distill-Qwen-1.5B/7B/14B/32B；Qwen2.5-32B-Instruct；QwQ-32B | Dense 架构级 | 未找到这些精确模型配方 | 无 | 精确 recipe 和 profile 缺失 |
| `Qwen3ForCausalLM` | Qwen3-32B | NVIDIA 有 16×H200 6P2D TP2 压测 Recipe，以及 8 GPU 1P1D Functional Overlay，均使用 NIXL | 未找到精确配方 | 无 | 模型 ID 命中，但当前 4×NL02 单机产品组合与官方资源/拓扑不等价 |
| `Qwen3MoeForCausalLM` | Qwen3-30B-A3B | MoE 架构级 | 当前模型页未找到 PD 章节 | 有用户自定义 Layerwise | 内部配方；非官方模型级配方 |
| `Qwen3MoeForCausalLM` | Qwen3-235B-A22B | MoE 架构级 | w8a8-rot 有 V1 配方 | 会命中 Qwen3-30 Layerwise | 权重、Connector 和拓扑不等价 |
| `Qwen3NextForCausalLM` | Qwen3-Next-80B-A3B-Instruct | Hybrid SSM/Mamba 架构级、受 Hybrid 限制 | 当前模型页未找到 PD 章节 | 无 | 模型级 recipe 和 profile 缺失 |
| `Qwen3_5ForConditionalGeneration` | Qwen3.5-27B；Qwen3.5-27B-w8a8；Qwen3.6-27B；Qwen3.6-27B-w8a8 | Hybrid SSM/Mamba；当前 P TP = D TP | w8a8 有 2 台 A3 配方 | 无 | profile、镜像物料和真机记录缺失 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5-397B-A17B-NVFP4 | Hybrid/MoE，NVIDIA 权重 | Ascend w8a8 配方不适用 | 会命中 Ascend Layerwise | 平台/精度隔离缺失 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5-397B-A17B-w8a8；Qwen3.5-397B-A17B | Hybrid/MoE 架构级 | w8a8 有 V1 配方 | 有，Layerwise | Connector 漂移；BF16 与 w8a8 分项记录 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5-35B-A3B；Qwen3.5-122B-A10B | NVIDIA 对 122B-A10B 的 FP8/H200 和 NVFP4/B200 各有 1P2D TP1 NIXL Recipe；35B 仍仅架构级 | 未找到精确配方 | 会命中 397B profile | 122B BF16 产品权重与官方量化权重不等价；397B profile 也不能作为精确配方 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.6-35B-A3B；Qwen3.6-35B-A3B-w8a8 | Hybrid/MoE 架构级 | 当前模型页只有普通部署，未找到 PD 章节 | 会命中 397B profile | 模型级 recipe 缺失；已有请求期 Hybrid KV 故障记录 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen-AgentWorld-35B-A3B | Hybrid/MoE 架构级 | 未找到精确配方 | 会命中 397B profile | 精确 recipe 缺失，架构 profile 超覆盖 |

### 6.5 MiniMax 与 Llama

| Architecture | Wings 模型名 | vLLM/NIXL | Ascend 模型级配方 | 本地 large-EP | 差异/缺项 |
|---|---|---|---|---|---|
| `MiniMaxM2ForCausalLM` | MiniMax-M2.5；MiniMax-M2.5-NVFP4；MiniMax-M2.5-w8a8 | MoE 架构级；NVFP4 为 NVIDIA 路径 | 未找到 M2.5 精确配方 | 无 | 平台分流、精确 recipe 和 profile 缺失 |
| `MiniMaxM2ForCausalLM` | MiniMax-M2.7；MiniMax-M2.7-w8a8；MiniMax-M2.7-w8a8-QuaRot | MoE 架构级 | w8a8-QuaRot 有配方 | 无 | profile、镜像物料和真机记录缺失 |
| `MiniMaxM3SparseForConditionalGeneration` | MiniMax-M3；MiniMax-M3-MXFP8 | 多模态边界未验证 | 有普通部署资料，未找到具体 PD 启动章节 | 无 | 精确 PD recipe 和 profile 缺失 |
| `LlamaForCausalLM` | LLaMA3-8B；LLaMA3.1-70B；LLaMA3.1-70B-Instruct；Meta-Llama-3.1-70B-Instruct；DeepSeek-R1-Distill-Llama-8B/70B | NVIDIA 有 `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` 的单节点 2P1D 和多节点 1P1D NIXL Recipe，但不是这些精确权重 | 未找到这些精确权重的 vLLM-Ascend 配方 | 无 | 不能跨模型版本、参数量和精度继承 NVIDIA 70B Recipe；Ascend 精确 recipe 也缺失 |

## 7. 本地 profile 与官方配方差异清单

| 本地 profile | 本地记录 | 当前官方记录 | 未对齐项 |
|---|---|---|---|
| `Qwen3MoeForCausalLM` | Qwen3-30B-A3B；Layerwise；用户自定义 3P1D；D 开 Prefix Cache 和异步调度 | Qwen3-30 当前页无 PD；Qwen3-235 使用 V1、P DP2 TP8、D DP8 TP4 | 同 architecture 跨模型复用；Connector、Proxy 和拓扑不同 |
| `DeepseekV32ForCausalLM` | Layerwise；`max_model_len=68000`；P 大 batch、D 小 batch | 当前 V3.2 页面命令为 V1；P DP2 TP16、D DP8 TP4 | Connector、Proxy/Metaserver 和请求顺序 |
| `GlmMoeDsaForCausalLM` | V1；固定 v0.23.0；P DP2 TP16、D DP16 TP4；D 显存 0.88；MTP=1；关闭 FUSED MC2 | GLM-5/5.1 拓扑同源，当前页面显存/MTP/通信环境值有差异；GLM-5.2 是独立 A3/A2 方案 | 5/5.1 的参数漂移；5.2 架构超覆盖 |
| `Qwen3_5MoeForConditionalGeneration` | 397B 来源；Layerwise；会命中 35B/122B、Qwen3.6、AgentWorld、NVFP4 | 397B 当前为 V1，P DP8 TP2、D DP16 TP2，D 关闭 Prefix Cache | Connector 漂移、模型/平台/精度超覆盖 |
| `DeepseekV4ForCausalLM` | Hybrid；A2/A3 overlay；P/D 关闭异步调度 | Flash 当前 D 示例含异步调度；Pro 有独立 A3/A2 拓扑；DSpark/MTP 参数分支不同 | Flash/Pro/基础 V4 未拆分；调度和推测参数差异 |

## 8. Ascend 硬件与组网清单

### 8.1 每个模型组合必须记录的硬件字段

| 类别 | 字段 |
|---|---|
| 卡与节点 | NPU 型号、单卡显存、每节点卡数、P 节点数、D 节点数 |
| 并行拓扑 | P/D 全局 DP、TP、EP；本地 DP；DP rank start |
| 权重 | 模型路径、精度/量化、MTP/DSpark draft 权重 |
| 容量 | 最大上下文、`max_num_batched_tokens`、`max_num_seqs`、显存比例 |
| KV | Connector、role、engine ID、KV port、extra config、KV dtype、block size |
| 调度 | Prefix Cache、Chunked Prefill、Async Scheduling、Hybrid KV manager |
| 版本 | 镜像、vLLM、vLLM-Ascend、CANN、torch、torch_npu、Mooncake |

### 8.2 物理网络

- 所有节点处于可路由网络；
- 节点内 NPU 通过 HCCS 互联；
- 节点间使用 RDMA；
- 容器能读取 `/etc/hccn.conf`；
- HCCN 链路、端口和网络健康检查为成功状态。

### 8.3 控制面与通信变量

| 类型 | 必填项 |
|---|---|
| 网卡 | `HCCL_IF_IP`、`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME` |
| DP master | `--data-parallel-address`、`--data-parallel-rpc-port` |
| rank | `--data-parallel-size`、本地 DP、`--data-parallel-start-rank`；非 master 使用 `--headless` |
| 设备 | `ASCEND_RT_VISIBLE_DEVICES` |
| Mooncake | KV port、bootstrap port、每实例唯一 engine ID / `PD_INDEX` |
| P/D 四元组 | `PD_PREFILL_TP_SIZE`、`PD_PREFILL_DP_SIZE`、`PD_DECODE_TP_SIZE`、`PD_DECODE_DP_SIZE` |

P/D 两侧必须看到相同的全局四元组，不能各自按本 Pod 卡数独立推导。不规则节点分配需要显式累计 rank start。

### 8.4 端口范围

AscendDirectTransport 随机端口范围：

- 8 NPU/节点：`20000-27999`，Mooncake `kv_port` 建议从 `28000` 以上规划；
- 16 NPU/节点：`20000-35999`，Mooncake `kv_port` 建议从 `36000` 以上规划。

还需分别登记 API 端口、DP RPC 端口、Proxy/Metaserver 端口和 Mooncake bootstrap 端口，避免同机实例复用。

## 9. 物料和版本字段

| 层次 | 核对项 | 典型失败证据 |
|---|---|---|
| Python 包 | vLLM、vLLM-Ascend、Mooncake、NIXL | Connector import/注册失败 |
| 框架 | torch、torch_npu、CUDA/CANN | 算子、ABI 或设备初始化失败 |
| Mooncake 主库 | `libtransfer_engine.so` | `cannot open shared object file` |
| Ascend transport | `ascend_transport.so` | `ldd` 显示 `not found` |
| 动态链接 | `LD_LIBRARY_PATH`、`ldconfig`、`LD_PRELOAD`、jemalloc | import 前失败或退出时原生堆损坏 |
| NIXL transport | NIXL 版本、UCX/LIBFABRIC 插件、网卡 | 握手、side-channel 或数据传输失败 |
| 官方/业务镜像 | 官方 tag、业务 tag、安装层差异 | 同命令在两镜像表现不同 |

## 10. 历史故障证据清单

| 场景 | 直接证据 | 发生阶段 | 记录分类 |
|---|---|---|---|
| Ascend 1P1D TP 冲突 | `Expected 2, but got 1`；Engine TP=2，KV `decode.tp_size=1` | KVTransferConfig 校验，传输前 | Wings 拓扑生成 |
| Qwen3.6 关闭 HMA | 多种 KV spec 无法统一 | KV cache 初始化 | 社区版本/Hybrid KV 能力边界 |
| Qwen3.6 打开 HMA | P/D 均启动，请求长期无返回 | 请求期 | Hybrid KV 传输/完成通知/调度闭环，根因未定 |
| Mooncake 主库不可见 | `libtransfer_engine.so: cannot open shared object file` | import | 镜像动态链接路径 |
| Ascend transport 不可见 | `ascend_transport.so => not found` | 原生依赖解析 | 镜像运行环境 |
| import 成功后堆损坏 | `corrupted size vs. prev_size` | 解释器退出/原生析构 | ABI、allocator 或原生库组合 |
| 首请求成功、第二请求崩溃 | `start_load_kv(): assert self.kv_recv_thread is not None` | 请求期状态复用 | Connector 生命周期、Prefix Cache 或 block 释放，根因未定 |
| NVIDIA Qwen3.5 Mamba state | 要求 `VLLM_SSM_CONV_STATE_LAYOUT=DS` | NIXL Hybrid state 传输 | 上游布局约束 + Wings 环境未注入 |
| NVIDIA PD 未下发 TP | 2 卡 Pod 最终命令无 TP，vLLM 默认 TP=1 | 最终命令生成 | 平台/Wings 拓扑输入 |
| large-EP 无 profile | `PD large-EP has no registered profile` | profile 选择 | Wings 精确配方缺失 |
| 只有一侧 DP>1 | 一侧命中 Layerwise，另一侧保留 standalone V1 | profile 选择 | Wings 按当前角色 DP 选择全局方案 |
| TP/DP 非法或超订 | TP=0、DP=0、4 卡 Pod TP=8 仍可能进入命令 | 启动前参数校验 | Wings fail-fast 缺口 |

## 11. 缺项分类字典

此表用于给每个模型/场景打标签；一个条目可以同时具有多个标签。

| 标签 | 判定依据 | 典型条目/证据 |
|---|---|---|
| `UPSTREAM-HARD-LIMIT` | 官方矩阵明确不支持，或官方同版本同命令稳定复现上游错误 | NIXL Encoder-Decoder；Hybrid Mamba 异构 TP/block 限制 |
| `UPSTREAM-NO-MODEL-RECIPE` | 有架构级能力，但未找到精确模型/权重/硬件 PD 命令 | DeepSeek-Coder-V2、Qwen3-30B、Qwen3-Next、Kimi-K2.7 等 |
| `WINGS-NO-PROFILE` | 官方有模型配方，本地无精确 large-EP profile | DeepSeek-R1/V3.1、GLM-4.7、Qwen3.5/3.6-27B、Kimi-K2.5/K2.6、MiniMax-M2.7 |
| `WINGS-PROFILE-DRIFT` | 本地 Connector、拓扑或参数与选定官方版本不同 | V3.2、Qwen3.5-397B |
| `WINGS-ARCH-OVERMATCH` | 同 architecture profile 命中不同模型/精度，但官方组合不同 | GLM-5→5.2、Qwen3-30→235、397B→35B/122B/AgentWorld |
| `MATERIAL-VERSION` | 官方实现位于不同 vLLM/vLLM-Ascend/CANN/torch_npu/Mooncake 版本 | 产品镜像版本早于教程版本 |
| `MATERIAL-PACKAGING` | Python 包或 `.so` 缺失、`ldd` 未解析 | Mooncake import/动态库失败 |
| `NETWORK-ORCHESTRATION` | IP、RDMA/HCCS、网卡、端口、rank、Proxy 或请求顺序不一致 | 端口冲突、rank 重叠、V1/Layerwise Proxy 不匹配 |
| `RUNTIME-REQUEST` | 启动完成但单请求/连续请求/并发失败 | 请求挂起、第二请求崩溃 |
| `VALIDATION-RESOURCE` | 官方要求的硬件规模当前未复现 | 48 张 A3、6 节点 GLM-5、8 节点 GLM-5.2 A2 等 |

`VALIDATION-RESOURCE` 只记录“同等硬件未验”，不替代社区能力、Wings profile 或物料状态。

## 12. 验证记录清单

| 等级 | 必须记录的检查 | 通过证据 |
|---|---|---|
| L0 注册 | 精确模型、权重、平台和 Connector 命中 | 不使用错误架构 profile |
| L1 静态命令 | P/D 最终命令、环境、端口、rank 与选定官方组合逐项比对 | 差异有来源和边界 |
| L2 启动 | 所有 P/D engine、Proxy/Metaserver 启动 | 无配置、模型加载、HCCL、Mooncake/NIXL 错误 |
| L3 单请求 | 通过 Proxy 完成一次请求 | HTTP 成功、输出正确、P/D KV 日志完整 |
| L4 连续请求 | 相同/不同前缀、第二请求、多轮 | 无线程丢失、block 泄漏、崩溃或挂起 |
| L5 并发和长上下文 | 多并发、长输入、不同输出长度、取消请求 | 无 OOM、超时、端口/lease 问题；记录 TTFT/TPOT |
| L6 精度和稳定性 | 精度集、长稳、异常恢复 | 精度无回退、长稳无泄漏、失败语义明确 |

## 13. 归档表固定字段

### 13.1 全量模型覆盖表

```text
模型名
architecture
权重/量化
目标硬件
vLLM 架构级 PD
vLLM 精确模型 recipe
vLLM-Ascend 精确模型 recipe
官方版本
官方 Connector
本地 profile
本地 Connector
验证等级
缺项标签
```

### 13.2 官方方案与本地差异表

```text
模型/权重
官方镜像/版本栈
硬件和节点数
P 全局/本地 DP、TP、EP、rank start
D 全局/本地 DP、TP、EP、rank start
Connector
Proxy/Metaserver 流程
KV/bootstrap/DP RPC 端口
网络要求
必要环境变量
Prefix Cache
Chunked Prefill
Async Scheduling
MTP/EAGLE/DSpark
官方已知问题
Wings 最终命令差异
```

### 13.3 故障与缺项表

```text
模型/场景
直接报错或失败现象
发生阶段
社区能力/recipe 状态
物料/版本状态
Wings profile/命令状态
组网状态
真机验证状态
缺项标签
解除该标签所需证据
```
