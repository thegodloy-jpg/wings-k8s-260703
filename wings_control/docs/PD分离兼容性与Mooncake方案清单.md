# PD 分离兼容性与 Mooncake 方案清单

> 核对日期：2026-08-11
> 核对对象：FusionOne AI 23.6.1 ARM/X86 模型兼容性表、`wings_control/utils/model_utils.py`、`wings_control/config/defaults/pd_config.json`、当前 Wings PD 配置生成逻辑、vLLM 与 vLLM-Ascend 官方文档。
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

### 4.2 上游 vLLM Mooncake 路径

上游 vLLM 另有通用 `MooncakeConnector` 使用指南，官方示例为 Qwen2.5-7B-Instruct 的一个 P 实例和一个 D 实例，通过 Mooncake transfer engine 和 Proxy 传输 KV。该示例证明 Connector/API 和部署骨架存在，不是 Wings NVIDIA 路径的默认 Connector，也不是 79 个模型的逐模型生产配方。

官方来源：

- [NIXL Compatibility Matrix](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/)
- [NIXL Usage Guide](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [MooncakeConnector Usage Guide](https://docs.vllm.ai/en/stable/features/mooncake_connector_usage/)

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

“vLLM/NIXL”列只登记架构级能力；“Ascend 模型级配方”只登记当前找到的精确教程或同模型族教程；“本地 large-EP”只登记 profile 查表结果。

### 6.1 DeepSeek

| Architecture | Wings 模型名 | vLLM/NIXL | Ascend 模型级配方 | 本地 large-EP | 差异/缺项 |
|---|---|---|---|---|---|
| `DeepseekV2ForCausalLM` | DeepSeek-Coder-V2-Instruct；DeepSeek-Coder-V2-Instruct-w8a8 | MLA 架构级 | 未找到精确配方 | 无 | 精确模型 recipe、profile、物料和真机记录缺失 |
| `DeepseekV3ForCausalLM` | DeepSeek-R1；DeepSeek-R1-0528；DeepSeek-V3；DeepSeek-V3-0324；DeepSeek-V3.1；以及对应 5 个 w8a8 名称 | MLA/MoE 架构级 | R1 有通用多节点示例；V3.1 有模型页配方；其余精确权重未逐一登记 | 无 | R1/V3.1 未接入；其余按权重分项核对 |
| `DeepseekV32ForCausalLM` | DeepSeek-V3.2；DeepSeek-V3.2-w8a8；DeepSeek-V3.2-Exp；DeepSeek-V3.2-Exp-W8A8 | Sparse MLA 架构级 | V3.2-w8a8 有模型配方；Exp 未找到独立配方 | 有，Layerwise | 当前官方 V3.2 命令为 V1；Exp 不能自动继承 |
| `DeepseekV4ForCausalLM` | DeepSeek-V4；DeepSeek-V4-w8a8；DeepSeek-V4-Flash；DeepSeek-V4-Flash-w8a8-mtp；DeepSeek-V4-Pro；DeepSeek-V4-Pro-w4a8-mtp | Hybrid KV/MoE，按具体 cache spec 记录 | Flash/Pro 精确量化权重有配方；基础 V4 未找到同等配方 | 有一个 Flash 来源的架构 profile | Flash、Pro、基础 V4 未拆成权重级 profile |

### 6.2 Kimi

| Architecture | Wings 模型名 | vLLM/NIXL | Ascend 模型级配方 | 本地 large-EP | 差异/缺项 |
|---|---|---|---|---|---|
| `KimiK25ForConditionalGeneration` | Kimi-K2.5；Kimi-K2.5-w4a8；Kimi-K2.6-w4a8；Kimi-K2.7；Kimi-K2.7-w4a8；Kimi-K2.7-Code | 完整多模态在 NIXL 矩阵中未验证 | K2.5-w4a8、K2.6-w4a8 有；K2.7/Code 未找到 | 无 | K2.5/K2.6 profile 缺失；K2.7/Code 模型级配方缺失 |
| `KimiK3ForConditionalGeneration` | Kimi-K3 | 完整多模态未验证 | 未找到 | 无 | 架构矩阵、模型级配方和本地 profile 均未登记 |

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
| `Qwen3ForCausalLM` | Qwen3-32B | Dense 架构级 | 未找到精确配方 | 无 | 精确 recipe 和 profile 缺失 |
| `Qwen3MoeForCausalLM` | Qwen3-30B-A3B | MoE 架构级 | 当前模型页未找到 PD 章节 | 有用户自定义 Layerwise | 内部配方；非官方模型级配方 |
| `Qwen3MoeForCausalLM` | Qwen3-235B-A22B | MoE 架构级 | w8a8-rot 有 V1 配方 | 会命中 Qwen3-30 Layerwise | 权重、Connector 和拓扑不等价 |
| `Qwen3NextForCausalLM` | Qwen3-Next-80B-A3B-Instruct | Hybrid SSM/Mamba 架构级、受 Hybrid 限制 | 当前模型页未找到 PD 章节 | 无 | 模型级 recipe 和 profile 缺失 |
| `Qwen3_5ForConditionalGeneration` | Qwen3.5-27B；Qwen3.5-27B-w8a8；Qwen3.6-27B；Qwen3.6-27B-w8a8 | Hybrid SSM/Mamba；当前 P TP = D TP | w8a8 有 2 台 A3 配方 | 无 | profile、镜像物料和真机记录缺失 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5-397B-A17B-NVFP4 | Hybrid/MoE，NVIDIA 权重 | Ascend w8a8 配方不适用 | 会命中 Ascend Layerwise | 平台/精度隔离缺失 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5-397B-A17B-w8a8；Qwen3.5-397B-A17B | Hybrid/MoE 架构级 | w8a8 有 V1 配方 | 有，Layerwise | Connector 漂移；BF16 与 w8a8 分项记录 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5-35B-A3B；Qwen3.5-122B-A10B | Hybrid/MoE 架构级 | 未找到精确配方 | 会命中 397B profile | 397B 参数不能作为精确配方 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.6-35B-A3B；Qwen3.6-35B-A3B-w8a8 | Hybrid/MoE 架构级 | 当前模型页只有普通部署，未找到 PD 章节 | 会命中 397B profile | 模型级 recipe 缺失；已有请求期 Hybrid KV 故障记录 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen-AgentWorld-35B-A3B | Hybrid/MoE 架构级 | 未找到精确配方 | 会命中 397B profile | 精确 recipe 缺失，架构 profile 超覆盖 |

### 6.5 MiniMax 与 Llama

| Architecture | Wings 模型名 | vLLM/NIXL | Ascend 模型级配方 | 本地 large-EP | 差异/缺项 |
|---|---|---|---|---|---|
| `MiniMaxM2ForCausalLM` | MiniMax-M2.5；MiniMax-M2.5-NVFP4；MiniMax-M2.5-w8a8 | MoE 架构级；NVFP4 为 NVIDIA 路径 | 未找到 M2.5 精确配方 | 无 | 平台分流、精确 recipe 和 profile 缺失 |
| `MiniMaxM2ForCausalLM` | MiniMax-M2.7；MiniMax-M2.7-w8a8；MiniMax-M2.7-w8a8-QuaRot | MoE 架构级 | w8a8-QuaRot 有配方 | 无 | profile、镜像物料和真机记录缺失 |
| `MiniMaxM3SparseForConditionalGeneration` | MiniMax-M3；MiniMax-M3-MXFP8 | 多模态边界未验证 | 有普通部署资料，未找到具体 PD 启动章节 | 无 | 精确 PD recipe 和 profile 缺失 |
| `LlamaForCausalLM` | LLaMA3-8B；LLaMA3.1-70B；LLaMA3.1-70B-Instruct；Meta-Llama-3.1-70B-Instruct；DeepSeek-R1-Distill-Llama-8B/70B | Dense 架构级 | 未找到这些精确权重的 vLLM-Ascend 配方 | 无 | NVIDIA 可按 NIXL 单模型建项；Ascend 精确 recipe 缺失 |

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
| `UPSTREAM-NO-MODEL-RECIPE` | 有架构级能力，但未找到精确模型/权重/硬件 PD 命令 | DeepSeek-Coder-V2、Qwen3-32B、Qwen3-Next、Kimi-K2.7/K3 等 |
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
