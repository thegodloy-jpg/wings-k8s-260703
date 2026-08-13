# PD 分离不可用场景归因分析

> 核对日期：2026-08-13
> 本文只回答一个问题：为什么现在有较多 PD 分离场景不能用，究竟是社区限制，还是我们的物料、适配和验证没有完成。

## 1. 结论

PD 不可用不能统一归因为“社区不支持”，也不能统一归因为“物料验证来不及”。当前是五类问题同时存在：

1. **社区硬限制**：架构或功能在官方矩阵中明确不支持或受限，例如 NIXL Encoder-Decoder 不支持，Hybrid SSM/Mamba 当前要求 P/D TP 一致。
2. **社区只有架构能力，没有具体模型方案**：能说“理论上有 PD 路径”，但没有对应权重、卡型、P/D 比例和性能参数，不能直接产品化。
3. **Wings 适配未完成或已有配方漂移**：社区已经给出模型级方案，但本地没有精确 profile，或 Connector、TP/DP、Proxy 流程与当前官方方案不一致。
4. **物料和运行环境不闭环**：镜像中的 vLLM/vLLM-Ascend、CANN、Mooncake/NIXL、Python 包、原生 so、ABI 或动态链接路径与官方方案不等价。
5. **同等硬件、组网和真机验证未完成**：部分官方方案需要 3–8 台 Atlas A2/A3、32–64 张 NPU 及 RDMA/HCCS 组网。没有同等资源只能说“未验证”，不能说“社区不支持”。

最关键的判断是：**对官方已给出模型级 PD 方案的模型，当前不能用主要应从 Wings 精确适配、物料版本、组网和真机验证中找原因；对官方只提供架构级能力、没有具体模型配方的场景，才应归为社区模型级方案不足。**

## 2. 我们当前的客观覆盖情况

- 代码中普通 LLM 兼容集包含 **18 个架构、79 个模型名**。
- Ascend large-EP PD 注册表只有 **5 个精确 architecture profile**。
- 5 个 profile 按架构名机械覆盖 29 个模型，但“同架构命中”不等于“同一 PD 方案”；剩余 50 个模型没有 large-EP 精确 profile。
- ARM/X86 产品兼容性表证明的是普通推理兼容，不是 PD 兼容。

| 本地 architecture | 本地 Connector | 机械覆盖数 | 直接问题 |
|---|---|---:|---|
| `Qwen3MoeForCausalLM` | `MooncakeLayerwiseConnector` | 2 | 原始配方是 Qwen3-30B 内部 3P1D，却会同时覆盖官方使用 V1 的 Qwen3-235B |
| `DeepseekV32ForCausalLM` | `MooncakeLayerwiseConnector` | 4 | 当前官方 DeepSeek-V3.2 教程使用 `MooncakeConnectorV1` |
| `GlmMoeDsaForCausalLM` | `MooncakeConnectorV1` | 9 | GLM-5 profile 会覆盖 GLM-5.2，但 GLM-5.2 官方拓扑和 Connector 不同 |
| `Qwen3_5MoeForConditionalGeneration` | `MooncakeLayerwiseConnector` | 8 | 会覆盖 35B/122B/397B、Qwen3.6 和 AgentWorld；当前 397B 官方使用 V1 |
| `DeepseekV4ForCausalLM` | `MooncakeHybridConnector` | 6 | Connector 方向对齐，但 V4-Flash 和 V4-Pro 有独立的硬件与拓扑配方 |

因此，“有5个profile”不能推导“29个模型都可用”，反而暴露了按 architecture 粗粒度复用造成的错误支持风险。

## 3. 社区官方 PD 能力和具体方案

### 3.1 vLLM Recipes：官方 PD 基线

本文的 NVIDIA 口径只审计 [vLLM 官方 Recipes](https://recipes.vllm.ai/)。审计基线固定为 [`vllm-project/recipes@9368396`](https://github.com/vllm-project/recipes/tree/93683962d28f2ed140c08030e2b9be0a50f646a0)：共有 **41 个模型族 Recipe 声明兼容 `pd_cluster`**。这个数字表示目录中存在可生成的 PD 配置，不等于 41 个模型已经完成性能和产品验收。

共享 [`pd_cluster.yaml`](https://github.com/vllm-project/recipes/blob/93683962d28f2ed140c08030e2b9be0a50f646a0/strategies/pd_cluster.yaml) 给出的默认骨架是：

| 组件 | 官方配置 | 作用与边界 |
|---|---|---|
| Prefill | 端口 8001；`NixlConnector`；`kv_role=kv_producer`；`kv_load_failure_policy=fail`；`--enforce-eager` | 生产 KV；不是 Mooncake |
| Decode | 端口 8002；`NixlConnector`；`kv_role=kv_consumer`；`FULL_DECODE_ONLY` | 消费 KV |
| Side channel | P 端 5557，D 端 5558；host 分别绑定 `$PREFILL_NODE_1`、`$DECODE_NODE_1` | 每个节点必须可路由，端口不能冲突 |
| 数据面环境 | `UCX_NET_DEVICES=all`、`NCCL_CUMEM_ENABLE=1`、`NCCL_MNNVL_ENABLE=1`、`NCCL_NVLS_ENABLE=1` | 跨机仍需与 NIC、IB/RoCE、GDR 和镜像内 NIXL/UCX 物料一致 |
| Router | `vllm-router --vllm-pd-disaggregation`，默认 `round_robin` | 负责把请求路由到 P/D；不负责传输 KV 张量 |
| 资源门槛 | 策略声明 `min_gpus: 8`、`multi_node: true` | 具体模型会按显存和模型 override 生成更大的 P/D 池，不能把 8 卡理解成所有模型的完整集群卡数 |

因此，vLLM Recipe 的默认 NVIDIA PD 是 **`NixlConnector` + NIXL 数据面 + vLLM Router**，不是 Mooncake。vLLM 的 [Mooncake 分布式 KV Store](https://github.com/vllm-project/recipes/blob/93683962d28f2ed140c08030e2b9be0a50f646a0/kv_store/kv_store_distributed_mooncake.yaml) 是另一个可选能力；与 PD 同时使用时，官方配置通过 `MultiConnector` 组合 `NixlConnector` 的 producer/consumer 与 `MooncakeStoreConnector`，并没有用 Mooncake 替换 P→D 的 NIXL 协议。

### 3.2 vLLM 官方 `pd_cluster` 全量模型清单

本节不再按当前项目兼容表筛选模型，只回答“vLLM 官方 Recipes 已经有哪些 NVIDIA PD 方案”。固定审计版本 [`vllm-project/recipes@9368396`](https://github.com/vllm-project/recipes/tree/93683962d28f2ed140c08030e2b9be0a50f646a0) 中：

- **41 个模型族**声明兼容 `pd_cluster`；
- 共登记 99 个 variant，其中排除 6 个 AMD-only variant 后，剩余 **93 个 NVIDIA 可选 variant 记录、92 个唯一权重 ID**；
- **31 个模型族**直接复用共享 `pd_cluster`；**10 个模型族**增加模型专项 override；
- 全部方案的 P→D KV 基线仍是 `NixlConnector`，默认 Router 是 `round_robin`，Mooncake 不是默认 P/D Connector。

| 厂商 | 数量 | 已有 `pd_cluster` 的模型族 |
|---|---:|---|
| DeepSeek | 6 | [DeepSeek-R1](https://recipes.vllm.ai/deepseek-ai/DeepSeek-R1?strategy=pd_cluster)、[V3](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V3?strategy=pd_cluster)、[V3.1](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V3.1?strategy=pd_cluster)、[V3.2](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V3.2?strategy=pd_cluster)、[V4-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash?strategy=pd_cluster)、[V4-Pro](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro?strategy=pd_cluster) |
| inclusionAI | 2 | [Ring-1T-FP8](https://recipes.vllm.ai/inclusionAI/Ring-1T-FP8?strategy=pd_cluster)、[Ring-2.6-1T](https://recipes.vllm.ai/inclusionAI/Ring-2.6-1T?strategy=pd_cluster) |
| Mindlab | 1 | [Macaron-V1-Coding-Venti](https://recipes.vllm.ai/mindlab-research/Macaron-V1-Coding-Venti?strategy=pd_cluster) |
| MiniMax | 5 | [M2](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M2?strategy=pd_cluster)、[M2.1](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M2.1?strategy=pd_cluster)、[M2.5](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M2.5?strategy=pd_cluster)、[M2.7](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M2.7?strategy=pd_cluster)、[M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3?strategy=pd_cluster) |
| Mistral | 2 | [Mistral-Large-3-675B](https://recipes.vllm.ai/mistralai/Mistral-Large-3-675B-Instruct-2512?strategy=pd_cluster)、[Mistral-Small-4-119B](https://recipes.vllm.ai/mistralai/Mistral-Small-4-119B-2603?strategy=pd_cluster) |
| Moonshot | 6 | [Kimi-K2-Instruct](https://recipes.vllm.ai/moonshotai/Kimi-K2-Instruct?strategy=pd_cluster)、[K2-Thinking](https://recipes.vllm.ai/moonshotai/Kimi-K2-Thinking?strategy=pd_cluster)、[K2.5](https://recipes.vllm.ai/moonshotai/Kimi-K2.5?strategy=pd_cluster)、[K2.6](https://recipes.vllm.ai/moonshotai/Kimi-K2.6?strategy=pd_cluster)、[K2.7-Code](https://recipes.vllm.ai/moonshotai/Kimi-K2.7-Code?strategy=pd_cluster)、[K3](https://recipes.vllm.ai/moonshotai/Kimi-K3?strategy=pd_cluster) |
| Qwen | 6 | [Qwen3-235B-2507](https://recipes.vllm.ai/Qwen/Qwen3-235B-A22B-Instruct-2507?strategy=pd_cluster)、[Qwen3-Coder-480B](https://recipes.vllm.ai/Qwen/Qwen3-Coder-480B-A35B-Instruct?strategy=pd_cluster)、[Qwen3-VL-235B](https://recipes.vllm.ai/Qwen/Qwen3-VL-235B-A22B-Instruct?strategy=pd_cluster)、[Qwen3.5-122B](https://recipes.vllm.ai/Qwen/Qwen3.5-122B-A10B?strategy=pd_cluster)、[Qwen3.5-397B](https://recipes.vllm.ai/Qwen/Qwen3.5-397B-A17B?strategy=pd_cluster)、[Qwen3.8-2.4T](https://recipes.vllm.ai/Qwen/Qwen3.8-2.4T-A95B?strategy=pd_cluster) |
| StepFun | 2 | [Step-3.5-Flash](https://recipes.vllm.ai/stepfun-ai/Step-3.5-Flash?strategy=pd_cluster)、[Step-3.7-Flash](https://recipes.vllm.ai/stepfun-ai/Step-3.7-Flash?strategy=pd_cluster) |
| Thinking Machines | 2 | [Inkling](https://recipes.vllm.ai/thinkingmachines/Inkling?strategy=pd_cluster)、[Inkling-Small](https://recipes.vllm.ai/thinkingmachines/Inkling-Small?strategy=pd_cluster) |
| Xiaomi | 3 | [MiMo-V2-Flash](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2-Flash?strategy=pd_cluster)、[MiMo-V2.5](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5?strategy=pd_cluster)、[MiMo-V2.5-Pro](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5-Pro?strategy=pd_cluster) |
| Z-AI / GLM | 6 | [GLM-4.5](https://recipes.vllm.ai/zai-org/GLM-4.5?strategy=pd_cluster)、[GLM-4.6](https://recipes.vllm.ai/zai-org/GLM-4.6?strategy=pd_cluster)、[GLM-4.7](https://recipes.vllm.ai/zai-org/GLM-4.7?strategy=pd_cluster)、[GLM-5](https://recipes.vllm.ai/zai-org/GLM-5?strategy=pd_cluster)、[GLM-5.1](https://recipes.vllm.ai/zai-org/GLM-5.1?strategy=pd_cluster)、[GLM-5.2](https://recipes.vllm.ai/zai-org/GLM-5.2?strategy=pd_cluster) |
| **合计** | **41** | 以上均来自 vLLM Recipes 的 `compatible_strategies: pd_cluster` |

41 个模型族并不代表 41 种 Connector。它们共享同一套 NIXL P/D 协议，主要差异在权重显存决定的共置/分节点规模，以及以下 10 个模型专项 override：

| 模型族 | 专项 PD 差异 |
|---|---|
| DeepSeek-V4-Flash | P/D 使用 DEP；启用 Hybrid KV manager、sleep mode 和 NCCL symmetric-memory 相关环境；P/D batch 参数分离 |
| DeepSeek-V4-Pro | P/D 使用 DEP；Hybrid KV；GB300 可缩为每侧 1 个 NVL4 节点，其他大权重组合按显存扩展 |
| MiniMax-M3 | Blackwell 下增加模型专项 Attention/indexer FP8 配置；P/D 协议仍为 NIXL |
| Kimi-K2.5、Kimi-K2.6 | P/D 使用 DEP；block size 64；D 端使用 `flashinfer_nvlink_one_sided` |
| Kimi-K3 | P 侧 TEP、D 侧 DEP；`VLLM_SSM_CONV_STATE_LAYOUT=DS`；D 关闭 Prefix Cache并限制 batch |
| Qwen3.5-122B、Qwen3.5-397B | P、D 两侧都注入 `VLLM_SSM_CONV_STATE_LAYOUT=DS` |
| Inkling、Inkling-Small | P/D pool 仅允许 DEP；Inkling 的 Hopper PD 被显式禁用，需使用 Blackwell |
| 其余 31 个模型族 | 无模型级 PD override，按共享 `pd_cluster` 和所选硬件/variant 的显存自动决定共置、每侧节点数及 TP |

需要保留一个成熟度边界：41 表示官方源码声明了 PD 生成路径，不等于全部组合经过专项性能验证。当前 41 个页面中有 39 个能直接取得默认 `pd_cluster.json`；Qwen3.8-2.4T 和 Inkling 的默认 JSON 端点未生成，原因分别与默认权重超出 H200 自动生成上限、Hopper PD 被禁用有关，但其模型页和源码仍明确声明 `pd_cluster`。

全部 NVIDIA variant、最低 vLLM 版本和默认拓扑分类见 [PD 分离兼容性与 Mooncake 方案清单](./PD分离兼容性与Mooncake方案清单.md#43-vllm-官方-pd-全量模型与-variant)。
### 3.3 vLLM-Ascend：已经有模型级官方方案

下表只列官方文档中同时给出了具体模型、硬件、P/D 拓扑和 Connector 的方案。模型页存在但没有 PD 章节，不计为模型级 PD 方案。

| 模型/官方示例权重 | 官方硬件和节点 | 官方全局 P 拓扑 | 官方全局 D 拓扑 | Connector | Wings 当前状态 |
|---|---|---|---|---|---|
| DeepSeek-R1-w8a8 | 4 台 Atlas 800T A3 | DP2 TP8 | DP32 TP1 | Layerwise 或 V1 两套示例 | 无 `DeepseekV3ForCausalLM` profile，属于本地未适配 |
| DeepSeek-V3.1-w8a8-mtp-QuaRot | 官方多节点方案 | DP2 TP8 | DP32 TP1 | `MooncakeConnectorV1` | 无精确 profile，属于本地未适配 |
| DeepSeek-V3.2-w8a8-mtp-QuaRot | 官方多节点方案 | DP2 TP16 | DP8 TP4 | 当前官方为 `MooncakeConnectorV1` | 本地为 Layerwise，属于方案漂移 |
| DeepSeek-V4-Flash-w8a8-mtp | A3/A2 均有方案 | A3 DP4 TP4；A2 DP8 TP1 | A3 DP16 TP1；A2 DP32 TP1 | `MooncakeHybridConnector` | Connector 对齐，版本、异步调度和真机稳定性待确认 |
| DeepSeek-V4-Pro-w4a8-mtp | A3 128GB×8 和 A2 64GB×8 均有多节点方案 | A3 DP2 TP16；A2 DP4 TP8 | A3 DP16 TP2；A2 DP8 TP4 | `MooncakeHybridConnector` | 没有 Pro 独立 recipe，不能继承 Flash 结论 |
| GLM-5/GLM-5.1-w8a8 | A3，2 个 P 节点+4 个 D 节点 | DP2 TP16 | DP16 TP4 | `MooncakeConnectorV1` | 拓扑和 Connector 基本同源，但本地性能参数有差异 |
| GLM-5.2-w4a8c8 A3 | 4 台 Atlas 800 A3：2P+2D | DP4 TP8 | DP32 TP1 | `MooncakeConnectorV1` | 无 GLM-5.2 独立 profile，会误用 GLM-5 配方 |
| GLM-5.2-w4a8c8 A2 | 8 台 Atlas 800 A2：4P+4D | DP4 TP8 | DP8 TP4 | `MultiConnector`：Mooncake V1 + AscendStore | Connector、拓扑和 KV Pool 均未适配 |
| Qwen3-235B-A22B-w8a8-rot | 3 台 Atlas 800 A3：1P+2D | DP2 TP8 | DP8 TP4 | `MooncakeConnectorV1` | 被 Qwen3-30 内部 Layerwise profile 覆盖，未精确对齐 |
| Qwen3.5-27B-w8a8 / Qwen3.6-27B-w8a8 | 2 台 Atlas 800 A3，物理 1P1D | DP8 TP2 | DP8 TP2 | `MooncakeConnectorV1` | 无精确 profile；Atlas 300I DUO 官方明确不支持该多节点 PD |
| Qwen3.5-397B-A17B-w8a8 | 3 台 Atlas 800 A3，共 48 NPU：1P+2D | DP8 TP2 | DP16 TP2 | 当前官方为 `MooncakeConnectorV1` | 本地为 Layerwise；D 端 Prefix Cache 还有官方已知问题 |
| Kimi-K2.5-w4a8 | 4 台 Atlas 800 A3 | DP2 TP8 | DP32 TP1 | `MooncakeConnectorV1` | 无精确 profile，属于本地未适配 |
| Kimi-K2.6-w4a8 | 4 台 Atlas 800 A3 | DP4 TP4 | DP8 TP4 | `MooncakeConnectorV1` | 无精确 profile，属于本地未适配 |
| MiniMax-M2.7-w8a8-QuaRot | 2 台 Atlas 800 A3，物理 1P1D | DP2 TP8 | DP2 TP8 | `MooncakeConnectorV1` | 无精确 profile，属于本地未适配 |

官方来源：

- [vLLM-Ascend 多节点 Mooncake PD](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md)
- [DeepSeek-V3.1](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.1.html)
- [DeepSeek-V3.2](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.2.html)
- [DeepSeek-V4-Flash](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/DeepSeek-V4-Flash.html)
- [DeepSeek-V4-Pro](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Pro.html)
- [GLM-5/5.1](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html)
- [GLM-5.2](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html)
- [Qwen3-235B-A22B](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-235B-A22B.html)
- [Qwen3.5-27B/Qwen3.6-27B](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html)
- [Qwen3.5-397B-A17B](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-397B-A17B.html)
- [Kimi-K2.5](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Kimi-K2.5.html)
- [Kimi-K2.6](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K2.6.html)
- [MiniMax-M2](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/MiniMax-M2.html)

### 3.4 vLLM 官方 Recipe 的可用边界

NVIDIA 侧不再用当前项目模型清单作为分母，只按 vLLM 官方 Recipe 自身统计：

- 源码层有 **41 个模型族**声明 `compatible_strategies: pd_cluster`；
- 排除 6 个 AMD-only variant 后，有 **93 条 NVIDIA variant 记录、92 个唯一权重 ID**；
- 默认 NVIDIA `pd_cluster.json` 端点为 **39/41 可取得、2/41 未生成**：Qwen3.8-2.4T 因默认 BF16 权重超过 H200 自动生成上限，Inkling 因 Hopper PD 被显式禁用；
- 39 个已生成端点中，DeepSeek-V4-Flash 和 Inkling-Small 的 H200 共置命令存在 GPU 可见数与本地 DP 数不一致，不能不经修正直接执行；
- DeepSeek-V4-Flash 在 RTX PRO 6000、Inkling 在 Hopper 上存在 Recipe 明确标注的硬件组合限制。

因此，“源码声明方案存在”“网站能生成默认 JSON”“命令能够直接执行”“已完成性能与长稳验证”是四个不同层级，不能把 41 个模型族统一写成已验证可用。

## 4. 不可用原因和责任边界

| 原因类别 | 判定标准 | 典型模型/问题 | 是否社区限制 |
|---|---|---|---|
| 社区硬限制 | 官方矩阵明确不支持，或官方同版本、同硬件、同命令仍稳定复现上游错误 | Encoder-Decoder；Hybrid Mamba 异构 TP/block size 限制 | 是 |
| 社区模型级方案缺口 | 只有架构矩阵，或 vLLM Recipes 模型页未声明兼容 `pd_cluster` | Qwen3-30B/Next、Qwen3.6-35B、通用 Kimi-K2.7 等 | 是，但是方案缺口，不代表架构绝对不可用 |
| Wings 适配缺口 | 官方已有方案，本地无精确 profile | DeepSeek-R1/V3.1、Qwen3.5/3.6-27B、Kimi-K2.5/2.6、MiniMax-M2.7 | 否 |
| 方案漂移/架构超覆盖 | 本地有 profile，但 Connector、TP/DP、Proxy 或模型边界与当前官方不同 | DeepSeek-V3.2、Qwen3.5-397B、GLM-5.2、Qwen3-235B、V4-Pro | 否 |
| 物料/版本缺口 | 版本不同，包/so 缺失，`ldd` 不可解析，或 import 后出现 ABI/allocator 崩溃 | `libtransfer_engine.so`、`ascend_transport.so`、`corrupted size vs. prev_size` | 否；只有确认官方等价物料也复现后才可转为上游问题 |
| 组网/编排缺口 | RDMA/HCCS、网卡、DP rank、RPC、Mooncake/NIXL 端口或 Proxy 流程不一致 | `Address already in use`、P/D Connector 不一致、side-channel 端口重复 | 否 |
| 真机验证/资源缺口 | 静态方案可生成，但缺少同等节点和网络，未通过连续请求、并发、长稳和故障恢复 | Qwen3.5-397B 需 48 NPU；GLM-5/5.1 需 6 节点；GLM-5.2 A2 需 8 节点 | 否，应标记“社区有方案，我们未完成验证” |

Ascend 官方多节点方案还明确要求同一可达局域网、节点内 HCCS、节点间 RDMA、容器可读 `/etc/hccn.conf`。每个节点 8 张 NPU 时 AscendDirectTransport 可使用 `20000-27999`，建议 Mooncake `kv_port >= 28000`；16 张 NPU 时范围扩大到 `20000-35999`，建议 `kv_port >= 36000`。这些是组网和产品化条件，不是模型社区支持状态。

## 5. 已遇到的具体问题和归因

| 场景 | 具体报错/现象 | 已确认的事实 | 归因 |
|---|---|---|---|
| Ascend 1P1D TP 冲突 | `KV transfer 'decode' config has a conflicting tensor parallel size. Expected 2, but got 1.` | Engine 使用 TP2，KV `decode.tp_size=1` | Wings standalone PD 拓扑生成错误；尚未进入 KV 传输，不是社区模型限制 |
| Qwen3.6 关闭 HMA | `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type.` | 模型同时存在 Attention 和 GDN/Mamba cache spec | Hybrid KV 架构/版本边界 |
| Qwen3.6 打开 HMA | P/D 均启动，请求长期无返回且无明确异常栈 | 只证明 KV cache 初始化通过，未证明多 cache group 传输和完成通知正常 | 请求期 Hybrid KV 传输/调度未闭环；证据不足以指定为某个上游 Bug |
| Mooncake 主库不可见 | `ImportError: libtransfer_engine.so: cannot open shared object file: No such file or directory` | `/usr/local/lib/libtransfer_engine.so` 实际存在 | 镜像动态链接路径/物料集成问题 |
| Mooncake 依赖库不可见 | `ImportError: ascend_transport.so: cannot open shared object file` / `ascend_transport.so => not found` | so 存在；增加 `LD_LIBRARY_PATH=/usr/local/lib:...` 后可解析 | 进程级动态链接搜索路径缺失 |
| Mooncake import 成功后崩溃 | `corrupted size vs. prev_size` / `Aborted` | 发生在 import 成功后的退出/析构阶段 | 原生库、ABI、allocator 或析构链问题；不再是动态库路径问题 |
| 首请求成功，第二请求崩溃 | `start_load_kv(): assert self.kv_recv_thread is not None`；前序有 `Delaying free of 2 blocks ...` | 启动、连接和至少一次 KV 传输已通过 | Connector 线程生命周期、Prefix Cache 或 block 复用问题；尚不能确认为上游 Bug |
| NVIDIA Qwen3.5 Mamba 布局 | `3-read Mamba conv transfer requires DS conv state layout. Set VLLM_SSM_CONV_STATE_LAYOUT=DS` | NIXL 对 Hybrid/Mamba 状态传输有明确布局约束 | 社区架构约束 + Wings 未注入必要环境变量 |
| NVIDIA PD 没有下发 TP | `device_count=2`、`PD_ROLE=P`，最终命令无 `--tensor-parallel-size`，vLLM 按 TP1 启动 | 平台未下发 `TP_SIZE`/`PD_PREFILL_TP_SIZE` | Wings/平台拓扑输入缺失，不是 NIXL 不支持 |
| large-EP 无精确 profile | `ValueError: PD large-EP has no registered profile for architecture=KimiK25ForConditionalGeneration` | Kimi-K2.5/K2.6 官方已有模型级方案 | 明确的 Wings 注册表适配缺口 |
| P/D Connector 不一致 | P 侧 DP>1 使用 Layerwise，D 侧 DP=1 保留 V1 | 两侧根据“当前角色 DP”选到不同协议 | Wings 全局 profile 选择边界问题 |
| TP/DP 非法或超订 | TP=0、DP=0 或 4 卡 Pod 配置 TP=8 仍可能进入最终命令 | PD override 可绕过普通容量校验 | Wings 启动前输入校验缺口 |

这些报错说明：实际问题已经横跨社区架构能力、Wings 配置生成、镜像动态库、原生 ABI、Connector 生命周期、组网和验证资源，所以不能用一个原因统一解释。

## 6. 针对主问题的正式回答

> 当前 PD 分离不能用的场景比较多，既有社区能力边界，也有我们的适配、物料和验证未完成，但不能统一说成“社区不支持”。
>
> NVIDIA 侧只按 vLLM 官方 Recipes 统计，不再与当前项目模型名取交集：源码共有 41 个模型族声明兼容 `pd_cluster`，排除 AMD-only variant 后有 93 条 NVIDIA variant 记录、92 个唯一权重 ID；31 个模型族复用共享策略，10 个包含模型专项 override。共享基线是 `NixlConnector`、NIXL 数据面和 `vllm-router`，不是统一 Mooncake。41 个模型族中，39 个能取得默认 NVIDIA 策略 JSON，Qwen3.8-2.4T 和 Inkling 的默认端点未生成；已生成项里 DeepSeek-V4-Flash 和 Inkling-Small 的 H200 共置命令还存在生成一致性问题。GLM-4.5 至 GLM-5.2 均已纳入官方清单，其中 GLM-5.2 默认方案是 16×H200、P/D 各 TP8、NIXL、round-robin Router。
>
> Ascend/vLLM-Ascend 侧已经为 DeepSeek-V3.1/V3.2、V4-Flash/V4-Pro、GLM-5/5.1/5.2、Qwen3-235B、Qwen3.5/3.6-27B、Qwen3.5-397B、Kimi-K2.5/K2.6、MiniMax-M2.7 等提供了具体 PD 方案。这些模型当前不能用时，主要不是社区没有方案，而是 Wings 只有 5 个 large-EP profile，且存在精确 profile 缺失、Connector/拓扑漂移、按架构超范围复用、镜像物料不等价和同等硬件验证未完成。
>
> 已发生的错误也能区分责任：`Expected 2, but got 1` 是 Wings TP 拓扑生成错误；`libtransfer_engine.so`/`ascend_transport.so` 不可见是镜像物料和链接环境问题；`corrupted size vs. prev_size` 是原生 ABI/allocator 问题；第二请求的 `kv_recv_thread is None` 是 Connector 状态复用问题；NIXL DS conv state 报错是 Hybrid/Mamba 布局约束与 Wings 环境适配缺失。因此每个不可用项必须标明是“社区硬限制”、“社区无模型级方案”、“Wings 未适配”、“物料不满足”还是“同等资源未验证”，不能笼统写成“PD 不支持”。

## 7. 量化总结和推荐 PD 策略

### 7.1 数量口径

| 统计对象 | 总数 | 已覆盖/已支持 | 未覆盖/未支持 | 必须注意的口径 |
|---|---:|---:|---:|---|
| 产品文本生成部署行 | 83 | ARM 45 行，X86 38 行 | 不适用 | 这是普通推理场景数，不是 PD 支持数 |
| 产品“平台+模型”去重记录 | 71 | ARM 35 个，X86 36 个 | 不适用 | 同一模型在 ARM/X86 分别计数，仍只表示普通推理 |
| Wings 代码模型名兼容集 | 79 | 29 个被 5 个 PD profile 按 architecture 机械命中（36.7%） | 50 个没有 large-EP profile（63.3%） | 29 只是机械命中，不是 29 个已验证支持 |
| vLLM/NIXL 官方架构类别 | 7 | 5 类基础 PD 支持（71.4%） | 1 类未验证，1 类明确不支持 | 支持：Dense、MLA、Sparse MLA、MoE、Hybrid SSM/Mamba；未验证：Multimodal；不支持：Encoder-Decoder |
| vLLM 官方 Recipe 模型族 | 41 | 41 个源码声明兼容 `pd_cluster` | 0 个源码未声明 | 仅表示该模型 YAML 有 PD 生成路径，不等于经过专项性能或产品验收 |
| vLLM 官方 NVIDIA variant | 93 条记录、92 个唯一权重 ID | 全部隶属上述 41 个模型族 | 另有 6 条 AMD-only variant 已排除 | Macaron 的 default/fp8 两个 label 指向同一权重 ID |
| vLLM 默认 NVIDIA 策略端点 | 41 个模型族 | 39 个返回 `pd_cluster.json` | 2 个未生成 | 已生成项中还有 2 个 H200 共置 DEP 命令存在 GPU 数与本地 DP 数冲突 |
| vLLM-Ascend 官方模型/硬件 PD 方案组 | 14 | 14 组都有官方方案 | 0 组 | 不能用 `79-14` 推导其他 65 个模型明确不支持 |

“官方不支持多少个”必须分成两种状态：

1. **明确不支持**：当前可明确量化的是 NIXL 矩阵中 1 个架构类别，即 Encoder-Decoder。
2. **Recipe 生成边界**：41 个 vLLM 模型族均在源码声明 PD；其中 2 个默认 NVIDIA JSON 未生成，另有 2 个已生成默认命令存在明显的共置拓扑冲突。这些状态应分别写成“源码有方案但默认端点未生成”或“默认命令需修正”，不能笼统写成“社区不支持”。

### 7.2 14 个 Ascend 官方方案的 Wings 就绪度

| Wings 状态 | 场景数 | 占 14 个官方方案的比例 | 对应场景 | 结论 |
|---|---:|---:|---|---|
| 基础 Connector/拓扑方向对齐 | 2 | 14.3% | DeepSeek-V4-Flash；GLM-5/5.1 | 可进入物料等价和真机验证，但尚不能宣称产品支持 |
| 有本地条目或同架构机械覆盖，但必须修订/拆分 | 6 | 42.9% | DeepSeek-V3.2、V4-Pro、GLM-5.2 A3、GLM-5.2 A2、Qwen3-235B、Qwen3.5-397B | 主要是 Connector 漂移、拓扑不同或架构超覆盖，属于 Wings 适配问题 |
| 没有精确 profile | 6 | 42.9% | DeepSeek-R1、DeepSeek-V3.1、Qwen3.5/3.6-27B、Kimi-K2.5、Kimi-K2.6、MiniMax-M2.7 | 官方有方案，本地未接入 |
| 需要本地适配处理的合计 | 12 | 85.7% | 上述“需修订/拆分”6组 + “无精确profile”6组 | 对这 12 组，当前阻塞主因不是社区无方案 |

从这个口径看，在已有官方 Ascend 方案的 14 组场景中，**12 组（85.7%）仍需要 Wings 补 profile 或修订现有配方**。这可以量化说明：对“社区已有方案”这一类场景，目前的主要矛盾在产品侧适配和验证闭环，而不是社区功能缺失。

就当前文档和仓库可见证据，14 组场景中尚未找到哪一组已完成“目标产品镜像 + 官方同等硬件 + 真实 RDMA 组网 + 连续请求/并发/长稳/故障恢复”的完整验收证据。因此，这里的 2 组“基础对齐”不能写成 2 组“已验收支持”；如果存在仓库外的正式验收报告，需要再回填该数字。

### 7.3 推荐 Connector 和 Mooncake 策略

| 场景 | 推荐策略 | 在 14 个 Ascend 官方方案中的数量 | 使用边界 |
|---|---|---:|---|
| vLLM Recipes 常规 Attention/MLA/MoE | `NixlConnector` | 不与 Ascend 14 组混算 | P/D 版本、模型哈希、KV dtype、Attention backend 和推测方法必须兼容；跨机配置 UCX/LIBFABRIC 和唯一 side-channel 端口 |
| vLLM Recipes Hybrid SSM/Mamba | `NixlConnector` + Hybrid 专项约束 | 不与 Ascend 14 组混算 | 当前 P TP = D TP；Qwen3.5-122B 的 P、D 均设置 `VLLM_SSM_CONV_STATE_LAYOUT=DS`；不使用 Cross-layer 和异构 block size |
| Ascend 常规 Attention/MLA/MoE | `MooncakeConnectorV1` 作为默认基线 | 10 组为 V1 唯一官方策略；DeepSeek-R1 另有 1 组可选 V1/Layerwise | 如 DeepSeek-V3.1/V3.2、GLM-5/5.1、GLM-5.2 A3、Qwen、Kimi、MiniMax-M2.7；不得无条件改为 Layerwise |
| 官方明确的 Layerwise 方案 | `MooncakeLayerwiseConnector` | 1 组：DeepSeek-R1 的可选方案 | 必须同时使用 Layerwise Proxy/metaserver 和对应请求顺序；它不是普通性能开关 |
| DeepSeek-V4 Hybrid KV | `MooncakeHybridConnector` | 2 组：V4-Flash、V4-Pro | 必须配套 Hybrid KV manager，并分别管理 Flash/Pro 的硬件、TP/DP、MTP/DSpark、Prefix Cache 和调度参数 |
| GLM-5.2 A2 KV Pool | `MultiConnector`：`MooncakeConnectorV1 + AscendStore` | 1 组 | 必须同时配置 AscendStore/KV Pool、Connector 顺序和 load failure policy，不能用单一 V1 代替 |

14 个 Ascend 官方方案的 Connector 分布为：

- **10 组（71.4%）**：官方唯一策略为 `MooncakeConnectorV1`；
- **1 组（7.1%）**：DeepSeek-R1 可在 V1 和 Layerwise 两套官方方案中选择；
- **2 组（14.3%）**：DeepSeek-V4-Flash/Pro 使用 `MooncakeHybridConnector`；
- **1 组（7.1%）**：GLM-5.2 A2 使用 `MultiConnector`。

因此，本地 Mooncake 策略应收敛为：**V1 是普通 Ascend PD 默认基线；Layerwise 只在官方明确要求且 Proxy 流程成套时使用；Hybrid 仅用于 DeepSeek-V4 类 Hybrid KV；MultiConnector 仅用于官方明确需要外部 KV Pool 的方案。**

### 7.4 为什么不存在一个通用的 MooncakeConnector 策略

Mooncake 解决的是 KV 跨实例传输基础能力，但 Connector 还定义了 **KV 如何组织、何时传输、P/D 如何交换元数据以及 Proxy 如何调度请求**。不同 Connector 不是同一协议的性能档位，而是不同的缓存和控制面协议，因此不能为所有模型固定一个 Connector。

| 差异维度 | `MooncakeConnectorV1` | `MooncakeLayerwiseConnector` | `MooncakeHybridConnector` | `MultiConnector` |
|---|---|---|---|---|
| KV 结构 | 常规 Attention/MLA/MoE KV | 同一模型的 KV 按 layer 逐层生产和消费 | 处理 Attention、Mamba/SSM 等多个 KV/cache group | 同时组合 Mooncake 与 AscendStore 等外部 KV Pool |
| 传输时机 | 通常在 P 完成所需 KV 后由 D 加载 | P 逐层推送，D 逐层消费 | 按不同 cache group 完成传输和就绪通知 | 先按 Connector 顺序查找/加载，失败时按策略回退 |
| 控制面 | 常规 P→D Proxy 流程 | 必须配套 Layerwise Proxy/metaserver 和其请求时序 | 必须配套 Hybrid KV manager 和模型缓存布局 | 必须额外配置 KV Pool、Connector 顺序和 failure policy |
| 关键拓扑/元数据 | P/D TP、DP、block、KV dtype、engine ID | 除常规拓扑外，还要保证 layer 级生产/消费关系一致 | 还要匹配多 cache group、HMA、MTP/DSpark 和缓存布局 | 还要匹配外部存储元数据和命中/回退语义 |
| 典型方案 | DeepSeek-V3.1/V3.2、GLM-5/5.1、Qwen、Kimi、MiniMax-M2.7 | DeepSeek-R1 的官方可选方案 | DeepSeek-V4-Flash/Pro | GLM-5.2 A2 |

不能通用的具体原因有：

1. **模型 KV 结构不同**：普通 Attention/MLA 与 Attention+Mamba/SSM 混合模型的 cache spec 不同。V1 不能自动获得 Hybrid Connector 的多 cache group 语义，Hybrid 也不是普通模型的“更强 V1”。
2. **请求顺序和完成通知不同**：V1 与 Layerwise 的产生/消费时机不同。只替换 Connector 名称而不替换 Proxy/metaserver，会导致 D 等待不会到来的 KV 完成事件，表现为请求挂起。
3. **P/D 拓扑不是通用常量**：不同官方方案的 P/D TP、DP、节点数、block size、KV dtype、engine ID 和端口都不同。即使 Connector 类型相同，也不能共用一份固定拓扑配置。
4. **缓存和调度开关必须与模型配套**：HMA、Prefix Cache、Chunked Prefill、Async Scheduling、MTP/DSpark 和 block 释放语义都会改变 Connector 行为。“能启动”不代表第二请求、缓存复用和长稳可用。
5. **部分方案依赖外部组件**：GLM-5.2 A2 需要 AscendStore/KV Pool，Layerwise 需要专用 Proxy/metaserver。这些能力不能通过单一 V1 或单一 Hybrid Connector 自动获得。
6. **版本和原生物料必须与方案一致**：同名 Connector 在不同 vLLM-Ascend/Mooncake/CANN 组合中也可能有元数据、ABI 或生命周期差异，不能用 Connector 名称代替物料版本验证。

如果强制设置一个全局默认 Connector，典型后果不是统一的“不支持”报错，而是：

- 初始化阶段的 KV spec 或 TP/DP 冲突；
- P/D 使用不同协议导致请求挂起；
- 首请求成功，但第二请求在缓存/线程复用时崩溃；
- 外部 KV Pool 未配置导致加载失败或无法回退；
- 镜像中的 Connector 名称存在，但原生库/ABI 不匹配。

因此，可以建立通用的**选择规则**，但不能建立一个通用的**Connector 实现**：

1. 当前 Wings 的 vLLM Recipe 路径进入 `NixlConnector` 策略，不进入 Ascend Mooncake 默认值；有共享 KV Store/Offload 需求时，再按官方组合使用 `MultiConnector(NixlConnector + MooncakeStoreConnector)`。
2. Ascend 必须先精确匹配“模型/权重+硬件+版本”的官方 recipe。
3. DeepSeek-V4 选 Hybrid；GLM-5.2 A2 选 V1+AscendStore MultiConnector；官方明确的 Layerwise 方案成套选 Layerwise；其余有官方 V1 配方的普通模型选 V1。
4. 没有精确官方 recipe 时不自动回退到 `default` Connector，而是标记“待适配/待验证”并进行独立配方验证。

### 7.5 最终量化结论

1. 普通推理兼容性有 83 个部署行，但它们不是 83 个 PD 支持场景。
2. Wings 代码中有 79 个模型名，只有 29 个被 PD profile 机械命中，50 个无 large-EP profile；29 个命中项也不能当作已支持。
3. vLLM/NIXL 在 7 个架构类别中，5 类基础支持、1 类未验证、1 类明确不支持。
4. vLLM 官方 Recipes 当前有 41 个模型族声明兼容 `pd_cluster`，排除 AMD-only 后为 93 条 NVIDIA variant 记录、92 个唯一权重 ID；31 个使用共享策略，10 个有模型专项 override。
5. 41 个模型族中有 39 个默认 NVIDIA `pd_cluster.json` 可取得、2 个未生成；39 个已生成项中又有 2 个 H200 共置 DEP 命令存在 GPU 数与本地 DP 数冲突。默认 PD 组合是 `NixlConnector` + NIXL + `vllm-router`，不是 Mooncake。
6. GLM-4.5、4.6、4.7、5、5.1、5.2 均有 vLLM 官方 Recipe；GLM-5.2 默认是 `zai-org/GLM-5.2-FP8`、vLLM 0.23.0、16×H200、P/D 各 TP8、NIXL、round-robin Router。
7. vLLM-Ascend 当前已找到 14 个具体模型/硬件 PD 方案组；其中 Wings 只有 2 组基础方向对齐，6 组需要修订或拆分，6 组无精确 profile。
8. 在这 14 个“社区已有方案”的 Ascend 场景中，12 个（85.7%）的直接阻塞点在 Wings 适配/配方对齐，而不是社区没有 PD 方案。
9. Connector 不应自由互换：vLLM Recipes 默认以 NIXL 完成 P→D KV 传输，Mooncake Store 仅在共享缓存/Offload 需求下通过 MultiConnector 叠加；Ascend 常规场景以 V1 为主，DeepSeek-R1 可选成套 Layerwise，DeepSeek-V4 使用 Hybrid，GLM-5.2 A2 使用 V1+AscendStore MultiConnector。可以统一选择规则，不能统一 Connector 实现。
