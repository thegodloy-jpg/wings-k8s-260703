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

### 3.1 vLLM/NVIDIA：NIXL 架构能力 + 部分模型级 Recipe

vLLM 当前的 [NixlConnector Compatibility Matrix](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/) 主要给出架构能力；NVIDIA Dynamo 的 [Production-Ready Recipes](https://github.com/ai-dynamo/dynamo/blob/main/recipes/README.md) 又补充了一批具体模型、硬件和 P/D 拓扑。两类证据必须分开：架构矩阵说明“机制能否支持”，模型 Recipe 才说明“某一组合如何部署”。

**NVIDIA 官方 Dynamo 的 vLLM PD Recipe 主线使用 `NixlConnector`，不是 `MooncakeConnector`。** Dynamo 用 NIXL 将 KV 从 P 端 GPU VRAM 直接传到 D 端 GPU VRAM。vLLM 上游确实另外提供 `MooncakeConnector` 的 Qwen2.5-7B 1P1D 通用示例，但那不是 NVIDIA Dynamo 模型 Recipe 的默认方案。Qwen3-32B Recipe 中的 “Mooncake conversation trace” 只是压测数据集名称，其 `deploy.yaml` 明确使用 `NixlConnector`。

| 架构类型 | 基础 PD | 主要限制 | 对我们的结论 |
|---|---:|---|---|
| Dense Transformer | 支持 | P/D 版本、模型哈希、KV dtype、Attention backend 等必须兼容 | 架构级候选，不是逐模型产品结论 |
| MLA / Sparse MLA | 支持 | 异构 TP、KV 布局和推测配置受限 | DeepSeek 系列仍需精确模型验证 |
| MoE | 支持 | 具体权重、量化、TP/DP 和调度组合需验证 | 不能只因为是 MoE 就宣称 PD 支持 |
| Hybrid SSM/Mamba | 基础 PD 支持 | 异构 TP 开发中，当前要求 P TP = D TP；Cross-layer 和异构 block size 不支持 | Qwen3.5/3.6 不能套用普通 Dense/MoE 结论 |
| Multimodal | 未验证 | 完整多模态请求路径未确认 | Kimi/MiniMax 的文本主干能力不等于多模态 PD 可用 |
| Encoder-Decoder | 不支持 | 官方矩阵明确不支持 | 属于社区硬限制 |

NIXL 还要求 P/D 的 vLLM/NIXL 版本、模型架构、dtype、KV heads/head size/layers、Attention backend、KV cache dtype 和推测解码方法兼容。跨机时需要可路由的 side-channel host，每个 worker 需要唯一端口，UCX/LIBFABRIC 等传输后端也必须进入物料。详见 [NixlConnector Usage Guide](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)。

### 3.2 NVIDIA Dynamo：官方 PD 必须按 Backend 拆分

审计基线固定为 2026-08-13 的 [`ai-dynamo/dynamo@22e80d2`](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes)。按路径包含 `disagg`/`disaggregated` 的显式部署清单统计，官方仓库不是只有 vLLM：

| Backend | PD `deploy*.yaml` 数 | 与当前兼容模型有关的典型方案 | 本文处理方式 |
|---|---:|---|---|
| vLLM | 28；另有 Qwen3-32B 的 5 个云厂商 overlay | Qwen3-32B、DeepSeek-R1、DeepSeek-V4-Flash/Pro | 按当前 X86 36 模型做精确 ID 和产品组合比对 |
| SGLang | 11 | DeepSeek-R1、DeepSeek-V4-Pro、GLM-5/5.2 | GLM 单独列出；不能混入 vLLM 4/36 分母 |
| TensorRT-LLM | 12 | DeepSeek-V3.2、Kimi-K2.5、Qwen3-235B/32B 等 | 记录为其他 backend 官方方案，不能当成 Wings vLLM 已支持 |

这些数字是固定提交下的 manifest 库存，不是模型支持率；同一模型可能有多个硬件、网络或云厂商变体。

#### 3.2.1 vLLM PD：当前 Wings 兼容模型子集

本小节回答“当前 Wings 兼容模型是否有可直接参考的 NVIDIA vLLM PD 方案”。只纳入 vLLM backend，优先列当前兼容表中的精确模型 ID，再列会被误认为可复用的同族权重；其他 backend 不进入“4/36”支持率分母。

在该固定版本中，按 `recipes` 路径包含 `disagg`/`disaggregated` 统计，共有 **28 个 vLLM PD `deploy.yaml`**，另有 Qwen3-32B 的 **5 个云厂商 overlay**。下表覆盖逐项比对所需的当前精确模型、容易误继承的同族权重，以及已单独要求核查的 Kimi-K3，共 **14 个 deploy manifest + 5 个 overlay**；Llama 的 GAIE 只是同一模型/拓扑的集成变体，不另算模型方案。模型名链接到实际权重页，Recipe 链接到对应配置；Connector、KV 数据面、Router 和 EP 通信要求分开描述。

| 模型/示例权重 | 官方 Recipe / 配置 | 官方硬件与物理形态 | P 全局拓扑 | D 全局拓扑 | Connector / 网络 / Router | 对当前兼容表的结论 |
|---|---|---|---|---|---|---|
| [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) | [Disagg KV Router](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b/vllm/disagg-kv-router)；Feature Recipe，部署✅/perf✅ | 16×H200，2 节点 | 6P，每个 TP2 | 2D，每个 TP2 | `NixlConnector`；NIXL + RDMA；KV-aware Router | 模型 ID 精确；当前产品为 4×NL02 单机，资源和拓扑不匹配 |
| [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) | [Cloud Provider Overlays](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b/vllm/cloud-providers)；Functional，5 个部署/无 perf | 8 GPU，1P1D；AKS IB、AWS EFA、GKE RoCE、Nebius IB、Nscale IB | 1P，TP4 | 1D，TP4 | `NixlConnector`；UCX 或 EFA/libfabric；未显式启用 KV Router | 功能基线未压测；具体 GPU 和网络资源取决于 provider overlay，不能替代当前产品组合验收 |
| [`Qwen/Qwen3-32B-FP8`](https://huggingface.co/Qwen/Qwen3-32B-FP8) | [vLLM Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b-fp8/vllm/disagg)；Production-ready，部署✅/perf✅ | 8×A100，单节点 | 2P，每个 TP2 | 1D，TP4 | `NixlConnector`；NIXL + UCX/RDMA；未显式启用 KV Router | 当前兼容表是 Qwen3-32B BF16，不是该 FP8 权重 |
| [`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`](https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic) | [单节点](https://github.com/ai-dynamo/dynamo/tree/main/recipes/llama-3-70b/vllm/disagg-single-node) / [多节点](https://github.com/ai-dynamo/dynamo/tree/main/recipes/llama-3-70b/vllm/disagg-multi-node)；Production-ready，部署✅/perf✅ | 8×H100/H200 单节点；或 16×H100/H200 两节点 | 单节点 2P×TP2；多节点 1P×TP8 | 单节点 1D×TP4；多节点 1D×TP8 | `NixlConnector`；单节点 NIXL 或跨节点高速网络；未显式启用 KV Router | 当前兼容表是 Llama-3-8B/Distill 权重，不是该 70B FP8 权重 |
| [`Qwen/Qwen3.5-122B-A10B-FP8`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) | [H200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3.5-122b/fp8/vllm/disagg-h200-agentic)；Production-ready，部署✅/perf✅ | 3×H200 | 1P，TP1 | 2D，每个 TP1 | `NixlConnector`；NIXL + IB/RDMA；KV-aware Router | 需要 DS conv state；无 MTP；D 关闭 Async Scheduling；当前产品 BF16 不精确匹配 |
| [`nvidia/Qwen3.5-122B-A10B-NVFP4`](https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4) | [B200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3.5-122b/nvfp4/vllm/disagg-b200-agentic)；Production-ready，部署✅/perf✅ | 3×B200 | 1P，TP1 | 2D，每个 TP1 | `NixlConnector`；NIXL + IB/RDMA；KV-aware Router | 同上；不能扩展成 397B-NVFP4 Recipe |
| [`deepseek-ai/DeepSeek-R1`](https://huggingface.co/deepseek-ai/DeepSeek-R1) | [vLLM Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-r1/vllm/disagg)；Production-ready，部署✅/无 perf | 32×H100/H200，4 节点 | 1P，跨 2 节点×8 GPU，DP16+EP16+TP1 | 1D，跨 2 节点×8 GPU，DP16+EP16+TP1 | P→D 使用 `NixlConnector` + NIXL/RDMA；DEP 的 DeepEP/NVSHMEM 另要求 IBGDA；未显式启用 KV Router | 模型 ID 精确；IBGDA 不是 KV Connector；当前 8×NH02 单机记录不能直接复用 |
| [`nvidia/DeepSeek-V4-Flash-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4) | [B200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-flash/vllm/disagg-b200-agentic)；Production-ready，部署✅/Benchmark✅ | 12×B200 | 2P，每个 TP4 | 1D，TP4 | `NixlConnector`；NIXL + UCX/GDR；KV-aware Router | 与 `nv-community/...` 仅同模型族；namespace、硬件和卡数均需重新验证 |
| [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | [H200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-flash/vllm/disagg-h200-agentic)；Production-ready，部署✅/Benchmark✅ | 28×H200 | 4P，每个 DP4+TP1+EP | 3D，每个 DP4+TP1+EP | `NixlConnector`；NIXL + UCX/GDR；KV-aware Router | 模型 ID 精确；当前 FP4、8×NH02 单机组合不精确 |
| [`nvidia/DeepSeek-V4-Pro-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4-Pro-NVFP4) | [B200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg-b200-agentic)；Production-ready，部署✅/Benchmark✅ | 16×B200 | 1P，TP8+EP | 1D，TP8+EP | `NixlConnector`；NIXL + UCX/GDR；KV-aware Router | 当前产品不是该 NVIDIA 权重 |
| [`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | [H200 Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg-h200-agentic)；Production-ready，部署✅/Benchmark✅ | 32×H200 | 1P，TP8+EP | 3D，每个 TP8+EP | `NixlConnector`；NIXL + UCX/GDR；KV-aware Router | 模型 ID 精确；当前 FP4、8×NH02 单机组合不精确；官方 H200 更推荐聚合方案 |
| [`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | [GB200 Experimental Disagg](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-v4/deepseek-v4-pro/vllm/disagg/gb200)；Experimental，部署✅/无 perf | 16×GB200，4 个 NVL4 节点 | 1P，跨 2 节点，DP8+TP1+EP | 1D，跨 2 节点，DP8+TP1+EP | `NixlConnector`；KV bulk 走 MNNVL/`cuda_ipc`，UCX TCP 仅作 active-message 控制面；DRA `ComputeDomain`；未显式启用 KV Router | 模型 ID 精确，但它是实验方案；当前 8×NH02 单机组合不匹配，且依赖自定义镜像、`max_model_len=9280` |
| [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) | [GB300](https://github.com/ai-dynamo/dynamo/tree/main/recipes/kimi-k3/vllm/disagg-gb300-agentic) / [GB200](https://github.com/ai-dynamo/dynamo/tree/main/recipes/kimi-k3/vllm/disagg-gb200-agentic) Disagg；Production-ready，部署✅/无 perf | 24×GB300 或 32×GB200 | GB300 1P×TP8；GB200 1P×TP16 | GB300 2D×TP8；GB200 1D×TP16 | `NixlConnector`；NIXL over MNNVL/RDMA；KV-aware Router；DRA `ComputeDomain` | 有精确模型 Recipe，但不在当前 X86 文本生成 36 模型中，也没有同等产品组合 |

当前 X86 表有 36 个去重文本生成模型：**4 个命中相同模型 ID 的 NVIDIA vLLM PD Recipe**，即 Qwen3-32B、DeepSeek-R1、DeepSeek-V4-Flash、DeepSeek-V4-Pro；**32 个没有相同模型 ID；0 个与产品记录的“模型+精度+卡型/卡数+P/D 拓扑”完整一致。** 新增 DeepSeek-V4-Pro GB200 只增加同一模型的实验拓扑，不会把“4 个模型 ID”变成 5。这里的 0 表示不能直接复制官方 Recipe，不表示社区明确禁止。

仓库内另有 **13 个**既不属于当前 X86 36 模型、也未被选作当前同族/Kimi 对照项的 vLLM PD manifest，因而不纳入上述 4/36 分母：[`GPT-OSS-120B`](https://github.com/ai-dynamo/dynamo/tree/main/recipes/gpt-oss-120b/vllm) 2 个、[`Nemotron-3-Ultra`](https://github.com/ai-dynamo/dynamo/tree/main/recipes/nemotron-3-ultra/vllm) 1 个、[`Nemotron-3.5-Lightning`](https://github.com/ai-dynamo/dynamo/tree/main/recipes/nemotron-3.5-lightning/vllm) 9 个、[`Qwen3-VL-32B-Instruct-FP8` 异构 PD](https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-vl-32b-fp8/vllm/hetero_hardware_disagg) 1 个。其硬件、P/D 组合和排除理由见明细文档；不能把“本节未展开”解释成 Dynamo 官方没有方案。

#### 3.2.2 GLM：官方有 SGLang PD，不是 vLLM PD

上一版没有展开 GLM，是因为 3.2 的分母被限定为 vLLM；但从“NVIDIA 官方方案是否存在”的总口径看，这会造成缺漏。Dynamo 当前提供 **4 个 GLM SGLang PD manifest**：

| 官方模型/权重 | 官方 Recipe | 硬件与 P/D 拓扑 | KV 传输 / Router | 对当前兼容表的结论 |
|---|---|---|---|---|
| [`nvidia/GLM-5-NVFP4`](https://huggingface.co/nvidia/GLM-5-NVFP4) | [GB200 UCX](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5-nvfp4/sglang/disagg) / [AWS EFA](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5-nvfp4/sglang/disagg/efa)；2 个 validated manifest，均有 perf | 20×GB200、5 节点；1P×4 GPU，TP4；1D 跨 4 节点×4 GPU，TP16/DP16/EP16 | SGLang `nixl`；UCX/MNNVL 或 NIXL LIBFABRIC/EFA；manifest 未显式启用 KV-aware Router | 与 X86 的 `ZhipuAI/GLM-5-FP8` 只是同模型族；namespace、NVFP4/FP8、SGLang/vLLM、GB200/NH02 均不同 |
| [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) | [B200 agentic disagg](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5.2/sglang/disagg-b200-agentic)；validated，有 benchmark | manifest 实际为 20×B200：3P，每个 4 GPU、TP4/DP4/EP4；1D×8 GPU、TP8/DP8 | SGLang `nixl` + UCX/IB；KV-aware Router；P 开 200 GiB HiCache | 当前 X86 36 模型没有 GLM-5.2；不能给 GLM-5/5.1 继承 |
| [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) | [H200 agentic disagg](https://github.com/ai-dynamo/dynamo/tree/22e80d275e85ef537ebed044ea7c62389be7c281/recipes/glm-5.2/sglang/disagg-h200-agentic)；validated，有 benchmark | 16×H200；1P×8 GPU、TP8/DP1/EP8；1D×8 GPU、TP8/DP8/EP1 | SGLang `nixl` + UCX/IB；KV-aware Router | 当前 X86 36 模型没有 GLM-5.2；它也不是 GLM-5.1 Recipe |

因此，GLM 的正确结论不是“社区没有 NVIDIA PD”，而是：**社区已经给出 GLM-5/5.2 的 SGLang+NIXL 方案；当前 Wings NVIDIA 产品记录是 vLLM，且 GLM-5 的权重/精度/卡型不一致，GLM-4.7、GLM-5.1 又没有精确官方 Recipe。** 当前阻塞首先是 backend 与产品组合不匹配；若决定引入 SGLang，还需新增运行时、镜像、Parser、NIXL bootstrap、Router 和组网适配，之后才进入同等硬件验证。

#### 3.2.3 其他 Backend 遗漏项闭环

继续从全量 manifest 反查后，除 GLM 外还有 7 个 SGLang 和 12 个 TensorRT-LLM manifest。与当前 X86 模型相关的组合不能只写成一句“其他 backend”，但也不能混入 vLLM 产品支持率：

| 与当前兼容表的关系 | 遗漏的官方 manifest | 官方组合摘要 | 结论 |
|---|---:|---|---|
| 精确模型 `deepseek-ai/DeepSeek-R1` | SGLang 2 + TensorRT-LLM 1 | SGLang：16×H200 TP8 1P1D、32×H200 TP16 1P1D；TRT-LLM：36×GB200，1 个 4-GPU P 节点+8 个 4-GPU D 节点 | 证明该模型有多 backend 社区方案；当前 8×NH02 vLLM 产品组合仍不匹配 |
| 精确模型 `deepseek-ai/DeepSeek-V4-Pro` | SGLang 2 | B200：16 卡 1P1D、每侧 TP8；GB200：16 卡 1P1D、每侧跨 2 节点 TP8 | 是 Day-0 Experimental SGLang 方案；不能替代当前 Wings vLLM 适配和验收 |
| 同模型族但非当前精确权重 | TensorRT-LLM 8 | `nvidia/DeepSeek-V3.2-NVFP4` 1 个；`Qwen/Qwen3-235B-A22B-FP8` 6 个；`Qwen/Qwen3-32B-FP8` 1 个 | 当前表分别是公共/不同量化权重；backend、精度、硬件和拓扑均需独立登记 |
| 不在当前 X86 36 模型 | SGLang 3 + TensorRT-LLM 3 | SGLang：Nemotron-3-Super 1、Qwen3.8 2；TRT-LLM：GPT-OSS-120B、Kimi-K2.5、Nemotron-3-Super 各 1 | 只用于闭合官方库存，不进入当前产品分母 |

由此闭环：SGLang 为 `GLM 4 + 其他 7 = 11`；TensorRT-LLM 为 `当前相关 9 + 当前表外 3 = 12`。这些补充不会新增 vLLM 的精确模型 ID 数：DeepSeek-R1、DeepSeek-V4-Pro 已经在 vLLM 的 4 个命中模型中，其余均为不同权重或当前表外模型。完整模型链接、P/D 组合和数据面见明细文档 4.3.4–4.3.5。

NVIDIA 生产跨机 PD 还要求 RDMA/IB/RoCE、RDMA device plugin、Pod 的 `rdma/ib` 和 `IPC_LOCK`、正确的 UCX NIC/transport、Dynamo Frontend/Router 与 ETCD/NATS。GB200/GB300 的 MNNVL 方案还依赖 VMM KV 注册和 DRA `ComputeDomain`。这些均属于 Recipe 的组成部分，不是换一个 Connector 名称即可省略。

逐模型权重、两套 Llama 拓扑、Qwen3.5 Hybrid 约束、GLM SGLang 方案、产品清单映射和官方逐条链接见 [PD 分离兼容性与 Mooncake 方案清单](./PD分离兼容性与Mooncake方案清单.md#43-nvidia-dynamo-官方模型级-pd-recipe按-backend-分层)。

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

### 3.4 只有架构能力或未找到精确模型级 PD 配方

当前兼容列表中的下列典型模型，没有找到与具体权重、卡型和 P/D 组合完全对应的官方方案：

- DeepSeek-Coder-V2，以及部分 Distill 模型；
- GLM-4/GLM-4.7 精确权重；
- Qwen2.5-32B、QwQ、Qwen3-8B/14B/30B/235B、Qwen3-Next 和 AgentWorld；
- Qwen3.5-35B/397B、Qwen3.6-35B；122B 只有不同量化权重的 NVIDIA Recipe；
- Kimi-K2.7；Kimi-K3 有 NVIDIA Recipe，但没有当前产品同等组合；
- MiniMax-M2.5/MiniMax-M3 精确方案；
- 当前列表中的 Llama 精确模型。

这类场景可以进入内部预研，但应标记为“架构级候选/模型级官方配方未找到”，这部分才是社区方案覆盖不足。

## 4. 不可用原因和责任边界

| 原因类别 | 判定标准 | 典型模型/问题 | 是否社区限制 |
|---|---|---|---|
| 社区硬限制 | 官方矩阵明确不支持，或官方同版本、同硬件、同命令仍稳定复现上游错误 | Encoder-Decoder；Hybrid Mamba 异构 TP/block size 限制 | 是 |
| 社区模型级方案缺口 | 只有架构矩阵，没有具体模型、权重、硬件和 P/D 命令 | Qwen3-30B/Next、Qwen3.6-35B、Kimi-K2.7、MiniMax-M3 等 | 是，但是方案缺口，不代表架构绝对不可用 |
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
> NVIDIA/vLLM 侧不是基于一个统一的 Mooncake 方案。vLLM 上游同时提供 NIXL 和 Mooncake Connector，但 NVIDIA Dynamo 的模型级 vLLM Recipe 主线使用 `NixlConnector`。Dynamo 已为 Qwen3-32B、Llama-3.3-70B、Qwen3.5-122B、DeepSeek-R1、DeepSeek-V4-Flash/Pro、Kimi-K3 等给出具体 P/D 方案，因此不能再概括为“只有架构级能力”。不过官方 Recipe 覆盖仍然很窄：当前 X86 36 个去重文本生成模型中，只有 4 个命中相同模型 ID，且 0 个与产品记录的精度、卡型/卡数和 P/D 拓扑完整一致。Hybrid SSM/Mamba 异构 TP、Encoder-Decoder 等仍有明确的社区功能限制。
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
| X86 去重文本生成模型对 NVIDIA Dynamo vLLM PD Recipe | 36 个模型 ID | 4 个命中相同模型 ID；0 个完整匹配产品组合 | 32 个无相同模型 ID Recipe；36 个均无完整产品组合 Recipe | 相同模型 ID：Qwen3-32B、DeepSeek-R1、DeepSeek-V4-Flash、DeepSeek-V4-Pro；完整组合还要求精度、卡型/卡数和 P/D 拓扑一致 |
| vLLM-Ascend 官方模型/硬件 PD 方案组 | 14 | 14 组都有官方方案 | 0 组 | 这 14 组就是第 3.3 节的逐行统计，不能用 `79-14` 推导其他 65 个模型明确不支持 |

“官方不支持多少个”必须分成两种状态：

1. **明确不支持**：当前可明确量化的是 NIXL 矩阵中 1 个架构类别，即 Encoder-Decoder。
2. **没有找到精确模型级方案**：NVIDIA 侧当前 X86 36 个去重文本生成模型中有 32 个没有相同模型 ID 的 Dynamo vLLM PD Recipe；其余 4 个也没有完整匹配产品精度、卡型/卡数和 P/D 拓扑。它们应标记“精确 Recipe 未找到/待移植验证”，不能标记“官方明确不支持”。

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
| NVIDIA Dense/MLA/Sparse MLA/MoE | `NixlConnector` | 不与 Ascend 14 组混算 | P/D 版本、模型哈希、KV dtype、Attention backend 和推测方法必须兼容；跨机配置 UCX/LIBFABRIC 和唯一 side-channel 端口 |
| NVIDIA Hybrid SSM/Mamba | `NixlConnector` + Hybrid 专项约束 | 不与 Ascend 14 组混算 | 当前 P TP = D TP；按 Recipe 设置 `VLLM_SSM_CONV_STATE_LAYOUT=DS`；Qwen3.5-122B PD 禁用 MTP、D 关闭 Async Scheduling；不使用 Cross-layer 和异构 block size |
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

1. 当前 Wings NVIDIA/vLLM 路径进入 `NixlConnector` 策略，不进入 Ascend Mooncake 默认值；SGLang/TensorRT-LLM 必须按各自 manifest 选择 NIXL、Mooncake 可选路径或 UCX/DEFAULT transceiver，不能沿用 vLLM 参数。
2. Ascend 必须先精确匹配“模型/权重+硬件+版本”的官方 recipe。
3. DeepSeek-V4 选 Hybrid；GLM-5.2 A2 选 V1+AscendStore MultiConnector；官方明确的 Layerwise 方案成套选 Layerwise；其余有官方 V1 配方的普通模型选 V1。
4. 没有精确官方 recipe 时不自动回退到 `default` Connector，而是标记“待适配/待验证”并进行独立配方验证。

### 7.5 最终量化结论

1. 普通推理兼容性有 83 个部署行，但它们不是 83 个 PD 支持场景。
2. Wings 代码中有 79 个模型名，只有 29 个被 PD profile 机械命中，50 个无 large-EP profile；29 个命中项也不能当作已支持。
3. vLLM/NIXL 在 7 个架构类别中，5 类基础支持、1 类未验证、1 类明确不支持。
4. NVIDIA Dynamo 官方 PD 不能只看 vLLM：固定审计提交中有 28 个 vLLM、11 个 SGLang、12 个 TensorRT-LLM 显式 PD manifest，另有 5 个 vLLM 云厂商 overlay；GLM 单独占 4 个 SGLang manifest。
5. NVIDIA 没有跨 backend 的统一 Connector：vLLM 模型 Recipe 主线使用 `NixlConnector`；SGLang 检入方案以 NIXL 为主，但 DeepSeek-R1 未显式固定 transfer backend，Nemotron-3-Super 还正式记录 Mooncake 替代路径；TensorRT-LLM 使用 UCX 或 `DEFAULT` cache transceiver。
6. 限定到当前 Wings vLLM 口径，X86 36 个去重文本生成模型中 4 个命中相同模型 ID，32 个没有相同模型 ID，0 个完整匹配产品组合。其他 backend 方案不改变这组 vLLM 统计。
7. vLLM-Ascend 当前已找到 14 个具体模型/硬件 PD 方案组；其中 Wings 只有 2 组基础方向对齐，6 组需要修订或拆分，6 组无精确 profile。
8. 在这 14 个“社区已有方案”的 Ascend 场景中，12 个（85.7%）的直接阻塞点在 Wings 适配/配方对齐，而不是社区没有 PD 方案。
9. Connector 不应自由互换：普通 Ascend 场景以 V1 为主，DeepSeek-R1 可选成套 Layerwise，DeepSeek-V4 使用 Hybrid，GLM-5.2 A2 使用 V1+AscendStore MultiConnector；NVIDIA 还必须继续按 backend 和具体 manifest 分流。可以统一选择规则，不能统一 Connector 实现。
