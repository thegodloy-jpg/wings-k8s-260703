# Qwen3.6 1P1D PD 分离问题与方案边界

> 结论：PD 可以统一拓扑和配置框架，但不存在跨模型、connector 和版本通用的运行 recipe。

## 1. 本次问题

当前组合为 Qwen3.6-35B-A3B、vLLM 0.21.0 + vLLM-Ascend runtime、Ascend 1P1D、`MooncakeConnectorV1`。问题分为三层：

| 阶段 | 现象 | 判断 |
| --- | --- | --- |
| 1P1D 拓扑校验 | `KV transfer 'decode' config has a conflicting tensor parallel size. Expected 2, but got 1.` | P/D peer TP 与实际单 service TP 不一致，属于拓扑配置问题 |
| KV cache 初始化 | `Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type.` | Qwen3.6 同时存在 Attention 与 GDN/Mamba 类 cache spec，关闭 HMA 后无法统一 |
| 请求执行 | 加入 `--no-disable-hybrid-kv-cache-manager` 后服务启动，但预测挂起 | HMA 只解决初始化；当前 connector 的多 cache group 传输或完成通知仍未验证通过 |

所以，全量打开 HMA 不是完整修复，只是把故障从启动阶段后移到了请求阶段。

## 2. 社区是否有完全相同的报错

### 2.1 HMA 启动错误：有完全相同的错误字符串和调用点

本次核心异常为：

```text
ValueError: Hybrid KV cache manager is disabled but failed to convert
the KV cache specs to one unified type.
```

社区存在完全相同的异常：

- [vLLM #36463](https://github.com/vllm-project/vllm/issues/36463) 的 Qwen3.5-27B 在关闭 HMA 后，进入同一个 `unify_hybrid_kv_cache_specs()` 并抛出完全相同的 `ValueError`。其环境是 NVIDIA + CPU KV offload，不是 Ascend PD，但错误机制一致：混合模型的多种 KV cache spec 无法被强制统一。
- [vLLM #23161](https://github.com/vllm-project/vllm/issues/23161) 讨论 V1 引擎对混合 KV layout 的限制，也记录了同一异常；其中还包含 Qwen3.6 GatedDeltaNet/Mamba cache 处理问题，但 Qwen3.6 案例的最终异常不同。

准确结论是：

- **错误文本和 vLLM 调用路径有完全相同的公开案例；**
- **尚未找到 Qwen3.6 + Ascend + MooncakeConnectorV1 + 1P1D 条件全部相同的公开复现。**

### 2.2 TP 冲突：未找到完全相同的公开案例

没有检索到与 `Expected 2, but got 1` 完全一致且同为 Qwen3.6 1P1D 的公开记录。该错误发生在 `KVTransferConfig` 校验阶段，说明 P/D 描述的 peer TP 与实际 TP 不一致，属于 Wings 1P1D 配置生成问题，不是 HMA 或 KV 数据传输问题。

### 2.3 打开 HMA 后请求挂起：只有同机制证据，不能称为完全相同

目前请求挂起没有对应的异常栈，因此不能与某个社区问题做“一模一样”的确认。只能确认上游后续持续修复相同风险区域：

- [vLLM-Ascend #8850](https://github.com/vllm-project/vllm-ascend/pull/8850) 为普通 Mooncake connector 增加多 cache group/Hybrid Attention 支持；
- [vLLM-Ascend #10342](https://github.com/vllm-project/vllm-ascend/pull/10342) 又修复 hybrid KV 的 block stride、压缩传输计算和 Mamba token 对齐；
- [v0.22.1rc1 发布说明](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.22.1rc1) 将这些列为 v0.21 之后的 Hybrid KV 修复。

这些证据说明上游 v0.21 版本线附近的 Hybrid KV 传输仍在演进，但不能在没有请求期 P/D 日志的情况下断言当前挂起就是其中某一个缺陷。

## 3. 当前 1P1D 方案边界

当前 Wings 1P1D 链路是：

```text
PD_ROLE=P/D
  -> _get_pd_config()
     默认 MooncakeConnectorV1，生成 P/D TP/DP 映射
  -> _apply_pd_external_lb()
     DP=1 只补 TP/DP，不读取大 EP recipe
  -> _apply_pd_final_guard()
     最终重申 TP/DP，当前还会对所有 vLLM PD 强制保留 HMA
```

对应代码见 [`config_loader.py`](../core/config_loader.py)。大 EP 则按模型架构从 [`pd_config.json`](../config/defaults/pd_config.json) 选择 connector、端口和 P/D 专属参数。

`1P1D` 只描述部署拓扑，不代表所有模型都应该使用同一个 connector 或 HMA 策略。当前 Qwen3.6 已证明：

- 统一 TP/DP 处理可以解决拓扑冲突；
- 全局 HMA 可以解决部分混合模型的启动；
- 但 connector 是否能正确传输所有 cache group，仍然是模型和版本相关能力。

## 4. 为什么不存在统一 PD 运行方案

| 变化维度 | 对 PD 的影响 |
| --- | --- |
| 模型架构 | 普通 Attention、MLA、Attention + GDN/Mamba 的 KV 状态不同 |
| Connector | V1、Layerwise、Hybrid、NIXL 的传输和完成通知语义不同 |
| 拓扑 | 1P1D、大 EP、P/D 异构 TP 会改变 rank 和 KV 分片映射 |
| 引擎版本 | HMA、多 cache group、stride 和 token 对齐能力持续变化 |
| 特性组合 | MTP、Prefix Cache、Chunked Prefill、Graph Mode 会改变 cache 生命周期 |

因此正确方向是统一选择框架：

```text
PD Profile = 模型 + 引擎版本 + 硬件 + 拓扑 + Connector 能力 + 特性组合
```

而不是统一为：

```text
所有 PD = MooncakeConnectorV1 + 全局 HMA
```

## 5. 当前建议

1. Qwen3.6 当前状态标记为“可启动、推理未通过”，不能声明支持 PD；
2. 保留 1P1D 的 TP/DP 隔离，不把大 EP 全局拓扑重新灌入；
3. 将 HMA 从全局布尔值收敛为模型 + connector + 版本能力项；
4. 验证 `MooncakeLayerwiseConnector + HMA` 时必须同时切换匹配的 proxy/metaserver、端口和请求流程，不能只改 connector 名称；
5. 另一条路线是成套升级匹配的 vLLM 与 vLLM-Ascend，不能只升级其中一个；
6. 支持结论至少要通过：P/D 启动、单请求、多轮、并发和长上下文，不能以 Pod Running 作为验收结果。

**最终结论：PD 可以统一角色、拓扑、profile 选择和验收流程，但最终 connector 与运行参数必须按模型和版本分别维护。**
