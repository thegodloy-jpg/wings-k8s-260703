# NVIDIA H20/L20 vLLM 0.23 Day0 场景适配说明

事实源：`wings_control/docs/AI Infra工作台_0 Day测试结果.xlsx`，以用户确认的 2026-07-09 NVIDIA 单机场景为准。

## 精确组合

| 场景 | 卡型 | 部署形态 | smart feature 落点 | 白名单 source |
| --- | --- | --- | --- | --- |
| DeepSeek V4-Flash-FP4+FP8 | G8600+H20 * 8 | 单机 | 复用现有 DeepSeek V4-Flash NV defaults/native/spec 逻辑；H20 spec 行标识为 vLLM 0.23 | `vllm-0.23` |
| GLM-5-FP8 | G8600+H20 * 8 | 单机 | MTP token=3、IndexCache topk=4、native offload | `vllm-0.23` |
| GLM4.7-FP8 | G8600+H20 * 8 | 单机 | MTP token=1、native offload | `vllm-0.23` |
| GLM5.1-FP8 | G8600+H20 * 8 | 单机 | MTP token=3、IndexCache topk=4、native offload | `vllm-0.23` |
| Qwen3-Embedding-0.6B-BF16 | G5500+L20 | 单机 | native offload | `vllm-0.23` |
| Qwen3.6-27B-BF16 | G5500+L20 | 单机 | MTP token=1、native offload | `vllm-0.23` |
| Qwen3.6-35B-A3B-BF16 | G5500+L20 | 单机 | MTP token=3、`moe_backend=triton`、native offload | `vllm-0.23` |
| bge-large-zh-v1.5-BF16 | G5500+L20 | 单机 | 只走 defaults 基线，不进 spec/sparse/offload 白名单 | 无白名单行 |
| bge-reranker-large-BF16 | G5500+L20 | 单机 | 只走 defaults 基线，不进 spec/sparse/offload 白名单 | 无白名单行 |

H20 同时覆盖 `h20-96` 和 `h20-141` 两个 card token；L20 使用 `l20`。

## source 字段口径

`source` 仅作为静态元信息，加载器不参与判定。当前收编版本统一为：

| 范围 | source |
| --- | --- |
| 本次 NVIDIA H20/L20 vLLM 0.23 smart-feature 白名单行 | `vllm-0.23` |
| 昨天收编的 Qwen3.5/Qwen3.6 vLLM-Ascend 0.21 smart-feature 白名单行 | `vllm-ascend-0.21` |

## 适配原则

1. 白名单只允许上述精确组合触发新增 Day0 特性，不用 `exclude_name_tokens` 反向排除精度变体。
2. 白名单使用静态字段 `precision_tokens`，匹配必须同时满足 `engine + name_tokens + precision_tokens + card_tokens`。
3. `exclude_name_tokens` 只保留型号边界用途，例如 GLM-5 不误命中 GLM-5.1/GLM-5.2，不再承载 FP8/BF16/NVFP4 这类精度排除。
4. 普通拉起参数继续放在 `wings_control/config/defaults/nvidia_default.json`；`kv_cache_dtype=fp8` 仍是普通参数，不作为 sparse。
5. native offload 继续复用现有卸载 backend 解析链路；白名单只声明 `backend=native`，内存 size 仍来自页面或 `auto` 自动计算。
6. 页面未填写 offload 内存时直接丢弃 native size，不引入 H20=200GB、L20=20GB 的隐式默认值。
7. 不输出 `calculate_kv_scales`，不新增 `fp8_kv` sparse 策略。

## 修改文件

| 文件 | 作用 |
| --- | --- |
| `wings_control/config/defaults/nvidia_default.json` | 承载本次 NVIDIA 场景中已有的普通默认参数 |
| `wings_control/config/smart_feature_whitelist.json` | 承载 MTP、IndexCache、native offload 白名单，并用 `precision_tokens` 收窄 vLLM 0.23 组合 |
| `wings_control/utils/model_utils.py` | 统一白名单匹配入口，增加 precision token 解析和命中判断 |
| `wings_control/engines/vllm_adapter.py` | MTP/IndexCache/native 仍复用现有 resolver，不新增 Day0 native 分支 |
| `wings_control/core/config_loader.py` | 继续按白名单结果收口 smart feature，不改变 defaults 合并逻辑 |
| `wings_control/core/wings_entry.py` | native 命中时沿用现有 LMCache patch 跳过逻辑 |
| `wings_control/docs/features/reasoning_parser/reason_parser.yaml` | 维护 parser/null 映射 |

## 验收点

1. `GLM5.1`、`Qwen3.6-27B` 这类未带精度后缀的模型名不命中 vLLM 0.23 Day0 白名单。
2. `GLM5.1-FP8` 命中 H20 vLLM 0.23 的 spec/sparse/offload。
3. `Qwen3.6-27B-BF16` 命中 L20 vLLM 0.23 的 spec/offload；`Qwen3.6-27B-FP8` 不命中。
4. `Qwen3-Embedding-0.6B-BF16` 只命中 offload。
5. `bge-large-zh-v1.5-BF16`、`bge-reranker-large-BF16` 不命中 spec/sparse/offload，只走 defaults 基线。
6. 同模型同 H20 场景优先使用 Excel Day0 参数，例如 `GLM4.7-FP8` 的 MTP token 为 1。
7. `smart_feature_whitelist.json` 中本次 NVIDIA 行 source 为 `vllm-0.23`，Qwen3.5/Qwen3.6 Ascend 0.21 收编行 source 为 `vllm-ascend-0.21`。
