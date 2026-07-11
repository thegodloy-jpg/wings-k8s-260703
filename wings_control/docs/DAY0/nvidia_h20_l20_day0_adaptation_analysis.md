# NVIDIA H20/L20 vLLM 0.23 Day0 场景适配说明

事实源：`wings_control/docs/AI Infra工作台_0 Day测试结果.xlsx`，以用户确认的 2026-07-09 NVIDIA 单机场景为准。

## 精确组合

| 场景 | 卡型 | 部署形态 | smart feature 落点 | 白名单 source |
| --- | --- | --- | --- | --- |
| DeepSeek V4-Flash-FP4+FP8 | G8600+H20 * 8 | 单机 | 复用现有 DeepSeek V4-Flash NV defaults/native/spec 逻辑；H20 spec 行标识为 vLLM 0.23 | `vllm-0.23` |
| GLM-5-FP8 | G8600+H20 * 8 | 单机 | MTP token=3、IndexCache topk=4、native offload | `vllm-0.23` |
| GLM4.7-FP8 | G8600+H20 * 8 | 单机 | MTP token=1、native offload | `vllm-0.23` |
| GLM5.1-FP8 | G8600+H20 * 8 | 单机 | MTP token=3、IndexCache topk=4、native offload | `vllm-0.23` |
| Qwen3-Embedding-0.6B-BF16 | G5500+L20、H20 | 单机 | native offload；L20 能力单向覆盖 H20 | `vllm-0.23` |
| Qwen3.6-27B-BF16 | G5500+L20、H20 | 单机 | MTP token=1、native offload；L20 能力单向覆盖 H20 | `vllm-0.23` |
| Qwen3.6-35B-A3B-BF16 | G5500+L20、H20 | 单机 | MTP token=3、`moe_backend=triton`、native offload；L20 能力单向覆盖 H20 | `vllm-0.23` |
| bge-large-zh-v1.5-BF16 | G5500+L20 | 单机 | 只走 defaults 基线，不进 spec/sparse/offload 白名单 | 无白名单行 |
| bge-reranker-large-BF16 | G5500+L20 | 单机 | 只走 defaults 基线，不进 spec/sparse/offload 白名单 | 无白名单行 |

H20 同时覆盖 `h20-96` 和 `h20-141` 两个 card token；L20 使用 `l20`。本次八场景内，L20 已有的 smart-feature 能力必须原样覆盖两个 H20 token，H20-only 能力不反向补到 L20。

表格场景名中的 `FP8` / `BF16` 是精度属性，不自动拼进白名单模型名。实际名称按开源启动命令：`zai-org/GLM-5`、`zai-org/GLM-4.7`、`zai-org/GLM-5.1`、`Qwen3-Embedding-0.6B`、`Qwen3.6-27B`、`Qwen3.6-35B-A3B`。

## source 字段口径

`source` 仅作为静态元信息，加载器不参与判定。当前收编版本统一为：

| 范围 | source |
| --- | --- |
| 本次 NVIDIA H20/L20 vLLM 0.23 smart-feature 白名单行 | `vllm-0.23` |
| 昨天收编的 Qwen3.5/Qwen3.6 vLLM-Ascend 0.21 smart-feature 白名单行 | `vllm-ascend-0.21` |

## 适配原则

1. 白名单相信上层传递的 `model_name` / `model_path` 准确，匹配必须同时满足 `engine + name_tokens + card_tokens`。
2. 白名单不解析独立精度字段，也不根据场景标签拼接 `FP8` / `BF16` 后缀；具有组织路径的 `name_tokens` 固定只写 `[组织/开源路径名, 准确模型名]` 两项，不增加第三种宽松拼写。
3. 只有开源模型名自身带量化后缀时才新增独立行，例如 Qwen W8A8、NVFP4；其他精度等待后续明确的开源模型名和需求。
4. 普通拉起参数继续放在 `wings_control/config/defaults/nvidia_default.json`；`kv_cache_dtype=fp8` 仍是普通参数，不作为 sparse。
5. native offload 继续复用现有卸载 backend 解析链路；白名单只声明 `backend=native`，内存 size 仍来自页面或 `auto` 自动计算。
6. 页面未填写 offload 内存时直接丢弃 native size，不引入 H20=200GB、L20=20GB 的隐式默认值。
7. 不输出 `calculate_kv_scales`，不新增 `fp8_kv` sparse 策略。
8. 卡型能力只做单向继承：L20 白名单行按相同策略分别补 `h20-96`、`h20-141`；H20 行不反向派生 L20。原本没有 smart feature 的 BGE 场景不新增空白名单行。

## 修改文件

| 文件 | 作用 |
| --- | --- |
| `wings_control/config/defaults/nvidia_default.json` | 承载本次 NVIDIA 场景中已有的普通默认参数 |
| `wings_control/config/smart_feature_whitelist.json` | 承载 MTP、IndexCache、native offload 白名单，按模型标识和卡型限定 vLLM 0.23 组合 |
| `wings_control/utils/model_utils.py` | 统一白名单匹配入口，复用 `model_name` / `model_path` 和卡型判断 |
| `wings_control/engines/vllm_adapter.py` | MTP/IndexCache/native 仍复用现有 resolver，不新增 Day0 native 分支 |
| `wings_control/core/config_loader.py` | 继续按白名单结果收口 smart feature，不改变 defaults 合并逻辑 |
| `wings_control/core/wings_entry.py` | native 命中时沿用现有 LMCache patch 跳过逻辑 |
| `wings_control/docs/features/reasoning_parser/reason_parser.yaml` | 维护 parser/null 映射 |

## 验收点

1. `zai-org/GLM-5.1`、`Qwen3.6-27B`、`Qwen3.6-35B-A3B` 等准确开源模型名命中对应行。
2. vLLM 0.23 的 GLM 行只保留 `[zai-org/...路径名, 准确模型名]` 两项，不派生 `glm5.1` 或 `zai-org/glm-5.1-fp8`。
3. `Qwen3.6-27B` 命中 L20 spec/offload，不新增 `Qwen3.6-27B-BF16` 派生行。
4. `Qwen3-Embedding-0.6B` 命中 offload，不新增 BF16 派生行。
5. `bge-large-zh-v1.5-BF16`、`bge-reranker-large-BF16` 不命中 spec/sparse/offload，只走 defaults 基线。
6. Qwen W8A8 / NVFP4 等开源名称自带后缀的场景继续命中各自独立行。
7. 同模型同 H20 场景优先使用 Excel Day0 参数，例如 `GLM4.7-FP8` 的 MTP token 为 1。
8. `smart_feature_whitelist.json` 中本次 NVIDIA 行 source 为 `vllm-0.23`，Qwen3.5/Qwen3.6 Ascend 0.21 收编行 source 为 `vllm-ascend-0.21`。
9. Qwen 三个 L20 场景在 `h20-96`、`h20-141` 上命中相同 smart feature；GLM 三个 H20-only 场景在 L20 上仍不命中。
