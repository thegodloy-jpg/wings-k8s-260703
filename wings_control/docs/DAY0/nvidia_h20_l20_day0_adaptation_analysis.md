# NVIDIA H20/L20 八场景适配说明

事实源：`wings_control/docs/AI Infra工作台_0 Day测试结果.xlsx`，2026-07-09 的 8 个 NVIDIA 场景。

## 适配原则

1. 只按 Excel 模型列的 8 个场景点对点适配，不扩展模型精度变体。
2. 普通拉起参数直接放入 `wings_control/config/defaults/nvidia_default.json`。
3. `smart_feature_whitelist.json` 只做特性触发白名单：MTP、IndexCache、native offload。
4. `kv_cache_dtype=fp8` 是普通参数，不作为 sparse；不新增 `fp8_kv`，不输出 `calculate_kv_scales`。
5. native offload 复用现有卸载 backend 解析链路；白名单只声明 `backend=native`，size 仍来自页面或 `auto` 自动计算。
6. 已有 NV 精度变体逻辑不归入本次 8 场景；新白名单行显式排除 `fp8/w4a8/w8a8/nvfp4`，避免误命中。

## 八场景落点

| 场景 | defaults | spec | sparse | offload |
| --- | --- | --- | --- | --- |
| GLM-5 / H20 | GLM CLI 基础参数、`glm47`、raw `-cc.pass_config.fuse_allreduce_rms=False` | `mtp`, token=3 | IndexCache topk=4 | native |
| GLM4.7 / H20 | GLM CLI 基础参数、`glm47` | `mtp`, token=1 | 无 | native |
| GLM5.1 / H20 | GLM CLI 基础参数、`glm47`、raw `-cc.pass_config.fuse_allreduce_rms=False` | `mtp`, token=3 | IndexCache topk=4 | native |
| Qwen3-Embedding-0.6B / L20 | `gpu_memory_utilization=0.9`, `kv_cache_dtype=fp8` | 无 | 无 | native |
| Qwen3.6-27B / L20 | `qwen3_coder`, `qwen3`, prefix cache, `kv_cache_dtype=fp8` | `mtp`, token=1 | 无 | native |
| Qwen3.6-35B-A3B / L20 | `qwen3_xml`, `qwen3`, prefix cache, `kv_cache_dtype=fp8` | `mtp`, token=3, `moe_backend=triton` | 无 | native |
| bge-large-zh-v1.5 / L20 | 基线参数、`gpu_memory_utilization=0.9` | 无 | 无 | 无 |
| bge-reranker-large / L20 | 基线参数、`gpu_memory_utilization=0.9` | 无 | 无 | 无 |

H20 同时配置 `h20-96` 和 `h20-141` 两个 card token；L20 配置 `l20`。

## 修改文件

| 文件 | 作用 |
| --- | --- |
| `wings_control/config/defaults/nvidia_default.json` | 放 8 场景普通默认参数 |
| `wings_control/config/smart_feature_whitelist.json` | 放 MTP、IndexCache、native offload 白名单 |
| `wings_control/engines/vllm_adapter.py` | MTP/IndexCache 走白名单；native 只扩展现有 offload backend 解析，不新增 Day0 native 分支 |
| `wings_control/core/config_loader.py` | offload 白名单 backend=native 时跳过 LMCache connector，GLM5.1 native backend 命中时不走旧禁用 guard |
| `wings_control/core/wings_entry.py` | native 白名单命中时不安装 LMCache patch |
| `wings_control/utils/model_utils.py` | 只补 `GLM4.7`、`GLM5.1` 这种非精度别名 |
| `wings_control/docs/features/reasoning_parser/reason_parser.yaml` | 补 parser/null 映射 |

## native offload 规则

- 页面开关开启 + offload 白名单命中 + `KV_MEM_OFFLOAD_SIZE` 有效，才输出 native CLI。
- `KV_MEM_OFFLOAD_SIZE=N`：输出 `--kv-offloading-size N`，节点级 GB，不按卡均分。
- `KV_MEM_OFFLOAD_SIZE=auto`：复用已有自动计算逻辑。
- 空值、非法值、`<=0`：直接不输出 native CLI，不回退 LMCache。
- H20=200GB、L20=20GB 只作为 Excel dry-run 验收输入，不是后端缺省值。

## 验收点

1. Qwen 三个 L20 场景有 `--kv-cache-dtype fp8`，无 `calculate_kv_scales`。
2. GLM-5/GLM5.1 有 IndexCache topk=4；GLM4.7 无 IndexCache。
3. MTP token：GLM-5=3，GLM4.7=1，GLM5.1=3，Qwen3.6-27B=1，Qwen3.6-35B-A3B=3。
4. Qwen3.6-35B-A3B 的 MTP JSON 包含 `moe_backend=triton`。
5. BGE 两个场景只走 defaults 基线，不进 spec/sparse/offload。
