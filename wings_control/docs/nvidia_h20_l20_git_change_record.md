# NVIDIA H20/L20 Day0 修改说明

## 总体目标

本次修改把 NVIDIA H20/L20 Day0 场景接入现有配置链路，而不是新增一套 H20/L20 专用启动分支。

核心链路：

```text
nvidia_default.json
  -> config_loader.py 选择普通启动 defaults

smart_feature_whitelist.json
  -> model_utils.py / config_loader.py 收口 smart feature
  -> vllm_adapter.py 渲染 vLLM 启动命令
```

最终效果：H20/L20 场景通过数据配置表达，命令生成继续复用现有 resolver，避免重复分支和隐式默认值。

## 修改摘要

| 改动点 | 修改了什么 | 为什么修改 | 修改后的效果 |
| --- | --- | --- | --- |
| NVIDIA defaults 增加卡型门禁 | 在 `wings_control/config/defaults/nvidia_default.json` 的 exact model defaults 增加 `card_tokens`；在 `wings_control/core/config_loader.py` 增加卡型匹配逻辑。 | 原来 exact defaults 主要按模型名命中，H20/L20 Day0 参数可能误套到其它 NVIDIA 卡。 | GLM H20-only 参数只在 `h20-96` / `h20-141` 生效；Qwen/BGE 的 L20 参数可按规则覆盖 H20；A100、B200、RTX Pro 等其它卡不会误触 Day0 defaults。 |
| smart feature 白名单收敛 | 调整 `wings_control/config/smart_feature_whitelist.json`，按 `spec` / `sparse` / `offload` 三表补 H20/L20 行；移除 `precision_tokens`；在 `wings_control/utils/model_utils.py` 简化 matcher。 | 上层下发的 `model_name` / `model_path` 已经是准确模型名，代码再解析 BF16/FP8 会引入第二套判断来源，并导致 L20 白名单 miss。 | 白名单只按 `engine + name_tokens + card_tokens` 判断；Qwen3.6 / Qwen-Embedding 的 L20 能力单向补到 H20；GLM H20-only 能力不反向补 L20；BGE 不进入 smart feature。 |
| smart feature 有效状态统一 | 在 `config_loader.apply_effective_feature_enablement()` 中把页面开关和白名单命中结果收口到 `_smart_feats`。 | 页面开关只表示用户请求，不能直接等价于最终能力可用。 | 后续 adapter、状态展示、patch 安装都优先看 `_smart_feats`，避免未命中白名单仍产生命令。 |
| 内存卸载重复逻辑删除/收敛 | 将 offload 场景判断统一到 `resolve_feature_whitelist_row_from_params()` 和 `resolve_offload_whitelist_backend()`；将 offload 开关、backend、variant、effective size 统一到 `vllm_adapter.py` 的 resolver。 | 原来多个模块分别判断模型、卡型、env、backend，容易出现命令、状态、patch 安装不一致。 | native / LMCache / MemCache 后端互斥关系统一；命令渲染、`advanced_features.json`、日志诊断使用同一套 offload 结果。 |
| native offload 容量入口收口 | `vllm_adapter._resolve_native_backend_offload_gb()` 统一只读取 `KV_MEM_OFFLOAD_SIZE`，不再从白名单读取容量，也不再保留旧 `LMCACHE_MAX_LOCAL_CPU_SIZE` 兼容入口。 | 白名单应只声明是否允许 native backend，不应承载运行时容量；旧入口和 fallback 会造成隐式默认容量。 | H20/L20 native offload 未填、非法或 auto 熔断时直接丢弃，不再偷偷补 H20=200GB、L20=20GB 等默认值。 |
| vLLM 命令渲染复用现有逻辑 | `wings_control/engines/vllm_adapter.py` 继续复用已有 spec、IndexCache、native offload builder，只从白名单行读取 token、topk、backend 等静态字段。 | 避免新增 H20/L20 专用命令生成分支。 | GLM/Qwen 的 MTP、IndexCache、native offload 都通过现有命令路径落地，行为更容易维护和测试。 |
| LMCache / MemCache / native 边界明确 | `wings_control/core/wings_entry.py` 识别 native/MemCache 场景并跳过 LMCache patch；`wings_control/features/kv_offload/memcache/hybrid.py` 不再维护第二份 Qwen Day0 模型 token 表。 | native offload、LMCache patch、MemCache hybrid 是互斥路径，不能同时介入。 | H20/L20 native offload 只走 vLLM 原生 CLI；MemCache 只在白名单 backend 为 `memcache` 且 `_smart_feats` 有效时生效。 |

## 场景效果

| 场景 | 修改后的效果 |
| --- | --- |
| GLM-4.7 / GLM-5 / GLM-5.1 + H20 | 命中 H20 defaults；按白名单启用 MTP、native offload；GLM-5 / GLM-5.1 启用 IndexCache。 |
| GLM-4.7 / GLM-5 / GLM-5.1 + L20 | 不命中 H20-only defaults 和 smart feature，避免误触。 |
| Qwen3.6-27B + L20/H20 | 命中 defaults；启用 MTP token=1 和 native offload。 |
| Qwen3.6-35B-A3B + L20/H20 | 命中 defaults；启用 MTP token=3、`moe_backend=triton` 和 native offload。 |
| Qwen3-Embedding-0.6B + L20/H20 | 命中 defaults；只启用 native offload，不启用 spec/sparse。 |
| bge-large / bge-reranker + L20/H20 | 只走 defaults 基线，不进入 spec/sparse/offload 白名单。 |

## 重点文件

| 文件 | 作用 |
| --- | --- |
| `wings_control/config/defaults/nvidia_default.json` | 普通启动 defaults 和 `card_tokens` 门禁数据。 |
| `wings_control/core/config_loader.py` | defaults 卡型选择、smart feature 有效状态收口。 |
| `wings_control/config/smart_feature_whitelist.json` | spec / sparse / offload 能力授权。 |
| `wings_control/utils/model_utils.py` | smart whitelist 统一 matcher 和 offload backend 查询入口。 |
| `wings_control/engines/vllm_adapter.py` | spec、IndexCache、native offload 命令渲染；offload variant / effective size 统一计算。 |
| `wings_control/core/wings_entry.py` | LMCache patch 安装判断和 `advanced_features.json` 状态展示。 |
| `wings_control/features/kv_offload/memcache/hybrid.py` | MemCache backend 边界和 Qwen Day0 MemCache 判断收敛。 |

## 验证

| 验证项 | 结果 |
| --- | --- |
| `python -m pytest tests/test_default_config.py -q` | `31 passed` |
| `python -m pytest tests/test_smart_feature_whitelist.py -q` | `83 passed` |
| `python -m pytest tests/test_smart_feature_whitelist.py tests/test_speculative_strategy.py tests/test_kv_offload_gating.py -q` | `186 passed` |
| `python -m json.tool wings_control/config/defaults/nvidia_default.json` | 通过 |
| `ConvertFrom-Json` 解析 `nvidia_default.json` | 通过 |

## 相关提交

| Commit | 说明 |
| --- | --- |
| `3c46c0f` | 首次接入 NVIDIA H20/L20 Day0 场景。 |
| `f96315b` | 收敛 smart whitelist 命名和 L20 -> H20 单向继承，移除 `precision_tokens`。 |
| `78b0af4` | 移除 native offload 隐式容量默认值。 |
| `300432a` | 给 NVIDIA exact defaults 增加 `card_tokens` 门禁。 |
| `52bca7d` | smart 白名单 source 归一化。 |
