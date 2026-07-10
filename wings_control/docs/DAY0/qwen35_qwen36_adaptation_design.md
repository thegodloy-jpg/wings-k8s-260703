# Qwen3.5 / Qwen3.6 Day0 优化参数适配设计

> 状态：编码适配基准稿。
>
> 本文以 `wings_control/docs/DAY0/qwen优化参数适配基准.xlsx` 为直接编码基准。该基准表由 `wings_control/docs/DAY0/qwen模型收编.xlsx` 的优化列整理而来，只描述优化后启动参数。Baseline 只用于对照和验收状态，不进入 Wings 适配参数。

## 目标

将 11 个 Qwen3.5 / Qwen3.6 Day0 优化场景纳入 Wings 启动链路，并让代码实现能够直接从场景基准表推导以下内容：

- `spec` 白名单与 `qwen3_5_mtp` 投机参数；
- MemCache `offload` 白名单、profile、配置文件和启动脚本；
- Function Call parser：统一 `qwen3_coder`；
- Reasoning / Thinking parser：按现有 `reason_parser.yaml` 的 `qwen3` 映射；
- EP、量化、`native_port_hint`、TP/DP、运行参数和模型路径；
- dry-run / unit test 需要校验的最终 vLLM 命令片段。

核心原则不变：页面开关、接口参数、环境变量或启动参数只能表达“请求启用某特性”，最终是否生效仍由 `smart_feature_whitelist.json` 按 `engine + model token + card token` 收敛。也就是说，页面下发的是请求态；代码需要根据白名单、场景 profile 和当前硬件信息计算有效态，再决定是否生成最终 vLLM 参数。

## 输入标准

### 适配基准文件

当前编码基准文件：

- `wings_control/docs/DAY0/qwen优化参数适配基准.xlsx`

该文件包含：

- `优化参数适配基准`：11 个优化场景的适配参数，并包含 `name_tokens`、`card_tokens`、芯片信息、`architecture`、`native_port_hint` 等编码识别字段；
- `适配规则`：白名单、offload、parser、sparse 等规则；
- `范围汇总`：场景数量和范围统计。

编码时只消费优化参数表。原始 `qwen模型收编.xlsx` 中的 `910C 基线` / `910B 基线` 只能作为验收对照，不能作为默认参数来源。

`qwen优化参数适配基准.xlsx` 是唯一事实源。本文的 Markdown 表只作为可读摘要和编码解释，字段、场景、参数或状态发生调整时，必须先更新 Excel，再同步本文摘要；实现和测试不应反向从 Markdown 表生成标准数据。

### 基准解释规则

- `MTP方法` 固定为 `qwen3_5_mtp`。
- `num_speculative_tokens` 按表逐场景使用。
- `Offload=是` 才能生成 MemCache 相关片段。
- `Offload=否` 的场景必须禁止 `MMC_LOCAL_CONFIG_PATH`、`AscendStoreConnector`、`--kv-transfer-config`。
- Function Call 全部为 `是`，`tool_call_parser` 固定为 `qwen3_coder`。
- Sparse 全部为 `否`，不进入 `sparse` 白名单。
- Qwen3.6 910C 场景的完整优化参数来自原始 Excel 的 `J6 / qwen36_serve.sh`，不是 wrapper 单元格本身。
- Qwen3.6 910B 场景的完整优化参数来自各自 `910B 优化` 单元格。

## 优化参数适配基准表

下表是编码实现的直接基准。字段名尽量与 `qwen优化参数适配基准.xlsx` 保持一致。

| 场景 | Excel来源 | Excel状态 | 模型名/Token | 引擎 | 卡型 | 精度 | TP | DP | Port | MTP方法 | tokens | MTP形态 | Offload | MemCache端口 | EP | 量化 | max_num_seqs | max_model_len | max_num_batched_tokens | gpu_memory_utilization | seed | served_model_name | Function Call | tool_call_parser | Sparse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-27B-910C | row2 / 910C 优化 | done | Qwen/Qwen3.5-27B | vllm_ascend | 910C | BF16 | 2 | 1 | 7878 | qwen3_5_mtp | 1 | MTP-only + MTP+MemCache | 是 | 50051/50061 | 否 | 否 | 8 | 131072 | 16384 | 0.75 | 1024 | qwen35 | 是 | qwen3_coder | 否 |
| Qwen3.5-27B-910B | row2 / 910B 优化 | done | Qwen/Qwen3.5-27B | vllm_ascend | 910B | BF16 | VISIBLE_COUNT(default 4) | 1 | 7898 | qwen3_5_mtp | 3 | MTP-only | 否 |  | 否 | 否 | 8 | 8192 | 8192 | 0.75 | 1024 | qwen35 | 是 | qwen3_coder | 否 |
| Qwen3.5-35B-A3B-910C | row3 / 910C 优化 | done | Qwen/Qwen3.5-35B-A3B | vllm_ascend | 910C | BF16 | 2 | 1 | 7897 | qwen3_5_mtp | 1 | MTP-only | 否 |  | 是 | 否 | 32 | 161072 | 8192 | 0.9 | 1024 | qwen35 | 是 | qwen3_coder | 否 |
| Qwen3.5-122B-A10B-910C | row4 / 910C 优化 | done | Qwen/Qwen3.5-122B-A10B | vllm_ascend | 910C | BF16 | 8 | 1 | 6901 | qwen3_5_mtp | 1 | MTP-only | 否 |  | 是 | 否 | 8 | 131072 | 16384 | 0.9 | 1024 | qwen35 | 是 | qwen3_coder | 否 |
| Qwen3.5-397B-A17B-910C | row5 / 910C 优化 | done | Qwen/Qwen3.5-397B-A17B | vllm_ascend | 910C | BF16 | 16 | 1 | 6901 | qwen3_5_mtp | 1 | MTP-only | 否 |  | 是 | 否 | 4 | 16384 | 16384 | 0.92 | 1024 | qwen35 | 是 | qwen3_coder | 否 |
| Qwen3.6-27B-910C | row6 / 910C 优化 + J6 | done | Qwen/Qwen3.6-27B | vllm_ascend | 910C | BF16 | 2 | 1 | 7799 | qwen3_5_mtp | 3 | MTP+MemCache | 是 | 50071/50081 | 否 | 否 | 32 | 131072 | 8192 | 0.9 | 1024 | qwen36 | 是 | qwen3_coder | 否 |
| Qwen3.6-27B-w8a8-910C | row7 / 910C 优化 + J6 | done | Eco-Tech/Qwen3.6-27B-w8a8 | vllm_ascend | 910C | INT8 / W8A8 | 2 | 1 | 7799 | qwen3_5_mtp | 3 | MTP+MemCache | 是 | 50071/50081 | 否 | quantization=ascend | 32 | 131072 | 8192 | 0.9 | 1024 | qwen36 | 是 | qwen3_coder | 否 |
| Qwen3.6-27B-w8a8-910B | row7 / 910B 优化 | done | Eco-Tech/Qwen3.6-27B-w8a8 | vllm_ascend | 910B | INT8 / W8A8 | 4 | 1 | 7899 | qwen3_5_mtp | 3 | MTP+MemCache | 是 | 50051/50061 | 否 | quantization=ascend | 32 | 131702 | 8192 | 0.9 | 1024 | qwen36 | 是 | qwen3_coder | 否 |
| Qwen3.6-35B-A3B-910C | row8 / 910C 优化 + J6 | done | Qwen/Qwen3.6-35B-A3B | vllm_ascend | 910C | BF16 / MoE | 2 | 1 | 7799 | qwen3_5_mtp | 3 | MTP+MemCache | 是 | 50071/50081 | 是 | 否 | 32 | 131072 | 8192 | 0.9 | 1024 | qwen36 | 是 | qwen3_coder | 否 |
| Qwen3.6-35B-A3B-w8a8-910C | row9 / 910C 优化 + J6 | 基线 done | Eco-Tech/Qwen3.6-35B-A3B-w8a8 | vllm_ascend | 910C | INT8 / W8A8 | 2 | 1 | 7799 | qwen3_5_mtp | 3 | MTP+MemCache | 是 | 50071/50081 | 是 | quantization=ascend | 32 | 131072 | 8192 | 0.9 | 1024 | qwen36 | 是 | qwen3_coder | 否 |
| Qwen3.6-35B-A3B-w8a8-910B | row9 / 910B 优化 | doing | Eco-Tech/Qwen3.6-35B-A3B-w8a8 | vllm_ascend | 910B | INT8 / W8A8 | 4 | 1 | 7899 | qwen3_5_mtp | 3 | MTP+MemCache | 是 | 50051/50061 | 是 | quantization=ascend | 32 | 131702 | 8192 | 0.9 | 1024 | qwen36 | 是 | qwen3_coder | 否 |

完整 `model_path` 以 `qwen优化参数适配基准.xlsx` 的 `model_path` 列为准。Markdown 表为了可读性不重复展开所有长路径。

## 场景识别和芯片信息

当前项目的 `smart_feature_whitelist.json` 使用小写子串匹配：`engine` 精确匹配，`name_tokens` 匹配 `model_name + model_path`，`card_tokens` 匹配标准化后的芯片 token。Ascend 910B / 910B3 统一收敛为 `910b`，Ascend 910C / A3 统一收敛为 `910c`。

| 场景 | name_tokens | card_tokens | 芯片信息 | architecture |
| --- | --- | --- | --- | --- |
| Qwen3.5-27B-910C | `qwen/qwen3.5-27b`, `qwen3.5-27b` | `910c` | Ascend 910C / A3 | `Qwen3_5ForConditionalGeneration` |
| Qwen3.5-27B-910B | `qwen/qwen3.5-27b`, `qwen3.5-27b` | `910b` | Ascend 910B / 910B3 / A2 | `Qwen3_5ForConditionalGeneration` |
| Qwen3.5-35B-A3B-910C | `qwen/qwen3.5-35b-a3b`, `qwen3.5-35b-a3b` | `910c` | Ascend 910C / A3 | `Qwen3_5MoeForConditionalGeneration` |
| Qwen3.5-122B-A10B-910C | `qwen/qwen3.5-122b-a10b`, `qwen3.5-122b-a10b` | `910c` | Ascend 910C / A3 | `Qwen3_5MoeForConditionalGeneration` |
| Qwen3.5-397B-A17B-910C | `qwen/qwen3.5-397b-a17b`, `qwen3.5-397b-a17b` | `910c` | Ascend 910C / A3 | `Qwen3_5MoeForConditionalGeneration` |
| Qwen3.6-27B-910C | `qwen/qwen3.6-27b`, `qwen3.6-27b` | `910c` | Ascend 910C / A3 | `Qwen3_5ForConditionalGeneration` |
| Qwen3.6-27B-w8a8-910C | `eco-tech/qwen3.6-27b-w8a8`, `qwen3.6-27b-w8a8` | `910c` | Ascend 910C / A3 | `Qwen3_5ForConditionalGeneration` |
| Qwen3.6-27B-w8a8-910B | `eco-tech/qwen3.6-27b-w8a8`, `qwen3.6-27b-w8a8` | `910b` | Ascend 910B / 910B3 / A2 | `Qwen3_5ForConditionalGeneration` |
| Qwen3.6-35B-A3B-910C | `qwen/qwen3.6-35b-a3b`, `qwen3.6-35b-a3b` | `910c` | Ascend 910C / A3 | `Qwen3_5MoeForConditionalGeneration` |
| Qwen3.6-35B-A3B-w8a8-910C | `eco-tech/qwen3.6-35b-a3b-w8a8`, `qwen3.6-35b-a3b-w8a8` | `910c` | Ascend 910C / A3 | `Qwen3_5MoeForConditionalGeneration` |
| Qwen3.6-35B-A3B-w8a8-910B | `eco-tech/qwen3.6-35b-a3b-w8a8`, `qwen3.6-35b-a3b-w8a8` | `910b` | Ascend 910B / 910B3 / A2 | `Qwen3_5MoeForConditionalGeneration` |

芯片 token 不从模型名推断，必须来自当前项目已有硬件识别链路。Day0 适配基准中的 `hardware_info.json` 只要求保留 `device` 和 `hardware_family` 两个字段，例如 910B 场景使用 `{"device": "ascend", "hardware_family": "Ascend910B_64G"}`；`details` 不是基准输入的一部分。运行时代码可以继续兼容旧的 `details[].name`，必要时也可以使用 `ENGINE_VERSION` 的 `-a2` / `-a3` 后缀兜底，但识别失败时不能默认命中白名单。

## Qwen3.6 910C wrapper 展开规则

原始 Excel 中 Qwen3.6 910C 优化参数来自同一个 `qwen36_serve.sh`。编码时必须展开为四个独立 scenario profile，不允许在运行时再解析 wrapper 文本。

| 场景 | wrapper 参数 | QUANT | EP | 量化 | MemCache端口 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B-910C | `27B` | `0` | `0` | 否 | `50071/50081` |
| Qwen3.6-27B-w8a8-910C | `27B-w8a8` | `1` | `0` | `quantization=ascend` | `50071/50081` |
| Qwen3.6-35B-A3B-910C | `35B-A3B` | `0` | `1` | 否 | `50071/50081` |
| Qwen3.6-35B-A3B-w8a8-910C | `35B-A3B-w8a8` | `1` | `1` | `quantization=ascend` | `50071/50081` |

上述四个场景共享 `TP=2`、`DP=1`、`native_port_hint=7799`、`max_num_seqs=32`、`max_model_len=131072`、`max_num_batched_tokens=8192`、`gpu_memory_utilization=0.90`、`seed=1024`、`served_model_name=qwen36`、`method=qwen3_5_mtp`、`num_speculative_tokens=3`。

## 编码数据模型

建议新增一个小型 Qwen Day0 scenario/profile resolver，数据结构可按下列字段组织。字段名不要求完全一致，但语义必须覆盖。

```python
QwenDay0Scenario = {
    "scenario": "Qwen3.6-27B-910C",
    "engine": "vllm_ascend",
    "name_tokens": ["qwen/qwen3.6-27b", "qwen3.6-27b"],
    "card_tokens": ["910c"],
    "chip": {
        "family": "ascend",
        "standard_card": "910c",
        "aliases": ["910c", "a3"],
    },
    "architecture": "Qwen3_5ForConditionalGeneration",
    "model_path_hint": "/var/aispace/model/ai-storage/ai-prod/platform/Qwen3.6-27B",
    "served_model_name": "qwen36",
    "native_port_hint": 7799,
    "tp": 2,
    "dp": 1,
    "spec": {
        "method": "qwen3_5_mtp",
        "num_speculative_tokens": 3,
        "enforce_eager": True,
    },
    "offload": {
        "enabled": True,
        "backend": "memcache",
        "connector": "AscendStoreConnector",
        "role": "kv_both",
        "meta_service_port": 50071,
        "config_store_port": 50081,
        "protocol": "device_rdma",
        "world_size": 256,
    },
    "defaults": {
        "max_num_seqs": 32,
        "max_model_len": 131072,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.90,
        "seed": 1024,
        "enable_expert_parallel": False,
        "quantization": None,
        "tool_call_parser": "qwen3_coder",
    },
}
```

所有 scenario profile 必须显式携带 `name_tokens`、`card_tokens`、`chip` 和 `architecture`，不能依赖调用方临时从场景名拆分。非 offload 场景的 `offload.enabled` 必须为 `False`，并且不能携带 connector、端口或 `kv_transfer_config`。

## 白名单规则

### spec

`spec` 覆盖基准表全部 11 个场景。

`smart_feature_whitelist.json` 中应按 `engine + name_tokens + card_tokens` 增加或补齐；具体 `name_tokens`、`card_tokens` 和 `architecture` 使用“场景识别和芯片信息”表，不再从显示名称临时推断：

- Qwen3.5:
  - `Qwen3.5-27B` on `910C`
  - `Qwen3.5-27B` on `910B`
  - `Qwen3.5-35B-A3B` on `910C`
  - `Qwen3.5-122B-A10B` on `910C`
  - `Qwen3.5-397B-A17B` on `910C`
- Qwen3.6:
  - `Qwen3.6-27B` on `910C`
  - `Qwen3.6-27B-w8a8` on `910C`
  - `Qwen3.6-27B-w8a8` on `910B`
  - `Qwen3.6-35B-A3B` on `910C`
  - `Qwen3.6-35B-A3B-w8a8` on `910C`
  - `Qwen3.6-35B-A3B-w8a8` on `910B`

### offload

`offload` 只覆盖：

- `Qwen3.5-27B-910C`
- Qwen3.6 全部 6 个场景

明确不覆盖：

- `Qwen3.5-27B-910B`
- `Qwen3.5-35B-A3B-910C`
- `Qwen3.5-122B-A10B-910C`
- `Qwen3.5-397B-A17B-910C`

这些非 offload 场景即使请求了 `ENABLE_KV_OFFLOAD=true`，有效状态也必须被白名单压掉，最终命令不能出现 MemCache 或 LMCache 片段。

### sparse

本批 Qwen3.5 / Qwen3.6 Ascend 场景全部不进入 `sparse` 白名单。

## 默认参数落点

### 设备默认值

设备级默认值仍由：

- `wings_control/config/defaults/ascend_default.json`
- `wings_control/config/defaults/nvidia_default.json`

管理。本批场景都是 `vllm_ascend`，应优先落在 `ascend_default.json`。

### per-scenario 参数

以下字段必须按基准表逐场景管理，不要提升为 architecture default，除非能证明该 architecture 下所有模型都共享同一值。

- `tp`
- `dp`
- `native_port_hint`
- `max_num_seqs`
- `max_model_len`
- `max_num_batched_tokens`
- `gpu_memory_utilization`
- `seed`
- `num_speculative_tokens`
- `enable_expert_parallel`
- `quantization`
- `served_model_name`
- MemCache port pair

### 端口落点规则

基准表中的 `Port` 来自 Day0 原生脚本，是 `native_port_hint`，用于 dry-run 对齐和无显式端口时的默认值，不代表必须覆盖 Wings/K8s 已有服务端口。

最终端口优先级：

1. 页面、接口、环境变量或部署配置显式下发的端口；
2. 当前项目已有的服务端口分配逻辑；
3. scenario profile 的 `native_port_hint`。

编码时不要把 `7878`、`7898`、`7899`、`7799`、`6901`、`7897` 直接写成不可覆盖常量。验收时 dry-run 可以校验 `native_port_hint`，但运行态必须允许上层部署端口覆盖。

### Function Call parser 落点规则

Function Call parser 统一为 `qwen3_coder`。为避免 exact-model defaults 直接返回时丢失 architecture default，第一轮实现必须满足以下二选一规则：

1. 每个 Qwen3.5 / Qwen3.6 exact-model scenario profile 或 exact-model defaults 都显式包含 `tool_call_parser=qwen3_coder`；
2. 或者修改 defaults merge 逻辑，保证 exact-model 配置会继承 architecture default 中的 `tool_call_parser`。

无论采用哪一种实现，单元测试都必须覆盖 exact-model 命中路径：Function Call 开关启用时，最终 engine config 同时保留 `enable_auto_tool_choice` 和 `tool_call_parser=qwen3_coder`。

## MemCache profile

MemCache 应作为独立 offload backend，而不是 LMCache 的 env 变体。

### 支持范围

Qwen MemCache profile 只覆盖：

- `Qwen3.5-27B-910C`
- `Qwen3.6-27B-910C`
- `Qwen3.6-27B-w8a8-910C`
- `Qwen3.6-27B-w8a8-910B`
- `Qwen3.6-35B-A3B-910C`
- `Qwen3.6-35B-A3B-w8a8-910C`
- `Qwen3.6-35B-A3B-w8a8-910B`

其他 Qwen3.5 场景必须返回空 profile。

### profile 字段

每个支持卸载的 profile 至少需要：

- `backend=memcache`
- `connector=AscendStoreConnector`
- `role=kv_both`
- `meta_service_url=tcp://127.0.0.1:<meta_port>`
- `config_store_url=tcp://127.0.0.1:<config_store_port>`
- `protocol=device_rdma`
- `world_size=256`
- `log_level=error`
- `MMC_LOCAL_CONFIG_PATH`
- `MMC_META_CONFIG_PATH`

端口按基准表：

- `Qwen3.5-27B-910C`：`50051/50061`
- Qwen3.6 910C：`50071/50081`
- Qwen3.6 910B：`50051/50061`

### 配置文件和服务生命周期

Qwen MemCache 应复用当前项目 `wings_control/features/kv_offload/memcache/` 下的 fragment 生成边界，而不是在 `vllm_adapter.py` 中拼接裸配置文本。

实现要求：

- engine 侧 prelude 渲染 `mmc_local.conf`，导出 `MMC_LOCAL_CONFIG_PATH`，并把 `start_memcache_master.sh` 写到 `WINGS_MEMCACHE_DIR`；
- master / sidecar 侧脚本渲染 `mmc_meta.conf`，导出 `MMC_META_CONFIG_PATH`，并执行 `python -c "from memcache_hybrid import MetaService; MetaService.main()"`；
- `WINGS_MEMCACHE_DIR` 默认沿用当前项目约定 `/shared-volume/memcache`，但允许部署层覆盖；
- `meta_service_url` 和 `config_store_url` 默认使用 `127.0.0.1` 仅适用于 engine 与 MetaService 共享网络命名空间；如果 MetaService 独立成 pod / service，必须由部署层提供服务 DNS 或 IP；
- engine 重试或 fallback 到非 offload 路径时，必须清理 `MMC_LOCAL_CONFIG_PATH`，并移除 `AscendStoreConnector` / `kv_transfer_config`；
- Qwen MemCache 不走 LMCache install/env/YAML 路径，也不复用 `CPUOffloadingConnector`。

MemCache 容量仍由页面或环境变量下发的 offload memory 决定。没有有效 offload memory 时，应按有效态关闭 MemCache，而不是保留脚本里的固定容量默认值。

### 生成命令要求

支持 offload 的最终 engine 脚本必须包含：

- `MMC_LOCAL_CONFIG_PATH`
- `--no-disable-hybrid-kv-cache-manager`
- `--kv-transfer-config`
- `AscendStoreConnector`
- 生成或引用 `start_memcache_master.sh`

非 offload 场景必须不包含上述任一 MemCache 片段。

## 投机解码规则

MTP 统一使用：

- `method=qwen3_5_mtp`
- `enforce_eager=true`

tokens：

- Qwen3.5 910C：`1`
- `Qwen3.5-27B-910B`：`3`
- Qwen3.6 全部：`3`

实现约束：`smart_feature_whitelist.json` 的 `spec` 场景行需要携带 `mtp_num_speculative_tokens`；`vllm_adapter.py` 在无真实 draft model 时从命中的白名单行读取该值，再回退到历史通用 MTP token 策略。不要在 adapter 中新增独立 Qwen 场景硬编码表。

如果用户提供真实 draft model 路径，仍应先走现有 draft/eagle 路径；本表只定义“无 draft model 时的 Qwen Day0 优化默认策略”。

## Function Call 和 Reasoning

### Function Call

本批全部支持 Function Call，parser 固定为：

- `qwen3_coder`

原始 Excel 的部分 Qwen3.6 脚本片段中出现 `hermes`，但这是脚本证据与产品口径的差异。Wings 适配必须按 `qwen3_coder` 收敛，dry-run 需要验证最终命令没有回落到 `hermes`。

Function Call 参数只在页面开关、接口参数、环境变量或模型默认策略要求启用时输出，不能因为场景支持就无条件注入。支持能力表示“可以启用”，不是“启动命令必然携带 `--enable-auto-tool-choice`”。

### Reasoning / Thinking

Reasoning parser 仍由：

- `ENABLE_AUTO_THINK_CHOICE`
- `--enable-auto-think-choice`
- `wings_control/docs/features/reasoning_parser/reason_parser.yaml`

控制。本批场景可继续映射到 `qwen3`，但只有 Thinking 启用时才注入。

## 适配流程

1. 加载启动参数和环境变量。
2. 加载设备默认值。
3. 识别模型 architecture、model token，并从硬件信息标准化出 `card_tokens`。
4. 按“场景识别和芯片信息”表解析 Qwen Day0 scenario profile。
5. 计算有效 smart feature：
   - 请求 `spec` + 命中 `spec` 白名单 -> 有效 `spec`；
   - 请求 `offload` + 命中 `offload` 白名单 -> 有效 `offload`；
   - 未命中时按当前规则抑制或降级。
6. 应用基准表中的 per-scenario 参数。
7. 按端口优先级决定最终端口，`native_port_hint` 只作为兜底。
8. 应用 Function Call 和 Reasoning gating，确保 exact-model 路径不会丢失 `tool_call_parser=qwen3_coder`。
9. 如果有效 offload profile 是 MemCache：
   - 渲染 `mmc_local.conf`；
   - 渲染 `mmc_meta.conf`；
   - 生成或引用 `start_memcache_master.sh`；
   - 注入 `AscendStoreConnector` 和 `kv_transfer_config`；
   - 跳过 LMCache 专用 install/env 路径。
10. 构建最终 vLLM 命令。
11. 按有效状态写入 startup status / advanced features，而不是直接使用原始 env。

## 验收清单

### 静态配置

- `smart_feature_whitelist.json` 中 `spec` 覆盖 11 个场景。
- `spec` 白名单 11 个场景行携带 `mtp_num_speculative_tokens`，数值与基准表一致。
- `smart_feature_whitelist.json` 中 `offload` 只覆盖 Qwen3.6 系列和 `Qwen3.5-27B-910C`。
- `smart_feature_whitelist.json` 中不新增本批 Qwen `sparse` 行。
- 白名单条目的 `name_tokens`、`card_tokens`、`architecture` 与“场景识别和芯片信息”表一致。
- `ascend_default.json` 或 scenario profile 能提供表中 per-scenario 参数。
- Function Call parser 最终为 `qwen3_coder`。
- Reasoning parser 关系仍由 `reason_parser.yaml` 管理。
- `qwen优化参数适配基准.xlsx` 与本文 Markdown 摘要的 11 个场景、MTP、offload、EP、量化、MemCache 端口保持一致。

### 单元测试

- `spec` 请求只有命中白名单才有效。
- `offload` 请求只有命中白名单才有效。
- Qwen3.5 非 offload 场景请求卸载后，最终命令没有 MemCache / LMCache 片段。
- Qwen3.6 和 `Qwen3.5-27B-910C` 请求卸载后，能解析出 MemCache profile。
- MTP tokens 与基准表一致。
- `910b` / `910c` 硬件 token 标准化后才能命中对应白名单；识别失败时不命中。
- EP / quantization 与基准表一致。
- Function Call 启用时输出 `qwen3_coder`。
- exact-model 命中路径下，Function Call 启用时同时保留 `enable_auto_tool_choice` 和 `tool_call_parser=qwen3_coder`。
- Thinking 启用时输出 `reasoning_parser=qwen3`。
- 显式端口存在时不会被 `native_port_hint` 覆盖；无显式端口时才使用基准表端口兜底。

### Dry Run

每个场景生成最终启动脚本，对比：

- model path
- served model name
- effective service port / `native_port_hint`
- TP / DP
- max 参数
- `gpu_memory_utilization`
- seed
- MTP method 和 tokens
- EP
- quantization
- Function Call parser
- Reasoning parser
- MemCache 文件、端口、connector 和 `kv_transfer_config`
- Markdown 摘要与 `qwen优化参数适配基准.xlsx` 的关键列一致

### Runtime Smoke

对 MemCache 场景：

- 启动 `start_memcache_master.sh`；
- 确认 `MetaService.main()` 可以成功 import；
- 确认 `MMC_META_CONFIG_PATH` 和 `MMC_LOCAL_CONFIG_PATH` 指向渲染出的文件；
- 确认 `mmc_local.conf` / `mmc_meta.conf` 中端口、`world_size=256`、`protocol=device_rdma`、`log_level=error` 与 profile 一致；
- 启动 engine；
- 发送一次请求；
- 确认没有 connector 初始化错误。

## 风险

- `qwen36_serve.sh` 把 910C 四个场景收敛到同一个 wrapper；如果不提前展开为四个独立 scenario profile，EP / quantization 容易串场。
- 如果 exact-model defaults 直接返回且缺少 `tool_call_parser`，可能导致 `qwen3_coder` 不被继承；实现必须通过显式 profile 字段或 defaults merge 修复。
- 如果 offload variant 没有显式区分，MemCache 和 LMCache 路径可能互相污染。
- `910b` / `910c` card token 解析是白名单命中的关键；解析失败会导致 `spec` 或 `offload` 完全失效。
- 如果只更新 Markdown 而没有同步 `qwen优化参数适配基准.xlsx`，编码基准会漂移；实现和测试以 Excel 为准。

## 实施边界

第一轮实现限制在：

- 增加 Qwen Day0 scenario/profile 数据；
- 补齐模型 token、芯片 token 和 architecture 识别信息；
- 更新 `spec` / `offload` 白名单；
- 补齐 Qwen Ascend 默认参数；
- 泛化 MemCache profile；
- 按端口优先级使用 `native_port_hint`；
- 增加单元测试和 dry-run 验证；
- 对齐 `qwen优化参数适配基准.xlsx`。

第一轮不重构无关的 PD、LMCache、sparse、Reasoning parser 或非 Qwen 模型路径。
