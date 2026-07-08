# Qwen3.5 / Qwen3.6 Day0 适配设计

> 状态：临时设计稿。
>
> 本文基于当前 checkout 以及 `D:\Download\Qwen*.sh` 原生脚本整理。明天提供准确特性矩阵后，需要刷新本文中的场景矩阵、白名单候选和待确认项。当前内容只作为适配框架，不作为最终产品事实。

## 目标

将 11 个 Qwen3.5 / Qwen3.6 Day0 模型场景纳入 Wings 启动链路，并明确以下能力的归属和适配方式：

- 投机解码；
- MemCache KV 卸载；
- Function Call parser；
- Reasoning / Thinking parser；
- 模型默认参数；
- 白名单控制下的有效启用状态；
- 最终生成的 vLLM 启动命令和辅助脚本。

设计上必须保留现有规则：用户页面、环境变量或启动参数只能表达“请求启用某特性”，最终是否生效仍由 `smart_feature_whitelist.json` 按 engine、model、card token 收敛。

## 输入依据

### 原生脚本来源

当前原生脚本来源为 `D:\Download\Qwen*.sh`。

本稿采用的临时判断规则如下：

- 如果原生脚本包含 `memcache`、`MEMCACHE=1`、`MMC_LOCAL_CONFIG_PATH`、`MMC_META_CONFIG_PATH`、`MetaService` 或 `AscendStoreConnector`，则暂按该场景原生支持 MemCache 卸载处理。
- 如果原生脚本只是通过 `MEMCACHE=1 ./qwen36_serve.sh` 间接启动，则暂按支持处理，但 Wings 侧实现时需要展开成显式的配置片段和启动片段。
- 如果没有任何 MemCache 信号，则在最终矩阵确认前，不把该场景归类为支持卸载。

### 当前仓库职责边界

- `wings_control/config/smart_feature_whitelist.json` 负责 `spec`、`sparse`、`offload` 等 smart feature 的准入。
- `wings_control/core/config_loader.py` 负责有效特性收敛、默认参数合并、Function Call gating、Reasoning parser gating 以及 `kv_transfer_config` 注入。
- `wings_control/engines/vllm_adapter.py` 负责最终 vLLM 命令渲染、投机配置渲染和 cache 相关环境片段。
- `wings_control/features/kv_offload/memcache/` 负责可复用的 MemCache shell 片段，但当前判断逻辑仍偏 Kimi K2.7 Code 专用。
- `wings_control/docs/features/reasoning_parser/reason_parser.yaml` 负责按 architecture/model 描述 reasoning parser 支持关系。
- `wings_control/config/defaults/ascend_default.json` 和 `wings_control/config/defaults/nvidia_default.json` 负责设备级默认值。

## 临时场景矩阵

下表列出当前 `wings_control/docs/DAY0` 中涉及的 11 个 Day0 场景。MemCache 卸载列按上面的原生脚本规则暂定。

| 场景 | 引擎 | 卡型 | 精度 | 投机 | MemCache 卸载 | 稀疏 | EP | Function Call | Reasoning | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-27B-910B | vllm_ascend | 910B | BF16 | qwen3_5_mtp | 是 | 否 | 否 | qwen3_coder 候选 | qwen3 候选 | 原生脚本包含 MetaService 启动片段和 mmc 配置说明。 |
| Qwen3.5-27B-910C | vllm_ascend | 910C | BF16 | qwen3_5_mtp | 是 | 否 | 否 | qwen3_coder 候选 | qwen3 候选 | 原生脚本包含 AscendStoreConnector、kv-transfer-config 和 MetaService 片段。 |
| Qwen3.5-122B-A10B-910C | vllm_ascend | 910C | BF16 | qwen3_5_mtp | 待确认 | 否 | 是 | qwen3_coder 候选 | qwen3 候选 | DAY0 文档中存在该场景，但当前未找到匹配的 `D:\Download` 原生脚本。 |
| Qwen3.5-35B-A3B-910C | vllm_ascend | 910C | BF16 | qwen3_5_mtp | 待确认 | 否 | 是 | qwen3_coder 候选 | qwen3 候选 | 该场景出现在 Qwen3.5-27B-910C 的原生/粘贴内容中，目前只看到 MTP 线索。 |
| Qwen3.5-397B-A17B-910C | vllm_ascend | 910C | BF16 | qwen3_5_mtp | 未见信号 | 待确认 | 是 | qwen3_coder 候选 | qwen3 候选 | 原生脚本未见 MemCache 信号；当前产品矩阵主要覆盖该家族的 NVIDIA 场景。 |
| Qwen3.6-27B-910C | vllm_ascend | 910C | BF16 | qwen3_5_mtp | 是 | 否 | 否 | qwen3_coder 或原生 parser 待确认 | qwen3 候选 | 原生脚本使用 `MEMCACHE=1 ./qwen36_serve.sh 27B`。 |
| Qwen3.6-27B-w8a8-910B | vllm_ascend | 910B | INT8 | qwen3_5_mtp | 是 | 否 | 否 | qwen3_coder 或 hermes 待确认 | qwen3 候选 | 原生脚本包含显式 AscendStoreConnector。 |
| Qwen3.6-27B-w8a8-910C | vllm_ascend | 910C | INT8 | qwen3_5_mtp | 是 | 否 | 否 | qwen3_coder 或原生 parser 待确认 | qwen3 候选 | 原生脚本使用 `MEMCACHE=1 ./qwen36_serve.sh 27B-w8a8`。 |
| Qwen3.6-35B-A3B-910C | vllm_ascend | 910C | BF16 或 Ascend recipe | qwen3_5_mtp | 是 | 否 | 是 | qwen3_coder 或原生 parser 待确认 | qwen3 候选 | 原生脚本使用 `MEMCACHE=1 ./qwen36_serve.sh 35B-A3B`。 |
| Qwen3.6-35B-A3B-w8a8-910B | vllm_ascend | 910B | INT8 | qwen3_5_mtp | 是 | 否 | 是 | qwen3_coder 或 hermes 待确认 | qwen3 候选 | 原生脚本包含显式 AscendStoreConnector。 |
| Qwen3.6-35B-A3B-w8a8-910C | vllm_ascend | 910C | INT8 | qwen3_5_mtp | 是 | 否 | 是 | qwen3_coder 或原生 parser 待确认 | qwen3 候选 | 原生脚本使用 `MEMCACHE=1 ./qwen36_serve.sh 35B-w8a8`。 |

## 设计原则

1. 区分“特性支持”和“特性请求”。
   页面、环境变量或启动参数可以请求 `spec`、`offload` 或 parser 能力，但最终有效状态必须继续由白名单收敛。

2. 将 MemCache 作为一类独立卸载变体处理。
   MemCache 不能实现成“换了一组环境变量的 LMCache”。它有独立 connector、独立服务进程和独立配置文件。

3. 让 MemCache 能在已确认的 Qwen 场景中复用。
   当前实现偏 Kimi 专用。Qwen 适配应复用渲染模板，但引入 Qwen 自己的 eligibility/profile 规则。

4. 不从 architecture 单独推断最终支持关系。
   本稿的临时依据是原生脚本证据。明天拿到准确特性矩阵后，以最终矩阵替换该临时规则。

5. 保留安全降级行为。
   如果请求了 `spec` 但未命中白名单，当前行为可以回退到 suffix 等策略；如果请求了 `offload` 但未命中白名单，则不能输出 MemCache 或 LMCache 相关片段。

## 建议架构

### 1. 场景注册表

建议引入一个小型内部 Qwen Day0 场景解析器。它不替代全局模型识别逻辑，只回答以下问题：

- 当前 model/card 是否属于已知 Qwen Day0 场景；
- 原生矩阵是否表明 MemCache 支持；
- 是否需要 EP；
- 是否需要量化；
- 默认 parser 或原生参数是否与现有默认值不同。

该注册表优先用数据描述，例如：

- model token 列表；
- engine；
- card token；
- architecture；
- 支持特性：`spec`、`offload`；
- offload backend：`memcache`；
- recipe hints：TP 默认值、EP 要求、量化、prefix cache 策略。

注册表应被 config/default/feature 逻辑消费，而不是让最终命令渲染器直接硬编码模型分支。

### 2. Smart Feature 白名单

只为最终矩阵确认的特性新增或更新白名单行。

基于当前临时设计：

- 已存在于 `spec` 的 Qwen3.6 行应保留。
- 原生脚本有 MemCache 证据的 Qwen3.5 / Qwen3.6 行，可以作为 `offload` 候选。
- 当前产品矩阵未确认的 Qwen3.5 行，应保持临时状态，等最终支持信息确认。
- 除非最终矩阵明确支持，否则不要把 Qwen 场景加入 `sparse`。

白名单匹配规则继续保持 `engine + model token + card token`。

### 3. MemCache 泛化

将现有 MemCache 支持从 Kimi 专用逻辑重构为“backend 通用 + model profile 判断”。

当前概念：

- `is_kimi_k27_code_memcache_params(...)`
- Kimi 专用路径决定是否渲染 MemCache。

目标概念：

- `resolve_memcache_profile(params, engine)` 返回空或 profile 对象。
- profile 描述：
  - backend name：`memcache`；
  - connector：`AscendStoreConnector`；
  - role：`kv_both`；
  - 默认 config directory；
  - 默认 meta service URL；
  - 默认 config store URL；
  - protocol；
  - world size；
  - DRAM size 来源；
  - profile 属于 Kimi 还是 Qwen。

渲染逻辑保持共享：

- 渲染 `mmc_local.conf`；
- 渲染 `mmc_meta.conf`；
- 渲染 `start_memcache_master.sh`；
- 导出 `MMC_LOCAL_CONFIG_PATH`；
- 生成 `kv_transfer_config`；
- 注入 `--no-disable-hybrid-kv-cache-manager`；
- MemCache profile 下跳过 LMCache patch/env exports。

### 4. MemCache 服务边界

MemCache 需要明确服务边界。最终启动脚本应包含或暴露：

- 写入 `mmc_local.conf` 的 engine prelude；
- 写入 `mmc_meta.conf` 的 master 脚本；
- 启动 `MetaService.main()` 的命令或说明；
- 可覆盖路径和端口的环境变量。

建议默认值：

- `WINGS_MEMCACHE_DIR=/shared-volume/memcache`；
- `WINGS_MEMCACHE_META_SERVICE_URL=tcp://127.0.0.1:5000`；
- `WINGS_MEMCACHE_CONFIG_STORE_URL=tcp://127.0.0.1:6000`；
- `WINGS_MEMCACHE_LOG_LEVEL=error`；
- `WINGS_MEMCACHE_WORLD_SIZE=256`；
- `WINGS_MEMCACHE_PROTOCOL=device_rdma`，当原生脚本要求 `device_sdma` 时按场景覆盖。

生成的 engine 脚本不应静默假设 MetaService 已经运行。它应选择以下方式之一：

- 在 engine 脚本旁生成 `start_memcache_master.sh` 并记录路径；
- 或在部署模式支持同进程/sidecar 启动时显式拉起 MetaService。

### 5. 投机解码

投机仍由以下输入控制：

- `ENABLE_SPECULATIVE_DECODE` / `SD_ENABLE`；
- `--enable-speculative-decode`；
- `SPECULATIVE_DECODE_MODEL_PATH`。

对这些 Qwen 场景：

- 无 draft model 的默认策略应为 `qwen3_5_mtp`；
- `num_speculative_tokens` 需要来自原生脚本或最终矩阵；
- 如果模型没有命中 `spec` 白名单，adapter 可以按当前规则回退到 suffix，而不是强行启用模型专用 MTP；
- 如果提供了真实 draft model 路径，仍应先进入 draft/eagle 分支，再考虑白名单驱动的 MTP 选择。

### 6. Function Call

Function Call 与 Reasoning 保持独立。

默认 parser 应来自设备默认值。当前仓库中很多 Ascend 路径对 Qwen3.5 / Qwen3.6 使用 `qwen3_coder`，部分原生脚本使用 `hermes`。

明天需要确认：

- 沿用当前仓库默认 parser；
- 使用原生脚本 parser；
- 或只在原生脚本证明差异的模型上设置 per-model override。

最终启动命令只有在 `ENABLE_AUTO_TOOL_CHOICE=true`、`--enable-auto-tool-choice` 被请求，或某个模型默认值明确要求启用时，才应输出 Function Call 相关参数。

### 7. Reasoning / Thinking

Reasoning parser 仍由以下输入控制：

- `ENABLE_AUTO_THINK_CHOICE`；
- `--enable-auto-think-choice`；
- `reason_parser.yaml`。

本稿暂按当前仓库已有 architecture/model 支持，将 Qwen3.5 / Qwen3.6 场景映射到 `qwen3`。

最终实现必须验证：

- 只有 think 开关启用时才注入 `reasoning_parser=qwen3`；
- Qwen thinking 默认值对应的 `default_chat_template_kwargs` 生成一致；
- request 级 `chat_template_kwargs` 仍能覆盖服务默认值。

### 8. 模型默认参数

默认值继续归设备 JSON 管理：

- Ascend 场景放在 `ascend_default.json`；
- NVIDIA 场景放在 `nvidia_default.json`。

不要把场景特有值推进 architecture 默认，除非该 architecture 下所有模型都确实共享这些值。

可能需要 per-scenario 或 per-model 管理的字段：

- `quantization=ascend`，用于 w8a8 / INT8 场景；
- `enable_expert_parallel=true`，用于 A3B / A10B / A17B MoE 场景；
- 原生脚本指定时的 `seed=1024`；
- `gpu_memory_utilization`；
- `max_model_len`；
- `max_num_seqs`；
- `max_num_batched_tokens`；
- `enable_prefix_caching` 或 `no_enable_prefix_caching`；
- `additional_config`；
- `compilation_config`；
- `tool_call_parser`。

### 9. 最终命令组装

生成的 vLLM 命令需要能清楚体现以下独立贡献：

- 基础模型参数和默认参数；
- Function Call flags；
- Reasoning parser flags；
- 投机配置；
- MemCache prelude 和 env exports；
- MemCache `kv_transfer_config`；
- 量化 / EP / prefix 相关设置。

对 MemCache 场景，最终命令证据必须包含：

- `MMC_LOCAL_CONFIG_PATH`；
- `--no-disable-hybrid-kv-cache-manager`；
- 包含 `AscendStoreConnector` 的 `--kv-transfer-config`；
- 生成或引用的 `start_memcache_master.sh`。

## 适配流程

1. 加载启动参数和环境变量。
2. 加载设备默认值。
3. 识别模型 architecture 和模型级默认值。
4. 解析 Qwen Day0 场景 profile。
5. 计算有效 smart feature：
   - 请求 `spec` + 命中白名单 -> 有效 `spec`；
   - 请求 `offload` + 命中白名单 -> 有效 `offload`；
   - 否则按当前规则抑制或降级。
6. 应用 Function Call 和 Reasoning gating。
7. 如果有效 offload profile 是 MemCache：
   - 渲染 MemCache 配置片段；
   - 注入 `AscendStoreConnector`；
   - 跳过 LMCache 专用 install/env 路径。
8. 构建最终 vLLM 命令。
9. 按有效状态写入 startup status / advanced features，而不是直接使用原始 env。

## 验证方案

### 静态配置测试

- 校验 `smart_feature_whitelist.json` 中的 Qwen 场景行。
- 校验 Qwen 模型默认值在需要时保持 model-specific。
- 校验未支持的 Qwen 场景不会误授予 `sparse`。
- 校验 parser 支持关系仍在 `reason_parser.yaml`，不要重复塞进 defaults。

### 单元测试

- 有效白名单测试：
  - 请求 `spec` 后，只有模型/卡型命中白名单才会有效；
  - 请求 `offload` 后，只有模型/卡型命中白名单才会有效。
- 投机策略测试：
  - Qwen Day0 场景命中白名单时使用 `qwen3_5_mtp`；
  - 未命中白名单时按预期降级。
- MemCache 测试：
  - Qwen MemCache profile 输出 `AscendStoreConnector`；
  - Qwen MemCache profile 渲染 `mmc_local.conf`；
  - Qwen MemCache profile 渲染 `start_memcache_master.sh`；
  - MemCache profile 不输出 LMCache install/patch env。
- Parser 测试：
  - Function Call 只在启用时出现；
  - Reasoning parser 只在 Thinking 启用时出现。

### Dry Run 测试

为每个支持场景生成最终启动脚本，并对比：

- model path；
- card token；
- TP/DP；
- EP；
- quantization；
- 投机配置；
- MemCache 文件和环境变量；
- Function Call parser；
- Reasoning parser；
- prefix 和 compilation config。

### Runtime Smoke 测试

对 MemCache 场景：

- 启动 `start_memcache_master.sh`；
- 确认 `MetaService.main()` 可以成功 import；
- 确认 `MMC_META_CONFIG_PATH` 和 `MMC_LOCAL_CONFIG_PATH` 指向渲染出的文件；
- 启动 engine；
- 发送一次请求；
- 确认没有 connector 初始化错误。

## 风险

- 临时特性矩阵可能高估支持范围，因为当前规则把原生脚本中出现 MemCache 信号视为支持。
- `qwen36_serve.sh` 隐藏了重要默认值，生产实现前必须展开。
- `hermes` 与 `qwen3_coder` parser 差异可能影响 Function Call 输出。
- 如果有效 offload variant 不显式，MemCache 和 LMCache 路径可能互相污染。
- `910b` / `910c` card token 解析是白名单命中的关键；如果卡型解析失败，对应白名单行会完全失效。
- 部分 Day0 脚本只是片段，不是完整可运行脚本，所以最终行为必须以生成命令为准，不能只看文件名。

## 最终矩阵待确认问题

1. 11 个场景中哪些是官方交付目标？
2. 哪些场景官方支持 MemCache 卸载？
3. 每个 w8a8 模型是否同时支持 910B 和 910C，还是只支持 `D:\Download` 下已有脚本示例？
4. Qwen3.5 Ascend 27B / 35B / 122B / 397B 是生产支持还是 Day0 示例？
5. Qwen3.6 的 Function Call parser 应使用 `qwen3_coder` 还是原生脚本中的 `hermes`？
6. 每个模型官方 `num_speculative_tokens` 是多少？
7. 每种卡型/平台的官方 MemCache protocol 和端口默认值是什么？
8. MemCache MetaService 应由 Wings 拉起、sidecar 拉起，还是由外部服务管理？
9. `/v1/startup/accel` 对 MemCache variant 应如何展示？

## 实施边界

在最终特性矩阵确认前，本文不建议直接进入完整实现。第一轮实现应限制在：

- 增加 scenario/profile 数据；
- 更新白名单行；
- 泛化 MemCache profile；
- 增加测试；
- 验证最终生成命令。

第一轮不应重构无关的 PD、LMCache、sparse 或 parser 行为。
