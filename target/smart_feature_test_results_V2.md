# AI Infra 工作台 0 Day 测试结果：Smart 特性收编核对（V2）

> 来源：`output/AI Infra工作台_0 Day测试结果_收编视图.xlsx`。使用 D 列“模型权重名称”、C 列“服务器芯片组合”和 E 列“推理引擎”，对照 `wings_control/config/smart_feature_whitelist.json` 生成备注。
>
> 标注规则：`MTP` / `suffix` / `Dflash` / `Eagle` → **SmartDecoding**；`IndexCache` / `KVCache FP8` → **SmartSparse**；`Native卸载` / `LMCache` / `LMCache Ascend` / `memcache` / `MoonCake Store` → **SmartKVcache**。备注中的“白名单”仅按模型、卡型、引擎静态匹配；不会判断页面开关、PD veto 或动态启动参数。
>
> H20 在源表未区分 96 GB 与 141 GB；本表按当前 0 Day H20 场景映射为 `h20-141`。
>
> 模型名维护注释：白名单 matcher 会对 `model_name + model_path` 统一小写后做子串匹配，大小写差异不影响命中；但 `-mtp`、`-mxfp8`、`w8a8` / `w4a8` 这类后缀是模型身份的一部分，不能随意删减。行 11/32 的 DeepSeek-V4-Flash 使用白名单口径 `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp`，不要改回源表短名 `DeepSeek-V4-Flash-w8a8`；行 21 的 MiniMax-M3 使用 `MiniMax/MiniMax-M3-MXFP8`，不要改回 `MiniMax-M3`；行 43 的 DeepSeek-V4-Pro 使用 `vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp`，短名 `DeepSeek-V4-Pro-w4a8` 不命中当前白名单；行 47 的 `Kimi-K2.7-Code` 可命中白名单小写 token `kimi-k2.7-code`，不是大小写问题，旧 `Kimi-K2.7-Code-w4a8` 仍不作为别名。
>
> 特性差异维护注释：行 2-5 的 G5500+300I A2(910B) Qwen3.6 场景按文档中的 910B 特性组合复用；其中行 2、3 与行 13、14 是同一 W8A8 模型名 + 910B 芯片组合，随 910B `backend=memcache` 白名单补齐后命中 SmartDecoding + SmartKVcache，行 4、5 基础名仍只命中 SmartDecoding。行 17 的 G5680+910B Qwen3.5-35B-A3B 同样复用 910C MemCache/卸载口径，白名单已补 SmartKVcache；行 33、45 的 GLM 910C 场景仍按现状保留当前白名单能力集合，暂不新增 SmartKVcache。

| Excel 行 | 芯片（C） | 模型名称（D） | 实际 Smart 特性（N） | 备注：实际特性分类 | 白名单静态匹配 |
| ---: | --- | --- | --- | --- | --- |
| 2 | G5500+300I A2(910B) | `Qwen3.6-27B-w8a8` | —SmartDecoding+SmartKVcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致（同模型名 + 910B 复用 MemCache） | SmartDecoding + SmartKVcache |
| 3 | G5500+300I A2(910B) | `Qwen3.6-35B-A3B-w8a8` | —SmartDecoding+SmartKVcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致（同模型名 + 910B 复用 MemCache） | SmartDecoding + SmartKVcache |
| 4 | G5500+300I A2(910B) | `Qwen3.6-27B` | —SmartDecoding+SmartKVcache | SmartDecoding；与白名单静态匹配一致（复用 910B 文档特性组合） | SmartDecoding |
| 5 | G5500+300I A2(910B) | `Qwen3.6-35B-A3B` | —SmartDecoding+SmartKVcache | SmartDecoding；与白名单静态匹配一致（复用 910B 文档特性组合） | SmartDecoding |
| 6 | G5500+L20 | `bge-large-zh-v1.5` | 无 | 无可归类 Smart 特性；与白名单静态匹配一致 | 未命中 |
| 7 | G5500+L20 | `bge-reranker-large` | 无 | 无可归类 Smart 特性；与白名单静态匹配一致 | 未命中 |
| 8 | G5500+L20 | `Qwen3.6-27B` | Native卸载,<br>MTP,KVCache FP8 | SmartDecoding + SmartSparse + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartSparse + SmartKVcache |
| 9 | G5500+L20 | `Qwen3.6-35B-A3B` | Native卸载,<br>MTP,KVCache FP8 | SmartDecoding + SmartSparse + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartSparse + SmartKVcache |
| 10 | G5500+L20 | `Qwen3-Embedding-0.6B` | KVCache FP8,Native卸载 | SmartSparse + SmartKVcache；与白名单静态匹配一致 | SmartSparse + SmartKVcache |
| 11 | G5680+910B*8  | `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp` | <br>MTP,IndexCache,LMCache Ascend | SmartDecoding + SmartSparse + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartSparse + SmartKVcache |
| 12 | G5680+910B*8  | `GLM-5.1-w8a8` | <br>MTP,IndexCache | SmartDecoding + SmartSparse；与白名单静态匹配一致 | SmartDecoding + SmartSparse |
| 13 | G5680+910B*8  | `Qwen3.6-27B-w8a8` | <br>MTP,memcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致（910B 复用 910C MemCache 口径） | SmartDecoding + SmartKVcache |
| 14 | G5680+910B*8  | `Qwen3.6-35B-A3B-w8a8` | <br>MTP,memcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致（910B 复用 910C MemCache 口径） | SmartDecoding + SmartKVcache |
| 15 | G5680+910B*8  | `Qwen3.5-397B-A17B-w8a8-mtp` | <br>MTP | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 16 | G5680+910B*8  | `Qwen3.5-27B` | <br>MTP | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 17 | G5680+910B*8  | `Qwen3.5-35B-A3B` | <br>MTP,memcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致（910B 复用 910C MemCache 口径） | SmartDecoding + SmartKVcache |
| 18 | G5680+910B*8  | `GLM-4.7-W8A8-floatmtp` | <br>MTP | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 19 | G5680+910B*8  | `Qwen3-Embedding-0.6B` | 无 | 无可归类 Smart 特性；与白名单静态匹配一致 | 未命中 |
| 20 | G5680+910B*8  | `MiniMax-M2.7-w8a8-QuaRot` | Eagle | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 21 | G6550+RTX PRO 5000 * 8 | `MiniMax/MiniMax-M3-MXFP8` | Native卸载,suffix | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 22 | G6550+RTX PRO 5000 * 8 | `MiniMax-M2.5-NVFP4` | suffix,Native卸载 | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 23 | G6550+RTX PRO 5000 * 8 | `Qwen3.5-122B-A10B` | <br>MTP,Native卸载 | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 24 | G6550+RTX PRO 5000 * 8 | `Qwen3.5-27B` | Native卸载,<br>MTP | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 25 | G6550+RTX PRO 5000 * 8 | `Qwen3.5-35B-A3B` | Native卸载 | SmartKVcache；与白名单静态匹配一致 | SmartKVcache |
| 26 | G6550+RTX PRO 5000 * 8 | `Qwen-AgentWorld-35B-A3B` | suffix,LMCache | SmartDecoding + SmartKVcache；需核对（白名单：SmartDecoding） | SmartDecoding |
| 27 | G8600+H20 * 8 | `DeepSeek-V4-Flash` | Native卸载,<br>MTP,IndexCache | SmartDecoding + SmartSparse + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartSparse + SmartKVcache |
| 28 | G8600+H20 * 8 | `GLM-4.7-FP8` | Native卸载,<br>MTP,KVCache FP8 | SmartDecoding + SmartSparse + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartSparse + SmartKVcache |
| 29 | G8600+H20 * 8 | `GLM-5.1-FP8` | Native卸载,<br>MTP,IndexCache | SmartDecoding + SmartSparse + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartSparse + SmartKVcache |
| 30 | G8600+H20 * 8 | `GLM-5-FP8` | Native卸载,<br>MTP,IndexCache | SmartDecoding + SmartSparse + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartSparse + SmartKVcache |
| 31 | G8600+H20 * 8 | `DeepSeek-V3.2` | MTP  IndexCache Native卸载 | SmartDecoding + SmartSparse + SmartKVcache；需核对（白名单：未命中；当前名称/卡型未命中 vLLM H20 白名单） | 未命中 |
| 32 | G8680V3+910C*8 | `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp` | <br>MTP,IndexCache,LMCache Ascend | SmartDecoding + SmartSparse + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartSparse + SmartKVcache |
| 33 | G8680V3+910C*8 | `GLM-5.1-w8a8` | <br>MTP,IndexCache,MoonCake Store | SmartDecoding + SmartSparse + SmartKVcache；差异保留（白名单：SmartDecoding + SmartSparse；GLM 910C 暂不补 SmartKVcache） | SmartDecoding + SmartSparse |
| 34 | G8680V3+910C*8 | `Qwen3.6-27B-w8a8` | <br>MTP,memcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 35 | G8680V3+910C*8 | `Qwen3.6-35B-A3B-w8a8` | <br>MTP,memcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 36 | G8680V3+910C*8 | `Qwen3.5-35B-A3B` | <br>MTP | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 37 | G8680V3+910C*8 | `Qwen3.5-27B` | <br>MTP,memcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 38 | G8680V3+910C*8 | `Qwen3.5-122B-A10B`改成Qwen3.5-122B-A10B-w8a8-mtp | <br>MTP | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 39 | G8680V3+910C*8 | `Qwen3.5-397B-A17B-w8a8-mtp` | <br>MTP | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 40 | G8680V3+910C*8 | `Qwen3.6-27B` | <br>MTP,memcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 41 | G8680V3+910C*8 | `Qwen3.6-35B-A3B` | <br>MTP,memcache | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 42 | G8680V3+910C*8 | `DeepSeek-Coder-V2-Instruct` | <br>MTP | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 43 | G8680V3+910C*8 | `vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp` | <br>MTP,IndexCache | SmartDecoding + SmartSparse；与白名单静态匹配一致 | SmartDecoding + SmartSparse |
| 44 | G8680V3+910C*8 | `GLM-4.7-W8A8-floatmtp` | <br>MTP | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 45 | G8680V3+910C*8 | `GLM-5.2-w8a8` | <br>MTP,MoonCake Store,PD分离 | SmartDecoding + SmartKVcache；差异保留（白名单：SmartDecoding；GLM 910C 暂不补 SmartKVcache） | SmartDecoding |
| 46 | G8680V3+910C*8 | `Kimi-K2.6-w4a8` | memcache,Dflash | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 47 | G8680V3+910C*8 | `Kimi-K2.7-Code` | memcache | SmartKVcache；与白名单静态匹配一致 | SmartKVcache |
| 48 | G8680V3+910C*8 | `MiniMax-M2.7-w8a8-QuaRot` | Eagle | SmartDecoding；与白名单静态匹配一致 | SmartDecoding |
| 49 | G8680V3+910C*8 | `GLM-5.2-w4a8` | <br>MTP,IndexCache(开不了),memcache | SmartDecoding  + SmartKVcache；需核对（白名单：未命中） | 未命中 |
| 50 | TokenBox RTX PRO 5000 * 8 | `MiniMax-M2.7-NVFP4` | suffix,LMCache | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |
| 51 | TokenBox RTX PRO 5000 * 8 | `Qwen3.5-397B-A17B-NVFP4` | <br>MTP,Native卸载 | SmartDecoding + SmartKVcache；与白名单静态匹配一致 | SmartDecoding + SmartKVcache |

## 汇总

- 共 50 条测试记录。
- N 列存在可归类 Smart 特性的记录：42 条。
- 按模型 / 卡型 / 引擎命中至少一项 Smart 白名单的记录：45 条。
- 实际分类与白名单静态匹配集合不同的记录：5 条。
- 其中按本轮裁定保持现状、不修改白名单的记录：2 条（行 33、45）。
- 仍建议后续核对/补齐的记录：3 条（行 26、31、49）；其中行 31、49 为当前模型名 + 芯片组合白名单未命中。

## 固定模型名 + 芯片特性复查结论

| 类型 | Excel 行 | 机型 / 模型 | 结论 |
| --- | --- | --- | --- |
| 文档白名单列已同步 | 8、9、10、27、29、50 | L20/H20/Pro5000 中已能完整命中特性集合的行 | 当前白名单静态匹配已覆盖文档实际特性，表格已按当前命中结果改为一致。 |
| 复用 910B 特性组合 | 2、3、4、5 | G5500+300I A2(910B) / Qwen3.6 相关模型 | 行 2、3 与行 13、14 是同模型名 + 910B，随 MemCache 白名单补齐后命中 SmartDecoding + SmartKVcache；行 4、5 基础名仍只命中 SmartDecoding。 |
| 已补白名单 | 13、14、17 | G5680+910B*8 / Qwen3.6、Qwen3.5 相关模型 | 文档实际特性包含 SmartKVcache，来源是复用 910C MemCache/卸载方式；当前白名单已补 910B `backend=memcache`，命中 SmartDecoding + SmartKVcache。 |
| 保持现状 | 33、45 | G8680V3+910C*8 / GLM-5.1-w8a8、GLM-5.2-w8a8 | 文档实际特性包含 SmartKVcache；当前白名单按现状分别保留 SmartDecoding + SmartSparse、SmartDecoding，暂不补 GLM 910C SmartKVcache。 |
| 仍需核对 | 26 | G6550+RTX PRO 5000 * 8 / Qwen-AgentWorld-35B-A3B | 文档实际特性为 SmartDecoding + SmartKVcache，当前白名单仅命中 SmartDecoding，缺 SmartKVcache。 |
| 场景缺失 | 31 | G8600+H20 * 8 / DeepSeek-V3.2 | 文档实际特性为 SmartDecoding + SmartSparse + SmartKVcache，当前模型名 + H20 芯片组合未命中白名单。 |
| 场景缺失 | 49 | G8680V3+910C*8 / GLM-5.2-w4a8 | 文档实际特性为 SmartDecoding + SmartKVcache，当前模型名 + 910C 芯片组合未命中白名单。 |

## 模型名复查结论

| 类型 | Excel 行 | 结论 |
| --- | --- | --- |
| 已按白名单模型名对齐 | 11、32 | 文档短名 `DeepSeek-V4-Flash-w8a8` 已改为 `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp`，当前命中 SmartDecoding + SmartSparse + SmartKVcache。 |
| 已按白名单模型名对齐 | 21 | 文档短名 `MiniMax-M3` 已改为 `MiniMax/MiniMax-M3-MXFP8`，当前命中 SmartDecoding + SmartKVcache。 |
| 已按白名单模型名对齐 | 43 | 文档短名 `DeepSeek-V4-Pro-w4a8` 已改为 `vllm-ascend/DeepSeek-V4-Pro-w4a8-mtp`，当前命中 SmartDecoding + SmartSparse。 |
| 名称大小写非问题 | 47 | `Kimi-K2.7-Code` 与白名单 token `kimi-k2.7-code` 可直接命中；旧 `Kimi-K2.7-Code-w4a8` 被排除，不应作为别名恢复。 |
| 白名单已有精确模型名 | 15、39 | `Qwen3.5-397B-A17B-w8a8-mtp` 已有 910B/910C 精确 spec 行，父 token 已排除 `w8a8-mtp`。 |
| 仍需确认模型身份 | 31 | 当前文档是 H20 + `DeepSeek-V3.2`，白名单没有 vLLM H20 对应模型 token；现有 `DeepSeek-V3.2-w8a8` 白名单属于 Ascend 910C，不可直接等价。 |
| 仍需确认模型身份 | 38 | 模型名单元格含“改成”说明，当前能靠 `Qwen3.5-122B-A10B` 父 token 命中；若后续确认实际是 `w8a8-mtp` 子型号，应同步白名单精确 token 后再清理文本。 |
| 仍需确认模型身份 | 49 | 文档为 `GLM-5.2-w4a8`，当前白名单是 `GLM-5.2-w8a8` 路径口径；`w4a8` / `w8a8` 不能仅靠大小写或宽匹配合并。 |
