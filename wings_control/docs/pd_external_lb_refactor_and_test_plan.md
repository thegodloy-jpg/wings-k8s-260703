# PD external-lb 重构与测试方案

> 状态：方案稿。
>
> 目标：在不改变当前 PD external-lb 行为的前提下，把散落在 `config_loader.py`、`vllm_adapter.py`、`wings_control.py` 中的 PD 规则收敛到清晰边界，降低后续新增模型、连接器或拓扑规则时的维护成本。

## 执行进展

截至 2026-07-11，阶段 1/2 已完成：

- 新增 `wings_control/core/pd_external_lb.py`，集中承载 PD external-lb env 解析、pd_config registry 加载、platform overlay、KV transfer config 和 plan 生成。
- `config_loader.py` 保留原 `_get_pd_external_lb_params()`、`_build_pd_external_lb_kv()`、`_resolve_ascend_platform()` wrapper，`_apply_pd_external_lb()` 改为消费 `PDExternalLbPlan`，`_apply_pd_topology_fallback()` / `_apply_pd_final_guard()` 复用同一套 TP/DP env resolver。
- `wings_control.py` 的 PD external-lb role gate 已从 `core.config_loader` 私有函数切到 `core.pd_external_lb.is_pd_external_lb_active()`。
- 新增 `tests/test_pd_external_lb_plan.py`，覆盖 plan 合并、显式 key 保留、legacy `_pd_*` 字段兼容和 role gate 不再 import `config_loader`。
- 暂未拆 `vllm_adapter.py` 的 shell 渲染；当前仍按既有 `_pd_external_lb` / `_pd_env` / `_pd_strip_env` / `_pd_engine_overrides` 字段消费，避免扩大改动面。

已验证：

```powershell
python -m py_compile wings_control\core\pd_external_lb.py wings_control\core\config_loader.py wings_control\wings_control.py tests\test_pd_external_lb_plan.py
python -m pytest tests\test_pd_external_lb_plan.py tests\test_pd_deepseek_v4_env.py tests\test_default_config.py tests\test_speculative_strategy.py tests\test_kv_offload_gating.py tests\test_smart_feature_whitelist.py -q
python -m pytest tests\test_log_analyzer.py -q
```

说明：`python -m pytest -q` 目前在收集阶段会因已有测试的 `sys.path.insert(.../wings_control)` 污染触发 `wings_control` 顶层模块/包名冲突，表现为 `wings_control.log_analyzer` 和 `wings_control.test_env_scripts` 无法按 package 导入。`tests/test_log_analyzer.py` 单独执行通过；该问题不属于本次 PD external-lb 行为改动。

## Dry-run 验证记录

截至 2026-07-11，本次已把 PD external-lb dry-run 验证补充到 `tests/test_pd_deepseek_v4_env.py`：

- `dp_size=1`：验证生成前台单命令，不出现 `for i in $(seq...)`、`wait -n`、`kill "${pids[@]}"`、`--data-parallel-external-lb`，并确认 `__PD_INDEX__` / `__PD_KVPORT__` 已替换为字面值。
- `dp_size>1`：验证生成 fork body，包含 `for i in $(seq...)`、`wait -n`、失败时 kill 全部子进程、`--data-parallel-external-lb`，并确认 KV 占位符替换为 shell 运行时变量 `$PD_INDEX` / `$KVPORT`。
- 多节点 rank 派生：验证 `NODE_IPS` + `RANK_IP` 能正确生成 `dp_rank_start`，避免多 pod rank 撞车。
- P/D 角色原有 dry-run：继续验证官方 env、无重复 export、Mooncake linker 前缀、generic/model env builder 不泄漏、`selected=pd_external_lb_isolated` 日志路径。

本次通过的命令：

```powershell
python -m py_compile wings_control\core\pd_external_lb.py wings_control\core\config_loader.py wings_control\wings_control.py tests\test_pd_external_lb_plan.py tests\test_pd_deepseek_v4_env.py
python -m pytest tests\test_pd_deepseek_v4_env.py -q
python -m pytest tests\test_pd_external_lb_plan.py tests\test_pd_deepseek_v4_env.py tests\test_default_config.py tests\test_speculative_strategy.py tests\test_smart_feature_whitelist.py -q
```

当前工作区另有非 PD dry-run 阻塞：

- `tests/test_kv_offload_gating.py` 当前失败 8 项，失败点集中在 `wings_control/engines/vllm_adapter.py` 的 KV offload 未提交改动：Qwen3.5 NVFP4 / DeepSeek-V4 native offload 容量来源被改为仅接受 `KV_MEM_OFFLOAD_SIZE`，且 DeepSeek-V4 Flash LMCache 默认 CPU pool 行为不再满足现有测试预期。
- 该文件不是本次 PD external-lb dry-run 改动触碰的文件；在继续跑包含 KV offload 的完整回归前，需要先决定是保留新的 KV offload 语义并同步测试，还是恢复原兼容语义。

## 背景

当前 PD external-lb 不是一个独立推理引擎，而是 vLLM / vLLM-Ascend 在 PD 分离部署下的一种启动拓扑。它和普通 single / distributed 路径的核心差异包括：

- 由 `PD_ROLE`、`PD_PREFILL_*`、`PD_DECODE_*`、`DP_SIZE_LOCAL`、`PD_INDEX` 等环境变量决定角色拓扑；
- 由 `pd_config.json` 决定 connector、角色 env、角色 engine 参数、platform overlay 和 `strip_env`；
- 命中 external-lb 后不进入 Ray / headless，而是每个 pod 作为对等 standalone 单元；
- adapter 侧走 `selected=pd_external_lb_isolated`，不复用普通 vLLM generic env chain；
- pod 内可能 fork 多个独立 `vllm serve`，需要按 service 重写 port、rank、visible devices、`KVPORT` 和 bootstrap port。

保留 `selected=pd_external_lb_isolated` 是合理的；需要收敛的是 PD 规则的所有权和重复解析。

## 当前问题

PD 逻辑目前横跨多个文件：

- `wings_control/core/config_loader.py`
  - `_get_pd_external_lb_params()`：解析 `PD_*` 拓扑。
  - `_apply_pd_external_lb()`：读取 `pd_config.json`，写入 `_pd_external_lb`、`_pd_env`、`_pd_strip_env`、`_pd_engine_overrides`。
  - `_apply_pd_topology_fallback()`：在未进入 external-lb 脚本路径时写入 TP/DP fallback。
  - `_apply_pd_final_guard()`：最终再次读取 `PD_*` env 并覆盖 TP/DP。
  - `_build_pd_external_lb_kv()`：构造 PD `kv_transfer_config`。
- `wings_control/engines/vllm_adapter.py`
  - `_prepare_engine_config()`：重申 `_pd_engine_overrides`。
  - `_build_pd_external_lb_env_cmds()`：构造 PD isolated env prelude。
  - `_build_vllm_pd_external_lb_script()`：清洗 base cmd、替换占位符、合并 env、渲染 dp=1 或 fork 脚本。
  - `build_start_script()`：选择 `selected=pd_external_lb_isolated`。
- `wings_control/wings_control.py`
  - role 判定直接 import `core.config_loader._get_pd_external_lb_params()`。

这导致同一类信息在多处被读取或重新解释。后续维护风险主要是：某处新增规则后，另一个产物路径没有同步，造成配置、脚本、状态或 role 判定不一致。

## 设计原则

- 不新建 `pd_adapter.py`。PD external-lb 不是 engine，最终仍由 `vllm_adapter` 生成 vLLM 启动脚本。
- 不把 PD 强行塞回普通 `_build_vllm_common_env_cmds()`。当前测试明确要求 generic/model env 不泄漏到 PD isolated path。
- 新文件只负责 PD 领域规则和 plan 生成，不负责最终 shell 拼接。
- 第一阶段保留 `_pd_external_lb`、`_pd_env`、`_pd_strip_env`、`_pd_engine_overrides` 这些 params 字段，降低迁移风险。
- `pd_config.json` 继续作为模型 / connector / env / role engine 参数 registry。新增模型优先改 JSON，不优先改 Python 分支。

## 推荐文件

新增：

```text
wings_control/core/pd_external_lb.py
```

推荐先用单文件，而不是 package。等 PD 逻辑继续增长后，再拆成 `core/pd_external_lb/plan.py`、`registry.py`、`topology.py`。

## 职责划分

### `core/pd_external_lb.py`

建议承载以下能力：

- `PDRoleTopology`
  - 角色、TP、DP、local DP、rank start、dp address、rpc port、`PD_INDEX`。
- `PDExternalLbPlan`
  - topology；
  - connector；
  - `engine_overrides`；
  - role env；
  - strip env；
  - `kv_transfer_config`；
  - platform 信息；
  - diagnostics 所需字段。
- `resolve_pd_role_topology()`
  - 唯一读取 `PD_ROLE`、`TP_SIZE`、`DP_SIZE`、`PD_*`、`DP_SIZE_LOCAL`、`NODE_IPS`、`RANK_IP`、`PD_INDEX` 的入口。
- `load_pd_config()`
  - 读取 `wings_control/config/defaults/pd_config.json`。
- `resolve_pd_registry_entry()`
  - 根据 model architecture 和 platform overlay 选中 registry entry。
- `build_pd_kv_transfer_config()`
  - 构造 `kv_transfer_config`，保留 `__PD_INDEX__` 和 `__PD_KVPORT__` 占位符。
- `build_pd_external_lb_plan()`
  - 输入 `cmd_known_params`、`model_info`、explicit keys，输出 plan 或 `None`。
- `is_pd_external_lb_requested()`
  - 给 `wings_control.py` 做 role 判定，避免 import `config_loader` 私有函数。

### `config_loader.py`

第一阶段保留薄包装：

- `_get_pd_external_lb_params()` 调 `resolve_pd_role_topology()`，保持旧调用兼容。
- `_apply_pd_external_lb()` 调 `build_pd_external_lb_plan()`，再把 plan 应用到 `final_engine_params`。
- `_apply_pd_final_guard()` 调 `resolve_pd_role_topology()`，不再自己重复解析 env。
- `_build_pd_external_lb_kv()` 可先变成 wrapper，后续再删除。

### `vllm_adapter.py`

第一阶段只消费既有字段，不做大迁移：

- `params["_pd_external_lb"]`
- `params["_pd_env"]`
- `params["_pd_strip_env"]`
- `params["_pd_engine_overrides"]`

后续第二阶段再考虑把 `_build_vllm_pd_external_lb_script()` 拆成：

- `_strip_pd_runtime_flags(cmd)`
- `_render_pd_service_cmd(plan, base_cmd)`
- `_render_pd_single_body(plan, service_cmd)`
- `_render_pd_fork_body(plan, service_cmd)`

### `wings_control.py`

把 role 判定从 `config_loader._get_pd_external_lb_params()` 改为新文件提供的稳定函数，例如：

```python
from core.pd_external_lb import is_pd_external_lb_requested
```

## 迁移步骤

### 阶段 0：补测试，不改行为

先补下文测试，确认当前代码通过。这样重构时可以用测试保护行为。

### 阶段 1：新建 `core/pd_external_lb.py`

移动或复制纯函数逻辑：

- PD topology 解析；
- pd_config 加载；
- platform overlay；
- KV transfer config 构造；
- plan 生成。

保留 `config_loader.py` 中原函数名作为 wrapper，避免一次性修改全部调用点。

### 阶段 2：替换调用点

- `config_loader._apply_pd_external_lb()` 改为消费 plan；
- `config_loader._apply_pd_final_guard()` 改为复用 topology resolver；
- `wings_control.py` 改为调用 `is_pd_external_lb_requested()`。

### 阶段 3：清理重复解析

确认测试稳定后，删除或降级旧的重复解析代码。保留必要 wrapper 一段时间，避免外部测试或脚本直接调用私有函数时立刻断裂。

### 阶段 4：可选拆分 adapter 脚本渲染

如果后续还要继续降低 `vllm_adapter.py` 复杂度，再拆 shell 渲染函数。不要和阶段 1 混在同一个改动里。

## 测试方案

测试目标：证明重构前后的以下行为不变。

### 1. Plan 单元测试

新增：

```text
tests/test_pd_external_lb_plan.py
```

建议用例：

| 用例 | 输入 | 断言 |
| --- | --- | --- |
| `test_no_pd_role_returns_none` | 无 `PD_ROLE` | 不生成 topology / plan |
| `test_prefill_topology_from_role_env` | `PD_ROLE=P`, `PD_PREFILL_TP_SIZE=4`, `PD_PREFILL_DP_SIZE=2` | role=P, TP=4, DP=2 |
| `test_decode_topology_from_role_env` | `PD_ROLE=D`, `PD_DECODE_TP_SIZE=1`, `PD_DECODE_DP_SIZE=8` | role=D, TP=1, DP=8 |
| `test_tp_dp_size_override_role_specific_env` | 同时设置 `TP_SIZE/DP_SIZE` 和 `PD_*` | `TP_SIZE/DP_SIZE` 优先 |
| `test_dp_size_local_defaults_to_one` | 未设置 `DP_SIZE_LOCAL` | local DP=1，保留 warning |
| `test_rank_start_from_rank_ip_and_node_ips` | `NODE_IPS=ip0,ip1`, `RANK_IP=ip1`, `DP_SIZE_LOCAL=2` | rank start=2 |
| `test_pd_index_defaults_by_role` | 无 `PD_INDEX` | P=0，D=1 |
| `test_pd_index_explicit_wins` | `PD_INDEX=7` | plan 中 pd_index_base=7 |
| `test_kv_transfer_config_keeps_placeholders` | 命中 registry | JSON 包含 `__PD_INDEX__`、`__PD_KVPORT__` |
| `test_disable_ascend_direct_removes_extra_flag` | `PD_DISABLE_ASCEND_DIRECT=true` | extra config 不含 `use_ascend_direct` |

这层测试保护 `PD_*` env 解析和 registry 解释。

### 2. Config Loader 集成测试

新增：

```text
tests/test_pd_external_lb_config_loader.py
```

建议用例：

| 用例 | 断言 |
| --- | --- |
| `test_apply_pd_external_lb_writes_legacy_param_fields` | 仍写入 `_pd_external_lb`、`_pd_env`、`_pd_strip_env`、`_pd_engine_overrides` |
| `test_apply_pd_external_lb_sets_distributed_false` | 命中 PD 后 `distributed=False` |
| `test_pd_engine_overrides_reapply_registry_values` | registry 中 role engine 参数进入 `_pd_engine_overrides` |
| `test_pd_topology_overrides_standard_device_count` | TP/DP 以 PD env 为准，不回退 `device_count` |
| `test_pd_final_guard_uses_same_topology_resolver` | final guard 修正被模型默认覆盖的 TP/DP |
| `test_non_pd_does_not_write_internal_pd_fields` | 无 `PD_ROLE` 时不产生 `_pd_*` 字段 |
| `test_user_explicit_keys_are_preserved_except_topology` | 用户显式非拓扑参数不被 registry 覆盖，TP/DP 仍按 PD 拓扑 |

这层保护 `config_loader.py` 的最终 params 行为。

### 3. Adapter 脚本回归测试

保留并扩展：

```text
tests/test_pd_deepseek_v4_env.py
```

已有用例应保留：

- P/D 两端官方 env 对齐；
- 无重复 export；
- 不走 generic env builders；
- 日志包含 `selected=pd_external_lb_isolated`；
- Mooncake linker 使用命令前缀，不全局 export `LD_LIBRARY_PATH`。

建议新增：

| 用例 | 断言 |
| --- | --- |
| `test_pd_external_lb_dp1_uses_single_foreground_command` | `dp_size=1` 时不出现 `for i in $(seq...)`，不出现 `wait -n`，命令包含字面 `--tensor-parallel-size` |
| `test_pd_external_lb_multi_dp_uses_fork_body` | `dp_size>1` 时出现 `for i in $(seq...)`、`wait -n`、`kill "${pids[@]}"`、`--data-parallel-external-lb` |
| `test_pd_external_lb_replaces_pd_placeholders` | `__PD_INDEX__`、`__PD_KVPORT__` 不泄漏到最终脚本 |
| `test_pd_external_lb_strip_env_removes_bootstrap_export_and_prefix` | `strip_env` 包含 `VLLM_MOONCAKE_BOOTSTRAP_PORT` 时不 export，也不追加 inline bootstrap |
| `test_pd_external_lb_keeps_isolated_env_even_if_generic_changes` | monkeypatch generic builders 后脚本不含 sentinel |

不要断言整段 shell 完全文本一致，只断言关键语义片段。

### 4. Role 判定测试

新增：

```text
tests/test_pd_external_lb_role.py
```

建议用例：

| 用例 | 断言 |
| --- | --- |
| `test_pd_external_lb_role_is_standalone` | 设置 `PD_ROLE` 后 `_determine_role()` 返回 `standalone` |
| `test_no_pd_role_keeps_regular_role_logic` | 无 `PD_ROLE` 时仍走原 master / worker 判定 |
| `test_invalid_pd_role_does_not_force_standalone` | 非 P/D 不触发 external-lb |

这层保护 `wings_control.py` 不因 import 改动丢失部署角色行为。

### 5. Smart Feature 隔离回归

PD external-lb 和 smart trio 的关系要保持现状：

- `apply_effective_feature_enablement()` 对 PD 仍 hard-disable `spec`、`sparse`、`offload` 请求；
- `pd_config.json` 中的 `speculative_config`、parser、`additional_config` 是 PD recipe defaults，不代表 smart feature gate 失效；
- SmartQoS、RAG、parser / thinking 不应被误判为 PD hard-disable 范围。

建议补或保留相关测试：

```text
tests/test_smart_feature_whitelist.py
tests/test_speculative_strategy.py
tests/test_kv_offload_gating.py
```

重点断言：

- PD_ROLE 下 `ENABLE_SPECULATIVE_DECODE` 请求不产生普通 smart spec 路径；
- PD_ROLE 下 offload 不注入 LMCache / MemCache 普通路径；
- `pd_config.json` 的 recipe 字段仍能进入 PD engine overrides；
- 非 PD 场景的 smart feature 行为不变。

### 6. Defaults / Registry 防回归

已有用例：

```text
tests/test_default_config.py::test_pd_default_templates_do_not_enable_auto_tool_choice
```

建议保留，并新增：

| 用例 | 断言 |
| --- | --- |
| `test_pd_config_has_default_registry_entry` | `pd_config.json` 仍有 `default` entry |
| `test_pd_registry_entries_have_required_keys` | 每个 entry 至少含 connector、kv_port、common、prefill、decode |
| `test_pd_platform_overrides_do_not_mutate_cached_registry` | 多次解析不同 platform 不污染 registry cache |

## 建议执行命令

阶段 0 和阶段 1 后优先跑：

```powershell
python -m pytest tests/test_pd_external_lb_plan.py tests/test_pd_external_lb_config_loader.py tests/test_pd_deepseek_v4_env.py tests/test_default_config.py::test_pd_default_templates_do_not_enable_auto_tool_choice
```

随后跑 smart / offload 相关回归：

```powershell
python -m pytest tests/test_smart_feature_whitelist.py tests/test_kv_offload_gating.py tests/test_speculative_strategy.py
```

最后跑针对启动脚本的最小全集：

```powershell
python -m pytest tests/test_pd_deepseek_v4_env.py tests/test_wings_start.py tests/test_start_command_logging_contract.py
```

## 验收标准

- 新增 `core/pd_external_lb.py` 后，非 PD 路径不出现 `_pd_*` 字段。
- `PD_ROLE=P/D` 的 external-lb 路径仍输出 `selected=pd_external_lb_isolated`。
- P/D 角色的 TP/DP、env、strip_env、connector、`kv_transfer_config` 与重构前一致。
- `wings_control.py` 的 external-lb role 判定仍为 `standalone`。
- 普通 vLLM / vLLM-Ascend single / distributed 路径不被 PD plan 逻辑影响。
- 测试只断言行为语义，不把整段 shell 作为快照锁死。

## 风险与规避

| 风险 | 规避方式 |
| --- | --- |
| 移动代码后 `config_loader.py` wrapper 与新文件语义不一致 | wrapper 只做委托，不保留第二份解析逻辑 |
| registry entry 被深合并后污染模块缓存 | plan 生成前对 entry 做 `deepcopy`，并加 cache pollution 测试 |
| adapter 侧仍重新读 env，导致 plan 不是唯一来源 | 第一阶段允许少量兼容，第二阶段逐步改为只消费 plan 字段 |
| 测试过度依赖 shell 格式 | 只断言关键 flag、env、占位符和控制流片段 |
| 非 PD 路径被误伤 | 每次重构后跑 non-PD defaults / smart / offload focused tests |
