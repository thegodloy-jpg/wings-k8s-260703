# Qwen Day0 current dry-run comparison

Source workbook: `wings_control\docs\DAY0\qwen模型收编.xlsx` (`Sheet1`, 12 rows x 16 columns)
Model rows: 8 | standard scenarios: 11 | reuse scenarios: 4
Hardware input standard: minimal hardware_info.json with `device` and `hardware_family`; no `details` input.
Function Call parser expectation: `qwen3_coder` per adaptation decision, even where the source script text still says `hermes`.
Qwen MemCache expectation: AscendStoreConnector config must not contain `kv_load_failure_policy`.

| Scenario | Result | Source | Hardware | Active features | Failed checks | start_command.sh |
|---|---:|---|---|---|---|---|
| Qwen3.5-27B-910C | PASS | row2 / 910C optimized | `Ascend910C` | `offload,spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.5-27B-910C\start_command.sh` |
| Qwen3.5-27B-910B | PASS | row2 / 910B optimized | `Ascend910B_64G` | `spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.5-27B-910B\start_command.sh` |
| Qwen3.5-35B-A3B-910C | PASS | row3 / 910C optimized | `Ascend910C` | `spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.5-35B-A3B-910C\start_command.sh` |
| Qwen3.5-122B-A10B-910C | PASS | row4 / 910C optimized | `Ascend910C` | `spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.5-122B-A10B-910C\start_command.sh` |
| Qwen3.5-397B-A17B-910C | PASS | row5 / 910C optimized | `Ascend910C` | `spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.5-397B-A17B-910C\start_command.sh` |
| Qwen3.6-27B-910C | PASS | row6 / 910C optimized + dependency script | `Ascend910C` | `offload,spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.6-27B-910C\start_command.sh` |
| Qwen3.6-27B-w8a8-910C | PASS | row7 / 910C optimized + dependency script | `Ascend910C` | `offload,spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.6-27B-w8a8-910C\start_command.sh` |
| Qwen3.6-27B-w8a8-910B | PASS | row7 / 910B optimized | `Ascend910B_64G` | `offload,spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.6-27B-w8a8-910B\start_command.sh` |
| Qwen3.6-35B-A3B-910C | PASS | row8 / 910C optimized + dependency script | `Ascend910C` | `offload,spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.6-35B-A3B-910C\start_command.sh` |
| Qwen3.6-35B-A3B-w8a8-910C | PASS | row9 / 910C optimized + dependency script | `Ascend910C` | `offload,spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.6-35B-A3B-w8a8-910C\start_command.sh` |
| Qwen3.6-35B-A3B-w8a8-910B | PASS | row9 / 910B optimized | `Ascend910B_64G` | `offload,spec` | - | `wings_control\docs\DAY0\dry_run\qwen_day0_current\Qwen3.6-35B-A3B-w8a8-910B\start_command.sh` |

## Checked Fields

`TP`, `DP`, `port`, `served_model_name`, max sequence/token limits, GPU memory utilization, seed, `tool_call_parser`, Function Call switch, MTP method/token/enforce_eager, EP, quantization, `additional_config`, `compilation_config`, `language_model_only`, sparse absence, MemCache ports, and absence of Qwen `kv_load_failure_policy`.

Overall result: PASS
