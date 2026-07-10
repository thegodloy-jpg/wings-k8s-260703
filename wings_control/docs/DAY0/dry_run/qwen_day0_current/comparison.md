# Qwen Day0 dry-run comparison

Source: local Markdown baseline because no .xlsx/.xls file is present in wings_control/docs/DAY0 at generation time.
Hardware input standard: minimal hardware_info.json with only device and hardware_family; no details.

| Scenario | Result | Hardware family | Failed checks | start_command.sh |
|---|---:|---|---|---|
| Qwen3.5-27B-910C | PASS | `Ascend910C` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.5-27B-910C/start_command.sh` |
| Qwen3.5-27B-910B | PASS | `Ascend910B_64G` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.5-27B-910B/start_command.sh` |
| Qwen3.5-35B-A3B-910C | PASS | `Ascend910C` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.5-35B-A3B-910C/start_command.sh` |
| Qwen3.5-122B-A10B-910C | PASS | `Ascend910C` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.5-122B-A10B-910C/start_command.sh` |
| Qwen3.5-397B-A17B-910C | PASS | `Ascend910C` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.5-397B-A17B-910C/start_command.sh` |
| Qwen3.6-27B-910C | PASS | `Ascend910C` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.6-27B-910C/start_command.sh` |
| Qwen3.6-27B-w8a8-910C | PASS | `Ascend910C` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.6-27B-w8a8-910C/start_command.sh` |
| Qwen3.6-27B-w8a8-910B | PASS | `Ascend910B_64G` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.6-27B-w8a8-910B/start_command.sh` |
| Qwen3.6-35B-A3B-910C | PASS | `Ascend910C` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.6-35B-A3B-910C/start_command.sh` |
| Qwen3.6-35B-A3B-w8a8-910C | PASS | `Ascend910C` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.6-35B-A3B-w8a8-910C/start_command.sh` |
| Qwen3.6-35B-A3B-w8a8-910B | PASS | `Ascend910B_64G` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.6-35B-A3B-w8a8-910B/start_command.sh` |
