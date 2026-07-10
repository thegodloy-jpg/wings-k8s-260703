# Qwen Day0 910B reuse dry-run comparison

Source: temporary 910B reuse extension; each default key is independent in ascend_default.json and temporarily copies the matching 910C Day0 optimized parameters.
Hardware input standard: minimal hardware_info.json with only device and hardware_family; no details.

| Scenario | Reuse from | Result | Hardware family | Failed checks | start_command.sh |
|---|---|---:|---|---|---|
| Qwen3.5-35B-A3B-910B-reuse | Qwen3.5-35B-A3B-910C | PASS | `Ascend910B_64G` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_910b_reuse_current/Qwen3.5-35B-A3B-910B-reuse/start_command.sh` |
| Qwen3.5-122B-A10B-910B-reuse | Qwen3.5-122B-A10B-910C | PASS | `Ascend910B_64G` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_910b_reuse_current/Qwen3.5-122B-A10B-910B-reuse/start_command.sh` |
| Qwen3.6-27B-910B-reuse | Qwen3.6-27B-910C | PASS | `Ascend910B_64G` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_910b_reuse_current/Qwen3.6-27B-910B-reuse/start_command.sh` |
| Qwen3.6-35B-A3B-910B-reuse | Qwen3.6-35B-A3B-910C | PASS | `Ascend910B_64G` | - | `wings_control/docs/DAY0/dry_run/qwen_day0_910b_reuse_current/Qwen3.6-35B-A3B-910B-reuse/start_command.sh` |
