#!/bin/bash
# =============================================================================
# MindIE 单机引擎环境初始化脚本
# 用途: 被 _build_base_env_commands() 读取并内联到 start_command.sh
# 来源: 参考 wings/config/set_mindie_single_env.sh，适配 sidecar 架构
#
# 注意: 此脚本在 engine 容器内执行，不是在 wings-control 容器内。
# =============================================================================

# set +u: CANN 环境脚本引用未绑定变量
set +u
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/ascend-toolkit/set_env.sh ascend-toolkit/set_env.sh; else source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; } || echo 'WARN: ascend-toolkit/set_env.sh not found'
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/nnal/atb/set_env.sh nnal/atb/set_env.sh; else source /usr/local/Ascend/nnal/atb/set_env.sh; fi; } || echo 'WARN: nnal/atb/set_env.sh not found'
[ -f /usr/local/Ascend/mindie/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/mindie/set_env.sh mindie/set_env.sh --backend=atb; else source /usr/local/Ascend/mindie/set_env.sh --backend=atb; fi; } || echo 'WARN: mindie/set_env.sh not found'
[ -f /opt/atb-models/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /opt/atb-models/set_env.sh atb-models/set_env.sh; else source /opt/atb-models/set_env.sh; fi; } || echo 'WARN: atb-models/set_env.sh not found'
set -u

export NPU_MEMORY_FRACTION=0.96
