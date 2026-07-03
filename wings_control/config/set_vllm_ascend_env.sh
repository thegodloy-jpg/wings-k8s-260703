#!/bin/bash
# =============================================================================
# vLLM-Ascend (华为昇腾) 引擎环境初始化脚本
# 用途: 被 _build_base_env_commands() 读取并内联到 start_command.sh
# 来源: 参考 wings/config/set_vllm_ascend_env.sh，适配 sidecar 架构
#
# 注意: 此脚本在 engine 容器内执行，不是在 wings-control 容器内。
#       因此路径应指向 engine 镜像中实际存在的位置。
#       engine 镜像预装 CANN toolkit，此处补充运行时环境变量。
# =============================================================================

# CANN/ATB 环境初始化脚本在 engine 容器内执行；若 helper 已注入，则打印 source 前后的环境变量差异。
set +u
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/ascend-toolkit/set_env.sh ascend-toolkit/set_env.sh; else source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; } || echo 'WARN: ascend-toolkit/set_env.sh not found'
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && { if command -v wings_source_env_with_diff >/dev/null 2>&1; then wings_source_env_with_diff /usr/local/Ascend/nnal/atb/set_env.sh nnal/atb/set_env.sh; else source /usr/local/Ascend/nnal/atb/set_env.sh; fi; } || echo 'WARN: nnal/atb/set_env.sh not found'
set -u

# Ascend 驱动库路径（libascend_hal.so 等位于此目录）
# 如果驱动目录不存在，说明宿主机驱动未挂载到容器
if [ -d /usr/local/Ascend/driver/lib64/driver ]; then
    export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
else
    echo '================================================================'
    echo 'FATAL: Ascend driver not found!'
    echo '  • /usr/local/Ascend/driver/lib64/driver directory is missing'
    echo '  • libascend_hal.so is required by torch_npu / vllm_ascend'
    echo ''
    echo 'FIX: Mount the host Ascend driver into the container:'
    echo '  volumeMounts:'
    echo '    - name: ascend-driver'
    echo '      mountPath: /usr/local/Ascend/driver'
    echo '  volumes:'
    echo '    - name: ascend-driver'
    echo '      hostPath:'
    echo '        path: /usr/local/Ascend/driver'
    echo '        type: Directory'
    echo '================================================================'
    exit 1
fi

# 昇腾通用环境变量
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


