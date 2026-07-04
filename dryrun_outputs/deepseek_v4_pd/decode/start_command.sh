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


# Pre-flight: verify Ascend driver is accessible
if [ ! -f /usr/local/Ascend/driver/lib64/driver/libascend_hal.so ]; then
    echo 'FATAL: libascend_hal.so not found at /usr/local/Ascend/driver/lib64/driver/'
    echo 'HINT: Ensure the host Ascend driver is mounted into the container (hostPath: /usr/local/Ascend/driver)'
    exit 1
fi
export HCCL_IF_IP=10.254.124.182
echo "[wings-env] export HCCL_IF_IP=${HCCL_IF_IP:-}"
export GLOO_SOCKET_IFNAME=xxxx
echo "[wings-env] export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
export TP_SOCKET_IFNAME=xxxx
echo "[wings-env] export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-}"
export HCCL_SOCKET_IFNAME=xxxx
echo "[wings-env] export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-}"
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
echo "[wings-env] export LD_PRELOAD=${LD_PRELOAD:-}"
export VLLM_RPC_TIMEOUT=3600000
echo "[wings-env] export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
echo "[wings-env] export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-}"
export HCCL_EXEC_TIMEOUT=204
echo "[wings-env] export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-}"
export OMP_PROC_BIND=false
echo "[wings-env] export OMP_PROC_BIND=${OMP_PROC_BIND:-}"
export OMP_NUM_THREADS=10
echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
echo "[wings-env] export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-}"
export TASK_QUEUE_ENABLE=1
echo "[wings-env] export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-}"
export HCCL_OP_EXPANSION_MODE=AIV
echo "[wings-env] export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-}"
export HCCL_BUFFSIZE=1024
echo "[wings-env] export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"
export HCCL_CONNECT_TIMEOUT=1200
echo "[wings-env] export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-}"
export PD_INDEX=1
echo "[wings-env] export PD_INDEX=${PD_INDEX:-}"
(
  pids=()
  for i in $(seq 0 7); do
    RANK=$((0 + i)); PORT=$((7100 + i))
    PD_INDEX=$PD_INDEX
    KVPORT=$((30000 + PD_INDEX * 100)); BOOTSTRAP=$((23100 + i))
    LO=$((i * 1)); HI=$((LO + 1 - 1)); CARDS=$(seq -s, $LO $HI)
    ASCEND_RT_VISIBLE_DEVICES=$CARDS vllm serve 'D:\project\wings-k8s-260703\dryrun_outputs\deepseek_v4_pd\fake_model' --trust-remote-code --max-model-len 1048576 --max-num-batched-tokens 120 --gpu-memory-utilization 0.9 --api-server-count 1 --max-num-seqs 60 --enable-expert-parallel --quantization ascend --block-size 128 --async-scheduling --safetensors-load-strategy prefetch --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}' --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --host 10.254.124.182 --served-model-name Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp --default-chat-template-kwargs '{"thinking":false}' --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"'"$KVPORT"'","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":4},"decode":{"dp_size":8,"tp_size":1}},"engine_id":"'"$PD_INDEX"'"}' --seed 1024 --no-enable-prefix-caching --reasoning-parser deepseek_v4 --no-disable-hybrid-kv-cache-manager --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' --port $PORT --tensor-parallel-size 1 --data-parallel-size 8 --data-parallel-rank $RANK --data-parallel-size-local 1 --data-parallel-address 10.254.124.182 --data-parallel-rpc-port 12777 --data-parallel-external-lb &
    pids+=($!)
  done
  wait -n || true
  echo "[pd] a service exited, tearing down pod" >&2
  kill "${pids[@]}" 2>/dev/null || true
  exit 1
)
