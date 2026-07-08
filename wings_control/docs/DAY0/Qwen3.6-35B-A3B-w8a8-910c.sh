#!/bin/bash
# Qwen3.6-35B-A3B-w8a8 feature run on 910C: MemCache offload + EP + MTP.
set +e

MODEL_DIR=${MODEL_DIR:-/var/aispace/model/ai-storage/ai-prod/platform/Qwen3.6-35B-A3B-w8a8}
SERVED=${SERVED:-qwen36}
PORT=${PORT:-7899}
TP_SIZE=${TP_SIZE:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
LOG_DIR=${LOG_DIR:-/home/swl/source/day0/qwen36_910c/logs}
mkdir -p "$LOG_DIR"

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9}
export HCCL_IF_IP=${HCCL_IF_IP:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-lo}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export MMC_LOCAL_CONFIG_PATH=${MMC_LOCAL_CONFIG_PATH:-/home/swl/source/day0/qwen36_910c/confs/mmc_local.conf}

KV_CONFIG=${KV_CONFIG:-'{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"0"}}'}
LOG_FILE="$LOG_DIR/qwen36_35b_a3b_w8a8_910c_memcache-$(date +%Y%m%d_%H%M%S).log"

{
  echo "[ENV] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
  echo "[ENV] TP_SIZE=${TP_SIZE}"
  echo "[ENV] MMC_LOCAL_CONFIG_PATH=${MMC_LOCAL_CONFIG_PATH}"
} > "$LOG_FILE"

vllm serve "$MODEL_DIR" \
    --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP_SIZE" --data-parallel-size 1 --enable-expert-parallel \
    --max-num-seqs 32 --max-model-len 140000 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --seed 1024 --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --quantization ascend \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config "$KV_CONFIG" \
    --async-scheduling \
    >> "$LOG_FILE" 2>&1
