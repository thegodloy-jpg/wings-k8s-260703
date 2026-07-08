#!/bin/bash
# Qwen3.5-397B-A17B feature run on 910C: TP=16 + EP + MTP, no MemCache offload.
set +e

cd "$(dirname "$0")"
MODEL_DIR=${MODEL_DIR:-/var/aispace/model/weight/Qwen3.5-397B-A17B}
SERVED=${SERVED:-qwen35}
PORT=${PORT:-6901}
TP_SIZE=${TP_SIZE:-16}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
LOG_DIR=${LOG_DIR:-$(dirname "$0")/logs}
mkdir -p "$LOG_DIR"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[ERR] missing model directory: $MODEL_DIR"
  exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
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

vllm serve "$MODEL_DIR" \
    --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP_SIZE" --data-parallel-size 1 --enable-expert-parallel \
    --max-num-seqs 4 --max-model-len 16384 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --seed 1024 --trust-remote-code \
    --language-model-only \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1,"enforce_eager":true}' \
    --additional-config '{"enable_cpu_binding":false,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"multistream_overlap_shared_expert":false}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    > "$LOG_DIR/397b_mtp-$(date +%Y%m%d_%H%M%S).log" 2>&1
