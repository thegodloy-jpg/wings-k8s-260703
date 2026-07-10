#!/bin/bash
# Qwen3.5-27B feature run on 910B: dense bf16 + MTP, no KV offload.
set +e

MODEL_DIR=${MODEL_DIR:-/var/aispace/model/ai-storage/ai-prod/platform/Qwen3.5-27B}
SERVED=${SERVED:-qwen35}
PORT=${PORT:-7898}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}
LOG_DIR=${LOG_DIR:-/var/aispace/jsh/work/day0-jsh/logs2}
mkdir -p "$LOG_DIR"

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}
export HCCL_IF_IP=${HCCL_IF_IP:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-lo}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}

VISIBLE_COUNT=$(awk -F',' '{print NF}' <<< "$ASCEND_RT_VISIBLE_DEVICES")
TP_SIZE=${TP_SIZE:-$VISIBLE_COUNT}
LOG_FILE="$LOG_DIR/qwen35_27b_910b-$(date +%Y%m%d_%H%M%S).log"

{
  echo "[ENV] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
  echo "[ENV] TP_SIZE=${TP_SIZE}"
} > "$LOG_FILE"

vllm serve "$MODEL_DIR" \
    --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED" \
    --tensor-parallel-size "$TP_SIZE" --data-parallel-size 1 \
    --max-num-seqs 8 --max-model-len 8192 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --seed 1024 --trust-remote-code \
    --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --additional-config '{"enable_cpu_binding":false,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"multistream_overlap_shared_expert":false}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    >> "$LOG_FILE" 2>&1
