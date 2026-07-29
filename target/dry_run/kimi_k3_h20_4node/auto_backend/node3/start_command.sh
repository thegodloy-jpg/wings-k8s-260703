#!/usr/bin/env bash
set -euo pipefail
mkdir -p /var/log/wings
rm -rf /var/log/wings/prometheus_multiproc
mkdir -p /var/log/wings/prometheus_multiproc
# --- wings: env echo helpers ---
wings_source_env_with_diff() {
    local script_path="$1"
    local label="${2:-$1}"
    if [ "$#" -ge 2 ]; then
        shift 2
    else
        shift 1
    fi
    if [ ! -f "$script_path" ]; then
        echo "[wings-env-source] WARN: $label not found: $script_path"
        return 0
    fi

    local before_file after_file
    before_file="$(mktemp)"
    after_file="$(mktemp)"
    env | sort > "$before_file" || true

    set +u
    # shellcheck disable=SC1090
    source "$script_path" "$@"
    local source_rc=$?
    set -u

    env | sort > "$after_file" || true
    comm -13 "$before_file" "$after_file" | sed "s|^|[wings-env-source] $label |" || true
    rm -f "$before_file" "$after_file"
    return "$source_rc"
}
# --- end wings env echo helpers ---
export PROMETHEUS_MULTIPROC_DIR=/var/log/wings/prometheus_multiproc
echo "[wings-env] export PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR:-}"

# --- log_analyzer: Worker节点不运行分析器 ---
echo "[log_analyzer] Worker节点(node_rank > 0)，跳过分析器启动"
# 确保共享卷目录存在
mkdir -p /shared-volume

export PYTHONUNBUFFERED=1
echo "[wings-env] export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-}"
exec > >(tee -a /var/log/wings/engine-full.log | grep --line-buffered -vE '"GET\s+/(health|metrics)\s|\b(Prefill|Decode) batch\b' | tee -a /var/log/wings/engine.log) 2>&1
ENGINE_START_EPOCH=$(date +%s)
export NCCL_SOCKET_IFNAME=ens3f3
echo "[wings-env] export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
export GLOO_SOCKET_IFNAME=ens3f3
echo "[wings-env] export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
export NCCL_NVLS_ENABLE=0
echo "[wings-env] export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-}"
export NCCL_DEBUG=WARN
echo "[wings-env] export NCCL_DEBUG=${NCCL_DEBUG:-}"
export VLLM_SSM_CONV_STATE_LAYOUT=DS
echo "[wings-env] export VLLM_SSM_CONV_STATE_LAYOUT=${VLLM_SSM_CONV_STATE_LAYOUT:-}"
echo '[wings-cmd] >>> exec vllm serve /var/ai-model/Kimi-K3/ --trust-remote-code --served-model-name kimi_k3 --gpu-memory-utilization 0.98 --no-enable-flashinfer-autotune --moe-backend marlin --disable-custom-all-reduce --distributed-timeout-seconds 1200 --max-num-seqs 32 --max-num-batched-tokens 8192 --max-model-len 40960 --tensor-parallel-size 32 -cc.pass_config.fuse_allreduce_rms=False --headless --distributed-executor-backend mp --nnodes 4 --node-rank 3 --master-addr 7.6.25.57 --master-port 29501'
vllm serve /var/ai-model/Kimi-K3/ --trust-remote-code --served-model-name kimi_k3 --gpu-memory-utilization 0.98 --no-enable-flashinfer-autotune --moe-backend marlin --disable-custom-all-reduce --distributed-timeout-seconds 1200 --max-num-seqs 32 --max-num-batched-tokens 8192 --max-model-len 40960 --tensor-parallel-size 32 -cc.pass_config.fuse_allreduce_rms=False --headless --distributed-executor-backend mp --nnodes 4 --node-rank 3 --master-addr 7.6.25.57 --master-port 29501 &
ENGINE_PID=$!
echo "[Engine] Engine PID: $ENGINE_PID"

# --- Engine process wait and exception handling (with crash retry) ---
echo "[Engine] Engine process monitor started, PID=$ENGINE_PID"
if wait "$ENGINE_PID"; then
  echo "[Engine] Engine process exited normally"
  echo "[引擎] 停止日志解析进程..."
  [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
  trap - EXIT
else
  EXIT_CODE=$?
  ENGINE_DURATION=$(( $(date +%s) - ENGINE_START_EPOCH ))
  echo "[Engine] Engine process exited abnormally, exit_code=$EXIT_CODE, runtime=${ENGINE_DURATION}s"
  echo "[Engine] ┌── Engine Crash Retry ──"
  echo "[Engine] │ Reason: Engine crashed (exit_code=$EXIT_CODE, runtime=${ENGINE_DURATION}s)"
  echo "[Engine] │ Action: Retrying engine startup with same parameters (attempt 2/2)"
  echo "[Engine] └── Retry command about to execute..."
  # 清理上一次启动残留：ray head/worker 进程 + 端口占用
  if command -v ray >/dev/null 2>&1; then
    echo "[Engine] Stopping leftover Ray cluster before retry..."
    echo '[wings-cmd] >>> ray stop --force >/dev/null 2>&1 || true'
    ray stop --force >/dev/null 2>&1 || true
  fi
  # 兜底：杀掉残留的 vLLM EngineCore / WorkerProc（父进程已死但子进程可能还在）
  pkill -9 -f 'vllm.*EngineCore' 2>/dev/null || true
  pkill -9 -f 'vllm.*WorkerProc' 2>/dev/null || true
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  echo "[Engine] Waiting 5s for port release before retry..."
  sleep 5
  ENGINE_START_EPOCH=$(date +%s)
export NCCL_SOCKET_IFNAME=ens3f3
echo "[wings-env] export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
export GLOO_SOCKET_IFNAME=ens3f3
echo "[wings-env] export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
export NCCL_NVLS_ENABLE=0
echo "[wings-env] export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-}"
export NCCL_DEBUG=WARN
echo "[wings-env] export NCCL_DEBUG=${NCCL_DEBUG:-}"
export VLLM_SSM_CONV_STATE_LAYOUT=DS
echo "[wings-env] export VLLM_SSM_CONV_STATE_LAYOUT=${VLLM_SSM_CONV_STATE_LAYOUT:-}"
echo '[wings-cmd] >>> exec vllm serve /var/ai-model/Kimi-K3/ --trust-remote-code --served-model-name kimi_k3 --gpu-memory-utilization 0.98 --no-enable-flashinfer-autotune --moe-backend marlin --disable-custom-all-reduce --distributed-timeout-seconds 1200 --max-num-seqs 32 --max-num-batched-tokens 8192 --max-model-len 40960 --tensor-parallel-size 32 -cc.pass_config.fuse_allreduce_rms=False --headless --distributed-executor-backend mp --nnodes 4 --node-rank 3 --master-addr 7.6.25.57 --master-port 29501'
vllm serve /var/ai-model/Kimi-K3/ --trust-remote-code --served-model-name kimi_k3 --gpu-memory-utilization 0.98 --no-enable-flashinfer-autotune --moe-backend marlin --disable-custom-all-reduce --distributed-timeout-seconds 1200 --max-num-seqs 32 --max-num-batched-tokens 8192 --max-model-len 40960 --tensor-parallel-size 32 -cc.pass_config.fuse_allreduce_rms=False --headless --distributed-executor-backend mp --nnodes 4 --node-rank 3 --master-addr 7.6.25.57 --master-port 29501 &
ENGINE_PID=$!
echo "[Engine] Engine PID: $ENGINE_PID (retry mode)"
  echo "[Engine] Retry engine started, waiting for process exit..."
  if wait "$ENGINE_PID"; then
    echo "[Engine] Engine process exited normally (retry mode)"
      echo "[引擎] 停止日志解析进程..."
      [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
      trap - EXIT
  else
    EXIT_CODE=$?
    echo "[Engine] Retry also failed, exit_code=$EXIT_CODE — unrecoverable"

      CURR_TIME=$(date -Iseconds)
      SCRIPT_START_EPOCH="${SCRIPT_START_EPOCH:-$(date +%s)}"
      START_TIME=$(date -Iseconds -d "@${SCRIPT_START_EPOCH}")
      ELAPSED_TIME=$(( $(date +%s) - SCRIPT_START_EPOCH ))

      cat >> "/shared-volume\progress.jsonl" <<EOF
{"progress": 0, "phase_code": "engine_crash", "phase_name": "引擎进程异常退出", "status": "failed", "key_log": "引擎进程异常退出，退出码: $EXIT_CODE", "curr_time": "$CURR_TIME", "start_time": "$START_TIME", "elapsed_time_s": $ELAPSED_TIME}
EOF

      echo "[引擎] 停止日志解析进程..."
      [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
      trap - EXIT

    exit "$EXIT_CODE"
  fi
fi
