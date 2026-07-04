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

# --- log_analyzer: 启动部署进度监控（仅master节点） ---
# 清空旧的日志文件，确保 log_analyzer 只分析新的日志（避免残留内容触发误判）
rm -f /var/log/wings/engine.log
rm -f /var/log/wings/engine-full.log
rm -f D:\project\wings-k8s-260703\dryrun_outputs\deepseek_v4_pd_project\decode\progress.jsonl

# 记录脚本开始时间（用于计算耗时）
SCRIPT_START_EPOCH=$(date +%s)

ANALYZER_CONFIG='{"engine": "vllm_ascend", "deployment_mode": "single", "hardware": "ascend", "nnodes": 1, "node_rank": 0, "distributed_backend": "ray", "tensor_parallel_size": 8, "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "model_path": "D:\\project\\wings-k8s-260703\\dryrun_outputs\\deepseek_v4_pd_project\\fake_model", "backend_port": 7100}'
echo "[log_analyzer] 配置信息: $ANALYZER_CONFIG"

# 启动日志分析器（后台）
# 清除旧 __pycache__，防止跨 Python 版本的 pyc magic number 不匹配
find D:\project\wings-k8s-260703\dryrun_outputs\deepseek_v4_pd_project\decode/log_analyzer -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cd D:\project\wings-k8s-260703\dryrun_outputs\deepseek_v4_pd_project\decode && python3 -B -m log_analyzer.log_analyzer \
    --config "$ANALYZER_CONFIG" \
    --log-file /var/log/wings/engine.log \
    --progress-file D:\project\wings-k8s-260703\dryrun_outputs\deepseek_v4_pd_project\decode\progress.jsonl &
LOG_ANALYZER_PID=$!
echo "[log_analyzer] 分析器PID: $LOG_ANALYZER_PID"

# 注册清理函数（等待分析器完全退出）
cleanup_analyzer() {
    local exit_code=$?
    echo "[log_analyzer] 停止分析器..."
    if [ -n "$LOG_ANALYZER_PID" ]; then
        kill $LOG_ANALYZER_PID 2>/dev/null || true
        # 等待分析器进程完全退出，确保完成收尾工作
        wait $LOG_ANALYZER_PID 2>/dev/null || true
    fi

    if [ -n "${ENGINE_PID:-}" ]; then
        echo "[cleanup] 发送 SIGTERM 给引擎进程..."
        kill -TERM "$ENGINE_PID" 2>/dev/null || true
    else
        # ENGINE_PID 未设置说明引擎启动前脚本就失败了（如 ray: command not found）
        # 写入失败进度，让上层感知到部署失败
        if [ "$exit_code" -ne 0 ]; then
            echo "[cleanup] 引擎启动前脚本异常退出，退出码: $exit_code"
            local curr_time
            curr_time=$(date -Iseconds)
            local start_time
            start_time=$(date -Iseconds -d "@${SCRIPT_START_EPOCH}")
            local elapsed
            elapsed=$(( $(date +%s) - SCRIPT_START_EPOCH ))
            cat >> "D:\project\wings-k8s-260703\dryrun_outputs\deepseek_v4_pd_project\decode\progress.jsonl" <<EARLY_FAIL_EOF
{"progress": 0, "phase_code": "script_error", "phase_name": "启动脚本执行失败", "status": "failed", "key_log": "引擎启动前脚本异常退出，退出码: $exit_code", "curr_time": "$curr_time", "start_time": "$start_time", "elapsed_time_s": $elapsed}
EARLY_FAIL_EOF
        fi
    fi
}
trap cleanup_analyzer EXIT  SIGTERM SIGINT

export PYTHONUNBUFFERED=1
echo "[wings-env] export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-}"
exec > >(tee -a /var/log/wings/engine-full.log | grep --line-buffered -vE '"GET\s+/(health|metrics)\s|\b(Prefill|Decode) batch\b' | tee -a /var/log/wings/engine.log) 2>&1
# Patch triton driver.py: Ascend NPU has no Triton backend, return dummy driver
python3 << 'TRITON_PATCH_EOF'
try:
    import triton.runtime, os
    drv_path = os.path.join(os.path.dirname(triton.runtime.__file__), 'driver.py')
    with open(drv_path) as f:
        src = f.read()
    if 'raise RuntimeError' in src and 'PATCHED_NPU' not in src:
        patch = '''
        # PATCHED_NPU: Ascend NPU has no Triton backend, provide dummy driver
        class _NpuDummyDrv:
            def get_current_target(self):
                import types; return types.SimpleNamespace(backend='npu', arch='Ascend910B', warp_size=0)
            def get_current_device(self): return 0
            def get_device_capability(self, *a): return (0, 0)
            def get_device_properties(self, device=0):
                try:
                    import torch_npu; n = torch_npu.npu.get_device_name(device); c = 20 if '910B' in str(n) else 30
                except Exception: c = 20
                return {'num_aicore': c, 'num_vectorcore': c}
            def __getattr__(self, name): return _NpuDummyDrv()
            def __call__(self, *a, **k): return self
            def __repr__(self): return '<NpuDummy>'
            def __int__(self): return 0
            def __bool__(self): return False
        return _NpuDummyDrv()'''
        src = src.replace(
            'raise RuntimeError(f"{len(active_drivers)} active drivers ({active_drivers}). There should only be one.")',
            patch.strip()
        )
        with open(drv_path, 'w') as f:
            f.write(src)
        print('[triton-patch] Patched', drv_path, 'for Ascend NPU')
    else:
        print('[triton-patch] Already patched or not needed')
except Exception as e:
    print(f'[triton-patch] Skip: {e}')
TRITON_PATCH_EOF
# --- wings: modelslim_config.py QuaRot compatibility patch ---
python3 << 'MODELSLIM_PATCH_EOF'
try:
    import importlib.util, pathlib
    spec = importlib.util.find_spec('vllm_ascend.quantization.modelslim_config')
    if spec and spec.origin:
        p = pathlib.Path(spec.origin)
        txt = p.read_text()
        old = 'self.quant_description[shard_prefix + ' + '"' + '.weight' + '"' + ']'
        new = 'self.quant_description.get(shard_prefix + ' + '"' + '.weight' + '"' + ')'
        if old in txt:
            p.write_text(txt.replace(old, new))
            print('[modelslim-patch] Patched modelslim_config.py: dict[] -> dict.get() for QuaRot compatibility')
        else:
            print('[modelslim-patch] Already patched or pattern not found')
    else:
        print('[modelslim-patch] modelslim_config module not found, skipping')
except Exception as e:
    print(f'[modelslim-patch] Skip: {e}')
MODELSLIM_PATCH_EOF
# --- end modelslim patch ---
ENGINE_START_EPOCH=$(date +%s)
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
    ASCEND_RT_VISIBLE_DEVICES=$CARDS vllm serve 'D:\project\wings-k8s-260703\dryrun_outputs\deepseek_v4_pd_project\fake_model' --trust-remote-code --max-model-len 1048576 --max-num-batched-tokens 120 --gpu-memory-utilization 0.9 --api-server-count 1 --max-num-seqs 60 --enable-expert-parallel --quantization ascend --block-size 128 --async-scheduling --safetensors-load-strategy prefetch --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}' --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --host 10.254.124.182 --served-model-name Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp --default-chat-template-kwargs '{"thinking":false}' --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"'"$KVPORT"'","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":4},"decode":{"dp_size":8,"tp_size":1}},"engine_id":"'"$PD_INDEX"'"}' --seed 1024 --no-enable-prefix-caching --reasoning-parser deepseek_v4 --no-disable-hybrid-kv-cache-manager --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' --port $PORT --tensor-parallel-size 1 --data-parallel-size 8 --data-parallel-rank $RANK --data-parallel-size-local 1 --data-parallel-address 10.254.124.182 --data-parallel-rpc-port 12777 --data-parallel-external-lb &
    pids+=($!)
  done
  wait -n || true
  echo "[pd] a service exited, tearing down pod" >&2
  kill "${pids[@]}" 2>/dev/null || true
  exit 1
) &
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
  # 一刀切：unset 所有补丁/加速层使能环境变量，退到最基本的启动命令
  # （不动 VLLM_ASCEND_ENABLE_* / VLLM_USE_V1 等常规性能 flag，它们不是补丁）
  echo "[Engine] Unsetting patch/accel env vars for retry: WINGS_ENGINE_PATCH_OPTIONS VLLM_EARS_TOLERANCE"
  unset WINGS_ENGINE_PATCH_OPTIONS
  unset VLLM_EARS_TOLERANCE
  echo "[Engine] Waiting 5s for port release before retry..."
  sleep 5
  ENGINE_START_EPOCH=$(date +%s)
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
    ASCEND_RT_VISIBLE_DEVICES=$CARDS vllm serve 'D:\project\wings-k8s-260703\dryrun_outputs\deepseek_v4_pd_project\fake_model' --trust-remote-code --max-model-len 1048576 --max-num-batched-tokens 120 --gpu-memory-utilization 0.9 --api-server-count 1 --max-num-seqs 60 --enable-expert-parallel --quantization ascend --block-size 128 --async-scheduling --safetensors-load-strategy prefetch --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}' --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}' --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --host 10.254.124.182 --served-model-name Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp --default-chat-template-kwargs '{"thinking":false}' --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"'"$KVPORT"'","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":4},"decode":{"dp_size":8,"tp_size":1}},"engine_id":"'"$PD_INDEX"'"}' --seed 1024 --no-enable-prefix-caching --reasoning-parser deepseek_v4 --no-disable-hybrid-kv-cache-manager --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' --port $PORT --tensor-parallel-size 1 --data-parallel-size 8 --data-parallel-rank $RANK --data-parallel-size-local 1 --data-parallel-address 10.254.124.182 --data-parallel-rpc-port 12777 --data-parallel-external-lb &
    pids+=($!)
  done
  wait -n || true
  echo "[pd] a service exited, tearing down pod" >&2
  kill "${pids[@]}" 2>/dev/null || true
  exit 1
) &
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

      cat >> "D:\project\wings-k8s-260703\dryrun_outputs\deepseek_v4_pd_project\decode\progress.jsonl" <<EOF
{"progress": 0, "phase_code": "engine_crash", "phase_name": "引擎进程异常退出", "status": "failed", "key_log": "引擎进程异常退出，退出码: $EXIT_CODE", "curr_time": "$CURR_TIME", "start_time": "$START_TIME", "elapsed_time_s": $ELAPSED_TIME}
EOF

      echo "[引擎] 停止日志解析进程..."
      [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
      trap - EXIT

    exit "$EXIT_CODE"
  fi
fi
