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
rm -f D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_shared/progress.jsonl

# 记录脚本开始时间（用于计算耗时）
SCRIPT_START_EPOCH=$(date +%s)

ANALYZER_CONFIG='{"engine": "vllm_ascend", "deployment_mode": "single", "hardware": "ascend", "nnodes": 1, "node_rank": 0, "distributed_backend": "mp", "tensor_parallel_size": 2, "model_name": "Qwen/Qwen3.6-35B-A3B", "model_path": "D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_model_configs/Qwen3.6-35B-A3B-910C", "backend_port": 7799}'
echo "[log_analyzer] 配置信息: $ANALYZER_CONFIG"

# 启动日志分析器（后台）
# 清除旧 __pycache__，防止跨 Python 版本的 pyc magic number 不匹配
find D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_shared/log_analyzer -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cd D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_shared && python3 -B -m log_analyzer.log_analyzer \
    --config "$ANALYZER_CONFIG" \
    --log-file /var/log/wings/engine.log \
    --progress-file D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_shared/progress.jsonl &
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
            cat >> "D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_shared/progress.jsonl" <<EARLY_FAIL_EOF
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
# --- wings-accel: install speculative decoding runtime deps (fault-tolerant) ---
if [ -f "/accel-volume/install.py" ]; then
    echo '[wings-accel] Installing speculative decoding runtime deps...'
    set +e
    python3 /accel-volume/install.py --install-runtime-deps
    SPEC_RC=$?
    set -e
    if [ $SPEC_RC -ne 0 ]; then
        echo "[wings-accel] WARNING: Speculative decoding runtime deps install failed (exit=$SPEC_RC), skipping. Service will continue without patches."
        python3 -c "import json, os; p='D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_shared/advanced_features.json'; d=json.load(open(p)) if os.path.exists(p) else {'engine':'','features':{}}; d.setdefault('features',{})['speculative_decode']=False; f=open(p+'.tmp','w'); json.dump(d,f,indent=4); f.close(); os.replace(p+'.tmp',p)"
    else
        echo '[wings-accel] Speculative decoding runtime deps installed successfully.'
    fi
else
    echo '[wings-accel] WARNING: /accel-volume/install.py not found, skipping speculative decoding runtime deps.'
fi
# --- wings-memcache: engine prelude ---
export WINGS_MEMCACHE_DIR="${WINGS_MEMCACHE_DIR:-/shared-volume/memcache}"
echo "[wings-env] export WINGS_MEMCACHE_DIR=${WINGS_MEMCACHE_DIR:-}"
export WINGS_MEMCACHE_META_SERVICE_URL="${WINGS_MEMCACHE_META_SERVICE_URL:-tcp://127.0.0.1:50071}"
echo "[wings-env] export WINGS_MEMCACHE_META_SERVICE_URL=${WINGS_MEMCACHE_META_SERVICE_URL:-}"
export WINGS_MEMCACHE_CONFIG_STORE_URL="${WINGS_MEMCACHE_CONFIG_STORE_URL:-tcp://127.0.0.1:50081}"
echo "[wings-env] export WINGS_MEMCACHE_CONFIG_STORE_URL=${WINGS_MEMCACHE_CONFIG_STORE_URL:-}"
export WINGS_MEMCACHE_LOG_LEVEL="${WINGS_MEMCACHE_LOG_LEVEL:-error}"
echo "[wings-env] export WINGS_MEMCACHE_LOG_LEVEL=${WINGS_MEMCACHE_LOG_LEVEL:-}"
export WINGS_MEMCACHE_WORLD_SIZE="${WINGS_MEMCACHE_WORLD_SIZE:-256}"
echo "[wings-env] export WINGS_MEMCACHE_WORLD_SIZE=${WINGS_MEMCACHE_WORLD_SIZE:-}"
export WINGS_MEMCACHE_PROTOCOL="${WINGS_MEMCACHE_PROTOCOL:-device_rdma}"
echo "[wings-env] export WINGS_MEMCACHE_PROTOCOL=${WINGS_MEMCACHE_PROTOCOL:-}"
export WINGS_MEMCACHE_DRAM_GB="40"
echo "[wings-env] export WINGS_MEMCACHE_DRAM_GB=${WINGS_MEMCACHE_DRAM_GB:-}"

mkdir -p "${WINGS_MEMCACHE_DIR}"
cat > "${WINGS_MEMCACHE_DIR}/mmc_local.conf" <<EOF
ock.mmc.meta_service_url = ${WINGS_MEMCACHE_META_SERVICE_URL}
ock.mmc.local_service.config_store_url = ${WINGS_MEMCACHE_CONFIG_STORE_URL}
ock.mmc.log_level = ${WINGS_MEMCACHE_LOG_LEVEL}
ock.mmc.local_service.world_size = ${WINGS_MEMCACHE_WORLD_SIZE}
ock.mmc.local_service.protocol = ${WINGS_MEMCACHE_PROTOCOL}
ock.mmc.local_service.dram.size = ${WINGS_MEMCACHE_DRAM_GB}GB
EOF
export MMC_LOCAL_CONFIG_PATH="${WINGS_MEMCACHE_DIR}/mmc_local.conf"
echo "[wings-env] export MMC_LOCAL_CONFIG_PATH=${MMC_LOCAL_CONFIG_PATH:-}"
cat > "${WINGS_MEMCACHE_DIR}/start_memcache_master.sh" <<'WINGS_MEMCACHE_MASTER'
#!/usr/bin/env bash
set -euo pipefail

export WINGS_MEMCACHE_DIR="${WINGS_MEMCACHE_DIR:-/shared-volume/memcache}"
echo "[wings-env] export WINGS_MEMCACHE_DIR=${WINGS_MEMCACHE_DIR:-}"
export WINGS_MEMCACHE_META_SERVICE_URL="${WINGS_MEMCACHE_META_SERVICE_URL:-tcp://127.0.0.1:50071}"
echo "[wings-env] export WINGS_MEMCACHE_META_SERVICE_URL=${WINGS_MEMCACHE_META_SERVICE_URL:-}"
export WINGS_MEMCACHE_CONFIG_STORE_URL="${WINGS_MEMCACHE_CONFIG_STORE_URL:-tcp://127.0.0.1:50081}"
echo "[wings-env] export WINGS_MEMCACHE_CONFIG_STORE_URL=${WINGS_MEMCACHE_CONFIG_STORE_URL:-}"
export WINGS_MEMCACHE_LOG_LEVEL="${WINGS_MEMCACHE_LOG_LEVEL:-error}"
echo "[wings-env] export WINGS_MEMCACHE_LOG_LEVEL=${WINGS_MEMCACHE_LOG_LEVEL:-}"

mkdir -p "${WINGS_MEMCACHE_DIR}"
cat > "${WINGS_MEMCACHE_DIR}/mmc_meta.conf" <<EOF
ock.mmc.meta_service_url = ${WINGS_MEMCACHE_META_SERVICE_URL}
ock.mmc.meta_service.config_store_url = ${WINGS_MEMCACHE_CONFIG_STORE_URL}
ock.mmc.log_level = ${WINGS_MEMCACHE_LOG_LEVEL}
EOF
export MMC_META_CONFIG_PATH="${WINGS_MEMCACHE_DIR}/mmc_meta.conf"
echo "[wings-env] export MMC_META_CONFIG_PATH=${MMC_META_CONFIG_PATH:-}"

echo '[wings-cmd] >>> exec python -c "from memcache_hybrid import MetaService; MetaService.main()"'
exec python -c "from memcache_hybrid import MetaService; MetaService.main()"
WINGS_MEMCACHE_MASTER
chmod +x "${WINGS_MEMCACHE_DIR}/start_memcache_master.sh"
# --- end wings-memcache: engine prelude ---
ENGINE_START_EPOCH=$(date +%s)
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
    echo "[wings-env] export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
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
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
echo "[wings-env] export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-}"
export HCCL_BUFFSIZE=512
echo "[wings-env] export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"
export OMP_PROC_BIND=false
echo "[wings-env] export OMP_PROC_BIND=${OMP_PROC_BIND:-}"
export OMP_NUM_THREADS=1
echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
export TASK_QUEUE_ENABLE=1
echo "[wings-env] export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-}"
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
echo "[wings-env] export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-}"
echo '[wings-cmd] >>> exec python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 131072 --seed 1024 --tensor-parallel-size 2 --data-parallel-size 1 --enable-expert-parallel --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --served-model-name qwen36 --async-scheduling --compilation-config '"'"'{"cudagraph_mode":"FULL_DECODE_ONLY"}'"'"' --additional-config '"'"'{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true}}'"'"' --tool-call-parser qwen3_coder --host 127.0.0.1 --port 7799 --model D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_model_configs/Qwen3.6-35B-A3B-910C --enable-auto-tool-choice --default-chat-template-kwargs '"'"'{"enable_thinking":false}'"'"' --kv-transfer-config '"'"'{"kv_connector":"AscendStoreConnector","kv...<truncated>'
python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 131072 --seed 1024 --tensor-parallel-size 2 --data-parallel-size 1 --enable-expert-parallel --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --served-model-name qwen36 --async-scheduling --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true}}' --tool-call-parser qwen3_coder --host 127.0.0.1 --port 7799 --model D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_model_configs/Qwen3.6-35B-A3B-910C --enable-auto-tool-choice --default-chat-template-kwargs '{"enable_thinking":false}' --kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache"}}' --no-disable-hybrid-kv-cache-manager --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' &
ENGINE_PID=$!
echo "[Engine] Engine PID: $ENGINE_PID (advanced features enabled)"

# --- Engine process wait and exception handling (with advanced feature fallback) ---
echo "[AdvFeature] Engine process monitor started, PID=$ENGINE_PID"
echo "[AdvFeature] Active advanced features: speculative_decode, lmcache_offload"
if wait "$ENGINE_PID"; then
  echo "[Engine] Engine process exited normally"
  echo "[引擎] 停止日志解析进程..."
  [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
  trap - EXIT
else
  EXIT_CODE=$?
  ENGINE_DURATION=$(( $(date +%s) - ENGINE_START_EPOCH ))
  echo "[Engine] Engine process exited abnormally, exit_code=$EXIT_CODE, runtime=${ENGINE_DURATION}s"

  # 一刀切策略：高级特性启用时崩溃 → 无条件禁用所有高级特性重试一次
  echo "[AdvFeature] ┌── Advanced Feature Fallback Triggered ──"
  echo "[AdvFeature] │ Reason: Engine crashed (exit_code=$EXIT_CODE, runtime=${ENGINE_DURATION}s)"
  echo "[AdvFeature] │ Features disabled: speculative_decode, lmcache_offload"
  echo "[AdvFeature] │ Action: Restarting engine without advanced features"
  echo "[AdvFeature] └── Fallback command about to execute..."
  echo "[Engine] Falling back to basic mode (disabled: speculative_decode, lmcache_offload)..."
  # 更新 advanced_features.json：引擎级特性全部置 false，RAG 保持不变
  cat > "D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_shared/advanced_features.json" <<'FEATURES_EOF'
{
    "engine": "vllm_ascend",
    "features": {
        "speculative_decode": false,
        "sparse_kv": false,
        "kv_offload": false,
        "rag_acc": false
    },
    "variants": {
        "speculative_decode": null,
        "sparse_kv": null,
        "kv_offload": null
    },
    "others": {
        "kv_mem_offload_size": 0
    }
}
FEATURES_EOF
  echo "[AdvFeature] Updated advanced_features.json: all engine features disabled"
  # 清理上一次启动残留：ray head/worker 进程 + 端口占用
  # （fallback 会重新执行 ray start --head，若旧 head 仍在则会因端口冲突失败）
  if command -v ray >/dev/null 2>&1; then
    echo "[AdvFeature] Stopping leftover Ray cluster before fallback restart..."
    echo '[wings-cmd] >>> ray stop --force >/dev/null 2>&1 || true'
    ray stop --force >/dev/null 2>&1 || true
  fi
  # 兜底：杀掉残留的 vLLM EngineCore / WorkerProc（父进程已死但子进程可能还在）
  pkill -9 -f 'vllm.*EngineCore' 2>/dev/null || true
  pkill -9 -f 'vllm.*WorkerProc' 2>/dev/null || true
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  # 一刀切：unset 所有补丁/加速层使能环境变量，退到最基本的启动命令
  # （不动 VLLM_ASCEND_ENABLE_* / VLLM_USE_V1 等常规性能 flag，它们不是补丁）
  echo "[AdvFeature] Unsetting patch/accel env vars for fallback: WINGS_ENGINE_PATCH_OPTIONS VLLM_EARS_TOLERANCE"
  unset WINGS_ENGINE_PATCH_OPTIONS
  unset VLLM_EARS_TOLERANCE
  echo "[Engine] Waiting 5s for port release before restart..."
  sleep 5
  ENGINE_START_EPOCH=$(date +%s)
# --- wings-memcache: fallback cleanup ---
unset MMC_LOCAL_CONFIG_PATH
# --- end wings-memcache: fallback cleanup ---
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
    echo "[wings-env] export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
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
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
echo "[wings-env] export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-}"
export HCCL_BUFFSIZE=512
echo "[wings-env] export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"
export OMP_PROC_BIND=false
echo "[wings-env] export OMP_PROC_BIND=${OMP_PROC_BIND:-}"
export OMP_NUM_THREADS=1
echo "[wings-env] export OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
export TASK_QUEUE_ENABLE=1
echo "[wings-env] export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-}"
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
echo "[wings-env] export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-}"
echo '[wings-cmd] >>> exec python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 131072 --seed 1024 --tensor-parallel-size 2 --data-parallel-size 1 --enable-expert-parallel --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --served-model-name qwen36 --async-scheduling --compilation-config '"'"'{"cudagraph_mode":"FULL_DECODE_ONLY"}'"'"' --additional-config '"'"'{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true}}'"'"' --tool-call-parser qwen3_coder --host 127.0.0.1 --port 7799 --model D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_model_configs/Qwen3.6-35B-A3B-910C --enable-auto-tool-choice --default-chat-template-kwargs '"'"'{"enable_thinking":false}'"'"' --no-disable-hybrid-kv-cache-manager'
python3 -m vllm.entrypoints.openai.api_server --trust-remote-code --max-model-len 131072 --seed 1024 --tensor-parallel-size 2 --data-parallel-size 1 --enable-expert-parallel --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --served-model-name qwen36 --async-scheduling --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":true}}' --tool-call-parser qwen3_coder --host 127.0.0.1 --port 7799 --model D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_model_configs/Qwen3.6-35B-A3B-910C --enable-auto-tool-choice --default-chat-template-kwargs '{"enable_thinking":false}' --no-disable-hybrid-kv-cache-manager &
ENGINE_PID=$!
echo "[Engine] Engine PID: $ENGINE_PID (advanced features disabled: speculative_decode, lmcache_offload, fallback mode)"
  echo "[AdvFeature] Fallback-mode engine started, waiting for process exit..."
  if wait "$ENGINE_PID"; then
    echo "[Engine] Engine process exited normally (fallback mode)"
    echo "[AdvFeature] Fallback-mode engine exited normally"
    echo "[引擎] 停止日志解析进程..."
    [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
    trap - EXIT
  else
    EXIT_CODE=$?
    echo "[Engine] Fallback mode also exited abnormally, exit_code=$EXIT_CODE"
    echo "[AdvFeature] ✗ Fallback mode also failed, exit_code=$EXIT_CODE — unrecoverable"

    CURR_TIME=$(date -Iseconds)
    SCRIPT_START_EPOCH="${SCRIPT_START_EPOCH:-$(date +%s)}"
    START_TIME=$(date -Iseconds -d "@${SCRIPT_START_EPOCH}")
    ELAPSED_TIME=$(( $(date +%s) - SCRIPT_START_EPOCH ))

    cat >> "D:/project/wings-k8s-260703/wings_control/docs/DAY0/dry_run/qwen_day0_current/_shared/progress.jsonl" <<EOF
{"progress": 0, "phase_code": "engine_crash", "phase_name": "引擎进程异常退出", "status": "failed", "key_log": "引擎进程异常退出，退出码: $EXIT_CODE", "curr_time": "$CURR_TIME", "start_time": "$START_TIME", "elapsed_time_s": $ELAPSED_TIME}
EOF

    echo "[引擎] 停止日志解析进程..."
    [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true
    trap - EXIT

    exit "$EXIT_CODE"
  fi
fi
