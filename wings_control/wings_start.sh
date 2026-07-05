#!/bin/bash
# =============================================================================
#  wings_start.sh — infer-control-sidecar-unified 启动入口
# =============================================================================
#
#  功能：
#    与 wings/wings/wings_start.sh 完全兼容的启动脚本，接受相同的 CLI 参数和
#    环境变量，但底层调用的是 sidecar 架构的 python -m wings_control。
#
#  设计原则：
#    - CLI 参数名称、语义、默认值与 wings_start.sh 100% 一致
#    - 环境变量优先级：CLI > 环境变量 > 脚本默认值
#    - 编排层零修改即可替换 wings 原有生产环境
#
#  架构差异（透明于调用方）：
#    ┌─ wings (A) ──────────────────────────┐
#    │ wings_start.sh → python -m wings.wings│  （单容器，进程内启动引擎）
#    └──────────────────────────────────────┘
#    ┌─ unified (B) ────────────────────────┐
#    │ wings_start.sh → python -m wings_control │  （Sidecar，生成脚本→共享卷）
#    │   ├── proxy    :18000                │
#    │   ├── health   :19000                │
#    │   └── engine   :17000 (另一容器)      │
#    └──────────────────────────────────────┘
#
#  用法：
#    bash wings_start.sh --model-name DeepSeek-R1 --model-path /weights
#    bash wings_start.sh --model-name Qwen2 --engine sglang --distributed
# =============================================================================

set -euo pipefail

# ===== 日志设置 =====
LOG_DIR="${LOG_DIR:-/var/log/wings}"
[ -d "$LOG_DIR" ] || mkdir -p "$LOG_DIR"
chmod 750 "$LOG_DIR" 2>/dev/null || true

LOG_FILE="$LOG_DIR/wings_start.log"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# 备份旧日志（保留最近 5 个）
ls -t "${LOG_FILE}".* 2>/dev/null | tail -n +6 | xargs rm -f -- 2>/dev/null || true
[ -f "$LOG_FILE" ] && mv "$LOG_FILE" "${LOG_FILE}.${TIMESTAMP}" 2>/dev/null || true

# 重定向所有输出到日志文件（同时保留 stdout 供 kubectl logs 采集）
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== [$(date)] Script started ====="

# ===== QAT 设备文件转移（与 A 保持一致） =====
if [ "${ENABLE_KV_QAT:-}" = "true" ]; then
    DEVICE_PATTERNS=("uio*" "qat_*" "usdm_drv")
    for pattern in "${DEVICE_PATTERNS[@]}"; do
        for device in /tmp/host_dev/$pattern; do
            if [ -e "$device" ] && [ ! -d "$device" ]; then
                dev_name=$(basename "$device")
                ln -sf "$device" "/dev/$dev_name"
            fi
        done
    done
fi

# ===== 帮助信息 =====
usage() {
    cat >&2 <<'EOF'
Usage: $0 [OPTIONS]
Options:
  --host <value>            Set the host address
  --port <value>            Set the port number
  --model-name <value>      Set the model name
  --trust-remote-code       Enable Trust remote code
  --dtype <value>           Set the data type
  --kv-cache-dtype <value>  Set the key-value cache data type
  --quantization <value>    Set the quantization
  --quantization-param-path <value> Set the quantization parameter path
  --gpu-memory-utilization <value> Set the GPU memory utilization
  --enable-chunked-prefill  Enable chunked prefill
  --block-size <value>       Set the block size
  --max-num-seqs <value>     Set the maximum number of sequences
  --seed <value>             Set the seed
  --enable-expert-parallel   Enable EP MOE
  --max-num-batched-tokens <value>   Set max batch tokens for prefill
  --enable-prefix-caching   Enable prefix caching
  --model-path <value>      Set the model path
  --engine <value>          Set the engine type
  --input-length <value>    Set the max input length
  --output-length <value>   Set the max output length
  --distributed             Enable distributed mode
  --config-file <value>     Specify a config file
  --gpu-usage-mode <value>  Specify gpu usage mode
  --device-count <value>    device count
  --model-type <value>      model type, should be llm or embedding or rerank or mmum or mmgm
  --save-path <value>       Top-level output dir for generated outputs (mmgm/wings)
  --enable-speculative-decode     Enable speculative decoding feature
  --speculative-decode-model-path <value>  Path to auxiliary model for speculative decoding
  --enable-sparse             Enable Sparse KV Cache feature
  --lc-sparse-threshold <value>  Set sparse attention threshold
  --total-budget <value>      Set sparse KV total memory budget ratio
  --local-kvstore-capacity <value>  Set local KVStore capacity
  --enable-rag-acc            Enable RAG acceleration feature
  --enable-auto-tool-choice   Enable function call feature
  --enable-smartqos           Enable SmartQoS priority scheduling feature
  --enable-auto-think-choice  Enable thinking (reasoning) feature
  --enable-otlp-traces        Enable OTLP traces feature (endpoint from OTLP_TRACES_ENDPOINT env, default localhost:4317)

Example:
  $0 --model-name my_model --model-path my_model_path --input-length 4096 --output-length 1024
EOF
    exit 1
}

# ===== 默认值（与 A 完全一致） =====
DEFAULT_MODEL_NAME=""
DEFAULT_MODEL_PATH="/weights"
DEFAULT_TRUST_REMOTE_CODE=""
DEFAULT_DTYPE=""
DEFAULT_KV_CACHE_DTYPE=""
DEFAULT_QUANTIZATION=""
DEFAULT_QUANTIZATION_PARAM_PATH=""
DEFAULT_GPU_MEMORY_UTILIZATION=""
DEFAULT_ENABLE_CHUNKED_PREFILL=""
DEFAULT_BLOCK_SIZE=""
DEFAULT_MAX_NUM_SEQS=""
DEFAULT_SEED=""
DEFAULT_ENABLE_EXPERT_PARALLEL=""
DEFAULT_ENGINE=""
DEFAULT_MAX_NUM_BATCHED_TOKENS=""
DEFAULT_ENABLE_PREFIX_CACHING=""
DEFAULT_INPUT_LENGTH=""
DEFAULT_OUTPUT_LENGTH=""
DEFAULT_SAVE_PATH="/opt/wings/outputs"
DEFAULT_LOG_ANALYZER="/opt/log_analyzer"

# ===== 解析命令行参数（与 A 完全一致） =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --host requires a value"; usage; }
            HOST="$2"; shift 2 ;;
        --port)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --port requires a value"; usage; }
            PORT="$2"; shift 2 ;;
        --model-name)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --model-name requires a value"; usage; }
            MODEL_NAME="$2"; shift 2 ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE=true; shift ;;
        --dtype)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --dtype requires a value"; usage; }
            DTYPE="$2"; shift 2 ;;
        --kv-cache-dtype)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --kv-cache-dtype requires a value"; usage; }
            KV_CACHE_DTYPE="$2"; shift 2 ;;
        --quantization)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --quantization requires a value"; usage; }
            QUANTIZATION="$2"; shift 2 ;;
        --quantization-param-path)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --quantization-param-path requires a value"; usage; }
            QUANTIZATION_PARAM_PATH="$2"; shift 2 ;;
        --gpu-memory-utilization)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --gpu-memory-utilization requires a value"; usage; }
            GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --enable-chunked-prefill)
            ENABLE_CHUNKED_PREFILL=true; shift ;;
        --block-size)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --block-size requires a value"; usage; }
            BLOCK_SIZE="$2"; shift 2 ;;
        --max-num-seqs)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --max-num-seqs requires a value"; usage; }
            MAX_NUM_SEQS="$2"; shift 2 ;;
        --seed)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --seed requires a value"; usage; }
            SEED="$2"; shift 2 ;;
        --enable-expert-parallel)
            ENABLE_EXPERT_PARALLEL=true; shift ;;
        --max-num-batched-tokens)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --max-num-batched-tokens requires a value"; usage; }
            MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --enable-prefix-caching)
            ENABLE_PREFIX_CACHING=true; shift ;;
        --model-path)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --model-path requires a value"; usage; }
            MODEL_PATH="$2"; shift 2 ;;
        --save-path)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --save-path requires a value"; usage; }
            SAVE_PATH="$2"; shift 2 ;;
        --engine)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --engine requires a value"; usage; }
            ENGINE="$2"; shift 2 ;;
        --input-length)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --input-length requires a value"; usage; }
            INPUT_LENGTH="$2"; shift 2 ;;
        --output-length)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --output-length requires a value"; usage; }
            OUTPUT_LENGTH="$2"; shift 2 ;;
        --distributed)
            DISTRIBUTED=true; shift ;;
        --config-file)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --config-file requires a value"; usage; }
            CONFIG_FILE="$2"; shift 2 ;;
        --gpu-usage-mode)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --gpu-usage-mode requires a value"; usage; }
            GPU_USAGE_MODE="$2"; shift 2 ;;
        --device-count)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --device-count requires a value"; usage; }
            DEVICE_COUNT="$2"; shift 2 ;;
        --model-type)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --model-type requires a value"; usage; }
            MODEL_TYPE="$2"; shift 2 ;;
        --enable-speculative-decode)
            ENABLE_SPECULATIVE_DECODE=true; shift ;;
        --speculative-decode-model-path)
            [[ -z "${2:-}" || "$2" == -* ]] && { echo "Error: --speculative-decode-model-path requires a value"; usage; }
            SPECULATIVE_DECODE_MODEL_PATH="$2"; shift 2 ;;
        --enable-sparse)
            ENABLE_SPARSE=true; shift ;;
        --enable-rag-acc)
            ENABLE_RAG_ACC=true; shift ;;
        --enable-auto-tool-choice)
            ENABLE_AUTO_TOOL_CHOICE=true; shift ;;
        --enable-smartqos)
            ENABLE_SMARTQOS=true; shift ;;
        --enable-auto-think-choice)
            ENABLE_AUTO_THINK_CHOICE=true; shift ;;
        --enable-otlp-traces)
            ENABLE_OTLP_TRACES=true; shift ;;
        -h|--help)
            usage ;;
        *)
            echo "Error: Unknown parameter: $1"
            echo "Did you mean one of these?"
            echo "  --host, --port, --model-name, --model-path, --engine"
            echo "  --input-length, --output-length, --distributed, --config-file"
            echo "  --enable-speculative-decode, --enable-sparse, --enable-rag-acc"
            usage ;;
    esac
done


# ===== 代理相关的环境控制（与 A 一致） =====
ENABLE_REASON_PROXY="${ENABLE_REASON_PROXY:-true}"

DEFAULT_HOST=""
DEFAULT_PORT="18000"
if [[ "${ENABLE_REASON_PROXY,,}" == "false" ]]; then
    BACKEND_PORT=${PORT:-$DEFAULT_PORT}
else
    # 优先使用用户通过 -e PROXY_PORT=xxx 注入的值，
    # 其次使用 PORT（兼容旧用法），最后使用缺省值 18000。
    PROXY_PORT=${PROXY_PORT:-${PORT:-$DEFAULT_PORT}}
    BACKEND_PORT="17000"
fi

# 设置默认值（如果未提供）
HOST=${HOST:-$DEFAULT_HOST}
MODEL_NAME=${MODEL_NAME:-$DEFAULT_MODEL_NAME}
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
ENGINE=${ENGINE:-$DEFAULT_ENGINE}
SAVE_PATH=${SAVE_PATH:-$DEFAULT_SAVE_PATH}

# 验证必要参数
if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: Model name is required"
    usage
fi


# ===== 导出环境变量给 wings_control（B 的 start_args_compat.py 通过 _env() 读取） =====
#
# 这里将 CLI 参数转为环境变量，使得 wings_control 中的 argparse 默认值能正确继承。
# wings_control / start_args_compat.py 的 argparse 会通过 _env("VAR", default) 读取。
#
export MODEL_NAME
export MODEL_PATH
export SAVE_PATH
[ -n "${ENGINE:-}" ]                  && export ENGINE
[ -n "${HOST:-}" ]                    && export HOST
# 代理关闭时才将 PORT 覆写为后端端口；代理开启时 PORT 保持用户指定值（proxy port，默认 18000）
if [[ "${ENABLE_REASON_PROXY,,}" == "false" ]]; then
    [ -n "${BACKEND_PORT:-}" ] && export PORT="$BACKEND_PORT"
fi
[ -n "${DTYPE:-}" ]                   && export DTYPE
[ -n "${KV_CACHE_DTYPE:-}" ]          && export KV_CACHE_DTYPE
[ -n "${QUANTIZATION:-}" ]            && export QUANTIZATION
[ -n "${QUANTIZATION_PARAM_PATH:-}" ] && export QUANTIZATION_PARAM_PATH
[ -n "${GPU_MEMORY_UTILIZATION:-}" ]  && export GPU_MEMORY_UTILIZATION
[ -n "${BLOCK_SIZE:-}" ]              && export BLOCK_SIZE
[ -n "${MAX_NUM_SEQS:-}" ]            && export MAX_NUM_SEQS
[ -n "${SEED:-}" ]                    && export SEED
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]  && export MAX_NUM_BATCHED_TOKENS
[ -n "${INPUT_LENGTH:-}" ]            && export INPUT_LENGTH
[ -n "${OUTPUT_LENGTH:-}" ]           && export OUTPUT_LENGTH
[ -n "${CONFIG_FILE:-}" ]             && export CONFIG_FILE
[ -n "${GPU_USAGE_MODE:-}" ]          && export GPU_USAGE_MODE
[ -n "${DEVICE_COUNT:-}" ]            && export DEVICE_COUNT
[ -n "${MODEL_TYPE:-}" ]              && export MODEL_TYPE
[ -n "${SPECULATIVE_DECODE_MODEL_PATH:-}" ] && export SPECULATIVE_DECODE_MODEL_PATH

# 布尔参数
[ "${TRUST_REMOTE_CODE:-}" = true ]          && export TRUST_REMOTE_CODE="true"
[ "${ENABLE_CHUNKED_PREFILL:-}" = true ]     && export ENABLE_CHUNKED_PREFILL="true"
[ "${ENABLE_EXPERT_PARALLEL:-}" = true ]     && export ENABLE_EXPERT_PARALLEL="true"
[ "${ENABLE_PREFIX_CACHING:-}" = true ]      && export ENABLE_PREFIX_CACHING="true"
[ "${DISTRIBUTED:-}" = true ]                && export DISTRIBUTED="true"
[ "${ENABLE_SPECULATIVE_DECODE:-}" = true ]  && export ENABLE_SPECULATIVE_DECODE="true"
[ "${ENABLE_SPARSE:-}" = true ]              && export ENABLE_SPARSE="true"
[ "${ENABLE_RAG_ACC:-}" = true ]             && export ENABLE_RAG_ACC="true"
[ "${ENABLE_AUTO_TOOL_CHOICE:-}" = true ]    && export ENABLE_AUTO_TOOL_CHOICE="true"
[ "${ENABLE_SMARTQOS:-}" = true ]            && export ENABLE_SMARTQOS="true"

# 代理 / 端口相关
export ENABLE_REASON_PROXY
export PROXY_PORT="${PROXY_PORT:-18000}"
# PORT 同步：start_args_compat.py 的 --port 默认值从 PORT 环境变量读取
export PORT="${PROXY_PORT}"
export RAG_ACC_ENABLED="${ENABLE_RAG_ACC:-false}"


# ===== 构建 wings_control CLI 参数 =====
#
# 虽然环境变量已导出，但 wings_control 也支持 CLI 参数（优先级高于环境变量）。
# 这里同时传递 CLI 参数以确保与 A 的行为一模一样。
# 使用数组而非字符串拼接，避免路径含空格时发生 word-splitting。
#
APP_ARGS=("--model-name" "$MODEL_NAME" "--model-path" "$MODEL_PATH")

[ -n "${SAVE_PATH:-}" ]               && APP_ARGS+=("--save-path" "$SAVE_PATH")
[ -n "${ENGINE:-}" ]                   && APP_ARGS+=("--engine" "$ENGINE")
[ "${TRUST_REMOTE_CODE:-}" = true ]    && APP_ARGS+=("--trust-remote-code")
[ -n "${DTYPE:-}" ]                    && APP_ARGS+=("--dtype" "$DTYPE")
[ -n "${KV_CACHE_DTYPE:-}" ]          && APP_ARGS+=("--kv-cache-dtype" "$KV_CACHE_DTYPE")
[ -n "${QUANTIZATION:-}" ]            && APP_ARGS+=("--quantization" "$QUANTIZATION")
[ -n "${QUANTIZATION_PARAM_PATH:-}" ] && APP_ARGS+=("--quantization-param-path" "$QUANTIZATION_PARAM_PATH")
[ -n "${GPU_MEMORY_UTILIZATION:-}" ]  && APP_ARGS+=("--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION")
[ "${ENABLE_CHUNKED_PREFILL:-}" = true ] && APP_ARGS+=("--enable-chunked-prefill")
[ -n "${BLOCK_SIZE:-}" ]              && APP_ARGS+=("--block-size" "$BLOCK_SIZE")
[ -n "${MAX_NUM_SEQS:-}" ]            && APP_ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
[ -n "${SEED:-}" ]                    && APP_ARGS+=("--seed" "$SEED")
[ "${ENABLE_EXPERT_PARALLEL:-}" = true ] && APP_ARGS+=("--enable-expert-parallel")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]  && APP_ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
[ "${ENABLE_PREFIX_CACHING:-}" = true ] && APP_ARGS+=("--enable-prefix-caching")
[ -n "${HOST:-}" ]                    && APP_ARGS+=("--host" "$HOST")
[ -n "${PROXY_PORT:-}" ]              && APP_ARGS+=("--port" "$PROXY_PORT")
[ -n "${INPUT_LENGTH:-}" ]            && APP_ARGS+=("--input-length" "$INPUT_LENGTH")
[ -n "${OUTPUT_LENGTH:-}" ]           && APP_ARGS+=("--output-length" "$OUTPUT_LENGTH")
[ -n "${CONFIG_FILE:-}" ]             && APP_ARGS+=("--config-file" "$CONFIG_FILE")
[ "${DISTRIBUTED:-}" = true ]         && APP_ARGS+=("--distributed")
[ -n "${GPU_USAGE_MODE:-}" ]          && APP_ARGS+=("--gpu-usage-mode" "$GPU_USAGE_MODE")
[ -n "${DEVICE_COUNT:-}" ]            && APP_ARGS+=("--device-count" "$DEVICE_COUNT")
[ -n "${MODEL_TYPE:-}" ]              && APP_ARGS+=("--model-type" "$MODEL_TYPE")
[ "${ENABLE_SPECULATIVE_DECODE:-}" = true ] && APP_ARGS+=("--enable-speculative-decode")
[ -n "${SPECULATIVE_DECODE_MODEL_PATH:-}" ] && APP_ARGS+=("--speculative-decode-model-path" "$SPECULATIVE_DECODE_MODEL_PATH")
[ "${ENABLE_SPARSE:-}" = true ]       && APP_ARGS+=("--enable-sparse")
[ "${ENABLE_RAG_ACC:-}" = true ]      && APP_ARGS+=("--enable-rag-acc")
[ "${ENABLE_AUTO_TOOL_CHOICE:-}" = true ] && APP_ARGS+=("--enable-auto-tool-choice")
[ "${ENABLE_SMARTQOS:-}" = true ]       && APP_ARGS+=("--enable-smartqos")
[ "${ENABLE_AUTO_THINK_CHOICE:-}" = true ] && APP_ARGS+=("--enable-auto-think-choice")
[ "${ENABLE_OTLP_TRACES:-}" = true ]   && APP_ARGS+=("--enable-otlp-traces")


# ===== 进入工作目录 =====
# B 项目的工作目录在 /opt/wings-control（Dockerfile WORKDIR）
cd "${APP_WORKDIR:-/opt/wings-control}" || exit 1

# 确保保存目录存在
if [ -n "${SAVE_PATH:-}" ]; then
    mkdir -p "$SAVE_PATH" 2>/dev/null || true
    echo "Resolved save-path: $SAVE_PATH"
fi

# 确保共享卷目录存在
mkdir -p "${SHARED_VOLUME_PATH:-/shared-volume}" 2>/dev/null || true

# 拷贝 log_analyzer 模块到共享卷
# 注意: 必须清除 __pycache__ 目录，因为 wings-control 容器(Python 3.10)
# 编译的 .pyc 文件与 engine 容器(Python 3.11+)不兼容，会导致 bad magic number 错误
if [ -d "${DEFAULT_LOG_ANALYZER:-/opt/log_analyzer}" ]; then
    echo "Copying ${DEFAULT_LOG_ANALYZER:-/opt/log_analyzer} module to /shared-volume..."
    cp -r "${DEFAULT_LOG_ANALYZER:-/opt/log_analyzer}" "${SHARED_VOLUME_PATH:-/shared-volume}/"
    find "${SHARED_VOLUME_PATH:-/shared-volume}/log_analyzer" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    echo "${DEFAULT_LOG_ANALYZER:-/opt/log_analyzer} copied successfully !"
fi

# 设置 PYTHONPATH
# 注意: /opt 必须位于 APP_WORKDIR 之前，否则 wings_control.py 文件
# 会遮蔽 /opt/wings_control 包（通过符号链接），导致 'wings_control' 被识别为
# 模块文件而非包，引发 ModuleNotFoundError
# Clear application bytecode before Python imports launcher modules. The app is
# exposed through both the flattened workdir and the /opt package alias.
find "${APP_WORKDIR:-/opt/wings-control}" "${WINGS_PACKAGE_DIR:-/opt/wings_control}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

export PYTHONPATH="/opt:${APP_WORKDIR:-/opt/wings-control}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"


# ===== 启动 Sidecar Launcher =====
#
# 与 A 的核心差异：A 直接启动 wings.wings + wings_proxy 两个进程；
# B 启动 wings_control，它内部自动管理 proxy + health 两个子进程，
# 并将引擎启动脚本写入共享卷（由另一个容器执行）。
#
echo "Starting wings application (sidecar launcher) with args: ${APP_ARGS[*]}"

# 记录新功能启用状态
[ "${ENABLE_SPECULATIVE_DECODE:-}" = true ] && echo "Speculative decode feature enabled"
[ -n "${SPECULATIVE_DECODE_MODEL_PATH:-}" ] && echo "Speculative decode model path: $SPECULATIVE_DECODE_MODEL_PATH"
[ "${ENABLE_RAG_ACC:-}" = true ] && echo "RAG acceleration feature enabled"

echo "Port plan: backend=${BACKEND_PORT} proxy=${PROXY_PORT} health=${HEALTH_PORT:-19000}"
echo "Enable proxy: ${ENABLE_REASON_PROXY}"

# 启动 launcher（前台运行，launcher 内部自带守护循环和信号处理）
# wings_control 内部会：
#   1. 解析参数 → 生成引擎启动脚本 → 写入共享卷
#   2. 启动 proxy (uvicorn :18000) 和 health (uvicorn :19000) 子进程
#   3. 进入守护循环，自动重启崩溃的子进程
#   4. 收到 SIGTERM/SIGINT 后优雅退出所有子进程
#
# 关键: 必须从 /opt 目录执行 python -m wings_control，
# 因为 Python -m 会将 cwd 作为 sys.path[0]。
# 若 cwd 为 /opt/wings-control/，则 wings_control.py(文件) 会遮蔽
# /opt/wings_control(包符号链接)，导致 ModuleNotFoundError。
# 从 /opt 启动时，sys.path[0]=/opt，Python 先找到 /opt/wings_control/(包)。
cd /opt

# 多 worker 场景下尝试调大 TCP listen backlog 上限
# 需要 privileged 容器或 CAP_NET_ADMIN；若无权限则忽略
sysctl -w net.core.somaxconn=65535 2>/dev/null || true

exec "${PYTHON_BIN}" -m wings_control "${APP_ARGS[@]}"
