"""启动参数兼容层。

目标是把旧的 `wings_start.sh` 参数语义迁移到 Python launcher 中，
让部署脚本、环境变量和历史参数名仍然可以继续复用。
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from dataclasses import dataclass

from utils.env_utils import get_master_ip, validate_ip

logger = logging.getLogger("wings-launcher")


def _env(name: str, default: str = "") -> str:
    """从环境变量读取字符串值，不存在或为空时返回默认值。

    所有 CLI 参数都优先支持从环境变量读取，
    这样 Dockerfile / K8s Deployment 可以直接通过 env 设置参数。

    注意：平台下发的 YAML 可能设置 ENGINE: (空字符串)，
    os.getenv 会返回 "" 而不是默认值。这里统一将空字符串视为未设置。
    """
    return os.getenv(name, default) or default


def _env_int(name: str, default: int) -> int:
    """从环境变量读取整数值，解析失败时返回默认值。"""
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """从环境变量读取浮点值，解析失败或值为 inf/nan 时返回默认值。"""
    try:
        val = float(_env(name, str(default)))
        if not math.isfinite(val):
            return default
        return val
    except ValueError:
        return default


def _to_bool(raw: str | bool) -> bool:
    """统一解析命令行和环境变量中的布尔值。

    支持多种真值写法："1", "true", "yes", "on"
    支持多种假值写法："0", "false", "no", "off"
    其他值抛出 ArgumentTypeError，防止静默误判。
    """
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {raw}")


def _add_bool(parser: argparse.ArgumentParser, flag: str, env_name: str, default: bool) -> None:
    """给 parser 添加兼容型布尔参数。

    使用 nargs="?" + const=True 的技巧，使参数支持以下三种用法：
    - --flag          → True（无值时取 const）
    - --flag true     → True（显式传值）
    - 省略            → 从环境变量 env_name 读取，再回退到 default
    """
    parser.add_argument(
        flag,
        nargs="?",
        const=True,
        default=_to_bool(_env(env_name, str(default).lower())),
        type=_to_bool,
    )


_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")


def _is_valid_host_or_ip(value: str) -> bool:
    """Allow legacy IPv4 values and DNS-style service names."""
    if not value:
        return False
    return validate_ip(value) or bool(_HOSTNAME_RE.fullmatch(value))


def _normalize_distributed_aliases(args: argparse.Namespace) -> None:
    """Keep legacy distributed aliases aligned with sidecar field names."""
    node_ips = str(getattr(args, "node_ips", "") or "").strip()
    nodes = str(getattr(args, "nodes", "") or "").strip()
    if node_ips and not nodes:
        args.nodes = node_ips
    elif nodes and not node_ips:
        args.node_ips = nodes

    master_ip = str(getattr(args, "master_ip", "") or "").strip()
    if not master_ip:
        args.master_ip = get_master_ip() or ""

    head_node_addr = str(getattr(args, "head_node_addr", "") or "").strip()
    if args.master_ip and (not head_node_addr or head_node_addr == "127.0.0.1"):
        args.head_node_addr = args.master_ip

    ray_head_ip = str(getattr(args, "ray_head_ip", "") or "").strip()
    if not ray_head_ip:
        args.ray_head_ip = args.master_ip or args.head_node_addr or ""


def _validate_distributed_args(args: argparse.Namespace) -> None:
    """Validate the distributed env contract used by legacy wings.

    重要前提假设 (RANK_IP reachability):
        RANK_IP 必须是一个从 **所有** 参与分布式推理的节点都可以互相访问的 IP 地址。
        在 K8s 环境中，通常使用 Pod IP（由 CNI 分配），而非宿主机 IP。
        若使用 hostNetwork: true，则 RANK_IP 为宿主机 IP，需确保节点间网络互通
        且 HCCL 通信端口（默认 60000+）未被防火墙阻断。
    """
    if not bool(getattr(args, "distributed", False)):
        return

    rank_ip = str(os.getenv("RANK_IP", "")).strip()
    if not validate_ip(rank_ip):
        raise ValueError("distributed mode requires a valid RANK_IP environment variable")

    node_ips = str(getattr(args, "node_ips", "") or getattr(args, "nodes", "") or "").strip()
    if not node_ips:
        raise ValueError("distributed mode requires NODE_IPS or --node-ips/--nodes")

    node_list = [ip.strip() for ip in node_ips.split(",") if ip.strip()]
    if len(node_list) < 1:
        raise ValueError("distributed mode requires at least 1 node in NODE_IPS")
    invalid_node = next((ip for ip in node_list if not _is_valid_host_or_ip(ip)), None)
    if invalid_node:
        raise ValueError(f"invalid distributed node address: {invalid_node}")

    args.node_ips = ",".join(node_list)
    args.nodes = args.node_ips

    current_nnodes = int(getattr(args, "nnodes", 1) or 1)
    if current_nnodes in {0, 1}:
        args.nnodes = len(node_list)
    elif current_nnodes != len(node_list):
        raise ValueError(
            f"nnodes={current_nnodes} does not match distributed topology size={len(node_list)}"
        )

    master_ip = str(getattr(args, "master_ip", "") or get_master_ip() or "").strip()
    if not _is_valid_host_or_ip(master_ip):
        raise ValueError("distributed mode requires MASTER_IP or --master-ip")
    args.master_ip = master_ip

    if not str(getattr(args, "head_node_addr", "") or "").strip() or args.head_node_addr == "127.0.0.1":
        args.head_node_addr = master_ip
    if not str(getattr(args, "ray_head_ip", "") or "").strip():
        args.ray_head_ip = master_ip


@dataclass(frozen=True)
class LaunchArgs:
    """launcher 所需的标准化参数集合。

    该 dataclass 是 CLI 解析后的规范化输出，frozen=True 保证创建后不可变。
    所有字段均为基本类型（str/int/float/bool），便于序列化、日志和传递。

    Attributes:
        host:           监听地址，默认 0.0.0.0（绑定所有网卡）
        port:           对外服务端口，默认 18000（proxy 层端口）
        model_name:     模型名称（必填），用于日志和 API 路由标识
        model_path:     模型权重文件路径，默认 /weights
        engine:         推理引擎类型：vllm / vllm_ascend / sglang / mindie
        input_length:   最大输入序列长度（tokens），用于计算 max_model_len
        output_length:  最大输出序列长度（tokens），用于计算 max_model_len
        config_file:    用户自定义 JSON 配置文件路径
        gpu_usage_mode: GPU 使用模式：full（完整卡）/ mig（MIG 切片）/ default
        device_count:   设备数量（GPU/NPU 数），影响张量并行度
        model_type:     模型类型：llm / embedding / rerank
        trust_remote_code: 是否信任远程代码（HuggingFace 模型加载）
        dtype:          推理精度类型：auto / float16 / bfloat16 等
        kv_cache_dtype: KV Cache 存储精度
        quantization:   量化方法：空串表示不量化，可选 awq / gptq / fp8 等
        quantization_param_path: 量化参数文件路径
        gpu_memory_utilization:  GPU 显存利用率上限，默认 0.9
        enable_chunked_prefill:  是否启用分块 prefill（降低首 token 延迟）
        block_size:     PagedAttention 块大小，默认 16
        max_num_seqs:   最大并发序列数，默认 32
        seed:           随机种子
        enable_expert_parallel: 是否启用 MOE 专家并行
        max_num_batched_tokens: 单批次最大 token 数
        enable_prefix_caching:  是否启用前缀缓存（加速共享前缀场景）
        enable_speculative_decode: 是否启用推测解码
        speculative_decode_model_path: 推测解码小模型路径
        enable_rag_acc: 是否启用 RAG 加速
        enable_auto_tool_choice: 是否自动选择工具调用
        enable_sparse:  是否启用 Sparse KV Cache (v2 新增)
        distributed:    是否启用多节点分布式推理
        nnodes:         分布式节点总数
        node_rank:      当前节点编号（0 为 head 节点）
        head_node_addr: head 节点 IP 地址
        distributed_executor_backend: 分布式后端：ray / dp_deployment
    """

    host: str
    port: int
    model_name: str
    model_path: str
    engine: str
    input_length: int
    output_length: int
    config_file: str
    gpu_usage_mode: str
    device_count: int
    model_type: str
    save_path: str
    trust_remote_code: bool
    dtype: str
    kv_cache_dtype: str
    quantization: str
    quantization_param_path: str
    gpu_memory_utilization: float
    enable_chunked_prefill: bool
    block_size: int
    max_num_seqs: int
    seed: int
    enable_expert_parallel: bool
    max_num_batched_tokens: int
    enable_prefix_caching: bool
    enable_speculative_decode: bool
    speculative_decode_model_path: str
    enable_rag_acc: bool
    enable_auto_tool_choice: bool
    enable_auto_think_choice: bool
    enable_sparse: bool
    distributed: bool
    nnodes: int
    node_rank: int
    head_node_addr: str
    distributed_executor_backend: str
    node_ips: str = ""
    nodes: str = ""
    master_ip: str = ""
    ray_head_ip: str = ""
    enable_smartqos: bool = False
    enable_otlp_traces: bool = False
    engine_config: dict | None = None
    _explicit_cli_keys: list[str] | None = None

    def to_namespace(self) -> argparse.Namespace:
        """转换为 argparse.Namespace，便于传递给配置合并层 config_loader。"""
        return argparse.Namespace(**self.__dict__)


def build_parser() -> argparse.ArgumentParser:
    """维护 launcher 的 CLI 契约。

    所有参数均支持环境变量回退（通过 _env / _env_int / _env_float）。
    参数名使用 kebab-case（如 --model-name），argparse 自动转为
    snake_case（如 model_name）供代码使用。
    """
    p = argparse.ArgumentParser(prog="wings-launcher-v4")
    p.add_argument("--host", default=_env("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=_env_int("PORT", 18000))
    p.add_argument("--model-name", default=_env("MODEL_NAME", ""))
    p.add_argument("--model-path", default=_env("MODEL_PATH", "/weights"))
    p.add_argument("--engine", default=_env("ENGINE", "vllm"))
    p.add_argument("--input-length", type=int, default=_env_int("INPUT_LENGTH", 4096))
    p.add_argument("--output-length", type=int, default=_env_int("OUTPUT_LENGTH", 1024))
    p.add_argument("--config-file", default=_env("CONFIG_FILE", ""))
    p.add_argument("--gpu-usage-mode", default=_env("GPU_USAGE_MODE", "full"))
    p.add_argument("--device-count", type=int, default=_env_int("DEVICE_COUNT", 1))
    p.add_argument("--model-type", default=_env("MODEL_TYPE", "auto"))
    p.add_argument("--save-path", default=_env("SAVE_PATH", "/opt/wings/outputs"))

    _add_bool(p, "--trust-remote-code", "TRUST_REMOTE_CODE", False)
    p.add_argument("--dtype", default=_env("DTYPE", "auto"))
    p.add_argument("--kv-cache-dtype", default=_env("KV_CACHE_DTYPE", "auto"))
    p.add_argument("--quantization", default=_env("QUANTIZATION", ""))
    p.add_argument("--quantization-param-path", default=_env("QUANTIZATION_PARAM_PATH", ""))
    p.add_argument("--gpu-memory-utilization", type=float, default=_env_float("GPU_MEMORY_UTILIZATION", 0.9))
    _add_bool(p, "--enable-chunked-prefill", "ENABLE_CHUNKED_PREFILL", False)
    p.add_argument("--block-size", type=int, default=_env_int("BLOCK_SIZE", 16))
    p.add_argument("--max-num-seqs", type=int, default=_env_int("MAX_NUM_SEQS", 32))
    p.add_argument("--seed", type=int, default=_env_int("SEED", 0))
    _add_bool(p, "--enable-expert-parallel", "ENABLE_EXPERT_PARALLEL", False)
    p.add_argument("--max-num-batched-tokens", type=int, default=_env_int("MAX_NUM_BATCHED_TOKENS", 4096))
    _add_bool(p, "--enable-prefix-caching", "ENABLE_PREFIX_CACHING", False)

    _add_bool(p, "--enable-speculative-decode", "ENABLE_SPECULATIVE_DECODE", False)
    p.add_argument("--speculative-decode-model-path", default=_env("SPECULATIVE_DECODE_MODEL_PATH", ""))
    _add_bool(p, "--enable-rag-acc", "ENABLE_RAG_ACC", False)
    _add_bool(p, "--enable-auto-tool-choice", "ENABLE_AUTO_TOOL_CHOICE", False)
    # 独立的思考模式开关，与 function call 的 enable_auto_tool_choice 并列对齐、完全解耦；默认关闭。
    # 关闭时：解析端剥离 reasoning_parser；生成端（仅 vllm/vllm_ascend）在启动命令注入
    # --default-chat-template-kwargs 设服务级默认非思考（sglang 无该启动能力，仅解析端生效）。
    _add_bool(p, "--enable-auto-think-choice", "ENABLE_AUTO_THINK_CHOICE", False)

    # --- v2 新增: Sparse KV / KVStore 参数 ---
    _add_bool(p, "--enable-sparse", "ENABLE_SPARSE", False)

    _add_bool(p, "--enable-smartqos", "ENABLE_SMARTQOS", False)

    _add_bool(p, "--enable-otlp-traces", "ENABLE_OTLP_TRACES", False)

    _add_bool(p, "--distributed", "DISTRIBUTED", False)

    p.add_argument("--nnodes", type=int, default=_env_int("NNODES", 1))
    p.add_argument("--node-rank", type=int, default=0)  # Master 分发时动态注入
    p.add_argument("--head-node-addr", default=_env("HEAD_NODE_ADDR", "127.0.0.1"))
    p.add_argument("--distributed-executor-backend", default=_env("DISTRIBUTED_EXECUTOR_BACKEND", "ray"))
    p.add_argument("--node-ips", default=_env("NODE_IPS", _env("NODES", "")))
    p.add_argument("--nodes", default=_env("NODES", _env("NODE_IPS", "")))
    p.add_argument("--master-ip", default=_env("MASTER_IP", ""))
    p.add_argument("--ray-head-ip", default=_env("RAY_HEAD_IP", ""))
    return p


# 支持的推理引擎白名单；不在此集合中的 engine 值将被 parse_launch_args 拒绝
SUPPORTED_ENGINES = {"vllm", "vllm_ascend", "sglang", "mindie"}


def _map_args_to_launch_kwargs(args, engine: str) -> dict:
    """将 argparse.Namespace 转换为 LaunchArgs 构造器所需的 kwargs 字典。"""
    return {
        "host": args.host,
        "port": args.port,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "engine": engine,
        "input_length": args.input_length,
        "output_length": args.output_length,
        "config_file": args.config_file,
        "gpu_usage_mode": args.gpu_usage_mode,
        "device_count": args.device_count,
        "model_type": args.model_type,
        "save_path": args.save_path,
        "trust_remote_code": bool(args.trust_remote_code),
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "quantization": args.quantization,
        "quantization_param_path": args.quantization_param_path,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enable_chunked_prefill": bool(args.enable_chunked_prefill),
        "block_size": args.block_size,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
        "enable_expert_parallel": bool(args.enable_expert_parallel),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "enable_speculative_decode": bool(args.enable_speculative_decode),
        "speculative_decode_model_path": args.speculative_decode_model_path,
        "enable_rag_acc": bool(args.enable_rag_acc),
        "enable_auto_tool_choice": bool(args.enable_auto_tool_choice),
        "enable_auto_think_choice": bool(args.enable_auto_think_choice),
        "enable_sparse": bool(args.enable_sparse),
        "enable_smartqos": bool(args.enable_smartqos),
        "enable_otlp_traces": bool(args.enable_otlp_traces),
        "distributed": bool(args.distributed),
        "nnodes": int(args.nnodes),
        "node_rank": int(args.node_rank),
        "head_node_addr": str(args.head_node_addr),
        "distributed_executor_backend": str(args.distributed_executor_backend),
        "node_ips": str(args.node_ips),
        "nodes": str(args.nodes),
        "master_ip": str(args.master_ip),
        "ray_head_ip": str(args.ray_head_ip),
        "engine_config": getattr(args, "engine_config", None),
        "_explicit_cli_keys": getattr(args, "_explicit_cli_keys", None),
    }


def parse_launch_args(argv: list[str] | None = None) -> LaunchArgs:
    """解析命令行参数并做最小合法性校验，返回标准化 LaunchArgs。

    校验规则：
    - model_name 为必填项（空值将抛出 ValueError）
    - engine 必须在 SUPPORTED_ENGINES 白名单中

    Args:
        argv: 命令行参数列表，None 时从 sys.argv 读取

    Returns:
        LaunchArgs: 校验通过的标准化参数集合

    Raises:
        ValueError: model_name 为空或 engine 不支持
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _normalize_distributed_aliases(args)
    _validate_distributed_args(args)

    if not args.model_name:
        raise ValueError("model_name is required")
    engine = str(args.engine).lower()
    if not engine:
        engine = "vllm"
        logger.info("[参数] ENGINE 未指定或为空，自动使用默认值: %s", engine)
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(
            f"unsupported engine '{engine}'; "
            f"supported engines: {sorted(SUPPORTED_ENGINES)}"
        )

    # Auto-sync ENGINE env var to ensure consistent reads across modules
    current_env = os.getenv("ENGINE", "")
    if current_env != engine:
        os.environ["ENGINE"] = engine
        logger.info("[Params] Syncing ENGINE env var: '%s' -> '%s'", current_env, engine)

    return LaunchArgs(**_map_args_to_launch_kwargs(args, engine))