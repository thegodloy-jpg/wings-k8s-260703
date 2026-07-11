# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""
vLLM 引擎适配器。

在 sidecar launcher 模式下，本模块仅负责命令拼装，不启动任何子进程。
生成的 shell 脚本将由 engine 容器读取并执行。

支持的引擎类型:
    - vllm:        NVIDIA GPU 版本，使用 NCCL 通信
    - vllm_ascend: 华为昇腾 NPU 版本，使用 HCCL 通信

分布式后端:
    - ray:           Ray 集群模式，支持多节点 TP
    - dp_deployment: 数据并行模式，支持多节点 DP
"""

import logging
import json
import os
import shlex
import stat
from typing import Dict, Any, List, Optional, Tuple

import yaml

from utils.model_utils import (ModelIdentifier, ModelIdentifierDraft,
                               INDEXCACHE_ARCHS, is_glm_moe_dsa_glm51,
                               is_qwen3_5_397b_nvfp4_vllm,
                               is_deepseek_v4_flash_rtx_pro_5000,
                               is_minimax_m27_rtx_pro_5000_vllm,
                               is_glm51_ascend_kvsparse_tmp_scope, is_glm52_model,
                               is_glm52_single_node_even, feature_allowed,
                               resolve_feature_whitelist_row_from_params,
                               resolve_offload_whitelist_backend,
                               resolve_sparse_topk)
from utils.device_utils import resolve_card_token

from utils.env_utils import get_local_ip, get_lmcache_env, \
    get_pd_role_env, get_qat_env, get_cold_start_env, \
    get_sparse_level_env
from utils.shell_env_utils import dedupe_env_exports
from utils.file_utils import safe_write_file, WriteOptions
try:
    from wings_control.features.kv_offload.memcache import hybrid as memcache_hybrid
except ImportError:
    from features.kv_offload.memcache import hybrid as memcache_hybrid  # type: ignore
from utils.vllm_helpers import (
    _format_cli_arg, _safe_int, _is_w8a8_quantize, _is_w4a8_quantize, _deep_merge_user_priority,
    _is_empty_engine_config_value, _parse_dict_like_config, Glm47DefaultMergeResult, Glm47InjectionStats,
    DistScriptCtx, DpDeploymentTopology,
)

try:
    from engines.vllm_distributed import _build_vllm_distributed_script, _resolve_dp_deployment_topology
except ImportError:
    from vllm_distributed import _build_vllm_distributed_script, _resolve_dp_deployment_topology  # noqa: F401

try:
    from wings_control.core.version_util import parse_engine_version_tuple, engine_version_platform
except ImportError:
    from core.version_util import parse_engine_version_tuple, engine_version_platform  # noqa: F811


def _sanitize_shell_path(path: str) -> str:
    """对路径进行 shell 安全转义，防止命令注入攻击。

    使用 shlex.quote() 进行标准 POSIX shell 转义，
    相比简单的正则过滤更安全且不会破坏包含空格的合法路径。

    Args:
        path: 原始文件路径字符串

    Returns:
        str: 经过 shell 安全转义的路径
    """
    return shlex.quote(path)


# ── RoCE 互联检测 ─────────────────────────────────────────────────────
_DEFAULT_RANK_TABLE_PATH_VLLM = "/workspace/rank_table_all.json"


def is_roce_distributed() -> bool:
    """Public wrapper — see :func:`_is_roce_distributed`."""
    return _is_roce_distributed()


def _is_roce_distributed() -> bool:
    """根据 ranktable 内容判断是否为 RoCE 互联场景。

    判断逻辑：读取 RANK_TABLE_PATH 指向的 ranktable JSON 文件，
    如果所有 device 条目中均不包含 ``super_device_id`` 字段，则认定为 RoCE 场景。
    包含 ``super_device_id`` 字段表示 HCCS 互联（非 RoCE）。

    Returns:
        True: RoCE 场景（ranktable 无 super_device_id）
        False: 非 RoCE 或无法判断（文件不存在/解析失败）
    """
    rank_table_path = (
        os.getenv("RANK_TABLE_PATH", "").strip() or _DEFAULT_RANK_TABLE_PATH_VLLM
    )
    if not os.path.isfile(rank_table_path):
        return False
    try:
        with open(rank_table_path, "r", encoding="utf-8") as f:
            rank_table = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    for server in rank_table.get("server_list", []):
        if not isinstance(server, dict):
            continue
        for device in server.get("device", []):
            if isinstance(device, dict) and "super_device_id" in device:
                return False
    return True

logger = logging.getLogger(__name__)

# ── 引擎版本解析 ──────────────────────────────────────────────────────
# vllm-ascend 从 v0.14 起，Ray 集群使用自定义资源 --resources='{"NPU": 1}'
# 代替 --num-gpus，以正确声明 Ascend NPU 设备。
# 低版本 (< 0.14) 沿用 --num-gpus（兼容 V1 行为）。
# 同时，v0.14 需要 Triton NPU 补丁和 --enforce-eager 标志。
_ASCEND_NPU_RESOURCE_MIN_VERSION = (0, 14)


def _parse_engine_version() -> tuple:
    """解析 ENGINE_VERSION 环境变量为 (major, minor) 元组。

    委托给 version_util.parse_engine_version_tuple()，
    支持 v0.17.0-20260325 等非标准格式。

    Returns:
        (major, minor) 整数元组；若未设置或格式异常，返回 (0, 17)（与 supported_features.json 对齐）。
    """
    return parse_engine_version_tuple()


def _get_ray_resource_flag(engine: str, params: dict) -> str:
    """根据引擎类型和版本返回 Ray 节点资源声明标志。

    版本策略：
      - vllm (NVIDIA):         始终使用 --num-gpus=1
      - vllm_ascend >= 0.14:   使用 --resources='{"NPU": 1}'（NPU 自定义资源）
      - vllm_ascend < 0.14:    使用 --num-gpus {tp_size}（兼容 V1 行为）

    可通过 RAY_RESOURCE_FLAG 环境变量完全覆盖自动检测结果。
    """
    # 环境变量覆盖 — 允许用户完全自定义
    override = os.getenv("RAY_RESOURCE_FLAG", "").strip()
    if override:
        logger.info("[ray] Using RAY_RESOURCE_FLAG override: %s", override)
        return override

    if engine != "vllm_ascend":
        return "--num-gpus=1"

    ver = _parse_engine_version()
    if ver >= _ASCEND_NPU_RESOURCE_MIN_VERSION:
        # device_count 已经是每节点的设备数（DEVICE_COUNT 环境变量），
        # 全局 TP = device_count * nnodes 在 _adjust_tensor_parallelism 中计算。
        # 此处直接用 device_count 作为每节点 NPU 资源数。
        npu_per_node = max(1, params.get("device_count", 1))
        logger.info("[ray] Ascend engine version %s >= 0.14, using --resources NPU=%d", ver, npu_per_node)
        return f"--resources='{{\"NPU\": {npu_per_node}}}'"
    else:
        tp_size = params.get("device_count", 1)
        logger.info("[ray] Ascend engine version %s < 0.14, using --num-gpus=%d (V1 compat)", ver, tp_size)
        return f"--num-gpus={tp_size}"


def get_ray_resource_flag(engine: str, params: dict) -> str:
    """Return the Ray resource flag for distributed vLLM scripts."""
    return _get_ray_resource_flag(engine, params)


def _need_triton_patch(engine: str) -> bool:
    """判断是否需要 Triton NPU 驱动补丁。

    仅 vllm_ascend >= 0.14 需要 Triton NPU 驱动补丁（解决 "0 active drivers" 崩溃）。
    此补丁是安全的一次性文件修改，不影响性能。
    """
    if engine != "vllm_ascend":
        return False
    ver = _parse_engine_version()
    return ver >= _ASCEND_NPU_RESOURCE_MIN_VERSION


def _need_enforce_eager(engine: str) -> bool:
    """判断是否需要 --enforce-eager 标志（跳过图编译）。

    A+X 环境（Ascend + NVIDIA GPU 混合部署）中，triton 和 triton-ascend
    版本冲突会导致 qkv_rmsnorm_rope 等算子无法正确注册
    (参见 vllm-ascend issue #6737, #6578)，需要通过 --enforce-eager 绕过。

    通过环境变量 ASCEND_ENFORCE_EAGER 控制：
      - true:  强制添加 --enforce-eager（用于 A+X 环境或遇到 triton 冲突时）
      - false: 不添加 --enforce-eager（默认，用于纯 Ascend 环境，可享受图编译性能优化）
    """
    if engine != "vllm_ascend":
        return False
    return os.getenv("ASCEND_ENFORCE_EAGER", "").lower() in ("true", "1", "yes")


def need_enforce_eager(engine: str) -> bool:
    """Return whether the start command should append --enforce-eager."""
    return _need_enforce_eager(engine)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_DEEPSEEK_ASCEND_DP_ARCHES = {
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    # V4 (Flash/Pro) 也走 dp_deployment：``_resolve_dp_deployment_topology``
    # 据此识别需要按 ``device_count/tp`` 计算 dp_size_local，否则会落到
    # ``(nnodes, 1, node_rank)`` 默认分支，与 V4-Flash 双机 DP=4 / V4-Pro DP=2 不符。
    "DeepseekV4ForCausalLM",
    "GlmMoeDsaForCausalLM",
    "KimiK25ForConditionalGeneration",
}

# GLM-5.2 单机(a3)官方 recipe 的全局 DP 度。单机每节点用 DP 个 replica，
# 每 replica 占 device_count // DP 卡，故 TP = device_count // _GLM52_SINGLE_NODE_DP，
# 且 TP × DP == device_count（精确占满本节点，不超订）。具名以替代散落的字面量 2。
_GLM52_SINGLE_NODE_DP = 2


def _default_deepseek_ascend_dp_tensor_parallel_size(
    model_architecture: str,
    device_count: Optional[int],
) -> Optional[int]:
    """Return the recommended default TP size for Ascend DP architectures."""
    if not device_count or device_count <= 0:
        return None
    if model_architecture == "KimiK25ForConditionalGeneration" and device_count in (8, 16):
        return device_count
    if model_architecture == "DeepseekV32ForCausalLM" and device_count in (8, 16):
        return device_count
    if model_architecture == "DeepseekV3ForCausalLM":
        return 4 if device_count >= 4 else device_count
    if model_architecture == "GlmMoeDsaForCausalLM" and device_count in (8, 16):
        return device_count
    return None


def default_deepseek_ascend_dp_tensor_parallel_size(
    model_architecture: str,
    device_count: Optional[int],
) -> Optional[int]:
    """Return the default TP size for Ascend DP architectures."""
    return _default_deepseek_ascend_dp_tensor_parallel_size(model_architecture, device_count)


def is_deepseek_ascend_dp_architecture(model_architecture: Optional[str]) -> bool:
    """Return whether the model architecture uses Ascend DP topology rules."""
    return model_architecture in _DEEPSEEK_ASCEND_DP_ARCHES


def _get_deepseek_ascend_dp_model_architecture(params: Dict[str, Any]) -> Optional[str]:
    """Return DeepSeek Ascend dp_deployment architecture, otherwise None."""
    if params.get("engine") != "vllm_ascend":
        return None
    if params.get("distributed_executor_backend") != "dp_deployment":
        return None
    model_path = params.get("model_path")
    if not model_path:
        return None
    model_info = ModelIdentifier(
        params.get("model_name"),
        model_path,
        params.get("model_type"),
    )
    if model_info.model_architecture in _DEEPSEEK_ASCEND_DP_ARCHES:
        return model_info.model_architecture
    return None


def get_deepseek_ascend_dp_model_architecture(params: Dict[str, Any]) -> Optional[str]:
    """Return the Ascend dp_deployment model architecture, otherwise None."""
    return _get_deepseek_ascend_dp_model_architecture(params)


def _need_triton_patch_and_eager(engine: str) -> bool:
    """兼容旧接口：判断是否需要 Triton 补丁或 enforce-eager。

    此函数已拆分为 _need_triton_patch() 和 _need_enforce_eager()，
    保留此函数仅用于向后兼容。
    """
    return _need_triton_patch(engine) or _need_enforce_eager(engine)

# 模块根目录：用于定位配置文件和环境脚本
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _inline_ascend_env_script(config_dir: str, engine: str) -> List[str]:
    """读取 Ascend 引擎环境脚本并内联为 shell 命令列表。

    脚本映射:
      - vllm_ascend → config/set_vllm_ascend_env.sh
      - mindie      → config/set_mindie_env.sh
    如脚本不存在则返回 fallback 命令；非 Ascend 引擎返回空列表。

    Args:
        config_dir: 配置目录路径
        engine:     引擎类型

    Returns:
        List[str]: 内联后的 shell 命令列表
    """
    script_map = {
        "vllm_ascend": "set_vllm_ascend_env.sh",
        "mindie": "set_mindie_env.sh",
    }
    script_name = script_map.get(engine)
    if not script_name:
        return []

    script_path = os.path.join(config_dir, script_name)
    if os.path.exists(script_path):
        return _read_and_inline_script(script_path, engine)

    logger.warning("Env script %s not found, using fallback for %s", script_path, engine)
    return _build_ascend_fallback_env(engine)


def _read_and_inline_script(script_path: str, engine: str) -> List[str]:
    """读取脚本文件内容并转为内联 shell 命令，附加驱动预检查。

    Args:
        script_path: 脚本文件完整路径
        engine:      引擎类型

    Returns:
        List[str]: 内联后的 shell 命令列表
    """
    commands = []
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n\r")
            if stripped.startswith("#!"):
                continue
            commands.append(stripped)
    logger.info("Inlined env script %s for engine %s (%d lines)",
                script_path, engine, len(commands))

    if engine == "vllm_ascend":
        commands.extend(_build_ascend_driver_check())
    return commands


def _build_ascend_driver_check() -> List[str]:
    """生成 Ascend 驱动预检查 shell 命令，驱动缺失时 exit 1。

    Returns:
        List[str]: 预检查 shell 命令列表
    """
    return [
        "# Pre-flight: verify Ascend driver is accessible",
        "if [ ! -f /usr/local/Ascend/driver/lib64/driver/libascend_hal.so ]; then",
        "    echo 'FATAL: libascend_hal.so not found at "
        "/usr/local/Ascend/driver/lib64/driver/'",
        "    echo 'HINT: Ensure the host Ascend driver is mounted "
        "into the container (hostPath: /usr/local/Ascend/driver)'",
        "    exit 1",
        "fi",
    ]


def _build_ascend_fallback_env(engine: str) -> List[str]:
    """在环境脚本缺失时生成 Ascend fallback 环境命令。

    Args:
        engine: 引擎类型

    Returns:
        List[str]: fallback 环境命令列表
    """
    if engine not in ("vllm_ascend", "mindie"):
        return []
    return [
        "export LD_LIBRARY_PATH=\"/usr/local/Ascend/driver/lib64/driver"
        ":/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}\"",
    ]


def _build_vllm_ascend_extensions(params) -> List[str]:
    """生成 vllm_ascend 扩展环境命令（昆仑 ATB、Qwen3Next 支持）。

    Args:
        params: 参数字典

    Returns:
        List[str]: 扩展环境命令列表
    """
    commands = []
    if params.get("engine_config", {}).get("use_kunlun_atb"):
        commands.append("export USE_KUNLUN_ATB=1")
        logger.info("kunlun atb is used")
    return commands


def _is_vllm_ascend_ray_distributed(params: Dict[str, Any], engine: str) -> bool:
    """判断当前是否为 vllm_ascend Ray 分布式执行场景。"""
    if engine != "vllm_ascend":
        return False
    if not params.get("distributed"):
        return False
    return params.get("distributed_executor_backend", "ray") == "ray"


def _filter_vllm_ascend_ray_incompatible_env(
    commands: List[str],
    params: Dict[str, Any],
    engine: str,
) -> List[str]:
    """过滤 vllm_ascend Ray 分布式不适用的环境变量。"""
    if not _is_vllm_ascend_ray_distributed(params, engine):
        return commands
    return [
        command for command in commands
        if command.strip() != "export HCCL_OP_EXPANSION_MODE=AIV"
    ]


def _filter_pd_incompatible_env(commands: List[str]) -> List[str]:
    """PD 分离（kv_producer/consumer）下剔除与之互斥的环境变量。

    vLLM-Ascend(≥0.20.2) 的 ``enable_balance_scheduling``（由 ``VLLM_ASCEND_BALANCE_SCHEDULING``
    驱动）仅 PD-mixed / 非 PD 可用；PD 分离下引擎会以 ValidationError 拒绝启动
    （"enable_balance_scheduling ... not supported in PD-disaggregated mode"）。
    各模型 env builder（GLM5/Kimi/MiniMax 等）会无条件注入该 flag，故在 env 汇总层
    按 PD 角色统一剔除——覆盖全模型 / 全启动路径（单机 / 分布式 / PD external-lb）。
    """
    if not get_pd_role_env():
        return commands
    kept = [c for c in commands if not c.strip().startswith("export VLLM_ASCEND_BALANCE_SCHEDULING=")]
    if len(kept) != len(commands):
        logger.info(
            "[PD] kv_role=producer/consumer 与 enable_balance_scheduling 互斥 → "
            "已剔除 VLLM_ASCEND_BALANCE_SCHEDULING（否则 vLLM-Ascend≥0.20.2 会 ValidationError）")
    return kept


def _build_vllm_ascend_forced_env_commands(params: Dict[str, Any], engine: str) -> List[str]:
    """构建 vllm_ascend 强制生效的通用环境变量。"""
    if engine != "vllm_ascend":
        return []

    commands = [
        "export OMP_PROC_BIND=${OMP_PROC_BIND:-false}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
        "export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}",
        "export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}",
    ]
    # RoCE 互联 + GLM5.1 分布式场景不注入 HCCL_OP_EXPANSION_MODE
    is_roce_glm51 = (
        params.get("distributed")
        and _get_deepseek_ascend_dp_model_architecture(params) == "GlmMoeDsaForCausalLM"
        and _is_roce_distributed()
    )
    if not _is_vllm_ascend_ray_distributed(params, engine) and not is_roce_glm51:
        commands.append("export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}")
    return commands


def _build_base_env_commands(params, engine: str, root: str) -> List[str]:
    """构建基础环境变量设置命令列表。

    仅 Ascend 引擎（vllm_ascend / mindie）需要环境初始化脚本，
    NVIDIA 引擎（vllm / sglang）无需额外设置。

    Args:
        params: 参数字典
        engine: 引擎类型 ('vllm', 'vllm_ascend', 'sglang', 'mindie')
        root:   项目根目录路径

    Returns:
        List[str]: shell 命令列表
    """
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
    )
    env_commands = _inline_ascend_env_script(config_dir, engine)
    if engine == "vllm_ascend":
        env_commands.extend(_build_vllm_ascend_extensions(params))
    return env_commands


# ── LMCache YAML 配置文件 ─────────────────────────────────────────────
# 当 cold_start 或 QAT 特性启用时，需要生成 YAML 配置文件供 LMCache 读取。
# 纯内存卸载场景不需要 YAML 文件（环境变量即可控制）。
_LMCACHE_CONFIG_FILENAME = "lmcache_config.yaml"
_LMCACHE_SHARED_VOLUME = os.getenv("SHARED_VOLUME_PATH", "/shared-volume")


def _build_lmcache_yaml_dict(
    engine: str,
    max_cpu_size: Optional[str] = None,
    local_cpu_enabled: Optional[bool] = None,
) -> dict:
    """根据环境变量构建 LMCache 的 YAML 配置字典。

    配置结构参考 LMCache 官方 YAML schema，包含以下可选段：
    - chunk_size: KV 缓存分块大小（默认 256）
    - local_cpu:  CPU 内存缓存配置
    - local_disk: 本地磁盘缓存配置
    - pre_caching: 冷启动预热配置（仅 cold_start 启用）
    - qat:         QAT 硬件压缩配置（仅 QAT 启用）

    Args:
        engine: 引擎类型（vllm / vllm_ascend）

    Returns:
        dict: 可被 yaml.dump() 序列化的配置字典
    """
    config: dict = {}

    # ── L2 门控（需求一 §A.1）──
    mem_enabled = (
        local_cpu_enabled
        if local_cpu_enabled is not None
        else os.getenv("ENABLE_KV_MEM_OFFLOAD", "false").strip().lower() == "true"
    )
    disk_enabled = os.getenv("ENABLE_KV_DISK_OFFLOAD", "false").strip().lower() == "true"

    # ── chunk_size ──
    chunk_size_str = os.getenv("LMCACHE_CHUNK_SIZE", "256")
    try:
        config["chunk_size"] = int(chunk_size_str)
    except (ValueError, TypeError):
        config["chunk_size"] = 256

    # ── local_cpu（仅在 L2 mem=true 时处理）──
    if mem_enabled:
        max_cpu_size = (max_cpu_size if max_cpu_size is not None else os.getenv("KV_MEM_OFFLOAD_SIZE", "")).strip()
        config["local_cpu"] = True
        if max_cpu_size and max_cpu_size.lower() != "auto":
            try:
                config["max_local_cpu_size"] = float(max_cpu_size)
            except (ValueError, TypeError):
                logger.warning("[LMCache YAML] Invalid KV_MEM_OFFLOAD_SIZE=%r, skip", max_cpu_size)

    # ── local_disk（仅在 L2 disk=true 时处理）──
    if disk_enabled:
        local_disk_path = os.getenv("KV_DISK_OFFLOAD_PATH", "").strip()
        if local_disk_path:
            config["local_disk"] = local_disk_path

        max_disk_size = os.getenv("KV_DISK_OFFLOAD_SIZE", "").strip()
        if max_disk_size:
            try:
                config["max_local_disk_size"] = float(max_disk_size)
            except (ValueError, TypeError):
                logger.warning("[LMCache YAML] Invalid KV_DISK_OFFLOAD_SIZE=%r, skip", max_disk_size)

    # ── pre_caching（冷启动预热）──
    if get_cold_start_env():
        pre_caching: dict = {
            "hash_algorithm": os.getenv("LMCACHE_PRE_CACHING_HASH", "sha256_cbor"),
            "manifest_write_interval": int(os.getenv("LMCACHE_MANIFEST_WRITE_INTERVAL", "1")),
            "maintenance": {"enabled": False},
            "full_sync": {"enabled": False},
        }
        config["pre_caching"] = pre_caching
        logger.info("[LMCache YAML] Cold-start pre_caching section enabled")

    # ── qat（QAT 硬件压缩，L3 门控已在 get_qat_env() 内）──
    if get_qat_env():
        qat_module = "kv_agent" if engine == "vllm" else os.getenv("LMCACHE_QAT_MODULE", "kv_agent")
        qat_section: dict = {
            "module_name": qat_module,
            "instance_num": int(os.getenv("KV_QAT_INSTANCE_NUM", "2")),
            "loss_level": int(os.getenv("KV_QAT_COMPRESS_LEVEL", "0")),
            "log_enabled": int(os.getenv("LMCACHE_QAT_LOG_ENABLED", "0")),
        }
        config["qat"] = qat_section
        logger.info("[LMCache YAML] QAT section enabled (module=%s)", qat_module)

    return config


def _need_lmcache_config_yaml(local_cpu_enabled: Optional[bool] = None) -> bool:
    """判断是否需要生成 LMCache YAML 配置文件。

    触发条件（任一满足即生成）：
      1. cold_start 或 QAT 特性启用（功能性段落）
      2. L2 内存卸载开关 ENABLE_KV_MEM_OFFLOAD=true
      3. L2 磁盘卸载开关 ENABLE_KV_DISK_OFFLOAD=true

    说明：LMCache 的容量类字段（max_size 等）在多数版本下仅识别
    YAML 文件，不保证从同名 env 自动注入；因此只要 L2 开关为 true，
    就必须落盘 YAML，否则会沉默失效（参数丢失 bug）。

    Returns:
        bool: 需要生成返回 True
    """
    if get_cold_start_env() or get_qat_env():
        return True
    if (
        local_cpu_enabled is not False
        and os.getenv("ENABLE_KV_MEM_OFFLOAD", "false").strip().lower() == "true"
    ):
        return True
    if os.getenv("ENABLE_KV_DISK_OFFLOAD", "false").strip().lower() == "true":
        return True
    return False


def _write_lmcache_config_yaml(
    engine: str,
    max_cpu_size: Optional[str] = None,
    local_cpu_enabled: Optional[bool] = None,
) -> Optional[str]:
    """生成并写入 LMCache YAML 配置文件到共享卷。

    条件：见 _need_lmcache_config_yaml()，覆盖 cold_start / QAT /
    CPU 卸载 / 磁盘卸载 等所有需要落盘的场景。
    写入路径：/shared-volume/lmcache_config.yaml

    Args:
        engine: 引擎类型

    Returns:
        str | None: 写入成功返回文件路径，无需写入或失败返回 None
    """
    if not _need_lmcache_config_yaml(local_cpu_enabled=local_cpu_enabled):
        return None

    config = _build_lmcache_yaml_dict(
        engine,
        max_cpu_size=max_cpu_size,
        local_cpu_enabled=local_cpu_enabled,
    )
    yaml_content = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)

    file_path = os.path.join(_LMCACHE_SHARED_VOLUME, _LMCACHE_CONFIG_FILENAME)
    os.makedirs(_LMCACHE_SHARED_VOLUME, exist_ok=True)

    ok = safe_write_file(
        file_path, yaml_content, is_json=False,
        options=WriteOptions(
            modes=stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
            atomic=True,
        ),
    )
    if ok:
        logger.info("[LMCache YAML] Config written to %s", file_path)
        return file_path
    else:
        logger.error("[LMCache YAML] Failed to write config to %s", file_path)
        return None


def _append_lmcache_env_export(env_commands: List[str], name: str, value: Optional[str] = None) -> None:
    """Append an explicit engine-side export for an LMCache environment variable."""
    if value is None:
        value = os.getenv(name, "").strip()
    if value:
        env_commands.append(f"export {name}={shlex.quote(value)}")


def _resolve_lmcache_lookup_server_worker_ids(params: Optional[Dict[str, Any]]) -> str:
    """Resolve LMCache lookup workers from the final DeepSeek-V4-Flash TP size."""
    explicit = os.getenv("LMCACHE_LOOKUP_SERVER_WORKER_IDS", "").strip()
    if explicit:
        return explicit

    tp_size = _safe_int((params or {}).get("tensor_parallel_size"))
    if not tp_size and params:
        platform = _ascend_platform_from_runtime(params)
        tp_size = _default_deepseek_v4_flash_tensor_parallel_size(platform)
    if not tp_size or tp_size < 1:
        tp_size = 4
    return ",".join(str(index) for index in range(tp_size))


def _is_glm51_nvidia_vllm_params(params: Optional[Dict[str, Any]], engine: str,
                                 model_info: Optional[ModelIdentifier] = None) -> bool:
    """Return True when current params describe GLM-5.1 on NVIDIA vLLM."""
    if engine != "vllm" or not params:
        return False
    try:
        info = model_info or ModelIdentifier(
            params.get("model_name"),
            params.get("model_path"),
            params.get("model_type"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[GLM-5.1 NV] Skip variant detection, ModelIdentifier failed: %s", exc)
        return False
    return is_glm_moe_dsa_glm51(
        info,
        model_name=params.get("model_name"),
        model_path=params.get("model_path"),
    )


_NATIVE_OFFLOAD_SIZE_ENV_NAMES = ("KV_MEM_OFFLOAD_SIZE",)
# Qwen3.5 NVFP4 是 23.6.0 既有 native KV offload 场景，历史上允许
# LMCACHE_MAX_LOCAL_CPU_SIZE 接管 native size。这个兼容策略属于代码侧
# 场景逻辑，不写入 smart_feature_whitelist.json；白名单只保留
# backend=native，避免把会随页面/env 变化的容量来源放进规则表。
_QWEN35_NVFP4_NATIVE_SIZE_ENV_NAMES = (
    "KV_MEM_OFFLOAD_SIZE",
    "LMCACHE_MAX_LOCAL_CPU_SIZE",
)


def _native_backend_size_env_names(params: Dict[str, Any], engine: str) -> Tuple[str, ...]:
    """返回 native offload size 可读取的 env 名称。

    设计约束：
      1. 白名单只决定当前场景是否允许 native backend；
      2. size 来源必须由页面/env 或代码侧场景兼容逻辑决定；
      3. 新增模型适配时不能再往白名单塞 ``size_*`` 字段。

    因此这里按模型场景选择 env 列表，而不是读取白名单行里的动态配置。
    """
    if engine == "vllm" and is_qwen3_5_397b_nvfp4_vllm(params, engine):
        # 兼容 23.6.0 Qwen3.5 NVFP4 既有管理口径：优先页面 KV_MEM_OFFLOAD_SIZE，
        # 旧 LMCACHE_MAX_LOCAL_CPU_SIZE 仍可接管，但该策略归代码场景逻辑所有，
        # 不放进 smart_feature_whitelist.json。
        return _QWEN35_NVFP4_NATIVE_SIZE_ENV_NAMES
    return _NATIVE_OFFLOAD_SIZE_ENV_NAMES


def _resolve_native_backend_offload_gb(params: Dict[str, Any], engine: str) -> int:
    """解析 native backend 的节点级 offload 容量。

    这里复用已有 ``_resolve_native_offload_gb``，保持页面输入、auto 公式、
    floor 熔断和日志口径一致。白名单行只提供 ``backend=native``，不提供
    fallback、env 名称或是否要求页面开关；这些属于运行时策略。

    当前两类口径：
      - Qwen3.5 NVFP4：兼容旧 LMCACHE_MAX_LOCAL_CPU_SIZE，未填时保留 200G fallback；
      - Day0 H20/L20 native：必须由页面/env 给出 KV_MEM_OFFLOAD_SIZE，缺失/非法即丢弃。
    """
    if engine == "vllm" and is_qwen3_5_397b_nvfp4_vllm(params, engine):
        return _resolve_native_offload_gb(
            params,
            fallback_gb=200,
            size_env_names=_QWEN35_NVFP4_NATIVE_SIZE_ENV_NAMES,
            log_context="KVCache Offload",
        )
    return _resolve_native_offload_gb(
        params,
        fallback_gb=None,
        size_env_names=_NATIVE_OFFLOAD_SIZE_ENV_NAMES,
        require_mem_switch=True,
        log_context="KVCache Offload",
    )


def _native_backend_auto_requested(params: Dict[str, Any], engine: str) -> bool:
    """判断 native backend 是否请求 auto 容量。

    auto 判定必须跟当前场景实际读取的 env 列表一致，这样 CLI、variant 和
    `/v1/startup/accel` 中的 floor_disabled 状态才能保持同源。
    """
    return any(
        os.getenv(env_name, "").strip().lower() == "auto"
        for env_name in _native_backend_size_env_names(params, engine)
    )


def _mtp_whitelist_override_row(params: Dict[str, Any], engine: str) -> Optional[dict]:
    """Return a spec whitelist row that explicitly owns MTP method selection."""
    row = resolve_feature_whitelist_row_from_params(params, engine, "spec")
    if not row or not row.get("mtp_method"):
        return None
    return row


# ── Offload 特例/后端判定（_build_cache_env_commands 与 resolve_offload_variant 共用，
#    使「守卫条件」成为单一真相源、二者天然同源同序）──
_OFFLOAD_NATIVE_NONE = ""                            # 无特例 → 走 LMCache 通用路径
_OFFLOAD_GLM51_NV_DISABLED = "glm51_nv_disabled"     # GLM-5.1·NV 强制关
_OFFLOAD_V4_FLASH_NATIVE = "v4_flash_native"         # V4-Flash·NV native --kv-offloading-backend
_OFFLOAD_NATIVE_BACKEND_VARIANT = "native_kv_offloading_backend"
KV_MEM_OFFLOAD_AUTO_FLOOR_DISABLED_ENV = "KV_MEM_OFFLOAD_AUTO_FLOOR_DISABLED"
_OFFLOAD_AUTO_FLOOR_VARIANT = "lmcache_cpu+auto+floor_disabled"
_OFFLOAD_CPU_AUTO_FLOOR_MODIFIER = "cpu_auto_floor_disabled"
_OFFLOAD_VARIANT_BY_SPECIAL = {                      # 特例 → resolve_offload_variant 的 variant 串
    _OFFLOAD_GLM51_NV_DISABLED: "disabled",
    _OFFLOAD_V4_FLASH_NATIVE: _OFFLOAD_NATIVE_BACKEND_VARIANT,
}


def _classify_offload_special_case(params: Optional[Dict[str, Any]], engine: str) -> str:
    """归类 offload 特例守卫（条件与原 _build_cache_env_commands 守卫逐字同序）。

    返回 ``_OFFLOAD_*`` 之一；``_OFFLOAD_NATIVE_NONE`` 表示无特例、应走 LMCache 通用路径。
    三特例均与 LMCache 互斥（强制关 / V4 走 native 后端）。
    """
    if resolve_offload_whitelist_backend(params, engine) == "native":
        return _OFFLOAD_NATIVE_NONE
    if _is_glm51_nvidia_vllm_params(params, engine):
        return _OFFLOAD_GLM51_NV_DISABLED
    if params and engine == "vllm" and _is_deepseek_v4_flash_params(params):
        return _OFFLOAD_V4_FLASH_NATIVE
    return _OFFLOAD_NATIVE_NONE


def _lmcache_engine_env_skip(special: str) -> bool:
    """三类 offload 特例下打印「跳过 LMCache engine 侧 env 导出」日志并返回 True。

    GLM-5.1·NV 强制关；V4-Flash·NV 走 native ``--kv-offloading-backend``。
    非特例返回 False（继续走 LMCache 路径）。
    """
    if special == _OFFLOAD_GLM51_NV_DISABLED:
        logger.warning(
            "[KVCache Offload] Forced disabled for GLM-5.1 on NVIDIA/vLLM; "
            "skipping LMCache engine-side env exports despite ENABLE_KV_OFFLOAD=true."
        )
        return True
    if special == _OFFLOAD_V4_FLASH_NATIVE:
        # [V4-Flash-NV-Day0] NV V4-Flash 用 native --kv-offloading-backend（见 _build_kv_offload_cmd）。
        logger.info(
            "[KVCache Offload] DeepSeek-V4-Flash (NV) uses native --kv-offloading-backend; "
            "skipping LMCache engine-side env exports."
        )
        return True
    return False


def _resolve_lmcache_cpu_env(params: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """解析 LMCache CPU 池两 env 值 ``(ENABLE_KV_MEM_OFFLOAD, KV_MEM_OFFLOAD_SIZE)``。

    auto 模式（``resolve_offload_cpu_capacity_gb`` 非 None）反向预算并写回「均卡」容量；
    熔断(<=0)清空 CPU 池；非 auto 透传现有 env。需求一 §3.0。
    """
    mem_enabled = os.getenv("ENABLE_KV_MEM_OFFLOAD", "false").strip().lower() == "true"
    if not mem_enabled:
        return "", ""
    local_cpu_value = os.getenv("ENABLE_KV_MEM_OFFLOAD", "").strip()
    max_cpu_size = os.getenv("KV_MEM_OFFLOAD_SIZE", "").strip()

    auto_total = resolve_offload_cpu_capacity_gb(params)
    if auto_total is None:
        if max_cpu_size and max_cpu_size.lower() != "auto":
            try:
                n_card = _safe_int(params.get("device_count")) if params else 1
                n_card = n_card or 1
                per_card = max(1, int(max_cpu_size) // n_card)
                logger.info(
                    "[KVCache Offload] custom CPU per-card = M_offload(%sG) / N_card(%d) = %dG "
                    "(KV_MEM_OFFLOAD_SIZE).",
                    max_cpu_size, n_card, per_card,
                )
                return local_cpu_value, str(per_card)
            except ValueError:
                logger.warning("[KVCache Offload] Invalid KV_MEM_OFFLOAD_SIZE=%r; using raw value.", max_cpu_size)
        return local_cpu_value, max_cpu_size
    if auto_total <= 0:
        # 熔断：容量低于下限 → 不建 CPU 卸载池（offload 退化为无 CPU 池）。
        logger.warning(
            "[KVCache Offload] auto CPU capacity below floor %dG -> skip CPU offload pool.",
            _OFFLOAD_MIN_GB,
        )
        return "", ""
    n_card = _safe_int(params.get("device_count")) or 1
    per_card = max(1, auto_total // n_card)
    local_cpu_value = local_cpu_value or "true"
    logger.info(
        "[KVCache Offload] auto CPU per-card = M_offload(%dG) / N_card(%d) = %dG "
        "(KV_MEM_OFFLOAD_SIZE).",
        auto_total, n_card, per_card,
    )
    return local_cpu_value, str(per_card)


def _is_lmcache_auto_cpu_floor_disabled(
    params: Optional[Dict[str, Any]],
    local_cpu_value: str,
    max_cpu_size: str,
) -> bool:
    """Return True when auto CPU capacity is below the minimum and must stay disabled."""
    if local_cpu_value or max_cpu_size:
        return False
    return is_kv_mem_offload_auto_floor_disabled(params)


def is_kv_mem_offload_auto_floor_disabled(params: Optional[Dict[str, Any]]) -> bool:
    """Return True when KV memory offload auto mode is fused by the 100G floor."""
    if os.getenv("ENABLE_KV_MEM_OFFLOAD", "false").strip().lower() != "true":
        return False
    if os.getenv("KV_MEM_OFFLOAD_SIZE", "").strip().lower() != "auto":
        return False
    return resolve_offload_cpu_capacity_gb(params or {}) == 0


def lmcache_auto_floor_disables_all_backends(params: Optional[Dict[str, Any]]) -> bool:
    """True when auto memory floor leaves no LMCache backend that needs a patch."""
    if not is_kv_mem_offload_auto_floor_disabled(params):
        return False
    if os.getenv("ENABLE_KV_DISK_OFFLOAD", "false").strip().lower() == "true":
        return False
    if get_qat_env() or get_cold_start_env():
        return False
    return True


def _is_offload_feature_effective(params: Optional[Dict[str, Any]], engine: str) -> bool:
    """Return whether offload survived the smart-feature gate."""
    if not params:
        return False
    smart_feats = params.get("_smart_feats")
    if smart_feats is not None:
        return "offload" in smart_feats
    return feature_allowed(
        engine,
        params.get("model_name"),
        params.get("model_path"),
        params.get("_smart_card_token") or resolve_card_token(),
        "offload",
    )


def _resolve_offload_backend(params: Optional[Dict[str, Any]], engine: str = "") -> Tuple[str, str]:
    """LMCache 后端选择，返回 ``(backend, cpu_mode)``。

    backend ∈ {native_kv_offloading_backend, lmcache_cpu_disk, lmcache_cpu, lmcache_disk, disabled}；
    cpu_mode ∈ {auto, custom, ""}（auto 即反向预算容量模式）。
    """
    if resolve_offload_whitelist_backend(params, engine) == "native":
        return _OFFLOAD_NATIVE_BACKEND_VARIANT, "native"
    auto_total = resolve_offload_cpu_capacity_gb(params)
    if auto_total is not None:
        has_cpu = auto_total > 0          # 熔断(0) → 无 CPU 池
        cpu_mode = "auto"
    else:
        has_cpu = os.getenv("ENABLE_KV_MEM_OFFLOAD", "false").strip().lower() == "true"
        cpu_mode = "custom" if has_cpu else ""
    has_disk = os.getenv("ENABLE_KV_DISK_OFFLOAD", "false").strip().lower() == "true"
    if has_cpu and has_disk:
        backend = "lmcache_cpu_disk"
    elif has_cpu:
        backend = "lmcache_cpu"
    elif has_disk:
        backend = "lmcache_disk"
    else:
        backend = "disabled"
    return backend, cpu_mode


def _build_deepseek_v4_flash_lmcache_env_commands(params: Optional[Dict[str, Any]]) -> List[str]:
    """Build vLLM-Ascend 0.21 LMCache dynamic offload env for DeepSeek-V4-Flash."""
    env_commands = ["export PYTHONHASHSEED=0"]
    offload_enabled = os.getenv("ENABLE_KV_OFFLOAD", "false").strip().lower() == "true"
    mem_offload_raw = os.getenv("ENABLE_KV_MEM_OFFLOAD")
    mem_enabled = (mem_offload_raw or "").strip().lower() == "true"
    mem_explicitly_disabled = (
        mem_offload_raw is not None
        and mem_offload_raw.strip().lower() in {"false", "0", "no", "off"}
    )
    auto_requested = os.getenv("KV_MEM_OFFLOAD_SIZE", "").strip().lower() == "auto"
    kv_mem_size_set = bool(os.getenv("KV_MEM_OFFLOAD_SIZE", "").strip())
    kv_mem_size_authoritative = mem_enabled and kv_mem_size_set
    default_cpu_pool = offload_enabled and not mem_explicitly_disabled and not auto_requested
    resolved_local_cpu, resolved_max_cpu_size = _resolve_lmcache_cpu_env(params)
    auto_cpu_floor_disabled = _is_lmcache_auto_cpu_floor_disabled(
        params,
        resolved_local_cpu,
        resolved_max_cpu_size,
    )

    local_cpu = os.getenv("LMCACHE_LOCAL_CPU", "").strip()
    if auto_cpu_floor_disabled:
        local_cpu = ""
    elif not local_cpu:
        if resolved_local_cpu:
            local_cpu = "True" if resolved_local_cpu.lower() == "true" else resolved_local_cpu
        elif mem_enabled or default_cpu_pool:
            local_cpu = "True"

    if kv_mem_size_authoritative:
        max_cpu_size = resolved_max_cpu_size
    else:
        max_cpu_size = os.getenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "").strip()
        if not max_cpu_size:
            max_cpu_size = resolved_max_cpu_size
    if not max_cpu_size and (mem_enabled or default_cpu_pool) and not kv_mem_size_authoritative:
        max_cpu_size = "40"

    _append_lmcache_env_export(env_commands, "LMCACHE_TRACK_USAGE", os.getenv("LMCACHE_TRACK_USAGE", "false"))
    if auto_cpu_floor_disabled:
        _append_lmcache_env_export(env_commands, KV_MEM_OFFLOAD_AUTO_FLOOR_DISABLED_ENV, "true")
    _append_lmcache_env_export(env_commands, "LMCACHE_LOCAL_CPU", local_cpu)
    _append_lmcache_env_export(env_commands, "LMCACHE_MAX_LOCAL_CPU_SIZE", max_cpu_size)
    _append_lmcache_env_export(env_commands, "LMCACHE_LOG_LEVEL", os.getenv("LMCACHE_LOG_LEVEL", "INFO"))
    _append_lmcache_env_export(env_commands, "LMCACHE_USE_LAYERWISE", os.getenv("LMCACHE_USE_LAYERWISE", "False"))
    _append_lmcache_env_export(env_commands, "LMCACHE_NUMA_MODE", os.getenv("LMCACHE_NUMA_MODE", "auto"))
    _append_lmcache_env_export(env_commands, "LMCACHE_CHUNK_SIZE", os.getenv("LMCACHE_CHUNK_SIZE", "1024"))
    _append_lmcache_env_export(
        env_commands,
        "LMCACHE_EXTRA_CONFIG",
        os.getenv("LMCACHE_EXTRA_CONFIG", '{"save_only_first_rank": false}'),
    )
    _append_lmcache_env_export(
        env_commands,
        "LMCACHE_LOOKUP_SERVER_WORKER_IDS",
        _resolve_lmcache_lookup_server_worker_ids(params),
    )
    return env_commands


def _build_cache_env_commands(engine: str, params: Optional[Dict[str, Any]] = None) -> List[str]:
    """构建 KVCache Offload 特性的环境变量设置命令。

    KVCache Offload 允许将 KV 缓存卸载到主机内存或远端存储。
    LMCache 所需的共享库已在 accel-volume 安装阶段通过平台对应的
    ``install.py`` 安装命令注入，无需再手动设置 LD_LIBRARY_PATH。

    只要传入了任何容量/路径类配置（CPU/Disk/cold_start/QAT），就会
    生成 LMCache YAML 配置文件并通过 ``LMCACHE_CONFIG_FILE`` 环境变量
    告知 LMCache，避免容量参数被沉默丢弃。

    Args:
        engine: 引擎类型

    Returns:
        List[str]: 环境变量设置命令列表，未启用时返回空列表

    环境变量:
        - ENABLE_KV_OFFLOAD: 是否启用 KVCache Offload (true/false)
        - LMCACHE_CONFIG_FILE: LMCache YAML 配置文件路径（自动生成）
    """
    env_commands = []
    if not get_lmcache_env():
        return env_commands
    

    # ── Smart 白名单二次守卫（与 config_loader._set_kv_cache_config 对齐）──
    # apply_effective_feature_enablement 通过 os.environ 收口，
    # 但 os.environ 可能在中间层被重置；此处直接查询白名单确保不被绕过。
    # 优先复用 C14 收口的 _smart_feats stash；stash 缺失时回退 feature_allowed()。
    _smart_feats = params.get("_smart_feats")
    if _smart_feats is not None:
        _offload_ok = "offload" in _smart_feats
    else:
        _offload_ok = feature_allowed(
            params.get("engine", ""), params.get("model_name"),
            params.get("model_path"), params.get("_smart_card_token") or resolve_card_token(), "offload",
        )
    if not _offload_ok:
        logger.info(
            "[SmartFeature] offload suppressed by whitelist in "
            "_build_cache_env_commands — skipping LMCache env exports."
        )
        return env_commands

    # 守卫（条件由 _classify_offload_special_case 统一裁定；互斥特例跳过 env 导出）
    special = _classify_offload_special_case(params, engine)
    if memcache_hybrid.is_memcache_hybrid_params(params, engine):
        logger.info("[MemCache] Model uses MemCache; skipping LMCache env exports.")
        return env_commands

    if _lmcache_engine_env_skip(special):
        return env_commands
    backend, _ = _resolve_offload_backend(params, engine)
    if backend == _OFFLOAD_NATIVE_BACKEND_VARIANT:
        logger.info(
            "[KVCache Offload] native KV offload backend selected; "
            "skipping LMCache engine-side env exports."
        )
        return env_commands

    # 跨实例Hash一致
    if params and engine == "vllm_ascend" and _is_deepseek_v4_flash_params(params):
        return _build_deepseek_v4_flash_lmcache_env_commands(params)

    env_commands.append('export PYTHONHASHSEED=0')
    _append_lmcache_env_export(env_commands, "ENABLE_KV_OFFLOAD", "true")
    _append_lmcache_env_export(env_commands, "LMCACHE_CHUNK_SIZE")

    # C4：auto 模式反向预算并写回「均卡」CPU 容量（LMCache 每 rank 一池，需 per-card）。需求一 §3.0。
    local_cpu_value, max_cpu_size = _resolve_lmcache_cpu_env(params)
    auto_cpu_floor_disabled = _is_lmcache_auto_cpu_floor_disabled(
        params,
        local_cpu_value,
        max_cpu_size,
    )
    if local_cpu_value or max_cpu_size:
        _append_lmcache_env_export(env_commands, "ENABLE_KV_MEM_OFFLOAD", local_cpu_value or "true")
        _append_lmcache_env_export(env_commands, "KV_MEM_OFFLOAD_SIZE", max_cpu_size)
    elif auto_cpu_floor_disabled:
        _append_lmcache_env_export(env_commands, KV_MEM_OFFLOAD_AUTO_FLOOR_DISABLED_ENV, "true")

    if os.getenv("ENABLE_KV_DISK_OFFLOAD", "false").strip().lower() == "true":
        _append_lmcache_env_export(env_commands, "ENABLE_KV_DISK_OFFLOAD", "true")
        _append_lmcache_env_export(env_commands, "KV_DISK_OFFLOAD_PATH")
        _append_lmcache_env_export(env_commands, "KV_DISK_OFFLOAD_SIZE")

    # 任何 LMCache 容量/功能段配置都会触发 YAML 生成并导出路径
    yaml_path = _write_lmcache_config_yaml(
        engine,
        max_cpu_size=max_cpu_size,
        local_cpu_enabled=False if auto_cpu_floor_disabled else None,
    )
    if yaml_path:
        env_commands.append(f'export LMCACHE_CONFIG_FILE={shlex.quote(yaml_path)}')
        logger.info("[KVCache Offload] LMCACHE_CONFIG_FILE exported -> %s", yaml_path)
    else:
        logger.warning(
            "[KVCache Offload] ENABLE_KV_OFFLOAD enabled but no LMCache config "
            "yaml generated. Capacity envs (KV_MEM_OFFLOAD_SIZE / "
            "KV_DISK_OFFLOAD_SIZE) may not take effect. "
            "Set ENABLE_KV_MEM_OFFLOAD=true (or any capacity env) to enable."
        )

    return env_commands


def resolve_offload_variant(params: Optional[Dict[str, Any]], engine: str) -> str:
    """纯函数：返回卸载 variant「后端[+模式][+修饰]」（advanced_features.json 监控用，无副作用）。

    镜像 ``_build_cache_env_commands`` 的后端选择 + C4 容量模式（需求一 §4.3）。
    ⚠ 与 ``_build_cache_env_commands`` 同源同序：守卫共用 ``_classify_offload_special_case``，
    新增/调整特例守卫只需改该分类器一处即可两侧同步。
    监控在写 JSON 时（早于产出口在脚本生成阶段运行）调用，故此处独立按 merged/env 推导，
    不依赖产出口先跑，也绝不改 engine_config。

    variant 形态：后端 ``[+auto|+custom][+qat][+cold_start]``，如
    ``lmcache_cpu+auto`` / ``lmcache_cpu+custom+qat`` / ``native_kv_offloading_backend`` / ``disabled``。
    """
    if not get_lmcache_env():
        return ""
    if memcache_hybrid.is_memcache_hybrid_params(params, engine):
        return (
            memcache_hybrid.MEMCACHE_OFFLOAD_VARIANT
            if memcache_hybrid.resolve_memcache_dram_gb(params)
            else "disabled"
        )
    # 守卫（与 _build_cache_env_commands 共用 _classify_offload_special_case，天然同序）：
    # GLM-5.1·NV 强制关 → disabled；V4 → native 后端。
    special = _classify_offload_special_case(params, engine)
    if special:
        variant = _OFFLOAD_VARIANT_BY_SPECIAL[special]
        if special == _OFFLOAD_V4_FLASH_NATIVE and _native_offload_auto_floor_disabled(params, special):
            return f"{variant}+auto+floor_disabled"
        return variant
    # LMCache 路径：CPU(auto/custom) / Disk / 分层（后端选择见 _resolve_offload_backend）
    backend, cpu_mode = _resolve_offload_backend(params, engine)
    if backend == _OFFLOAD_NATIVE_BACKEND_VARIANT:
        size_gb = _resolve_native_backend_offload_gb(params or {}, engine)
        if size_gb <= 0:
            if _native_backend_auto_requested(params or {}, engine):
                return f"{backend}+auto+floor_disabled"
            return "disabled"
        return backend
    auto_cpu_floor_disabled = is_kv_mem_offload_auto_floor_disabled(params)
    if backend == "disabled":            # offload on 但无任何容量段（含熔断）
        if auto_cpu_floor_disabled:
            return _OFFLOAD_AUTO_FLOOR_VARIANT
        return "disabled"
    variant = backend
    if cpu_mode and backend in ("lmcache_cpu", "lmcache_cpu_disk"):
        variant += "+" + cpu_mode
    if auto_cpu_floor_disabled:
        variant += "+" + _OFFLOAD_CPU_AUTO_FLOOR_MODIFIER
    if get_qat_env():
        variant += "+qat"
    if get_cold_start_env():
        variant += "+cold_start"
    return variant


def _parse_kv_mem_size_gb(value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        size_gb = int(raw)
    except (TypeError, ValueError):
        logger.warning("[KVCache Offload] Invalid resolved KV memory size=%r; reporting null.", raw)
        return None
    return size_gb if size_gb >= 0 else None


def resolve_effective_kv_mem_offload_size(
    params: Optional[Dict[str, Any]],
    engine: str,
    variant: Optional[str] = None,
) -> Optional[int]:
    """Return the effective KV memory offload size in GB for status reporting.

    The returned value follows the engine-facing landing unit:
      * LMCache reports the per-card value exported to LMCache.
      * Native vLLM reports the node-level value used by --kv-offloading-size.
      * MemCache reports the DRAM value written to mmc_local.conf.
      * Floor-disabled/discarded memory offload reports 0.
    """
    params = params or {}
    if not get_lmcache_env():
        return None

    resolved_variant = variant if variant is not None else resolve_offload_variant(params, engine)
    if not resolved_variant or resolved_variant == "disabled":
        return None
    if "floor_disabled" in resolved_variant:
        return 0
    if resolved_variant == memcache_hybrid.MEMCACHE_OFFLOAD_VARIANT:
        return memcache_hybrid.resolve_memcache_dram_gb(params)
    if resolved_variant.startswith("native_kv_offloading_backend"):
        special = _classify_offload_special_case(params, engine)
        if special == _OFFLOAD_V4_FLASH_NATIVE:
            size_gb = _resolve_v4_flash_offload_gb(params)
        elif _resolve_offload_backend(params, engine)[0] == _OFFLOAD_NATIVE_BACKEND_VARIANT:
            size_gb = _resolve_native_backend_offload_gb(params, engine)
        else:
            return None
        return max(0, int(size_gb))
    if resolved_variant.startswith("lmcache_cpu"):
        _, max_cpu_size = _resolve_lmcache_cpu_env(params)
        return _parse_kv_mem_size_gb(max_cpu_size)
    return None


def _native_offload_auto_floor_disabled(
    params: Optional[Dict[str, Any]],
    special: str,
) -> bool:
    """Return whether a native KV offload special case is disabled by auto floor."""
    if not params:
        params = {}
    if special == _OFFLOAD_V4_FLASH_NATIVE:
        return is_kv_mem_offload_auto_floor_disabled(params)
    return False


def _build_qat_env_commands(engine) -> List[str]:
    """构建 KVCache QAT 压缩特性的环境变量设置命令。

    QAT (QuickAssist Technology) 是 Intel 的硬件压缩加速技术，
    可用于压缩 KV 缓存以减少内存占用和传输开销。

    注意:
        - 当前仅 vllm (NVIDIA) 支持 QAT 压缩
        - vllm_ascend 不支持，会自动禁用并打印警告

    Args:
        engine: 引擎类型

    Returns:
        List[str]: LMCACHE_QAT_ENABLED 设置命令列表

    环境变量:
        - LMCACHE_QAT: 是否启用 QAT 压缩 (true/false)
    """
    env_commands = []
    if not get_qat_env():
        return env_commands

    if engine == "vllm":
        env_commands.append('export LMCACHE_QAT_ENABLED=True')
    else:
        env_commands.append('export LMCACHE_QAT_ENABLED=False')
        logger.warning("[KVCache Offload] QAT compression feature is not supported by the current engine %s, "
                       "it has been automatically disabled", engine)
    return env_commands


def _build_pd_role_env_commands(engine: str, current_ip: str, network_interface: str) -> List[str]:
    """构建 PD 分离部署的环境变量设置命令。

    PD 分离 (Prefill-Decode Disaggregation) 是一种高级部署架构，
    将 Prefill 和 Decode 阶段分离到不同节点，以优化资源利用率。

    vllm (NVIDIA) 场景:
        - 使用 NIXL 协议进行 KV 传输
        - 设置 VLLM_NIXL_SIDE_CHANNEL_HOST

    vllm_ascend 场景:
        - 使用 HCCL 进行跨节点通信
        - 需要设置多个网络接口环境变量
        - 依赖 CANN 和 ATB 工具包

    Args:
        engine:           引擎类型 ('vllm' 或 'vllm_ascend')
        current_ip:       当前节点 IP 地址
        network_interface: 网络接口名称 (如 'eth0')

    Returns:
        List[str]: PD 分离所需的环境变量设置命令

    环境变量:
        - PD_ROLE: PD 角色 ('P' 或 'D')
        - VLLM_LLMDD_RPC_PORT: LLMDataDist RPC 端口号
    """
    env_commands = []
    if get_pd_role_env():
        if engine == "vllm":
            env_commands.append(f'export VLLM_NIXL_SIDE_CHANNEL_HOST={shlex.quote(current_ip)}')
        elif engine == "vllm_ascend":
            rpc_port = os.getenv('VLLM_LLMDD_RPC_PORT', "5569")
            mooncake_bootstrap_port = os.getenv('VLLM_MOONCAKE_BOOTSTRAP_PORT', "23000")
            # CANN 环境初始化已由 _build_base_env_commands() 完成，此处不再重复
            env_commands.extend([
                f"export HCCL_IF_IP={shlex.quote(current_ip)}",
                f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
                f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
                f"export HCCL_SOCKET_IFNAME={shlex.quote(network_interface)}",
                "export OMP_PROC_BIND=false",
                f"export OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', '100')}",
                "export VLLM_USE_V1=1",
                "export LCCL_DETERMINISTIC=1",
                "export HCCL_DETERMINISTIC=true",
                "export CLOSE_MATMUL_K_SHIFT=1",
                f"export VLLM_LLMDD_RPC_PORT={rpc_port}",
                f"export VLLM_MOONCAKE_BOOTSTRAP_PORT={mooncake_bootstrap_port}",
                f"export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:{os.getenv('NPU_MAX_SPLIT_SIZE_MB', '256')}",
                # mooncake-transfer-engine 的 Ascend 传输后端 (ascend_transport.so)
                # 安装在 /usr/local/lib，需追加到 LD_LIBRARY_PATH 以便运行时加载
                'export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"',
            ])
    return env_commands


def _build_distributed_env_commands(params: Dict[str, Any], current_ip: str,
                                    network_interface: str, engine: str) -> List[str]:
    """构建分布式环境变量设置命令（扩展点）。

    当前返回空列表，分布式 NCCL/HCCL 环境设置已在
    _build_pd_role_env_commands 和 build_start_script 内部的
    Ray 初始化块中处理。

    根据 distributed_executor_backend（ray / dp_deployment）和引擎类型
    （vllm / vllm_ascend）设置对应的网络通信环境变量。

    Args:
        params:            参数字典
        current_ip:        当前节点 IP
        network_interface: 网络接口名称
        engine:            引擎类型

    Returns:
        List[str]: 环境变量设置命令列表
    """
    env_commands: List[str] = []
    if params.get("distributed", False):
        backend = params.get("distributed_executor_backend")
        if backend == "ray":
            env_commands = _build_ray_network_env_commands(current_ip, network_interface, engine)
        elif backend == "dp_deployment":
            env_commands = _build_dp_network_env_commands(params, current_ip, network_interface, engine)
    return env_commands


def _build_ray_network_env_commands(current_ip: str, network_interface: str, engine: str) -> List[str]:
    """Build Ray distributed network environment commands."""
    if engine == "vllm":
        return [
            f"export VLLM_HOST_IP={shlex.quote(current_ip)}",
            f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
            f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
            f"export NCCL_SOCKET_IFNAME={shlex.quote(network_interface)}",
        ]
    if engine == "vllm_ascend":
        return [
            f"export HCCL_IF_IP={shlex.quote(current_ip)}",
            f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
            f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
            "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
            "export ASCEND_PROCESS_LOG_PATH=/tmp/ray_vllm010",
            # Ascend NPU 首次推理 JIT 编译算子耗时可能远超 Ray 编译DAG默认 300s 超时
            "export RAY_CGRAPH_get_timeout=" + os.getenv('RAY_CGRAPH_get_timeout', '3600'),
        ]
    return []


def _build_dp_network_env_commands(
    params: Dict[str, Any],
    current_ip: str,
    network_interface: str,
    engine: str,
) -> List[str]:
    """Build dp_deployment network environment commands."""
    if engine == "vllm":
        return [
            f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
            f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
            f"export NCCL_SOCKET_IFNAME={shlex.quote(network_interface)}",
            f"export VLLM_NIXL_SIDE_CHANNEL_PORT={params.get('nixl_port', '')}",
            "export NCCL_IB_DISABLE=0",
            "export NCCL_CUMEM_ENABLE=0",
            "export NCCL_NET_GDR_LEVEL=SYS",
        ]
    if engine == "vllm_ascend":
        return _build_ascend_dp_network_env_commands(params, current_ip, network_interface)
    return []


def _build_ascend_dp_network_env_commands(
    params: Dict[str, Any],
    current_ip: str,
    network_interface: str,
) -> List[str]:
    """Build vLLM-Ascend dp_deployment network environment commands."""
    dp_arch = _get_deepseek_ascend_dp_model_architecture(params)
    is_glm5_dp = dp_arch == "GlmMoeDsaForCausalLM"
    omp_default = '1' if is_glm5_dp else '10'
    hccl_buffsize_default = '1024' if is_glm5_dp else '1024'
    omp_threads = os.getenv('OMP_NUM_THREADS', omp_default)
    hccl_buffsize = os.getenv('HCCL_BUFFSIZE', hccl_buffsize_default)
    commands = [
        f"export HCCL_IF_IP={shlex.quote(current_ip)}",
        f"export GLOO_SOCKET_IFNAME={shlex.quote(network_interface)}",
        f"export TP_SOCKET_IFNAME={shlex.quote(network_interface)}",
        f"export HCCL_SOCKET_IFNAME={shlex.quote(network_interface)}",
        "export OMP_PROC_BIND=false",
        f"export OMP_NUM_THREADS={omp_threads}",
        f"export HCCL_BUFFSIZE={hccl_buffsize}",
        'echo "[wings-env] final HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"',
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_OP_EXPANSION_MODE=AIV",
    ]
    if dp_arch in ("DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"):
        # OPP 自定义算子路径是 DeepSeek 系列专属，套到 GLM 等其它架构会加载错算子。
        commands.extend([
            "export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/"
            "latest/opp/deepseek-v32/vendors/customize:${ASCEND_CUSTOM_OPP_PATH:-}",
            "export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/"
            "opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH:-}",
        ])
    return commands


def _build_ascend910_9362_env_commands(params: Dict[str, Any], engine: str) -> List[str]:
    """构建 Ascend910_9362 设备特定环境变量命令。

    当满足以下条件时，添加特定的环境变量：
    1. 通过 torch_npu 检测设备名称为 Ascend910_9362
    2. 模型结构为 DeepseekV32ForCausalLM 或 DeepseekV3ForCausalLM
    3. 引擎为 vllm_ascend
    4. 不是 dp_deployment 分布式模式（避免与 _build_distributed_env_commands 重复）

    Args:
        params: 参数字典
        engine: 引擎类型

    Returns:
        List[str]: 环境变量导出命令列表
    """
    env_commands = []
    distributed_backend = params.get("distributed_executor_backend")

    # 从硬件信息 JSON 或环境变量中获取设备名称（不依赖 torch_npu SDK）
    device_name = None
    for detail in params.get("device_details") or []:
        if isinstance(detail, dict) and detail.get("name"):
            device_name = str(detail.get("name")).strip()
            break
    if device_name:
        logger.info("[Ascend910_9362] Detected device from hardware info: %s", device_name)

    if device_name != "Ascend910_9362":
        return env_commands

    if engine != "vllm_ascend":
        return env_commands

    if distributed_backend == "dp_deployment":
        return env_commands

    if not params.get("model_path"):
        return env_commands

    model_info = ModelIdentifier(
        params.get("model_name"),
        params.get("model_path"),
        params.get("model_type")
    )

    if model_info.model_architecture in ["DeepseekV32ForCausalLM", "DeepseekV3ForCausalLM"]:
        env_commands.extend([
            "export OMP_PROC_BIND=false",
            "export OMP_NUM_THREADS=10",
            "export HCCL_BUFFSIZE=1024"
        ])
        logger.info("[Ascend910_9362] Set environment variables for %s", model_info.model_architecture)

    return env_commands


def _build_glm4moe_ascend_env(arch: str) -> List[str]:
    """构建 GLM-4.7 (Glm4MoeForCausalLM) Ascend 环境变量命令。"""
    logger.info("[GLM-4.7] Set Ascend environment variables for %s", arch)
    return [
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
        "export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
        "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
    ]


def _build_glm_moe_dsa_ascend_env(arch: str) -> List[str]:
    """构建 GLM-5/5.1 (GlmMoeDsaForCausalLM) Ascend 环境变量命令。"""
    logger.info("[GLM-5] Set Ascend environment variables for %s", arch)
    return [
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export HCCL_BUFFSIZE=1024",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
    ]


def _build_qwen3_ascend_env(arch: str) -> List[str]:
    """构建 Qwen3 密集模型 (Qwen3ForCausalLM) Ascend 环境变量命令。

    适用于 Qwen3-32B 等密集架构。
    """
    logger.info("[Qwen3] Set Ascend environment variables for %s", arch)
    return [
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
    ]


def _build_qwen35_ascend_env(arch: str) -> List[str]:
    """构建 Qwen3.5 (Qwen3_5ForConditionalGeneration) Ascend 环境变量命令。"""
    logger.info("[Qwen3.5] Set Ascend environment variables for %s", arch)
    return [
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
    ]


def _build_qwen35moe_ascend_env(arch: str) -> List[str]:
    """构建 Qwen3.5-MoE (Qwen3_5MoeForConditionalGeneration) Ascend 环境变量命令。"""
    logger.info("[Qwen3.5-MoE] Set Ascend environment variables for %s", arch)
    return [
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
    ]


def _build_minimaxm2_ascend_env(arch: str) -> List[str]:
    """构建 MiniMax-M2.5 (MiniMaxM2ForCausalLM) Ascend 环境变量命令。

    注入 MiniMax-M2.5 在 Ascend 910B 上所需的环境变量：
    - OMP_NUM_THREADS:                限制 OpenMP 线程数
    - TASK_QUEUE_ENABLE:              启用 Ascend task queue
    - VLLM_USE_GRAPH:                  启用 NPU Graph 加速
    - VLLM_USE_V1:                     启用 vLLM V1 多进程架构
    - VLLM_ASCEND_ENABLE_FLASHCOMM1:   启用 FlashComm 通信优化（EP 密集通信场景）
    - VLLM_ASCEND_ENABLE_FUSED_MC2:    启用 Fused MC2 通信优化
    - VLLM_ASCEND_BALANCE_SCHEDULING:  启用 Ascend 均衡调度
    - VLLM_TORCH_COMPILE:              关闭 torch.compile（Ascend 910B 兼容性）

    注意: HCCL_OP_EXPANSION_MODE、HCCL_BUFFSIZE、PYTORCH_NPU_ALLOC_CONF、jemalloc
    和系统性能调优已由 set_vllm_ascend_env.sh 全局设置，此处不重复注入。
    """
    logger.info("[MiniMax-M2.5] Set Ascend environment variables for %s", arch)
    return [
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
        "export VLLM_USE_GRAPH=1",
        "export VLLM_USE_V1=1",
        "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
        "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
        "export VLLM_TORCH_COMPILE=0",
    ]


def _build_deepseekv32_ascend_env(arch: str) -> List[str]:
    """构建 DeepSeek V3.2 (DeepseekV32ForCausalLM) Ascend 环境变量命令。"""
    logger.info("[DeepSeek V3.2] Set Ascend environment variables for %s", arch)
    # DeepSeek V3.2 独有变量
    return [
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=10",
        "export VLLM_USE_V1=1",
        "export HCCL_BUFFSIZE=512",
        "export VLLM_ASCEND_ENABLE_MLAPO=1",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
    ]


def _build_llama_ascend_env(arch: str) -> List[str]:
    """构建 LLaMA3.1 (LlamaForCausalLM) Ascend 环境变量命令。"""
    logger.info("[LLaMA3.1] Set Ascend environment variables for %s", arch)
    return [
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
    ]


def _is_kimik25_distributed() -> bool:
    """判断当前是否为 Kimi-K2.5 分布式部署场景。

    通过 DISTRIBUTED 环境变量判断是否处于分布式模式：
    - 分布式: 不开EP，且不注入 VLLM_ASCEND_ENABLE_FLASHCOMM1，
              并新增 HCCL_INTRA_PCIE_ENABLE，HCCL_INTRA_ROCE_ENABLE
    - 非分布式: 开EP，注入 VLLM_ASCEND_ENABLE_FLASHCOMM1=1

    Returns:
        True 如果 DISTRIBUTED 环境变量为 true/1
    """
    engine_version = os.getenv("DISTRIBUTED", "")
    
    return engine_version in [True, "true", 1]


def _build_kimik25_ascend_env(arch: str) -> List[str]:
    """构建 Kimi-K2.5 (KimiK25ForConditionalGeneration) Ascend 环境变量命令。

    注入 Kimi-K2.5 在 Ascend 上所需的环境变量：
    - HCCL_OP_EXPANSION_MODE=AIV: 启用 AIV 通信优化
    - PYTORCH_NPU_ALLOC_CONF=expandable_segments:True: 启用 NPU 内存扩展段
    - OMP_PROC_BIND=false / OMP_NUM_THREADS=1: OpenMP 线程绑定与数量限制
    - TASK_QUEUE_ENABLE=1: 启用任务队列
    - HCCL_BUFFSIZE=1024: HCCL 缓冲区大小
    - VLLM_ASCEND_ENABLE_MLAPO=1: 启用 MLAPO 优化
    - VLLM_ASCEND_BALANCE_SCHEDULING=1: 启用负载均衡调度
    - VLLM_ENGINE_READY_TIMEOUT_S=3600: 引擎就绪超时时间

    根据 DISTRIBUTED 是否分布式 决定分支注入：
    - 非分布式: 注入 VLLM_ASCEND_ENABLE_FLASHCOMM1=1（FlashComm 通信优化）
    - 分布式: 注入 HCCL_INTRA_PCIE_ENABLE=1 / HCCL_INTRA_ROCE_ENABLE=0
              （HCCL 网卡绑定配置，不注入 FlashComm1）
    """
    is_distributed = _is_kimik25_distributed()
    logger.info(
        "[Kimi-K2.5] Set Ascend environment variables for %s (DISTRIBUTED=%s)",
        arch, is_distributed,
    )
    env_vars = [
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
        "export PYTHONHASHSEED=0",
        'export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"',
        "export HCCL_BUFFSIZE=1024",
        "export VLLM_ASCEND_ENABLE_MLAPO=1",
        "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
        "export VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ]
    if not is_distributed:
        env_vars.append("export VLLM_ASCEND_ENABLE_FLASHCOMM1=1")
    else:
        env_vars.append("export HCCL_INTRA_PCIE_ENABLE=1")
        env_vars.append("export HCCL_INTRA_ROCE_ENABLE=0")
    return env_vars


def _build_glm5_ascend_env(arch: str, platform: str = "") -> List[str]:
    """构建 GLM-5 (GlmMoeDsaForCausalLM) Ascend 环境变量命令。

    数值对齐 vllm-ascend 官方 GLM-5 W8A8 多机部署文档：
    HCCL_OP_EXPANSION_MODE=AIV / OMP_NUM_THREADS=1 / HCCL_BUFFSIZE=1024 /
    VLLM_ASCEND_BALANCE_SCHEDULING=1。

    A3（W8A8 官方双机命令）额外追加 ``VLLM_ASCEND_ENABLE_MLAPO=1``；
    A2 / 未识别保持不变。
    """
    logger.info("[GLM-5] Set Ascend environment variables for %s (platform=%s)", arch, platform or "auto")
    env = [
        "export HCCL_OP_EXPANSION_MODE=AIV",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export HCCL_BUFFSIZE=1024",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export VLLM_ASCEND_BALANCE_SCHEDULING=1",
    ]
    if platform == "a3":
        env.append("export VLLM_ASCEND_ENABLE_MLAPO=1")
    return env


# DeepSeek-V4 专属路径必须以真实架构为边界，而不是只看 model_name。
# 这里刻意不包含 DeepseekV3ForCausalLM：部分启动参数可能把 V3 权重改名成
# V4-Flash/Pro，但那不能触发 V4 的 MTP、TP/DP、专属 offload 默认值。
# 如果后续确有真实 V4 权重沿用 V3 架构名，必须先用真实 config.json 证据和
# 负向测试补齐，再调整白名单，避免把 V3 路径误导入 V4 特例。
_DEEPSEEK_V4_FLASH_ARCHES = {
    "DeepseekV4ForCausalLM",
    "DeepSeekV4ForCausalLM",
}


_DEEPSEEK_V4_CPU_OFFLOAD_ARCHES = {
    "DeepseekV4ForCausalLM",
    "DeepSeekV4ForCausalLM",
}


_ASCEND_A2_PLATFORM_TOKENS = {"a2", "atlas-a2", "atlas_a2", "910b"}
_ASCEND_A3_PLATFORM_TOKENS = {"a3", "atlas-a3", "atlas_a3", "910c"}


def _match_ascend_platform(value: Any) -> str:
    """Normalize an Ascend platform/chip token to a2 or a3."""
    token = str(value or "").strip().lower()
    if not token:
        return ""
    if any(marker in token for marker in _ASCEND_A3_PLATFORM_TOKENS):
        return "a3"
    if any(marker in token for marker in _ASCEND_A2_PLATFORM_TOKENS):
        return "a2"
    return ""


def _ascend_platform_from_runtime(params: Dict[str, Any]) -> str:
    """Return Ascend platform from hardware-info first, then ENGINE_VERSION."""
    for detail in params.get("device_details") or []:
        platform = _match_ascend_platform(detail.get("name"))
        if platform:
            return platform
    return engine_version_platform() or "a2"


_DEEPSEEK_V4_IDENTITY_CONFIG_KEYS = (
    "_name_or_path",
    "name_or_path",
    "model_name",
    "model_id",
    "base_model_name",
    "source_model",
)


def _append_identity_candidate(candidates: List[str], value: Any) -> None:
    """把模型身份候选值扁平化进 candidates。

    identity 来源可能是字符串、列表或 config.json 中的嵌套字段。这里不做语义判断，
    只负责收集可搜索文本，真正的 V4/V4-Pro/V4-Flash 判定留给后续 gate。
    """
    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            candidates.append(value)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_identity_candidate(candidates, item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _append_identity_candidate(candidates, item)


def _load_model_identity_config(
    params: Dict[str, Any],
    model_info: Optional[ModelIdentifier] = None,
) -> Dict[str, Any]:
    """读取模型 config.json，用作 V4 身份和架构证据。

    优先复用已构造的 ModelIdentifier.config，避免重复读文件；只有调用方没有
    model_info 时才从 model_path/config.json 兜底读取。读取失败只返回空字典，
    因为模型身份识别是默认值注入的守卫，不应该阻断非 V4 模型启动。
    """
    config = getattr(model_info, "config", None)
    if isinstance(config, dict):
        return config

    model_path = params.get("model_path")
    if not model_path:
        return {}
    config_path = os.path.join(str(model_path), "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("[DeepSeek-V4] Skip model identity config %s: %s", config_path, exc)
        return {}


def _deepseek_v4_identity_text(
    params: Dict[str, Any],
    model_info: Optional[ModelIdentifier] = None,
) -> str:
    """拼接可用于识别 DeepSeek-V4 变体的身份文本。

    取值范围覆盖 CLI model_name、served_model_name、模型路径、engine_config 中
    的模型字段，以及 config.json 里的常见来源字段。这样可以兼容用户改名、
    上层只传 served_model_name、或权重目录中保留原始 _name_or_path 的场景。
    """
    candidates: List[str] = []
    for value in (
        params.get("model_name"),
        params.get("served_model_name"),
        params.get("model_path"),
    ):
        _append_identity_candidate(candidates, value)

    engine_config = params.get("engine_config") or {}
    for value in (
        engine_config.get("served_model_name"),
        engine_config.get("model"),
    ):
        _append_identity_candidate(candidates, value)

    config = _load_model_identity_config(params, model_info)
    for key in _DEEPSEEK_V4_IDENTITY_CONFIG_KEYS:
        _append_identity_candidate(candidates, config.get(key))

    return " ".join(str(item).lower() for item in candidates if item)


def _is_deepseek_v4_flash_params(
    params: Dict[str, Any],
    model_info: Optional[ModelIdentifier] = None,
) -> bool:
    """Return True for DeepSeek-V4-Flash launch params without affecting other DeepSeek models.

    判定顺序是“名称证据 + 架构否决”：
    - 名称出现 V4-Flash 时，如果存在 config 架构且不是 V4，直接否决；
    - 名称缺失但架构是 V4 时，仅在身份文本同时包含 v4/flash 时兜底命中。
    这样可以覆盖用户改名场景，同时避免 V3 架构因名称相似误入 V4-Flash 路径。
    """
    text = _deepseek_v4_identity_text(params, model_info)
    arch_match = _deepseek_v4_arch_matches(params, model_info)
    if "deepseek-v4-flash" in text or "deepseek_v4_flash" in text or "deepseekv4flash" in text:
        return arch_match is not False
    if arch_match is True:
        return "v4" in text and "flash" in text
    return False


def is_deepseek_v4_pro_adapted_scope(
    params: Dict[str, Any],
    model_info: Optional[ModelIdentifier] = None,
) -> bool:
    """V4-Pro 适配范围闸门：仅 A3 双机分布式，落在此范围才注入 Pro 专属默认与环境。

    单机/A2/Ray 单机等场景一律不进入 V4-Pro 路径，避免覆盖通用 DeepSeek 默认。
    """
    if params.get("engine") != "vllm_ascend":
        return False
    if not _is_deepseek_v4_pro_params(params, model_info):
        return False
    if _ascend_platform_from_runtime(params) != "a3":
        return False
    if not bool(params.get("distributed")):
        return False
    nnodes = _safe_int(params.get("nnodes")) or 0
    return nnodes == 2


def _extract_quantize_from_config(config: Dict[str, Any]) -> Optional[str]:
    """从 config.json 提取量化字段，等价于 ``ModelIdentifier.identify_model_quantize``。

    单独抽出来给 ``_is_deepseek_v4_pro_params`` 在 ``model_info`` 缺席时使用，
    避免在每个 caller 处重新构造 ``ModelIdentifier``。
    """
    if not isinstance(config, dict):
        return None
    if "quantize" in config and config["quantize"]:
        return str(config["quantize"])
    quant_cfg = config.get("quantization_config")
    if isinstance(quant_cfg, dict):
        method = quant_cfg.get("quant_method")
        if method:
            return str(method)
    return None


def _deepseek_v4_arch_matches(
    params: Dict[str, Any],
    model_info: Optional[ModelIdentifier] = None,
) -> Optional[bool]:
    """Return V4 architecture match when config evidence exists, otherwise None.

    返回三态而不是 bool：
    - True：明确是 V4 架构；
    - False：明确不是 V4 架构，应否决 V4 特例；
    - None：没有架构证据，只能让名称/量化等弱信号继续判断。
    """
    arch = getattr(model_info, "model_architecture", None)
    if arch:
        return arch in _DEEPSEEK_V4_FLASH_ARCHES

    archs = _load_model_identity_config(params, model_info).get("architectures") or []
    if archs:
        return archs[0] in _DEEPSEEK_V4_FLASH_ARCHES
    return None


def _is_deepseek_v4_pro_params(
    params: Dict[str, Any],
    model_info: Optional[ModelIdentifier] = None,
) -> bool:
    """Return True for DeepSeek-V4-Pro launch params; strictly exclusive with V4-Flash.

    Pro 的名称可能被上层省略或改写，因此除显式 ``DeepSeek-V4-Pro`` 名称外，
    还允许“V4 架构 + w4a8 量化指纹”兜底。任何包含 flash 的文本都优先交给
    Flash 分支，防止 Pro 默认覆盖 Flash 的 TP=8 / DP 推导。
    """
    text = _deepseek_v4_identity_text(params, model_info)
    # V4-Flash 与 V4-Pro 必须互斥：名字中带 flash 一律视为 V4-Flash，由专用路径处理。
    if "flash" in text:
        return False
    if "deepseek-v4-pro" in text or "deepseek_v4_pro" in text or "deepseekv4pro" in text:
        return _deepseek_v4_arch_matches(params, model_info) is not False

    arch_is_deepseek_v4 = _deepseek_v4_arch_matches(params, model_info) is True
    if arch_is_deepseek_v4 and "v4" in text and "pro" in text:
        return True
    if arch_is_deepseek_v4:
        # 权重指纹兜底：V4-Pro 是 w4a8 量化的 DeepSeek-V4，V4-Flash 是 w8a8。
        # 当用户改名 / 把权重塞进通用路径以至于 identity text 丢失 "pro" 标记时，
        # 仍可由 config.json 的 quantize / quantization_config.quant_method 字段识别。
        # 这能阻止 V4-Pro 在 hardware_info.json 缺失 details 字段时被误判为 A2 →
        # 闸门拒绝注入 TP/DP → 抛 "DeepSeek Ascend DP requires positive TP" 的崩溃。
        quantize = getattr(model_info, "model_quantize", None) if model_info else None
        if not quantize:
            quantize = _extract_quantize_from_config(_load_model_identity_config(params, model_info))
        if _is_w4a8_quantize(quantize):
            return True
    return False


def _build_deepseek_v4_flash_env(params: Dict[str, Any]) -> List[str]:
    """构建 DeepSeek-V4-Flash A2/A3 vLLM-Ascend 专属环境变量。

    A2 (910B) 关键开关 ``USE_MULTI_GROUPS_KV_CACHE=1`` 与
    ``VLLM_ASCEND_ENABLE_FLASHCOMM1=1`` 与 A3 一致，缺一会触发 MTP + MoE
    场景的 ``is_kv_cache_spec_uniform`` ``'list' object has no attribute 'merge'``
    崩溃（KV cache spec 跨层非均匀）。
    """
    platform = _ascend_platform_from_runtime(params)
    common = [
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=10",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        'export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"',
        "export HCCL_BUFFSIZE=1024",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
        "export TASK_QUEUE_ENABLE=1",
        'export HCCL_OP_EXPANSION_MODE="AIV"',
    ]
    if platform == "a3":
        logger.info("[DeepSeek-V4-Flash] Set Ascend A3 environment variables")
        return common

    logger.info("[DeepSeek-V4-Flash] Set Ascend A2 environment variables")
    return common


def _build_deepseek_v4_pro_env(params: Dict[str, Any]) -> List[str]:
    """构建 DeepSeek-V4-Pro A3 双机 vLLM-Ascend 专属环境变量。

    与 vLLM-Ascend 官方 V4-Pro 双机参考脚本严格对齐。差异点：``HCCL_BUFFSIZE=2048``
    （Pro 长上下文/MoE 通信量更大，沿用 1024 会触发 HCCL OOM 风险）。

    """
    logger.info("[DeepSeek-V4-Pro] Set Ascend A3 environment variables")
    return [
        "export USE_MULTI_BLOCK_POOL=1",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=10",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export USE_MULTI_GROUPS_KV_CACHE=1",
        "export HCCL_BUFFSIZE=2048",
        "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
        "export HCCL_OP_EXPANSION_MODE=AIV",
        'export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"',
    ]


def _build_glm5_model_env(params: Dict[str, Any], arch: str) -> List[str]:
    """GLM-5/5.1/5.2 (GlmMoeDsa) 的 Ascend 模型 env（从 _build_model_env_commands 抽出，避免主函数超长）。

      * 基础 env 由 _build_glm5_ascend_env 按平台给（A3 含 MLAPO）；
      * RoCE 互联：剔除 HCCL_OP_EXPANSION_MODE=AIV，追加 MLAPO/FUSED_MC2=0；
      * GLM-5.2 专属：固定 VLLM_VERSION=0.21.0（官方单/双机 recipe，仅 5.2，不注入 5.0/5.1）。
    """
    env_commands = _build_glm5_ascend_env(arch, _ascend_platform_from_runtime(params))
    if params.get("distributed") and _is_roce_distributed():
        env_commands = [c for c in env_commands if "HCCL_OP_EXPANSION_MODE" not in c]
        env_commands.append("export VLLM_ASCEND_ENABLE_MLAPO=1")
        env_commands.append("export VLLM_ASCEND_ENABLE_FUSED_MC2=0")
        logger.info(
            "[GLM-5.1 RoCE] Removed HCCL_OP_EXPANSION_MODE, "
            "appended VLLM_ASCEND_ENABLE_MLAPO=1, VLLM_ASCEND_ENABLE_FUSED_MC2=0"
        )
    if is_glm52_model(params.get("model_name"), params.get("model_path")):
        env_commands.append("export VLLM_VERSION=0.21.0")
        logger.info("[GLM-5.2] pinned VLLM_VERSION=0.21.0 (single/dual recipe)")
    return env_commands


def _build_model_env_commands(params: Dict[str, Any], engine: str) -> List[str]:
    """构建模型架构特定的环境变量命令（支持 NVIDIA 和 Ascend）。

    根据模型架构注入引擎官方文档推荐的环境变量。

    已覆盖的 Ascend 架构:
    - Glm4MoeForCausalLM (GLM-4.7): TOPK 优化, FlashComm, Fused MC2
    - GlmMoeDsaForCausalLM (GLM-5/5.1): DSA MTP 基础运行时变量
    - Qwen3_5ForConditionalGeneration (Qwen3.5-27B): TASK_QUEUE_ENABLE
    - Qwen3_5MoeForConditionalGeneration (Qwen3.5-397B): TASK_QUEUE_ENABLE
    - MiniMaxM2ForCausalLM (MiniMax-M2.5): FlashComm
    - DeepseekV32ForCausalLM (DeepSeek V3.2): MLAPO, FlashComm, VLLM_USE_V1
    - LlamaForCausalLM (LLaMA3.1-70B): 基础 NPU 内存/线程优化

    Args:
        params: 参数字典
        engine: 引擎类型

    Returns:
        List[str]: 环境变量导出命令列表
    """
    if engine not in ("vllm", "vllm_ascend"):
        return []

    model_path = params.get("model_path")
    if not model_path:
        return []

    model_info = ModelIdentifier(
        params.get("model_name"),
        params.get("model_path"),
        params.get("model_type")
    )
    arch = model_info.model_architecture

    # [Qwen3.5-397B-A17B-NVFP4] 运行时配方：注入全部四个 env
    # 这些 env 必须在 vllm 进程启动前 export，故写入 start_command.sh 的 env 段。
    if is_qwen3_5_397b_nvfp4_vllm(params, engine):
        logger.info("[NVFP4] inject FP4 runtime env (deep_gemm/flashinfer_moe) for %s on vllm", arch)
        return [
            'export VLLM_DEEP_GEMM_WARMUP=skip',
            'export VLLM_USE_DEEP_GEMM=0',
            'export VLLM_FLASHINFER_MOE_BACKEND=latency',
            'export VLLM_USE_FLASHINFER_MOE_FP4=1',
        ]

    # [MiniMax-M2.7 + RTX-PRO-5000] 运行时配方：注入 LMCache env 集
    # （LMCACHE_* / VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS 等，与 minimax-2.7-zyy.txt 对齐）。
    # 这些 env 必须在 vllm 进程启动前 export，故写入 start_command.sh 的 env 段；
    # _build_cache_env_commands 的通用 LMCache 流程已对本场景跳过，由本分支接管。
    if is_minimax_m27_rtx_pro_5000_vllm(params, engine):
        logger.info("[MiniMax-M2.7] inject LMCache runtime env for %s on vllm (RTX-PRO-5000)", arch)
        return _build_minimax_m27_rtx_pro_5000_env_commands(params)

    if engine == "vllm_ascend" and _is_deepseek_v4_flash_params(params, model_info):
        return _build_deepseek_v4_flash_env(params)
    if engine == "vllm_ascend" and is_deepseek_v4_pro_adapted_scope(params, model_info):
        return _build_deepseek_v4_pro_env(params)

    if engine == "vllm_ascend":
        _arch_env_builders = {
            "Glm4MoeForCausalLM": _build_glm4moe_ascend_env,
            "GlmMoeDsaForCausalLM": _build_glm5_ascend_env,
            "Qwen3ForCausalLM": _build_qwen3_ascend_env,
            "Qwen3_5ForConditionalGeneration": _build_qwen35_ascend_env,
            "Qwen3_5MoeForConditionalGeneration": _build_qwen35moe_ascend_env,
            "MiniMaxM2ForCausalLM": _build_minimaxm2_ascend_env,
            "DeepseekV32ForCausalLM": _build_deepseekv32_ascend_env,
            "LlamaForCausalLM": _build_llama_ascend_env,
            "KimiK25ForConditionalGeneration": _build_kimik25_ascend_env,
        }
    else:
        _arch_env_builders = {}

    builder = _arch_env_builders.get(arch)
    if not builder:
        return []
    # GLM-5 (GlmMoeDsaForCausalLM) 在 A3 上需要追加 VLLM_ASCEND_ENABLE_MLAPO=1，
    # 与 vllm-ascend W8A8 官方双机命令对齐；其它架构构造器签名不变。
    if engine == "vllm_ascend" and arch == "GlmMoeDsaForCausalLM":
        return _build_glm5_model_env(params, arch)
    return builder(arch)


def _build_env_commands(params: Dict[str, Any], current_ip: str, network_interface: str, root: str) -> List[str]:
    """组装完整的环境变量设置命令列表。

    按顺序调用各子模块构建环境设置，创建完整的环境初始化流程：
    1. 基础环境（CANN/ATB 工具包）
    2. KVCache Offload 环境
    3. QAT 压缩环境
    4. PD 分离环境
    5. 分布式环境（扩展点）

    Args:
        params:            参数字典，包含 engine 等配置
        current_ip:        当前节点 IP 地址
        network_interface: 网络接口名称
        root:              项目根目录

    Returns:
        List[str]: 所有环境变量设置命令的有序列表
    """
    engine = params.get("engine")
    env_commands = []

    env_commands.extend(_build_base_env_commands(params, engine, root))
    env_commands.extend(_build_cache_env_commands(engine, params))
    env_commands.extend(_build_qat_env_commands(engine))
    env_commands.extend(_build_pd_role_env_commands(engine, current_ip, network_interface))
    env_commands.extend(_build_distributed_env_commands(params, current_ip, network_interface, engine))
    env_commands.extend(_build_ascend910_9362_env_commands(params, engine))
    env_commands.extend(_build_model_env_commands(params, engine))
    env_commands = _filter_vllm_ascend_ray_incompatible_env(env_commands, params, engine)
    env_commands.extend(_build_vllm_ascend_forced_env_commands(params, engine))

    return env_commands


_DP_TOPOLOGY_KEYS = (
    "tensor_parallel_size",
    "data_parallel_size",
    "data_parallel_size_local",
    "data_parallel_start_rank",
)


def _strip_internal_engine_config_keys(params: Dict[str, Any], engine_config: Dict[str, Any]) -> None:
    """清理 adapter 内部字段，并应用必须在 CLI 渲染前完成的硬约束。

    这里处理的是“不应该出现在 vLLM CLI 中”的字段：
    - use_kunlun_atb / enable_sparse 等由 adapter 自己消费；
    - ascend_platform / hardware_platform 是旧平台 override 字段，不再渲染到 CLI；
    - GLM-5.1 NVIDIA/vLLM 不允许 KV offload，即使上游合并了 kv_transfer_config
      也要在脚本生成前删除，避免运行时加载不兼容 connector。
    """
    if _is_glm51_nvidia_vllm_params(params, params.get("engine", "vllm")):
        removed = engine_config.pop("kv_transfer_config", None)
        if removed is not None:
            logger.warning(
                "[KVCache Offload] Forced disabled for GLM-5.1 on NVIDIA/vLLM; "
                "removed upstream kv_transfer_config=%s",
                removed,
            )
    engine_config.pop("use_kunlun_atb", None)
    engine_config.pop("enable_sparse", None)
    engine_config.pop("ascend_platform", None)
    engine_config.pop("hardware_platform", None)


def _apply_generic_deepseek_ascend_dp_defaults(
    params: Dict[str, Any],
    engine_config: Dict[str, Any],
    explicit_keys: set,
) -> None:
    """通用 DeepSeek Ascend dp_deployment 默认值。

    V3/V3.1/V3.2 等普通 DeepSeek DP 走这里：自动推 TP、关闭 prefix cache、
    开 EP/async。V4-Flash 与 V4-Pro 各自有完整默认值注入函数，必须提前跳过，
    否则通用 DP 逻辑会覆盖 V4 专属的 prefix cache、TP/DP 和 MTP 配置。
    """
    if not _is_deepseek_ascend_dp_deployment(params):
        return
    if _is_deepseek_v4_pro_params(params) or _is_deepseek_v4_flash_params(params):
        return

    model_architecture = _get_deepseek_ascend_dp_model_architecture(params)
    device_count = _safe_int(params.get("device_count"))
    default_tp = _default_deepseek_ascend_dp_tensor_parallel_size(model_architecture or "", device_count)
    # [GLM-5.2] 单机(nnodes==1)按官方 recipe 跑 DP=2 → TP=device_count//2（16卡 → TP8/DP2）；
    # 双机维持每节点 TP=device_count（每节点 1 个 DP replica，TP16），由
    # _resolve_dp_deployment_topology 推 DP-local=1/DP=2。按 is_glm52 子串标识，复杂名亦稳。
    if model_architecture == "GlmMoeDsaForCausalLM" and is_glm52_single_node_even(params):
        default_tp = device_count // _GLM52_SINGLE_NODE_DP
        _set_if_not_explicit(engine_config, explicit_keys, "data_parallel_size", _GLM52_SINGLE_NODE_DP)
    if default_tp and "tensor_parallel_size" not in explicit_keys:
        engine_config["tensor_parallel_size"] = default_tp

    prefix_cache_explicit = bool(
        explicit_keys.intersection({"enable_prefix_caching", "no_enable_prefix_caching"})
    )
    if (
        not prefix_cache_explicit
        and engine_config.get("enable_prefix_caching") not in (None, False, "False", 0, "0")
    ):
        logger.warning(
            "[DeepSeek Ascend DP] prefix caching is incompatible with the "
            "dp_deployment path; forcing --no-enable-prefix-caching."
        )
    if not prefix_cache_explicit:
        engine_config.pop("enable_prefix_caching", None)
        engine_config["no_enable_prefix_caching"] = True

    if (
        "enable_expert_parallel" not in explicit_keys
        and engine_config.get("enable_expert_parallel") in (None, False, "False", 0, "0")
    ):
        logger.info(
            "[DeepSeek Ascend DP] enabling expert parallel to align with "
            "vLLM-Ascend DeepSeek multi-node launch examples."
        )
    if "enable_expert_parallel" not in explicit_keys:
        engine_config["enable_expert_parallel"] = True
    
    # KimiK25: EP 由 是否分布式决定。分布式：不开EP。 非分布式：开EP
    if (model_architecture == "KimiK25ForConditionalGeneration"
        and _is_kimik25_distributed()
    ):
        engine_config.pop("enable_expert_parallel")



def _apply_glm5_dsa_distributed_fixups(
    params: Dict[str, Any],
    engine_config: Dict[str, Any],
    explicit_keys: set,
) -> None:
    """对齐 vllm-ascend 官方 GLM-5/5.1 双机模板的两处修正。

    仅作用于 GlmMoeDsaForCausalLM + dp_deployment（双机分布式）：
      (#prefix) prefix caching：撤销通用 DeepSeek DP 路径的 force-off —— 官方 GLM5
                单机/双机模板均开启 prefix caching；
      (#additional) additional_config：A3（910C）双机不下发 fuse_muls_add /
                multistream_overlap_shared_expert / enable_npugraph_ex —— 官方 A3
                多机模板特意省略，这些图优化开关在长上下文 decode replay 时触发
                MTE 地址越界崩溃。A2 双机仍保留。

    必须在 ``_apply_generic_deepseek_ascend_dp_defaults`` 之后调用，才能覆盖它对
    prefix caching 的强制关闭。``explicit_keys`` 始终优先，用户显式配置不被覆盖。
    """
    if params.get("engine") != "vllm_ascend":
        return
    if _get_deepseek_ascend_dp_model_architecture(params) != "GlmMoeDsaForCausalLM":
        return

    # 不强制关闭 prefix caching（官方 GLM5 单机/双机均开启）
    if not explicit_keys.intersection({"enable_prefix_caching", "no_enable_prefix_caching"}):
        engine_config.pop("no_enable_prefix_caching", None)
        engine_config["enable_prefix_caching"] = True

    # 仅 A3（910C）双机移除 additional_config（GLM-5.1 规避全图 decode replay MTE 越界）。
    # GLM-5.2 已稳定，豁免剥除——保留 fuse_muls_add/multistream/enable_npugraph_ex 三键图优化。
    if "additional_config" not in explicit_keys:
        if (_ascend_platform_from_runtime(params) == "a3"
                and not is_glm52_model(params.get("model_name"), params.get("model_path"))):
            engine_config.pop("additional_config", None)


def _writeback_dp_topology_to_params(params: Dict[str, Any], engine_config: Dict[str, Any]) -> None:
    """Mirror DP topology keys back into params["engine_config"] so downstream readers see them.

    Why: ``_resolve_dp_deployment_topology`` reads TP from ``params["engine_config"]``. Without this
    writeback, V4-Flash/Pro (not in ``_default_deepseek_ascend_dp_tensor_parallel_size`` fallback table)
    would crash with "requires a positive tensor_parallel_size".
    """
    params_engine_config = params.setdefault("engine_config", {})
    for key in _DP_TOPOLOGY_KEYS:
        if key in engine_config:
            params_engine_config[key] = engine_config[key]


def _prepare_engine_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """准备最终传给 ``_build_vllm_cmd_parts`` 的 engine_config。

    执行顺序不能随意调整：
    1. 先删除内部字段，避免它们被格式化成 vLLM CLI；
    2. 再注入 V4/GLM/DeepSeek DP 等模型专属默认；
    3. 最后处理旧版 ``task`` 字段和 DP 拓扑回写。
    DP 回写必须在返回前完成，因为分布式脚本拓扑计算后续读取的是
    ``params["engine_config"]``，而不是本函数局部变量。
    """
    engine_config = dict(params.get("engine_config", {}))
    _strip_internal_engine_config_keys(params, engine_config)
    explicit_keys = set(params.get("_explicit_cli_keys") or [])

    _apply_deepseek_v4_flash_engine_defaults(params, engine_config, explicit_keys)
    _apply_deepseek_v4_pro_engine_defaults(params, engine_config, explicit_keys)
    _apply_deepseek_v4_flash_nv_engine_defaults(params, engine_config, explicit_keys)
    # MiniMax-M2.7-NVFP4 + RTX-PRO-5000 + vLLM(NVIDIA) TP/DP 动态策略
    # （TP=min(4,device_count) + DP=device_count/TP，与 DeepSeek-V4-Flash-NV 同构）
    _apply_minimax_m27_nvfp4_nv_engine_defaults(params, engine_config, explicit_keys)
    # block_size 固定 256 必须在所有可能写入 block_size 的默认注入之后执行，
    # 以确保最终值恒为 256（覆盖 json 默认与用户显式值）。
    _force_deepseek_v4_flash_nv_block_size(params, engine_config)
    _apply_glm5_ascend_engine_defaults(params, engine_config, explicit_keys)
    _apply_generic_deepseek_ascend_dp_defaults(params, engine_config, explicit_keys)
    _apply_glm5_dsa_distributed_fixups(params, engine_config, explicit_keys)
   

    # ── GLM-5.1 RoCE 互联场景：强制剔除 async_scheduling / enable_expert_parallel，
    #    并将投机推理 speculative_config 替换为 RoCE 适配版本 ──
    is_glm5_roce = (
        params.get("engine") == "vllm_ascend"
        and _get_deepseek_ascend_dp_model_architecture(params) == "GlmMoeDsaForCausalLM"
        and _is_roce_distributed()
    )
    if params.get("distributed") and is_glm5_roce:
        removed_keys = []
        if engine_config.pop("async_scheduling", None) is not None:
            removed_keys.append("async_scheduling")
        if engine_config.pop("enable_expert_parallel", None) is not None:
            removed_keys.append("enable_expert_parallel")
        # 投机推理使用 RoCE 适配配置
        if params.get("enable_speculative_decode"):
            engine_config["speculative_config"] = {
                "num_speculative_tokens": 3,
                "method": "deepseek_mtp",
            }
            logger.info(
                "[GLM-5.1 RoCE] Replaced speculative_config with "
                "num_speculative_tokens=3, method=deepseek_mtp"
            )
        if removed_keys:
            logger.info(
                "[GLM-5.1 RoCE] Forcibly removed engine_config keys: %s",
                ", ".join(removed_keys),
            )

    # PD external-lb：pd_config.json 注册表是 PD 引擎参数的唯一真相源。上面的模型默认注入器
    # （_apply_*_engine_defaults）用 _force_set_* / _merge_dict_default_* 回填了部分键，会覆盖
    # _apply_pd_external_lb 写入的注册表值；故在所有注入器之后重申注册表覆盖。
    # 仅 PD external-lb 命中（非 PD 部署 _pd_engine_overrides 为空 → 行为字节级不变）。
    # value=None 表示该角色应删除该 base 键（如 Prefill 删除 base 注入的 compilation_config）。
    pd_overrides = params.get("_pd_engine_overrides")
    if pd_overrides:
        pd_topology_keys = {"tensor_parallel_size", "data_parallel_size"}
        for k, v in pd_overrides.items():
            if k in explicit_keys and k not in pd_topology_keys:
                continue
            if v is None:
                engine_config.pop(k, None)
            else:
                engine_config[k] = v

    removed_task = engine_config.pop("task", None)
    if removed_task and removed_task != "generate":
        logger.info("[vLLM] Mapping deprecated task=%s to --runner pooling", removed_task)
        engine_config.setdefault("runner", "pooling")

    _writeback_dp_topology_to_params(params, engine_config)
    # 同步 speculative_config 回 params，阻止 _should_append_auto_speculative_config 重复合成
    if "speculative_config" in engine_config:
        params.setdefault("engine_config", {})["speculative_config"] = engine_config["speculative_config"]
    return engine_config


def prepare_params_for_startup_status(params: Dict[str, Any]) -> None:
    """Write back final topology defaults needed by startup status reporting.

    ``advanced_features.json`` is written before the shell is executed, but its
    offload status must match the command that ``build_start_script`` will
    render. Reuse the same engine-config preparation step so auto KV offload
    formulas see the final TP/DP values.
    """
    engine = str((params or {}).get("engine") or "")
    if engine not in {"vllm", "vllm_ascend"}:
        return
    try:
        prepared_config = _prepare_engine_config(params)
    except Exception:  # noqa: BLE001
        logger.exception(
            "[StartupStatus] Failed to prepare final engine config for status reporting."
        )
        raise
    if not isinstance(prepared_config, dict):
        raise TypeError(
            "_prepare_engine_config must return a dict, "
            f"got {type(prepared_config).__name__}"
        )
    # 显式消费准备结果并回写最终拓扑；该操作与 _prepare_engine_config 内部回写
    # 幂等，既满足状态计算需要，也避免调用方静默丢弃非空返回值。
    _writeback_dp_topology_to_params(params, prepared_config)


def _set_if_not_explicit(
    engine_config: Dict[str, Any],
    explicit_keys: set,
    key: str,
    value: Any,
) -> None:
    """Set engine_config[key] unless user explicitly supplied the key."""
    if key in explicit_keys:
        return
    if engine_config.get(key) in (None, "", [], {}):
        engine_config[key] = value


def _merge_dict_default_if_not_explicit(
    engine_config: Dict[str, Any],
    explicit_keys: set,
    key: str,
    default_value: Dict[str, Any],
) -> None:
    """Deep-merge a dict default while preserving user-supplied values."""
    if key in explicit_keys:
        return
    current = engine_config.get(key)
    if current in (None, "", [], {}):
        engine_config[key] = dict(default_value)
        return
    parsed_current = _parse_dict_like_config(current)
    if parsed_current is None:
        return
    engine_config[key] = _deep_merge_user_priority(parsed_current, default_value)


def _set_deepseek_v4_flash_additional_config(
    engine_config: Dict[str, Any],
    explicit_keys: set,
    platform: str,
) -> None:
    """Set DeepSeek-V4-Flash additional_config defaults with platform-specific override.

    A3 长上下文场景额外开启 ascend_compilation_config 与 multistream_dsa_preprocess，
    对齐 vllm-ascend DeepSeek-V4-Flash 官方 benchmark 启动模板。
    """
    if "additional_config" in explicit_keys:
        return
    current = _parse_dict_like_config(engine_config.get("additional_config")) or {}
    # ``enable_cpu_binding`` 必须是 bool（``_format_cli_arg`` 对字符串 "true"
    # 会作为普通字符串发出 ``--... 'true'``，与 vLLM 0.18 期望的 bool 不符）。
    current.setdefault("enable_cpu_binding", True)
    # 官方 A2/A3 启动模板均显式关闭 ``multistream_overlap_shared_expert``：
    # 该项在 V4-Flash MoE+MTP 路径会与 KV cache spec 合并步骤冲突。
    current["multistream_overlap_shared_expert"] = False
    if platform == "a3":
        current.setdefault("multistream_dsa_preprocess", False)
        ascend_compile = _parse_dict_like_config(current.get("ascend_compilation_config")) or {}
        ascend_compile.setdefault("enable_npugraph_ex", True)
        ascend_compile.setdefault("enable_static_kernel", False)
        current["ascend_compilation_config"] = ascend_compile
    engine_config["additional_config"] = current


def _force_set_if_not_explicit(
    engine_config: Dict[str, Any],
    explicit_keys: set,
    key: str,
    value: Any,
) -> None:
    """Overwrite engine_config[key] unless user supplied it explicitly.

    Why: ``_set_if_not_explicit`` skips non-empty existing values, which would silently keep
    upstream defaults (e.g. ``enable_expert_parallel=False`` → MoE+MTP crash).
    """
    if key not in explicit_keys:
        engine_config[key] = value


def _default_deepseek_v4_flash_tensor_parallel_size(platform: str) -> int:
    """Return the platform-specific DeepSeek-V4-Flash TP default."""
    return 4 if platform == "a3" else 8


def _compute_deepseek_v4_flash_data_parallel_size(
    params: Dict[str, Any],
    tensor_parallel_size: int,
) -> int:
    """按整集群剩余卡拉满 DP。

    典型结果：A2 单机 8 卡 + TP=8 -> DP=1；A3 单机 16 卡 + TP=4 -> DP=4；
    A3 双机 16x2 卡 + TP=4 -> DP=8。这里只负责计算 Flash 的推荐 DP，
    显式 CLI/ENV 覆盖仍由调用方的 explicit_keys 判断保护。
    """
    device_count = _safe_int(params.get("device_count")) or 8
    is_distributed = bool(params.get("distributed"))
    nnodes = _safe_int(params.get("nnodes")) or (2 if is_distributed else 1)
    total_cards = device_count * (nnodes if is_distributed else 1)
    return max(1, total_cards // max(1, tensor_parallel_size))


def _apply_deepseek_v4_flash_capacity_and_topology(
    params: Dict[str, Any],
    engine_config: Dict[str, Any],
    explicit_keys: set,
    platform: str,
) -> None:
    """V4-Flash TP/DP 拓扑与 A3 专属 CLI 字段。

    max_model_len / max_num_batched_tokens / max_num_seqs / gpu_memory_utilization
    等共用默认由 ascend_default.json 的 DeepSeek-V4-Flash 条目承载，loader 已注入
    engine_config，此处不再重复 force-set。max_model_len 完全由用户控制（JSON 默认 +
    --input-length/--output-length 合成或 config-file 覆盖），不再按平台注入长上下文默认。
    """
    if platform == "a3":
        _force_set_if_not_explicit(engine_config, explicit_keys, "api_server_count", 1)
    default_tp = _default_deepseek_v4_flash_tensor_parallel_size(platform)
    tp_size = _safe_int(engine_config.get("tensor_parallel_size")) or default_tp
    # A2 保持 TP=8；A3 对齐 vllm-ascend 官方 Flash A3 模板，默认 TP=4/DP=4（16 卡）。
    # 用户显式 TP 仍优先，避免覆盖调试或未来官方拓扑变体。
    if "tensor_parallel_size" not in explicit_keys:
        engine_config["tensor_parallel_size"] = default_tp
        params["tensor_parallel_size"] = default_tp
        tp_size = default_tp
    if "data_parallel_size" not in explicit_keys:
        engine_config["data_parallel_size"] = _compute_deepseek_v4_flash_data_parallel_size(
            params, tp_size,
        )
    # MoE 必须开 EP，否则 KV cache spec 形状不一致 + MTP → 'list' has no 'merge'。
    _force_set_if_not_explicit(engine_config, explicit_keys, "enable_expert_parallel", True)


def _apply_deepseek_v4_flash_engine_defaults(
    params: Dict[str, Any],
    engine_config: Dict[str, Any],
    explicit_keys: set,
) -> None:
    """Apply DeepSeek-V4-Flash vLLM-Ascend launch defaults without touching other models.

    Gate 条件必须同时满足：engine 是 vllm_ascend，且身份/架构判定为 V4-Flash。
    函数体只保留运行时派生字段；quantization/block/tokenizer/parser 等静态
    vLLM CLI 默认值由 ascend_default.json 的 DeepSeek-V4-Flash 条目承载。
    """
    if params.get("engine") != "vllm_ascend":
        return
    if not _is_deepseek_v4_flash_params(params):
        return

    platform = _ascend_platform_from_runtime(params)
    # Speculative decoding is controlled only by the upstream SmartFeature
    # switch + whitelist gate. Do not force-enable it from model defaults.
    _apply_deepseek_v4_flash_capacity_and_topology(params, engine_config, explicit_keys, platform)
    _merge_dict_default_if_not_explicit(
        engine_config, explicit_keys, "compilation_config",
        {"cudagraph_mode": "FULL_DECODE_ONLY"},
    )


def _apply_deepseek_v4_pro_engine_defaults(
    params: Dict[str, Any],
    engine_config: Dict[str, Any],
    explicit_keys: set,
) -> None:
    """Apply runtime-only DeepSeek-V4-Pro launch defaults that cannot live in JSON.

    适配范围：仅 A3 双机（nnodes==2、distributed==True）。其它部署形态不进入此分支。
    max_model_len、quantization、compilation_config、additional_config 等静态字段仍由
    ``ascend_default.json`` 的 ``DeepSeek-V4-Pro`` 条目承载。并行拓扑必须依赖运行时
    ``device_count`` / ``nnodes`` 动态推导，避免把官方 16 卡双机示例固化进默认 JSON：

    - TP = device_count（每节点一个 TP replica）
    - DP-local = device_count / TP
    - DP = DP-local * nnodes
    - DP-start-rank = node_rank * DP-local
    """
    if params.get("engine") != "vllm_ascend":
        return
    if not is_deepseek_v4_pro_adapted_scope(params):
        return
    device_count = _safe_int(params.get("device_count"))
    nnodes = _safe_int(params.get("nnodes")) or 1
    node_rank = _safe_int(params.get("node_rank")) or 0
    if not device_count or device_count <= 0:
        raise ValueError(
            "DeepSeek-V4-Pro requires a positive device_count to compute TP/DP topology"
        )

    tp_size = _safe_int(engine_config.get("tensor_parallel_size")) or device_count
    if tp_size <= 0 or device_count % tp_size != 0:
        raise ValueError(
            "DeepSeek-V4-Pro requires device_count to be divisible by tensor_parallel_size: "
            f"device_count={device_count}, tensor_parallel_size={tp_size}"
        )
    dp_size_local = device_count // tp_size
    dynamic_topology = {
        "tensor_parallel_size": tp_size,
        "data_parallel_size": dp_size_local * nnodes,
        "data_parallel_size_local": dp_size_local,
        "data_parallel_start_rank": node_rank * dp_size_local,
    }
    for key, value in dynamic_topology.items():
        if key not in explicit_keys:
            engine_config[key] = value
    logger.info(
        "[DeepSeek-V4-Pro] dynamic topology from device_count/nnodes: "
        "TP=%d, DP=%d, DP-local=%d, DP-start-rank=%d",
        tp_size,
        dynamic_topology["data_parallel_size"],
        dp_size_local,
        dynamic_topology["data_parallel_start_rank"],
    )
    # MTP (enable_speculative_decode) 由上层 CLI/ENV (--enable-speculative-decode)
    # 控制，本路径不再默认强制开启。模型权重虽含 MTP head，但 LMCache、调试场景
    # 不一定希望同时启用投机解码，强制 True 会让用户无法关闭。
    if params.get("rpc_port") in (None, "", 13355, "13355"):
        params["rpc_port"] = 13399
    params["_force_data_parallel_start_rank_on_rank0"] = True


def _apply_deepseek_v4_flash_nv_engine_defaults(
    params: Dict[str, Any],
    engine_config: Dict[str, Any],
    explicit_keys: set,
) -> None:
    """DeepSeek-V4-Flash on rtx_pro_5000_72G + vllm(NVIDIA) 的 TP/DP 动态策略。

    对用户不可见（不由 CLI 传入）：
      - TP = min(4, device_count)
      - DP = device_count / TP，必须整除，否则报错

    NVIDIA vLLM 路径：``_apply_deepseek_v4_flash_engine_defaults`` 对 vllm_ascend 提前
    return，故 NV 的 TP/DP 由本函数接管。判定复用 ``is_deepseek_v4_flash_rtx_pro_5000``，
    与 config_loader._set_parallelism_params 的短路条件逐字一致（config_loader 让位 ⇔
    adapter 接管）。仅当用户未显式指定 TP/DP 时注入（explicit_keys 优先）。
    """
    if not is_deepseek_v4_flash_rtx_pro_5000(params, params.get("engine", "vllm")):
        return

    device_count = _safe_int(params.get("device_count")) or 0
    if device_count <= 0:
        raise ValueError(
            "DeepSeek-V4-Flash(NV) TP/DP 策略需要正的 device_count，"
            "当前为 %s；请通过 --device-count 或 DEVICE_COUNT 指定。" % params.get("device_count")
        )

    tp = min(4, device_count)
    if device_count % tp != 0:
        raise ValueError(
            "DeepSeek-V4-Flash(NV) DP 策略要求 device_count(%d) 能被 TP(%d) 整除，"
            "当前 %d %% %d = %d；请调整 --device-count。" % (device_count, tp, device_count, tp, device_count % tp)
        )
    dp = device_count // tp

    if "tensor_parallel_size" not in explicit_keys:
        engine_config["tensor_parallel_size"] = tp
        params["tensor_parallel_size"] = tp
        logger.info("[DeepSeek-V4-Flash-NV] tensor_parallel_size = %d (min(4, %d))", tp, device_count)
    if "data_parallel_size" not in explicit_keys:
        engine_config["data_parallel_size"] = dp
        params["data_parallel_size"] = dp
        logger.info("[DeepSeek-V4-Flash-NV] data_parallel_size = %d (%d / %d)", dp, device_count, tp)


def _apply_minimax_m27_nvfp4_nv_engine_defaults(
    params: Dict[str, Any],
    engine_config: Dict[str, Any],
    explicit_keys: set,
) -> None:
    """MiniMax-M2.7-NVFP4 on rtx_pro_5000_72G + vllm(NVIDIA) 的 TP/DP 动态策略。

    与 ``_apply_deepseek_v4_flash_nv_engine_defaults`` 同构（TP/DP 公式一致）：
      - TP = min(4, device_count)
      - DP = device_count / TP，必须整除，否则报错

    config_loader._set_parallelism_params 已对 ``is_minimax_m27_rtx_pro_5000_vllm`` 短路
    （不套用通用 TP=device_count 公式），TP/DP 由本函数接管；判定与 config_loader 短路
    共用同一函数，保证「config_loader 让位 ⇔ adapter 接管」逐字一致。仅当用户未显式
    指定 TP/DP 时注入（explicit_keys 优先）。
    """
    if not is_minimax_m27_rtx_pro_5000_vllm(params, params.get("engine", "vllm")):
        return

    device_count = _safe_int(params.get("device_count")) or 0
    if device_count <= 0:
        raise ValueError(
            "MiniMax-M2.7-NVFP4(NV) TP/DP 策略需要正的 device_count，"
            "当前为 %s；请通过 --device-count 或 DEVICE_COUNT 指定。" % params.get("device_count")
        )

    tp = min(4, device_count)
    if device_count % tp != 0:
        raise ValueError(
            "MiniMax-M2.7-NVFP4(NV) DP 策略要求 device_count(%d) 能被 TP(%d) 整除，"
            "当前 %d %% %d = %d；请调整 --device-count。" % (device_count, tp, device_count, tp, device_count % tp)
        )
    dp = device_count // tp

    if "tensor_parallel_size" not in explicit_keys:
        engine_config["tensor_parallel_size"] = tp
        params["tensor_parallel_size"] = tp
        logger.info("[MiniMax-M2.7-NVFP4-NV] tensor_parallel_size = %d (min(4, %d))", tp, device_count)
    if "data_parallel_size" not in explicit_keys:
        engine_config["data_parallel_size"] = dp
        params["data_parallel_size"] = dp
        logger.info("[MiniMax-M2.7-NVFP4-NV] data_parallel_size = %d (%d / %d)", dp, device_count, tp)


# V4-Flash KV 稀疏路径（IndexCache / FLASHMLA_SPARSE_DSV4）要求 block_size 恒为 256，
# 与卡型无关。即使用户显式传入其它值也以 256 覆盖，
# 避免 block_size 不匹配导致启动失败或性能劣化。
_DEEPSEEK_V4_FLASH_NV_BLOCK_SIZE = 256


def _force_deepseek_v4_flash_nv_block_size(
    params: Dict[str, Any],
    engine_config: Dict[str, Any],
) -> None:
    """DeepSeek-V4-Flash + vllm(NVIDIA) 固定 ``--block-size 256``。
    本函数对所有 NVIDIA 卡型生效（``engine == "vllm"``），且强制覆盖用户
    显式值——block_size 是 V4-Flash KV 稀疏路径的硬性约束，不可由 CLI 改写。
    """
    if params.get("engine") != "vllm":
        return
    if not _is_deepseek_v4_flash_params(params):
        return
    prev = engine_config.get("block_size")
    if prev not in (None, _DEEPSEEK_V4_FLASH_NV_BLOCK_SIZE):
        logger.info(
            "[DeepSeek-V4-Flash-NV] 强制 block_size=%d（覆盖原值 %r）；"
            "V4-Flash KV 稀疏路径要求固定 256。",
            _DEEPSEEK_V4_FLASH_NV_BLOCK_SIZE, prev,
        )
    engine_config["block_size"] = _DEEPSEEK_V4_FLASH_NV_BLOCK_SIZE


def _read_lmcache_max_local_cpu_gb() -> Optional[int]:
    """读取页面传入的 ``LMCACHE_MAX_LOCAL_CPU_SIZE``（GB）。

    仅负责 env 解析，不掺入缺省值策略。native offload 会直接复用该 GB 值；
    只有 LMCache env 渲染路径需要把整节点容量折算成每卡容量。

    Returns:
        int: env 已设且合法时的页面容量；``None``: 未设或非法（caller 按各自缺省处理）。
    """
    raw_size = os.getenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "").strip()
    if not raw_size:
        return None
    try:
        return int(raw_size)
    except ValueError:
        logger.warning(
            "[KV Offload] Invalid LMCACHE_MAX_LOCAL_CPU_SIZE=%r; "
            "falling back to per-model default.", raw_size,
        )
        return None


# ── C4: KV 卸载 auto 容量反向预算（需求一 §3.0）────────────────────────────────
# 两路卸载（LMCache / native CPUOffloading）复用同一 M_offload，仅落地单位不同：
#   LMCache  KV_MEM_OFFLOAD_SIZE = M_offload ÷ N_card（均卡/per-card）
#   native   cpu_swap_space_gb / --kv-offloading-size = M_offload（整节点，不除卡数）
_OFFLOAD_ENGINE_SELF_PER_WORKER_GB = 7   # 每 worker 常驻（CANN/torch_npu/.so/激活）线性系数
_OFFLOAD_ENGINE_SELF_BASE_GB = 3         # 固定开销
_OFFLOAD_MARGIN_RATIO = 0.10             # 安全垫
_OFFLOAD_MIN_GB = 100                    # 熔断下限：低于此不建卸载池


def _offload_parallel_size(params: Dict[str, Any], key: str) -> int:
    """从 params 或 engine_config 读取并行度（TP/DP），缺省 1。"""
    val = _safe_int(params.get(key))
    if val:
        return val
    ec = params.get("engine_config")
    if isinstance(ec, dict):
        val = _safe_int(ec.get(key))
        if val:
            return val
    return 1


def resolve_offload_cpu_capacity_gb(
    params: Dict[str, Any],
    size_env_name: str = "KV_MEM_OFFLOAD_SIZE",
) -> Optional[int]:
    """C4：auto 模式反向预算「本节点总」CPU 卸载容量 M_offload (GiB)。需求一 §3.0。

    判定（靠 size_env_name 对应 env 的字面值 =auto；默认 KV_MEM_OFFLOAD_SIZE）：
        <size_env_name> == "auto" 且 AVAILABLE_POD_MEM_SIZE 非空 → auto；
        否则（custom 带 GB 值 / 无 POD_MEM_SIZE）→ 返回 None，调用方走原透传逻辑。

    公式（M_swap=0 由 swap_space=0 原子绑定保证）：
        M_offload = M_container − (7G×TP×DP + 3G) − M_container×10%

    本函数只算「本节点总额」M_offload；两路卸载共用，仅落地单位不同：
        LMCache 落地需再 ÷ N_card（均卡）；native/CPUOffloading 直接用整节点 M_offload。

    Returns:
        None: 非 auto（custom 透传或无 AVAILABLE_POD_MEM_SIZE）→ 调用方按原逻辑处理。
        0:    auto 命中但 M_offload < 熔断下限 → 调用方不建卸载池。
        >0:   auto「本节点总」容量 M_offload。
    """
    max_cpu = os.getenv(size_env_name, "").strip()
    pod_mem = os.getenv("AVAILABLE_POD_MEM_SIZE", "").strip()

    # 判定: ENABLE_KV_MEM_OFFLOAD=true 且 size_env_name 对应 env == "auto" 且 AVAILABLE_POD_MEM_SIZE 非空 → auto 自算
    if os.getenv("ENABLE_KV_MEM_OFFLOAD", "false").strip().lower() != "true":
        return None
    if max_cpu.lower() != "auto":
        return None  # custom 带 GB 值或未设置 → 透传
    if not pod_mem:
        return None
    try:
        # AVAILABLE_POD_MEM_SIZE is supplied by the upper layer in MiB.
        m_container = float(pod_mem) / 1024.0
    except (TypeError, ValueError):
        logger.warning("[KVCache Offload] Invalid AVAILABLE_POD_MEM_SIZE=%r; auto capacity skipped.", pod_mem)
        return None
    tp = _offload_parallel_size(params, "tensor_parallel_size")
    dp = _offload_parallel_size(params, "data_parallel_size")
    m_engine_self = _OFFLOAD_ENGINE_SELF_PER_WORKER_GB * (tp * dp) + _OFFLOAD_ENGINE_SELF_BASE_GB
    m_margin = m_container * _OFFLOAD_MARGIN_RATIO
    m_offload = m_container - m_engine_self - m_margin  # M_swap=0（auto 强制 swap_space=0）
    logger.info(
        "[KVCache Offload] AVAILABLE_POD_MEM_SIZE=%sMB -> %.2fG container memory.",
        pod_mem, m_container,
    )
    if m_offload < _OFFLOAD_MIN_GB:
        return 0
    return int(m_offload)


def _resolve_native_offload_gb(
    params: Dict[str, Any],
    *,
    fallback_gb: Optional[int],
    size_env_names: Tuple[str, ...] = ("KV_MEM_OFFLOAD_SIZE",),
    require_mem_switch: bool = False,
    log_context: str = "KVCache Offload",
) -> int:
    """Resolve node-level native KV offload size while preserving caller fallback policy."""
    if (
        require_mem_switch
        and os.getenv("ENABLE_KV_MEM_OFFLOAD", "false").strip().lower() != "true"
    ):
        logger.info("[%s] native memory offload switch is disabled.", log_context)
        return 0

    selected_env = size_env_names[0] if size_env_names else "KV_MEM_OFFLOAD_SIZE"
    raw_size = ""
    for env_name in size_env_names or ("KV_MEM_OFFLOAD_SIZE",):
        raw_size = os.getenv(env_name, "").strip()
        selected_env = env_name
        if raw_size:
            break

    if raw_size.lower() == "auto":
        auto_total = resolve_offload_cpu_capacity_gb(params, size_env_name=selected_env)
        if auto_total is not None:
            logger.info(
                "[%s] native auto size = M_offload(%dG) (整节点，不除卡数).",
                log_context,
                int(auto_total),
            )
            return int(auto_total)
        if fallback_gb is not None:
            return int(fallback_gb)
        logger.warning(
            "[%s] %s=auto but auto capacity is unavailable; native request discarded.",
            log_context,
            selected_env,
        )
        return 0

    if not raw_size:
        if fallback_gb is not None:
            return int(fallback_gb)
        logger.warning("[%s] %s is empty; native request discarded.", log_context, selected_env)
        return 0

    try:
        size_gb = int(raw_size)
    except (TypeError, ValueError):
        if fallback_gb is not None:
            logger.warning(
                "[%s] Invalid %s=%r; falling back to %d GB.",
                log_context,
                selected_env,
                raw_size,
                fallback_gb,
            )
            return int(fallback_gb)
        logger.warning(
            "[%s] invalid %s=%r; native request discarded.",
            log_context,
            selected_env,
            raw_size,
        )
        return 0
    if size_gb <= 0 and fallback_gb is None:
        logger.warning(
            "[%s] non-positive %s=%s; native request discarded.",
            log_context,
            selected_env,
            size_gb,
        )
        return 0
    return size_gb


def _resolve_v4_flash_offload_gb(params: Dict[str, Any]) -> int:
    """Resolve KV-offload CPU size (GB) shared by ascend ``cpu_swap_space_gb``
    and NV native ``--kv-offloading-size``.

    取值规则（整节点口径，两路径同源同值）：
      * **auto**（KV_MEM_OFFLOAD_SIZE=auto + AVAILABLE_POD_MEM_SIZE 非空）：
        直接用反向预算「本节点总」M_offload，native **不除卡数**（需求一 §3.0）；
      * ``KV_MEM_OFFLOAD_SIZE`` 未设/非法 → 200（默认平铺，不乘）；
      * 自定义值来自页面，单位 GB，native **直接复用**，不按卡数放大。
    """
    return _resolve_native_offload_gb(
        params,
        fallback_gb=200,
        log_context="DeepSeek-V4 KV Offload",
    )


_GLM5_A2_ADDITIONAL_CONFIG: Dict[str, Any] = {
    "fuse_muls_add": True,
    "multistream_overlap_shared_expert": True,
    "ascend_compilation_config": {"enable_npugraph_ex": True},
}


def _apply_glm52_ascend_recipe(params: Dict[str, Any], engine_config: Dict[str, Any],
                               explicit_keys: set) -> bool:
    """GLM-5.2 专属 engine 默认：async_scheduling + enable_expert_parallel 必产；单机
    (nnodes==1, 偶数卡)官方 recipe 全局 DP=2 → TP=device_count//2（覆盖所有 backend，含非
    dp_deployment / 页面未下发 TP）；双机不动 TP，由 _resolve_dp_deployment_topology 推。

    命中 GLM-5.2 返回 True（调用方应提前 return，不进 GLM-5.1 的 EP 默认处理）。单机 TP 能落地
    依赖 config_loader._set_parallelism_params 对 GLM-5.2 单机短路（两处共用 is_glm52_single_node_even，
    须配套，否则 _set_if_not_explicit 只填空值、覆盖不掉被预置的 TP）。
    """
    if not is_glm52_model(params.get("model_name"), params.get("model_path")):
        return False
    _set_if_not_explicit(engine_config, explicit_keys, "async_scheduling", True)
    _set_if_not_explicit(engine_config, explicit_keys, "enable_expert_parallel", True)
    if is_glm52_single_node_even(params):
        _dc = _safe_int(params.get("device_count"))
        _set_if_not_explicit(engine_config, explicit_keys, "tensor_parallel_size", _dc // _GLM52_SINGLE_NODE_DP)
        _set_if_not_explicit(engine_config, explicit_keys, "data_parallel_size", _GLM52_SINGLE_NODE_DP)
    logger.info("[GLM-5.2 Ascend] async_scheduling + EP (+单机 DP=%d) ensured", _GLM52_SINGLE_NODE_DP)
    return True


def _ensure_glm51_ascend_ep_enabled(params: Dict[str, Any], model_info,
                                    engine_config: Dict[str, Any], explicit_keys: set) -> None:
    """GLM-5.1 Ascend keeps EP enabled by default.

    The strict dry-run contract currently emits ``--enable-expert-parallel`` for
    GLM-5.1 Ascend.  This hook preserves explicit user values and fills the
    default ``True`` only when the caller did not set the key.
    """
    if not is_glm51_ascend_kvsparse_tmp_scope(
        model_info, params.get("engine"),
        model_name=params.get("model_name"),
        model_path=params.get("model_path"),
    ):
        return
    if "enable_expert_parallel" not in explicit_keys:
        engine_config["enable_expert_parallel"] = True


def _apply_glm5_ascend_engine_defaults(
    params: Dict[str, Any],
    engine_config: Dict[str, Any],
    explicit_keys: set,
) -> None:
    """[GLM-5/5.1 Ascend] 注入 ``additional_config`` 三键默认值。

    依据 vllm-ascend 官方 W8A8 双机命令（A2/A3 一致）：
      * 传 ``--additional-config '{fuse_muls_add,
        multistream_overlap_shared_expert, ascend_compilation_config.enable_npugraph_ex}'``

    行为：A2 / A3 一致；深合并默认三键，用户显式声明的键值保留。

    GLM-5.1 Ascend 默认保持 ``enable_expert_parallel=True``，与当前假跑输出中的
    ``--enable-expert-parallel`` 保持一致；用户显式配置仍优先。
    """
    if params.get("engine") != "vllm_ascend":
        return
    # PD external-lb：GLM-5 的 PD 参数由 pd_config 注册表控制（官方 GLM5 PD 命令
    # 使用 --enable-expert-parallel），不走非 PD 路径的 additional_config 注入。
    if params.get("_pd_external_lb"):
        return
    try:
        model_info = ModelIdentifier(
            params.get("model_name"),
            params.get("model_path"),
            params.get("model_type"),
        )
    except Exception as exc:
        logger.debug(
            "[GLM-5/5.1 Ascend] Skip additional_config defaults; "
            "ModelIdentifier failed: %s", exc,
        )
        return
    if model_info.model_architecture != "GlmMoeDsaForCausalLM":
        return

    _merge_dict_default_if_not_explicit(
        engine_config,
        explicit_keys,
        "additional_config",
        _GLM5_A2_ADDITIONAL_CONFIG,
    )
    logger.info(
        "[GLM-5/5.1 Ascend] ensure additional_config defaults applied",
    )

    # GLM-5.2 提前收口(async/EP/单机 DP=2)；命中则不进下面仅对 GLM-5.1 的 EP 默认处理。
    if _apply_glm52_ascend_recipe(params, engine_config, explicit_keys):
        return
    # GLM-5.1：默认开启 EP，显式配置优先。
    _ensure_glm51_ascend_ep_enabled(params, model_info, engine_config, explicit_keys)


def _is_deepseek_ascend_dp_deployment(params: Dict[str, Any]) -> bool:
    """Determine whether current launch uses DeepSeek Ascend dp_deployment."""
    return _get_deepseek_ascend_dp_model_architecture(params) is not None


def is_deepseek_ascend_dp_deployment(params: Dict[str, Any]) -> bool:
    """Return True when params target an Ascend dp_deployment architecture."""
    return _is_deepseek_ascend_dp_deployment(params)


# ── GLM-4.7-W8A8 引擎参数注入（仅针对量化变体，避免污染同架构 BF16 模型）──
# 触发条件：架构 == Glm4MoeForCausalLM 且 config.json 量化字段命中 W8A8 别名表
# 合并策略：
#   * 标量字段：用户已显式给出则不覆盖（user > injected）
#   * dict 字段（additional_config / compilation_config）：
#       做 **深合并**，用户给出的 sub-key 优先，未给出的 sub-key 注入
_GLM47_W8A8_ENGINE_DEFAULTS: Dict[str, Any] = {
    "use_vllm_serve": True,
    "data_parallel_size": 2,
    "tensor_parallel_size": 8,
    "enable_expert_parallel": True,
    "async_scheduling": True,
    "quantization": "ascend",
    "seed": 1024,
    "max_model_len": 133000,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 16,
    "gpu_memory_utilization": 0.9,
    "additional_config": {
        # 官方 GLM-4.7-W8A8 强推荐
        "enable_shared_expert_dp": True,
        "ascend_fusion_config": {"fusion_ops_gmmswigluquant": False},
    },
    # 推测解码不在此处承载：完全交由"上层开关 + launcher 自动合成"路径产出，
    # 避免在 ascend 默认 / 架构指纹注入这条"第三入口"上再硬编码任何 spec 字段。
    # 编译图：cudagraph 全量解码模式，覆盖常用并发档位
    "compilation_config": {
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
        "cudagraph_mode": "FULL_DECODE_ONLY",
    },
}

# 需要做 dict 深合并的字段（不能整体覆盖）
_GLM47_W8A8_DEEP_MERGE_KEYS = {
    "additional_config",
    "compilation_config",
}





def _merge_glm47_dict_default(existing: Any, default_val: Dict[str, Any]) -> Glm47DefaultMergeResult:
    """Merge a GLM-4.7 W8A8 dict default."""
    if _is_empty_engine_config_value(existing):
        return Glm47DefaultMergeResult(dict(default_val), "injected")
    parsed_existing = _parse_dict_like_config(existing)
    if parsed_existing is not None:
        merged = _deep_merge_user_priority(parsed_existing, default_val)
        action = "deep_merged" if merged != existing else "unchanged"
        return Glm47DefaultMergeResult(merged if merged != existing else None, action)
    return Glm47DefaultMergeResult(None, "skipped_non_dict")


def _apply_glm47_w8a8_default(
    engine_config: Dict[str, Any],
    key: str,
    default_val: Any,
    explicit_keys: Optional[set] = None,
    force_non_explicit: bool = False,
) -> str:
    """Apply one GLM-4.7 W8A8 default while preserving explicit user values."""
    explicit_keys = explicit_keys or set()
    if key in explicit_keys:
        return "skipped"
    existing = engine_config.get(key)
    if key in _GLM47_W8A8_DEEP_MERGE_KEYS and isinstance(default_val, dict):
        result = _merge_glm47_dict_default(existing, default_val)
        if result.value is not None:
            engine_config[key] = result.value
        return result.action
    if force_non_explicit:
        action = "overridden" if not _is_empty_engine_config_value(existing) else "injected"
        engine_config[key] = default_val
        return action
    if not _is_empty_engine_config_value(existing):
        return "skipped"
    engine_config[key] = default_val
    return "injected"


def _get_glm47_w8a8_model_info(params: Dict[str, Any]) -> Optional[ModelIdentifier]:
    """Return model info when params describe a GLM-4.7 W8A8 vLLM model."""
    engine = params.get("engine", "vllm")
    if engine not in ("vllm", "vllm_ascend"):
        return None

    model_path = params.get("model_path")
    if not model_path:
        return None

    try:
        info = ModelIdentifier(
            params.get("model_name", ""),
            model_path,
            params.get("model_type", "auto"),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[GLM-4.7-W8A8] Skip injection, ModelIdentifier failed: %s", e)
        return None

    if info.model_architecture != "Glm4MoeForCausalLM":
        return None
    return info if _is_w8a8_quantize(info.model_quantize) else None


def _record_glm47_default_action(
    engine_config: Dict[str, Any],
    key: str,
    action: str,
    stats: Glm47InjectionStats,
) -> None:
    """Record and log the result of applying one GLM-4.7 W8A8 default."""
    if action == "injected":
        stats.injected.append(key)
    elif action == "deep_merged":
        stats.deep_merged.append(key)
    elif action == "skipped_non_dict":
        logger.warning(
            "[GLM-4.7-W8A8] %s already present as non-dict (%s); "
            "keeping user value and skipping default injection for this key.",
            key, type(engine_config.get(key)).__name__,
        )
        stats.skipped.append(key)
    elif action == "skipped":
        stats.skipped.append(key)


def _log_glm47_w8a8_summary(
    info: ModelIdentifier,
    engine_config: Dict[str, Any],
    stats: Glm47InjectionStats,
) -> None:
    """Log GLM-4.7 W8A8 engine_config injection summary."""
    if not stats.injected and not stats.deep_merged:
        return
    logger.info(
        "[GLM-4.7-W8A8] Engine config tuning for arch=%s quantize=%s | "
        "injected=%s | deep_merged=%s | user_kept=%s",
        info.model_architecture, info.model_quantize, stats.injected, stats.deep_merged, stats.skipped,
    )
    try:
        summary = {k: engine_config.get(k) for k in _GLM47_W8A8_ENGINE_DEFAULTS.keys()}
        logger.info(
            "[GLM-4.7-W8A8] Final engine_config for tuned keys:\n%s",
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[GLM-4.7-W8A8] Skip summary dump: %s", e)


def _inject_glm47_w8a8_engine_config(params: Dict[str, Any], force_non_explicit: bool = False) -> None:
    """检测 GLM-4.7-W8A8 模型，**就地**向 engine_config 追加调优默认字段。

    设计要点：
      * 仅当 (架构 == Glm4MoeForCausalLM) 且 (quantize 命中 W8A8) 时触发
      * 标量字段：用户优先；dict 字段：深合并，用户的 sub-key 优先
      * BF16 / 同架构非量化变体（如 GLM-4.5）不会被影响
      * 仅对 vllm / vllm_ascend 引擎生效
      * 不承载推测解码：spec 完全交给"上层开关 + launcher 自动合成"两入口
    """
    info = _get_glm47_w8a8_model_info(params)
    if info is None:
        return

    engine_config = params.setdefault("engine_config", {})
    explicit_keys = set(params.get("_explicit_cli_keys") or [])
    stats = Glm47InjectionStats()
    for key, default_val in _GLM47_W8A8_ENGINE_DEFAULTS.items():
        action = _apply_glm47_w8a8_default(
            engine_config, key, default_val, explicit_keys, force_non_explicit,
        )
        _record_glm47_default_action(engine_config, key, action, stats)

    _log_glm47_w8a8_summary(info, engine_config, stats)


def _build_vllm_cmd_parts(params: Dict[str, Any]) -> str:
    """构建 vLLM 核心启动命令字符串。

    将 engine_config 字典转换为 vLLM CLI 参数格式：
    python3 -m vllm.entrypoints.openai.api_server --arg1 value1 ...

    Args:
        params: 参数字典，必须包含 engine_config 字典

    Returns:
        str: 完整的 vLLM 启动命令字符串
    """
    engine_config = _prepare_engine_config(params)
    use_vllm_serve = bool(engine_config.pop("use_vllm_serve", False))
    extra_cli_args = engine_config.pop("extra_cli_args", [])
    if use_vllm_serve:
        model_value = engine_config.pop("model", params.get("model_path", "/weights"))
        cmd_parts = ["vllm", "serve", shlex.quote(str(model_value))]
    else:
        cmd_parts = ["python3", "-m", "vllm.entrypoints.openai.api_server"]

    for arg, value in engine_config.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if arg == "max_num_batched_tokens":
            try:
                if int(value) <= 0:
                    logger.warning("Skip invalid max_num_batched_tokens=%s; vLLM requires >=1", value)
                    continue
            except (TypeError, ValueError):
                logger.warning("Skip non-integer max_num_batched_tokens=%s", value)
                continue

        arg_name = f"--{arg.replace('_', '-')}"
        cmd_parts.extend(_format_cli_arg(arg_name, value))

    allowed_raw_args = {"-cc.pass_config.fuse_allreduce_rms=False"}
    if isinstance(extra_cli_args, str):
        extra_cli_args = [extra_cli_args]
    for raw_arg in extra_cli_args or ():
        if raw_arg in allowed_raw_args:
            cmd_parts.append(raw_arg)
        else:
            logger.warning("[vLLM] rejected unapproved raw CLI arg: %r", raw_arg)

    return " ".join(cmd_parts)


# ── 推测解码 (Speculative Decoding) ──────────────────────────────────────


def _normalize_speculative_draft_path(value: Any) -> str:
    """Return a real draft path, or empty string for no-draft sentinel values."""
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in ("none", "null"):
        return ""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        inner = raw[1:-1].strip()
        if not inner or inner.lower() in ("none", "null"):
            return ""
    return raw


def _is_mtp_or_suffix_strategy(params: Dict[str, Any], engine: str) -> bool:
    """判断当前投机推理策略是否为 MTP 或 suffix（即非草稿模型方案）。

    当启用投机推理且未指定草稿模型路径时，策略一定是 MTP 或 suffix。
    vllm_ascend 不注入 VLLM_EARS_TOLERANCE，Ascend 侧无需该参数。
    MiniMax-M2.7 (NV) 不注入 VLLM_EARS_TOLERANCE（与 minimax-2.7-zyy.txt 对齐，无此项）。
    """
    if not params.get("enable_speculative_decode"):
        return False
    if engine != "vllm":
        return False
    if is_minimax_m27_rtx_pro_5000_vllm(params, engine):
        return False
    if _normalize_speculative_draft_path(params.get("speculative_decode_model_path")):
        return False
    return True


def _build_speculative_env_commands(params: Dict[str, Any], engine: str) -> List[str]:
    """构建 MTP / suffix 投机推理策略所需的环境变量命令。

    当投机推理采用 MTP 或 suffix 策略时，默认注入
    ``VLLM_EARS_TOLERANCE=0.5`` 环境变量以控制容忍度参数。

    Args:
        params: 参数字典
        engine: 引擎类型

    Returns:
        List[str]: 环境变量设置命令列表，未启用时返回空列表
    """
    if not _is_mtp_or_suffix_strategy(params, engine):
        return []
    logger.info("[AdvFeature-SpecDecode] MTP/suffix strategy detected, "
                "injecting VLLM_EARS_TOLERANCE=0.5")
    return ['export VLLM_EARS_TOLERANCE=0.5']


def _build_trace_env_commands(params: Dict[str, Any], engine: str) -> List[str]:
    """构建 OTLP trace 相关的环境变量命令。

    仅在 enable_otlp_traces 为 True 时生效。
    仅对 vllm / vllm_ascend 引擎生效。

    Args:
        params: 参数字典
        engine: 引擎类型

    Returns:
        List[str]: 环境变量设置命令列表，未启用时返回空列表
    """
    if not params.get("enable_otlp_traces"):
        return []
    if engine not in ("vllm", "vllm_ascend"):
        return []

    logger.info("[AdvFeature-EnableTrace] vLLM OTLP trace env commands injected")
    return [
        'export OTEL_SERVICE_NAME="vllm-server"',
        'export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true',
        'export OTEL_RESOURCE_ATTRIBUTES="service.instance.id=$(hostname),k8s.pod.name=$(hostname)"',
    ]


# Qwen3.5 系列（混合/线性注意力）架构集合：dense 27B 与 MoE 397B-A17B。
# 投机方法按 engine 收口：vLLM 原生用裸 "mtp"，vLLM-Ascend 官方模板用 "qwen3_5_mtp"
_QWEN35_ARCHES = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)


def _is_qwen35_arch(model_architecture: Optional[str]) -> bool:
    """Return whether the architecture is a Qwen3.5 (hybrid linear-attn) family model."""
    return model_architecture in _QWEN35_ARCHES


def _resolve_mtp_method(model_architecture: str, engine: str) -> str:
    # Qwen3.5 系列按 engine 收口（见 _QWEN35_ARCHES 注释）：
    #   vllm        → "mtp"          （NV 原生 vLLM 注册的方法名）
    #   vllm_ascend → "qwen3_5_mtp"  （vLLM-Ascend 官方模板要求）
    if _is_qwen35_arch(model_architecture):
        return "mtp" if engine == "vllm" else "qwen3_5_mtp"
    mtp_methods_by_arch = {
        "DeepseekV3ForCausalLM": "mtp",
        "DeepseekV32ForCausalLM": "mtp",
        # DeepSeek-V4 (Flash/Pro) 在 vLLM-Ascend 官方模板中使用 deepseek_mtp。
        "DeepseekV4ForCausalLM": "deepseek_mtp",
        "GlmMoeDsaForCausalLM": "deepseek_mtp",
        "Qwen3NextForCausalLM": "qwen3_next_mtp",
        "Glm4MoeForCausalLM": "glm4_moe_mtp",
    }
    return mtp_methods_by_arch.get(model_architecture, "")


def _resolve_whitelist_mtp_num_speculative_tokens(params: Dict[str, Any], engine: str) -> Optional[int]:
    """从 spec 白名单场景行读取逐场景 MTP token 数。

    MTP token 数属于模型+芯片场景契约。把它放在白名单行上，可以避免
    adapter 内部再维护一张容易漂移的局部表；新增 Day0 场景行，或同一模型
    在不同芯片上使用不同 token 数时（例如 Qwen3.5-27B 的 910C 与 910B），
    也能保持单一事实源。
    """
    row = resolve_feature_whitelist_row_from_params(params, engine, "spec")
    if not row:
        return None
    raw_tokens = row.get("mtp_num_speculative_tokens")
    if raw_tokens is None:
        return None
    try:
        tokens = int(raw_tokens)
    except (TypeError, ValueError):
        logger.warning(
            "[AdvFeature-SpecDecode] Invalid mtp_num_speculative_tokens=%r in spec whitelist row.",
            raw_tokens,
        )
        return None
    if tokens <= 0:
        logger.warning(
            "[AdvFeature-SpecDecode] Non-positive mtp_num_speculative_tokens=%s in spec whitelist row.",
            tokens,
        )
        return None
    return tokens


def _lmcache_requires_suffix_speculative_strategy(
    params: Dict[str, Any],
    engine: str,
    model_info: ModelIdentifier,
) -> bool:
    """Return True when effective LMCache offload should force suffix over MTP."""
    if not get_lmcache_env():
        return False
    if not _is_offload_feature_effective(params, engine):
        return False
    if lmcache_auto_floor_disables_all_backends(params):
        logger.info(
            "[KVCache Offload] auto memory offload capacity below floor and no "
            "LMCache backend is active; keeping mtp speculative strategy."
        )
        return False
    if _resolve_offload_backend(params, engine)[0] == _OFFLOAD_NATIVE_BACKEND_VARIANT:
        logger.info("[KVCache Offload] native KV offload coexists with MTP.")
        return False
    if _is_glm51_nvidia_vllm_params(params, engine, model_info):
        logger.warning(
            "[KVCache Offload] Forced disabled for GLM-5.1 on NVIDIA/vLLM; "
            "ignoring ENABLE_KV_OFFLOAD for speculative strategy selection."
        )
        return False
    if engine == "vllm_ascend" and _is_deepseek_v4_flash_params(params, model_info):
        logger.info(
            "[KVCache Offload] DeepSeek-V4-Flash uses LMCacheAscendConnectorV1Dynamic "
            "(coexists with MTP); keeping mtp speculative strategy."
        )
        return False
    if engine == "vllm" and _is_deepseek_v4_flash_params(params, model_info):
        logger.info(
            "[KVCache Offload] DeepSeek-V4-Flash (NV) uses native KV offload "
            "(coexists with MTP); keeping mtp speculative strategy."
        )
        return False
    if engine == "vllm" and is_qwen3_5_397b_nvfp4_vllm(params, engine):
        logger.info(
            "[KVCache Offload] Qwen3.5-397B-A17B-NVFP4 (NV) uses native KV offload "
            "(coexists with MTP); keeping mtp speculative strategy."
        )
        return False
    if engine == "vllm_ascend" and memcache_hybrid.is_memcache_hybrid_params(params, engine):
        # 标记为 "MTP+MemCache" 的 Qwen Day0 优化行走 AscendStoreConnector，
        # 不是历史上会强制切到 suffix 的 LMCacheConnector。这里保留 MTP，
        # 让 spec 白名单中的逐场景 mtp_num_speculative_tokens 仍能落到最终
        # --speculative-config。
        logger.info(
            "[KVCache Offload] MemCache Hybrid coexists with MTP; "
            "keeping mtp speculative strategy."
        )
        return False
    return True


def resolve_speculative_strategy(params: Dict[str, Any], engine: str) -> str:
    """Return the speculative decoding strategy selected for vLLM."""
    if engine not in ("vllm", "vllm_ascend"):
        return ""

    mtp_row = _mtp_whitelist_override_row(params, engine)
    if mtp_row and params.get("enable_speculative_decode"):
        smart_feats = params.get("_smart_feats")
        if smart_feats is None or "spec" in smart_feats:
            return str(mtp_row.get("mtp_method") or "mtp")

    draft_path = params.get("speculative_decode_model_path")
    normalized_draft_path = _normalize_speculative_draft_path(draft_path)
    _draft_raw = str(draft_path).strip().lower() if draft_path else ""
    logger.info(
        "[SpecDecode-DIAG] resolve_speculative_strategy entry: "
        "raw_draft_path=%r stripped_lower=%r engine=%s",
        draft_path, _draft_raw, engine,
    )
    # "none" / 空串 / None 均视为「无草稿模型」，回落 MTP/suffix 路径。
    # K8s ConfigMap 常以 SPECULATIVE_DECODE_MODEL_PATH=none 表示未指定，
    # 但字符串 "none" 为 truthy，会被误判为有效草稿模型路径导致生成 draft_model 配置。
    if normalized_draft_path:
        logger.info(
            "[SpecDecode-DIAG] draft_path is real (not none/empty) → entering draft_model branch"
        )
        draft_model_info = ModelIdentifierDraft(normalized_draft_path)
        if 'eagle3' in draft_model_info.draft_model_architecture.lower():
            return "eagle3"
        return "draft_model"
    if draft_path:
        logger.info(
            "[SpecDecode-DIAG] draft_path=%r filtered as 'none' → falling through to MTP/suffix",
            draft_path,
        )
    else:
        logger.info(
            "[SpecDecode-DIAG] draft_path is None/empty → falling through to MTP/suffix"
        )
    model_info = ModelIdentifier(
        params.get("model_name"),
        params.get("model_path"),
        params.get("model_type"),
    )
    if model_info.model_architecture == "Qwen3NextForCausalLM" and engine == "vllm_ascend":
        return "suffix"

    mtp_method = _resolve_mtp_method(model_info.model_architecture, engine)
    if mtp_method:
        # §2.3 白名单 gate：spec 不在白名单 → suffix 地板（恒产 suffix，不返回空）。
        #   GLM-5.1·Ascend 命中 spec 白名单后使用自身 MTP 方法 deepseek_mtp；
        #   未命中时仍回落 suffix，避免非白名单模型误产 MTP。
        #   优先复用 C14 收口（hardware_env 解析卡型最准）stash 的白名单结论；adapter 内拿不到
        #   hardware_env 时才回退 resolve_card_token() 读取 hardware_info.json。
        smart_feats = params.get("_smart_feats")
        if smart_feats is not None:
            spec_ok = "spec" in smart_feats
        else:
            spec_ok = feature_allowed(
                engine, params.get("model_name"), params.get("model_path"),
                params.get("_smart_card_token") or resolve_card_token(), "spec",
            )
        if not spec_ok:
            logger.info("[SpecDecode] spec not in whitelist -> suffix floor (arch=%s)",
                        model_info.model_architecture)
            return "suffix"
        
        # Only an actually blocking LMCache backend should downgrade MTP to suffix.
        return "suffix" if _lmcache_requires_suffix_speculative_strategy(
            params, engine, model_info,
        ) else mtp_method

    return "suffix"


def _format_speculative_result(config_entries: List[str], compact: bool = False) -> str:
    """将推测解码配置列表格式化为 --speculative-config 命令行参数。"""
    body = "{" + ", ".join(config_entries) + "}"
    if compact:
        try:
            body = json.dumps(json.loads(body), ensure_ascii=False, separators=(",", ":"))
        except json.JSONDecodeError:
            body = "{" + ",".join(config_entries) + "}"
    result = f" --speculative-config '{body}'"
    logger.info("[AdvFeature-SpecDecode] Generated params: %s", result.strip())
    return result


def _handle_draft_model_case(params: Dict[str, Any], config: List[str]) -> None:
    """处理有草稿模型的推测解码配置"""
    draft_path = _normalize_speculative_draft_path(params.get("speculative_decode_model_path"))
    # 对路径中的双引号和反斜杠进行 JSON 转义，防止 JSON-in-shell 注入
    safe_path = draft_path.replace('\\', '\\\\').replace('"', '\\"')
    config.append(f'"model": "{safe_path}"')
    config.append('"draft_tensor_parallel_size": 1')
    draft_model_info = ModelIdentifierDraft(draft_path)

    if 'eagle3' in draft_model_info.draft_model_architecture.lower():
        logger.info('--- Using the Eagle3 speculative decoding approach ---')
        config.append('"method" : "eagle3"')
        num_spec_tokens = 4
        config.append(f'"num_speculative_tokens": {num_spec_tokens}')
    else:
        logger.info('--- Using the draft model speculative decoding approach ---')
        config.append('"method" : "draft_model"')
        num_spec_tokens = 4
        config.append(f'"num_speculative_tokens": {num_spec_tokens}')


def _handle_mtp_case(model_info: ModelIdentifier, mtp_support_models: List[Any],
                     mtp_types: List[str], config: List[str]) -> None:
    """处理 MTP 推测解码配置"""
    logger.info('--- Using the MTP speculative decoding approach ---')

    for i, model_group in enumerate(mtp_support_models):
        if model_info.model_architecture in model_group:
            config.append(f'"method": "{mtp_types[i]}"')
            break
    # MTP 强制 num_speculative_tokens=3（官方 GLM-4.7 / DeepSeek-V3 推荐值）
    config.append('"num_speculative_tokens": 3')


def _handle_suffix_case(config: List[str]) -> None:
    """处理 suffix 推测解码配置"""
    logger.info('--- Using the suffix speculative decoding approach ---')
    config.append('"method" : "suffix"')
    config.append('"num_speculative_tokens": 5')
    config.append('"suffix_decoding_max_cached_requests": 1000')


def _normalize_mtp_strategy(
    params: Dict[str, Any],
    engine: str,
    model_info: ModelIdentifier,
    strategy: str,
) -> Tuple[str, bool, bool, bool]:
    """归一化 MTP 方法，并返回后续 token/eager 判定所需场景标记。"""
    is_v4_flash = _is_deepseek_v4_flash_params(params, model_info)
    is_v4_flash_pro5000 = (
        engine == "vllm"
        and is_deepseek_v4_flash_rtx_pro_5000(params, engine)
    )
    is_qwen35_nvfp4_native = is_qwen3_5_397b_nvfp4_vllm(params, engine)
    if (
        model_info.model_architecture == "Glm4MoeForCausalLM"
        and _is_w8a8_quantize(model_info.model_quantize)
    ):
        strategy = "mtp"
    # V4-Flash 的官方模板在 NVIDIA 和 Ascend 上都使用裸 ``mtp``。
    if engine in {"vllm", "vllm_ascend"} and strategy.endswith("_mtp") and is_v4_flash:
        strategy = "mtp"
    return strategy, is_v4_flash, is_v4_flash_pro5000, is_qwen35_nvfp4_native


def _resolve_mtp_speculative_token_count(
    params: Dict[str, Any],
    engine: str,
    model_info: ModelIdentifier,
    is_v4_flash: bool,
    is_v4_flash_pro5000: bool,
) -> int:
    """按白名单和模型配方优先级计算 MTP token 数。"""
    whitelist_tokens = _resolve_whitelist_mtp_num_speculative_tokens(params, engine)
    if whitelist_tokens is not None:
        return whitelist_tokens
    if is_v4_flash_pro5000:
        return 2

    glm52_ascend = engine == "vllm_ascend" and is_glm52_model(
        params.get("model_name"),
        params.get("model_path"),
    )
    glm51_ascend = engine == "vllm_ascend" and is_glm_moe_dsa_glm51(
        model_info,
        model_name=params.get("model_name"),
        model_path=params.get("model_path"),
    )
    uses_single_token = (
        _is_deepseek_v4_pro_params(params)
        or is_v4_flash
        or (
            model_info.model_architecture == "GlmMoeDsaForCausalLM"
            and not glm51_ascend
        )
    )
    return 1 if not glm52_ascend and uses_single_token else 3


def _build_mtp_speculative_cmd(
    params: Dict[str, Any],
    engine: str,
    model_info: ModelIdentifier,
    strategy: str,
) -> str:
    """构建已经选定 MTP 策略后的 speculative-config。"""
    row = _mtp_whitelist_override_row(params, engine)
    if row:
        tokens = _resolve_whitelist_mtp_num_speculative_tokens(params, engine)
        if tokens is None:
            logger.warning(
                "[Speculative Decode] MTP whitelist row has no valid token count; skipping."
            )
            return ""
        method = str(row.get("mtp_method") or "mtp")
        config = [
            f'"method":"{method}"',
            f'"num_speculative_tokens":{tokens}',
        ]
        moe_backend = row.get("mtp_moe_backend")
        if moe_backend:
            config.append(f'"moe_backend":"{moe_backend}"')
        return _format_speculative_result(config, compact=True)

    strategy, is_v4_flash, is_v4_flash_pro5000, is_qwen35_nvfp4_native = (
        _normalize_mtp_strategy(params, engine, model_info, strategy)
    )
    token_count = _resolve_mtp_speculative_token_count(
        params,
        engine,
        model_info,
        is_v4_flash,
        is_v4_flash_pro5000,
    )
    config = [
        f'"method": "{strategy}"',
        f'"num_speculative_tokens": {token_count}',
    ]

    # Qwen3.5 MTP 与非 Pro5000 的 V4-Flash 只让投机头退回 eager，
    # 不影响主模型顶层的图编译策略。
    if (
        (_is_qwen35_arch(model_info.model_architecture) and not is_qwen35_nvfp4_native)
        or (is_v4_flash and not is_v4_flash_pro5000)
    ):
        config.append('"enforce_eager": true')
    if (
        model_info.model_architecture == "Glm4MoeForCausalLM"
        and _is_w8a8_quantize(model_info.model_quantize)
    ):
        config.append('"speculative_token_range": "256,512"')
    return _format_speculative_result(config, compact=is_qwen35_nvfp4_native)


def _build_speculative_cmd(params: Dict[str, Any], engine: str) -> str:
    """推测解码方案的自动选取。

    根据模型架构自动选择最优的推测解码策略：
    1. 如有草稿模型 → eagle3 / draft_model
    2. Qwen3NextForCausalLM + vllm_ascend → suffix
    3. DeepSeek/GLM-5/Qwen3Next/Glm4Moe → MTP
    4. 其他 → suffix

    Args:
        params: 参数字典
        engine: 引擎类型 ('vllm' 或 'vllm_ascend')

    Returns:
        str: --speculative-config 参数字符串，未启用时返回空字符串
    """
    model_info = ModelIdentifier(params.get("model_name"),
                                 params.get("model_path"),
                                 params.get("model_type"))
    logger.info("[AdvFeature-SpecDecode] Model architecture detection: %s (model_name=%s)",
                model_info.model_architecture, params.get("model_name"))

    strategy = resolve_speculative_strategy(params, engine)
    if not strategy:
        logger.info("[AdvFeature-SpecDecode] engine='%s' does not support speculative decode, skipping", engine)
        return ""

    spec_draft_raw = params.get("speculative_decode_model_path")
    normalized_draft_path = _normalize_speculative_draft_path(spec_draft_raw)
    logger.info(
        "[SpecDecode-DIAG] _build_speculative_cmd entry: "
        "enable_spec_decode=%s draft_path=%r engine=%s strategy=%s engine_has_spec_config=%s",
        params.get("enable_speculative_decode"), spec_draft_raw, engine, strategy,
        bool((params.get("engine_config") or {}).get("speculative_config")),
    )

    mtp_row = _mtp_whitelist_override_row(params, engine)
    if spec_draft_raw and not mtp_row:
        if normalized_draft_path:
            logger.info("[AdvFeature-SpecDecode] Draft model path detected: %s, using draft_model strategy",
                        normalized_draft_path)
            speculative_config_temp = []
            _handle_draft_model_case(params, speculative_config_temp)
            return _format_speculative_result(speculative_config_temp)
        logger.info("[AdvFeature-SpecDecode] Draft model path is '%s' — treated as no draft model, "
                    "falling through to strategy='%s'",
                    spec_draft_raw, strategy)

    if strategy == "suffix":
        logger.info("[AdvFeature-SpecDecode] Architecture %s → suffix strategy",
                    model_info.model_architecture)
        speculative_config_temp = []
        _handle_suffix_case(speculative_config_temp)
        return _format_speculative_result(speculative_config_temp)

    if strategy == "mtp" or strategy.endswith("_mtp"):
        logger.info("[AdvFeature-SpecDecode] Architecture %s → MTP strategy (%s)",
                    model_info.model_architecture, strategy)
        return _build_mtp_speculative_cmd(params, engine, model_info, strategy)

    return ""


def build_speculative_cmd(params: Dict[str, Any], engine: str) -> str:
    """Build the speculative decoding CLI fragment."""
    return _build_speculative_cmd(params, engine)


def resolve_effective_speculative_details(
    params: Dict[str, Any], engine: str,
) -> Optional[Dict[str, Any]]:
    """Return explicit MTP details from the same whitelist row used by the CLI."""
    if not params.get("enable_speculative_decode"):
        return None
    smart_feats = params.get("_smart_feats")
    if smart_feats is not None and "spec" not in smart_feats:
        return None
    row = _mtp_whitelist_override_row(params, engine)
    if not row:
        return None
    tokens = _resolve_whitelist_mtp_num_speculative_tokens(params, engine)
    if tokens is None:
        return None
    return {
        "method": str(row.get("mtp_method") or "mtp"),
        "num_speculative_tokens": tokens,
        "moe_backend": row.get("mtp_moe_backend"),
    }


def _should_append_auto_speculative_config(params: Dict[str, Any]) -> bool:
    """白名单：仅在 enable_speculative_decode=True 且 engine_config.speculative_config
    不存在时让 launcher 合成 spec config。

    设计：投机推理仅两个入口——
      (1) 上层显式 ``engine_config.speculative_config`` dict → 命中第一条 return False
      (2) 上层 ``enable_speculative_decode=True`` 开关 → launcher 自动合成
    ascend_default.json / 架构指纹注入 一律不再承载 spec 字段。
    """
    if not params.get("enable_speculative_decode"):
        return False
    engine_config = params.get("engine_config") or {}
    return not bool(engine_config.get("speculative_config"))


def should_append_auto_speculative_config(params: Dict[str, Any]) -> bool:
    """Return True when launcher should synthesize speculative_config itself."""
    return _should_append_auto_speculative_config(params)


# ── KV Sparse（IndexCache / FP8 KV CACHE）───────────────────────────────

# 当 enable_sparse=true 时，根据模型架构决定 KV 稀疏策略：
#   - INDEXCACHE_ARCHS 中的架构 → IndexCache 加速
#   - 其他架构 → FP8 KV CACHE 量化


def _resolve_sparse_level() -> str:
    """解析 SmartKVSparse 有效精度/性能档位（需求一 §2.4）。"""
    return get_sparse_level_env()


def _resolve_sparse_topk(params: Dict[str, Any], engine: str, sparse_level: str, default: int) -> int:
    card_token = params.get("_smart_card_token") or resolve_card_token()
    return resolve_sparse_topk(
        engine,
        params.get("model_name"),
        params.get("model_path"),
        card_token,
        sparse_level,
        default=default,
    )



def _build_kv_sparse_cmd(params: Dict[str, Any], engine: str) -> str:
    """构建 KV 稀疏特性的启动命令参数。

    vllm (NVIDIA) 完整支持；vllm_ascend 仅 GLM-5.1 走 IndexCache（临时白名单）。
    根据模型架构决定策略：
      - vllm + IndexCache 架构（GlmMoeDsa/DeepseekV32）：返回 --hf-overrides CLI 参数
      - vllm + 其他架构：直接修改 engine_config 注入 kv_cache_dtype=fp8，返回空字符串
      - vllm_ascend + GLM-5.1（单机/双机）：返回 --hf-overrides；其他 ascend 场景返回空串
        参见 [GLM5.1-Ascend-Tmp]，等 vllm-ascend 支持 indexcache 补丁后合并入 vllm 主分支。

    **必须在 _build_vllm_cmd_parts 之前调用**，以便 FP8 参数正确合入基础命令，
    避免与 engine_config 中已有的 kv_cache_dtype 产生重复。

    Args:
        params: 参数字典（FP8 路径会就地修改 engine_config）
        engine: 引擎类型

    Returns:
        str: 额外的 CLI 参数字符串（IndexCache 返回 --hf-overrides，FP8 返回空串）
    """
    if engine not in ("vllm", "vllm_ascend"):
        return ""

    # 需求一 §2.4/P5：档位由 SPARSE_LEVEL 决定，topk 来自 sparse 独立白名单行。
    sparse_level = _resolve_sparse_level()
    logger.info("[KV Sparse] effective SPARSE_LEVEL=%s (engine=%s)", sparse_level, engine)

    model_info = ModelIdentifier(
        params.get("model_name"),
        params.get("model_path"),
        params.get("model_type"),
    )
    arch = model_info.model_architecture
    sparse_row = resolve_feature_whitelist_row_from_params(params, engine, "sparse")
    if sparse_row and sparse_row.get("strategy") == "indexcache":
        topk = _resolve_sparse_topk(params, engine, sparse_level, default=4)
        logger.info(
            "[KV Sparse] sparse whitelist -> IndexCache topk=%s",
            topk,
        )
        return (
            " --hf-overrides "
            f"'{{\"use_index_cache\":true,\"index_topk_freq\":{topk}}}'"
        )

    # [GLM5.1-Ascend-Tmp] vllm_ascend 路径：
    # 仅 GLM-5.1 走 IndexCache，不写 engine_config，
    # 不触发 indexcache 补丁安装（补丁仍由 _collect_indexcache_patch_features 的
    # engine 门控屏蔽于 ascend 之外）。
    if engine == "vllm_ascend":
        if is_glm51_ascend_kvsparse_tmp_scope(
            model_info, engine,
            model_name=params.get("model_name"),
            model_path=params.get("model_path"),
        ):
            logger.info(
                "[GLM5.1-Ascend-Tmp] vllm_ascend + GLM-5.1 → "
                "IndexCache via --hf-overrides (no patch install)"
            )
            topk = _resolve_sparse_topk(params, engine, sparse_level, default=8)
            return f" --hf-overrides '{{\"use_index_cache\": true, \"index_topk_freq\": {topk}}}'"
        logger.info(
            "[KV Sparse] engine=vllm_ascend arch=%s not GLM-5.1; "
            "KV sparse is no-op on ascend", arch,
        )
        return ""

    # [V4-Flash-NV-Day0] NV V4-Flash 走 IndexCache（use_index_cache），引擎内置、不装补丁。
    # 刻意不把 DeepseekV4ForCausalLM 加入 INDEXCACHE_ARCHS，使 _collect_indexcache_patch_features
    # 因架构不在白名单天然返回 []，从而跳过 indexcache 补丁安装。
    if _is_deepseek_v4_flash_params(params, model_info):
        logger.info("[KV Sparse] DeepSeek-V4-Flash (NV) → IndexCache use_index_cache "
                    "(--hf-overrides, no patch install)")
        topk = _resolve_sparse_topk(params, engine, sparse_level, default=4)
        return f" --hf-overrides '{{\"use_index_cache\": true, \"index_topk_freq\": {topk}}}'"

    # [MiniMax-M2.7-NVFP4-NV] kv_cache_dtype=fp8 已由 nvidia_default.json 提供，
    # 跳过通用 FP8 路径注入的 calculate_kv_scales（与 minimax-2.7-zyy.txt 对齐，无此项）。
    if is_minimax_m27_rtx_pro_5000_vllm(params, engine):
        logger.info("[KV Sparse] MiniMax-M2.7 (NV) -> no-op (fp8 from json, no calculate_kv_scales)")
        return ""

    if arch in INDEXCACHE_ARCHS:
        logger.info("[KV Sparse] Architecture %s → IndexCache strategy (--hf-overrides)", arch)
        topk = _resolve_sparse_topk(params, engine, sparse_level, default=4)
        return f" --hf-overrides '{{\"index_topk_freq\": {topk}}}'"
    else:
        logger.info("[KV Sparse] Architecture %s → FP8 KV CACHE strategy (kv_cache_dtype=fp8)", arch)
        engine_config = params.setdefault("engine_config", {})
        engine_config["kv_cache_dtype"] = "fp8"
        engine_config["calculate_kv_scales"] = True
        return ""


def resolve_sparse_variant(params: Dict[str, Any], engine: str) -> str:
    """纯函数：返回稀疏 variant 名（advanced_features.json 监控用，无副作用）。

    镜像 ``_build_kv_sparse_cmd`` 的分支（需求一 §4.2）。⚠ 二者须同步修改。
    与产出口的区别：本函数**不修改 engine_config**（fp8 分支仅报名，无副作用）。
    """
    if engine not in ("vllm", "vllm_ascend"):
        return ""                        # none（engine 否决）
    model_info = ModelIdentifier(params.get("model_name"), params.get("model_path"),
                                 params.get("model_type"))
    arch = model_info.model_architecture
    sparse_level = get_sparse_level_env()
    sparse_row = resolve_feature_whitelist_row_from_params(params, engine, "sparse")
    if sparse_row and sparse_row.get("strategy") == "indexcache":
        topk = _resolve_sparse_topk(params, engine, sparse_level, default=4)
        return f"indexcache_use_index_cache_topk{topk}"
    if engine == "vllm_ascend":
        if is_glm51_ascend_kvsparse_tmp_scope(
            model_info, engine,
            model_name=params.get("model_name"), model_path=params.get("model_path"),
        ):
            topk = _resolve_sparse_topk(params, engine, sparse_level, default=8)
            return f"indexcache_use_index_cache_topk{topk}"
        return "noop"                    # Ascend 非 GLM-5.1
    if _is_deepseek_v4_flash_params(params, model_info):
        topk = _resolve_sparse_topk(params, engine, sparse_level, default=4)
        return f"indexcache_use_index_cache_topk{topk}"
    # [MiniMax-M2.7-NVFP4-NV] 同步 _build_kv_sparse_cmd：no-op（fp8 来自 json，无 calculate_kv_scales）
    if is_minimax_m27_rtx_pro_5000_vllm(params, engine):
        return "noop"
    if arch in INDEXCACHE_ARCHS:
        topk = _resolve_sparse_topk(params, engine, sparse_level, default=4)
        return f"indexcache_topk{topk}"
    return "fp8"



def _build_kv_offload_cmd(params: Dict[str, Any], engine: str) -> str:
    """构建 NVIDIA/vLLM native KV 卸载 CLI 片段。

    - 仅 ``engine == "vllm"`` 且模型命中特例时生效（Ascend 0.21 走 LMCache dynamic）。
    - 复用 ``ENABLE_KV_OFFLOAD`` 总开关（get_lmcache_env）作为触发条件。
    - size 按模型复用对应 native 容量解析器。
    - 与 LMCache env 路径互斥：命中时 ``_build_cache_env_commands`` 跳过 LMCache 导出。
    - fallback 时由 ``_wings_fallback_no_kv_offload`` 抑制（崩溃回退退回基线命令）。
    """
    if engine != "vllm":
        return ""
    if params.get("_wings_fallback_no_kv_offload"):
        return ""
    smart_feats = params.get("_smart_feats")
    if smart_feats is not None and "offload" not in smart_feats:
        logger.info("[KV Offload] offload not in effective smart features; skipping native offload CLI.")
        return ""
    if not get_lmcache_env():
        return ""

    if _resolve_offload_backend(params, engine)[0] == _OFFLOAD_NATIVE_BACKEND_VARIANT:
        size_gb = _resolve_native_backend_offload_gb(params, engine)
        logger.info("[KV Offload] whitelist backend native -> --kv-offloading-size=%dGB", size_gb)
    elif _is_deepseek_v4_flash_params(params):
        size_gb = _resolve_v4_flash_offload_gb(params)
        logger.info("[KV Offload] DeepSeek-V4-Flash (NV) -> native backend, "
                    "--kv-offloading-size=%dGB", size_gb)
    else:
        return ""
    if size_gb <= 0:
        logger.warning(
            "[KV Offload] native offload auto capacity below floor; "
            "skipping --kv-offloading-backend."
        )
        return ""
    return f" --kv-offloading-backend native --kv-offloading-size {size_gb}"


# ── MiniMax-M2.7 + RTX-PRO-5000 + vLLM 集成（融入通用流程）──────────────────────
# 不再走 build_start_script 早退分支，而是仿 DeepSeek-V4-Flash-NV / Qwen3.5 通过扩展点
# 条件分支融入通用流程：
#   * 固定 CLI 字段（trust_remote_code / kv_cache_dtype / served_model_name / moe_backend /
#     tool_call_parser / enable_auto_tool_choice / speculative_config / use_vllm_serve）
#     外置到 nvidia_default.json 的 MiniMaxM2ForCausalLM -> MiniMax-M2.7 -> vllm ->
#     rtx_pro_5000_72G 子块，由通用 4 层合并注入 engine_config。
#   * reasoning_parser 由 reason_parser.yaml 注入（MiniMax-M2.7 系列设为 minimax_m2），
#     受 enable_auto_think_choice 开关控制（与 Qwen3.5 一致）。
#   * tensor_parallel_size / data_parallel_size 由 _apply_minimax_m27_nvfp4_nv_engine_defaults
#     动态推导（TP=min(4,device_count) + DP=device_count/TP），在 _prepare_engine_config 调用。
#   * kv_transfer_config 由 config_loader._set_kv_cache_config 的 MiniMax 特例注入
#     （含通用 LMCacheConnectorV1 缺失的 kv_load_failure_policy:recompute）。
#   * 环境变量由 _build_model_env_commands 的 MiniMax 分支注入（仿 Qwen3.5 NVFP4 运行时配方，
#     返回 txt 的 LMCache env 集）；_build_cache_env_commands 的通用 LMCache 流程对本场景跳过。
#   * _build_kv_sparse_cmd 对 MiniMax 早退，跳过 calculate_kv_scales 注入（txt 无此项）。
_MINIMAX_M27_DISK_SUFFIX = "kvcache"


def _resolve_minimax_m27_lmcache_max_cpu_size(params: Dict[str, Any]) -> Optional[str]:
    """解析 MiniMax-M2.7 配方下 ``LMCACHE_MAX_LOCAL_CPU_SIZE``（每卡 GB）。

    复用通用 ``resolve_offload_cpu_capacity_gb`` 反向预算「本节点总」M_offload，再 ÷ 卡数
    （LMCache 每 rank 一池需按卡平摊），与通用 LMCache 路径 ``_resolve_lmcache_cpu_env``
    同源同公式（C4，需求一 §3.0）。auto 熔断（< 100G 不建池）亦复用通用语义，与
    DeepSeek-V4-Flash-NV 一致。

      * ``KV_MEM_OFFLOAD_SIZE == "auto"``：``resolve_offload_cpu_capacity_gb`` 返回 None（非 auto
        透传）/ 0（熔断）/>0（本节点总）；仅 >0 时 ÷ device_count 得每卡值。
      * 非 auto：``int(KV_MEM_OFFLOAD_SIZE) // device_count``。

    Returns:
        解析出的每卡容量字符串（无引号，由调用方加引号）；无法计算（auto 非命中 / 熔断 / 值非法）
        时返回 ``None``，调用方据此省略该 env。
    """
    n_card = _safe_int(params.get("device_count")) or 1
    raw_size = os.getenv("KV_MEM_OFFLOAD_SIZE", "").strip()

    if raw_size.lower() == "auto":
        auto_total = resolve_offload_cpu_capacity_gb(params)
        if auto_total is None or auto_total <= 0:
            # 非 auto 透传 / 缺 POD_MEM_SIZE / 熔断（< 100G 不建池）-> 省略该 env
            logger.info(
                "[MiniMax-M2.7] auto CPU capacity not available (resolve_offload_cpu_capacity_gb=%s); "
                "skip LMCACHE_MAX_LOCAL_CPU_SIZE.", auto_total,
            )
            return None
        per_card = max(1, auto_total // n_card)
        logger.info(
            "[MiniMax-M2.7] auto CPU per-card = M_offload(%dG) / N_card(%d) = %dG.",
            auto_total, n_card, per_card,
        )
        return str(per_card)

    try:
        total = int(raw_size)
    except ValueError:
        logger.warning(
            "[MiniMax-M2.7] Invalid KV_MEM_OFFLOAD_SIZE=%r; skip LMCACHE_MAX_LOCAL_CPU_SIZE.",
            raw_size,
        )
        return None
    per_card = max(1, total // n_card)
    logger.info(
        "[MiniMax-M2.7] custom CPU per-card = M_offload(%dG) / N_card(%d) = %dG.",
        total, n_card, per_card,
    )
    return str(per_card)


def _build_minimax_m27_rtx_pro_5000_env_commands(params: Dict[str, Any]) -> List[str]:
    """构建 MiniMax-M2.7 RTX-PRO-5000 配方的固定 LMCache 环境变量。

    引号风格逐字对齐 minimax-2.7-zyy.txt：数字/布尔无引号、字符串双引号、JSON 单引号。
    所有 LMCache 卸载相关变量仅在 ``ENABLE_KV_MEM_OFFLOAD`` 或 ``ENABLE_KV_DISK_OFFLOAD``
    至少一个为 true 时才添加（不卸载则 LMCache 不启用，相关配置无意义）。
    """
    mem_offload = os.getenv("ENABLE_KV_MEM_OFFLOAD", "false").strip().lower() == "true"
    disk_offload = os.getenv("ENABLE_KV_DISK_OFFLOAD", "false").strip().lower() == "true"

    cmds: List[str] = [
        "export PYTHONHASHSEED=0",
        "export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900",
    ]

    if not (mem_offload or disk_offload):
        return cmds

    # 内存卸载段：容量 + 启用标志
    # 不回传 ENABLE_KV_MEM_OFFLOAD 开关：它是 xperf `-e` 传入的 wings 内部门控
    # （仅用于决定是否构建本段 env），vLLM/LMCache 运行时实际读取 LMCACHE_* 系列。
    if mem_offload:
        max_cpu_size = _resolve_minimax_m27_lmcache_max_cpu_size(params)
        if max_cpu_size is not None:
            cmds.append(f'export LMCACHE_MAX_LOCAL_CPU_SIZE="{max_cpu_size}"')
        cmds.append('export LMCACHE_LOCAL_CPU="True"')

    # LMCache 通用配置（chunk size / ODirect / pre-caching hash）：仅在启用任一卸载时生效
    cmds.append("export LMCACHE_CHUNK_SIZE=256")
    cmds.append('export LMCACHE_EXTRA_CONFIG=\'{"use_odirect": true}\'')
    cmds.append('export LMCACHE_PRE_CACHING_HASH_ALGORITHM="sha256_cbor_64bit"')

    # 磁盘卸载段：路径 + 大小（同样不回传 ENABLE_KV_DISK_OFFLOAD 开关）
    if disk_offload:
        disk_base = os.getenv("KV_DISK_OFFLOAD_PATH", "").strip()
        if disk_base:
            cmds.append(f'export LMCACHE_LOCAL_DISK="{disk_base}/{_MINIMAX_M27_DISK_SUFFIX}"')
        disk_size = os.getenv("KV_DISK_OFFLOAD_SIZE", "").strip()
        if disk_size:
            cmds.append(f'export LMCACHE_MAX_LOCAL_DISK_SIZE="{disk_size}"')

    return cmds


def build_start_command(params: Dict[str, Any]) -> str:
    """为 launcher 生成 vLLM 启动命令字符串（旧版接口）。

    此函数仅执行命令拼装，不启动任何子进程。
    返回的命令不包含环境变量设置，适合简单场景。

    Args:
        params: 参数字典

    Returns:
        str: vLLM 启动命令字符串

    Raises:
        ValueError: 分布式模式不支持此简化接口

    建议:
        推荐使用 build_start_script() 获取完整脚本
    """
    if params.get("distributed", False):
        raise ValueError("Launcher MVP does not support distributed mode for vLLM.")
    return _build_vllm_cmd_parts(params)




def _filter_glm52_single_node_task_queue(
    commands: List[str],
    params: Dict[str, Any],
    engine: str,
) -> List[str]:
    """单机 GLM-5.2(a3) 对齐官方 recipe：剔除 ``TASK_QUEUE_ENABLE``（官方单机命令不设）。

    ``TASK_QUEUE_ENABLE`` 由两个 vllm_ascend 通用源注入——内联 ``set_vllm_ascend_env.sh``
    的字面量 ``=1`` 与 ``_build_vllm_ascend_forced_env_commands`` 的软默认 ``=${...:-1}``——
    去重后合并为一条。此处在 ``dedupe_env_exports`` **之后**按变量名剔除其 export 行；
    echo 行由 ``_inject_env_echo`` 依 export 生成，export 没了 echo 自然不生成，无需单独删。

    范围与单机 TP=device_count//DP + DP2 配方严格一致：仅 ``is_glm52_single_node_even`` 命中
    （vllm_ascend + GLM-5.2 + 单机偶数卡 + **a3**）才剔除；双机 / a2(910B) / 其它模型保留。
    """
    if engine != "vllm_ascend":
        return commands
    if not is_glm52_single_node_even(params):
        return commands
    kept = [c for c in commands if "TASK_QUEUE_ENABLE" not in c]
    if len(kept) != len(commands):
        logger.info("[GLM-5.2 single-node] removed TASK_QUEUE_ENABLE to align with official recipe")
    return kept


def _build_vllm_common_env_cmds(params: Dict[str, Any], engine: str) -> List[str]:
    """构建 vLLM 公共环境变量命令链（对所有部署模式均适用）。"""
    # sidecar 容器无 GPU/NPU，使用环境变量代替 netifaces 探测网络接口
    current_ip = os.getenv("POD_IP", get_local_ip())
    net_if = os.getenv("NETWORK_INTERFACE", os.getenv("GLOO_SOCKET_IFNAME", "eth0"))
    cmds: List[str] = []
    cmds.extend(_build_base_env_commands(params, engine, root_dir))
    cmds.extend(_build_cache_env_commands(engine, params))

    cmds.extend(_build_qat_env_commands(engine))
    cmds.extend(_build_pd_role_env_commands(engine, current_ip, net_if))
    cmds.extend(_build_speculative_env_commands(params, engine))
    cmds.extend(_build_trace_env_commands(params, engine))
    # 架构专用环境变量（GLM-4.7 / Qwen3 / Qwen3.5 / MiniMax-M2.5 / DeepSeek V3.2 / LLaMA 等）
    # 之前只在未被引用的 _build_env_commands 里调用，导致架构专用 env 一行都没进 start_command.sh
    cmds.extend(_build_model_env_commands(params, engine))
    # Ascend910_9362 专用 env 也一并挂上
    cmds.extend(_build_ascend910_9362_env_commands(params, engine))
    cmds = _filter_vllm_ascend_ray_incompatible_env(cmds, params, engine)
    cmds.extend(_build_vllm_ascend_forced_env_commands(params, engine))
    # PD 分离（kv_producer/consumer）下剔除互斥的 VLLM_ASCEND_BALANCE_SCHEDULING（全模型/全路径）。
    cmds = _filter_pd_incompatible_env(cmds)
    # 多个 builder（内联 set_vllm_ascend_env.sh / 架构块 / forced 软默认）会重复导出同名变量，
    # 这里收口去重，保证每个变量最终只有一条 export 生效（等价最终值，不动累加型与块内导出）。
    cmds = dedupe_env_exports(cmds)
    # 单机 GLM-5.2(a3) 对齐官方 recipe：去重后剔除 TASK_QUEUE_ENABLE（官方单机命令不设）。
    cmds = _filter_glm52_single_node_task_queue(cmds, params, engine)
    return cmds


def _build_pd_external_lb_env_cmds(params: Dict[str, Any], engine: str) -> List[str]:
    """Build the minimal env prelude for PD external-lb scripts.

    PD external-lb has role-specific env from pd_config.json. Do not reuse the
    generic vLLM env chain here, because that chain injects standalone Ascend
    and model-family defaults before PD role settings are applied.
    """
    current_ip = os.getenv("POD_IP", get_local_ip())
    net_if = os.getenv("NETWORK_INTERFACE", os.getenv("GLOO_SOCKET_IFNAME", "eth0"))
    cmds: List[str] = []

    if engine == "vllm":
        cmds.append(f"export VLLM_NIXL_SIDE_CHANNEL_HOST={shlex.quote(current_ip)}")
    elif engine == "vllm_ascend":
        cmds.extend([
            f"export HCCL_IF_IP={shlex.quote(current_ip)}",
            f"export GLOO_SOCKET_IFNAME={shlex.quote(net_if)}",
            f"export TP_SOCKET_IFNAME={shlex.quote(net_if)}",
            f"export HCCL_SOCKET_IFNAME={shlex.quote(net_if)}",
        ])
        if _is_deepseek_v4_flash_params(params):
            cmds.append(
                'export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"'
            )

    cmds = dedupe_env_exports(cmds)
    logger.info(
        "[PD external-lb env] isolated_builder=True engine=%s env_names=%s",
        engine,
        sorted(
            line.split(" ", 1)[1].split("=", 1)[0]
            for line in cmds
            if line.startswith("export ") and "=" in line
        ),
    )
    return cmds


def _build_vllm_single_script(
    params: Dict[str, Any],
    cmd: str,
    common_env_cmds: List[str],
    engine: str,
    sparse_args: str,
) -> str:
    """组装单机模式的 bash 脚本体并返回。"""
    env_prefix = "\n".join(common_env_cmds) + "\n" if common_env_cmds else ""
    speculative_extra = (
        _build_speculative_cmd(params, engine)
        if _should_append_auto_speculative_config(params)
        else ""
    )
    # A+X 环境下需要 --enforce-eager 绕过 triton 版本冲突（与 Ray 路径一致）
    eager_flag = " --enforce-eager" if _need_enforce_eager(engine) else ""
    # [V4-Flash-NV-Day0] NV V4-Flash native KV 卸载（与 LMCache 互斥，fallback 自动剥除）
    kv_offload_extra = _build_kv_offload_cmd(params, engine)

    return env_prefix + f"exec {cmd}{eager_flag}{speculative_extra}{sparse_args}{kv_offload_extra}\n"


def _build_vllm_pd_external_lb_script(params: Dict[str, Any], cmd: str,
                                      common_env_cmds: List[str], pd_ext: Dict[str, Any]) -> str:
    """生成 PD external-lb（模式 A）的 fork 启动脚本。

    对齐官方 launch_online_dp.py：pod 内 fork ``dp_size_local`` 个独立 vllm serve，
    逐 service：rank = dp_rank_start + i、port = base_port + i、卡组 = [i*tp, (i+1)*tp)。
    任一 service 退出即整 pod 退出（wait -n + kill 全部 + exit 1），交由编排层整组重启
    （EP all-to-all 下单 rank 缺失会让整域 hang）。

    base ``cmd`` 由 ``_build_vllm_cmd_parts`` 生成，已含 model / 引擎参数 /
    kv-transfer-config（含 MooncakeConnectorV1 的 ``__PD_RANK__`` 占位符）。本函数
    剥离其中的单进程 ``--port`` 与并行度相关 flag，循环里按 service 重新追加。
    """
    import re as _re

    tp = pd_ext["tp_size"]
    local = pd_ext["dp_size_local"]
    start = pd_ext["dp_rank_start"]
    dp_size = pd_ext["dp_size"]
    addr = pd_ext.get("dp_address", "")
    # --data-parallel-rpc-port 按角色硬编码（config_loader 已置 pd_ext.rpc_port=12890/12777，
    # 刻意不读 env）；此处 or 仅作防御性兜底，常量须与 config_loader 保持一致。
    rpc = pd_ext.get("rpc_port") or ("12890" if pd_ext.get("role") == "P" else "12777")
    # PD_INDEX 由上层下发（env），默认 P=0/D=1（config_loader 已处理），wings 透传不计算
    pd_index_base = pd_ext.get("pd_index_base", 0)

    # 端口基址：优先取 base cmd 里的 --port，否则回退 ENGINE_PORT
    m = _re.search(r"--port\s+(\S+)", cmd)
    base_port = m.group(1) if m else os.getenv("ENGINE_PORT", "18000")

    # 剥离单进程 --port 与并行度 flag（循环里按 service 重新追加）
    svc_cmd = cmd
    for flag in ("--port", "--tensor-parallel-size", "--data-parallel-size",
                 "--data-parallel-size-local", "--data-parallel-rank",
                 "--data-parallel-start-rank", "--data-parallel-address",
                 "--data-parallel-rpc-port"):
        svc_cmd = _re.sub(rf"\s*{flag}\s+\S+", "", svc_cmd)
    svc_cmd = _re.sub(r"\s*--data-parallel-external-lb\b", "", svc_cmd)
    svc_cmd = _re.sub(r"\s*--headless\b", "", svc_cmd)
    # 占位符 → 让 bash 在单引号 JSON 内展开 shell 变量：
    #   engine_id / kv_port 均由 PD_INDEX 派生（跨 P/D 全局连续，所有 connector 统一）。
    bootstrap_base = pd_ext.get("bootstrap_base", 23000)
    svc_cmd = svc_cmd.replace("__PD_INDEX__", "'\"$PD_INDEX\"'")
    svc_cmd = svc_cmd.replace("__PD_KVPORT__", "'\"$KVPORT\"'")
    connector = pd_ext.get("connector", "")

    role_env = params.get("_pd_env") or {}
    env_lines = list(common_env_cmds)
    for k, v in role_env.items():
        env_lines.append(f"export {k}={shlex.quote(str(v))}")
    # PD_INDEX 透传给 bash 环境（config_loader 已处理默认值 P=0/D=1），fork 脚本不计算直接引用
    env_lines.append(f"export PD_INDEX={pd_index_base}")
    # L3：common_env/角色 env 追加在 base 之后（bash 后者生效）；对整段去重，使注册表覆盖值收口、
    # 消掉 base 的同名重复（common_env_cmds 内部已去重，这里把角色 env 一并纳入再收口一次）。
    env_lines = dedupe_env_exports(env_lines)
    # strip_env（注册表 _pd_strip_env，按场景/平台/角色声明）：剔除本部署不应出现的 env export，
    # 对齐官方"不设某些 env"的口径。仅声明了 strip_env 的条目（如 GLM5 A2）生效，
    # 空集 → 不过滤，其它 PD 模型行为不变。
    strip_env = set(params.get("_pd_strip_env") or [])
    def _export_name(line: str):
        m = _re.match(r"\s*export ([A-Za-z_][A-Za-z0-9_]*)=", line)
        return m.group(1) if m else None

    env_names_before_strip = [
        name for name in (_export_name(line) for line in env_lines) if name
    ]
    if strip_env:
        env_lines = [c for c in env_lines if _export_name(c) not in strip_env]
    env_names_after_strip = [
        name for name in (_export_name(line) for line in env_lines) if name
    ]
    logger.info(
        "[PD external-lb env merge] role=%s connector=%s base_env=%s role_env=%s "
        "strip_env=%s stripped_env=%s final_env=%s",
        pd_ext.get("role"),
        connector,
        sorted(name for name in (_export_name(line) for line in common_env_cmds) if name),
        sorted(role_env.keys()),
        sorted(strip_env),
        sorted(set(env_names_before_strip) - set(env_names_after_strip)),
        sorted(env_names_after_strip),
    )

    # bootstrap 端口逐 service 唯一，供 MooncakeConnectorV1/Layerwise/Hybrid；若 strip_env 含
    # VLLM_MOONCAKE_BOOTSTRAP_PORT（如官方 p2p MooncakeConnector 不设），内联前缀也一并省去。
    rt_prefix = "ASCEND_RT_VISIBLE_DEVICES=$CARDS"
    linker_prelude = []
    process_env_prefix = ""
    if "Mooncake" in connector:
        linker_prelude.append("ldconfig /usr/local/lib >/dev/null 2>&1 || true")
        process_env_prefix = "LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-} "
    rt_prefix = f"{process_env_prefix}{rt_prefix}"
    if "VLLM_MOONCAKE_BOOTSTRAP_PORT" not in strip_env:
        rt_prefix += " VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP"

    # fork 主体包进子 shell，使其作为单个可后台化单元被上层监控
    # （wings_entry._strip_exec_and_backgroundify 给末行 ')' 追加 ' &' + ENGINE_PID=$!）。
    # 任一 service 退出 → 子 shell exit 1 → 上层 crash-retry 整 pod 重启（EP all-to-all 语义）。
    # dp_size=1（1P1D）：单进程，不带 --data-parallel-external-lb（vllm-ascend 校验 dp_size>1），
    # 不套 fork 子 shell，直接以前台单命令启动。将 '"$PD_INDEX"' / '"$KVPORT"' 替换为字面值。
    if dp_size == 1:
        _pd_idx = str(pd_index_base)
        svc_cmd = svc_cmd.replace("'\"$PD_INDEX\"'", _pd_idx)
        svc_cmd = svc_cmd.replace("'\"$KVPORT\"'", str(30000 + pd_index_base * 100))
        rt_prefix_1 = (
            f"{process_env_prefix}ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $(({tp} - 1)))"
        )
        if "VLLM_MOONCAKE_BOOTSTRAP_PORT" not in strip_env:
            rt_prefix_1 += f" VLLM_MOONCAKE_BOOTSTRAP_PORT={bootstrap_base}"
        single_cmd = f"{rt_prefix_1} {svc_cmd} --port {base_port} --tensor-parallel-size {tp}"
        return "\n".join(env_lines + linker_prelude + [single_cmd]) + "\n"

    fork_body = [
        "(",
        "  pids=()",
        f"  for i in $(seq 0 {local - 1}); do",
        f"    RANK=$(({start} + i)); PORT=$(({base_port} + i))",
        "    PD_INDEX=$PD_INDEX",
        f"    KVPORT=$((30000 + PD_INDEX * 100)); BOOTSTRAP=$(({bootstrap_base} + i))",
    ]
    fork_body += [
        f"    LO=$((i * {tp})); HI=$((LO + {tp} - 1)); CARDS=$(seq -s, $LO $HI)",
        (f"    {rt_prefix}"
         f" {svc_cmd} --port $PORT"
         f" --tensor-parallel-size {tp} --data-parallel-size {dp_size}"
         f" --data-parallel-rank $RANK --data-parallel-size-local 1"
         f" --data-parallel-address {shlex.quote(addr)} --data-parallel-rpc-port {rpc}"
         f" --data-parallel-external-lb &"),
        "    pids+=($!)",
        "  done",
        '  wait -n || true',
        '  echo "[pd] a service exited, tearing down pod" >&2',
        '  kill "${pids[@]}" 2>/dev/null || true',
        "  exit 1",
        ")",
    ]
    return "\n".join(env_lines + linker_prelude + fork_body) + "\n"


def build_start_script(params: Dict[str, Any]) -> str:
    """生成完整的 bash 启动脚本体（start_command.sh 内容，不含 shebang）。

    这是 vLLM 适配器的主要入口，生成的脚本将写入共享卷，
    由 engine 容器读取并执行。

    支持的部署模式:

    1. 单机 vllm:
       exec python3 -m vllm.entrypoints.openai.api_server ...

    2. 单机 vllm_ascend:
       source /usr/local/Ascend/.../set_env.sh  # 加载 CANN 环境
       exec python3 -m vllm.entrypoints.openai.api_server ...

    3. Ray 分布式 (rank0 - head 节点) / Ray 分布式 (rank>0 - worker 节点)

    4. DP 分布式 (dp_deployment 后端):
       exec python3 -m vllm... --data-parallel-address ... --data-parallel-rank ...

    Args:
        params: 参数字典，包含 engine/distributed/nnodes/node_rank 等关键字段

    Returns:
        str: 完整的 bash 脚本体（不含 shebang）
    """
    engine = params.get("engine", "vllm")
    # KV 稀疏：必须在 _build_vllm_cmd_parts 之前调用，
    # FP8 路径会就地修改 engine_config，避免 --kv-cache-dtype 重复。
    # enable_sparse 已由 config_loader.apply_effective_feature_enablement (§2.0 C14) 收口为
    # 「有效开关」（开关 on 且命中白名单才为真，无 forced）。原 _force_kv_sparse_* 已按 §0 裁定1 删除。
    should_emit_sparse = bool(params.get("enable_sparse"))
    sparse_args = _build_kv_sparse_cmd(params, engine) if should_emit_sparse else ""
    # GLM-4.7-W8A8 引擎参数注入（必须在 _build_vllm_cmd_parts 之前，且只动 W8A8 量化变体）
    _inject_glm47_w8a8_engine_config(params, force_non_explicit=True)
    cmd = _build_vllm_cmd_parts(params)
    is_distributed = params.get("distributed", False)
    nnodes = params.get("nnodes", 1)
    pd_ext = params.get("_pd_external_lb")
    if pd_ext:
        # PD external-lb（模式 A）：pod 内 fork dp_size_local 个独立 vllm serve
        logger.info(
            "[vllm_adapter.env_path] selected=pd_external_lb_isolated engine=%s "
            "role=%s tp_size=%s dp_size=%s dp_size_local=%s distributed=%s nnodes=%s",
            engine,
            pd_ext.get("role"),
            pd_ext.get("tp_size"),
            pd_ext.get("dp_size"),
            pd_ext.get("dp_size_local"),
            is_distributed,
            nnodes,
        )
        common_env_cmds = _build_pd_external_lb_env_cmds(params, engine)
        script = _build_vllm_pd_external_lb_script(params, cmd, common_env_cmds, pd_ext)
    elif is_distributed and nnodes > 1:
        logger.info(
            "[vllm_adapter.env_path] selected=distributed_common engine=%s "
            "distributed=%s nnodes=%s",
            engine,
            is_distributed,
            nnodes,
        )
        common_env_cmds = _build_vllm_common_env_cmds(params, engine)
        script = _build_vllm_distributed_script(params, cmd, common_env_cmds, engine, sparse_args)
    else:
        logger.info(
            "[vllm_adapter.env_path] selected=single_common engine=%s "
            "distributed=%s nnodes=%s",
            engine,
            is_distributed,
            nnodes,
        )
        common_env_cmds = _build_vllm_common_env_cmds(params, engine)
        script = _build_vllm_single_script(params, cmd, common_env_cmds, engine, sparse_args)

    script = _inject_env_echo(script)

    return script


def _inject_env_echo(script: str) -> str:
    """在脚本中每条 'export VAR=...' 语句前插入 echo 打印，方便排查环境变量注入情况。

    同时对关键命令行（python3 引擎启动 / ray start / source set_env）前置
    `echo "[wings-cmd] >>> ..."`，便于在 engine.log 里快速定位每条实际执行的命令。

    打印格式：`[wings-env] export VAR=<value>`，使用 `${VAR}` 在 bash 运行时
    展开实际值。注意：值会原样进日志，不再脱敏；如有 token / API key 等敏感
    变量，请避免通过 export 注入或在调用方自行脱敏后再传入。

    为便于排查最终执行环境，脚本内显式 export 的变量都会追加一行
    `[wings-env] export VAR=<value>` 日志。

    Args:
        script: 原始 bash 脚本字符串

    Returns:
        str: 插入 echo 打印后的脚本字符串
    """
    import re as _re
    lines = script.splitlines(keepends=True)
    result = []
    # Echo matched command lines before execution; keep the preview complete.
    # exec \./... 涵盖 MindIE 等用 exec ./bin/daemon 形式启动的可执行文件；
    # \./[A-Za-z0-9_./-]+ 去除 .sh 限制，同时覆盖 ./bin/mindieservice_daemon & 等无扩展名程序。
    cmd_prefix_re = _re.compile(
        r'^(exec\s+(?:python3?|vllm\s+serve|\./\S+)|vllm\s+serve|python3?\s+-m\s+vllm|python3?\s+-m\s+sglang|'
        r'python3?\s+\S*/install\.py|\(?cd\s+\S+\s+&&\s+python3?\s+install\.py|'
        r'ray\s+(start|stop|status)|source\s+/|nohup\s+|\./[A-Za-z0-9_./-]+)'
    )
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        m = _re.match(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)', stripped)
        if m:
            var_name = m.group(1)
            # 先输出 export 本身，再 echo（${VAR:-} 兜底，避免 `set -u` 触发
            # unbound variable，且能反映 export 后的实际值——含 LD_LIBRARY_PATH
            # 这种追加合并的最终结果）。
            result.append(line)
            indent = line[: len(line) - len(stripped)]
            next_line = lines[idx + 1].lstrip() if idx + 1 < len(lines) else ""
            already_echoed = (
                f"[wings-env] export {var_name}=" in next_line
                or f"[mindie-env] {var_name}=" in next_line
            )
            if not already_echoed:
                result.append(
                    f'{indent}echo "[wings-env] export {var_name}=${{{var_name}:-}}"\n'
                )
            continue
        if cmd_prefix_re.match(stripped):
            indent = line[: len(line) - len(stripped)]
            # Keep the command intact so the logged line can be replayed directly.
            preview = stripped.rstrip("\n").rstrip("&").rstrip()
            preview_safe = preview.replace("'", "'\"'\"'")
            already_echoed = bool(result and "[wings-cmd] >>>" in result[-1])
            if not already_echoed:
                result.append(f"{indent}echo '[wings-cmd] >>> {preview_safe}'\n")
        result.append(line)
    return "".join(result)


def start_vllm_distributed(params: Dict):
    """分布式模式入口（sidecar MVP 中不支持）。

    Raises:
        RuntimeError: sidecar 架构不允许直接启动进程
    """
    raise RuntimeError("分布式模式在 sidecar launcher MVP 中已禁用。")


def start_engine(params: Dict[str, Any]):
    """旧版兼容接口（sidecar launcher 模式中已禁用）。

    在 sidecar 架构中，适配器不允许直接启动推理进程。
    应使用 build_start_script() 生成脚本，写入共享卷，
    由 engine 容器执行。

    Raises:
        RuntimeError: 始终抛出，阻止意外调用
    """
    raise RuntimeError(
        "start_engine 在 launcher 模式中已禁用。"
        "请使用 build_start_command() 并将结果写入共享卷。"
    )
