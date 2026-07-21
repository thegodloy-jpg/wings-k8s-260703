"""面向模型场景的 Ascend MemCache Hybrid offload 辅助函数。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from utils.env_utils import (
        OFFLOAD_MIN_GB,
        get_lmcache_env,
        resolve_offload_cpu_capacity_gb,
    )
    from utils.device_utils import resolve_card_token
    from utils.model_utils import (
        feature_allowed,
        is_kimi_k26_family,
        is_kimi_k27_code_family,
        resolve_feature_whitelist_row_from_params,
        resolve_offload_whitelist_backend,
    )
except ImportError:
    from wings_control.utils.env_utils import (  # type: ignore
        OFFLOAD_MIN_GB,
        get_lmcache_env,
        resolve_offload_cpu_capacity_gb,
    )
    from wings_control.utils.device_utils import resolve_card_token  # type: ignore
    from wings_control.utils.model_utils import (  # type: ignore
        feature_allowed,
        is_kimi_k26_family,
        is_kimi_k27_code_family,
        resolve_feature_whitelist_row_from_params,
        resolve_offload_whitelist_backend,
    )

logger = logging.getLogger(__name__)

MEMCACHE_OFFLOAD_VARIANT = "memcache"

_TEMPLATE_DIR = Path(__file__).resolve().parent
_DEFAULT_META_SERVICE_URL = "tcp://127.0.0.1:5000"
_DEFAULT_CONFIG_STORE_URL = "tcp://127.0.0.1:6000"
_DEFAULT_PROTOCOL = "device_rdma"

_ASCEND_MEMCACHE_PROTOCOLS = (
    ("910c", "device_sdma"),
    ("910b", "device_rdma"),
)

# Qwen Day0 的 MemCache 端口/协议是启动脚本契约，不是白名单规则。
# 白名单只声明 backend=memcache；具体 endpoint/protocol 放在这里，仍允许
# 部署层通过 WINGS_MEMCACHE_* env 覆盖最终值。
_QWEN35_DAY0_PROFILE = (
    "tcp://127.0.0.1:50051",
    "tcp://127.0.0.1:50061",
    "device_sdma",
)
_QWEN36_DAY0_PROFILE = (
    "tcp://127.0.0.1:50071",
    "tcp://127.0.0.1:50081",
    "device_rdma",
)
_KIMI_K27_DAY0_PROFILE = (
    _DEFAULT_META_SERVICE_URL,
    _DEFAULT_CONFIG_STORE_URL,
    "device_sdma",
)


def empty_memcache_hybrid_fragment() -> dict:
    return {
        "enabled": False,
        "engine_prelude": "",
        "fallback_cleanup": "",
        "master_script": "",
        "env": {},
    }


def is_kimi_k27_code_memcache_params(params: Optional[Dict[str, Any]], engine: str) -> bool:
    """判断当前参数是否应使用 Kimi K2.7 Code MemCache offload。"""
    return is_kimi_k27_code_family(params, engine)


def is_kimi_k26_memcache_params(params: Optional[Dict[str, Any]], engine: str) -> bool:
    return is_kimi_k26_family(params, engine)


def is_qwen_day0_memcache_params(params: Optional[Dict[str, Any]], engine: str) -> bool:
    """判断当前参数是否命中 Qwen Day0 MemCache 支持场景。"""
    if not params or engine != "vllm_ascend":
        return False

    # 这里不要再维护第二份 Qwen 模型 token 表。Day0 矩阵对模型和芯片都敏感，
    # MemCache 能力必须跟随特性 gating 和 dry-run 命令生成已经使用的同一条
    # offload 白名单行。arch 检查用于把该 helper 限定在 Qwen 场景，即使后续
    # 其它 offload 模型也复用相同的 MemCache 传输。
    row = resolve_feature_whitelist_row_from_params(
        params,
        engine,
        "offload",
        require_enabled=True,
    )
    if not row:
        return False
    return (
        row.get("backend") == MEMCACHE_OFFLOAD_VARIANT
        and str(row.get("arch", "")).startswith("Qwen3_5")
    )


def is_memcache_hybrid_params(params: Optional[Dict[str, Any]], engine: str) -> bool:
    """判断当前参数是否应使用 MemCache Hybrid offload 路径。"""
    backend = resolve_offload_whitelist_backend(params, engine)
    if backend:
        return backend == MEMCACHE_OFFLOAD_VARIANT
    smart_feats = (params or {}).get("_smart_feats")
    if smart_feats is not None:
        return "offload" in smart_feats and (
            is_kimi_k27_code_memcache_params(params, engine)
            or is_kimi_k26_memcache_params(params, engine)
        )
    return False


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_memcache_per_card_gb(params: Dict[str, Any], total_gb: int) -> int:
    """将页面下发或自动计算的节点总容量均分为 MemCache 每卡容量。

    页面容量字段的口径与 LMCache 一致，表示当前节点可用于 KV offload 的总内存。
    ``mmc_local.conf`` 中的 ``dram.size`` 则是单卡本地服务容量，因此必须按
    ``device_count`` 均分，避免每张卡都重复申请整节点容量。
    """
    device_count = _safe_int(params.get("device_count")) or 1
    per_card_gb = max(1, total_gb // device_count)
    logger.info(
        "[MemCache] per-card DRAM = node offload memory(%dG) / N_card(%d) = %dG.",
        total_gb,
        device_count,
        per_card_gb,
    )
    return per_card_gb


def resolve_memcache_dram_gb(params: Optional[Dict[str, Any]]) -> Optional[int]:
    """从节点 offload memory 解析 MemCache 单卡本地 DRAM 容量。"""
    params = params or {}
    raw_size = os.getenv("KV_MEM_OFFLOAD_SIZE", "").strip()
    size_env_name = "KV_MEM_OFFLOAD_SIZE"
    if not raw_size:
        return None
    if raw_size.lower() == "auto":
        auto_total = resolve_offload_cpu_capacity_gb(params, size_env_name=size_env_name)
        if auto_total and auto_total > 0:
            return _resolve_memcache_per_card_gb(params, int(auto_total))
        if auto_total == 0:
            logger.info(
                "[MemCache] Page offload memory auto capacity is below %dG floor; "
                "disabling MemCache.",
                OFFLOAD_MIN_GB,
            )
        return None
    try:
        size_gb = int(raw_size)
    except (TypeError, ValueError):
        logger.warning("[MemCache] Invalid page offload memory size=%r; disabling MemCache.", raw_size)
        return None
    if size_gb <= 0:
        return None
    return _resolve_memcache_per_card_gb(params, size_gb)


def _resolve_memcache_profile_defaults(
    params: Optional[Dict[str, Any]],
    engine: str,
) -> tuple[str, str, str]:
    """返回场景拥有的 MemCache endpoint 和按卡型解析的传输协议默认值。

    shell 模板仍允许部署层通过 WINGS_MEMCACHE_META_SERVICE_URL 和
    WINGS_MEMCACHE_CONFIG_STORE_URL、WINGS_MEMCACHE_PROTOCOL 覆盖。这里的默认值
    只决定部署层未显式下发时渲染什么。白名单只表达是否允许 memcache backend；
    端口和协议属于启动参数默认值，不放入 smart_feature_whitelist.json。
    """
    params = params or {}
    profile = _resolve_qwen_day0_memcache_profile(params, engine)
    if not profile and is_kimi_k27_code_memcache_params(params, engine):
        profile = _KIMI_K27_DAY0_PROFILE
    if not profile:
        profile = (
            _DEFAULT_META_SERVICE_URL,
            _DEFAULT_CONFIG_STORE_URL,
            _DEFAULT_PROTOCOL,
        )

    meta_service_url, config_store_url, fallback_protocol = profile
    return (
        meta_service_url,
        config_store_url,
        _resolve_memcache_protocol(params, fallback_protocol),
    )


def _resolve_memcache_protocol(params: Dict[str, Any], fallback_protocol: str) -> str:
    """已识别的昇腾卡严格按卡型选协议，未知卡型沿用原场景默认值。"""
    card_token = str(params.get("_smart_card_token") or resolve_card_token()).lower()
    for expected_card, protocol in _ASCEND_MEMCACHE_PROTOCOLS:
        if expected_card in card_token:
            return protocol
    return fallback_protocol


def _resolve_qwen_day0_memcache_profile(
    params: Dict[str, Any],
    engine: str,
) -> Optional[tuple[str, str, str]]:
    """在白名单确认允许 MemCache 后，返回 Qwen Day0 静态启动 profile。

    这里故意先调用 ``is_qwen_day0_memcache_params``：端口/协议默认值只能在
    offload 已经通过页面开关和白名单 gating 后生效，不能反过来靠模型名绕过
    smart-feature 有效状态。这样既保留 Qwen3.5/Qwen3.6 标准脚本差异，又不把
    端口、协议这类部署参数塞回白名单。
    """
    if not is_qwen_day0_memcache_params(params, engine):
        return None
    text = " ".join(
        str(params.get(key, "") or "").lower()
        for key in ("model_name", "model_path")
    )
    # Qwen3.5 的 50051/50061 与 Qwen3.6 的 50071/50081 仍按模型 profile 选择；
    # 传输协议由上层统一按 910C/910B 卡型覆盖，避免模型规则和硬件规则交叉。
    if "qwen3.5-27b" in text or "qwen3.5-35b-a3b" in text:
        return _QWEN35_DAY0_PROFILE
    if "qwen3.6-" in text:
        return _QWEN36_DAY0_PROFILE
    return None


def _memcache_offload_allowed(engine: str, merged: dict | None) -> bool:
    if not merged:
        return False
    smart_feats = merged.get("_smart_feats")
    if smart_feats is not None:
        return "offload" in smart_feats
    if not get_lmcache_env():
        return False
    return feature_allowed(
        engine,
        merged.get("model_name"),
        merged.get("model_path"),
        merged.get("_smart_card_token") or resolve_card_token(),
        "offload",
    )


def _read_template(filename: str) -> str:
    return (_TEMPLATE_DIR / filename).read_text(encoding="utf-8")


def build_memcache_hybrid_fragment(engine: str, merged: dict | None) -> dict:
    """构建可直接拼接进启动脚本的 MemCache Hybrid 片段。"""
    if (
        not merged
        or not is_memcache_hybrid_params(merged, engine)
        or not _memcache_offload_allowed(engine, merged)
    ):
        return empty_memcache_hybrid_fragment()

    dram_gb = resolve_memcache_dram_gb(merged)
    if not dram_gb:
        return empty_memcache_hybrid_fragment()

    meta_service_url, config_store_url, protocol = _resolve_memcache_profile_defaults(merged, engine)
    master_script = (
        _read_template("memcache_master.sh")
        .rstrip()
        .replace("{meta_service_url}", meta_service_url)
        .replace("{config_store_url}", config_store_url)
    )
    engine_prelude = (
        _read_template("memcache_engine_prelude.sh")
        .replace("{dram_gb}", str(dram_gb))
        .replace("{meta_service_url}", meta_service_url)
        .replace("{config_store_url}", config_store_url)
        .replace("{protocol}", protocol)
        .replace("{master_script}", master_script)
    )
    return {
        "enabled": True,
        "engine_prelude": engine_prelude,
        "fallback_cleanup": (
            "# --- wings-memcache: fallback cleanup ---\n"
            "unset MMC_LOCAL_CONFIG_PATH\n"
            "# --- end wings-memcache: fallback cleanup ---\n"
        ),
        "master_script": master_script,
        "env": {"WINGS_MEMCACHE_DRAM_GB": str(dram_gb)},
    }
