"""MemCache Hybrid helpers for Kimi offload."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from wings_control.utils.device_utils import resolve_card_token
    from wings_control.utils.model_utils import feature_allowed
except ImportError:
    from utils.device_utils import resolve_card_token  # type: ignore
    from utils.model_utils import feature_allowed  # type: ignore

logger = logging.getLogger(__name__)

MEMCACHE_OFFLOAD_VARIANT = "memcache"

_OFFLOAD_ENGINE_SELF_PER_WORKER_GB = 7
_OFFLOAD_ENGINE_SELF_BASE_GB = 3
_OFFLOAD_MARGIN_RATIO = 0.10
_OFFLOAD_MIN_GB = 100
_TEMPLATE_DIR = Path(__file__).resolve().parent


def empty_memcache_hybrid_fragment() -> dict:
    return {
        "enabled": False,
        "engine_prelude": "",
        "fallback_cleanup": "",
        "master_script": "",
        "env": {},
    }


def is_kimi_k27_code_memcache_params(params: Optional[Dict[str, Any]], engine: str) -> bool:
    """Return whether params should use Kimi K2.7 Code MemCache offload."""
    if not params or engine != "vllm_ascend":
        return False
    text = " ".join(
        str(params.get(key, "") or "").lower()
        for key in ("model_name", "model_path")
    )
    return "kimi-k2.7-code" in text and "w4a8" not in text


def build_memcache_ascend_store_config() -> Dict[str, Any]:
    """Build the vLLM AscendStoreConnector config used by MemCache Hybrid."""
    return {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "lookup_rpc_port": "0",
            "backend": "memcache",
        },
    }


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _offload_parallel_size(params: Dict[str, Any], key: str) -> int:
    val = _safe_int(params.get(key))
    if val:
        return val
    engine_config = params.get("engine_config")
    if isinstance(engine_config, dict):
        val = _safe_int(engine_config.get(key))
        if val:
            return val
    return 1


def _resolve_offload_cpu_capacity_gb(
    params: Dict[str, Any],
    size_env_name: str = "KV_MEM_OFFLOAD_SIZE",
) -> Optional[int]:
    max_cpu = os.getenv(size_env_name, "").strip()
    pod_mem = os.getenv("AVAILABLE_POD_MEM_SIZE", "").strip()
    if os.getenv("ENABLE_KV_MEM_OFFLOAD", "false").strip().lower() != "true":
        return None
    if max_cpu.lower() != "auto":
        return None
    if not pod_mem:
        return None
    try:
        m_container = float(pod_mem) / 1024.0
    except (TypeError, ValueError):
        logger.warning("[MemCache] Invalid AVAILABLE_POD_MEM_SIZE=%r; auto capacity skipped.", pod_mem)
        return None
    tp = _offload_parallel_size(params, "tensor_parallel_size")
    dp = _offload_parallel_size(params, "data_parallel_size")
    m_engine_self = _OFFLOAD_ENGINE_SELF_PER_WORKER_GB * (tp * dp) + _OFFLOAD_ENGINE_SELF_BASE_GB
    m_margin = m_container * _OFFLOAD_MARGIN_RATIO
    m_offload = m_container - m_engine_self - m_margin
    if m_offload < _OFFLOAD_MIN_GB:
        return 0
    return int(m_offload)


def resolve_memcache_dram_gb(params: Optional[Dict[str, Any]]) -> Optional[int]:
    """Resolve MemCache local DRAM size from the page-owned offload memory."""
    params = params or {}
    raw_size = os.getenv("KV_MEM_OFFLOAD_SIZE", "").strip()
    size_env_name = "KV_MEM_OFFLOAD_SIZE"
    if not raw_size:
        raw_size = os.getenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "").strip()
        size_env_name = "LMCACHE_MAX_LOCAL_CPU_SIZE"
    if not raw_size:
        return None
    if raw_size.lower() == "auto":
        auto_total = _resolve_offload_cpu_capacity_gb(params, size_env_name=size_env_name)
        if auto_total and auto_total > 0:
            return int(auto_total)
        return None
    try:
        size_gb = int(raw_size)
    except (TypeError, ValueError):
        logger.warning("[MemCache] Invalid page offload memory size=%r; disabling MemCache.", raw_size)
        return None
    if size_gb <= 0:
        return None
    return size_gb


def _memcache_offload_allowed(engine: str, merged: dict | None) -> bool:
    if os.getenv("ENABLE_KV_OFFLOAD", "").strip().lower() != "true":
        return False
    if not merged:
        return False
    smart_feats = merged.get("_smart_feats")
    if smart_feats is not None:
        return "offload" in smart_feats
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
    """Build directly concatenable MemCache Hybrid startup fragments."""
    if (
        not merged
        or not is_kimi_k27_code_memcache_params(merged, engine)
        or not _memcache_offload_allowed(engine, merged)
    ):
        return empty_memcache_hybrid_fragment()

    dram_gb = resolve_memcache_dram_gb(merged)
    if not dram_gb:
        return empty_memcache_hybrid_fragment()

    master_script = _read_template("memcache_master.sh").rstrip()
    engine_prelude = (
        _read_template("memcache_engine_prelude.sh")
        .replace("{dram_gb}", str(dram_gb))
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
