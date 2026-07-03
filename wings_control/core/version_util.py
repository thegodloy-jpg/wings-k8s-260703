# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""ENGINE_VERSION 环境变量统一解析与规范化。

上层传入的版本号格式可能不规范，例如：
    v0.17.0-20260325, 0.14.rc1, V0.17, 0.12.0.post1, latest, nightly-20260301

本模块提供两个函数：
    - parse_engine_version_tuple(): 返回 (major, minor) 整数元组，用于版本比较
    - normalize_engine_version():   返回规范化的 "major.minor.patch" 字符串
"""

import logging
import os
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# 未设置或无法解析时的默认版本元组
# 与 supported_features.json 中的 is_default 版本对齐
DEFAULT_VERSION_TUPLE = (0, 17)


def parse_engine_version_tuple(ver_str: str | None = None) -> tuple[int, int]:
    """将版本字符串解析为 (major, minor) 元组。

    解析策略：
    1. 去除前导 v/V 前缀
    2. 提取前两段数字（以 '.' 分隔）
    3. 忽略后续内容（patch、预发布标签、日期后缀等）

    支持格式示例：
        "0.17"            → (0, 17)
        "0.17.0"          → (0, 17)
        "v0.17.0"         → (0, 17)
        "V0.17.0-20260325"→ (0, 17)
        "0.14.rc1"        → (0, 14)
        "0.12.0.post1"    → (0, 12)

    Args:
        ver_str: 版本字符串。若为 None，则从 ENGINE_VERSION 环境变量读取。

    Returns:
        (major, minor) 整数元组；无法解析时返回 DEFAULT_VERSION_TUPLE。
    """
    if ver_str is None:
        ver_str = os.getenv("ENGINE_VERSION", "").strip()
    else:
        ver_str = ver_str.strip()

    if not ver_str:
        return DEFAULT_VERSION_TUPLE

    # 去除前导 v/V
    cleaned = re.sub(r'^[vV]', '', ver_str)

    # 提取前两段数字
    match = re.match(r'(\d+)\.(\d+)', cleaned)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    # 可能只有一个数字（如 "17"）
    match_single = re.match(r'(\d+)', cleaned)
    if match_single:
        logger.warning(
            "ENGINE_VERSION='%s' only has major version, "
            "defaulting minor to 0",
            ver_str,
        )
        return (int(match_single.group(1)), 0)

    logger.warning(
        "ENGINE_VERSION='%s' format unrecognized, defaulting to %s",
        ver_str,
        ".".join(str(v) for v in DEFAULT_VERSION_TUPLE),
    )
    return DEFAULT_VERSION_TUPLE


# 卡型号 token 归一化映射表。
# 输入来源多样（env 直传、镜像版本后缀、hardware_info.json 的 name、device_utils 回退值），
# 写法差异大：大小写、分隔符(-_.空格)、厂商前缀。统一小写并去除分隔符后再查表，
_CARD_TOKEN_ALIASES: dict[str, str] = {
    # ── Ascend ──
    "a2": "a2", "atlasa2": "a2", "910b": "a2", "ascend910b": "a2",
    "a3": "a3", "atlasa3": "a3", "910c": "a3", "ascend910c": "a3",
    # ── NVIDIA RTX Pro 5000 ──
    "rtxpro500048g": "rtx_pro_5000_48G",
    "rtxpro500072g": "rtx_pro_5000_72G",
    # ── NVIDIA H20 ──
    "h2096g": "h20_96G",
    "h20141g": "h20_141G",
}


def _normalize_card_token(raw: str) -> str:
    """卡型号字符串归一化：小写 + 去除 ``-`` ``_`` ``.`` 及空白分隔符。"""
    return re.sub(r"[-_.\s]+", "", (raw or "").lower())


def _canonical_card_model(raw: Any) -> str:
    """把任意写法的卡型号映射到规范 key，未识别返回空串。

    先做 token 精确匹配，再对完整硬件名走子串匹配（覆盖带厂商前缀/显存/型号后缀
    的写法，如 ``"NVIDIA RTX PRO 5000 72GB Blackwell"`` → ``rtx_pro_5000_72G``）。
    未命中任何别名时返回空串（而非原样回传 raw），使 ``resolve_card_model``
    的优先级链能继续 fall-through 到下一档信号源，避免被一段无法识别的设备名短路。
    """
    token = _normalize_card_token(str(raw) if raw is not None else "")
    if not token:
        return ""
    if token in _CARD_TOKEN_ALIASES:
        return _CARD_TOKEN_ALIASES[token]
    return ""


def _extract_card_token(raw: Any) -> str:
    """从卡型信号提取卡型 token：白名单命中返回规范名，否则返回检测到的名称。

    - 白名单命中：返回规范名称（如 ``rtx_pro_5000_72G``），与
      ``nvidia_default.json`` 配置 key 对齐，保证卡型专属配置能命中；
    - 白名单未命中：原样返回检测到的名称（``str(raw)``），使未登记卡型
      （A100 / B200 / 910A …）也能被检出回传。

    ``raw`` 为 ``None`` 时返回空串。

    示例::

        "rtx_pro_5000_72G"           → "rtx_pro_5000_72G"   # 白名单命中
        "NVIDIA RTX PRO 5000 72GB"   → "rtx_pro_5000_72G"   # 白名单子串命中
        "NVIDIA A100 80GB"           → "NVIDIA A100 80GB"   # 原样
        "Ascend 910A"                → "Ascend 910A"        # 原样
    """
    canonical = _canonical_card_model(raw)
    if canonical:
        return canonical
    return str(raw) if raw is not None else ""


def _parse_card_model_from_engine_version(ver_str: str | None = None) -> str:
    """从 ``ENGINE_VERSION`` 的版本号分段中解析卡型号。

    镜像构建版本号以卡型后缀标识芯片/GPU，如::

        "0.13.0rc3-a3"            → a3
        "0.21.0-rtx_pro_5000_72G" → rtx_pro_5000_72G
        "0.21.0-h20_141g"         → h20_141G

    从右向左扫描各 ``-`` 分段，对每段走 :func:`_extract_card_token`（白名单优先 +
    检测名兜底），并跳过纯版本号/纯数字日期段（不含字母或不含数字）——否则
    ``0.17.0`` 会被误判为卡型 ``0_17_0``。从右向左可避开日期后缀（如
    ``-20260325``）。无可识别分段时返回空串。

    Args:
        ver_str: 版本字符串。为 None 时从 ``ENGINE_VERSION`` 环境变量读取。

    Returns:
        卡型 token 或空串。
    """
    if ver_str is None:
        ver_str = os.getenv("ENGINE_VERSION", "")
    ver = (ver_str or "").strip()
    if not ver:
        return ""
    # 去除前导 v/V 后按 '-' 拆分逐段识别
    segments = re.sub(r"^[vV]", "", ver).split("-")
    for seg in reversed(segments):
        token = _extract_card_token(seg)
        if not token:
            continue
        # 跳过纯版本号(0_17_0 无字母)/纯数字日期(20260325 无字母)段
        if not (any(c.isalpha() for c in token) and any(c.isdigit() for c in token)):
            continue
        return token
    return ""


def engine_version_platform(ver_str: str | None = None) -> str | None:
    """从 ``ENGINE_VERSION`` 后缀解析 Ascend 芯片：``'a3'``(910C) / ``'a2'``(910B)。

    镜像构建版本号以 ``-a3`` / ``-a2`` 后缀标识芯片（如 ``"0.21.0-a3"`` → a3/910C）。
    无可识别后缀时返回 ``None``（由调用方决定默认）。

    本函数是 engine-version → 芯片 的【单一归口】：
      * ``vllm_adapter._get_engine_config_platform`` 复用它（取代原内联 ``endswith("-a3")``）；
      * ``model_utils.is_glm52_single_node_even`` 复用它做单机 GLM-5.2 的 a3 门控。

    Args:
        ver_str: 版本字符串。为 None 时从 ``ENGINE_VERSION`` 环境变量读取。

    Returns:
        ``"a3"`` / ``"a2"`` / ``None``。
    """
    if ver_str is None:
        ver_str = os.getenv("ENGINE_VERSION", "")
    ver = (ver_str or "").strip().lower()
    if ver.endswith("-a3"):
        return "a3"
    if ver.endswith("-a2"):
        return "a2"
    return None


def resolve_card_model(
    params: Dict[str, Any] | None = None,
) -> str:
    """获取当前运行环境的卡型号 —— 全局统一归口。

    按以下 4 档优先级依次取卡型，命中即返回，全未命中返回空串 ``""``::

      1. detect_hardware().hardware_family（探测到的硬件真相源）
      2. WINGS_DEVICE_NAME（显式声明环境变量）
      3. ENGINE_VERSION 后缀（镜像构建版本号携带的卡型标识）
      4. engine_config 声明字段：card_model / ascend_platform / hardware_platform

    每档信号经 :func:`_extract_card_token` 处理：命中白名单返回规范名
    （如 ``rtx_pro_5000_72G``，与 ``nvidia_default.json`` 配置 key 对齐），
    未命中返回检测到的名称（未登记卡型亦能检出）。

    Args:
        params: 启动参数字典（可选）；提供时从中读取 engine_config 声明字段。

    Returns:
        规范卡型名（白名单命中）或检测到的卡型名称（未登记）；无信号时返回 ``""``。
    """
    params = params or {}
    engine_config = params.get("engine_config") or {}

    # 1) 探测到的硬件真相源：detect_hardware().hardware_family
    try:
        from core.hardware_detect import detect_hardware
        family = detect_hardware().get("hardware_family")
        token = _extract_card_token(family)
        if token:
            return token
    except Exception as exc:  # noqa: BLE001
        logger.debug("[resolve_card_model] detect_hardware failed: %s", exc)

    # 2) WINGS_DEVICE_NAME（显式声明环境变量）
    token = _extract_card_token(os.getenv("WINGS_DEVICE_NAME", ""))
    if token:
        return token

    # 3) ENGINE_VERSION 后缀（镜像构建版本号携带的卡型标识）
    token = _parse_card_model_from_engine_version()
    if token:
        return token

    # 4) engine_config 声明字段
    for cfg_key in ("card_model", "ascend_platform", "hardware_platform"):
        token = _extract_card_token(engine_config.get(cfg_key))
        if token:
            return token

    return ""


def normalize_engine_version(ver_str: str | None = None) -> str:
    """将版本字符串规范化为 "major.minor.0" 格式。

    用于需要干净版本字符串的场景（如 JSON 构造、传递给 install.py 等）。
    输出 3 段格式以确保与 supported_features.json 中的版本 key（如 "0.17.0"）
    直接匹配，避免 PEP 440 字符串比较不一致问题。

    Args:
        ver_str: 版本字符串。若为 None，则从 ENGINE_VERSION 环境变量读取。

    Returns:
        规范化版本字符串，如 "0.17.0"；无法解析时返回 "0.0.0"。
    """
    major, minor = parse_engine_version_tuple(ver_str)
    return f"{major}.{minor}.0"