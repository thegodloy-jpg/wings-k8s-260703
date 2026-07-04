# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""硬件环境静态探测模块 (core/hardware_detect.py)。

为 config_loader 提供设备类型和数量信息。

工作原理:
    在 sidecar 架构中，推理引擎和 sidecar 分属不同容器，因此 sidecar
    无法直接访问 GPU/NPU 硬件。本模块改为从 hardware_info.json 读取
    设备类型/型号信息，而不是直接调用 torch/pynvml 进行探测。

支持的输入:
    - WINGS_HARDWARE_FILE: 硬件信息 JSON 文件路径
      （优先，默认 /shared-volume/hardware_info.json）
    - device_count: 由调用方从 --device-count 传入

探测口径:
    硬件型号只来自 JSON 文件；设备数量只来自 device_count。

输出格式示例::

    {
        "device": "nvidia" | "ascend" | "unknown",
        "count": int,
        "details": [
            {
                "device_id": int,
                "name": str,
                "total_memory": float,   # GB
                "used_memory": float,    # GB
                "free_memory": float,    # GB
                "util": int,             # GPU 利用率 % (仅 NVIDIA)
                "vendor": str            # "Nvidia" 或 "Ascend"
            }
        ],
        "units": "GB"
    }

Sidecar 契约:
    - 探测应使用最佳努力策略，不应因任何探测失败而崩溃
    - 避免破坏异构节点（混合 GPU/NPU 环境）的兼容性
"""

import json
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _normalize_device(raw: str) -> str:
    """将用户输入的设备类型字符串标准化为内部统一格式。

    支持多种别名映射：
      - 'nvidia'/'gpu'/'cuda' -> 'nvidia'
      - 'ascend'/'npu'        -> 'ascend'
      - 其他未识别的值回退到 'nvidia'

    Args:
        raw: 原始设备类型字符串（大小写不敏感）

    Returns:
        str: 标准化后的设备类型 ('nvidia' 或 'ascend')
    """
    val = (raw or "").strip().lower()
    mapping = {
        "nvidia": "nvidia",
        "gpu": "nvidia",
        "cuda": "nvidia",
        "ascend": "ascend",
        "npu": "ascend",
    }
    return mapping.get(val, "unknown")


def _parse_count(raw: str) -> int:
    """解析设备数量字符串，确保返回至少为 1 的正整数。

    异常输入（非数字字符串、负数、零）均回退到默认值 1，
    避免配置错误导致 launcher 崩溃。

    Args:
        raw: 原始设备数量字符串

    Returns:
        int: 解析后的设备数量（>= 1）
    """
    try:
        value = int((raw or "1").strip())
        return value if value > 0 else 1
    except Exception as _:
        return 1


def _load_hardware_from_file(file_path: str, device_count: int) -> Dict[str, Any]:
    """从 JSON 文件加载完整硬件信息。

    JSON 文件应包含与原始 wings 项目 detect_hardware() 输出一致的格式：
      - device:  设备类型 ('nvidia' 或 'ascend')
      - count:   设备数量
      - details: 每张卡的详细信息列表
      - units:   显存单位 ('GB')

    Args:
        file_path: JSON 文件的绝对路径

    Returns:
        Dict[str, Any]: 硬件信息字典

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误
        ValueError: 缺少必要字段或字段类型错误
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 校验必要字段
    if not isinstance(data, dict):
        raise ValueError("hardware info JSON must be a dict")
    if "device" not in data:
        raise ValueError("hardware info JSON must contain 'device' field")

    # 标准化 device 字段
    data["device"] = _normalize_device(data["device"])
    data.setdefault("units", "GB")

    # 确保 count 为正整数
    details = data.get("details")
    if not isinstance(details, list):
        details = []
    hardware_family = str(data.get("hardware_family") or "").strip()
    if not details and hardware_family:
        details = [{"name": hardware_family}]
    data["details"] = details

    data["count"] = _parse_count(str(device_count))

    return data


def detect_hardware(device_count: int = 1) -> Dict[str, Any]:
    """探测硬件环境信息。

    硬件型号只从 JSON 文件获取（由 WINGS_HARDWARE_FILE 指定路径，
    默认 /shared-volume/hardware_info.json）；设备数量只使用调用方传入的
    device_count。

    JSON 文件方式可提供每张卡的完整信息（型号、显存、利用率等），
    激活 VRAM 校验和 CUDA Graph 尺寸自动计算等下游功能。

    Returns:
        Dict[str, Any]: 硬件环境描述字典，包含以下字段：
            - device:  设备类型 ('nvidia'、'ascend' 或 'unknown')
            - count:   设备数量
            - details: 设备详情列表
            - units:   显存单位 (GB)
    """
    # ── 策略 1: 从 JSON 文件加载 ──────────────────────────────────────────
    hw_file = os.getenv(
        "WINGS_HARDWARE_FILE",
        "/shared-volume/hardware_info.json",
    )
    if os.path.isfile(hw_file):
        try:
            result = _load_hardware_from_file(hw_file, device_count)
            logger.info(
                "Loaded hardware info from file %s: device=%s, count=%d, cards=%d",
                hw_file, result["device"], result["count"], len(result["details"]),
            )
            for i, detail in enumerate(result["details"]):
                logger.info(
                    "  Card %d: name=%s, total=%.2f GB, free=%.2f GB, used=%.2f GB%s",
                    detail.get("device_id", i),
                    detail.get("name", "unknown"),
                    detail.get("total_memory", 0),
                    detail.get("free_memory", 0),
                    detail.get("used_memory", 0),
                    ", util=%d%%" % detail["util"] if "util" in detail else "",
                )
            return result
        except Exception as e:
            logger.warning(
                "Failed to load hardware file %s: %s; using unknown hardware context",
                hw_file, e,
            )

    result = {"device": "unknown", "count": _parse_count(str(device_count)), "details": [], "units": "GB"}
    logger.warning("Hardware info file %s unavailable; using %s", hw_file, result)
    return result

