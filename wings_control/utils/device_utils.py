"""设备级辅助方法，用于硬件能力检查和资源内省。

已重构 ── 从 JSON 文件读取硬件信息，不再依赖 torch/pynvml/torch_npu。

数据来源:
    本模块从 hardware_info.json 文件加载硬件信息（由 hardware_probe.py
    在 GPU/NPU 可访问的容器中预先生成）。JSON 文件路径通过环境变量
    WINGS_HARDWARE_FILE 配置，默认为 /shared-volume/hardware_info.json。

功能概述:
    本模块提供跨平台的设备信息查询功能，支持：
    - NVIDIA GPU (CUDA)
    - 华为昇腾 NPU (Ascend)
    - CPU 回退

主要功能:
    - is_npu_available()       : 检查昇腾 NPU 是否可用
    - get_available_device()   : 返回当前可用设备类型 (cuda/npu/cpu)
    - gpu_count()              : 返回可见 GPU/NPU 数量
    - get_nvidia_gpu_info()    : 获取 NVIDIA GPU 详情列表
    - get_device_info()        : 获取完整硬件环境信息
    - is_h20_gpu()             : 根据显存大小判断 H20 GPU 型号
    - check_pcie_cards()       : 通过 lspci 检测 PCIe 设备

Sidecar 架构契约:
    - 硬件信息来自 JSON 文件，无需 GPU/NPU SDK
    - 硬件探测应优雅降级，不在缺少文件时崩溃
    - 避免对异构节点做硬编码假设
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.

import json
import os
import re
import subprocess
import threading
from typing import List, Dict, Literal, Any
import logging

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "npu", "cpu"]
DEVICE_TO_ENV_NAME = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
}

# ── 设备类型映射（JSON 中的 device 值 → DeviceType） ──────────────────────
_DEVICE_TO_DEVICETYPE = {
    "nvidia": "cuda",
    "ascend": "npu",
}

# ── JSON 硬件信息缓存 ────────────────────────────────────────────────
_hardware_cache: Dict[str, Any] = {}
_hardware_cache_lock = threading.Lock()


def _get_hardware_info() -> Dict[str, Any]:
    """从 JSON 文件加载硬件信息（带缓存）。

    首次调用时读取 JSON 文件并缓存，后续调用直接返回缓存。
    若文件不存在或解析失败，回退到环境变量构建基本信息。

    Returns:
        Dict[str, Any]: 硬件信息字典，包含 device/count/details/units
    """
    if _hardware_cache:
        return _hardware_cache

    with _hardware_cache_lock:
        if _hardware_cache:
            return _hardware_cache

        hw_file = os.getenv(
            "WINGS_HARDWARE_FILE",
            "/shared-volume/hardware_info.json",
        )

        # 策略 1: 从 JSON 文件加载
        if os.path.isfile(hw_file):
            try:
                with open(hw_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "device" in data:
                    details = data.get("details")
                    if not isinstance(details, list):
                        details = []
                    hardware_family = str(data.get("hardware_family") or "").strip()
                    if not details and hardware_family:
                        details = [{"name": hardware_family}]
                    data["details"] = details
                    try:
                        data["count"] = max(int(data.get("count") or 0), 0)
                    except (TypeError, ValueError):
                        data["count"] = 0
                    data.setdefault("units", "GB")
                    _hardware_cache.update(data)
                    logger.info("device_utils: loaded hardware info from %s", hw_file)
                    return _hardware_cache
            except Exception as e:
                logger.warning("device_utils: failed to load %s: %s", hw_file, e)

        # hardware_info.json 不可用时不再从环境变量推断卡型。
        fallback = {"device": "unknown", "count": 0, "details": [], "units": "GB"}
        _hardware_cache.update(fallback)
        logger.warning("device_utils: hardware info unavailable: %s", fallback)
        return _hardware_cache


def _get_device_type_from_hw() -> DeviceType:
    """将 JSON 中的 device 字段（nvidia/ascend）转换为 DeviceType（cuda/npu/cpu）。"""
    hw = _get_hardware_info()
    return _DEVICE_TO_DEVICETYPE.get(hw.get("device", "nvidia"), "cpu")


# ── 设备可用性检查 ────────────────────────────────────────────────────


def is_npu_available() -> bool:
    """检查昇腾 NPU (Ascend) 是否可用。

    基于 JSON 硬件信息中的设备类型判断，不再依赖 torch_npu。

    Returns:
        bool: NPU 可用返回 True，否则返回 False
    """
    hw = _get_hardware_info()
    return hw.get("device") == "ascend" and hw.get("count", 0) > 0


def get_available_device() -> DeviceType:
    """获取当前系统可用的设备类型。

    基于 JSON 硬件信息判断，设备类型映射：
      - nvidia → 'cuda'
      - ascend → 'npu'
      - 未知/无设备 → 'cpu'

    Returns:
        DeviceType: 'cuda'、'npu' 或 'cpu'
    """
    hw = _get_hardware_info()
    count = hw.get("count", 0)
    if count <= 0:
        return "cpu"
    return _get_device_type_from_hw()


def is_device_available(device: str) -> bool:
    """检查指定设备类型是否可用。

    Args:
        device: 设备类型字符串 ('cuda', 'npu', 'cpu')

    Returns:
        bool: 设备可用返回 True
    """
    if device == "cpu":
        return True
    available = get_available_device()
    return available == device


# ── 设备信息查询 ──────────────────────────────────────────────────────


def get_available_device_env_name():
    """获取当前可用设备对应的可见设备环境变量名。

    映射关系：
    - CUDA GPU -> 'CUDA_VISIBLE_DEVICES'
    - 昇腾 NPU -> 'ASCEND_RT_VISIBLE_DEVICES'
    - CPU -> None

    Returns:
        str | None: 环境变量名称字符串，若当前设备为 CPU 则返回 None。
    """
    return DEVICE_TO_ENV_NAME.get(get_available_device())


def gpu_count() -> int:
    """获取当前可见的 GPU/NPU 设备数量。

    基于 JSON 硬件信息中的 count 字段。

    Returns:
        int: 可用设备数量
    """
    hw = _get_hardware_info()
    return hw.get("count", 0)


def get_nvidia_gpu_info() -> List[Dict[str, Any]]:
    """获取所有 NVIDIA GPU 的详细信息列表。

    从 JSON 硬件信息中读取，不再依赖 pynvml。

    Returns:
        List[Dict]: GPU 信息字典列表，每个字典包含:
            {
                "device_id": int,
                "name": str,
                "total_memory": float,   # GB
                "used_memory": float,    # GB
                "free_memory": float,    # GB
                "util": int,             # GPU 利用率 %
                "vendor": str
            }

    注意:
        若当前设备非 NVIDIA 或无详情数据，返回空列表。
    """
    hw = _get_hardware_info()
    if hw.get("device") != "nvidia":
        return []
    return hw.get("details", [])


def get_device_info() -> Dict[str, Any]:
    """检测并汇总当前系统的计算设备信息。

    从 JSON 硬件信息文件读取，返回格式与原始 wings 项目一致。
    设备类型使用 DeviceType 格式（cuda/npu/cpu），而非 hardware_detect
    的内部格式（nvidia/ascend）。

    Returns:
        Dict[str, Any]: 设备信息字典，结构如下：
            {
                "device": str,       # 设备类型 ('cuda' / 'npu' / 'cpu')
                "count": int,        # 可用设备数量
                "details": List[Dict],  # 每张设备的详细信息列表
                "units": str,        # 内存单位，固定为 'GB'
            }
    """
    hw = _get_hardware_info()
    device_type = _get_device_type_from_hw()
    return {
        "device": device_type,
        "count": hw.get("count", 0),
        "details": hw.get("details", []),
        "units": hw.get("units", "GB"),
    }


def is_hf_accelerate_supported(device: str) -> bool:
    """检查指定设备是否支持 HuggingFace Accelerate 库的加速特性。

    目前仅 CUDA GPU 和昇腾 NPU 支持 Accelerate 加速，CPU 不支持。

    Args:
        device: 设备类型字符串，可选值为 'cuda'、'npu'、'cpu'。

    Returns:
        bool: 支持 Accelerate 返回 True，否则返回 False。
    """
    return device == "cuda" or device == "npu"


# ── GPU 型号判断 ──────────────────────────────────────────────────────


def is_h20_gpu(total_memory: float, tolerance_gb: float = 10.0) -> str:
    """根据显存大小判断是否为 H20 系列 GPU。

    Args:
        total_memory: GPU 显存大小 (GB)
        tolerance_gb: 允许的误差范围 (GB)

    Returns:
        str: "H20-96G" 或 "H20-141G"，如果不匹配则返回空字符串
    """
    if abs(total_memory - 96) <= tolerance_gb:
        return "H20-96G"
    elif abs(total_memory - 141) <= tolerance_gb:
        return "H20-141G"
    return ""


def resolve_card_token(hardware_env: Dict[str, Any] = None) -> str:
    """返回小写卡型标识，用于 Smart 三特性白名单匹配。

    来源优先级：
        1. hardware_env.details[0].name（如 "Ascend910B3" → 含 "910b"）
        2. hardware_env.hardware_family（如 "Ascend910B_64G"）
        3. 未显式传入 hardware_env 时，读取 hardware_info.json 的同名字段
        4. ENGINE_VERSION 后缀（只保留 a2/a3 平台兜底，不读取 WINGS_DEVICE_NAME）

    解析失败返回空串。此时 Ascend 白名单各行的 910b/910c **永不匹配 → 整条
    Ascend 白名单 miss**（见需求一 §0.1#2：MaaS 须保证 Ascend 部署的
    hardware_info.json 填 details[0].name/hardware_family，或设置 ENGINE_VERSION 后缀）。

    Args:
        hardware_env: 硬件环境字典（device/count/details/hardware_family）；
                      产出口拿不到时省略，读取 hardware_info.json 后再走
                      ENGINE_VERSION a2/a3 兜底。

    Returns:
        str: 小写卡型标识子串，未知返回 ""。
    """
    def _is_generic_device_name(value: str) -> bool:
        normalized = (value or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        return normalized in {"", "ascend", "npu", "huawei", "huaweiascend", "nvidia", "gpu", "cuda", "unknown"}

    if hardware_env is None:
        hardware_env = _get_hardware_info()

    name = ""
    if hardware_env:
        details = hardware_env.get("details") or []
        if details and isinstance(details[0], dict):
            detail_name = str(details[0].get("name", "")).strip()
            if not _is_generic_device_name(detail_name):
                name = detail_name
        # ── 兜底：hardware_info.json 的 hardware_family 字段 ──
        # K8s sidecar 模式下 hardware_info.json 常缺少 count/details，
        # 但包含 hardware_family（如 "Ascend910B_64G" → 含 "910b"）。
        if not name:
            hw_family = str(hardware_env.get("hardware_family", "")).strip()
            if hw_family:
                name = hw_family.lower()
    name = (name or "").strip().lower()
    if str((hardware_env or {}).get("device") or "").lower() == "nvidia":
        compact = re.sub(r"[^a-z0-9]+", "", name)
        if "h20" in compact and "141" in compact:
            return "h20-141"
        if "h20" in compact and "96" in compact:
            return "h20-96"
        if "rtxpro5000" in compact and "72" in compact:
            return "rtxpro5000-72"
        if "rtxpro5000" in compact and "48" in compact:
            return "rtxpro5000-48"
    if not name:
        from core.version_util import engine_version_platform
        platform = engine_version_platform() or ""
        if platform == "a3":
            return "ascend910c"
        if platform == "a2":
            return "ascend910b"
    return name

# ── PCIe 设备检测（lspci，不依赖 torch） ─────────────────────────────────────


def check_pcie_cards(device_id="d802", subsystem_id="4000"):
    """检查指定的device_id,subsystem_id的pcie设备是否存在

    Args:
        device_id (str): 要查找的目标设备ID，默认值为"d802"
        subsystem_id (str): 要匹配的目标子系统ID，默认值为"4000"

    Returns:
        tuple: (is_exist, count, bdf_list) - 设备是否存在、总数和BDF列表
        - is_exist (bool): 如果找到至少一个匹配设备则返回True
        - count (int): 匹配设备的总数量
        - bdf_list (list): 匹配设备的BDF地址列表

    其他:
        常用的device id/subsystem id 与设备对应关系
        d500/0110  300I Pro标卡
        d500/0100  300I Duo标卡
        d802/3000  910B4 模组
        d802/3005  910B4-1 模组
        d802/4000  300I A2标卡
        d803/3003  Ascend910 (910C模组)
    """
    try:
        result = subprocess.run(
            ['/usr/bin/lspci', '-d', f':{device_id}'],
            capture_output=True, text=True, check=True
        )

        if not result.stdout.strip():
            return False, 0, []

        device_bdfs = []
        for line in result.stdout.strip().split('\n'):
            if line and ':' in line:
                bdf = line.split()[0]
                device_bdfs.append(bdf)

        count = 0
        matched_bdfs = []
        for bdf in device_bdfs:
            detail_result = subprocess.run(
                ['/usr/bin/lspci', '-vvv', '-s', bdf],
                capture_output=True, text=True, check=True
            )
            if f'Device {subsystem_id}' in detail_result.stdout:
                count += 1
                matched_bdfs.append(bdf)

        return count > 0, count, matched_bdfs

    except subprocess.CalledProcessError as e:
        error_msg = str(e)
        if "command not found" in error_msg or "No such file or directory" in error_msg:
            logger.warning("lspci command is not available (install pciutils if PCIe detection is needed)")
            return False, 0, []
        else:
            logger.warning("lspci command execution failed: %s", error_msg)
            return False, 0, []

    except FileNotFoundError:
        logger.warning("lspci command not found (install pciutils if PCIe detection is needed)")
        return False, 0, []

    except ValueError as e:
        logger.error("Result parsing failed: %s", str(e))
        return False, 0, []

    except Exception as e:
        logger.error("Unexpected error occurred: %s", str(e))
        return False, 0, []
