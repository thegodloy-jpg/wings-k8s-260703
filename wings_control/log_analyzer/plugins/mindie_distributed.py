# -*- coding: utf-8 -*-
"""MindIE 分布式（Ascend）插件。

针对 MindIE 多节点分布式部署场景（Ascend NPU）的日志分析插件。
"""

from __future__ import annotations

from typing import Dict, List

from ..log_analyzer import BaseLogPatternPlugin


class MindIEDistributedPlugin(BaseLogPatternPlugin):
    """MindIE 分布式 Ascend 插件。

    适配 MindIE 引擎在昇腾 NPU 上的多节点分布式部署场景。
    阶段划分（仅 master 节点运行分析器）：
      init(5%) → hccl_init(10%) → engine_booting(20%) →
      model_loading(50%) → server_checking(1%) → ready(1%)
    """

    def get_error_patterns(self) -> List[Dict]:
        """返回错误匹配模式。"""
        return [
            {
                "pattern": r"ConfigManager init exception",
                "error_type": "MindIEConfigError"
            },
            {
                "pattern": r"Failed to initialize BackendConfig from JSON",
                "error_type": "MindIEConfigError"
            },
            {
                "pattern": r"The size of npuDeviceIds \(subset\) does not equal to worldSize",
                "error_type": "MindIENpuDeviceIdsMismatch"
            },
            {
                "pattern": r"\[mindie\] ERROR: daemon exited with code",
                "error_type": "MindIEDaemonFailed"
            },
            {
                "pattern": r"Traceback \(most recent call last\):",
                "error_type": "Exception"
            },
            {
                "pattern": r"RuntimeError:",
                "error_type": "RuntimeError"
            },
            {
                "pattern": r"HCCL communi[ca]+tion error",
                "error_type": "HCCLError"
            },
            {
                "pattern": r"Aborted \(core dumped\)",
                "error_type": "CoreDump"
            },
            {
                "pattern": r"OOM|out of memory|Cannot allocate memory",
                "error_type": "OOM"
            },
        ]

    def get_stage_definitions(self) -> Dict[str, Dict]:
        """返回阶段定义和权重。"""
        return {
            "init": {
                "weight": 5,
                "name": "wings-control初始化"
            },
            "accel_enabling": {
                "weight": 4,
                "name": "加速特性注入"
            },
            "hccl_init": {
                "weight": 10,
                "name": "HCCL分布式初始化"
            },
            "engine_booting": {
                "weight": 20,
                "name": "MindIE引擎启动"
            },
            "model_loading": {
                "weight": 55,
                "name": "模型加载"
            },
            "server_checking": {
                "weight": 3,
                "name": "服务启动"
            },
            "ready": {
                "weight": 3,
                "name": "启动终态"
            },
        }

    def get_log_patterns(self) -> List[Dict]:
        """返回日志匹配模式。"""
        patterns = []
        patterns.extend(self._get_hccl_init_patterns())
        patterns.extend(self._get_engine_booting_patterns())
        patterns.extend(self._get_model_loading_patterns())
        patterns.extend(self._get_server_checking_patterns())
        patterns.extend(self._get_ready_patterns())
        return patterns

    def _get_hccl_init_patterns(self) -> List[Dict]:
        """获取 hccl_init 阶段的日志模式。"""
        return [
            {
                "pattern": r"export RANK_TABLE_FILE=",
                "phase_code": "hccl_init",
                "progress_calc": 10,
            },
            {
                "pattern": r"chmod 640 .*hccl_ranktable\.json",
                "phase_code": "hccl_init",
                "progress_calc": 11,
            },
            {
                "pattern": r"export HCCL_WHITELIST_DISABLE=1",
                "phase_code": "hccl_init",
                "progress_calc": 12,
            },
        ]

    def _get_engine_booting_patterns(self) -> List[Dict]:
        """获取 engine_booting 阶段的日志模式。"""
        return [
            {
                "pattern": r"\[mindie\] Loaded original config\.json",
                "phase_code": "engine_booting",
                "progress_calc": 20,
            },
            {
                "pattern": r"\[mindie\] config\.json merge-updated successfully",
                "phase_code": "engine_booting",
                "progress_calc": 24,
            },
            {
                "pattern": r"ConfigManager: Load Config from",
                "phase_code": "engine_booting",
                "progress_calc": 27,
            },
            {
                "pattern": r"LogLevelDynamicHandler start",
                "phase_code": "engine_booting",
                "progress_calc": 28,
            },
            {
                "pattern": r"\[mindie\] Daemon started as PID \d+",
                "phase_code": "engine_booting",
                "progress_calc": 30,
            },
            {
                "pattern": r"g_mainPid = \d+",
                "phase_code": "engine_booting",
                "progress_calc": 32,
            },
        ]

    def _get_model_loading_patterns(self) -> List[Dict]:
        """获取 model_loading 阶段的日志模式。"""
        return [
            {
                "pattern": r"Start to load model",
                "phase_code": "model_loading",
                "progress_calc": 35,
            },
            {
                "pattern": r"Loading model weights",
                "phase_code": "model_loading",
                "progress_calc": 40,
            },
            {
                "pattern": r"Loading safetensors checkpoint",
                "phase_code": "model_loading",
                "progress_calc": 42,
            },
            {
                "pattern": r"model loading progress.*?(\d+)%",
                "phase_code": "model_loading",
                "progress_calc": lambda m, curr: int(42 + int(m.group(1)) * 0.35),
            },
            {
                "pattern": r"Finish loading model",
                "phase_code": "model_loading",
                "progress_calc": 78,
            },
            {
                "pattern": r"Model loaded successfully",
                "phase_code": "model_loading",
                "progress_calc": 80,
            },
            {
                "pattern": r"Warmup finished",
                "phase_code": "model_loading",
                "progress_calc": 85,
            },
        ]

    def _get_server_checking_patterns(self) -> List[Dict]:
        """获取 server_checking 阶段的日志模式。"""
        return [
            {
                "pattern": r"Started server process",
                "phase_code": "server_checking",
                "progress_calc": 99,
            },
            {
                "pattern": r"Waiting for application startup\.",
                "phase_code": "server_checking",
                "progress_calc": 99,
            },
            {
                "pattern": r"Application startup complete\.",
                "phase_code": "server_checking",
                "progress_calc": 99,
            },
            {
                "pattern": r"Uvicorn running on http://0\.0\.0\.0:\d+",
                "phase_code": "server_checking",
                "progress_calc": 99,
            },
        ]

    def _get_ready_patterns(self) -> List[Dict]:
        """获取 ready 阶段的日志模式。"""
        return [
            {
                "pattern": r"MindIE service started successfully",
                "phase_code": "ready",
                "progress_calc": 100,
            },
            {
                "pattern": r"The server is fired up and ready to roll!",
                "phase_code": "ready",
                "progress_calc": 100,
            },
            {
                "pattern": r"INFO:\s+127\.0\.0\.1:\d+ - \"GET /model_info HTTP/1\.1\" 200 OK",
                "phase_code": "ready",
                "progress_calc": 100,
            },
        ]
