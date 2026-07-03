# -*- coding: utf-8 -*-
"""SGLang单机单GPU插件。

针对SGLang单机单GPU部署场景的日志分析插件。
"""

from __future__ import annotations

from typing import Dict, List

from ..log_analyzer import BaseLogPatternPlugin


class SGLangSingleGPUPlugin(BaseLogPatternPlugin):
    """SGLang单机单GPU插件。"""

    def get_error_patterns(self) -> List[Dict]:
        """返回错误匹配模式。

        每个 pattern 必须足够精确，避免在日志行中仅因提及错误
        类型名称即触发误判。对 Python 异常类型统一要求尾随冒号
        （如 ``RuntimeError:``），确保匹配的是实际异常行而非普通
        日志文本。
        """
        return [
            {
                "pattern": r"Out of memory",
                "error_type": "OOM"
            },
            {
                "pattern": r"RuntimeError:",
                "error_type": "RuntimeError"
            },
            {
                "pattern": r"ValueError:",
                "error_type": "ValueError"
            },
            {
                "pattern": r"AttributeError:",
                "error_type": "AttributeError"
            },
            {
                "pattern": r"(?:Import|ModuleNotFound)Error:",
                "error_type": "ImportError"
            },
            {
                "pattern": r"FileNotFoundError:",
                "error_type": "FileNotFoundError"
            },
            {
                "pattern": r"Application startup failed",
                "error_type": "StartupFailed"
            },
            {
                "pattern": r"Traceback \(most recent call last\):",
                "error_type": "Exception"
            }
        ]

    def get_stage_definitions(self) -> Dict[str, Dict]:
        """返回阶段定义和权重。"""
        return {
            "init": {
                "weight": 5,
                "name": "wings-control初始化"
            },
            "accel_enabling": {
                "weight": 15,
                "name": "加速特性注入"
            },
            "engine_booting": {
                "weight": 10,
                "name": "engine初始化"
            },
            "model_loading": {
                "weight": 30,
                "name": "模型加载"
            },
            "kv_cache_allocating": {
                "weight": 10,
                "name": "KV缓存分配"
            },
            "cuda_graph_capturing": {
                "weight": 28,
                "name": "CUDA图捕获"
            },
            "server_checking": {
                "weight": 1,
                "name": "服务启动"
            },
            "ready": {
                "weight": 1,
                "name": "启动终态"
            }
        }

    def get_log_patterns(self) -> List[Dict]:
        """返回日志匹配模式。"""
        patterns = []
        patterns.extend(self._get_engine_booting_patterns())
        patterns.extend(self._get_model_loading_patterns())
        patterns.extend(self._get_kv_cache_allocating_patterns())
        patterns.extend(self._get_cuda_graph_capturing_patterns())
        patterns.extend(self._get_server_checking_patterns())
        patterns.extend(self._get_ready_patterns())
        return patterns

    def _get_engine_booting_patterns(self) -> List[Dict]:
        """获取engine_booting阶段的日志模式。"""
        return [
            {
                "pattern": r"Attention backend not specified\. Use fa3 backend by default\.",
                "phase_code": "engine_booting",
                "progress_calc": 22
            },
            {
                "pattern": r"Using default HuggingFace chat template with detected content format",
                "phase_code": "engine_booting",
                "progress_calc": 25
            },
            {
                "pattern": r"Mamba selective_state_update backend initialized: triton",
                "phase_code": "engine_booting",
                "progress_calc": 27
            },
            {
                "pattern": r"Init torch distributed begin\.",
                "phase_code": "engine_booting",
                "progress_calc": 28
            },
            {
                "pattern": r"Init torch distributed ends\. elapsed=\d+\.\d+ s",
                "phase_code": "engine_booting",
                "progress_calc": 30
            }
        ]

    def _get_model_loading_patterns(self) -> List[Dict]:
        """获取model_loading阶段的日志模式。"""
        return [
            {
                "pattern": r"Load weight begin\. avail mem=\d+\.\d+ GB",
                "phase_code": "model_loading",
                "progress_calc": 32
            },
            {
                "pattern": r"Loading safetensors checkpoint shards:\s*(\d+)% Completed",
                "phase_code": "model_loading",
                "progress_calc": lambda m, curr: int(36 + (int(m.group(1)) * 0.24))
            },
            {
                "pattern": r"Load weight end\. elapsed=\d+\.\d+ s, type=\S+",
                "phase_code": "model_loading",
                "progress_calc": 60
            }
        ]

    def _get_kv_cache_allocating_patterns(self) -> List[Dict]:
        """获取kv_cache_allocating阶段的日志模式。"""
        return [
            {
                "pattern": r"Using KV cache dtype: torch\.\w+",
                "phase_code": "kv_cache_allocating",
                "progress_calc": 62
            },
            {
                "pattern": r"KV Cache is allocated\. #tokens: \d+, K size: \d+\.\d+ GB, V size: \d+\.\d+ GB",
                "phase_code": "kv_cache_allocating",
                "progress_calc": 68
            },
            {
                "pattern": r"Memory pool end\. avail mem=\d+\.\d+ GB",
                "phase_code": "kv_cache_allocating",
                "progress_calc": 70
            }
        ]

    def _get_cuda_graph_capturing_patterns(self) -> List[Dict]:
        """获取cuda_graph_capturing阶段的日志模式。"""
        return [
            {
                "pattern": r"Capture cuda graph begin\. This can take up to several minutes\. avail mem=\d+\.\d+ GB",
                "phase_code": "cuda_graph_capturing",
                "progress_calc": 72
            },
            {
                "pattern": r"Capturing batches \(bs=\d+ avail_mem=\d+\.\d+ GB\):\s*(\d+)%",
                "phase_code": "cuda_graph_capturing",
                "progress_calc": lambda m, curr: int(74 + (int(m.group(1)) * 0.24))
            },
            {
                "pattern": (
                    r"Capture cuda graph end\. Time elapsed: \d+\.\d+ s\. "
                    r"mem usage=\d+\.\d+ GB\. avail mem=\d+\.\d+ GB\."
                ),
                "phase_code": "cuda_graph_capturing",
                "progress_calc": 98
            }
        ]

    def _get_server_checking_patterns(self) -> List[Dict]:
        """获取server_checking阶段的日志模式。"""
        return [
            {
                "pattern": r"Started server process",
                "phase_code": "server_checking",
                "progress_calc": 99
            },
            {
                "pattern": r"Waiting for application startup\.",
                "phase_code": "server_checking",
                "progress_calc": 99
            },
            {
                "pattern": r"Application startup complete\.",
                "phase_code": "server_checking",
                "progress_calc": 99
            },
            {
                "pattern": r"Uvicorn running on http://0\.0\.0\.0:\d+",
                "phase_code": "server_checking",
                "progress_calc": 99
            }
        ]

    def _get_ready_patterns(self) -> List[Dict]:
        """获取ready阶段的日志模式。"""
        return [
            {
                "pattern": r"The server is fired up and ready to roll!",
                "phase_code": "ready",
                "progress_calc": 100
            },
            {
                "pattern": r"INFO:\s+127\.0\.0\.1:\d+ - \"GET /model_info HTTP/1\.1\" 200 OK",
                "phase_code": "ready",
                "progress_calc": 100
            },
            {
                "pattern": r"INFO:\s+127\.0\.0\.1:\d+ - \"POST /generate HTTP/1\.1\" 200 OK",
                "phase_code": "ready",
                "progress_calc": 100
            }
        ]