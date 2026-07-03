# -*- coding: utf-8 -*-
"""vLLM-Ascend单机插件。

针对vLLM-Ascend单机部署场景的日志分析插件。
"""

from __future__ import annotations

from typing import Dict, List

from .vllm_single import VLLMSingleGPUPlugin


class VLLMAscendSinglePlugin(VLLMSingleGPUPlugin):
    """vLLM-Ascend单机插件 - 继承自vLLM单机插件，适配昇腾NPU。"""

    def get_log_patterns(self) -> List[Dict]:
        """返回日志匹配模式。"""
        patterns = super().get_log_patterns()

        # 添加昇腾NPU特有模式
        ascend_patterns = [
            # NPU初始化
            {
                "pattern": r"Platform plugin ascend is activated",
                "phase_code": "engine_booting",
                "progress_calc": 20
            }
        ]

        return ascend_patterns + patterns

    def get_error_patterns(self) -> List[Dict]:
        """返回错误匹配模式。"""
        patterns = super().get_error_patterns()

        # 添加昇腾NPU特有错误模式
        ascend_errors = [
            {
                "pattern": r"NPU out of memory",
                "error_type": "NPUOOM"
            },
            {
                "pattern": r"HCCL error",
                "error_type": "HCCLError"
            },
            {
                "pattern": r"Ascend runtime error",
                "error_type": "AscendRuntimeError"
            },
            {
                "pattern": r"Torch NPU error",
                "error_type": "TorchNPUError"
            },
            {
                "pattern": r"CANN error",
                "error_type": "CANNError"
            }
        ]

        return patterns + ascend_errors