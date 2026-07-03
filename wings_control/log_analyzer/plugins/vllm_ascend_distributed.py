# -*- coding: utf-8 -*-
"""vLLM-Ascend分布式插件。

针对vLLM-Ascend分布式部署场景的日志分析插件。
"""

from __future__ import annotations

from typing import Dict, List

from .vllm_distributed import VLLMDistributedPlugin


class VLLMAscendDistributedPlugin(VLLMDistributedPlugin):
    """vLLM-Ascend分布式插件 - 继承自vLLM分布式插件，适配昇腾NPU。"""

    def get_log_patterns(self) -> List[Dict]:
        """返回日志匹配模式。"""
        patterns = super().get_log_patterns()

        # 添加昇腾NPU分布式特有模式
        ascend_distributed_patterns = [
            # NPU集群初始化
            {
                "pattern": r"Initializing NPU cluster",
                "phase_code": "ray_cluster_init",
                "progress_calc": 9
            },
            {
                "pattern": r"NPU cluster initialized",
                "phase_code": "ray_cluster_init",
                "progress_calc": 10
            },

            # HCCL分布式初始化
            {
                "pattern": r"Initializing HCCL for distributed",
                "phase_code": "distributed_setup",
                "progress_calc": 21
            },
            {
                "pattern": r"HCCL distributed initialized",
                "phase_code": "distributed_setup",
                "progress_calc": 23
            },

            # NPU设备发现
            {
                "pattern": r"Discovering NPU devices across nodes",
                "phase_code": "worker_registration",
                "progress_calc": 16
            },
            {
                "pattern": r"NPU devices discovered:\s*(\d+)",
                "phase_code": "worker_registration",
                "progress_calc": 17
            }
        ]

        return ascend_distributed_patterns + patterns

    def get_error_patterns(self) -> List[Dict]:
        """返回错误匹配模式。"""
        patterns = super().get_error_patterns()

        # 添加昇腾NPU分布式特有错误模式
        ascend_distributed_errors = [
            {
                "pattern": r"NPU cluster initialization failed",
                "error_type": "NPUClusterInitFailed"
            },
            {
                "pattern": r"HCCL distributed error",
                "error_type": "HCCLDistributedError"
            },
            {
                "pattern": r"NPU device discovery failed",
                "error_type": "NPUDeviceDiscoveryFailed"
            },
            {
                "pattern": r"NPU communication error",
                "error_type": "NPUCommunicationError"
            }
        ]

        return patterns + ascend_distributed_errors