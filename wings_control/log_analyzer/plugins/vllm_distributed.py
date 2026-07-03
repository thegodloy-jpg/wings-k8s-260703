# -*- coding: utf-8 -*-
"""vLLM分布式插件。

针对vLLM分布式部署场景的日志分析插件。
"""

from __future__ import annotations

from typing import Dict, List

from .vllm_single import VLLMSingleGPUPlugin


def _make_offset_calc(orig_fn, offset, ceiling):
    """创建带偏移量和上界的进度计算函数。

    Args:
        orig_fn: 原始进度计算函数
        offset: 进度偏移量
        ceiling: 进度上界

    Returns:
        调整后的进度计算函数
    """
    def adjusted_calc(match, curr):
        return min(orig_fn(match, curr) + offset, ceiling)
    return adjusted_calc


class VLLMDistributedPlugin(VLLMSingleGPUPlugin):
    """vLLM分布式插件 - 继承自单机插件，增加分布式特有阶段。"""

    def get_stage_definitions(self) -> Dict[str, Dict]:
        """返回阶段定义和权重。"""
        # 添加分布式特有阶段
        distributed_stages = {
            "ray_cluster_init": {
                "weight": 5,
                "name": "Ray集群初始化"
            },
            "worker_registration": {
                "weight": 5,
                "name": "Worker节点注册"
            },
            "distributed_setup": {
                "weight": 5,
                "name": "分布式环境配置"
            }
        }

        # 合并阶段定义，调整权重
        # init: 5% -> 3%
        # ray_cluster_init: 5%
        # worker_registration: 5%
        # distributed_setup: 5%
        # accel_enabling: 15% -> 12%
        # engine_booting: 10% -> 8%
        # model_loading: 30% -> 25%
        # model_compiling: 15% -> 12%
        # cuda_graph_capturing: 23% -> 20%
        # server_checking: 1% -> 1%
        # ready: 1% -> 1%

        return {
            "init": {"weight": 3, "name": "wings-control初始化"},
            "ray_cluster_init": {"weight": 5, "name": "Ray集群初始化"},
            "worker_registration": {"weight": 5, "name": "Worker节点注册"},
            "distributed_setup": {"weight": 5, "name": "分布式环境配置"},
            "accel_enabling": {"weight": 12, "name": "加速特性注入"},
            "engine_booting": {"weight": 8, "name": "engine初始化"},
            "model_loading": {"weight": 28, "name": "模型加载"},
            "model_compiling": {"weight": 12, "name": "模型编译"},
            "cuda_graph_capturing": {"weight": 20, "name": "CUDA图捕获"},
            "server_checking": {"weight": 1, "name": "服务启动"},
            "ready": {"weight": 1, "name": "启动终态"}
        }

    def get_log_patterns(self) -> List[Dict]:
        """返回日志匹配模式。"""
        patterns = super().get_log_patterns()

        # 添加分布式特有模式
        distributed_patterns = [
            {"pattern": r"Starting Ray cluster", "phase_code": "ray_cluster_init", "progress_calc": 8},
            {"pattern": r"Ray cluster initialized", "phase_code": "ray_cluster_init", "progress_calc": 10},
            {"pattern": r"Worker node registered", "phase_code": "worker_registration", "progress_calc": 15},
            {"pattern": r"All workers registered", "phase_code": "worker_registration", "progress_calc": 18},
            {"pattern": r"Setting up distributed environment", "phase_code": "distributed_setup", "progress_calc": 20},
            {"pattern": r"Distributed environment ready", "phase_code": "distributed_setup", "progress_calc": 23}
        ]

        # 调整后续阶段的进度值
        # accel_enabling: 23-35%
        # engine_booting: 35-43%
        # model_loading: 43-68%
        # model_compiling: 68-80%
        # cuda_graph_capturing: 80-99%
        # server_checking: 99%
        # ready: 100%

        # 更新现有模式的进度计算
        for pattern in patterns:
            if pattern["phase_code"] == "engine_booting":
                if isinstance(pattern["progress_calc"], int):
                    pattern["progress_calc"] = min(35 + (pattern["progress_calc"] - 20), 43)
            elif pattern["phase_code"] == "model_loading":
                if isinstance(pattern["progress_calc"], int):
                    pattern["progress_calc"] = min(43 + (pattern["progress_calc"] - 30), 68)
                elif callable(pattern["progress_calc"]):
                    pattern["progress_calc"] = _make_offset_calc(
                        pattern["progress_calc"], offset=7, ceiling=68
                    )
            elif pattern["phase_code"] == "model_compiling":
                if isinstance(pattern["progress_calc"], int):
                    pattern["progress_calc"] = min(68 + (pattern["progress_calc"] - 60), 80)
            elif pattern["phase_code"] == "cuda_graph_capturing":
                if isinstance(pattern["progress_calc"], int):
                    pattern["progress_calc"] = min(80 + (pattern["progress_calc"] - 75), 99)
                elif callable(pattern["progress_calc"]):
                    pattern["progress_calc"] = _make_offset_calc(
                        pattern["progress_calc"], offset=4, ceiling=99
                    )
            elif pattern["phase_code"] == "server_checking":
                if isinstance(pattern["progress_calc"], int):
                    pattern["progress_calc"] = 99

        return distributed_patterns + patterns

    def get_error_patterns(self) -> List[Dict]:
        """返回错误匹配模式。"""
        patterns = super().get_error_patterns()

        # 添加分布式特有错误模式
        distributed_errors = [
            {
                "pattern": r"Ray cluster initialization failed",
                "error_type": "RayInitFailed"
            },
            {
                "pattern": r"Worker registration timeout",
                "error_type": "WorkerTimeout"
            },
            {
                "pattern": r"Connection refused",
                "error_type": "ConnectionRefused"
            },
            {
                "pattern": r"NCCL error",
                "error_type": "NCCLError"
            }
        ]

        return patterns + distributed_errors