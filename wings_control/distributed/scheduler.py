"""分布式任务调度器。

移植自 wings/distributed/scheduler.py，适配 sidecar 包路径。

功能概述:
    根据可配置策略从活跃 Worker 节点中选择最优节点，并在失败时自动重试。
    支持三种调度策略:
      - least_load: 最小负载优先（默认）
      - round_robin: 轮询
      - random: 随机

Sidecar 适配:
    - 包路径从 wings.distributed -> distributed
    - 逻辑与 wings 版本完全一致
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.

import logging
import random
import threading
import time
from typing import Dict, Optional

import requests

from config.settings import settings
from distributed.monitor import MonitorService


class SchedulerPolicy:
    ROUND_ROBIN = "round_robin"
    LEAST_LOAD = "least_load"
    RANDOM = "random"


class TaskScheduler:
    """分布式任务调度器。

    在 Master 节点中使用，根据策略选择 Worker 节点并转发请求。
    内置重试机制（默认最多 3 次，每次间隔 1 秒）。

    Attributes:
        monitor:     MonitorService 实例，提供活跃节点列表
        policy:      当前调度策略
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
    """

    def __init__(self, monitor: MonitorService):
        self.monitor = monitor
        self.policy = SchedulerPolicy.LEAST_LOAD
        self.max_retries = 3
        self.retry_delay = 1
        self._rr_index = 0  # round-robin 轮询计数器
        self._rr_lock = threading.Lock()  # 保护 _rr_index 的并发访问

    @staticmethod
    def start():
        """启动调度器（当前仅记录日志）。"""
        logging.info("Task scheduler started successfully")

    @staticmethod
    def stop():
        """停止调度器（当前仅记录日志）。"""
        logging.info("Task scheduler stopped successfully")

    @staticmethod
    def _node_workload(item: tuple) -> float:
        """提取节点负载值，兼容 dict 和对象两种格式。"""
        _, value = item
        return value["workload"] if isinstance(value, dict) else value.workload

    @staticmethod
    def _least_load(nodes: Dict) -> str:
        """最少负载策略：选择负载最低的节点。"""
        return min(nodes.items(), key=TaskScheduler._node_workload)[0]

    def set_policy(self, policy: str):
        """设置调度策略。"""
        self.policy = policy
        logging.info("Scheduling policy set to: %s", policy)

    def schedule(self, url: str, data: Dict, retries: int = 0) -> dict:
        """调度任务到 Worker 节点。

        Args:
            url:     Worker 端的 API 路径（如 /api/start_engine）
            data:    请求 body
            retries: 当前重试次数（内部递归用）

        Returns:
            dict: Worker 响应的 JSON

        Raises:
            Exception: 达到最大重试次数时抛出
        """
        if retries >= self.max_retries:
            raise Exception("Maximum retry attempts reached")

        node_id = self._select_node()
        if not node_id:
            raise Exception("No available worker nodes")

        for attempt in range(retries, self.max_retries):
            try:
                return self._forward_request(node_id, url, data)
            except Exception as e:
                logging.warning(
                    "Task scheduling failed "
                    "(attempt %d/%d): %s", attempt + 1, self.max_retries, e
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_delay)
                    node_id = self._select_node()
                    if not node_id:
                        raise Exception("No available worker nodes") from e
                else:
                    raise
        raise RuntimeError("Maximum retry attempts reached")

    def select_worker(self) -> Optional[Dict]:
        """选择工作节点并返回完整节点信息。"""
        node_id = self._select_node()
        if not node_id:
            return None
        return self.monitor.get_active_nodes().get(node_id)

    def _round_robin(self, nodes: Dict) -> str:
        """轮询策略：按顺序依次选择节点。"""
        keys = list(nodes.keys())
        with self._rr_lock:
            selected = keys[self._rr_index % len(keys)]
            self._rr_index += 1
        return selected

    def _select_node(self) -> Optional[str]:
        """根据策略选择节点。"""
        active_nodes = self.monitor.get_active_nodes()
        if not active_nodes:
            return None

        if self.policy == SchedulerPolicy.ROUND_ROBIN:
            return self._round_robin(active_nodes)
        elif self.policy == SchedulerPolicy.LEAST_LOAD:
            return self._least_load(active_nodes)
        else:
            return random.choice(list(active_nodes.keys()))

    def _forward_request(self, node_id: str, url: str, data: Dict) -> dict:
        """转发请求到工作节点。"""
        node_info = self.monitor.get_active_nodes().get(node_id)
        if not node_info:
            raise Exception(f"Node {node_id} does not exist")

        node_url = f"http://{node_info['ip']}:{node_info['port']}{url}"
        try:
            response = requests.post(node_url, json=data, timeout=settings.HTTP_REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(
                "Failed to forward request to node %s: %s", node_id, e
            )
            raise
