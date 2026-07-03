"""分布式节点健康监控服务。

移植自 wings/distributed/monitor.py，适配 sidecar 包路径。

功能概述:
    在 Master 节点中运行，线程安全地管理所有 Worker 节点的状态。
    - 注册节点: 创建 NodeStatus 记录节点 IP、端口、状态、负载
    - 更新心跳: Worker 定期上报心跳，重置丢失计数和更新负载
    - 健康检查: 后台线程定期巡检，连续丢失 max_missed_heartbeats 次后移除节点

Sidecar 适配:
    - 包路径从 wings.distributed → distributed
    - 逻辑与 wings 版本完全一致，无其他改动
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.

import logging
import threading
import time
from typing import Dict


class NodeStatus:
    """单个工作节点的运行时状态快照。

    Attributes:
        node_id:           节点唯一标识（通常为 worker_<uuid>）
        ip:                节点 IP 地址
        port:              节点 Worker API 端口
        last_heartbeat:    最近一次心跳时间戳
        missed_heartbeats: 连续丢失心跳计数
        status:            节点状态：active / inactive / failed
        workload:          节点当前负载（0.0 ~ 1.0）
    """

    def __init__(self, node_id: str, ip: str, port: int):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.last_heartbeat = time.time()
        self.missed_heartbeats = 0
        self.status = "active"
        self.workload = 0.0


class MonitorService:
    """分布式节点健康监控服务。

    在 Master 节点的 main 线程中初始化，通过后台守护线程持续巡检
    所有已注册 Worker 节点的心跳状态。

    线程安全：所有对 self.nodes 的读写均在 self.lock 保护下进行。

    默认配置:
        - check_interval:          30 秒巡检一次
        - max_missed_heartbeats:   连续丢失 10 次后移除节点（10×30s ≈ 5 分钟）
    """

    def __init__(self):
        self.nodes: Dict[str, NodeStatus] = {}
        self.lock = threading.Lock()
        self.check_interval = 30
        self.max_missed_heartbeats = 10  # 10×30s = ~5 分钟剔除死节点
        self._stop_event = threading.Event()
        self.thread = None

    def register_node(self, node_id: str, ip: str, port: int):
        """注册新节点。"""
        with self.lock:
            if node_id not in self.nodes:
                self.nodes[node_id] = NodeStatus(node_id, ip, port)
                logging.info("New node registered: %s (%s:%d)", node_id, ip, port)

    def update_heartbeat(self, node_id: str, workload: float = 0.0):
        """更新节点心跳。"""
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = time.time()
                self.nodes[node_id].workload = workload
                self.nodes[node_id].missed_heartbeats = 0
                self.nodes[node_id].status = "active"
            else:
                logging.warning("Unknown node heartbeat: %s", node_id)

    def start(self):
        """启动后台监控线程。"""
        self._stop_event.clear()
        self.thread = threading.Thread(
            target=self._check_nodes,
            daemon=True,
        )
        self.thread.start()
        logging.info("Monitoring service started")

    def stop(self):
        """停止监控线程。"""
        self._stop_event.set()
        if self.thread:
            self.thread.join()
        logging.info("Monitoring service stopped")

    def get_active_nodes(self) -> Dict[str, dict]:
        """获取所有活跃节点（返回普通 dict 以便 JSON 序列化）。"""
        with self.lock:
            return {
                node_id: {
                    "node_id": status.node_id,
                    "ip": status.ip,
                    "port": status.port,
                    "status": status.status,
                    "workload": status.workload,
                    "last_heartbeat": status.last_heartbeat,
                }
                for node_id, status in self.nodes.items()
                if status.status == "active"
            }

    def _check_nodes(self):
        """定期巡检节点心跳，移除长期无响应的节点。"""
        while not self._stop_event.is_set():
            current_time = time.time()
            with self.lock:
                for node_id, status in list(self.nodes.items()):
                    time_since_last = current_time - status.last_heartbeat
                    if time_since_last > self.check_interval * 1.5:
                        status.missed_heartbeats += 1
                        logging.warning(
                            "Node %s missed heartbeat #%d",
                            node_id, status.missed_heartbeats,
                        )
                        if status.missed_heartbeats >= self.max_missed_heartbeats:
                            del self.nodes[node_id]
                            logging.error(
                                "Node %s removed due to %d "
                                "consecutive missed heartbeats",
                                node_id, self.max_missed_heartbeats,
                            )
            time.sleep(self.check_interval)
