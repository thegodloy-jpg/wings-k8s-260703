"""分布式主节点（Master）服务。

移植自 wings/distributed/master.py，适配 sidecar 脚本生成模式。

功能概述:
    Master 是分布式系统的入口和协调中心，通过 FastAPI 暴露:
      POST /api/nodes/register  - Worker 注册
      GET  /api/nodes           - 查询所有活跃节点
      POST /api/start_engine    - 启动推理引擎（分布式/单机）
      POST /api/inference       - 分发推理任务
      POST /api/heartbeat       - 接收心跳

核心流程 (start_engine):
    1. 分布式模式: 解析 nodes 参数，通过 ThreadPoolExecutor 并发
       向每个 Worker 发送启动请求，Worker 生成脚本写入共享卷
    2. 单机模式: 调度器选最优 Worker，转发启动请求

与 wings 版本的核心差异:
    - wings: Worker 调用 start_engine_service() 直接启动进程
    - sidecar: Worker 调用 build_start_script() 生成脚本 -> 写入共享卷
    - 新增: Master 在分发时自动计算并注入 nnodes/node_rank/head_node_addr

Sidecar 架构契约:
    - Master 负责分布式协调，不直接操作引擎进程
    - 所有引擎启动通过 Worker -> 共享卷 -> engine 容器的链路完成
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.

from __future__ import annotations

import concurrent.futures
import copy
import json
import logging
import socket
import time
from pathlib import Path
import os
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from distributed.monitor import MonitorService
from distributed.scheduler import TaskScheduler
from distributed.worker import HeartbeatRequest
from utils.env_utils import get_master_port

# 注意：不在模块级别调用 basicConfig，避免与 setup_root_logging 冲突
logger = logging.getLogger(__name__)


app = FastAPI(title="Wings Distributed Inference Master Node")


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class NodeInfo(BaseModel):
    node_id: str
    ip: str
    port: int
    status: str = "active"
    last_heartbeat: float = 0.0  # 由注册逻辑在创建时显式设置为 time.time()
    workload: float = 0.0


class RegisterRequest(BaseModel):
    node_id: str
    ip: str
    port: int


class InferenceRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    input_data: str
    parameters: Optional[Dict] = None


class StartEngineRequest(BaseModel):
    engine: str
    params: Dict[str, Any]


# ---------------------------------------------------------------------------
# Global service instances — lazy-initialized in start_master()
# ---------------------------------------------------------------------------

monitor_service: MonitorService = None   # type: ignore[assignment]
task_scheduler: TaskScheduler = None     # type: ignore[assignment]

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/api/nodes/register")
async def register_node(request: RegisterRequest):
    """工作节点注册接口。"""
    monitor_service.register_node(request.node_id, request.ip, request.port)
    monitor_service.update_heartbeat(request.node_id, 0.0)
    logger.info(
        "Node registered successfully: %s (%s:%s)",
        request.node_id,
        request.ip,
        request.port,
    )
    return {"status": "success"}


@app.get("/api/nodes")
async def get_nodes():
    """获取所有活跃节点状态。"""
    return {"nodes": list(monitor_service.get_active_nodes().values())}


@app.post("/api/start_engine")
async def start_engine(request: StartEngineRequest):
    """启动引擎服务。

    根据 params.distributed 决定走分布式还是单机模式:
      - 分布式: 向所有节点并发分发，自动注入 nnodes/node_rank/head_node_addr
      - 单机:   调度器选最优 Worker 转发
    """
    try:
        params = request.model_dump()
        logger.info("Received engine start request")

        if params["params"].get("distributed"):
            return await _handle_distributed_mode(params)
        else:
            return await _handle_single_mode(params)

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


# ===== 分布式模式处理 =====

async def _handle_distributed_mode(params: dict):
    """处理分布式启动逻辑。

    与 wings 版本的差异:
      - wings: 需要 params.nodes（逗号分隔的 IP 列表）
      - sidecar: 优先从 params.nodes 获取，若无则从已注册节点自动获取
      - 新增: 自动计算并注入 nnodes/node_rank/head_node_addr 到每个节点的 params
    """
    # 获取节点列表: 优先使用显式指定，否则取所有已注册节点
    nodes_str = params["params"].get("nodes")
    active_nodes = monitor_service.get_active_nodes()

    if nodes_str:
        node_ips = [_resolve_to_ip(h.strip()) for h in nodes_str.split(",")]
    else:
        # 从已注册节点自动获取 IP 列表
        node_ips = [info["ip"] for info in active_nodes.values()]
        if len(node_ips) < 1:
            raise HTTPException(
                status_code=400,
                detail="No nodes available for distributed mode",
            )

    # 校验 node_ips 拓扑一致性：去重并检查是否有重复 IP
    unique_ips = list(dict.fromkeys(node_ips))  # 保持顺序去重
    if len(unique_ips) != len(node_ips):
        logger.warning(
            "Duplicate IPs detected in node list (%d -> %d unique), deduplicating",
            len(node_ips), len(unique_ips),
        )
        node_ips = unique_ips

    # 校验显式指定的 nnodes 与实际节点数是否一致
    # 如果不一致，自动修正为实际节点数，避免下游 adapter 层
    # （如 mindie_adapter._resolve_distributed_topology）严格校验时抛 ValueError
    expected_nnodes = params["params"].get("nnodes")
    if expected_nnodes is not None and int(expected_nnodes) != len(node_ips):
        logger.warning(
            "nnodes mismatch: params.nnodes=%s but %d node IPs provided, "
            "auto-correcting nnodes to %d",
            expected_nnodes, len(node_ips), len(node_ips),
        )
        params["params"]["nnodes"] = len(node_ips)

    nodes_to_port = _map_active_nodes(active_nodes)

    return {
        "results": await _distribute_requests(node_ips, nodes_to_port, params)
    }


def _resolve_to_ip(host: str) -> str:
    """将 hostname 解析为 IP；解析失败时原样返回。"""
    try:
        return socket.gethostbyname(host)
    except socket.error:
        return host


def _map_active_nodes(active_nodes: dict) -> dict:
    """构建 IP → port 映射表。"""
    return {node["ip"]: node["port"] for node in active_nodes.values()}


async def _distribute_requests(
    node_ips: list, node_map: dict, params: dict
) -> dict:
    """向所有节点并发分发启动请求。

    为每个节点自动注入分布式参数:
      - nnodes:         总节点数
      - node_rank:      当前节点编号（按 node_ips 顺序）
      - head_node_addr: 第一个节点的 IP（rank 0）
    """
    results = {}
    nnodes = len(node_ips)
    head_addr = node_ips[0]  # rank 0 的 IP 作为 head

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_node = {}
        for rank, node_ip in enumerate(node_ips):
            # 为每个节点构造独立的 params 副本，注入分布式信息
            node_params = _inject_distributed_params_with_topology(
                params,
                nnodes=nnodes,
                node_rank=rank,
                head_addr=head_addr,
                node_ips=node_ips,
            )
            port = node_map.get(node_ip)
            if port is None:
                logger.error(
                    "No registered port for node %s (rank %d), skipping",
                    node_ip, rank,
                )
                results[node_ip] = {"status": "error", "detail": f"node {node_ip} not in active registry"}
                continue
            future = executor.submit(
                _send_single_request,
                node_ip,
                port,
                node_params,
            )
            future_to_node[future] = node_ip

        for future in concurrent.futures.as_completed(future_to_node):
            node_ip = future_to_node[future]
            results[node_ip] = _process_node_response(future, node_ip)

    return results


def _inject_distributed_params(
    params: dict, nnodes: int, node_rank: int, head_addr: str
) -> dict:
    """为单个节点注入分布式参数。

    这是与 wings 的核心对齐点: 原来这些值由 K8s YAML 注入环境变量，
    现在改为 Master 在分发时动态计算并写入 params。
    """
    node_params = copy.deepcopy(params)
    node_params["params"]["distributed"] = True
    node_params["params"]["nnodes"] = nnodes
    node_params["params"]["node_rank"] = node_rank
    node_params["params"]["head_node_addr"] = head_addr
    # nodes 字段保留以兼容 wings 的 adapter 逻辑
    node_params["params"]["nodes"] = head_addr  # 仅保留 head 的 IP 供 adapter 使用
    return node_params


def _inject_distributed_params_with_topology(
    params: dict, nnodes: int, node_rank: int, head_addr: str, node_ips: list[str]
) -> dict:
    """Inject distributed params and preserve the full cluster topology."""
    resolved_head_addr = params["params"].get("master_ip") or head_addr
    node_params = _inject_distributed_params(
        params,
        nnodes=nnodes,
        node_rank=node_rank,
        head_addr=resolved_head_addr,
    )
    all_node_ips = ",".join(node_ips)
    node_params["params"]["node_ips"] = all_node_ips
    node_params["params"]["nodes"] = all_node_ips
    node_params["params"]["master_ip"] = node_params["params"].get("master_ip") or resolved_head_addr
    node_params["params"]["ray_head_ip"] = (
        node_params["params"].get("ray_head_ip") or node_params["params"]["master_ip"]
    )
    return node_params


def _send_single_request(node_ip: str, port: int, params: dict):
    """向单个 Worker 节点发送启动请求。"""
    worker_url = f"http://{node_ip}:{port}"
    return requests.post(
        f"{worker_url}/api/start_engine",
        json={"engine": params["engine"], "params": params["params"]},
        timeout=120,
    )


def _process_node_response(future, node_ip: str) -> dict:
    """处理单个节点的响应结果。"""
    try:
        response = future.result()
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error("Node %s startup failed: %s", node_ip, e)
        return {"status": "error", "detail": str(e)}


# ===== 单机模式处理 =====

async def _handle_single_mode(params: dict):
    """处理单机模式启动逻辑。"""
    worker_node = task_scheduler.select_worker()
    if not worker_node:
        raise HTTPException(
            status_code=503, detail="No available worker nodes"
        )
    response = _send_single_request(
        worker_node["ip"], worker_node["port"], params
    )
    try:
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/api/inference")
async def distribute_inference(request: InferenceRequest):
    """分发推理任务。"""
    try:
        result = task_scheduler.schedule("/api/inference", request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/api/heartbeat")
async def receive_heartbeat(request: HeartbeatRequest):
    """接收工作节点心跳。"""
    monitor_service.update_heartbeat(request.node_id, request.workload)
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def start_master():
    """启动主节点服务。"""
    global monitor_service, task_scheduler

    # 延迟初始化：仅在未被 _start_master_api_thread 预初始化时创建
    if monitor_service is None:
        monitor_service = MonitorService()
        monitor_service.start()
    if task_scheduler is None:
        task_scheduler = TaskScheduler(monitor_service)
        task_scheduler.start()

    config_path = Path(__file__).parent.parent / "config" / "defaults" / "distributed_config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    # 优先使用 COORDINATOR_PORT，避免与 HCCL 的 MASTER_PORT 冲突。
    # MASTER_PORT 在 MindIE 中用于 HCCL 集合通信（默认 27070），
    # 而此处是 wings 分布式协调 API 端口（默认 16000），二者不同。
    coordinator_port = os.getenv("COORDINATOR_PORT")
    if coordinator_port:
        try:
            master_port = int(coordinator_port)
        except ValueError:
            logger.warning("Invalid COORDINATOR_PORT=%r, falling back", coordinator_port)
            coordinator_port = None
    if not coordinator_port:
        master_port = get_master_port()
        if not master_port:
            master_port = cfg["master"]["port"]

    uvicorn.run(app, host="0.0.0.0", port=master_port)


if __name__ == "__main__":
    start_master()
