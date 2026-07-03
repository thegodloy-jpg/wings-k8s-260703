"""分布式工作节点（Worker）服务。

移植自 wings/distributed/worker.py，适配 sidecar 脚本生成模式。

功能概述:
    每个 Worker 节点的职责:
    1. 启动时向 Master 注册自身 IP 和端口
    2. 启动后台线程定期发送心跳
    3. 暴露 /api/start_engine 接口，收到请求后:
       - 调用 engine adapter 的 build_start_script() 生成 bash 脚本
       - 将脚本写入共享卷 /shared-volume/start_command.sh
       - engine 容器读取并执行脚本

与 wings 版本的核心差异:
    - wings: Worker 直接调用 start_engine_service() 通过 subprocess 启动引擎
    - sidecar: Worker 只生成脚本写入共享卷，由 engine 容器执行
    - 新增 /api/node_info 接口，供 Master 查询本节点分布式信息

Sidecar 架构契约:
    - Worker 不直接启动引擎进程
    - 脚本写入路径由 settings.SHARED_VOLUME_PATH 控制
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.

from __future__ import annotations

import atexit
import dataclasses as _dc
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from config.settings import settings
from utils.env_utils import get_local_ip, get_master_ip, get_master_port, get_worker_port
from utils.file_utils import safe_write_file

# 注意：不在模块级别调用 basicConfig，避免与 setup_root_logging 冲突
logger = logging.getLogger(__name__)

app = FastAPI(title="Wings Distributed Inference Worker Node")


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class HeartbeatRequest(BaseModel):
    node_id: str
    workload: float


class InferenceRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    input_data: str
    parameters: Optional[Dict] = None


class EngineStartRequest(BaseModel):
    engine: str
    params: Dict[str, Any]


# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------


class WorkerConfig:
    """Worker 节点配置。

    从环境变量和 distributed_config.json 加载 Master 地址和本机端口。

    Attributes:
        node_id:            唯一标识 (worker_<uuid>)
        master_ip:          Master 节点 IP
        master_url:         Master API 完整 URL
        ip:                 本机 IP
        port:               Worker API 端口
        heartbeat_interval: 心跳间隔（秒）
    """

    def __init__(self, master_ip: str | None = None):
        self.node_id = "worker_" + str(uuid.uuid4())
        self.master_ip = master_ip if master_ip else get_master_ip()
        self._load_config(self.master_ip)
        self.heartbeat_interval = 30

    def _load_config(self, master_ip: str | None = None):
        config_path = (
            Path(__file__).parent.parent / "config" / "defaults" / "distributed_config.json"
        )
        try:
            with open(config_path) as f:
                _config = json.load(f)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Distributed config not found: {config_path}. "
                "Ensure infer-control-sidecar-unified is installed correctly."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed JSON in {config_path}: {exc}"
            ) from exc

        self.ip = get_local_ip()

        # 优先读 COORDINATOR_PORT，避免与 HCCL 的 MASTER_PORT 混淆。
        # get_master_port() 读取的 MASTER_PORT 本意用于 HCCL（默认 27070），
        # 此处需要的是 wings 协调 API 端口（默认 16000），二者不同。
        # 不再回退到 get_master_port()/(MASTER_PORT)，避免与 HCCL 端口冲突。
        coordinator_port = os.getenv("COORDINATOR_PORT")
        if not coordinator_port:
            coordinator_port = _config["master"]["port"]
        self.master_url = (
            f"http://{master_ip or _config['master']['host']}:{coordinator_port}"
        )

        worker_port = get_worker_port()
        if not worker_port:
            worker_port = _config["workers"]["port"]
        self.port = worker_port


# Global config — initialised by start_worker() / __main__
config: WorkerConfig | None = None
_config_lock = threading.Lock()  # 保护 config 的并发访问


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/start_engine")
async def start_engine_api(request: EngineStartRequest):
    """收到 Master 的引擎启动指令后，走完整 launcher 管线并写入共享卷。

    流程（与 standalone 模式的 build_launcher_plan 完全一致）:
      1. 从 request.params 重建 LaunchArgs（包含 Master 注入的分布式参数）
      2. 推导 PortPlan
      3. 调用 build_launcher_plan() 走硬件探测 → 配置合并 → adapter 脚本生成
      4. 将完整 bash 脚本写入 /shared-volume/start_command.sh
      5. engine 容器轮询该文件并执行

    与直接调用 start_engine_service() 的区别:
      - build_launcher_plan() 会执行 detect_hardware()、load_and_merge_configs()
      - 对不同硬件（GPU/NPU）的 Worker 节点能正确适配
      - host/port 注入逻辑按 node_rank 自动处理（rank0 绑定端口，其余不绑定）
    """
    try:
        from core.start_args_compat import LaunchArgs
        from core.port_plan import derive_port_plan
        from core.wings_entry import build_launcher_plan

        # ---- 1. 从 params 重建 LaunchArgs ----
        # 注意：仅提取 LaunchArgs 已定义的字段。engine_config 也属于
        # LaunchArgs 的可选透传字段，用于保持 Master 下发的上层 vLLM 参数
        # 在所有节点一致；Worker 仍会在 build_launcher_plan() 中补充本地硬件
        # 探测与 rank 专属 host/port/headless 调整。
        la_fields = {f.name for f in _dc.fields(LaunchArgs)}
        la_kwargs = {k: v for k, v in request.params.items() if k in la_fields}
        la_kwargs["engine"] = request.engine  # 确保 engine 来自顶层字段
        launch_args = LaunchArgs(**la_kwargs)

        # ---- 2. 推导 PortPlan ----
        port_plan = derive_port_plan(
            port=launch_args.port,
            enable_reason_proxy=settings.ENABLE_REASON_PROXY,
            health_port=settings.HEALTH_PORT,
        )

        # ---- 3. build_launcher_plan（硬件探测 + 配置合并 + 脚本生成） ----
        launcher_plan = build_launcher_plan(launch_args, port_plan)

        # ---- 4. 写入共享卷 ----
        shared_dir = settings.SHARED_VOLUME_PATH
        os.makedirs(shared_dir, exist_ok=True)
        script_path = os.path.join(
            shared_dir, settings.START_COMMAND_FILENAME
        )
        ok = safe_write_file(script_path, launcher_plan.command, is_json=False)
        if not ok:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to write script to {script_path}",
            )

        logger.info("Engine start script written to %s", script_path)
        return {
            "status": "started",
            "message": "Engine start script written to shared volume",
        }

    except HTTPException:
        raise  # 已经是 HTTP 错误，直接抛出
    except Exception as e:
        logger.error("Engine script generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Engine script generation failed: {e}",
        ) from e


@app.get("/api/node_info")
async def node_info():
    """返回本节点的基本信息，供 Master 查询。"""
    return {
        "node_id": config.node_id if config else "unknown",
        "ip": config.ip if config else "unknown",
        "port": config.port if config else 0,
    }


# ---------------------------------------------------------------------------
# Master registration & heartbeat
# ---------------------------------------------------------------------------


def register_with_master(max_retries: int = 30, retry_interval: float = 5.0):
    """向 Master 注册本节点，带重试机制。

    分布式场景下（尤其 podManagementPolicy=Parallel），Worker 和 Master
    同时启动，Worker 首次注册往往早于 Master API 就绪。
    通过轮询重试确保最终注册成功。
    """
    with _config_lock:
        cfg = config
    register_url = f"{cfg.master_url}/api/nodes/register"
    data = {
        "node_id": cfg.node_id,
        "ip": cfg.ip,
        "port": cfg.port,
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(register_url, json=data, timeout=settings.REGISTRATION_TIMEOUT)
            response.raise_for_status()
            logger.info(
                "Successfully registered with master node: %s (attempt %d)",
                cfg.master_url,
                attempt,
            )
            return True
        except Exception as e:
            logger.warning(
                "Registration attempt %d/%d failed: %s",
                attempt, max_retries, e,
            )
            if attempt < max_retries:
                time.sleep(retry_interval)
    logger.error(
        "Registration failed after %d attempts, giving up", max_retries
    )
    return False


# 心跳停止事件，用于优雅关闭心跳线程
_heartbeat_stop = threading.Event()


def send_heartbeat():
    """后台线程：定期向 Master 发送心跳，失败时指数退避。"""
    consecutive_failures = 0
    max_backoff = 300  # 最大退避间隔（秒）
    while not _heartbeat_stop.is_set():
        with _config_lock:
            cfg = config
        try:
            heartbeat_url = f"{cfg.master_url}/api/heartbeat"
            data = {"node_id": cfg.node_id, "workload": 0.0}
            response = requests.post(heartbeat_url, json=data, timeout=settings.HEARTBEAT_TIMEOUT)
            response.raise_for_status()
        except Exception as e:
            consecutive_failures += 1
            logger.error("Heartbeat failed (attempt %d): %s", consecutive_failures, e)
        else:
            consecutive_failures = 0  # 成功则重置
        # 指数退避：正常时用 heartbeat_interval，失败时按 2^n 增长到 max_backoff
        if consecutive_failures > 0:
            backoff = min(cfg.heartbeat_interval * (2 ** consecutive_failures), max_backoff)
            _heartbeat_stop.wait(backoff)
        else:
            _heartbeat_stop.wait(cfg.heartbeat_interval)


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------


def is_port_available(port: int) -> bool:
    """检查端口是否可用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except socket.error:
            return False


def start_worker(worker_config: WorkerConfig | None = None):
    """启动 Worker 节点服务。

    1. 检查端口可用
    2. 向 Master 注册
    3. 启动心跳守护线程
    4. 启动 FastAPI 服务
    """
    global config
    if worker_config:
        with _config_lock:
            config = worker_config
    elif config is None:
        with _config_lock:
            config = WorkerConfig()

    if not is_port_available(config.port):
        logger.error(
            "Port %d is already in use, service startup failed", config.port
        )
        return

    if not register_with_master():
        logger.error("Worker startup aborted because registration with master failed")
        sys.exit(1)

    heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
    heartbeat_thread.start()

    # 确保退出时停止心跳线程
    def _shutdown_heartbeat(*_args):
        _heartbeat_stop.set()

    atexit.register(_shutdown_heartbeat)
    # 不在信号处理函数中调用 sys.exit()，避免与 uvicorn 事件循环竞争。
    # 仅停止心跳，由 uvicorn 自己的信号处理来完成优雅关闭。
    # signal.signal() 只能在主线程中调用；当 start_worker 被上层通过
    # daemon thread 调用时（如 wings_control._run_worker_api），需跳过
    # 信号注册，此时依赖 atexit 回调完成心跳线程清理。
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_a: _shutdown_heartbeat())
    else:
        logger.info(
            "start_worker running in non-main thread, "
            "skipping signal handler registration (atexit still active)"
        )

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-ip", help="Master node IP address", default=None
    )
    args = parser.parse_args()
    with _config_lock:
        config = WorkerConfig(args.master_ip)
    start_worker()
