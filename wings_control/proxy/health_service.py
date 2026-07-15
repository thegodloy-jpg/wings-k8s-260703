# -*- coding: utf-8 -*-
"""独立健康服务。

与 gateway 中的 `/health` 不同，这个模块单独跑在健康端口上，
便于 Kubernetes 探针在 proxy 高负载时仍然可靠读取健康状态。

同时整合了 monitor_proxy 的功能，透传 Engine 侧的监控接口。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict
from config.settings import settings

import httpx
import uvicorn
from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse


from proxy.health_router import (
    _jittered_sleep_base,
    build_health_body,
    build_v1_health_body,
    build_health_headers,
    init_health_state,
    map_http_code_from_state,
    teardown_health_monitor,
    tick_observe_and_advance,
)
from utils.log_config import setup_root_logging, LOGGER_HEALTH
from proxy.speaker_logging import configure_worker_logging
from utils.progress_utils import StartupProgressManager
from proxy.tags import build_backend_url, make_upstream_headers
from proxy.gateway import _copy_entity_headers

setup_root_logging(stderr_level="ERROR")
_logger = logging.getLogger(LOGGER_HEALTH)

# 配置 worker 日志：归一化 uvicorn/httpx 子 logger 格式，
# 安装 /health 日志过滤器以抑制 httpx 高频探活噪声。
configure_worker_logging()

# health 服务的 httpx 活动仅有后端探活轮询，全部是低价值重复日志。
# 将 httpx 日志级别提升至 WARNING，彻底消除噪声。
# 注意：设置父 logger "httpx" 的级别会通过 effective level 影响所有子 logger
# （如 httpx._client），因此无需逐个设置。
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 单独的 FastAPI 应用，通常监听 `HEALTH_SERVICE_PORT`。
app = FastAPI()

# 独立健康服务对外监听端口，通常由 launcher 注入。
HEALTH_SERVICE_PORT = int(os.getenv("HEALTH_SERVICE_PORT", "19000"))

# 监控接口白名单（从 monitor_proxy 迁移）
ALLOWED_ENGINE_ENDPOINTS = [
    "/metrics",
    "/version",
    "/v1/models",
    "/load",
]


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化健康服务所需的资源。

    初始化内容包括：
      1. 创建异步 HTTP 客户端，用于轮询后端 /health 接口。
      2. 初始化健康状态字典（分数、连续状态计数等）。
      3. 初始化启动进度管理器。
      4. 启动后台健康轮询任务。
    """
    app.state.client = httpx.AsyncClient()
    app.state.health = init_health_state()
    app.state.progress_manager = StartupProgressManager(os.getenv("ENGINE", "vllm"))
    app.state.health_task = asyncio.create_task(health_monitor_loop(), name="health-monitor")


async def health_monitor_loop():
    """后台健康轮询循环，周期性探测后端引擎状态。

    不断调用 tick_observe_and_advance() 更新健康状态机，
    并根据当前状态动态调整轮询间隔（包含随机抱动以避免雷群效应）。
    发生异常时仅记录警告日志而不中断循环，确保健康探测始终运行。
    """
    while True:
        try:
            await tick_observe_and_advance(app.state.health, app.state.client)
        except Exception as e:
            _logger.warning("health_monitor_error: %s", e)
        await asyncio.sleep(_jittered_sleep_base(app.state.health))


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源。

    依次取消后台健康轮询任务，然后关闭异步 HTTP 客户端，
    确保连接池和文件句柄被正确释放。
    """
    await teardown_health_monitor(app)
    await app.state.client.aclose()


@app.get("/v1/health")
async def v1_health_check():
    """返回当前健康状态。

    根据健康状态机的当前状态映射为 HTTP 状态码（200/503），
    并在响应头中注入状态摘要信息。

    Returns:
        Response | JSONResponse: 健康检查响应，HTTP 200 表示健康，503 表示异常。
    """
    h = app.state.health
    code = map_http_code_from_state(h)
    headers = build_health_headers(h)

    body = build_v1_health_body(h, code)
    return JSONResponse(status_code=code, content=body, headers=headers)


@app.get("/health")
async def health_check(minimal: bool = False):
    """返回当前健康状态。

    根据健康状态机的当前状态映射为 HTTP 状态码（200/503），
    并在响应头中注入状态摘要信息。

    Args:
        minimal: 为 True 时返回空 body 的精简响应（仅状态码 + 头部），
            适用于 K8s livenessProbe。为 False 时返回包含详细分数、
            连续状态计数等信息的 JSON body。

    Returns:
        Response | JSONResponse: 健康检查响应，HTTP 200 表示健康，503 表示异常。
    """
    h = app.state.health
    code = map_http_code_from_state(h)
    headers = build_health_headers(h)

    if minimal:
        return Response(status_code=code, headers=headers)

    body = build_health_body(h, code)
    return JSONResponse(status_code=code, content=body, headers=headers)


@app.head("/health")
async def health_head():
    """轻量级 HEAD 健康接口，供 Kubernetes 探针使用。

    仅返回 HTTP 状态码和状态头部，不包含响应 body，
    最大限度减少健康探测的网络开销。

    Returns:
        Response: 空 body 响应，状态码 200（健康）或 503（异常）。
    """
    h = app.state.health
    code = map_http_code_from_state(h)
    headers = build_health_headers(h)
    return Response(status_code=code, headers=headers)


async def _detect_engine_version(client: httpx.AsyncClient) -> str | None:
    """尝试从运行中的后端获取真实引擎版本号。

    向后端 /version 端点发起 GET 请求（vLLM/vLLM-Ascend 标准接口）。
    成功后将结果缓存在 app.state 中，后续调用直接返回缓存值。

    Returns:
        str | None: 引擎版本字符串（如 "0.17.0rc1"），失败时返回 None。
    """
    cached = getattr(app.state, "cached_engine_version", None)
    if cached:
        return cached
    try:
        url = f"http://{settings.ENGINE_HOST}:{settings.ENGINE_PORT}/version"
        resp = await client.get(url, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            version = data.get("version", "")
            if version:
                app.state.cached_engine_version = version
                return version
    except Exception as exc:  # noqa: BLE001
        _logger.debug("Failed to detect engine version: %s", exc)
    return None

@app.get("/v1/startup/progress")
async def get_startup_progress():
    """获取部署进度信息

    Returns:
        JSONResponse: 包含进度信息的响应
    """
    from utils.progress_utils import read_progress_file, build_progress_response
    
    progress_file = settings.PROGRESS_FILE
    try:
        if os.path.exists(progress_file):
            file_progress = read_progress_file(progress_file)
            progress_data = app.state.progress_manager.update_from_file(file_progress)
        else:
            # 文件不存在，使用初始化信息
            progress_data = app.state.progress_manager.get_initial_progress_data()
        
        return JSONResponse(status_code=200, content=build_progress_response(progress_data))
    except Exception as e:
        _logger.error(f"Failed to get progress info: {e}")
        from utils.progress_utils import create_error_progress_data
        error_data = create_error_progress_data(e)
        response_body = build_progress_response(
            error_data, f"Failed to get progress info: {str(e)}"
        )
        return JSONResponse(status_code=200, content=response_body)


def _is_endpoint_allowed(path: str) -> bool:
    """检查请求路径是否在白名单中。

    Args:
        path: 请求路径

    Returns:
        bool: 是否允许访问
    """
    return path in ALLOWED_ENGINE_ENDPOINTS


def _create_forbidden_response(full_path: str) -> JSONResponse:
    """创建403禁止访问响应。

    Args:
        full_path: 请求路径

    Returns:
        JSONResponse: 403错误响应
    """
    _logger.warning(f"Rejected unauthorized engine endpoint proxy: {full_path}")
    return JSONResponse(
        status_code=403,
        content={
            "code": 403,
            "msg": f"Endpoint '{full_path}' is not allowed for proxying",
            "data": None,
        },
    )


def _create_timeout_error_response(target_url: str, error: Exception) -> JSONResponse:
    """创建504超时错误响应。

    Args:
        target_url: 目标URL
        error: 异常对象

    Returns:
        JSONResponse: 504错误响应
    """
    _logger.error(f"Engine request timeout: {target_url}, error: {error}")
    return JSONResponse(
        status_code=504,
        content={
            "code": 504,
            "msg": "Engine request timeout",
            "data": None
        }
    )


def _create_connection_error_response(error: Exception) -> JSONResponse:
    """创建503连接错误响应。

    Args:
        error: 异常对象

    Returns:
        JSONResponse: 503错误响应
    """
    _logger.error(f"Cannot connect to engine: {settings.ENGINE_HOST}:{settings.ENGINE_PORT}, error: {error}")
    return JSONResponse(
        status_code=503,
        content={
            "code": 503,
            "msg": "Engine connection failed",
            "data": None
        }
    )


def _create_generic_error_response(error: Exception) -> JSONResponse:
    """创建500通用错误响应。

    Args:
        error: 异常对象

    Returns:
        JSONResponse: 500错误响应
    """
    _logger.error(f"Engine request failed: {error}")
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "msg": f"Engine request failed: {str(error)}",
            "data": None
        }
    )


async def _proxy_get_once(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    timeout: httpx.Timeout | float,
    path: str,
) -> Response:
    """统一的一次性 GET 透传实现。

    Args:
        client: httpx 异步客户端
        url: 目标 URL
        headers: 请求头
        timeout: 超时配置
        path: 请求路径（用于判断是否为 /metrics 接口）
    """
    resp = await client.get(url, headers=headers, timeout=timeout)
    entity_headers = _copy_entity_headers(resp)

    # 对于 /metrics 接口，添加 POD_NAME 到响应头
    if path == "/metrics":
        pod_name = os.getenv("POD_NAME", "")
        entity_headers["X-Pod-Name"] = pod_name

    # 直接把下游返回的 body/状态码/实体头转发给上游
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=entity_headers,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# FastAPI matches routes in registration order. Keep health-owned APIs above the
# catch-all proxy route so they are not rejected by the engine endpoint whitelist.
@app.get("/v1/startup/accel")
async def get_startup_accel():
    """获取加速特性使能信息。

    单一真相源是 advanced_features.json（settings.ADVANCED_FEATURES_FILE）：
    由 wings_entry 在脚本生成阶段写入；shell 层在补丁失败时回写。
    data 字段保持 advanced_features.json 的完整对象；接口只补齐缺省标准字段，
    不裁剪未来新增的顶层状态字段。

    Returns:
        JSONResponse: 包含加速特性信息的响应
    """
    try:
        state = _read_advanced_features_state(settings.ADVANCED_FEATURES_FILE)
        accel_data = _build_accel_data(state)
        return _build_accel_response(accel_data)
    except Exception as e:
        _logger.error(f"Failed to get acceleration feature info: {e}")
        accel_data = _build_accel_data({})
        return _build_accel_response(accel_data, f"Failed to get acceleration feature info: {str(e)}")


@app.api_route("/{path:path}", methods=["GET"])
async def proxy_engine_monitor_request(path: str, request: Request):
    """透传Engine侧的监控接口。

    支持白名单机制，只允许透传预定义的安全接口。
    保持原始响应格式，不做任何修改。
    直接原样透传给引擎侧17000端口。
    对于 /metrics 接口，会在响应头中添加 POD_NAME。

    Args:
        path: 请求路径
        request: 原始请求对象

    Returns:
        Response: Engine的原始响应
    """
    # 构建完整路径
    full_path = f"/{path}"

    # 白名单检查
    if not _is_endpoint_allowed(full_path):
        return _create_forbidden_response(full_path)

    # 目标 URL
    target_url = build_backend_url(full_path)

    try:
        # 上游请求头（同步函数，返回 dict）
        upstream_headers = make_upstream_headers(request)

        # 统一超时：/metrics 可能稍大，给足 read；普通接口也可共用
        timeout = httpx.Timeout(
            connect=5.0,
            read=15.0,
            write=10.0,
            pool=5.0,
        )

        return await _proxy_get_once(app.state.client, target_url, upstream_headers, timeout, full_path)

    except httpx.TimeoutException as e:
        return _create_timeout_error_response(target_url, e)
    except httpx.ConnectError as e:
        return _create_connection_error_response(e)
    except httpx.RequestError as e:
        # 兜底网络异常（DNS/握手/断连等），更宽泛
        _logger.error(f"Engine request network error: {e.__class__.__name__}: {e}")
        return _create_connection_error_response(e)
    except Exception as e:
        return _create_generic_error_response(e)


def _read_advanced_features_state(file_path: str) -> dict:
    """读取 advanced_features.json（页面状态汇报文件，使能真相源）。

    该文件是单个 JSON 对象，当前标准字段为 ``engine/features/variants/others``。
    接口层返回完整对象，避免后续新增顶层字段后被健康接口静默丢弃。

    Args:
        file_path: advanced_features.json 路径

    Returns:
        dict: advanced_features.json 的完整对象；文件缺失、损坏或非对象时返回空字典。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _logger.debug("advanced_features.json unavailable: %s", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _build_accel_data(state: dict | None = None) -> dict:
    """构建加速特性响应数据。

    Args:
        state: advanced_features.json 的完整对象

    Returns:
        dict: 加速特性数据
    """
    data = dict(state) if isinstance(state, dict) else {}
    # 保持老消费者依赖的四个标准字段稳定，同时不删除 advanced_features.json
    # 中未来新增的顶层字段，例如请求态、回退原因或诊断信息。
    data["engine"] = data.get("engine") or os.getenv("ENGINE", "vllm")
    if not isinstance(data.get("features"), dict):
        data["features"] = {}
    if not isinstance(data.get("variants"), dict):
        data["variants"] = {}
    if not isinstance(data.get("others"), dict):
        data["others"] = {}
    return data


def _build_accel_response(accel_data: dict, message: str = "") -> JSONResponse:
    """构建加速特性响应。

    Args:
        accel_data: 加速特性数据
        message: 响应消息

    Returns:
        JSONResponse: 加速特性响应
    """
    return JSONResponse(status_code=200, content={
        "code": 200,
        "msg": message,
        "data": accel_data
    })


def run_standalone():
    """以独立进程方式启动健康服务，供本地开发调试使用。

    监听地址固定为 0.0.0.0，端口由环境变量 HEALTH_SERVICE_PORT 决定（默认 19000）。
    生产环境通常由 launcher 通过 uvicorn 启动，不使用此入口。
    """
    uvicorn.run(app, host="0.0.0.0", port=HEALTH_SERVICE_PORT)


if __name__ == "__main__":
    run_standalone()
