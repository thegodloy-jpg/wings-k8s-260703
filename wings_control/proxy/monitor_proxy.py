# -*- coding: utf-8 -*-
"""监控API透传服务。

该服务专门用于透传Engine侧的监控接口，支持白名单机制。
独立运行在19100端口（可通过环境变量MONITOR_PROXY_PORT配置）。
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from proxy import proxy_config as C
from proxy.http_client import create_async_client
from proxy.tags import build_backend_url, make_upstream_headers
from proxy.gateway import _copy_entity_headers
from proxy.speaker_logging import configure_worker_logging

configure_worker_logging()
logger = logging.getLogger(__name__)

# 监听端口
MONITOR_PROXY_PORT = int(os.getenv("MONITOR_PROXY_PORT", "19100"))

# 后端URL（从环境变量读取，默认使用 proxy_config 中的值）
BACKEND_URL = os.getenv("BACKEND_URL", C.BACKEND_URL)
logger.info(f"Monitor proxy backend URL: {BACKEND_URL}")

# 白名单
ALLOWED_ENGINE_ENDPOINTS = [
    "/metrics",
    "/version",
    "/v1/models",
    "/load",
]

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """初始化 httpx.AsyncClient（必须 await 工厂函数）。"""
    app.state.client = await create_async_client()
    logger.info(f"Monitor proxy started, listening on port: {MONITOR_PROXY_PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    """关闭 httpx.AsyncClient（容错：client 可能未初始化成功）。"""
    client = getattr(app.state, "client", None)
    if client is not None:
        await client.aclose()
    logger.info("Monitor proxy stopped")


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
    logger.warning(f"Rejected unauthorized engine endpoint proxy: {full_path}")
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
    logger.error(f"Engine request timeout: {target_url}, error: {error}")
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
    logger.error(f"Cannot connect to engine: {C.BACKEND_URL}, error: {error}")
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
    logger.error(f"Engine request failed: {error}")
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
) -> Response:
    """统一的一次性 GET 透传实现。"""
    resp = await client.get(url, headers=headers, timeout=timeout)
    entity_headers = _copy_entity_headers(resp)
    # 直接把下游返回的 body/状态码/实体头转发给上游
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=entity_headers,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.api_route("/{path:path}", methods=["GET"])
async def proxy_engine_request(path: str, request: Request):
    """透传Engine侧的监控接口。

    支持白名单机制，只允许透传预定义的安全接口。
    保持原始响应格式，不做任何修改。

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
            connect=C.METRICS_CONNECT_TIMEOUT if hasattr(C, "METRICS_CONNECT_TIMEOUT") else 5.0,
            read=C.HTTPX_READ_TIMEOUT if hasattr(C, "HTTPX_READ_TIMEOUT") else 15.0,
            write=C.HTTPX_WRITE_TIMEOUT if hasattr(C, "HTTPX_WRITE_TIMEOUT") else 10.0,
            pool=C.HTTPX_POOL_TIMEOUT if hasattr(C, "HTTPX_POOL_TIMEOUT") else 5.0,
        )

        return await _proxy_get_once(app.state.client, target_url, upstream_headers, timeout)

    except httpx.TimeoutException as e:
        return _create_timeout_error_response(target_url, e)
    except httpx.ConnectError as e:
        return _create_connection_error_response(e)
    except httpx.RequestError as e:
        # 兜底网络异常（DNS/握手/断连等），更宽泛
        logger.error(f"Engine request network error: {e.__class__.__name__}: {e}")
        return _create_connection_error_response(e)
    except Exception as e:
        return _create_generic_error_response(e)


def run_standalone():
    """独立进程启动。"""
    uvicorn.run(app, host="0.0.0.0", port=MONITOR_PROXY_PORT)


if __name__ == "__main__":
    run_standalone()
