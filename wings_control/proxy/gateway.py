# -*- coding: utf-8 -*-
"""主业务代理入口。

职责可以概括为三类：
1. 对外暴露 OpenAI 兼容接口，并把请求转发到 backend engine；
2. 对流式和非流式响应采用不同的回传策略，兼顾首包延迟和吞吐；
3. 维护观测头、重试信息以及 `/health`、`/metrics` 等辅助接口。
"""

from __future__ import annotations
import asyncio
import inspect
import json
import random
import time
import os
from typing import Any, AsyncIterator, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# RAG 加速
from rag_acc.rag_app import is_rag_scenario, rag_acc_chat
from rag_acc.extract_dify_info import is_dify_scenario

from . import proxy_config as C
from .http_client import create_async_client
from .queueing import QueueGate
from .tags import (
    want_stream,
    rebuild_request_json,
    make_upstream_headers,
    read_json_body,
    jlog, elog, ms,
    build_backend_url,
)
from .speaker_logging import configure_worker_logging

# Anthropic ↔ OpenAI 协议转换（仅 SmartQoS 模式下启用）
from .anthropic_converter import (
    convert_anthropic_request_to_openai,
    convert_openai_response_to_anthropic,
    convert_openai_stream_to_anthropic,
)

# 健康状态机在 `health_router.py` 中维护，gateway 只负责对外暴露结果。
from .health_router import (
    setup_health_monitor,
    teardown_health_monitor,
    map_http_code_from_state,
    build_health_body,
    build_health_headers,
)


class _DictAttrView:
    """Dict wrapper allowing attribute-style access.

    用于替代 fastchat 的 ChatCompletionRequest，避免其对
    content=null 等字段做严格校验导致报错。
    """

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getattr__(self, name: str):
        return self._data.get(name)

    def __repr__(self) -> str:
        return repr(self._data)

configure_worker_logging()

# =============================================================================
# 全局配置常量
# =============================================================================
# 从环境变量读取 POD_NAME，用于在响应头中标识当前 Pod
POD_NAME = os.getenv("POD_NAME", "")

# 由 launcher 通过 uvicorn 启动的 FastAPI 应用。
app = FastAPI()

# backend 地址由 launcher 注入环境变量，这里只做读取。
app.state.backend = C.BACKEND_URL

# 做 httpx 版本兼容：有的版本 `send()` 支持 timeout，有的版本不支持。
_SEND_HAS_TIMEOUT = "timeout" in inspect.signature(httpx.AsyncClient.send).parameters


# =============================================================================
# 内部函数：HTTP 请求发送与重试逻辑
# =============================================================================


async def _raw_send(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    content: bytes | None = None,
    headers: dict | None = None,
    stream: bool = False,
    timeout: httpx.Timeout | None = None,
) -> httpx.Response:
    """统一封装底层发送逻辑，屏蔽不同 httpx 版本的签名差异。

    Args:
        client: httpx 异步客户端实例
        method: HTTP 方法 ("GET"/"POST"/...)
        url:    请求目标 URL
        content: 请求体字节数据
        headers: 请求头字典
        stream:  是否使用流式接收响应
        timeout: 可选超时配置

    Returns:
        httpx.Response: 后端响应对象

    实现说明:
        httpx 不同版本的 send() 方法签名不同，有些支持 timeout 参数，
        有些不支持。通过 _SEND_HAS_TIMEOUT 标志动态决定是否传递 timeout。
    """
    req = client.build_request(method, url, content=content, headers=headers)
    if _SEND_HAS_TIMEOUT and timeout is not None:
        return await client.send(req, stream=stream, timeout=timeout)
    return await client.send(req, stream=stream)

# =============================================================================
# 重试策略配置
# =============================================================================

# 固定可重试异常和状态码集合，避免重试范围无限扩大。
_RETRIABLE_EXC = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_RETRIABLE_5XX = (502, 503, 504)


class _ErrorDedup:
    """去重连续相同错误日志，避免刷屏。

    同一类错误（err_type + detail 相同）在 window 秒内只记录首次。
    窗口过后首次出现时再记录（并附带被抑制的次数）。
    """

    def __init__(self, window: float = 30.0):
        self._window = window
        # key -> (first_seen_time, suppressed_count)
        self._seen: dict[str, tuple[float, int]] = {}

    def should_log(self, err_type: str, detail: str) -> tuple[bool, int]:
        """判断是否应该记录此错误。

        Returns:
            (should_log, suppressed_count)
        """
        key = f"{err_type}:{detail}"
        now = time.time()

        if key in self._seen:
            first_time, count = self._seen[key]
            if now - first_time < self._window:
                # 窗口内，抑制
                self._seen[key] = (first_time, count + 1)
                return False, count + 1
            # 窗口过期，重新开始
            self._seen[key] = (now, 0)
            return True, count
        # 首次出现
        self._seen[key] = (now, 0)
        return True, 0


_error_dedup = _ErrorDedup()

# =============================================================================
# 重试辅助函数
# =============================================================================


def _should_retry_status(stream: bool, status_code: int, attempt: int, total: int) -> bool:
    """判断是否应该基于 HTTP 状态码进行重试。

    Args:
        stream:      是否为流式请求
        status_code: 后端返回的 HTTP 状态码
        attempt:     当前尝试次数
        total:       最大尝试次数

    Returns:
        bool: True 表示应该重试，False 表示不重试

    设计决策:
        - 仅对流式请求的部分 5xx (502/503/504) 做重试
        - 普通请求保持行为更可预期，不重试
    """
    # 仅对流式请求的部分 5xx 做重试，普通请求保持行为更可预期。
    return stream and status_code in _RETRIABLE_5XX and attempt < total


async def _close_resp_quiet(resp: httpx.Response) -> None:
    """安静关闭响应对象，忽略关闭过程中的异常。

    用于重试逻辑中当需要关闭旧响应时调用，
    防止关闭异常影响重试流程。

    Args:
        resp: 要关闭的 httpx.Response 对象
    """
    try:
        await resp.aclose()
    except Exception as e:
        C.logger.error("Failed to close response: %s", e)


def _mark_retry_count(resp: httpx.Response, attempt: int) -> None:
    """将实际重试次数记录到响应的 extensions 字典中。

    后续可通过 X-Retry-Count 头透传给客户端，便于调试和观测。

    Args:
        resp:    后端响应对象
        attempt: 当前尝试序号 (1-based)，重试次数 = attempt - 1
    """
    # 将“真实重试了几次”挂在 response 上，后续可透传给客户端观察。
    try:
        resp.extensions["app_retry_count"] = attempt - 1
    except Exception as e:
        C.logger.error("Failed to set retry count in response extensions: %s", e)


async def _log_and_wait_status_retry(rid: str | None, attempt: int, status: int, interval: float, t0: float) -> None:
    """记录状态码重试日志并等待指定间隔后再重试。

    在基于 HTTP 状态码触发重试时调用，先记录一条结构化日志，
    然后异步休眠 interval 秒，为下一次重试预留后端恢复窗口。

    Args:
        rid:      请求 ID，用于日志关联
        attempt:  当前尝试序号 (1-based)
        status:   后端返回的 HTTP 状态码
        interval: 两次重试之间的等待时间（秒）
        t0:       本次尝试的起始时间戳（perf_counter）
    """
    elog(
        "retry_status",
        rid=rid, attempt=attempt, status=status,
        next_wait_ms=int(interval * 1000), elapsed=ms(time.perf_counter() - t0),
    )
    await asyncio.sleep(interval)


def _is_retriable_exception(e: Exception) -> bool:
    """判断异常是否属于可重试类型。

    仅将连接错误 (ConnectError)、连接超时 (ConnectTimeout)
    和连接池耗尽 (PoolTimeout) 视为可重试异常，其余异常直接上抛。

    Args:
        e: 捕获到的异常实例

    Returns:
        bool: True 表示该异常可以安全重试
    """
    return isinstance(e, _RETRIABLE_EXC)


async def _log_and_maybe_wait_exception(e: Exception, **ctx) -> bool:
    """记录异常日志，若可重试则等待后返回 True。

    对捕获到的异常执行以下流程：
    1. 记录结构化日志（含异常类型、详情、是否可重试）；
    2. 若该异常属于可重试类型且尚未用尽重试次数，则休眠 interval 秒并返回 True；
    3. 否则返回 False，由调用方决定是否上抛。

    Args:
        e: 捕获到的异常实例
        **ctx: 上下文关键字参数，包含：
            - rid (str | None):    请求 ID
            - attempt (int):       当前尝试序号
            - total (int):         最大尝试次数
            - interval (float):    重试等待间隔（秒）
            - t0 (float):         本次尝试起始时间戳

    Returns:
        bool: True 表示已等待完毕、可以重试；False 表示不可重试
    """
    rid = ctx.get("rid")
    attempt = ctx.get("attempt")
    total = ctx.get("total")
    interval = ctx.get("interval")
    t0 = ctx.get("t0")

    retriable = _is_retriable_exception(e)
    err_type_str = e.__class__.__name__
    detail_str = str(e)
    should_log_it, suppressed = _error_dedup.should_log(err_type_str, detail_str)
    if should_log_it:
        extra = {}
        if suppressed > 0:
            extra["prev_suppressed"] = suppressed
        elog(
            "retry_exception",
            rid=rid,
            attempt=attempt,
            err_type=err_type_str,
            detail=detail_str,
            retriable=retriable,
            next_wait_ms=(int(interval * 1000) if retriable and attempt < total else 0),
            elapsed=ms(time.perf_counter() - t0),
            **extra,
        )
    if retriable and attempt < total:
        await asyncio.sleep(interval)
        return True
    return False


async def _send_with_fixed_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    content: bytes | None = None,
    headers: dict | None = None,
    stream: bool = False,
    timeout: httpx.Timeout | None = None,
    rid: str | None = None,
) -> httpx.Response:
    """执行带固定重试策略的请求发送。

    重试策略:
        - 最多重试 RETRY_TRIES 次（包含首次调用）
        - 每次重试间隔 RETRY_INTERVAL_MS 毫秒
        - 仅重试连接错误、超时、连接池用尽
        - 流式请求额外重试 502/503/504

    Args:
        client:  httpx 异步客户端
        method:  HTTP 方法
        url:     目标 URL
        content: 请求体字节
        headers: 请求头
        stream:  是否流式接收
        timeout: 超时配置
        rid:     请求 ID（用于日志）

    Returns:
        httpx.Response: 成功的后端响应

    Raises:
        HTTPException(502): 后端连接错误或持续 5xx
    """
    total = max(1, int(C.RETRY_TRIES))
    interval = max(0, int(C.RETRY_INTERVAL_MS)) / 1000.0
    last_exc: Exception | None = None
    last_status: int | None = None

    for attempt in range(1, total + 1):
        t0 = time.perf_counter()
        try:
            resp = await _raw_send(
                client, method, url,
                content=content, headers=headers, stream=stream, timeout=timeout
            )

            # Retry on server-side error: check 5xx status code
            if _should_retry_status(stream, resp.status_code, attempt, total):
                last_status = resp.status_code
                await _close_resp_quiet(resp)
                await _log_and_wait_status_retry(rid, attempt, last_status, interval, t0)
                continue

            # Mark final retry count on response extension header
            _mark_retry_count(resp, attempt)
            return resp

        except Exception as e:
            last_exc = e
            #
            if await _log_and_maybe_wait_exception(e, rid=rid, attempt=attempt, total=total, interval=interval, t0=t0):
                continue

            # Wrap connection error as HTTP 502 Bad Gateway
            if isinstance(e, httpx.RequestError):
                raise HTTPException(502, "backend connect error") from e
            raise

    #
    if last_exc is not None:
        if isinstance(last_exc, httpx.RequestError):
            elog("retry_final_fail", rid=rid, tries=total, final_error=str(last_exc))
            raise HTTPException(502, "backend connect error") from last_exc
        elog("retry_final_fail", rid=rid, tries=total, final_error=str(last_exc))
        raise last_exc
    if last_status is not None:
        elog("retry_final_fail", rid=rid, tries=total, final_status=last_status)
        raise HTTPException(502, f"backend error after retries (status {last_status})")
    elog("retry_final_fail", rid=rid, tries=total, reason="unknown")
    raise HTTPException(502, "backend error after retries")


#   http


@app.on_event("startup")
async def _startup():
    """初始化 HTTP 客户端、gate 和健康监控任务。"""
    configure_worker_logging(force=True)
    C.log_boot_plan()
    app.state.client = await create_async_client()
    app.state.gate = QueueGate()

    # 启动后台健康轮询，让 `/health` 能读到持续更新的状态。
    setup_health_monitor(app)
    C.logger.info("Reason-Proxy is starting on %s:%s (health monitor loop enabled)", C.HOST, C.PORT)
    # 启动确认日志
    C.logger.info("Proxy ready: http://0.0.0.0:%s -> backend %s", C.PORT, C.BACKEND_URL)


@app.on_event("shutdown")
async def _shutdown():
    """应用关闭时清理资源。

    依次执行：
    1. 停止后台健康监控任务（teardown_health_monitor）；
    2. 关闭 httpx 异步客户端连接池。

    Raises:
        不会向外抛出异常；CancelledError 被静默吞掉以兼容 Starlette 关闭流程。
    """
    try:
        await teardown_health_monitor(app)
    except asyncio.CancelledError:
        # Swallow CancelledError to match Starlette shutdown flow
        pass
    await app.state.client.aclose()


#


def _copy_entity_headers(resp: httpx.Response) -> Dict[str, str]:
    """从后端响应中提取需要透传给客户端的实体头。

    保留以下类别的响应头：
    - 所有 X-* 自定义头（含后端观测头）
    - content-type、content-encoding（保持内容编码语义）
    - etag、last-modified（缓存验证头）

    Args:
        resp: 后端返回的 httpx.Response 对象

    Returns:
        Dict[str, str]: 筛选后的响应头字典
    """
    return {
        k: v for k, v in resp.headers.items()
        if k.lower().startswith("x-") or k.lower() in ("content-type", "content-encoding", "etag", "last-modified")
    }


def _merge_obs_and_retry_headers(
        gate: QueueGate,
        queue_headers: Dict[str, str],
        resp: httpx.Response) -> Dict[str, str]:
    """合并观测头与重试计数头，生成最终的附加响应头。

    将排队观测头（X-InFlight、X-Queued-Wait 等）与请求重试次数
    (X-Retry-Count) 合并为一个字典，便于一次性附加到客户端响应。

    Args:
        gate:          QueueGate 实例，提供 obs_headers() 方法
        queue_headers: 排队阶段产生的队列相关头信息
        resp:          后端响应对象，其 extensions 字段可能含 app_retry_count

    Returns:
        Dict[str, str]: 合并后的附加响应头字典
    """
    merged = gate.obs_headers(queue_headers)
    ext = getattr(resp, "extensions", None)
    if isinstance(ext, dict):
        retry_cnt = ext.get("app_retry_count")
        if retry_cnt is not None:
            merged["X-Retry-Count"] = str(retry_cnt)
    # 添加 Pod 名称到响应头
    merged["X-Pod-Name"] = POD_NAME
    return merged


def _content_length(resp: httpx.Response) -> int | None:
    """从响应头中安全提取 Content-Length 值。

    若响应头中包含合法的 Content-Length 数值字符串，则返回对应整数；
    否则返回 None。用于非流式转发中判断是否可以一次性缓冲返回。

    Args:
        resp: 后端返回的 httpx.Response 对象

    Returns:
        int | None: 内容长度（字节），无法解析时返回 None
    """
    try:
        v = resp.headers.get("content-length")
        return int(v) if v is not None and v.isdigit() else None
    except Exception as _:
        return None


async def _send_nonstream_request(
    client: httpx.AsyncClient,
    upstream_path: str,
    body_bytes: bytes,
    req: Request,
    rid: str | None,
) -> httpx.Response:
    """向后端发送非流式请求（但以 stream=True 接收响应）。

    虽然业务语义上为非流式请求，但在 HTTP 传输层仍使用 stream=True
    来接收响应，以便后续根据 Content-Length 决定一次性读取还是按块转发，
    从而避免大响应体撑爆内存。

    Args:
        client:        httpx 异步客户端实例
        upstream_path: 后端路由路径（如 "/v1/chat/completions"）
        body_bytes:    序列化后的请求体字节
        req:           原始客户端请求对象（用于提取头信息）
        rid:           请求 ID，用于日志追踪

    Returns:
        httpx.Response: 后端响应对象（未读取 body）

    Raises:
        HTTPException(502): 后端连接失败或持续返回 5xx
    """
    url = build_backend_url(upstream_path)
    return await _send_with_fixed_retries(
        client, "POST", url,
        content=body_bytes,
        headers=make_upstream_headers(req, want_gzip=False),
        stream=True,  #
        rid=rid,
    )


async def _pipe_nonstream(req: Request, r: httpx.Response, rid: str | None,
                          gate: QueueGate | None = None) -> AsyncIterator[bytes]:
    """按块管道转发非流式响应体数据到客户端。

    当非流式响应的 Content-Length 超过 NONSTREAM_THRESHOLD 阈值时，
    改为按块流式转发，避免将大体积 JSON 响应完整缓冲在代理内存中。
    转发过程中持续检测客户端是否已断开连接，若断开则提前终止。

    当 gate 不为 None 且 GATE_EARLY_RELEASE=false 时，在流式传输
    结束后释放闸门槽位，确保在整个后端请求期间保持并发占用。

    Args:
        req:  原始客户端请求对象，用于检测连接状态
        r:    后端 httpx.Response 对象（stream 模式，未读取 body）
        rid:  请求 ID，用于日志追踪
        gate: 排队控制器（非早释放模式下传入，用于延迟释放）

    Yields:
        bytes: 后端响应体的字节块
    """
    try:
        async for chunk in r.aiter_bytes():
            if not chunk:
                continue
            if await req.is_disconnected():
                elog("client_disconnected_nonstream", rid=rid)
                break
            yield chunk
    finally:
        await r.aclose()
        if gate is not None and not C.GATE_EARLY_RELEASE:
            await gate.release()
            jlog("gate_released_after_response", rid=rid, mode="nonstream_pipe")


async def _acquire_gate_early_nonstream(req: Request, gate: QueueGate, rid: str | None) -> Dict[str, str]:
    """为非流式请求获取排队闸门。

    根据 GATE_EARLY_RELEASE 配置决定是否立即释放：
    - 早释放模式（GATE_EARLY_RELEASE=true）：acquire 后立即 release，
      闸门仅做准入速率控制，不限制后端并发数。
    - 持有模式（GATE_EARLY_RELEASE=false，默认）：acquire 后保持占用，
      由调用方在后端响应完成后手动 release，实现真正的并发限制。

    Args:
        req:  原始客户端请求对象
        gate: QueueGate 排队控制器实例
        rid:  请求 ID，用于日志追踪

    Returns:
        Dict[str, str]: 排队相关的头信息（如 X-Queued-Wait）

    Raises:
        HTTPException: 排队超时或并发数超限时抛出
    """
    queue_headers = await gate.acquire(dict(req.headers))
    if C.GATE_EARLY_RELEASE:
        await gate.release()
        jlog("gate_acquired_released_early", rid=rid, wait_hdr=queue_headers.get("X-Queued-Wait"))
    else:
        jlog("gate_acquired_held", rid=rid, wait_hdr=queue_headers.get("X-Queued-Wait"))
    return queue_headers


async def _forward_nonstream(req: Request, upstream_path: str):
    """完整的非流式请求转发流程。

    以 stream=True 方式从后端接收响应，再根据响应体大小选择返回策略：
    1) 读取并校验 JSON 请求体；
    2) 获取/释放排队闸门，记录排队等待耗时；
    3) 向后端发送请求并获取响应对象（未消费 body）；
    4) 若 Content-Length ≤ NONSTREAM_THRESHOLD，一次性读取后以 Response 返回；
       否则使用 StreamingResponse 按块管道转发，避免大响应撑爆内存。

    Args:
        req:           原始客户端 Request 对象
        upstream_path: 后端路由路径（如 "/v1/chat/completions"）

    Returns:
        Response | StreamingResponse | JSONResponse:
            正常时返回后端响应内容；排队异常时返回包含错误detail 的 JSONResponse。
    """
    client: httpx.AsyncClient = app.state.client
    gate: QueueGate = app.state.gate
    rid = req.headers.get("x-request-id")

    #
    body_bytes = await read_json_body(req, rid)
    jlog("req_recv", rid=rid, path=str(req.url.path), body_len=len(body_bytes))

    #
    try:
        queue_headers = await _acquire_gate_early_nonstream(req, gate, rid)
    except HTTPException as e:
        elog("gate_acquire_error", rid=rid, status=e.status_code, detail=str(e.detail))
        return JSONResponse(status_code=e.status_code,
                            content={"detail": e.detail},
                            headers=gate.obs_headers(e.headers))

    #
    try:
        r = await _send_nonstream_request(client, upstream_path, body_bytes, req, rid)

        # Merge observation headers and retry headers
        merged = _merge_obs_and_retry_headers(gate, queue_headers, r)
        entity_headers = _copy_entity_headers(r)

        #
        content_len = _content_length(r)
        if content_len and content_len <= C.NONSTREAM_THRESHOLD:
            data = await r.aread()
            await r.aclose()
            if not C.GATE_EARLY_RELEASE:
                await gate.release()
                jlog("gate_released_after_response", rid=rid, mode="nonstream")
            return Response(
                data,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
                headers={**entity_headers, **merged},
            )

        #
        return StreamingResponse(
            _pipe_nonstream(req, r, rid, gate=gate),
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
            headers={**entity_headers, **merged},
        )
    except Exception as _:
        # 异常时也要释放闸门，避免泄漏
        if not C.GATE_EARLY_RELEASE:
            await gate.release()
            jlog("gate_released_on_error", rid=rid, mode="nonstream")
        raise


# =============================================================================
# Anthropic → OpenAI 协议转换处理（SmartQoS 模式）
# =============================================================================


async def _handle_anthropic_with_conversion(
    req: Request,
    payload: Dict[str, Any],
    rid: str,
) -> Response | StreamingResponse | JSONResponse:
    """处理 Anthropic /v1/messages 请求，转换为 OpenAI 格式后转发到 /v1/chat/completions。

    在 SmartQoS 模式下启用，实现:
    1. 请求体: Anthropic → OpenAI 格式转换（支持 text/image/thinking/tool_use/tool_result）
    2. 转发: 转换后的请求发送到后端 /v1/chat/completions
    3. 响应体: OpenAI → Anthropic 格式转换（非流式/流式均支持）
    4. Priority 注入: 从 metadata.priority 或显式参数提取，解决 Anthropic 协议
       不支持 priority 调度的问题

    Args:
        req:     原始客户端 Request 对象
        payload: 解析后的 Anthropic 请求体字典
        rid:     请求 ID，用于日志追踪

    Returns:
        Response | StreamingResponse | JSONResponse:
            正常时返回 Anthropic 格式的响应；异常时返回错误 JSONResponse。
    """
    # 1. Anthropic → OpenAI 请求体转换
    try:
        openai_payload = convert_anthropic_request_to_openai(payload)
    except Exception as e:
        elog("anthropic_convert_error", rid=rid, detail=str(e))
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to convert Anthropic request: {e}"},
        )

    jlog("anthropic_converted_to_openai", rid=rid,
         model=openai_payload.get("model", ""),
         has_tools="tools" in openai_payload,
         has_priority="priority" in openai_payload)

    # 2. 序列化为 bytes
    body_bytes = json.dumps(openai_payload, ensure_ascii=False).encode("utf-8")

    # 3. 转发到 /v1/chat/completions
    is_stream = want_stream(openai_payload.get("stream", False))

    if is_stream:
        return await _forward_anthropic_stream(req, body_bytes, rid)
    else:
        return await _forward_anthropic_nonstream(req, body_bytes, rid)


async def _forward_anthropic_stream(
    req: Request,
    body_bytes: bytes,
    rid: str,
) -> StreamingResponse | JSONResponse:
    """转发转换后的 OpenAI 流式请求，并将响应转换回 Anthropic 格式。"""
    client: httpx.AsyncClient = app.state.client
    gate: QueueGate = app.state.gate

    jlog("req_recv", rid=rid, path="/v1/chat/completions", body_len=len(body_bytes))

    try:
        queue_headers = await _acquire_gate_early(req, gate, rid)
    except HTTPException as e:
        elog("gate_acquire_error", rid=rid, status=e.status_code, detail=str(e.detail))
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
            headers=gate.obs_headers(e.headers),
        )

    try:
        r = await _send_stream_request(client, "/v1/chat/completions", body_bytes, req, rid)
    except Exception:
        if not C.GATE_EARLY_RELEASE:
            await gate.release()
            jlog("gate_released_on_error", rid=rid, mode="anthropic_stream")
        raise

    passthrough = _build_passthrough_headers(r, gate, queue_headers)
    return StreamingResponse(
        convert_openai_stream_to_anthropic(
            _stream_gen(req, r, rid, gate=gate)
        ),
        status_code=r.status_code,
        media_type="text/event-stream",
        headers=passthrough,
    )


async def _forward_anthropic_nonstream(
    req: Request,
    body_bytes: bytes,
    rid: str,
) -> Response | JSONResponse:
    """转发转换后的 OpenAI 非流式请求，并将响应转换回 Anthropic 格式。"""
    client: httpx.AsyncClient = app.state.client
    gate: QueueGate = app.state.gate

    jlog("req_recv", rid=rid, path="/v1/chat/completions", body_len=len(body_bytes))

    try:
        queue_headers = await _acquire_gate_early_nonstream(req, gate, rid)
    except HTTPException as e:
        elog("gate_acquire_error", rid=rid, status=e.status_code, detail=str(e.detail))
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
            headers=gate.obs_headers(e.headers),
        )

    # 统一 try/finally 确保 release，与 _forward_nonstream 对齐
    try:
        r = await _send_nonstream_request(client, "/v1/chat/completions", body_bytes, req, rid)

        # 读取完整响应体
        data = await r.aread()
        await r.aclose()

        # OpenAI → Anthropic 响应转换
        openai_response = json.loads(data)
        anthropic_response = convert_openai_response_to_anthropic(openai_response)
        anthropic_bytes = json.dumps(anthropic_response, ensure_ascii=False).encode("utf-8")

        merged = _merge_obs_and_retry_headers(gate, queue_headers, r)
        entity_headers = _copy_entity_headers(r)

        return Response(
            anthropic_bytes,
            status_code=r.status_code,
            media_type="application/json",
            headers={**entity_headers, **merged},
        )
    finally:
        # 统一释放闸门：正常返回、异常、CancelledError 均能覆盖
        if not C.GATE_EARLY_RELEASE:
            await gate.release()
            jlog("gate_released_after_response", rid=rid, mode="anthropic_nonstream")


#


def _should_flush_first_packet(buf: bytearray, first_flush_done: bool, now: float, last_flush: float) -> bool:
    """判断流式传输中是否应该刷出首包数据。

    首包刷出策略用于优化 TTFT（首 Token 到达时间），满足以下任一条件即刷出：
    - 缓冲区已累积 ≥ FIRST_FLUSH_BYTES 字节
    - 启用了分隔符刷出且缓冲区包含 SSE 分隔符 "\\n\\n"
    - 距离上次刷出已超过 FIRST_FLUSH_MS 毫秒

    Args:
        buf:              当前字节缓冲区
        first_flush_done: 是否已完成首包刷出
        now:              当前时间戳（perf_counter）
        last_flush:       上次刷出的时间戳

    Returns:
        bool: True 表示应立即刷出缓冲区内容
    """
    if first_flush_done:
        return False
    if len(buf) >= C.FIRST_FLUSH_BYTES:
        return True
    if C.ENABLE_DELIM_FLUSH and b"\n\n" in buf:
        return True
    if C.FIRST_FLUSH_MS and (now - last_flush) >= C.FIRST_FLUSH_MS:
        return True
    return False


def _should_flush(buf: bytearray, dyn_bytes: int, last_flush: float, now: float) -> bool:
    """判断流式传输中是否应该刷出后续数据包。

    在首包已刷出之后的常规刷出判断，满足以下任一条件即触发刷出：
    - 缓冲区已累积 ≥ dyn_bytes 字节（含随机抖动）
    - 启用了分隔符刷出且缓冲区包含 SSE 分隔符 "\\n\\n"
    - 距离上次刷出已超过 STREAM_FLUSH_MS 毫秒

    Args:
        buf:        当前字节缓冲区
        dyn_bytes:  动态字节阈值（含随机抖动，防止多连接同步刷出）
        last_flush: 上次刷出的时间戳（perf_counter）
        now:        当前时间戳

    Returns:
        bool: True 表示应立即刷出缓冲区内容
    """
    return (
        len(buf) >= dyn_bytes or
        (C.ENABLE_DELIM_FLUSH and b"\n\n" in buf) or
        (now - last_flush) >= C.STREAM_FLUSH_MS
    )


async def _stream_gen(req: Request, r: httpx.Response, rid: str,
                      gate: QueueGate | None = None) -> AsyncIterator[bytes]:
    """流式响应生成器：按自适应策略刷出后端 SSE 数据块。

    从后端流式响应 r 中逐块读取数据，通过三级刷出策略平衡
    首包延迟 (TTFT) 和整体吞吐：
    1. 快速路径：首个小包（≤ FAST_PATH_BYTES）直接 yield，零延迟；
    2. 首包策略：累积到 FIRST_FLUSH_BYTES 或遇到分隔符时尽快刷出；
    3. 常规策略：按动态字节阈值 + 时间窗口 + 分隔符三重条件刷出。

    生成器结束时自动关闭后端响应连接。当 gate 不为 None 且
    GATE_EARLY_RELEASE=false 时，同时释放闸门槽位。

    Args:
        req:  原始客户端请求，用于检测连接断开
        r:    后端 httpx.Response 对象（stream 模式）
        rid:  请求 ID，用于日志追踪
        gate: 排队控制器（非早释放模式下传入，用于延迟释放）

    Yields:
        bytes: 刷出的字节块
    """
    buf = bytearray()
    last_flush = time.perf_counter()
    first_flush_done = False
    dyn_base = max(C.STREAM_FLUSH_BYTES, int(C.MAX_CONN / 4))

    try:
        async for chunk in r.aiter_raw():
            if not chunk:
                continue
            if await req.is_disconnected():
                elog("client_disconnected_stream", rid=rid)
                break

            #
            if not first_flush_done and len(chunk) <= C.FAST_PATH_BYTES:
                yield chunk
                first_flush_done = True
                continue

            buf.extend(chunk)
            now = time.perf_counter()

            # Flush first packet to minimize time-to-first-token
            if _should_flush_first_packet(buf, first_flush_done, now, last_flush):
                yield bytes(buf)
                buf.clear()
                first_flush_done = True
                last_flush = now
                continue

            # Dynamic flush: check buffer size threshold with jitter
            dyn_bytes = dyn_base + random.randint(0, max(1, dyn_base // 8))
            if _should_flush(buf, dyn_bytes, last_flush, now):
                yield bytes(buf)
                buf.clear()
                last_flush = now
    finally:
        if buf:
            yield bytes(buf)
        await r.aclose()
        if gate is not None and not C.GATE_EARLY_RELEASE:
            await gate.release()
            jlog("gate_released_after_response", rid=rid, mode="stream")


def _build_passthrough_headers(r: httpx.Response, gate: QueueGate, queue_headers: Dict[str, str]) -> Dict[str, str]:
    """构建流式响应的透传头集合。

    合并以下头信息用于 SSE/Chunked 流式响应：
    - 后端返回的所有 X-* 自定义头
    - X-Retry-Count（若发生了重试）
    - X-Pod-Name（当前 Pod 名称）
    - X-Accel-Buffering: no（禁止 Nginx 缓冲 SSE）
    - Cache-Control: no-transform（防止中间代理压缩 SSE）
    - 排队观测头（X-InFlight、X-Queued-Wait 等）

    Args:
        r:              后端 httpx.Response 对象
        gate:           QueueGate 排队控制器实例
        queue_headers:  排队阶段产生的头信息

    Returns:
        Dict[str, str]: 合并后的完整透传头字典
    """
    headers = {k: v for k, v in r.headers.items() if k.lower().startswith("x-")}

    ext = getattr(r, "extensions", None)
    if isinstance(ext, dict):
        retry_cnt = ext.get("app_retry_count")
        if retry_cnt is not None:
            headers["X-Retry-Count"] = str(retry_cnt)

    # 添加 Pod 名称到响应头
    headers["X-Pod-Name"] = POD_NAME

    headers.setdefault("X-Accel-Buffering", "no")
    headers.setdefault("Cache-Control", "no-transform")
    headers.update(gate.obs_headers(queue_headers))
    return headers


async def _send_stream_request(
    client: httpx.AsyncClient,
    upstream_path: str,
    body_bytes: bytes,
    req: Request,
    rid: str,
) -> httpx.Response:
    """向后端发送流式请求并返回流式响应对象。

    使用专用的超时配置：connect 超时较短以快速发现后端不可达，
    read 超时为 None（流式场景下无法预估完整响应时长），
    write/pool 使用全局默认值。

    Args:
        client:        httpx 异步客户端实例
        upstream_path: 后端路由路径（如 "/v1/chat/completions"）
        body_bytes:    序列化后的请求体字节
        req:           原始客户端请求对象（用于提取头信息）
        rid:           请求 ID，用于日志追踪

    Returns:
        httpx.Response: 后端流式响应对象（未消费 body）

    Raises:
        HTTPException(502): 后端连接失败或持续返回 5xx
    """
    url = build_backend_url(upstream_path)
    return await _send_with_fixed_retries(
        client, "POST", url,
        content=body_bytes,
        headers=make_upstream_headers(req, want_gzip=False),
        stream=True,
        timeout=httpx.Timeout(
            connect=C.STREAM_BACKEND_CONNECT_TIMEOUT,
            read=None,
            write=C.HTTPX_WRITE_TIMEOUT,
            pool=C.HTTPX_POOL_TIMEOUT,
        ),
        rid=rid,
    )


async def _acquire_gate_early(req: Request, gate: QueueGate, rid: str) -> Dict[str, str]:
    """为流式请求获取排队闸门。

    根据 GATE_EARLY_RELEASE 配置决定是否立即释放：
    - 早释放模式（GATE_EARLY_RELEASE=true）：acquire 后立即 release，
      闸门仅做准入速率控制，不限制后端并发数。
    - 持有模式（GATE_EARLY_RELEASE=false，默认）：acquire 后保持占用，
      由 _stream_gen 在流式传输结束后手动 release，实现真正的并发限制。

    Args:
        req:  原始客户端请求对象
        gate: QueueGate 排队控制器实例
        rid:  请求 ID，用于日志追踪

    Returns:
        Dict[str, str]: 排队相关的头信息（如 X-Queued-Wait）

    Raises:
        HTTPException: 排队超时或并发数超限时抛出
    """
    queue_headers = await gate.acquire(dict(req.headers))
    if C.GATE_EARLY_RELEASE:
        await gate.release()
        jlog("gate_acquired_released_early", rid=rid, wait_hdr=queue_headers.get("X-Queued-Wait"))
    else:
        jlog("gate_acquired_held", rid=rid, wait_hdr=queue_headers.get("X-Queued-Wait"))
    return queue_headers


async def _forward_stream(req: Request, upstream_path: str):
    """完整的流式请求转发流程（SSE / Chunked Transfer）。

    处理步骤：
    1. 读取并校验 JSON 请求体；
    2. 获取/释放排队闸门，记录排队等待耗时；
    3. 向后端发送请求并获取流式响应对象；
    4. 构建透传头（X-*、SSE 禁缓冲等）；
    5. 以 StreamingResponse 包装 _stream_gen 生成器返回给客户端。

    失败处理：
    - 排队异常时直接返回 JSONResponse 错误；
    - 后端连接失败时由 _send_with_fixed_retries 重试后抛出 502。

    Args:
        req:           原始客户端 Request 对象
        upstream_path: 后端路由路径（如 "/v1/chat/completions"）

    Returns:
        StreamingResponse | JSONResponse:
            正常时返回 SSE 流式响应；排队异常时返回错误 JSONResponse。
    """
    client: httpx.AsyncClient = app.state.client
    gate: QueueGate = app.state.gate
    rid = req.headers.get("x-request-id") or ""

    body_bytes = await read_json_body(req, rid)
    jlog("req_recv", rid=rid, path=str(req.url.path), body_len=len(body_bytes))

    try:
        queue_headers = await _acquire_gate_early(req, gate, rid)
    except HTTPException as e:
        elog("gate_acquire_error", rid=rid, status=e.status_code, detail=str(e.detail))
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
            headers=gate.obs_headers(e.headers)
        )

    try:
        r = await _send_stream_request(client, upstream_path, body_bytes, req, rid)
    except Exception as _:
        # 后端连接异常时释放闸门
        if not C.GATE_EARLY_RELEASE:
            await gate.release()
            jlog("gate_released_on_error", rid=rid, mode="stream")
        raise

    passthrough = _build_passthrough_headers(r, gate, queue_headers)
    return StreamingResponse(
        _stream_gen(req, r, rid, gate=gate),
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "text/event-stream"),
        headers=passthrough,
    )


async def handle_rag_scenario(req: Request, upstream_path: str):
    """处理 RAG 加速场景的流式请求（v2 新增功能）。

    当 RAG_ACC_ENABLED 为 true 时，检测请求是否匹配 RAG / Dify 场景，
    若匹配则走 Map-Reduce 加速路径，否则回退到普通流式转发。
    """
    body = await req.body()
    rid = req.headers.get("x-request-id")

    # 强制跳过
    if b"/no_rag_acc" in body:
        jlog("rag acceleration skipped forcibly", rid=rid)
        return await _forward_stream(req, upstream_path)

    # 将请求体包装为支持属性访问的视图（不依赖 fastchat 校验）
    try:
        payload_dict = json.loads(body)
        chat_input = _DictAttrView(payload_dict)
    except Exception as e:
        elog("rag_parse_error", rid=rid, detail=str(e))
        return await _forward_stream(req, upstream_path)

    # 非 RAG 请求
    is_rag = is_rag_scenario(chat_input, req)
    is_dify = is_dify_scenario(chat_input)
    if not is_rag and not is_dify:
        jlog("not rag and dify scenario", rid=rid)
        return await _forward_stream(req, upstream_path)

    jlog("rag acceleration enabled", rid=rid, backend=C.BACKEND_URL)
    return await rag_acc_chat(chat_input, request=req, backend_url=C.BACKEND_URL + upstream_path)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """聊天补全接口，根据 `stream` 字段自动切换转发路径。"""
    rid = req.headers.get("x-request-id")
    try:
        payload: Dict[str, Any] = await req.json()
    except Exception as e:
        elog("chat_json_parse_error", rid=rid, detail=str(e))
        payload = {}

    if want_stream(payload.get("stream", False)):
        # RAG 加速场景拦截（v2 新增）
        if C.RAG_ACC_ENABLED:
            return await handle_rag_scenario(req, "/v1/chat/completions")
        return await _forward_stream(req, "/v1/chat/completions")
    return await _forward_nonstream(req, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(req: Request):
    """传统 completion 接口。"""
    rid = req.headers.get("x-request-id")
    try:
        payload: Dict[str, Any] = await req.json()
    except Exception as e:
        elog("completions_json_parse_error", rid=rid, detail=str(e))
        payload = {}
    if want_stream(payload.get("stream", False)):
        return await _forward_stream(req, "/v1/completions")
    return await _forward_nonstream(req, "/v1/completions")


@app.post("/v1/messages")
async def messages(req: Request):
    """Anthropic Messages 兼容接口。

    根据 ENABLE_SMARTQOS 配置决定处理方式:
    - SmartQoS 模式（True）: 将 Anthropic 请求转换为 OpenAI 格式后转发到
      /v1/chat/completions，再将 OpenAI 响应转换回 Anthropic 格式。
      同时支持在转换时注入 priority 参数，解决 Anthropic 协议不支持
      priority 调度的问题。
    - 普通模式（False）: 直接透传 Anthropic 请求到后端 /v1/messages。
    """
    rid = req.headers.get("x-request-id") or ""
    try:
        payload: Dict[str, Any] = await req.json()
    except Exception as e:
        elog("messages_json_parse_error", rid=rid, detail=str(e))
        payload = {}

    # SmartQoS 模式: Anthropic → OpenAI 转换后转发到 /v1/chat/completions
    smartqos_enabled = os.getenv("ENABLE_SMARTQOS", "false").lower() == "true"
    if smartqos_enabled:
        return await _handle_anthropic_with_conversion(req, payload, rid)

    # 普通模式: 直接透传到后端 /v1/messages
    if want_stream(payload.get("stream", False)):
        return await _forward_stream(req, "/v1/messages")
    return await _forward_nonstream(req, "/v1/messages")


@app.post("/v1/responses")
async def responses(req: Request):
    """Responses API 兼容入口。"""
    rid = req.headers.get("x-request-id")
    try:
        payload: Dict[str, Any] = await req.json()
    except Exception as e:
        elog("responses_json_parse_error", rid=rid, detail=str(e))
        payload = {}
    if want_stream(payload.get("stream", False)):
        return await _forward_stream(req, "/v1/responses")
    return await _forward_nonstream(req, "/v1/responses")


@app.post("/v1/rerank")
async def rerank(req: Request):
    """重排序接口，将请求透传到后端 /v1/rerank 端点。"""
    return await _forward_nonstream(req, "/v1/rerank")


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    """向量嵌入接口，将请求透传到后端 /v1/embeddings 端点。"""
    return await _forward_nonstream(req, "/v1/embeddings")


@app.post("/tokenize")
async def tokenize(req: Request):
    """分词接口，将请求透传到后端的 /tokenize 端点。

    兼容性说明:
        vLLM 使用 ``{"text": "..."}`` 字段名;
        SGLang 使用 ``{"prompt": "..."}`` 字段名。
        代理层采用透传策略，不做字段翻译——调用方需根据实际
        后端引擎使用对应的字段名。(B-02)
    """
    return await _forward_nonstream(req, "/tokenize")


def _extract_metrics_headers(r: httpx.Response) -> dict:
    """保留 metrics 所需的关键响应头。"""
    headers = {k: v for k, v in r.headers.items() if k.lower().startswith("x-")}
    headers["Content-Type"] = r.headers.get(
        "content-type",
        "text/plain; version=0.0.4; charset=utf-8",
    )
    return headers


async def _pipe_metrics(req: Request, r: httpx.Response):
    """按块回传 metrics 数据。"""
    try:
        async for chunk in r.aiter_bytes():
            if not chunk:
                continue
            if await req.is_disconnected():
                elog("client_disconnected_metrics", rid=req.headers.get("x-request-id"))
                break
            yield chunk
    finally:
        await r.aclose()


@app.get("/metrics")
async def metrics(req: Request):
    """透传 backend 的 `/metrics`。"""
    client: httpx.AsyncClient = app.state.client
    url = build_backend_url("/metrics")

    r = await _send_with_fixed_retries(
        client, "GET", url,
        headers=make_upstream_headers(req, want_gzip=False),
        stream=True,
        timeout=httpx.Timeout(
            connect=C.METRICS_CONNECT_TIMEOUT,
            read=None,
            write=C.HTTPX_WRITE_TIMEOUT,
            pool=C.HTTPX_POOL_TIMEOUT,
        ),
        rid=req.headers.get("x-request-id"),
    )

    headers = _extract_metrics_headers(r)
    return StreamingResponse(
        _pipe_metrics(req, r),
        status_code=r.status_code,
        headers=headers,
    )


# `/health` 返回的是 health 状态机的当前快照，而不是现场临时探测。


@app.get("/health")
async def health_get(request: Request):
    """返回完整健康状态 JSON。"""
    h = app.state.health
    code = map_http_code_from_state(h)
    headers = build_health_headers(h)
    body = build_health_body(h, code)
    return JSONResponse(status_code=code, content=body, headers=headers)


@app.head("/health")
async def health_head(request: Request):
    """返回仅含状态码和头部的健康检查结果，适合 K8s 探针。"""
    h = app.state.health
    code = map_http_code_from_state(h)
    headers = build_health_headers(h)
    return Response(status_code=code, headers=headers)


# 模型列表接口直接透传 backend，便于前端/SDK 获取当前已加载模型。


@app.get("/v1/models")
async def models_proxy(request: Request):
    """模型列表查询接口。"""
    rid = request.headers.get("x-request-id")
    url = build_backend_url("/v1/models")
    try:
        upstream_headers = make_upstream_headers(request)
        resp = await app.state.client.get(url, headers=upstream_headers, timeout=10.0)
        entity_headers = _copy_entity_headers(resp)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=entity_headers,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except Exception as e:
        elog("models_proxy_error", rid=rid, detail=str(e))
        raise HTTPException(status_code=502, detail="backend unavailable") from e


@app.get("/v1/version")
async def version_proxy(req: Request):
    """返回 sidecar 自身版本信息，便于部署排查。"""
    rid = req.headers.get("x-request-id")

    version = os.getenv("WINGS_VERSION", "25.0.0.1")
    build_date = os.getenv("WINGS_BUILD_DATE", "2025-08-30")

    return JSONResponse(
        status_code=200,
        content={
            "WINGS_VERSION": version,
            "WINGS_BUILD_DATE": build_date
        }
    )

