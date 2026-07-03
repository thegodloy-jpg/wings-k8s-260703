# -*- coding: utf-8 -*-
"""
tagging.py - 标签常量和辅助工具。

功能:
  - want_stream(): 布尔参数解析
  - make_upstream_headers(): 生成转发头
  - read_json_body(): 读取并校验 JSON 请求体
  - jlog()/elog(): JSON 结构化日志
  - ms(): 时间格式化
  - build_backend_url(): URL 构建 (scheme+host+port)
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional
from urllib.parse import urlsplit
import json
from fastapi import HTTPException, Request
import orjson

from . import proxy_config as C


def want_stream(v: Any) -> bool:
    """判断请求是否要求流式响应。

    将各种类型的输入统一解析为布尔值，用于决定是否以 SSE 流式方式返回推理结果。
    支持 bool、int/float、以及字符串形式的真值（'1'/'true'/'yes'/'y'/'t'/'on'）。

    Args:
        v: 请求体中 stream 字段的原始值，可以是 bool / int / float / str 类型。

    Returns:
        bool: True 表示客户端请求流式响应，False 表示非流式。
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t", "on")
    return False


def make_upstream_headers(req: Request, want_gzip: bool = False) -> Dict[str, str]:
    """构造转发到后端引擎的 HTTP 请求头。

    从原始请求中提取需要透传的头部（Authorization、X-Request-Id），
    并设置 content-type、accept-encoding、connection 等固定头部。

    Args:
        req: 客户端发来的原始 FastAPI Request 对象。
        want_gzip: 是否在 accept-encoding 中使用 gzip。
            默认使用 identity（不压缩），避免增加 TTFT 延迟。

    Returns:
        Dict[str, str]: 构建好的请求头字典，至少包含 content-type、
            accept-encoding 和 connection 三个键。
    """
    h = {
        "content-type": "application/json",
        "accept-encoding": "gzip" if want_gzip else "identity",
        "connection": "keep-alive",
    }
    auth = req.headers.get("authorization")
    if auth:
        h["authorization"] = auth
    rid = req.headers.get("x-request-id")
    if rid:
        h["x-request-id"] = rid
    return h


async def read_json_body(req: Request, rid: Optional[str]) -> bytes:
    """读取并校验请求体中的 JSON 数据。

    依次执行以下检查：
      1. 读取原始请求体字节。
      2. 尝试用 orjson 反序列化以验证 JSON 合法性，失败则返回 HTTP 400。

    Args:
        req: 客户端发来的 FastAPI Request 对象。
        rid: 请求 ID，用于错误日志关联。

    Returns:
        bytes: 经过校验的原始 JSON 字节串（未修改内容）。

    Raises:
        HTTPException: 400 - 请求体不是合法 JSON。
    """
    body = await req.body()
    try:
        _ = orjson.loads(body)
    except Exception as e:
        elog("req_json_error", rid=rid, error=str(e))
        raise HTTPException(400, f"invalid json: {e}") from e
    return body


def jlog(evt: str, **fields):
    """ INFO JSON """
    try:
        log_entry = {"evt": evt, **fields}
        C.logger.info(json.dumps(log_entry, ensure_ascii=False))
    except (TypeError, ValueError) as e:
        C.logger.error("Failed to serialize log entry: %s", log_entry, exc_info=e)
    except Exception as e:
        C.logger.error("Unexpected error while writing log", exc_info=e)


def elog(evt: str, **fields):
    """ ERROR JSON """
    try:
        log_entry = {"evt": evt, **fields}
        C.logger.error(json.dumps(log_entry, ensure_ascii=False))
    except (TypeError, ValueError) as e:
        C.logger.error("Failed to serialize log entry: %s", log_entry, exc_info=e)
    except Exception as e:
        C.logger.error("Unexpected error while writing log", exc_info=e)


def ms(sec: float) -> str:
    """ -> x.xms"""
    return f"{sec * 1000:.1f}ms"


def _backend_origin() -> str:
    """从 BACKEND_URL 中提取 origin 部分（scheme://host[:port]）。

    仅保留协议和主机端口，丢弃 path、query、fragment 等部分。
    如果 BACKEND_URL 中包含路径组件，会输出警告日志。

    Returns:
        str: 格式为 ``scheme://netloc`` 的 origin 字符串，
            例如 ``http://127.0.0.1:17000``。

    Raises:
        ValueError: 当 BACKEND_URL 缺少 scheme 或 netloc 时抛出。
    """
    u = urlsplit(C.BACKEND_URL.strip())
    if not u.scheme or not u.netloc:
        raise ValueError(
            f"BACKEND_URL must include scheme and netloc "
            f"(e.g., 'http://127.0.0.1:17000'); got: {C.BACKEND_URL!r}"
        )
    if u.path and u.path not in ("", "/"):
        C.logger.warning(
            "BACKEND_URL '%s' contains unexpected path '%s'; only scheme://host:port is used",
            C.BACKEND_URL, u.path,
        )
    return f"{u.scheme}://{u.netloc}"


def build_backend_url(path: str) -> str:
    """拼接后端引擎的完整 URL。

    将 origin（scheme+host+port）与指定 path 拼接成完整的后端请求 URL。
    如果 path 不以 '/' 开头，会自动补全。

    Args:
        path: API 路径，例如 ``/v1/chat/completions``。

    Returns:
        str: 完整的后端 URL，例如 ``http://10.0.0.8:17000/v1/chat/completions``。
    """
    if not path.startswith("/"):
        path = "/" + path
    return _backend_origin() + path


def rebuild_request_json(req: Request, new_payload: Dict[str, Any]) -> Request:
    """用新的 JSON payload 重建 Request 对象。

    复制原始请求的 scope 并替换请求体内容，同时更新 content-length 头部
    以匹配新 payload 的字节长度。返回一个全新的 Request 实例。

    Args:
        req: 原始的 FastAPI Request 对象。
        new_payload: 新的请求体字典，将被序列化为 JSON。

    Returns:
        Request: 携带新 payload 的全新 Request 对象。
    """
    new_bytes = json.dumps(new_payload, ensure_ascii=False).encode("utf-8")

    # 1.  content-length
    new_scope = req.scope.copy()
    headers = []
    for header_name, header_value in req.scope["headers"]:
        if header_name == b"content-length":
            header_value = str(len(new_bytes)).encode()
        headers.append((header_name, header_value))
    new_scope["headers"] = headers

    # 2.  receive
    async def receive() -> dict:
        return {"type": "http.request", "body": new_bytes, "more_body": False}

    return Request(new_scope, receive)