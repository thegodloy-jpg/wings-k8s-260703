# -*- coding: utf-8 -*-
"""
http_client.py - 异步 HTTP 客户端工厂。

负责:
  - 创建配置统一的 httpx.AsyncClient 实例
  - 启用 HTTP/2、连接池、keepalive 等特性
  - 不设置 base_url，由 gateway 拼接完整 URL
"""

from __future__ import annotations
import httpx
from . import proxy_config as C


async def create_async_client() -> httpx.AsyncClient:
    """创建配置统一的异步 HTTP 客户端实例。

    配置说明:
        - max_connections:          最大连接数 (C.MAX_CONN)
        - max_keepalive_connections: 最大 keep-alive 连接数 (C.MAX_KEEPALIVE)
        - keepalive_expiry:         keep-alive 超时秒数
        - http2:                    是否启用 HTTP/2 (C.HTTP2_ENABLED)
        - connect timeout:          连接超时 (C.HTTPX_CONNECT_TIMEOUT)
        - read timeout:             None - 流式响应可能持续很长
        - write timeout:            写入超时 (C.HTTPX_WRITE_TIMEOUT)
        - pool timeout:             连接池超时 (C.HTTPX_POOL_TIMEOUT)

    Returns:
        httpx.AsyncClient: 配置完成的客户端实例

    注意:
        - trust_env=False 禁止继承系统代理配置，避免本地后端走代理
        - follow_redirects=False 不跟随重定向，避免性能不可预测
    """
    limits = httpx.Limits(
        max_connections=C.MAX_CONN,
        max_keepalive_connections=C.MAX_KEEPALIVE,
        keepalive_expiry=C.KEEPALIVE_EXPIRY,
    )

    # 注意: 当指定 transport 时，AsyncClient 的 limits 参数会被忽略，
    # 因此必须将 limits 传给 AsyncHTTPTransport 构造函数。
    transport = httpx.AsyncHTTPTransport(
        http2=C.HTTP2_ENABLED,
        limits=limits,
    )

    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(
            connect=C.HTTPX_CONNECT_TIMEOUT,
            read=None,  # No read timeout: streaming responses may take unbounded time
            write=C.HTTPX_WRITE_TIMEOUT,
            pool=C.HTTPX_POOL_TIMEOUT,
        ),
        follow_redirects=False,
        headers={"connection": "keep-alive"},
        trust_env=False,  # Ignore OS-level proxy settings managed by Wings
        # base_url not set; resolved dynamically per-request in gateway
    )
    return client