# -*- coding: utf-8 -*-
"""Proxy 运行时配置。

默认值优先考虑稳定性：
- 避免过于激进的连接池配置
- 默认禁用 HTTP/2（部分上游引擎在 HTTP/1.1 上更稳定）
- 流式 chat 流量对瞬时传输问题比较敏感
"""

import argparse
import logging
import os

from utils.log_config import setup_root_logging, LOGGER_PROXY

# Proxy: only show ERROR on console, full logs saved to file
setup_root_logging(stderr_level="ERROR")
logger = logging.getLogger(LOGGER_PROXY)


def parse_args():
    """解析代理专属的命令行参数，保留 launcher 的参数不受影响。

    支持的参数:
        --backend: 后端引擎地址 (BACKEND_URL)
        --host:    代理监听地址 (HOST)
        --port:    代理监听端口 (PORT)

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", default=os.getenv("BACKEND_URL", "http://127.0.0.1:17000"))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "18000")))
    parsed_args, _ = parser.parse_known_args()
    return parsed_args


args = parse_args()
BACKEND_URL = args.backend.strip()
HOST = args.host
PORT = args.port

BACKEND_PROBE_TIMEOUT = int(os.getenv("BACKEND_PROBE_TIMEOUT", "3600"))

# Do not inherit system proxy settings for local backend traffic.
logger.info("Clearing system proxy environment variables to prevent httpx from picking them up")
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(key, None)

# Streaming flush policy.
FAST_PATH_BYTES = int(os.getenv("FAST_PATH_BYTES", "128"))
FIRST_FLUSH_BYTES = int(os.getenv("FIRST_FLUSH_BYTES", "256"))
FIRST_FLUSH_MS = float(os.getenv("FIRST_FLUSH_MS", "0.0"))
STREAM_FLUSH_BYTES = int(os.getenv("STREAM_FLUSH_BYTES", "8192"))
STREAM_FLUSH_MS = float(os.getenv("STREAM_FLUSH_MS", "0.006"))
NONSTREAM_THRESHOLD = int(os.getenv("NONSTREAM_PIPE_THRESHOLD", str(256 * 1024)))

# Connection pool defaults (aligned with wings production defaults).
MAX_CONN = int(os.getenv("HTTPX_MAX_CONNECTIONS", "2048"))
MAX_KEEPALIVE = int(os.getenv("HTTPX_MAX_KEEPALIVE", "256"))
KEEPALIVE_EXPIRY = float(os.getenv("HTTPX_KEEPALIVE_EXPIRY", "30"))

# HTTP/2 enabled by default (aligned with wings production defaults).
HTTP2_ENABLED = os.getenv("HTTP2_ENABLED", "true").lower() != "false"
H2_MAX_STREAMS = int(os.getenv("HTTP2_MAX_STREAMS", "128"))

# Retry policy for transient backend errors (aligned with wings: 3 tries, 100ms interval).
RETRY_TRIES = int(os.getenv("RETRY_TRIES", "3"))
RETRY_INTERVAL_MS = int(os.getenv("RETRY_INTERVAL_MS", "100"))
ENABLE_DELIM_FLUSH = os.getenv("ENABLE_DELIM_FLUSH", "true").lower() != "false"

# Client and per-endpoint timeout tuning.
HTTPX_CONNECT_TIMEOUT = float(os.getenv("HTTPX_CONNECT_TIMEOUT", "20"))
HTTPX_WRITE_TIMEOUT = float(os.getenv("HTTPX_WRITE_TIMEOUT", "20"))
HTTPX_POOL_TIMEOUT = float(os.getenv("HTTPX_POOL_TIMEOUT", "30"))
STREAM_BACKEND_CONNECT_TIMEOUT = float(os.getenv("STREAM_BACKEND_CONNECT_TIMEOUT", "20"))
METRICS_CONNECT_TIMEOUT = float(os.getenv("METRICS_CONNECT_TIMEOUT", "10"))
STATUS_CONNECT_TIMEOUT = float(os.getenv("STATUS_CONNECT_TIMEOUT", "10"))
STATUS_READ_TIMEOUT = float(os.getenv("STATUS_READ_TIMEOUT", "30"))

WARMUP_CONN = int(os.getenv("WARMUP_CONN", str(min(MAX_KEEPALIVE or 50, 200))))
WARMUP_PROMPT = os.getenv("WARMUP_PROMPT", "").strip()
WARMUP_ROUNDS = int(os.getenv("WARMUP_ROUNDS", "1"))
WARMUP_TIMEOUT = float(os.getenv("WARMUP_TIMEOUT", "10"))

GLOBAL_PASS_THROUGH_LIMIT = int(os.getenv("GLOBAL_PASS_THROUGH_LIMIT", "1024"))
GLOBAL_QUEUE_MAXSIZE = int(os.getenv("GLOBAL_QUEUE_MAXSIZE", "1024"))

# 上限 128：允许高并发场景下分配更多 worker 进程处理转发请求。
# 每个 worker 约 65MB RAM，128 workers ≈ 8GB，在生产服务器上可接受。
_MAX_PROXY_WORKERS = 32
WORKERS = min(int(os.getenv("PROXY_WORKERS", "32")), _MAX_PROXY_WORKERS)
WORKER_INDEX = int(os.getenv("WORKER_INDEX", "-1"))
RAG_ACC_ENABLED = os.getenv("RAG_ACC_ENABLED", "false").lower() != "false"


def _split_strict(total: int, workers: int, idx: int) -> int:
    """将全局配额严格均分给每个 worker。

    使用整除 + 余数策略：前 ``total % workers`` 个 worker 各多分配 1，
    保证所有 worker 的配额之和严格等于 total。

    Args:
        total: 全局总配额（如并发上限或队列容量）。
        workers: worker 总数，若 <= 0 则直接返回 total。
        idx: 当前 worker 索引（0-based），若不在 [0, workers) 范围内则返回 base 值。

    Returns:
        int: 当前 worker 分配到的本地配额。
    """
    if workers <= 0:
        return total
    base = total // workers
    extra = total % workers
    if 0 <= idx < workers:
        return base + (1 if idx < extra else 0)
    return base


# ---------------------------------------------------------------------------
# 排队和并发控制参数
#
# 设计思路：
#   - 全局配额（GLOBAL_*）用于描述整个 Pod/容器级别的并发上限和队列容量。
#   - 本地配额（LOCAL_*）通过 _split_strict() 将全局配额均分给每个 worker，
#     避免多 worker 进程之间超发。
#   - 双闸门模型（Gate-0 / Gate-1）用于分层流控：
#       Gate-0: 零等待的快速通道（容量 = GATE0_LOCAL_CAP）
#       Gate-1: 弹性缓冲通道（容量 = LOCAL_PASS_THROUGH_LIMIT - GATE0_LOCAL_CAP）
# ---------------------------------------------------------------------------
LOCAL_PASS_THROUGH_LIMIT = _split_strict(GLOBAL_PASS_THROUGH_LIMIT, WORKERS, WORKER_INDEX)
LOCAL_QUEUE_MAXSIZE = _split_strict(GLOBAL_QUEUE_MAXSIZE, WORKERS, WORKER_INDEX)
MAX_INFLIGHT = LOCAL_PASS_THROUGH_LIMIT
QUEUE_MAXSIZE = LOCAL_QUEUE_MAXSIZE
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "15.0"))
 
QUEUE_REJECT_POLICY = os.getenv("QUEUE_REJECT_POLICY", "drop_oldest").lower()
QUEUE_OVERFLOW_MODE = os.getenv("QUEUE_OVERFLOW_MODE", "block").lower()
 
GATE0_TOTAL = WORKERS
GATE0_LOCAL_CAP = _split_strict(GATE0_TOTAL, WORKERS, WORKER_INDEX)
GATE1_LOCAL_CAP = max(0, LOCAL_PASS_THROUGH_LIMIT - GATE0_LOCAL_CAP)

USE_GLOBAL_GATE = os.getenv("USE_GLOBAL_GATE", "false").lower() == "true"
GATE_SOCK = os.getenv("GATE_SOCK", "")

# ---------------------------------------------------------------------------
# 早释放开关（Gate Early Release）
#
# 默认 true（开启早释放）：acquire 后立即 release，闸门仅做
# 准入速率控制（rate limiting），不限制后端并发数。
# 设置为 false 时关闭早释放：闸门在整个后端请求期间保持占用，
# 真正限制后端并发数（concurrency limiting）。
# ---------------------------------------------------------------------------
GATE_EARLY_RELEASE = os.getenv("GATE_EARLY_RELEASE", "true").lower() == "true"


def log_boot_plan():
    """在服务启动时输出当前生效的代理运行时配置摘要。

    输出内容包括：后端地址、worker 布局、全局/本地并发参数、
    闸门容量、HTTP/2 状态、重试策略及超时配置。
    供运维人员在日志中快速确认配置是否符合预期。
    """
    logger.info(
        "Plan: WORKERS=%s INDEX=%s | GLOBAL(inflight=%s, queue=%s) -> LOCAL(inflight=%s, queue=%s) | "
        "GATE0_TOTAL=%s -> G0_LOCAL=%s, G1_LOCAL=%s | HTTP2=%s H2_MAX_STREAMS=%s | "
        "RETRY_TRIES=%s INTERVAL=%sms | CONNECT=%ss POOL=%ss | GATE_EARLY_RELEASE=%s",
        WORKERS,
        WORKER_INDEX,
        GLOBAL_PASS_THROUGH_LIMIT,
        GLOBAL_QUEUE_MAXSIZE,
        LOCAL_PASS_THROUGH_LIMIT,
        LOCAL_QUEUE_MAXSIZE,
        GATE0_TOTAL,
        GATE0_LOCAL_CAP,
        GATE1_LOCAL_CAP,
        HTTP2_ENABLED,
        H2_MAX_STREAMS,
        RETRY_TRIES,
        RETRY_INTERVAL_MS,
        HTTPX_CONNECT_TIMEOUT,
        HTTPX_POOL_TIMEOUT,
        GATE_EARLY_RELEASE,
    )
