"""
=============================================================================
 Launcher 主入口模块 (wings_control.py)
=============================================================================

功能概述：
    这是 sidecar 控制容器的主入口，负责整个推理服务的生命周期管理。
    它不直接运行推理引擎，而是作为协调器完成以下三个核心职责：

核心职责：
    1. 参数解析与脚本生成
       - 解析 CLI 启动参数和环境变量
       - 结合端口规划（PortPlan）生成 engine 启动脚本
       - 调用 build_launcher_plan() 完成配置合并和命令拼装

    2. 启动脚本传递
       - 将生成的 shell 脚本写入共享卷 (/shared-volume/start_command.sh)
       - engine 容器通过挂载同一共享卷读取脚本并执行
       - 实现跨容器的命令传递和参数同步

    3. 子服务托管
       - 启动并守护 proxy（反向代理）和 health（健康检查）两个 FastAPI 子服务
       - 监控子进程状态，异常退出时自动拉起（守护进程模式）
       - 处理系统信号（SIGINT/SIGTERM）实现优雅退出

    4. 分布式协调（DISTRIBUTED=true 时激活）
       - 通过 RANK_IP vs MASTER_IP 比较自动判断 master/worker 角色
       - Master: 生成 rank0 脚本 + 启动 Master API + 等待 Worker 注册后分发启动指令
       - Worker: 启动 Worker API + 向 Master 注册 + 接收启动指令写入共享卷
       - 支持 MASTER_IP 为 DNS 名称（通过 DNS 解析后比较）

Sidecar 架构说明：
    ┌─────────────────────────────────────────────────────────────┐
    │                      K8s Pod                                │
    │  ┌─────────────────────┐    ┌─────────────────────────────┐ │
    │  │   Launcher 容器     │    │      Engine 容器            │ │
    │  │  (wings-control)      │    │  (vllm/sglang/mindie)       │ │
    │  │                     │    │                             │ │
    │  │  wings_control.py ───────┤────┤──> start_command.sh         │ │
    │  │       ↓             │    │         ↓                   │ │
    │  │  proxy:18000        │    │    engine:17000             │ │
    │  │  health:19000       │    │                             │ │
    │  └─────────────────────┘    └─────────────────────────────┘ │
    │              ↑                          ↑                   │
    │              └──────── 共享卷 ───────────┘                   │
    │                   /shared-volume/                           │
    └─────────────────────────────────────────────────────────────┘

关键设计点：
    - launcher 本身不直接启动推理引擎进程，避免跨容器进程管理的复杂性
    - 通过共享卷传递脚本，实现 launcher 与 engine 容器的解耦
    - 分布式场景下，只有 rank0 节点暴露 proxy，其他节点仅保留 health 服务

使用方式：
    # 作为模块运行
    python -m wings_control --model-name DeepSeek-R1 --model-path /weights

    # 或通过 run() 函数调用
    from wings_control import run
    sys.exit(run(['--model-name', 'MyModel', ...]))
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Sequence

import requests

from config.settings import settings
from core.port_plan import PortPlan, derive_port_plan
from core.start_args_compat import LaunchArgs, parse_launch_args
from core.wings_entry import build_launcher_plan
from utils.env_utils import get_local_ip, get_master_ip, get_node_ips
from utils.file_utils import safe_write_file, WriteOptions
from utils.log_config import setup_root_logging, LOGGER_LAUNCHER
from utils.noise_filter import install_noise_filters

setup_root_logging()
logger = logging.getLogger(LOGGER_LAUNCHER)

# 安装噪声过滤器：抑制 /health 访问日志、batch 噪声、pynvml FutureWarning
# 旧版 wings.py 在模块加载时调用，新版需在 launcher 入口显式调用
install_noise_filters()

_CHILD_STRUCTURED_LOG_RE = re.compile(
    r"^(?:\[WINGS-CONTROL\]\s+)?"
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?\s+"
    r"\[(?P<level>WARNING|ERROR|CRITICAL)\]\s+"
    r"\[(?P<logger>[^\]]+)\]\s+"
    r"(?P<message>.*)$",
    re.IGNORECASE,
)

_CHILD_LEVEL_RE = re.compile(
    r"\[(?P<bracket>WARNING|ERROR|CRITICAL)\]"
    r"|^(?P<prefix>WARNING|ERROR|CRITICAL):"
    r"|\b(?P<error>Traceback|Exception|Error:)",
    re.IGNORECASE,
)

_LOG_LEVELS = {
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "TRACEBACK": logging.ERROR,
    "EXCEPTION": logging.ERROR,
    "ERROR:": logging.ERROR,
}


def _normalize_child_log_line(
    proc_name: str,
    line: str,
) -> tuple[str, int, str] | None:
    """Parse a managed child-process log line for launcher relay."""
    text = line.rstrip()
    if not text:
        return None

    structured = _CHILD_STRUCTURED_LOG_RE.match(text)
    if structured:
        level_name = structured.group("level").upper()
        return (
            structured.group("logger"),
            _LOG_LEVELS[level_name],
            f"[{proc_name}] {structured.group('message')}",
        )

    level_match = _CHILD_LEVEL_RE.search(text)
    if not level_match:
        return None

    level_name = (
        level_match.group("bracket")
        or level_match.group("prefix")
        or level_match.group("error")
        or "WARNING"
    ).upper()
    return LOGGER_LAUNCHER, _LOG_LEVELS.get(level_name, logging.WARNING), f"[{proc_name}] {text}"


@dataclass
class DistTopology:
    """分布式拓扑信息，描述多节点集群的基本参数。

    Attributes:
        nnodes:    集群总节点数
        head_addr: head 节点地址（rank-0 IP 或主机名）
        node_ips:  所有节点 IP 列表
    """

    nnodes: int
    head_addr: str
    node_ips: list[str]


@dataclass
class DispatchOptions:
    """分发引擎到 Worker 节点的可选配置。

    Attributes:
        max_retries:    最大重试次数
        retry_interval: 重试间隔（秒）
    """

    max_retries: int = 3
    retry_interval: float = 3.0


@dataclass
class ManagedProc:
    """描述一个由 launcher 托管的子进程。

    这个数据类封装了子进程的完整元数据，用于 launcher 的进程守护循环。
    支持进程启动、停止、状态检查等生命周期操作。

    Attributes:
        name: 进程名称标识（如 'proxy'、'health'），用于日志和调试
        argv: 进程启动命令行参数列表，第一个元素通常是 python 解释器路径
        env:  进程环境变量字典，包含 BACKEND_URL、PORT 等运行时配置
        proc: subprocess.Popen 实例，进程未启动时为 None

    使用示例：
        >>> proxy = ManagedProc(
        ...     name='proxy',
        ...     argv=['python', '-m', 'uvicorn', 'proxy.gateway:app'],
        ...     env={'PORT': '18000', 'BACKEND_URL': 'http://127.0.0.1:17000'}
        ... )
        >>> _start(proxy)  # 启动进程
        >>> _stop(proxy)   # 停止进程
    """

    name: str           # 进程名称标识，用于日志打印
    argv: list[str]     # 命令行参数列表[python, -m, uvicorn, ...]
    env: dict[str, str] # 环境变量字典，继承自父进程并添加服务特定变量
    proc: subprocess.Popen | None = None  # 实际的子进程句柄
    dedupe_output: bool = False      # 是否过滤重复的 uvicorn worker 启动日志
    crash_count: int = 0            # 连续崩溃计数器
    last_start_ts: float = 0.0     # 上次启动时间戳
    backoff_until: float = 0.0     # 退避期截止时间戳


def _start_output_filter_thread(proc: ManagedProc) -> None:
    """启动守护线程，从子进程 stdout 读取并过滤低优先级日志。

    只转发 WARNING / ERROR / CRITICAL 级别的消息到 launcher 日志，
    所有 INFO 及以下级别（含 uvicorn worker 启动日志、proxy 应用日志）静默丢弃。
    这显著减少了多 worker 场景下的日志噪声。
    """
    def _relay() -> None:
        try:
            for raw in iter(proc.proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                relay = _normalize_child_log_line(proc.name, line)
                if relay is not None:
                    logger_name, level, message = relay
                    logging.getLogger(logger_name).log(level, message)
        except (ValueError, OSError):
            pass

    t = threading.Thread(target=_relay, name=f"filter-{proc.name}", daemon=True)
    t.start()


def _start(proc: ManagedProc) -> None:
    """启动单个托管子进程。

    使用 subprocess.Popen 创建子进程，继承当前进程的标准输入输出。
    启动失败时仅记录错误日志，不抛出异常，允许守护循环继续尝试。

    Args:
        proc: 待启动的托管进程对象，启动后 proc.proc 将被设置为 Popen 实例

    注意事项:
        - 子进程使用指定的 env 字典作为环境变量，不会自动继承父进程变量
        - 启动失败通常是由于命令不存在或权限不足，需检查 argv[0] 路径
        - 启动成功后需要通过 poll() 检查进程是否正常运行
        - dedupe_output=True 时，stdout/stderr 重定向到过滤线程，
          仅保留一条 uvicorn worker 启动日志
    """
    logger.info("Starting subprocess %s: %s", proc.name, " ".join(proc.argv))
    try:
        if proc.dedupe_output:
            # 多 worker 场景：管道化输出并通过过滤线程去重
            proc.proc = subprocess.Popen(
                proc.argv, env=proc.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            _start_output_filter_thread(proc)
        else:
            # 使用 Popen 创建子进程，env 参数完全替换（非继承）父进程环境
            proc.proc = subprocess.Popen(proc.argv, env=proc.env)
    except OSError as e:
        # OSError 通常表示可执行文件不存在或权限问题
        logger.error("Failed to start %s: %s", proc.name, e)


def _kill_process_forcefully(proc: ManagedProc) -> None:
    """向已超过 SIGTERM 等待期的进程发送 SIGKILL。

    等待最多 5 秒；若仍未退出则放弃并记录警告。
    在容器环境中，僵尸进程由 init 进程收割，影响较小。
    """
    logger.warning("%s did not respond to SIGTERM, sending SIGKILL", proc.name)
    proc.proc.kill()  # type: ignore[union-attr]
    try:
        proc.proc.wait(timeout=5)  # type: ignore[union-attr]
    except subprocess.TimeoutExpired:
        logger.warning("%s still running after SIGKILL, giving up", proc.name)


def _stop(proc: ManagedProc) -> None:
    """优雅停止托管子进程，必要时强制终止。

    采用两阶段停止策略：
    1. 首先发送 SIGTERM 信号请求优雅退出，等待最多 10 秒。
    2. 若进程未响应，调用 _kill_process_forcefully() 发送 SIGKILL。
    3. 若仍未退出，放弃等待并记录警告。

    Args:
        proc: 待停止的托管进程对象，停止后 proc.proc 将被置为 None
    """
    if not proc.proc:
        return  # 进程从未启动或已被清理

    if proc.proc.poll() is None:
        logger.info("Sending SIGTERM to %s (pid=%d)", proc.name, proc.proc.pid)
        proc.proc.terminate()
        try:
            proc.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _kill_process_forcefully(proc)
    proc.proc = None  # 清理引用


def _restart_if_needed(proc: ManagedProc) -> None:
    """检查进程状态，必要时自动重启（带崩溃循环保护）。

    该函数在主循环中周期性调用，实现子进程的自动恢复：
    - 进程从未启动 → 立即启动
    - 进程正在运行 → 不做操作
    - 进程已退出   → 记录退出码并重启（带指数退避）

    崩溃循环保护策略：
    - 进程在 30 秒内退出视为崩溃，连续崩溃计数器递增
    - 连续崩溃时指数退避：等待 min(2^crash_count, 60) 秒后再重启
    - 进程稳定运行超过 30 秒后退出，崩溃计数器重置
    """
    crash_threshold_sec = 30  # 小于此时间视为崩溃
    max_backoff_sec = 60      # 最大退避时间

    if not proc.proc:
        # 进程从未启动或已标记为待重启
        # 检查是否在退避等待期内（崩溃循环保护）
        if proc.backoff_until and time.time() < proc.backoff_until:
            return  # 退避期内，跳过本轮
        proc.last_start_ts = time.time()
        _start(proc)
        return

    # poll() 返回 None 表示进程仍在运行
    code = proc.proc.poll()
    if code is None:
        return  # 进程正常运行，无需操作

    # ── 进程已退出，记录崩溃/正常退出并清理句柄 ──
    uptime = time.time() - proc.last_start_ts if proc.last_start_ts else 0

    # 清理已退出进程句柄，下轮将通过 "not proc.proc" 分支重启
    proc.proc = None

    if uptime < crash_threshold_sec:
        # 短时间内退出视为崩溃
        proc.crash_count += 1
        backoff = min(2 ** proc.crash_count, max_backoff_sec)
        proc.backoff_until = time.time() + backoff
        logger.warning(
            "[Restart] %s exited abnormally (code=%s, uptime=%.1fs), "
            "crash_count=%d, waiting %ds before restart...",
            proc.name, code, uptime, proc.crash_count, backoff,
        )
        # 退避期内不重启，下轮若已过退避期则经 "not proc.proc" 分支启动
    else:
        # 稳定运行后退出，重置崩溃计数器并立即重启
        proc.crash_count = 0
        logger.warning("[Restart] %s exited (code=%s, uptime=%.1fs), resetting crash count and restarting immediately",
                       proc.name, code, uptime)
        proc.last_start_ts = time.time()
        _start(proc)


# Prometheus multi-process metrics 共享目录，放在日志共享卷下
_PROMETHEUS_MULTIPROC_DIR = "/var/log/wings/prometheus_multiproc"


def _build_child_env(port_plan: PortPlan) -> dict[str, str]:
    """为 proxy/health 子进程准备环境变量。"""
    env = os.environ.copy()

    # 后端地址：sidecar 与 engine 在同一 Pod 内共享网络命名空间。
    # 分布式模式下 RANK_IP（Pod IP）可直接访问 engine；
    # 单机/本地开发无 RANK_IP 时回退到 127.0.0.1。
    rank_ip = os.getenv("RANK_IP")
    backend_host = rank_ip if rank_ip else "127.0.0.1"

    env["BACKEND_URL"] = f"http://{backend_host}:{port_plan.backend_port}"
    env["BACKEND_HOST"] = backend_host
    env["BACKEND_PORT"] = str(port_plan.backend_port)
    env["PORT"] = str(port_plan.proxy_port)
    env["PROXY_PORT"] = str(port_plan.proxy_port)
    env["HEALTH_PORT"] = str(port_plan.health_port)
    env["HEALTH_SERVICE_PORT"] = str(port_plan.health_port)

    # 引擎类型：K8s 部署通过 ENGINE 环境变量传入（如 mindie/vllm/sglang），
    # 子进程（health_router/proxy/health_service）统一通过 ENGINE 读取。
    engine = os.getenv("ENGINE", "vllm")
    env["ENGINE"] = engine.strip().lower()

    # Prometheus 多进程指标汇总目录。
    # 当 proxy 以多 worker 模式运行时(--workers > 1)，prometheus_client
    # 需要此目录来汇总各 worker 进程的 Gauge/Counter 指标，否则 /metrics
    # 只能返回当前响应进程的部分数据。
    # 目录位于日志共享卷 /var/log/wings 下，engine 容器也使用同一路径。
    _ensure_prometheus_multiproc_dir()
    env["PROMETHEUS_MULTIPROC_DIR"] = _PROMETHEUS_MULTIPROC_DIR

    return env


def _ensure_prometheus_multiproc_dir() -> None:
    """确保 Prometheus 多进程目录存在且为空（清除上次残留的 .db 文件）。"""
    import shutil
    try:
        if os.path.isdir(_PROMETHEUS_MULTIPROC_DIR):
            shutil.rmtree(_PROMETHEUS_MULTIPROC_DIR)
        os.makedirs(_PROMETHEUS_MULTIPROC_DIR, exist_ok=True)
    except OSError as e:
        logger.warning("Failed to prepare PROMETHEUS_MULTIPROC_DIR=%s: %s",
                       _PROMETHEUS_MULTIPROC_DIR, e)


def _build_proxy_proc(
    port_plan: PortPlan, env: dict, python_bin: str, uvicorn_mod: str, proxy_workers: int
) -> "ManagedProc":
    """构建 proxy 进程的 ManagedProc 对象。"""
    # 仅打印 ERROR 级别到 stdout/stderr，完整日志由 RotatingFileHandler 写入文件。
    # 应用自身日志（wings-proxy / log-center）的 ERROR 仍可见。
    uvicorn_log_level = "error"
    proxy_argv = [
        python_bin, "-m", uvicorn_mod, settings.PROXY_APP,
        "--host", "0.0.0.0", "--port", str(port_plan.proxy_port),
        "--log-level", uvicorn_log_level,
    ]
    if proxy_workers > 1:
        proxy_argv += ["--workers", str(proxy_workers)]
        # 多 worker 场景下增大 TCP backlog，防止高并发时连接被拒
        proxy_argv += ["--backlog", "8192"]
    return ManagedProc(name="proxy", argv=proxy_argv, env=env.copy(), dedupe_output=(proxy_workers > 1))


def _build_health_proc(
    port_plan: PortPlan, env: dict, python_bin: str, uvicorn_mod: str
) -> "ManagedProc":
    """构建 health 进程的 ManagedProc 对象。"""
    return ManagedProc(
        name="health",
        argv=[
            python_bin, "-m", uvicorn_mod, settings.HEALTH_APP,
            "--host", "0.0.0.0", "--port", str(port_plan.health_port),
            "--log-level", "info",
        ],
        env=env.copy(),
    )


def _detect_cgroup_memory_mb() -> int | None:
    """读取 cgroup 内存限制（兼容 v1/v2），返回 MB；读取失败返回 None。"""
    paths = [
        "/sys/fs/cgroup/memory.max",                  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes", # cgroup v1
    ]
    for p in paths:
        try:
            raw = Path(p).read_text().strip()
            if raw == "max" or not raw.isdigit():
                continue
            val = int(raw)
            if val > 0:
                return val // (1024 * 1024)
        except (OSError, ValueError):
            continue
    return None


def _auto_proxy_workers(max_workers: int = 128) -> int:
    """根据容器内存限制自动计算最优 Worker 数。

    公式：workers = (可用内存MB - 200) / 65 - 2，结果夹在 [1, max_workers]。
    若无法读取 cgroup 限制（非容器环境），退回默认值 4。
    """
    mem_mb = _detect_cgroup_memory_mb()
    if mem_mb is None:
        return 4
    workers = int((mem_mb - 200) / 65) - 2
    workers = max(1, min(workers, max_workers))
    logger.info("Auto-detected cgroup memory limit: %d MB -> PROXY_WORKERS=%d", mem_mb, workers)
    return workers


def _build_processes(port_plan: PortPlan) -> list[ManagedProc]:
    """构造 launcher 需要托管的 proxy、health 进程。"""
    env = _build_child_env(port_plan)
    python_bin = settings.PYTHON_BIN
    uvicorn_mod = settings.UVICORN_MODULE

    # 确定 proxy worker 数量：
    # 1. 优先使用 PROXY_WORKERS 环境变量（用户显式指定）
    # 2. 未指定时自动读取 cgroup 内存限制计算最优值
    # 3. 非容器环境退回默认值 4
    # 上限 128：每个 worker 约 65MB RAM。
    max_proxy_workers = 128
    env_val = os.getenv("PROXY_WORKERS")
    if env_val is not None:
        proxy_workers = min(int(env_val), max_proxy_workers)
    else:
        proxy_workers = _auto_proxy_workers(max_proxy_workers)
    env["PROXY_WORKERS"] = str(proxy_workers)

    return [
        _build_proxy_proc(port_plan, env, python_bin, uvicorn_mod, proxy_workers),
        _build_health_proc(port_plan, env, python_bin, uvicorn_mod),
    ]


def _write_start_command(script_text: str) -> str:
    """将 engine 启动脚本写入共享卷（原子写入 + 宽松权限）。"""
    shared_dir = settings.SHARED_VOLUME_PATH
    os.makedirs(shared_dir, exist_ok=True)
    path = os.path.join(shared_dir, settings.START_COMMAND_FILENAME)
    # 权限 0o644：engine 容器即使非 root 也能读取。
    # atomic=True：先写临时文件再 rename，防止 engine 读到截断的脚本。
    ok = safe_write_file(
        path, script_text, is_json=False,
        options=WriteOptions(
            modes=stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
            atomic=True,
        ),
    )
    if not ok:
        raise RuntimeError(f"failed to write start command: {path}")
    logger.info("start command written: %s", path)
    return path


# ---------------------------------------------------------------------------
# 分布式模式辅助函数
# ---------------------------------------------------------------------------


def _resolve_ip_or_self(ip: str) -> str:
    """尝试 DNS 解析 ip；解析失败时返回原始字符串。"""
    try:
        return socket.gethostbyname(ip)
    except socket.error as exc:
        logger.debug("DNS resolution failed for '%s': %s", ip, exc)
        return ip


def _ips_match(local_ip: str, master_ip: str) -> bool:
    """判断 local_ip 与 master_ip 是否指向同一节点（支持 DNS 解析）。

    优先做原始字符串比较（快速路径），匹配失败时再解析 DNS（慢速路径）。
    """
    if local_ip == master_ip:
        return True
    return _resolve_ip_or_self(local_ip) == _resolve_ip_or_self(master_ip)


def _determine_role() -> str:
    """判断当前 Pod 在分布式集群中的角色。

    通过 DISTRIBUTED 环境变量判断是否为分布式模式:
      - 非分布式 → "standalone"（沿用原有单机流程）
      - 分布式 + PD external-lb（PD_ROLE + DP_SIZE≥1）→ "standalone"（对等 pod，见设计 §13.7）
      - 分布式且 RANK_IP == MASTER_IP → "master"（含 DNS 解析）
      - 分布式且 RANK_IP != MASTER_IP → "worker"

    RANK_IP 由上层（MaaS）传入，每个 Pod 唯一，不依赖 NODE_RANK 环境变量。

    Returns:
        "standalone" | "master" | "worker"
    """
    distributed = os.getenv("DISTRIBUTED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not distributed:
        return "standalone"

    # PD external-lb（PD_ROLE + DP_SIZE≥1）下，每个 pod 是对等的独立单元：自带 proxy/health/
    # monitor + 本地引擎 fork，跨 pod 靠 vLLM DP rendezvous（--data-parallel-address）组域。
    # Ray master/worker 编排（head 起 API、worker 仅 headless、health 端口偏移、BACKEND_URL 指
    # master、worker 不监控本地引擎）是为「单引擎跨节点」设计的，套在 external-lb 上会范式错位
    # （见设计文档 §13.7）。故 external-lb 命中时强制按 standalone 处理，绕开 master/worker。
    # 门控信号 = 引擎脚本分发同源的 _get_pd_external_lb_params()（非空 ⇔ PD_ROLE 且 DP_SIZE≥1），
    # 含 1P1D（DP_SIZE=1 默认），两层判定一致。非 PD 的 Ray 分布式无 PD_ROLE → 此处不命中，行为字节级不变。
    try:
        from core.config_loader import _get_pd_external_lb_params
        if _get_pd_external_lb_params() is not None:
            logger.info(
                "[role] PD external-lb active (PD_ROLE + DP_SIZE≥1) → standalone peer pod; "
                "skipping Ray master/worker orchestration (see design §13.7)"
            )
            return "standalone"
    except Exception as exc:  # noqa: BLE001 — 守卫：探测失败绝不可阻断角色判定，回退原逻辑
        logger.debug("[role] PD external-lb probe skipped (%s); fall back to master/worker", exc)

    master_ip = get_master_ip()
    local_ip = get_local_ip()  # 来自 RANK_IP 环境变量

    if not master_ip:
        logger.warning("DISTRIBUTED=true but MASTER_IP not set, falling back to standalone")
        return "standalone"

    if _ips_match(local_ip, master_ip):
        local_r = _resolve_ip_or_self(local_ip)
        master_r = _resolve_ip_or_self(master_ip)
        logger.info(
            "Role determined: MASTER (RANK_IP=%s→%s, MASTER_IP=%s→%s)",
            local_ip, local_r, master_ip, master_r,
        )
        return "master"

    local_r = _resolve_ip_or_self(local_ip)
    master_r = _resolve_ip_or_self(master_ip)
    logger.info(
        "Role determined: WORKER (RANK_IP=%s→%s, MASTER_IP=%s→%s)",
        local_ip, local_r, master_ip, master_r,
    )
    return "worker"


def _get_expected_nodes() -> list[str]:
    """从 NODE_IPS 环境变量获取集群全部节点 IP 列表。

    若 NODE_IPS 未设置，回退到仅包含本机 IP 的单元素列表。
    """
    node_ips_str = get_node_ips()
    if not node_ips_str:
        return [get_local_ip()]
    return [ip.strip() for ip in node_ips_str.split(",") if ip.strip()]


def _override_distributed_args(
    launch_args: LaunchArgs,
    *,
    distributed: bool,
    nnodes: int,
    node_rank: int,
    head_node_addr: str,
    node_ips: str | None = None,
    nodes: str | None = None,
    master_ip: str | None = None,
    ray_head_ip: str | None = None,
) -> LaunchArgs:
    """创建 LaunchArgs 副本，覆盖分布式相关字段。

    由于 LaunchArgs 是 frozen dataclass，使用 dataclasses.replace 创建变体。
    """
    return dataclasses.replace(
        launch_args,
        distributed=distributed,
        nnodes=nnodes,
        node_rank=node_rank,
        head_node_addr=head_node_addr,
        node_ips=node_ips if node_ips is not None else launch_args.node_ips,
        nodes=nodes if nodes is not None else launch_args.nodes,
        master_ip=master_ip if master_ip is not None else launch_args.master_ip,
        ray_head_ip=ray_head_ip if ray_head_ip is not None else launch_args.ray_head_ip,
    )


def _load_distributed_config() -> dict:
    """加载 config/distributed_config.json 配置。"""
    config_path = Path(__file__).parent / "config" / "defaults" / "distributed_config.json"
    with open(config_path) as f:
        return json.load(f)


def _resolve_host_to_ip(host: str) -> str:
    """将 DNS 名称解析为 IP 地址；已是 IP 或解析失败时原样返回。

    用于处理 NODE_IPS 中可能包含的 DNS 名称（如 'infer-1.infer-hl'），
    使其能与 Worker 注册时上报的 Pod IP 进行正确比较。
    """
    try:
        return socket.gethostbyname(host)
    except socket.error as exc:
        logger.debug("DNS resolution failed for host '%s': %s", host, exc)
        return host


def _wait_for_worker_registration(
    master_url: str,
    worker_ips: list[str],
    max_wait_sec: int = 300,
    poll_interval: int = 5,
    max_retries: int = 2,
) -> bool:
    """轮询 Master API 直到所有 worker 完成注册，超时或重试耗尽返回 False。"""
    resolved_workers = {_resolve_host_to_ip(ip) for ip in worker_ips}

    def _check_once() -> bool:
        try:
            resp = requests.get(f"{master_url}/api/nodes", timeout=10)
            resp.raise_for_status()
            registered = {n["ip"] for n in resp.json().get("nodes", [])}
            return resolved_workers.issubset(registered)
        except Exception as exc:
            logger.debug("Waiting for workers to register: %s", exc)
            return False

    for attempt in range(max_retries + 1):
        start_time = time.time()
        while time.time() - start_time < max_wait_sec:
            if _check_once():
                logger.info(
                    "All %d worker nodes registered with master (resolved: %s)",
                    len(worker_ips), resolved_workers,
                )
                return True
            time.sleep(poll_interval)
        if attempt < max_retries:
            logger.warning(
                "Timed out (%ds) waiting for worker registration (attempt %d/%d). "
                "Expected: %s. Retrying...",
                max_wait_sec, attempt + 1, max_retries + 1, worker_ips,
            )
        else:
            logger.error(
                "Timed out (%ds) waiting for worker registration after %d retries. "
                "Expected: %s",
                max_wait_sec * (attempt + 1), attempt, worker_ips,
            )
    return False


def _wait_for_worker_api_ready(
    worker_ip: str,
    worker_port: int,
    max_attempts: int = 10,
    interval: float = 2.0,
) -> bool:
    """探测 Worker API 是否已开始监听，最多尝试 *max_attempts* 次。"""
    url = f"http://{worker_ip}:{worker_port}/api/node_info"
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            logger.debug("Worker %s:%d API ready (attempt %d)", worker_ip, worker_port, attempt)
            return True
        except Exception as exc:
            if attempt < max_attempts:
                logger.debug(
                    "Worker %s:%d API not ready (attempt %d/%d): %s",
                    worker_ip, worker_port, attempt, max_attempts, exc,
                )
                time.sleep(interval)
    logger.warning("Worker %s:%d API not ready after %d attempts", worker_ip, worker_port, max_attempts)
    return False


def _try_dispatch_to_worker(
    worker_ip: str,
    worker_port: int,
    rank: int,
    params: dict,
    options: DispatchOptions,
) -> bool:
    """向单个 Worker 节点发送引擎启动请求，带有限重试。

    Returns:
        True if dispatched successfully, False otherwise.
    """
    for attempt in range(1, options.max_retries + 1):
        try:
            resp = requests.post(
                f"http://{worker_ip}:{worker_port}/api/start_engine",
                json={"engine": params.get("engine", "vllm"), "params": params},
                timeout=30,
            )
            resp.raise_for_status()
            logger.info(
                "Distributed start command to worker rank %d (%s): %s",
                rank, worker_ip, resp.json(),
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to distribute to worker rank %d (%s) (attempt %d/%d): %s",
                rank, worker_ip, attempt, options.max_retries, exc,
            )
            if attempt < options.max_retries:
                time.sleep(options.retry_interval)
    return False


def _dispatch_engine_to_workers(
    worker_ips: list[str],
    worker_port: int,
    launch_args: LaunchArgs,
    topology: DistTopology,
    options: DispatchOptions | None = None,
) -> int:
    """向每个 Worker 节点发送引擎启动指令，带 ready 探测和有限重试。

    Returns:
        int: 分发失败的 Worker 数量（0 = 全部成功）
    """
    if options is None:
        options = DispatchOptions()
    topology_csv = ",".join(topology.node_ips)
    base_params = launch_args.to_namespace().__dict__
    failed_count = 0
    for rank, worker_ip in enumerate(worker_ips, start=1):
        params = {
            **base_params,
            "distributed": True,
            "nnodes": topology.nnodes,
            "node_rank": rank,
            "head_node_addr": topology.head_addr,
            "node_ips": topology_csv,
            "nodes": topology_csv,
            "master_ip": launch_args.master_ip or topology.head_addr,
            "ray_head_ip": launch_args.ray_head_ip or launch_args.master_ip or topology.head_addr,
        }

        # 先探测 Worker API 是否已在监听
        if not _wait_for_worker_api_ready(worker_ip, worker_port):
            logger.error(
                "Skipping worker rank %d (%s): API never became ready",
                rank, worker_ip,
            )
            failed_count += 1
            continue

        # 带有限重试的分发
        if not _try_dispatch_to_worker(worker_ip, worker_port, rank, params, options):
            logger.error(
                "Permanently failed to distribute to worker rank %d (%s) after %d retries",
                rank, worker_ip, options.max_retries,
            )
            failed_count += 1

    return failed_count


def _wait_and_distribute_to_workers(
    node_ips: list[str],
    launch_args: LaunchArgs,
    master_url: str,
) -> None:
    """后台线程：等待所有 Worker 注册后向其分发引擎启动指令。

    流程:
      1. 轮询 Master /api/nodes 接口，等待所有 worker 节点就绪（最多 5 分钟）
      2. 注册完成后，逐个向 worker 的 /api/start_engine 发送启动请求
      3. 为每个 worker 注入正确的 nnodes / node_rank / head_node_addr

    Args:
        node_ips:    全部节点 IP 列表（index 0 = master/rank0）
        launch_args: 标准化启动参数
        master_url:  Master API 地址（用于查询节点注册情况）
    """
    dist_config = _load_distributed_config()
    worker_port = int(os.getenv("WORKER_PORT", str(dist_config["workers"]["port"])))
    worker_ips = node_ips[1:]  # 排除 rank 0（Master 自身已处理）
    if not worker_ips:
        logger.info("No worker nodes to distribute to (single-node distributed)")
        return

    # 检测重复 IP — 同一 worker 不应被分发两次
    unique_ips = list(dict.fromkeys(worker_ips))  # 保序去重
    if len(unique_ips) < len(worker_ips):
        logger.warning(
            "Duplicate worker IPs detected in node_ips: %s → deduplicated to %s",
            worker_ips, unique_ips,
        )
        worker_ips = unique_ips

    if not _wait_for_worker_registration(master_url, worker_ips):
        return
    head_addr = launch_args.head_node_addr or launch_args.master_ip or node_ips[0]
    topo = DistTopology(nnodes=len(node_ips), head_addr=head_addr, node_ips=node_ips)
    failed = _dispatch_engine_to_workers(worker_ips, worker_port, launch_args, topo)
    if failed > 0:
        logger.critical(
            "DISTRIBUTED STARTUP INCOMPLETE: %d/%d worker(s) failed to receive start command. "
            "MindIE HCCL requires ALL nodes to be started — cluster may hang indefinitely.",
            failed, len(worker_ips),
        )


def _generate_rank0_script(
    launch_args: LaunchArgs,
    port_plan: PortPlan,
    node_ips: list[str],
    head_addr: str,
    master_ip: str,
) -> LaunchArgs:
    """生成 rank 0 脚本并写入共享卷，返回含分布式参数的 master_args。"""
    nnodes = len(node_ips)
    topology_csv = ",".join(node_ips)
    master_args = _override_distributed_args(
        launch_args,
        distributed=True,
        nnodes=nnodes,
        node_rank=0,
        head_node_addr=head_addr,
        node_ips=topology_csv,
        nodes=topology_csv,
        master_ip=master_ip,
        ray_head_ip=launch_args.ray_head_ip or master_ip,
    )
    launcher_plan = build_launcher_plan(master_args, port_plan)
    script_path = _write_start_command(launcher_plan.command)
    logger.info("master start command written to %s", script_path)
    return master_args


def _start_master_api_thread(master_port: int) -> threading.Thread:
    """后台启动 Master FastAPI 协调服务线程（带异常保护和自动重启）。"""

    def _run_master_api():
        max_restarts = 5
        restart_count = 0
        while restart_count < max_restarts:
            try:
                import uvicorn
                from distributed.master import app as master_app
                from distributed.master import MonitorService, TaskScheduler
                import distributed.master as master_mod

                if restart_count == 0:
                    master_mod.monitor_service = MonitorService()
                    master_mod.task_scheduler = TaskScheduler(master_mod.monitor_service)
                    master_mod.monitor_service.start()
                    master_mod.task_scheduler.start()
                logger.info(
                    "Master API uvicorn starting on port %d (attempt %d)",
                    master_port,
                    restart_count + 1,
                )
                uvicorn.run(master_app, host="0.0.0.0", port=master_port,
                            log_level="error")
                break  # normal exit
            except Exception as _:
                restart_count += 1
                logger.exception(
                    "Master API thread crashed (attempt %d/%d), "
                    "restarting in 3s...",
                    restart_count,
                    max_restarts,
                )
                time.sleep(3)
        if restart_count >= max_restarts:
            logger.error(
                "Master API exceeded max restarts (%d), giving up",
                max_restarts,
            )

    master_thread = threading.Thread(target=_run_master_api, daemon=True)
    master_thread.start()
    logger.info("Master API starting on port %d", master_port)
    # 等待 Master API 就绪，而非固定 sleep（最多 10 秒，0.5 秒间隔轮询）
    for _probe in range(20):
        time.sleep(0.5)
        try:
            resp = requests.get(f"http://127.0.0.1:{master_port}/api/nodes", timeout=2)
            if resp.status_code < 500:
                logger.info("Master API ready on port %d", master_port)
                break
        except Exception:  # noqa: BLE001
            logger.debug("Master API probe attempt %d failed", _probe + 1)
    else:
        logger.warning("Master API not confirmed ready after 10s, proceeding anyway")
    return master_thread


def _check_and_restart_all(processes: list[ManagedProc]) -> bool:
    """检查所有子进程状态，按需重启，返回是否全部已退出。"""
    all_dead = True
    for proc in processes:
        _restart_if_needed(proc)
        if proc.proc and proc.proc.poll() is None:
            all_dead = False
    return all_dead


def _daemon_loop(
    processes: list[ManagedProc],
    label: str,
) -> int:
    """信号处理 + 子进程守护循环（master/worker/standalone 共用）。"""
    stop_event = Event()

    def _on_signal(signum: int, _frame: object) -> None:
        logger.info("received signal: %s", signum)
        stop_event.set()

    # signal.signal() 只能在主线程中调用；加防御性检查避免未来重构时引入问题
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    else:
        logger.warning(
            "_daemon_loop called from non-main thread, "
            "signal handlers not registered"
        )

    exit_code = 0
    try:
        while not stop_event.is_set():
            if _check_and_restart_all(processes) and processes:
                logger.error("%s: all managed processes have exited", label)
                exit_code = 1
                break
            time.sleep(settings.PROCESS_POLL_SEC)
    finally:
        for proc in processes:
            _stop(proc)
        logger.info("%s shutdown complete", label)
    return exit_code


def _run_master_mode(
    launch_args: LaunchArgs,
    port_plan: PortPlan,
) -> int:
    """Master 模式主流程。

    1. 生成 rank 0 引擎启动脚本并写入共享卷 → engine 容器自动执行
    2. 后台启动 Master FastAPI 协调服务（端口来自 distributed_config.json）
    3. 启动 proxy + health 子服务
    4. 后台等待 Worker 注册完成后分发启动指令到各 Worker
    5. 进入守护循环，监控 proxy/health 子进程状态
    """
    local_ip = get_local_ip()
    node_ips = [ip.strip() for ip in (launch_args.node_ips or launch_args.nodes).split(",") if ip.strip()]
    if not node_ips:
        node_ips = _get_expected_nodes()
    master_ip = launch_args.master_ip or get_master_ip() or local_ip
    head_addr = launch_args.head_node_addr or master_ip or local_ip

    # ---- 1. 生成 rank 0 脚本写入共享卷 ----
    master_args = _generate_rank0_script(launch_args, port_plan, node_ips, head_addr, master_ip)

    # ---- 2. 后台启动 Master FastAPI ----
    dist_config = _load_distributed_config()
    # 使用 COORDINATOR_PORT 避免与 HCCL 的 MASTER_PORT 语义冲突。
    # MASTER_PORT 在 MindIE engine 中用于 HCCL 集合通信（默认 27070），
    # 而此处是 wings 分布式协调 API 端口（默认 16000），二者不同。
    master_port = int(
        os.getenv("COORDINATOR_PORT",
                  os.getenv("MASTER_PORT", str(dist_config["master"]["port"])))
    )
    master_url = f"http://127.0.0.1:{master_port}"
    _master_api_thread = _start_master_api_thread(master_port)
    logger.info(
        "Master API thread started: name=%s alive=%s",
        _master_api_thread.name,
        _master_api_thread.is_alive(),
    )

    # ---- 3. 启动 proxy + health 子服务 ----
    processes = _build_processes(port_plan)
    for proc in processes:
        _start(proc)

    # ---- 4. 后台等待 Worker 注册并分发 ----
    dist_thread = threading.Thread(
        target=_wait_and_distribute_to_workers,
        args=(node_ips, master_args, master_url),
        daemon=True,
    )
    dist_thread.start()

    # ---- 5. 守护循环 ----
    logger.info(
        "Master mode running: master_api=%d backend=%d proxy=%d health=%d",
        master_port,
        port_plan.backend_port,
        port_plan.proxy_port,
        port_plan.health_port,
    )
    return _daemon_loop(processes, "Master mode")


def _run_worker_mode(
    launch_args: LaunchArgs,
    port_plan: PortPlan,
) -> int:
    """Worker 模式主流程。

    1. 后台启动 Worker FastAPI 服务（自动向 Master 注册 + 心跳守护）
    2. 仅启动 health 子服务（非 rank0 不暴露 proxy）
    3. 进入守护循环
    4. 引擎启动脚本由 Master 分发后通过 Worker API 写入共享卷

    注意:
      Worker 启动时不写 start_command.sh。脚本在 Master 完成分发后由
      Worker 的 /api/start_engine 端点生成并写入共享卷。
    """
    master_ip = get_master_ip()

    # ---- 1. 后台启动 Worker FastAPI ----
    def _run_worker_api():
        from distributed.worker import WorkerConfig, start_worker

        worker_cfg = WorkerConfig(master_ip=master_ip)
        start_worker(worker_cfg)

    worker_thread = threading.Thread(target=_run_worker_api, daemon=True)
    worker_thread.start()
    logger.info("Worker API starting, registering with master at %s", master_ip)

    # 等待 Worker FastAPI 就绪（最多 10 秒，0.5 秒间隔轮询）
    dist_config = _load_distributed_config()
    worker_port = int(os.getenv("WORKER_PORT", str(dist_config["workers"]["port"])))
    for _probe in range(20):
        time.sleep(0.5)
        try:
            resp = requests.get(
                f"http://127.0.0.1:{worker_port}/api/node_info", timeout=2,
            )
            if resp.status_code < 500:
                logger.info("Worker API ready on port %d", worker_port)
                break
        except Exception:  # noqa: BLE001
            logger.debug("Worker API probe attempt %d failed", _probe + 1)
    else:
        logger.warning(
            "Worker API not confirmed ready after 10s, proceeding anyway",
        )

    # ---- 2. 启动 health 子服务（使用偏移端口避免 hostNetwork 冲突） ----
    # Worker 的 health 端口在基准端口上偏移 +1（如 19000 → 19001），
    # 避免 hostNetwork 模式下与同一宿主机上 Master Pod 的 19000 端口冲突。
    # K8s StatefulSet 中 Worker Pod 的 readinessProbe/livenessProbe 需对应配置。
    #
    # Worker 本地没有 vLLM API server（只运行 Ray worker），因此将
    # BACKEND_URL 指向 Master 的 backend 端口。分布式场景下，Master backend
    # 健康即表示包含此 Worker 的 Ray 集群正常工作。
    worker_health_port = port_plan.health_port + 1
    master_backend_url = f"http://{master_ip}:{port_plan.backend_port}"
    worker_port_plan = PortPlan(
        enable_proxy=port_plan.enable_proxy,
        backend_port=port_plan.backend_port,
        proxy_port=port_plan.proxy_port,
        health_port=worker_health_port,
    )
    # Worker 只启动 health 服务，不启动 proxy
    processes = [p for p in _build_processes(worker_port_plan) if p.name == "health"]
    for proc in processes:
        proc.env["BACKEND_URL"] = master_backend_url
        _start(proc)
    logger.info(
        "Worker health probing master backend at %s",
        master_backend_url,
    )

    logger.info(
        "Worker mode running: health=%d "
        "(waiting for master to dispatch engine start)",
        worker_health_port,
    )
    return _daemon_loop(processes, "Worker mode")


# ---------------------------------------------------------------------------
# 主入口 — 辅助函数
# ---------------------------------------------------------------------------


def _stop_processes(processes: list) -> None:
    """逐个停止托管进程，忽略单个进程的停止错误。"""
    for proc in processes:
        try:
            _stop(proc)
        except Exception as stop_err:
            logger.debug("Failed to stop process during cleanup: %s", stop_err)


def _check_all_running(processes: list, attempt_label: str) -> bool:
    """检查所有进程在短暂等待后是否仍在运行，返回 True 表示全部正常。"""
    all_ok = True
    for proc in processes:
        if proc.proc and proc.proc.poll() is not None:
            code = proc.proc.returncode
            logger.error(
                "Process '%s' exited immediately with code %s %s",
                proc.name, code, attempt_label,
            )
            all_ok = False
    return all_ok


def _launch_attempt(launch_args: LaunchArgs, port_plan: PortPlan, attempt_label: str) -> list:
    """执行一次启动尝试，返回已启动的进程列表；失败时抛出异常。"""
    logger.info("Starting launcher plan generation %s ...", attempt_label)
    launcher_plan = build_launcher_plan(launch_args, port_plan)
    logger.info("Launcher plan generated successfully %s", attempt_label)

    logger.info("Writing start command to shared volume %s ...", attempt_label)
    start_cmd_path = _write_start_command(launcher_plan.command)
    logger.info("Start command written to: %s %s", start_cmd_path, attempt_label)

    processes = _build_processes(port_plan)
    # 分布式场景下只有 rank0 暴露 proxy，其余 rank 保留 health 即可。
    if getattr(launch_args, "node_rank", 0) > 0:
        processes = [p for p in processes if p.name != "proxy"]

    logger.info("Starting %d managed processes %s ...", len(processes), attempt_label)
    for proc in processes:
        _start(proc)

    time.sleep(2)  # 短暂等待，检查进程是否立即崩溃
    if not _check_all_running(processes, attempt_label):
        raise RuntimeError(
            f"One or more processes crashed immediately after startup {attempt_label}"
        )

    logger.info(
        "launcher running: backend=%s proxy=%s health=%s",
        port_plan.backend_port,
        port_plan.proxy_port,
        port_plan.health_port,
    )
    return processes


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def _run_standalone_mode(launch_args: LaunchArgs, port_plan: PortPlan) -> int:
    """Standalone 模式主循环（含首次启动失败重试）。"""
    max_launch_retries = 1  # 最多重试 1 次（共尝试 2 次）
    retry_delay_sec = 5     # 重试前等待秒数
    processes: list = []

    has_spec_decode = getattr(launch_args, "enable_speculative_decode", False)
    if has_spec_decode:
        logger.info("[AdvFeature] Standalone mode startup, speculative decode enabled")
        logger.info("[AdvFeature] Max launch retries: %d, retry delay: %ds",
                     max_launch_retries, retry_delay_sec)

    for attempt in range(max_launch_retries + 1):
        attempt_label = f"(attempt {attempt + 1}/{max_launch_retries + 1})"
        try:
            logger.info("[Startup] Service starting %s ...", attempt_label)
            processes = _launch_attempt(launch_args, port_plan, attempt_label)
            logger.info("[Startup] Service started successfully %s", attempt_label)
            break  # 启动成功，退出重试循环
        except Exception as e:
            logger.error("[Startup] Service startup failed %s: %s", attempt_label, e, exc_info=True)
            _stop_processes(processes)
            processes = []
            if attempt < max_launch_retries:
                logger.warning(
                    "[Startup] Retrying service startup in %ds (remaining retries: %d) ...",
                    retry_delay_sec, max_launch_retries - attempt,
                )
                time.sleep(retry_delay_sec)
            else:
                logger.critical(
                    "[Startup] All %d startup attempts exhausted. Service startup failed.",
                    max_launch_retries + 1,
                )
                return 1

    logger.info(
        "launcher running: backend=%s proxy=%s health=%s",
        port_plan.backend_port,
        port_plan.proxy_port,
        port_plan.health_port,
    )
    return _daemon_loop(processes, "launcher")


def run(argv: Sequence[str] | None = None) -> int:
    """launcher 主流程。

    根据 _determine_role() 判断角色:
      - standalone: 沿用原有单机流程（build_launcher_plan → 写脚本 → 守护 proxy/health）
      - master:     Master 协调模式（写 rank0 脚本 + Master API + 分发 Worker）
      - worker:     Worker 等待模式（Worker API + 仅 health，等 Master 分发脚本）
    """
    launch_args = parse_launch_args(list(argv) if argv is not None else None)
    port_plan = derive_port_plan(
        port=launch_args.port,
        enable_reason_proxy=settings.ENABLE_REASON_PROXY,
        health_port=settings.HEALTH_PORT,
    )

    # 当前版本必须启用 proxy。
    if not port_plan.enable_proxy:
        logger.error("ENABLE_REASON_PROXY=false is not supported in v4 MVP")
        return 2

    # ---- 分布式角色分支 ----
    role = _determine_role()
    logger.info("Launcher role: %s", role)

    if role == "master":
        return _run_master_mode(launch_args, port_plan)
    if role == "worker":
        return _run_worker_mode(launch_args, port_plan)

    # ---- standalone 模式（支持首次启动失败重试） ----
    return _run_standalone_mode(launch_args, port_plan)


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
