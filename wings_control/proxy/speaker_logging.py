# -*- coding: utf-8 -*-
"""
multi-worker 日志级别控制。

仅让部分 worker 输出 INFO 级别日志，避免多 worker 场景下日志重复。

使用方法:
1) 在 FastAPI/uvicorn 启动时调用 configure_worker_logging()
2) 日志级别会自动根据 worker 索引和配置决定

配置环境变量:
- LOG_INFO_SPEAKERS       : INFO 级别日志的 speaker 数量 (默认 1)
- LOG_WORKER_COUNT        : 总 worker 数，如未设置则读取 --workers 或 WEB_CONCURRENCY/UVICORN_WORKERS
- KEEP_ACCESS_LOG         : 是否保留 uvicorn.access，0/false 关闭
- LOG_SPEAKER_INDEXES     : 明确指定哪些 worker 为 speaker，如 "0,2"，需配合 WORKER_INDEX 使用
- WORKER_INDEX            : 当前 worker 的索引，通常由 uvicorn 设置

若不指定 LOG_SPEAKER_INDEXES 或 WORKER_INDEX，则用 pid-hash % LOG_WORKER_COUNT 决定。
"""
import logging
import os
import re
import sys
import zlib
from typing import List, Optional


# =========================
#
# =========================
class LogConstants:
    """日志格式化与 speaker 决策相关的常量集合。

    集中管理环境变量前缀、speaker 决策键名、默认 worker 数量
    以及需要被归一化的 logger 名称列表，方便外部 patch 替换。
    """

    #
    ENV_PREFIX = "LOG_"

    # Environment variable key for caching the speaker decision
    SPEAKER_DECISION_ENV = "_SPEAKER_DECISION"

    # Default expected worker count when actual count is unknown
    DEFAULT_WORKER_COUNT = 8

    # Logger names to normalize (NOTSET level, propagate to root)
    NORMALIZE_LOGGERS = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.server",
        "uvicorn.lifespan",
        "httpx",
        "httpcore",
    ]


#
DEFAULT_WORKER_COUNT = LogConstants.DEFAULT_WORKER_COUNT
NORMALIZE_LOGGERS = LogConstants.NORMALIZE_LOGGERS
SPEAKER_DECISION_ENV = LogConstants.SPEAKER_DECISION_ENV


#
_CONFIGURED_ONCE = False

# Module-level logger for this module
_lg = logging.getLogger(__name__)


def _env_bool(key: str, default: bool = False) -> bool:
    """读取环境变量并解析为布尔值，不存在或无法识别时返回默认值。

    Args:
        key: 环境变量名称。
        default: 环境变量未设置或值无法识别时的默认返回值。

    Returns:
        解析后的布尔值。识别 '1/true/yes/y/on' 为 True，
        '0/false/no/n/off' 为 False，其余返回 default。
    """
    v = os.getenv(key, "")
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_int(key: str, default: int) -> int:
    """读取环境变量并解析为整数，解析失败时返回默认值。

    Args:
        key: 环境变量名称。
        default: 环境变量未设置或无法转换为整数时的默认返回值。

    Returns:
        解析后的整数值，解析异常时返回 default。
    """
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception as _:
        return default


def _parse_csv_ints(s: str) -> List[int]:
    """将逗号分隔的数字字符串解析为整数列表。

    非整数项会被忽略并记录警告日志。
    示例: '0,2,5' -> [0, 2, 5]

    Args:
        s: 逗号分隔的数字字符串，如 '0,2,5'。

    Returns:
        解析成功的整数列表，无效项被跳过。
    """
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            _lg.warning("ignore non-int csv item: %r", part)
    return out


def _discover_worker_count() -> int:
    """自动发现当前部署的 worker 总数。

    按以下优先级依次尝试:
    1) 环境变量 LOG_WORKER_COUNT（手动指定，优先级最高）
    2) WEB_CONCURRENCY / UVICORN_WORKERS（--workers 参数设置）
    3) 均未配置时返回 0，由调用方决定回退策略

    Returns:
        检测到的 worker 数量；未检测到时返回 0。
    """
    # 1)
    n = _env_int("LOG_WORKER_COUNT", 0)
    if n > 0:
        return n

    # 2)
    for k in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "PROXY_WORKERS"):
        v = os.getenv(k, "").strip()
        if not v:
            continue
        if not v.isdecimal():  # 避免 try/except；只接受十进制数字
            _lg.debug("Env %s=%r is not a positive integer string", k, v)
            continue
        n = int(v)
        if n > 0:
            return n
        _lg.debug("Env %s=%r parsed <= 0, ignored", k, v)

    # 3)
    return 0


def _is_speaker_by_index(allowed_indexes: List[int], worker_index: Optional[int]) -> Optional[bool]:
    """根据显式配置的 LOG_SPEAKER_INDEXES 和 WORKER_INDEX 判断当前 worker 是否为 speaker。

    当 WORKER_INDEX 未设置时无法判断，返回 None 以便调用方回退到 PID-hash 策略。

    Args:
        allowed_indexes: 允许作为 speaker 的 worker 索引列表。
        worker_index: 当前 worker 的索引；为 None 表示未设置。

    Returns:
        True/False 表示是否为 speaker；None 表示无法判断（缺少 WORKER_INDEX）。
    """
    if worker_index is None:
        return None
    return worker_index in allowed_indexes


def _is_speaker_by_pid_hash(pid: int, speakers_quota: int, worker_count: int) -> bool:
    """通过 PID 的 CRC32 哈希值对 worker_count 取模，判断当前进程是否为 speaker。

    哈希结果落在 [0, speakers_quota) 区间内则为 speaker。

    Args:
        pid: 当前进程的 PID。
        speakers_quota: 允许输出 INFO 级别日志的 worker 数量，最小为 1。
        worker_count: worker 总数；若 <=0 则回退为 max(8, speakers_quota)，
                      确保在 worker 数未知时仍能合理分配。

    Returns:
        True 表示当前进程为 speaker，应输出 INFO 级别日志；否则 False。
    """
    speakers_quota = max(1, speakers_quota)
    if worker_count <= 0:
        # worker 数未知且未设置多 worker 环境变量 → 极可能是单 worker 模式（sidecar 默认）。
        # 单 worker 时直接设为 speaker，确保启动日志可见。
        return True
    h = (zlib.crc32(str(pid).encode("utf-8")) & 0xFFFFFFFF)
    return (h % worker_count) < speakers_quota


def _ensure_root_handler():
    """
     root logger  StreamHandlerstderr
     logger
    使用 log_config 中的统一格式。
    """
    from utils.log_config import LOG_FORMAT, LOG_DATE_FORMAT
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(stream=sys.stderr)
        fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        h.setFormatter(fmt)
        root.addHandler(h)


def _quiet_uvicorn_access(keep: bool):
    """
     uvicorn.access
    - keep=False handler HTTP
    - keep=True
    """
    lg = logging.getLogger("uvicorn.access")
    if not keep:
        lg.disabled = True
        lg.propagate = False
        try:
            lg.handlers.clear()
        except Exception as e:
            logging.debug("handlers.clear() unavailable: %s", e)
            lg.handlers[:] = []
    else:
        lg.disabled = False


def _normalize_children():
    """
     logger
    -  logger  NOTSET root
    -  logger  propagate  True root
     worker  INFO root
    """
    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.server",
        "uvicorn.lifespan",
        "httpx",
        "httpcore",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.NOTSET)   # logger
        lg.propagate = True           # 交给 root 统一控制


# ==== NEW: /health  ===================================
class _DropByRegex(logging.Filter):
    """
    Filter record.getMessage()

    -  message uvicorn  AccessLogger
    -
        * uvicorn.access   "GET /health HTTP/1.1"
        * httpx / httpcore   /health
    """
    def __init__(self, patterns: List[re.Pattern]):
        super().__init__()
        self._patterns = patterns

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception as _:
            return True
        for p in self._patterns:
            if p.search(msg):
                return False
        return True


def _install_health_log_filters() -> None:
    """
     /health
    - DROP_HEALTH_ACCESS=1 uvicorn.access  "GET /health"
      * HEALTH_ACCESS_DROP_REGEX \"GET\\s+/health\\b
    - DROP_OUTBOUND_HEALTH=1 httpx/httpcore  '/health'
      * OUTBOUND_HEALTH_DROP_REGEX /health
    """
    # 1)  inbound  /health
    if _env_bool("DROP_HEALTH_ACCESS", True):
        pat = os.getenv("HEALTH_ACCESS_DROP_REGEX", r"\"GET\s+/health\b")
        try:
            patterns = [re.compile(pat)]
            access_logger = logging.getLogger("uvicorn.access")
            access_logger.addFilter(_DropByRegex(patterns))
            _lg.debug("Installed uvicorn.access /health drop filter: %r", pat)
        except re.error as e:
            _lg.warning("Invalid HEALTH_ACCESS_DROP_REGEX=%r, skip filter. err=%s", pat, e)

    # 2)  outbound  /health httpx/httpcore  DEBUG
    if _env_bool("DROP_OUTBOUND_HEALTH", True):
        pat2 = os.getenv("OUTBOUND_HEALTH_DROP_REGEX", r"/health")
        try:
            patterns2 = [re.compile(pat2)]
            # Python logging.Filter 仅对直接创建日志的 logger 生效，
            # 不会在 propagate 链上对父 logger 的 filter 做检查。
            # httpx 库实际使用 httpx._client 等子 logger 记录请求日志，
            # 因此需同时在父 logger 和已知子 logger 上安装过滤器。
            for name in (
                "httpx", "httpx._client",
                "httpcore", "httpcore._async", "httpcore._sync",
            ):
                lg = logging.getLogger(name)
                lg.addFilter(_DropByRegex(patterns2))
            _lg.debug("Installed httpx/httpcore /health drop filter: %r", pat2)
        except re.error as e:
            _lg.warning("Invalid OUTBOUND_HEALTH_DROP_REGEX=%r, skip filter. err=%s", pat2, e)
# ==== NEW END =================================================================


def configure_worker_logging(force: bool = False) -> bool:
    """配置当前 worker 的日志级别，决定其是否为 speaker（输出 INFO 日志）。

    应在 FastAPI/uvicorn 启动时调用，默认仅执行一次（force=False）。
    执行流程:
    1) 读取环境变量获取 speakers_quota、worker_count、access 日志策略
    2) 通过 worker 索引匹配或 PID-hash 决定 speaker 身份
    3) 确保 root logger 有 handler，配置 uvicorn.access 日志
    4) 归一化子 logger，安装 /health 过滤器
    5) 设置 root 日志级别：speaker=INFO，非 speaker=WARNING
    6) 将决策结果写入 _SPEAKER_DECISION 环境变量供后续快速查询

    Args:
        force: 是否强制重新配置。为 False 时若已配置过则直接返回缓存结果。

    Returns:
        True 表示当前 worker 为 speaker（INFO 级别），False 表示非 speaker（WARNING 级别）。
    """
    global _CONFIGURED_ONCE
    if _CONFIGURED_ONCE and not force:
        #
        return bool(int(os.getenv("_SPEAKER_DECISION", "0")))

    # Max number of workers that may emit INFO-level logs
    speakers_quota = max(1, _env_int("LOG_INFO_SPEAKERS", 1))
    # Total worker count for speaker ratio calculation
    worker_count = _discover_worker_count()
    # Whether to preserve uvicorn.access HTTP request logs
    keep_access = _env_bool("KEEP_ACCESS_LOG", False)

    # Explicit speaker selection via index list and WORKER_INDEX env
    allowed_indexes_env = os.getenv("LOG_SPEAKER_INDEXES", "").strip()
    allowed_indexes = _parse_csv_ints(allowed_indexes_env) if allowed_indexes_env else []
    worker_index_env = os.getenv("WORKER_INDEX")
    worker_index = None
    if worker_index_env and worker_index_env.strip().lstrip("-").isdigit():
        worker_index = int(worker_index_env)

    pid = os.getpid()

    #
    # 1)  WORKER_INDEX
    # 2)  PID  worker_count
    decision_by_index = None
    if allowed_indexes:
        decision_by_index = _is_speaker_by_index(allowed_indexes, worker_index)
    if decision_by_index is None:
        is_speaker = _is_speaker_by_pid_hash(pid, speakers_quota, worker_count)
    else:
        is_speaker = bool(decision_by_index)

    # Ensure root logger has a StreamHandler attached
    _ensure_root_handler()

    # Configure or suppress uvicorn.access log output
    _quiet_uvicorn_access(keep_access)

    # Normalize child loggers to propagate logs to root handler
    _normalize_children()

    # ==== NEW:  /health wings  ====
    _install_health_log_filters()
    # ==== NEW END ====

    # root
    root = logging.getLogger()
    root.setLevel(logging.INFO if is_speaker else logging.WARNING)

    # Non-speaker workers: disable file handler to avoid duplicate file logs.
    # Only the speaker worker writes complete logs to the shared log file.
    if not is_speaker:
        try:
            from utils.log_config import disable_file_handler
            disable_file_handler()
        except ImportError:
            pass

    # Log speaker designation when this worker is the INFO speaker
    if is_speaker:
        logging.getLogger("log-center").info(
            "worker(pid=%s) is SPEAKER=1  (quota=%s, workers=%s, idx=%s, allowed=%s)",
            pid, speakers_quota, worker_count, worker_index, allowed_indexes or None
        )

    #
    os.environ["_SPEAKER_DECISION"] = "1" if is_speaker else "0"
    _CONFIGURED_ONCE = True
    return is_speaker


# =========================
#
# =========================
def _patch_worker_logging(constants: type = LogConstants) -> None:
    """根据 LogConstants 中的常量对模块内部函数进行猴子补丁。

    使 _normalize_children、_is_speaker_by_pid_hash、configure_worker_logging
    三个函数在运行时使用 constants 中定义的值，而非硬编码默认值。
    这允许外部通过修改 LogConstants 来自定义行为。

    补丁内容:
    - _normalize_children: 使用 constants.NORMALIZE_LOGGERS 替代硬编码列表
    - _is_speaker_by_pid_hash: 使用 constants.DEFAULT_WORKER_COUNT 替代默认值 8
    - configure_worker_logging: 同步 constants.SPEAKER_DECISION_ENV 与内部键名

    Args:
        constants: 包含日志常量的类，默认为 LogConstants。
    """
    mod = sys.modules[__name__]

    # 1)  _normalize_children
    def _normalize_children_patched():
        for name in list(constants.NORMALIZE_LOGGERS):
            lg = logging.getLogger(name)
            lg.setLevel(logging.NOTSET)
            lg.propagate = True

    setattr(mod, "_normalize_children", _normalize_children_patched)

    # 2)  _is_speaker_by_pid_hash
    _orig_pid_hash = getattr(mod, "_is_speaker_by_pid_hash")

    def _is_speaker_by_pid_hash_patched(pid: int, speakers_quota: int, worker_count: int) -> bool:
        speakers_quota_local = max(1, speakers_quota)
        if worker_count <= 0:
            # worker 数未知且未设置多 worker 环境变量 → 极可能是单 worker 模式（sidecar 默认）。
            # 单 worker 时直接设为 speaker，确保启动日志和 jlog 可见。
            return True
        h = (zlib.crc32(str(pid).encode("utf-8")) & 0xFFFFFFFF)
        return (h % worker_count) < speakers_quota_local

    setattr(mod, "_is_speaker_by_pid_hash", _is_speaker_by_pid_hash_patched)

    # 3)  configure_worker_logging
    _orig_configure = getattr(mod, "configure_worker_logging")

    def configure_worker_logging_wrapped(*args, **kwargs):
        alias_key = getattr(constants, "SPEAKER_DECISION_ENV", "_SPEAKER_DECISION")
        default_key = "_SPEAKER_DECISION"

        #
        if alias_key != default_key and alias_key in os.environ and default_key not in os.environ:
            os.environ[default_key] = os.environ[alias_key]

        try:
            result = _orig_configure(*args, **kwargs)
        finally:
            #
            if alias_key != default_key and default_key in os.environ:
                os.environ[alias_key] = os.environ[default_key]

        return result

    setattr(mod, "configure_worker_logging", configure_worker_logging_wrapped)


# 默认自动打补丁；如需关闭可设置环境变量 LOG_PATCH_DISABLE=1
if os.getenv("LOG_PATCH_DISABLE", "").strip().lower() not in ("1", "true", "yes", "y", "on"):
    _patch_worker_logging(LogConstants)
