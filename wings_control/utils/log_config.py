"""
统一日志格式配置模块。

提供集中化的日志格式常量和初始化函数，确保 wings-control 容器内
所有组件（launcher、proxy、health）使用一致的日志格式。

典型用法::

    from utils.log_config import setup_root_logging, LOGGER_LAUNCHER
    setup_root_logging()
    logger = logging.getLogger(LOGGER_LAUNCHER)
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time

# ---------------------------------------------------------------------------
# 统一格式常量
# ---------------------------------------------------------------------------

#: 默认日志格式 — [name] 标签唯一标识组件，kubectl --all-containers 再叠加容器名
LOG_COMPONENT = os.getenv("LOG_COMPONENT", "WINGS-CONTROL")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    f"%(asctime)s.%(msecs)03d [%(levelname)s] {LOG_COMPONENT} "
    "[%(name)s#%(funcName)s:%(lineno)d] %(message)s",
)

#: 日期格式
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# 标准 logger 名称常量 — 各模块引用保证命名一致
# ---------------------------------------------------------------------------

LOGGER_LAUNCHER = "wings-launcher"
LOGGER_PROXY = "wings-proxy"
LOGGER_HEALTH = "wings-health"

# ---------------------------------------------------------------------------
# 日志文件配置
# ---------------------------------------------------------------------------

#: 日志文件路径（环境变量覆盖）
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "/var/log/wings/wings_control.log")

#: 单文件最大体积 50MB
LOG_MAX_BYTES = 50 * 1024 * 1024

#: 保留 5 个备份
LOG_BACKUP_COUNT = 5


# ---------------------------------------------------------------------------
# DedupErrorFilter — suppress duplicate error messages within a time window
# ---------------------------------------------------------------------------

class DedupErrorFilter(logging.Filter):
    """Suppress duplicate ERROR/CRITICAL messages within a configurable window.

    Only affects records at ERROR level or above. Records below ERROR
    pass through unconditionally.

    Args:
        window_sec: Seconds within which identical messages are suppressed.
        max_cache:  Maximum number of cached message keys; oldest evicted when full.
    """

    def __init__(self, window_sec: float = 60.0, max_cache: int = 256):
        super().__init__()
        self._window = window_sec
        self._max_cache = max_cache
        self._seen: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.ERROR:
            return True
        try:
            key = record.getMessage()
        except Exception as _:
            return True
        now = time.monotonic()
        last = self._seen.get(key)
        if last is not None and (now - last) < self._window:
            return False  # duplicate within window — suppress
        # Evict oldest if cache full
        if len(self._seen) >= self._max_cache:
            oldest_key = min(self._seen, key=self._seen.get)  # type: ignore[arg-type]
            del self._seen[oldest_key]
        self._seen[key] = now
        return True


class WingsControlFormatter(logging.Formatter):
    """Format normal logs and launcher-relayed child logs without duplicate prefixes."""

    @staticmethod
    def _format_child_time(child_time: object) -> str:
        text = str(child_time).strip().replace(",", ".")
        if len(text) == len("YYYY-MM-DD HH:MM:SS"):
            return f"{text}.000"
        return text

    def format(self, record: logging.LogRecord) -> str:
        child_component = getattr(record, "wings_child_component", None)
        if child_component:
            child_time = getattr(record, "wings_child_time", None)
            if not child_time:
                child_time = self._format_record_time(record)
            else:
                child_time = self._format_child_time(child_time)
            child_source = getattr(record, "wings_child_source", None)
            if not child_source:
                child_source = f"{record.name}#{child_component}"
            return (
                f"{child_time} [{record.levelname}] {LOG_COMPONENT} "
                f"[{child_source}] {record.getMessage()}"
            )
        return super().format(record)

    def _format_record_time(self, record: logging.LogRecord) -> str:
        return f"{self.formatTime(record, self.datefmt)}.{int(record.msecs):03d}"


def _make_formatter() -> WingsControlFormatter:
    return WingsControlFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def _resolve_log_levels(
    level: str | None,
    stderr_level: str | None,
) -> tuple[int, int]:
    """解析日志级别字符串为 logging 整数常量。

    Returns:
        (log_level, stderr_log_level)
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, level.upper(), logging.INFO)

    if stderr_level is None:
        stderr_level = os.getenv("LOG_STDERR_LEVEL", level)
    stderr_log_level = getattr(logging, stderr_level.upper(), log_level)
    return log_level, stderr_log_level


def _configure_stderr_handlers(root: logging.Logger, stderr_log_level: int) -> None:
    """设置所有 stderr StreamHandler 的日志级别。"""
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(stderr_log_level)


def _install_dedup_filter(root: logging.Logger) -> None:
    """在所有 stderr StreamHandler 上安装重复消息抑制过滤器。"""
    dedup = DedupErrorFilter(
        window_sec=float(os.getenv("LOG_DEDUP_WINDOW_SEC", "60")),
    )
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.addFilter(dedup)


def _setup_file_handler(root: logging.Logger) -> None:
    """尝试添加 RotatingFileHandler — 写入共享日志卷。

    若目录不可写（如未挂载 log-volume），则跳过，仅保留 stderr 输出。
    若 root logger 已有指向同一文件的 RotatingFileHandler，则不重复添加。
    """
    already_has = any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, "baseFilename", None) == os.path.abspath(LOG_FILE_PATH)
        for h in root.handlers
    )
    if already_has:
        return

    log_dir = os.path.dirname(LOG_FILE_PATH)
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_make_formatter())
        root.addHandler(file_handler)
    except OSError:
        root.warning("Cannot write log file to %s, file logging disabled", LOG_FILE_PATH)


def setup_root_logging(
    level: str | None = None,
    stderr_level: str | None = None,
) -> None:
    """一次性配置 root logger，确保全局统一格式。

    使用 ``logging.basicConfig(force=True)`` 覆盖已有配置，
    保证无论导入顺序如何，格式始终一致。

    同时尝试添加 RotatingFileHandler 写入 LOG_FILE_PATH，
    若目录不可写则跳过（仅保留 stderr 输出）。

    Args:
        level: 日志级别字符串 (DEBUG/INFO/WARNING/ERROR)。
               未指定时读取 LOG_LEVEL 环境变量，默认 INFO。
        stderr_level: stderr handler 的独立级别。未指定时跟随 level。
                      Proxy 场景可设为 "ERROR" 仅在控制台打印错误。
    """
    log_level, stderr_log_level = _resolve_log_levels(level, stderr_level)

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stderr,
        force=True,
    )

    root = logging.getLogger()
    formatter = _make_formatter()
    for handler in root.handlers:
        handler.setFormatter(formatter)
    _configure_stderr_handlers(root, stderr_log_level)
    _install_dedup_filter(root)
    _setup_file_handler(root)


def disable_file_handler() -> None:
    """Remove RotatingFileHandler from root logger.

    Called by non-speaker workers in multi-worker mode to avoid
    duplicate file logging — only the speaker worker writes to file.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.handlers.RotatingFileHandler):
            root.removeHandler(h)
            try:
                h.close()
            except Exception as exc:  # pylint: disable=broad-except
                logging.getLogger(__name__).debug("Failed to close file handler: %s", exc)
