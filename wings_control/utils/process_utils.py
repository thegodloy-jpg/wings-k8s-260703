"""进程管理辅助方法，用于启动等待、PID 记录、输出流处理。

功能概述:
    本模块提供进程管理相关的工具函数:
    - wait_for_process_startup() : 等待子进程启动成功（按成功标志消息检测）
    - log_process_pid()          : 将 PID 写入文件以供外部监控
    - log_stream()               : 启动后台线程转发子进程 stdout/stderr 到日志

Sidecar 架构契约:
    - 不在启动检查中无限阻塞
    - 进程监控诊断信息清晰可读
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

import os
import subprocess
import threading
import time
import logging
import re
from typing import Union

from utils.file_utils import safe_write_file

logger = logging.getLogger(__name__)

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOG_DIR = os.path.join(root_dir, 'logs')

_LOG_LEVEL_PATTERNS = (
    (logging.CRITICAL, re.compile(r"\b(CRITICAL|FATAL)\b", re.IGNORECASE)),
    (
        logging.ERROR,
        re.compile(r"\b(ERROR|ERR|EXCEPTION|TRACEBACK|FAILED|FAILURE)\b|Error:", re.IGNORECASE),
    ),
    (logging.WARNING, re.compile(r"\b(WARNING|WARN)\b", re.IGNORECASE)),
    (logging.DEBUG, re.compile(r"\bDEBUG\b", re.IGNORECASE)),
    (logging.INFO, re.compile(r"\bINFO\b", re.IGNORECASE)),
)


def infer_log_level(line: str, default_level: int = logging.INFO) -> int:
    """Infer a logging level from a captured process output line."""
    for level, pattern in _LOG_LEVEL_PATTERNS:
        if pattern.search(line):
            return level
    return default_level


def wait_for_process_startup(
    process: subprocess.Popen,
    success_message: str,
    _logger: logging.Logger = None,
    timeout_sec: int = 300
) -> bool:
    """等待子进程启动并检测成功标志消息。

    启动后台线程分别读取 stdout 和 stderr，一旦检测到 success_message
    即认为服务启动成功；若进程退出且返回码非 0 则抛出异常。

    Args:
        process:         已启动的 Popen 对象
        success_message: 用于判断启动成功的字符串标志
        _logger:         可选自定义 logger，默认使用本模块 logger
        timeout_sec:     最大等待秒数（默认 300s），超时抛出 TimeoutError

    Returns:
        bool: True 表示检测到启动成功消息，False 表示进程正常退出但未检测到

    Raises:
        TimeoutError: 超过 timeout_sec 仍未检测到成功消息
        RuntimeError: 进程异常退出（returncode != 0）
    """
    if _logger is None:
        _logger = logger

    started = threading.Event()

    def _log_stream(stream, default_level):
        for line in stream:
            stripped = line.strip()
            if stripped:
                _logger.log(infer_log_level(stripped, default_level), stripped)
                if success_message in stripped:
                    started.set()

    #
    stdout_thread = threading.Thread(
        target=_log_stream,
        args=(process.stdout, logging.INFO),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=_log_stream,
        args=(process.stderr, logging.ERROR),
        daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    #
    deadline = time.monotonic() + timeout_sec
    while True:
        if started.is_set():
            _logger.info("Detected service startup success message: %s", success_message)
            return True

        if process.poll() is not None:  #
            if process.returncode != 0:
                raise RuntimeError(f"Process startup failed with return code: {process.returncode}")
            return False

        if time.monotonic() >= deadline:
            _logger.error("Process startup timed out after %ds waiting for: %s", timeout_sec, success_message)
            raise TimeoutError(
                f"Process startup timed out after {timeout_sec}s "
                f"waiting for success message: {success_message}"
            )

        time.sleep(1)


def log_process_pid(
    name: str,
    parent_pid: Union[int, None] = None,
    child_pid: Union[int, None] = None,
    log_dir: str = _LOG_DIR
) -> None:
    """将进程 PID 写入文件，供外部监控工具读取。

    文件格式:
        parent:<pid>\n
        child:<pid>\n

    Args:
        name:       进程标识（作为文件名前缀）
        parent_pid: 父进程 PID（可选）
        child_pid:  子进程 PID（可选）
        log_dir:    PID 文件存储目录（默认 wings/logs）
    """
    try:
        #
        os.makedirs(log_dir, exist_ok=True)

        #
        pid_file = os.path.join(log_dir, f"{name}_pid.txt")

        # PID
        pid_content = ""
        if parent_pid is not None:
            pid_content += f"parent:{parent_pid}\n"
        if child_pid is not None:
            pid_content += f"child:{child_pid}\n"
        safe_write_file(pid_file, pid_content)

        #
        log_msg = f"Logged process PID - name: {name}"
        if parent_pid is not None:
            log_msg += f", parent: {parent_pid}"
        if child_pid is not None:
            log_msg += f", child: {child_pid}"
        log_msg += f" to {pid_file}"

        logger.info(log_msg)
    except Exception as e:
        logger.error("Failed to log PID: %s", e, exc_info=True)
        raise


def log_stream(process):
    def _log_stdout():
        try:
            for line in process.stdout:
                stripped = line.strip()
                if stripped:
                    logger.log(infer_log_level(stripped, logging.INFO), stripped)
        except Exception as e:
            logger.error("Log stdout error: %s", e)

    def _log_stderr():
        try:
            for line in process.stderr:
                stripped = line.strip()
                if stripped:
                    logger.log(infer_log_level(stripped, logging.ERROR), stripped)
        except Exception as e:
            logger.error("Log stderr error: %s", e)

    stdout_thread = threading.Thread(target=_log_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_log_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    logger.info("Service started successfully. Process and log threads are running independently.")
