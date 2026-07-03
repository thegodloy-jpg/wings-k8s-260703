# -*- coding: utf-8 -*-
"""
日志与输出噪声过滤模块。
------------------------------------------------

过滤高频低价值的日志消息，改善运行时信号清晰度:

- uvicorn 的 access 日志中 "GET /health ..." 探针请求
- 推理引擎的 "Prefill batch." / "Decode batch." 调试输出
- pynvml 的 FutureWarning 警告（torch.cuda 触发）
- 覆盖 stdout/stderr 过滤引擎内部的高频噪声 print

配置环境变量:

- NOISE_FILTER_DISABLE=1            : 完全禁用所有过滤
- HEALTH_FILTER_ENABLE=0/1           : /health 日志过滤（默认 1）
- HEALTH_PATH_REGEX=...              : "\"GET\\s+/health\\b"
- BATCH_NOISE_FILTER_ENABLE=0/1      : Prefill/Decode 噪声过滤（默认 1）
- BATCH_NOISE_REGEX=...              : "(?i)\\b(?:prefill|decode)\\b[^\\n]{0,200}\\bbatch\\b"
- PYNVML_FILTER_ENABLE=0/1           : pynvml 警告过滤（默认 1）
- PYNVML_NOISE_REGEX=...             : "(?i)(pynvml package is deprecated|\\bimport pynvml\\b)"
- STDIO_FILTER_ENABLE=0/1            : stdout/stderr 过滤（默认 1）
"""
from __future__ import annotations
import logging
import os
import re
import sys
import warnings
from typing import Iterable


# -----------------  -----------------
def _env_bool(k: str, default: bool) -> bool:
    """读取环境变量并解析为布尔值。

    Args:
        k: 环境变量名称。
        default: 环境变量未设置或为空时的默认返回值。

    Returns:
        若值为 '1/true/yes/y/on' 返回 True，否则返回 default。
    """
    v = os.getenv(k, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")

if _env_bool("NOISE_FILTER_DISABLE", False):
    #
    def install_noise_filters(*_args, **_kwargs) -> None:
        return
else:
    _HEALTH_ON = _env_bool("HEALTH_FILTER_ENABLE", True)
    _BATCH_ON = _env_bool("BATCH_NOISE_FILTER_ENABLE", True)
    _PYNVML_ON = _env_bool("PYNVML_FILTER_ENABLE", True)
    _STDIO_ON = _env_bool("STDIO_FILTER_ENABLE", True)

    _HEALTH_RE = os.getenv("HEALTH_PATH_REGEX",
                           r'"(GET\s+/health\b|POST\s+/api/heartbeat\b|GET\s+/api/nodes\b)')
    _BATCH_RE = os.getenv("BATCH_NOISE_REGEX",
                         r'(?i)\b(?:prefill|decode)\b[^\n]{0,200}\bbatch\b')
    _PYNVML_RE = os.getenv("PYNVML_NOISE_REGEX",
                          r'(?i)(pynvml package is deprecated|\bimport pynvml\b)')

    _HEALTH_PAT = re.compile(_HEALTH_RE)
    _BATCH_PAT = re.compile(_BATCH_RE)
    _PYNVML_PAT = re.compile(_PYNVML_RE)


    class _DropByRegex(logging.Filter):
        """按正则表达式丢弃日志记录的 logging Filter。

        当日志消息匹配任意一个给定 pattern 时，该记录会被丢弃。

        Attributes:
            pats: 编译后的正则表达式元组
        """
        __slots__ = ("pats",)


        def __init__(self, patterns: Iterable[re.Pattern]):
            super().__init__()
            self.pats = tuple(patterns)


        def filter(self, record: logging.LogRecord) -> int:
            """判断日志记录是否应保留。

            遍历所有 pattern，若消息匹配任意一个则丢弃该记录。

            Args:
                record: 待过滤的日志记录对象。

            Returns:
                0 表示丢弃该记录，1 表示保留。
            """
            try:
                msg = record.getMessage()
            except Exception as _:
                msg = record.msg if isinstance(record.msg, str) else str(record.msg)
            for p in self.pats:
                if p.search(msg):
                    return 0  #
            return 1


    class _LineFilterIO:
        """按行过滤的 stdout/stderr 包装器。

        将原始输出流包装，匹配指定 pattern 的行会被丢弃。
        用于过滤引擎库内部的高频噪声 print。

        Attributes:
            _under: 原始输出流
            _buf:   待处理的缓冲区
            _pats:  过滤用正则表达式元组
        """
        __slots__ = ("_under", "_buf", "_pats", "_closed", "name")


        def __init__(self, under, patterns: Iterable[re.Pattern], name: str):
            self._under = under
            self._buf = ""
            self._pats = tuple(patterns)
            self._closed = False
            self.name = getattr(under, "name", name)


        def __getattr__(self, item):
            # 代理所有未覆盖的属性到原始输出流
            return getattr(self._under, item)


        def fileno(self):
            return self._under.fileno() if hasattr(self._under, "fileno") else -1


        def isatty(self):
            return self._under.isatty() if hasattr(self._under, "isatty") else False


        def write(self, s: str):
            """将内容写入缓冲区，按行过滤后输出到原始流。

            数据先追加到内部缓冲区，遇到换行符时逐行检查是否匹配
            过滤 pattern。匹配的行被丢弃，不匹配的行合并后写入原始流。
            未遇到换行符的尾部数据会留在缓冲区，等待后续 write 或 flush。

            Args:
                s: 待写入的字符串内容。

            Returns:
                实际写入原始流的字节数；若全部被过滤则返回 0。
            """
            if self._closed:
                return 0
            if not isinstance(s, str):
                s = str(s)
            self._buf += s
            out = []
            while True:
                nl = self._buf.find("\n")
                if nl < 0:
                    break
                line = self._buf[:nl+1]
                self._buf = self._buf[nl+1:]
                # 逐行检查：匹配噪声 pattern 的行被丢弃
                if not any(p.search(line) for p in self._pats):
                    out.append(line)
            if out:
                return self._under.write("".join(out))
            return len(s)


        def flush(self):
            """刷新缓冲区，将未以换行符结尾的尾部数据写入原始流。

            尾部数据同样经过 pattern 过滤：匹配则丢弃，不匹配则写出。
            最后调用原始流的 flush 确保数据落盘。

            Returns:
                原始流 flush 的返回值；流已关闭时返回 None。
            """
            if self._closed:
                return None
            # 将缓冲区中剩余的尾部数据（未以换行符结尾）过滤后写出
            if self._buf:
                tail = self._buf
                self._buf = ""
                if not any(p.search(tail) for p in self._pats):
                    self._under.write(tail)
            result = self._under.flush()
            return result


        def close(self):
            """关闭过滤流：先 flush 缓冲区，再标记为已关闭。

            Returns:
                True 表示 flush 成功后关闭，False 表示 flush 异常后关闭。
                若已关闭则直接返回 True。
            """
            if not self._closed:
                flush_success = False
                try:
                    self.flush()
                except Exception as e:
                    # flush 失败时记录调试日志，仍需标记关闭
                    logging.getLogger(__name__).debug(
                        f"Flush failed during close: {e}"
                    )
                else:
                    # flush 成功
                    flush_success = True
                finally:
                    self._closed = True
                return flush_success
            return True


    def _attach_filter_to(logger_name: str, filt: logging.Filter) -> None:
        """将过滤器添加到指定 logger 及其所有 handler 上。

        Args:
            logger_name: 目标 logger 的名称。
            filt: 要添加的 logging.Filter 实例。
        """
        try:
            lg = logging.getLogger(logger_name)
        except (TypeError, ValueError) as e:
            # logger 名称无效时记录警告
            logging.getLogger(__name__).warning(
                f"Invalid logger name '{logger_name}': {e}"
            )
            return

        # 同时添加到 logger 自身和其所有 handler
        try:
            for handler in list(lg.handlers):
                handler.addFilter(filt)
            lg.addFilter(filt)
        except (AttributeError, RuntimeError) as e:
            logging.getLogger(__name__).debug(
                f"Failed to add filter to logger '{logger_name}': {e}"
            )


    def _install_logging_filters() -> None:
        """将噪声过滤器安装到推理引擎相关的 logger 上。

        根据 HEALTH_ON / BATCH_ON / PYNVML_ON 开关收集对应的正则 pattern，
        创建 _DropByRegex 过滤器并挂载到 sglang、vllm、vllm_ascend、mindie、
        wings 等推理引擎 logger 上。
        """
        pats = []
        if _HEALTH_ON:
            pats.append(_HEALTH_PAT)
        if _BATCH_ON:
            pats.append(_BATCH_PAT)
        if _PYNVML_ON:
            pats.append(_PYNVML_PAT)
        if not pats:
            return
        filt = _DropByRegex(pats)

        # root 及推理引擎相关 logger（uvicorn/httpx/sglang/vllm 等）
        targets = [
            "sglang", "sglang.server", "sglang.runtime",
            "vllm", "vllm.entrypoints", "vllm.engine",
            "vllm_ascend", "mindie", "wings",
            "uvicorn.access",
        ]
        for name in targets:
            _attach_filter_to(name, filt)


    def _install_warning_filters() -> None:
        """安装 Python warnings 过滤器，抑制 pynvml 的 FutureWarning。

        torch.cuda 初始化时会触发 pynvml 弃用警告，对推理服务无实际意义，
        通过 warnings.filterwarnings 将其静默。
        """
        if not _PYNVML_ON:
            return
        # 过滤 torch.cuda 初始化时触发的 pynvml FutureWarning
        try:
            warnings.filterwarnings(
                "ignore",
                message=r".*pynvml package is deprecated.*",
                category=FutureWarning,
                module=r"torch\.cuda(\..*)?$",
            )
        except Exception as e:
            # 安装失败不影响主流程，仅记录调试日志
            logging.getLogger(__name__).debug(
                f"Failed to install pynvml warning filter: {e}"
            )


    def _install_stdio_filters() -> None:
        """用 _LineFilterIO 包装 sys.stdout 和 sys.stderr 实现行级噪声过滤。

        根据 HEALTH_ON / BATCH_ON / PYNVML_ON 开关收集 pattern，
        将标准输出/错误流替换为过滤包装器。引擎库内部直接 print 的
        高频噪声（如 batch 日志）会被按行拦截丢弃。
        """
        if not _STDIO_ON:
            return
        pats = []
        if _HEALTH_ON:
            pats.append(_HEALTH_PAT)
        if _BATCH_ON:
            pats.append(_BATCH_PAT)
        if _PYNVML_ON:
            pats.append(_PYNVML_PAT)
        if not pats:
            return
        try:
            # 用过滤包装器替换标准输出流和标准错误流
            sys.stdout = _LineFilterIO(sys.stdout, pats, name="stdout")
            sys.stderr = _LineFilterIO(sys.stderr, pats, name="stderr")
        except Exception as e:
            # 替换失败不影响主流程，保持原始流不变
            logging.getLogger(__name__).debug(
                f"Failed to install stdio filters: {e}"
            )


    def install_noise_filters() -> None:
        """一次性安装所有噪声过滤器，应在应用 import 阶段尽早调用。

        依次执行:
        1) 将正则过滤器挂载到推理引擎相关 logger 上
        2) 安装 warnings 过滤器抑制 pynvml FutureWarning
        3) 用行级过滤包装器替换 stdout/stderr

        若 NOISE_FILTER_DISABLE=1 则本函数为空操作（在模块加载时已替换为 no-op）。
        """
        _install_logging_filters()
        _install_warning_filters()
        _install_stdio_filters()