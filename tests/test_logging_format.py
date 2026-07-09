import importlib
import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))


def _load_launcher_module():
    module_path = Path("wings_control/wings_control.py").resolve()
    spec = importlib.util.spec_from_file_location("_wings_control_launcher_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_log_format_includes_wings_control_project_prefix(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    from utils import log_config

    reloaded = importlib.reload(log_config)

    assert (
        reloaded.LOG_FORMAT
        == "%(asctime)s.%(msecs)03d [%(levelname)s] WINGS-CONTROL [%(name)s#%(funcName)s:%(lineno)d] %(message)s"
    )


def test_normal_log_output_includes_component_function_and_line(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    from utils import log_config

    reloaded = importlib.reload(log_config)
    record = logging.LogRecord(
        "wings_control.core.config_loader",
        logging.INFO,
        pathname="",
        lineno=2675,
        msg="Set global environment variable WINGS_ENGINE=%s",
        args=("vllm",),
        exc_info=None,
        func="load_and_merge_configs",
    )
    record.created = datetime(2026, 7, 8, 16, 8, 24, 900000).timestamp()
    record.msecs = 900

    formatted = reloaded.WingsControlFormatter(
        reloaded.LOG_FORMAT,
        datefmt=reloaded.LOG_DATE_FORMAT,
    ).format(record)

    assert (
        formatted
        == "2026-07-08 16:08:24.900 [INFO] WINGS-CONTROL [wings_control.core.config_loader#load_and_merge_configs:2675] Set global environment variable WINGS_ENGINE=vllm"
    )


def test_child_structured_error_output_keeps_payload_with_child_metadata():
    launcher = _load_launcher_module()

    relay = launcher._normalize_child_log_line(
        "proxy",
        '2026-07-07 09:43:42 [ERROR] [wings-proxy] {"evt": "retry_exception"}',
    )

    assert relay.logger_name == "wings-launcher"
    assert relay.level == logging.WARNING
    assert relay.message == '{"evt": "retry_exception"}'
    assert relay.extra == {
        "wings_child_component": "proxy",
        "wings_child_time": "2026-07-07 09:43:42",
        "wings_child_source": "wings-proxy",
    }


def test_child_timestamp_first_source_output_keeps_child_source(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    launcher = _load_launcher_module()

    from utils import log_config

    reloaded = importlib.reload(log_config)
    relay = launcher._normalize_child_log_line(
        "proxy",
        '2026-07-08 16:08:24.900 [ERROR] WINGS-CONTROL [wings-proxy#handle_request:486] {"evt": "retry_exception"}',
    )
    record = logging.LogRecord(
        relay.logger_name,
        relay.level,
        pathname="",
        lineno=0,
        msg=relay.message,
        args=(),
        exc_info=None,
    )
    for key, value in relay.extra.items():
        setattr(record, key, value)

    formatted = reloaded.WingsControlFormatter(
        reloaded.LOG_FORMAT,
        datefmt=reloaded.LOG_DATE_FORMAT,
    ).format(record)

    assert (
        formatted
        == '2026-07-08 16:08:24.900 [WARNING] WINGS-CONTROL [wings-proxy#handle_request:486] {"evt": "retry_exception"}'
    )


def test_child_project_prefixed_error_output_formats_hierarchy_once(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    launcher = _load_launcher_module()

    from utils import log_config

    reloaded = importlib.reload(log_config)
    relay = launcher._normalize_child_log_line(
        "proxy",
        '[WINGS-CONTROL] 2026-07-07 09:43:42 [ERROR] [wings-proxy] {"evt": "retry_exception"}',
    )
    record = logging.LogRecord(
        relay.logger_name,
        relay.level,
        pathname="",
        lineno=0,
        msg=relay.message,
        args=(),
        exc_info=None,
    )
    for key, value in relay.extra.items():
        setattr(record, key, value)

    formatted = reloaded.WingsControlFormatter(
        reloaded.LOG_FORMAT,
        datefmt=reloaded.LOG_DATE_FORMAT,
    ).format(record)

    assert (
        formatted
        == '2026-07-07 09:43:42.000 [WARNING] WINGS-CONTROL [wings-proxy] {"evt": "retry_exception"}'
    )


def test_child_component_first_proxy_output_formats_hierarchy_once(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    launcher = _load_launcher_module()

    from utils import log_config

    reloaded = importlib.reload(log_config)
    relay = launcher._normalize_child_log_line(
        "proxy",
        '[WINGS-CONTROL][wings-proxy] [ERROR]2026-07-08 03:12:04 {"evt": "retry_exception"}',
    )
    record = logging.LogRecord(
        relay.logger_name,
        relay.level,
        pathname="",
        lineno=0,
        msg=relay.message,
        args=(),
        exc_info=None,
    )
    for key, value in relay.extra.items():
        setattr(record, key, value)

    formatted = reloaded.WingsControlFormatter(
        reloaded.LOG_FORMAT,
        datefmt=reloaded.LOG_DATE_FORMAT,
    ).format(record)

    assert (
        formatted
        == '2026-07-08 03:12:04.000 [WARNING] WINGS-CONTROL [wings-proxy] {"evt": "retry_exception"}'
    )


def test_child_timestamp_first_proxy_output_formats_hierarchy_once(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    launcher = _load_launcher_module()

    from utils import log_config

    reloaded = importlib.reload(log_config)
    relay = launcher._normalize_child_log_line(
        "proxy",
        '2026-07-08 03:12:04 [WINGS-CONTROL][wings-proxy] [ERROR] {"evt": "retry_exception"}',
    )
    record = logging.LogRecord(
        relay.logger_name,
        relay.level,
        pathname="",
        lineno=0,
        msg=relay.message,
        args=(),
        exc_info=None,
    )
    for key, value in relay.extra.items():
        setattr(record, key, value)

    formatted = reloaded.WingsControlFormatter(
        reloaded.LOG_FORMAT,
        datefmt=reloaded.LOG_DATE_FORMAT,
    ).format(record)

    assert (
        formatted
        == '2026-07-08 03:12:04.000 [WARNING] WINGS-CONTROL [wings-proxy] {"evt": "retry_exception"}'
    )


def test_log_analyzer_uses_wings_control_project_prefix():
    source = Path("wings_control/log_analyzer/log_analyzer.py").read_text(encoding="utf-8")

    assert (
        "%(asctime)s.%(msecs)03d [%(levelname)s] WINGS-CONTROL [%(name)s#%(funcName)s:%(lineno)d] %(message)s"
        in source
    )
