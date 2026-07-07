import importlib
import importlib.util
import logging
import sys
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
        == "[WINGS-CONTROL] %(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )


def test_child_structured_error_output_is_relayed_once_with_child_level_and_logger():
    launcher = _load_launcher_module()

    relay = launcher._normalize_child_log_line(
        "proxy",
        '2026-07-07 09:43:42 [ERROR] [wings-proxy] {"evt": "retry_exception"}',
    )

    assert relay == (
        "wings-proxy",
        logging.ERROR,
        '[proxy] {"evt": "retry_exception"}',
    )


def test_child_structured_error_output_accepts_project_prefixed_format():
    launcher = _load_launcher_module()

    relay = launcher._normalize_child_log_line(
        "proxy",
        '[WINGS-CONTROL] 2026-07-07 09:43:42 [ERROR] [wings-proxy] {"evt": "retry_exception"}',
    )

    assert relay == (
        "wings-proxy",
        logging.ERROR,
        '[proxy] {"evt": "retry_exception"}',
    )


def test_log_analyzer_uses_wings_control_project_prefix():
    source = Path("wings_control/log_analyzer/log_analyzer.py").read_text(encoding="utf-8")

    assert "[WINGS-CONTROL] %(asctime)s [%(levelname)s] [%(name)s] %(message)s" in source
