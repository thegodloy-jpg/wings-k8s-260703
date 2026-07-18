import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402


def _build_command(engine: str, engine_config: dict) -> str:
    return vllm_adapter._build_vllm_cmd_parts(
        {
            "engine": engine,
            "engine_config": engine_config,
        }
    )


def _assert_error_stack_logging_is_not_forced(engine: str) -> None:
    command = vllm_adapter._build_vllm_cmd_parts(
        {
            "engine": engine,
            "engine_config": {},
        }
    )

    assert "--log-error-stack" not in command
    assert "--no-log-error-stack" not in command


def test_vllm_ascend_start_command_does_not_force_error_stack_logging():
    _assert_error_stack_logging_is_not_forced("vllm_ascend")


def test_vllm_start_command_does_not_force_error_stack_logging():
    _assert_error_stack_logging_is_not_forced("vllm")


def test_vllm_start_command_preserves_explicit_error_stack_logging_config():
    command = _build_command(
        "vllm",
        {
            "no_log_error_stack": True,
            "log_error_stack": False,
        },
    )

    assert "--log-error-stack" not in command
    assert command.count("--no-log-error-stack") == 1
