import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402


def _assert_error_stack_logging_is_forced(engine: str) -> None:
    command = vllm_adapter._build_vllm_cmd_parts(
        {
            "engine": engine,
            "engine_config": {
                "no_log_error_stack": True,
                "log_error_stack": False,
            },
        }
    )

    assert command.count("--log-error-stack") == 1
    assert "--no-log-error-stack" not in command


def test_vllm_ascend_start_command_forces_error_stack_logging():
    _assert_error_stack_logging_is_forced("vllm_ascend")


def test_vllm_start_command_forces_error_stack_logging():
    _assert_error_stack_logging_is_forced("vllm")


def test_non_vllm_engine_does_not_force_error_stack_logging():
    command = vllm_adapter._build_vllm_cmd_parts(
        {
            "engine": "sglang",
            "engine_config": {"log_error_stack": False},
        }
    )

    assert "--log-error-stack" not in command
