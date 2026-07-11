import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402


def test_vllm_ascend_start_command_forces_error_stack_logging():
    command = vllm_adapter._build_vllm_cmd_parts(
        {
            "engine": "vllm_ascend",
            "engine_config": {
                "no_log_error_stack": True,
                "log_error_stack": False,
            },
        }
    )

    assert command.count("--log-error-stack") == 1
    assert "--no-log-error-stack" not in command


def test_vllm_start_command_does_not_force_error_stack_logging():
    command = vllm_adapter._build_vllm_cmd_parts(
        {
            "engine": "vllm",
            "engine_config": {"log_error_stack": False},
        }
    )

    assert "--log-error-stack" not in command
