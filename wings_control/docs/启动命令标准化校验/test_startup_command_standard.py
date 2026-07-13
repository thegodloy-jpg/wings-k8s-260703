import json
import sys
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))

import startup_command_standard as tool  # noqa: E402


def test_normalize_wings_log_and_multiline_json():
    raw = """
[AI Service Log] [wings-env] export OMP_NUM_THREADS=1
[AI Service Log] [wings-cmd] >>> exec python3 -m vllm.entrypoints.openai.api_server \\
  --model /models/qwen \\
  --port 18000 \\
  --speculative-config '{"num_speculative_tokens":3,"method":"qwen3_5_mtp"}'
"""

    normalized = tool.normalize_command(raw)

    assert normalized.valid
    assert normalized.environment == {"OMP_NUM_THREADS": 1}
    assert normalized.flags["model"] == "/models/qwen"
    assert normalized.flags["port"] == 18000
    assert normalized.flags["speculative-config"] == {
        "method": "qwen3_5_mtp",
        "num_speculative_tokens": 3,
    }
    assert any(item.code == "remove_wings_log_prefix" for item in normalized.repairs)


def test_compare_ignores_order_quoting_and_entrypoint_form():
    standard = tool.normalize_command(
        "vllm serve /models/qwen --port 18000 "
        "--speculative-config '{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3}'"
    )
    actual = tool.normalize_command(
        "python3 -m vllm.entrypoints.openai.api_server "
        "--speculative-config '{\"num_speculative_tokens\": 3, \"method\": \"qwen3_5_mtp\"}' "
        "--port 18000 --model /models/qwen"
    )

    report = tool.compare_commands(standard, actual)

    assert report["result"] == "PASS"
    assert report["summary"]["failed"] == 0


def test_compare_detects_value_change_and_extra_cli():
    standard = tool.normalize_command(
        "vllm serve /models/qwen --port 18000 --tensor-parallel-size 2"
    )
    actual = tool.normalize_command(
        "vllm serve /models/qwen --port 18000 --tensor-parallel-size 4 --enable-expert-parallel"
    )

    report = tool.compare_commands(standard, actual)

    assert report["result"] == "FAIL"
    failed_paths = {item["path"] for item in report["differences"] if item["result"] == "FAIL"}
    assert "tensor-parallel-size" in failed_paths
    assert "enable-expert-parallel" in failed_paths


def test_invalid_input_does_not_invent_engine_command():
    normalized = tool.normalize_command("some text without an engine command")

    assert not normalized.valid
    assert [item.code for item in normalized.errors] == ["engine_command_missing"]


def test_malformed_json_flag_is_invalid_input():
    normalized = tool.normalize_command(
        "vllm serve /models/qwen --speculative-config '{bad json}'"
    )

    assert not normalized.valid
    assert "json_flag_parse_error" in [item.code for item in normalized.errors]


def test_generate_current_project_command_from_existing_qwen_scenario(tmp_path):
    scenario_path = TOOL_DIR / "examples" / "qwen35-27b-910b.scenario.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    standard_path = (
        tool.REPO_ROOT
        / "wings_control"
        / "docs"
        / "DAY0"
        / "dry_run"
        / "qwen_day0_current"
        / "Qwen3.5-27B-910B"
        / "exec_command.txt"
    )

    actual_script, runtime = tool.generate_wings_command(scenario, tmp_path)
    standard = tool.normalize_command(standard_path.read_text(encoding="utf-8"))
    actual = tool.normalize_command(actual_script)
    report = tool.compare_commands(standard, actual, scenario, runtime)

    assert report["result"] == "PASS", tool.comparison_markdown(report)
    assert (tmp_path / "actual_start_command.sh").exists()
    assert (tmp_path / "merged_params.json").exists()
