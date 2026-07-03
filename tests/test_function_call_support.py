import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINGS_CONTROL = ROOT / "wings_control"
if str(WINGS_CONTROL) not in sys.path:
    sys.path.insert(0, str(WINGS_CONTROL))


from core import config_loader  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402


def test_enable_auto_tool_choice_uses_yaml_function_call_parser(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["FunctionCallUnitTestForCausalLM"]}),
        encoding="utf-8",
    )

    support_path = tmp_path / "function_call_support.yaml"
    support_path.write_text(
        """
feature: function_call
field: tool_call_parser
engines: [vllm, vllm_ascend, sglang]
architectures:
  - name: FunctionCallUnitTestForCausalLM
    config:
      vllm_ascend: {default: unit_parser}
    models: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_loader, "FUNCTION_CALL_SUPPORT_PATH", support_path, raising=False)
    if hasattr(config_loader, "_load_function_call_support"):
        config_loader._load_function_call_support.cache_clear()

    args = parse_launch_args(
        [
            "--model-name",
            "Unit-FC",
            "--model-path",
            str(model_dir),
            "--engine",
            "vllm_ascend",
            "--enable-auto-tool-choice",
        ]
    )

    merged = config_loader.load_and_merge_configs(
        {"device": "ascend", "count": 1, "details": []},
        args.to_namespace(),
    )

    assert merged["engine_config"]["tool_call_parser"] == "unit_parser"
    assert merged["engine_config"]["enable_auto_tool_choice"] is True
