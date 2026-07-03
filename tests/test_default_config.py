import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_device_default_model_deploy_config_overrides_common_default(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "vllm_default.json",
        {
            "max_model_len": 4096,
            "model_deploy_config": {
                "llm": {
                    "default": {
                        "vllm": {
                            "tool_call_parser": "common_parser",
                        }
                    }
                }
            },
        },
    )
    _write_json(
        tmp_path / "nvidia_default.json",
        {
            "model_deploy_config": {
                "llm": {
                    "default": {
                        "vllm": {
                            "tool_call_parser": "device_parser",
                        }
                    }
                }
            }
        },
    )

    monkeypatch.setattr(config_loader, "DEFAULT_CONFIG_DIR", str(tmp_path))

    config = config_loader._load_default_config({"device": "nvidia"})

    assert config["max_model_len"] == 4096
    assert (
        config["model_deploy_config"]["llm"]["default"]["vllm"]["tool_call_parser"]
        == "device_parser"
    )


def test_nvidia_default_contains_function_call_model_defaults():
    config = config_loader._load_default_config({"device": "nvidia"})

    model_defaults = config["model_deploy_config"]["llm"]["DeepseekV3ForCausalLM"]
    assert model_defaults["default"]["vllm"]["tool_call_parser"] == "deepseek_v3"
