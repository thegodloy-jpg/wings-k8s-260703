import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402


class _FakeDeepSeekV4Info:
    config = {}
    model_quantize = "w8a8"

    @staticmethod
    def identify_model_architecture():
        return "DeepseekV4ForCausalLM"

    @staticmethod
    def identify_model_type():
        return "llm"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _as_dict(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


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


def test_deepseek_v4_flash_ascend_a2_defaults_are_selected_without_static_topology(monkeypatch):
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a2")

    config = config_loader._get_model_specific_config(
        {"device": "ascend"},
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "distributed": False,
        },
        _FakeDeepSeekV4Info(),
    )

    assert config["max_model_len"] == 133120
    assert config["max_num_batched_tokens"] == 8192
    assert config["max_num_seqs"] == 32
    assert config["no_enable_prefix_caching"] is True
    assert "enable_prefix_caching" not in config
    assert "tensor_parallel_size" not in config
    assert "data_parallel_size" not in config
    additional_config = _as_dict(config["additional_config"])
    assert additional_config["enable_dsa_cp"] is True
    assert additional_config["ascend_compilation_config"] == {
        "enable_npugraph_ex": True,
        "enable_static_kernel": False,
    }


def test_deepseek_v4_flash_ascend_a3_defaults_are_selected_without_static_topology(monkeypatch):
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")

    config = config_loader._get_model_specific_config(
        {"device": "ascend"},
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "distributed": False,
        },
        _FakeDeepSeekV4Info(),
    )

    assert config["max_model_len"] == 1048576
    assert config["max_num_batched_tokens"] == 10240
    assert config["max_num_seqs"] == 64
    assert config["api_server_count"] == 1
    assert "enable_prefix_caching" not in config
    assert "no_enable_prefix_caching" not in config
    assert "tensor_parallel_size" not in config
    assert "data_parallel_size" not in config
    additional_config = _as_dict(config["additional_config"])
    assert "enable_dsa_cp" not in additional_config
    assert additional_config["ascend_compilation_config"] == {
        "enable_npugraph_ex": True,
        "enable_static_kernel": False,
    }
