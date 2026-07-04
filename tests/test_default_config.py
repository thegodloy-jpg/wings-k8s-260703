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
    assert config["no_disable_hybrid_kv_cache_manager"] is True
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
    assert config["no_disable_hybrid_kv_cache_manager"] is True
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


def test_deepseek_v4_flash_reasoning_parser_support_file_is_loaded(monkeypatch):
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a2")

    config_loader._load_reasoning_parser_support.cache_clear()
    config = config_loader._get_model_specific_config(
        {"device": "ascend"},
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "distributed": False,
            "enable_auto_think_choice": True,
        },
        _FakeDeepSeekV4Info(),
    )

    assert config["reasoning_parser"] == "deepseek_v4"


def test_reasoning_parser_support_source_file_uses_short_name():
    assert config_loader.REASONING_PARSER_SUPPORT_PATH.name == "reason_parser.yaml"
    assert config_loader.REASONING_PARSER_SUPPORT_PATH.exists()


def test_explicit_served_model_name_overrides_model_name(monkeypatch):
    monkeypatch.setenv("SERVED_MODEL_NAME", "dsv4")

    config = config_loader._get_model_specific_config(
        {"device": "ascend"},
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "distributed": False,
            "served_model_name": "dsv4",
        },
        _FakeDeepSeekV4Info(),
    )

    assert config["served_model_name"] == "dsv4"


def test_vllm_final_config_drops_generic_chat_template():
    final = config_loader._merge_final_config(
        {
            "chat_template": "/usr/local/serving/models/chat_template.jinja",
            "default_chat_template_kwargs": {"thinking": False},
        },
        {"engine": "vllm_ascend"},
    )

    assert "chat_template" not in final["engine_config"]
    assert final["engine_config"]["default_chat_template_kwargs"] == {"thinking": False}


def _mapping_path() -> str:
    return str(
        Path(config_loader.DEFAULT_CONFIG_DIR)
        / config_loader.DEFAULT_CONFIG_FILES["engine_parameter_mapping"]
    )


def _clear_removed_param_env(monkeypatch):
    for env_name in (
        "DTYPE",
        "KV_CACHE_DTYPE",
        "GPU_MEMORY_UTILIZATION",
        "BLOCK_SIZE",
        "SEED",
        "QUANTIZATION",
        "ENABLE_CHUNKED_PREFILL",
        "ENABLE_PREFIX_CACHING",
        "ENABLE_EXPERT_PARALLEL",
    ):
        monkeypatch.delenv(env_name, raising=False)


def test_removed_page_tuning_defaults_are_not_backfilled_when_not_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    _clear_removed_param_env(monkeypatch)

    params = {}
    engine_cmd_parameter = {
        "dtype": "auto",
        "kv_cache_dtype": "auto",
        "quantization": "awq",
        "quantization_param_path": "/tmp/quant.json",
        "gpu_memory_utilization": 0.9,
        "enable_chunked_prefill": False,
        "block_size": 16,
        "seed": 0,
        "enable_expert_parallel": False,
        "enable_prefix_caching": False,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 4096,
        "trust_remote_code": True,
    }

    config_loader._set_common_params(params, engine_cmd_parameter, _mapping_path())

    for key in (
        "dtype",
        "kv_cache_dtype",
        "quantization",
        "quantization_param_path",
        "gpu_memory_utilization",
        "enable_chunked_prefill",
        "block_size",
        "seed",
        "enable_expert_parallel",
        "enable_prefix_caching",
    ):
        assert key not in params
    assert params["max_num_seqs"] == 32
    assert params["max_num_batched_tokens"] == 4096
    assert params["trust_remote_code"] is True


def test_removed_page_tuning_defaults_still_allow_explicit_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest", "--gpu-memory-utilization", "0.8"])
    _clear_removed_param_env(monkeypatch)

    params = {"gpu_memory_utilization": 0.92}
    engine_cmd_parameter = {"gpu_memory_utilization": 0.8}

    config_loader._set_common_params(params, engine_cmd_parameter, _mapping_path())

    assert params["gpu_memory_utilization"] == 0.8


def _model_deploy_config(device: str) -> dict:
    return config_loader._load_default_config({"device": device})["model_deploy_config"]


def _assert_engine_fields(config: dict, expected: dict) -> None:
    for key, value in expected.items():
        assert config[key] == value


def _assert_engines(config: dict, engines: tuple[str, ...], expected: dict) -> None:
    for engine in engines:
        _assert_engine_fields(config[engine], expected)


def _contains_quantization_ascend(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_quantization_ascend(v) for v in value.values()) or (
            value.get("quantization") == "ascend"
        )
    if isinstance(value, list):
        return any(_contains_quantization_ascend(v) for v in value)
    return False


def test_ascend_defaults_follow_parameter_reduction_plan():
    llm = _model_deploy_config("ascend")["llm"]

    for arch, names in {
        "DeepseekV3ForCausalLM": (
            "default",
            "DeepSeek-R1-w8a8",
            "DeepSeek-V3.1-w8a8",
        ),
        "Qwen3MoeForCausalLM": ("default", "Qwen3-235B-A22B"),
    }.items():
        for name in names:
            _assert_engines(
                llm[arch][name],
                ("vllm_ascend", "vllm_ascend_distributed"),
                {"enable_expert_parallel": True},
            )

    _assert_engines(
        llm["DeepseekV32ForCausalLM"]["default"],
        ("vllm_ascend", "vllm_ascend_distributed"),
        {"enable_expert_parallel": True},
    )
    _assert_engines(
        llm["Glm4MoeForCausalLM"]["default"],
        ("vllm_ascend", "vllm_ascend_distributed"),
        {"enable_expert_parallel": True},
    )

    deepseek_v4 = llm["DeepseekV4ForCausalLM"]
    assert deepseek_v4["DeepSeek-V4-Flash-A2"]["vllm_ascend"]["no_enable_prefix_caching"] is True
    assert "enable_prefix_caching" not in deepseek_v4["DeepSeek-V4-Flash-A2"]["vllm_ascend"]
    assert "enable_prefix_caching" not in deepseek_v4["DeepSeek-V4-Flash-A3"]["vllm_ascend"]
    assert "no_enable_prefix_caching" not in deepseek_v4["DeepSeek-V4-Flash-A3"]["vllm_ascend"]
    _assert_engine_fields(
        deepseek_v4["DeepSeek-V4-Pro"]["vllm_ascend_distributed"],
        {"enable_chunked_prefill": True, "enable_prefix_caching": True},
    )

    glm5 = llm["GlmMoeDsaForCausalLM"]
    _assert_engines(
        glm5["default"],
        ("vllm_ascend", "vllm_ascend_distributed"),
        {"enable_expert_parallel": True},
    )
    assert glm5["GLM-5.2-w8a8"]["vllm_ascend"]["gpu_memory_utilization"] == 0.95
    assert glm5["GLM-5.2-w8a8"]["vllm_ascend_distributed"]["gpu_memory_utilization"] == 0.93

    qwen35_dense = llm["Qwen3_5ForConditionalGeneration"]
    assert "enable_prefix_caching" not in qwen35_dense["default"]["vllm_ascend"]
    for model_name in ("Qwen3.5-27B-w8a8", "Qwen3.6-27B-w8a8"):
        _assert_engines(
            qwen35_dense[model_name],
            ("vllm_ascend", "vllm_ascend_distributed"),
            {
                "seed": 1024,
                "quantization": "ascend",
                "gpu_memory_utilization": 0.9,
                "no_enable_prefix_caching": True,
            },
        )

    qwen35_moe = llm["Qwen3_5MoeForConditionalGeneration"]
    for engine in ("vllm_ascend", "vllm_ascend_distributed"):
        default_cfg = qwen35_moe["default"][engine]
        assert default_cfg["enable_expert_parallel"] is True
        assert "quantization" not in default_cfg
        assert "gpu_memory_utilization" not in default_cfg
        assert "enable_prefix_caching" not in default_cfg
    _assert_engines(
        qwen35_moe["Qwen3.6-35B-A3B"],
        ("vllm_ascend", "vllm_ascend_distributed"),
        {"seed": 1024},
    )
    _assert_engines(
        qwen35_moe["Qwen-AgentWorld-35B-A3B"],
        ("vllm_ascend", "vllm_ascend_distributed"),
        {"enable_expert_parallel": True, "gpu_memory_utilization": 0.9},
    )

    _assert_engines(
        llm["Qwen3NextForCausalLM"]["default"],
        ("vllm_ascend", "vllm_ascend_distributed"),
        {"gpu_memory_utilization": 0.8},
    )
    _assert_engines(
        llm["KimiK25ForConditionalGeneration"]["default"],
        ("vllm_ascend", "vllm_ascend_distributed"),
        {"quantization": "ascend", "seed": 1024},
    )


def test_nvidia_defaults_follow_parameter_reduction_plan():
    config = _model_deploy_config("nvidia")
    llm = config["llm"]

    assert not _contains_quantization_ascend(config)

    moe_entries = {
        "DeepseekV3ForCausalLM": (
            "default",
            "DeepSeek-R1",
            "DeepSeek-V3.1",
        ),
        "DeepseekV32ForCausalLM": ("default",),
        "Qwen3MoeForCausalLM": (
            "default",
            "Qwen3-Coder-480B-A35B-Instruct",
            "Qwen3-Coder-30B-A3B-Instruct",
        ),
        "Qwen3_5MoeForConditionalGeneration": (
            "default",
            "Qwen-AgentWorld-35B-A3B",
        ),
        "Glm4MoeForCausalLM": ("default",),
        "GlmMoeDsaForCausalLM": ("default",),
    }
    for arch, names in moe_entries.items():
        for name in names:
            _assert_engines(
                llm[arch][name],
                ("vllm", "vllm_distributed"),
                {"enable_expert_parallel": True},
            )

    deepseek_v4_default = llm["DeepseekV4ForCausalLM"]["default"]
    for engine in ("vllm", "vllm_distributed"):
        assert "kv_cache_dtype" not in deepseek_v4_default[engine]
        assert "block_size" not in deepseek_v4_default[engine]
        assert "gpu_memory_utilization" not in deepseek_v4_default[engine]
        assert "enable_expert_parallel" not in deepseek_v4_default[engine]
    deepseek_flash = llm["DeepseekV4ForCausalLM"]["DeepSeek-V4-Flash"]
    expected_deepseek_flash = {
        "kv_cache_dtype": "fp8",
        "block_size": 256,
        "gpu_memory_utilization": 0.92,
        "enable_expert_parallel": True,
    }
    _assert_engine_fields(deepseek_flash["vllm"]["default"], expected_deepseek_flash)
    _assert_engine_fields(deepseek_flash["vllm_distributed"], expected_deepseek_flash)
    _assert_engine_fields(
        deepseek_flash["vllm"]["rtx_pro_5000_72G"],
        {
            "kv_cache_dtype": "fp8",
            "block_size": 256,
            "gpu_memory_utilization": 0.9,
        },
    )
    for arch in (
        "Qwen3ForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "MiniMaxM2ForCausalLM",
        "LlamaForCausalLM",
        "Glm4ForCausalLM",
        "Qwen3NextForCausalLM",
    ):
        for engine in ("vllm", "vllm_distributed"):
            assert "gpu_memory_utilization" not in llm[arch]["default"][engine]
            assert "block_size" not in llm[arch]["default"][engine]
