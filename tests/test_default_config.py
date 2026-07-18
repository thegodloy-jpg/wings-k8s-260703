import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from engines import sglang_adapter, vllm_adapter  # noqa: E402


class _FakeDeepSeekV4Info:
    config = {}
    model_quantize = "w8a8"

    @staticmethod
    def identify_model_architecture():
        return "DeepseekV4ForCausalLM"

    @staticmethod
    def identify_model_type():
        return "llm"


class _FakeModelInfo:
    config = {}
    model_quantize = ""

    def __init__(self, model_name, architecture, model_type="llm"):
        self.model_name = model_name
        self.model_path = Path("/models") / model_name
        self.model_architecture = architecture
        self.model_type = model_type

    def identify_model_architecture(self):
        return self.model_architecture

    def identify_model_type(self):
        return self.model_type


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _as_dict(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def test_deepseek_v4_flash_repo_name_selects_pro5000_vllm_defaults():
    arch_dict = _model_deploy_config("nvidia")["llm"]["DeepseekV4ForCausalLM"]
    scenario = config_loader._SpecialEngineScenario(
        deepseek_v4_flash_vllm_nvidia=True,
    )

    config = config_loader._match_model_engine_config(
        arch_dict,
        "deepseek-ai/deepseek-v4-flash",
        "vllm",
        scenario,
        _FakeDeepSeekV4Info(),
        {
            "device": "nvidia",
            "hardware_family": "NVIDIA RTX PRO 5000 72GB Blackwell",
        },
    )

    assert config["use_vllm_serve"] is True
    assert config["attention_backend"] == "FLASHMLA_SPARSE_DSV4"
    assert "rtx_pro_5000_72G" not in config


def test_model_card_profile_key_reuses_model_name_and_card_tokens():
    arch_dict = {
        "Model-H20-96G": {
            "card_tokens": ["h20-96"],
            "sglang": {"selected": "h20"},
        },
        "Model": {
            "vllm": {"selected": "base"},
        },
    }

    assert config_loader._match_model_engine_config(
        arch_dict,
        "model",
        "sglang",
        config_loader._SpecialEngineScenario(),
        _FakeModelInfo("Model", "Arch"),
        {"device": "nvidia", "details": [{"name": "NVIDIA H20 96GB"}]},
    ) == {"selected": "h20"}
    assert config_loader._match_model_engine_config(
        arch_dict,
        "model",
        "sglang",
        config_loader._SpecialEngineScenario(),
        _FakeModelInfo("Model", "Arch"),
        {"device": "nvidia", "details": [{"name": "NVIDIA A100 80GB"}]},
    ) == {}


def test_deepseek_sglang_h20_defaults_follow_detected_hardware(monkeypatch):
    monkeypatch.setenv("WINGS_H20_MODEL", "H20-96G")
    arch_dict = _model_deploy_config("nvidia")["llm"]["DeepseekV3ForCausalLM"]
    scenario = config_loader._SpecialEngineScenario()
    cases = [
        (
            {"device": "nvidia", "details": [{"name": "NVIDIA H20 96GB"}]},
            0.9,
            None,
        ),
        (
            {
                "device": "nvidia",
                "details": [{"name": "NH02(141GB) / G8600 V7", "total_memory": 141}],
            },
            0.95,
            8,
        ),
    ]

    for hardware, expected_memory_fraction, expected_dp in cases:
        config = config_loader._match_model_engine_config(
            arch_dict,
            "deepseek-v3.1",
            "sglang",
            scenario,
            _FakeModelInfo("DeepSeek-V3.1", "DeepseekV3ForCausalLM"),
            hardware,
        )
        cmd = sglang_adapter._build_sglang_cmd_parts({"engine_config": config})

        assert config["mem_fraction_static"] == expected_memory_fraction
        assert config.get("dp") == expected_dp
        assert "card_tokens" not in config
        assert "--H20-96G" not in cmd
        assert "--H20-141G" not in cmd
        assert "--card-tokens" not in cmd
        assert "--tool-call-parser deepseekv31" in cmd


def test_deepseek_sglang_unknown_card_does_not_emit_h20_profile_args(monkeypatch):
    monkeypatch.delenv("WINGS_H20_MODEL", raising=False)
    arch_dict = _model_deploy_config("nvidia")["llm"]["DeepseekV3ForCausalLM"]
    config = config_loader._match_model_engine_config(
        arch_dict,
        "deepseek-v3.1",
        "sglang",
        config_loader._SpecialEngineScenario(),
        _FakeModelInfo("DeepSeek-V3.1", "DeepseekV3ForCausalLM"),
        {"device": "nvidia", "details": [{"name": "NVIDIA A100 80GB"}]},
    )

    assert config == {}


def test_deepseek_v4_flash_repo_name_gets_pro5000_defaults_through_real_selector():
    params = config_loader._get_model_specific_config(
        {
            "device": "nvidia",
            "count": 8,
            "details": [{"name": "NVIDIA RTX PRO 5000 72GB Blackwell"}],
            "hardware_family": "NVIDIA RTX PRO 5000 72GB Blackwell",
        },
        {
            "engine": "vllm",
            "model_name": "deepseek-ai/DeepSeek-V4-Flash",
            "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
            "model_type": "llm",
            "distributed": False,
            "device_count": 8,
        },
        _FakeDeepSeekV4Info(),
    )

    assert params["use_vllm_serve"] is True
    assert params["attention_backend"] == "FLASHMLA_SPARSE_DSV4"
    assert "default" not in params
    assert "rtx_pro_5000_72G" not in params


def test_device_default_config_loads_device_file_directly(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "nvidia_default.json",
        {
            "device_marker": "nvidia",
            "model_deploy_config": {
                "llm": {
                    "default": {
                        "vllm": {
                            "tool_call_parser": "device_parser",
                        }
                    }
                }
            },
        },
    )
    _write_json(
        tmp_path / "ascend_default.json",
        {
            "device_marker": "ascend",
            "model_deploy_config": {
                "llm": {
                    "default": {
                        "vllm_ascend": {
                            "tool_call_parser": "ascend_parser",
                        }
                    }
                }
            }
        },
    )

    monkeypatch.setattr(config_loader, "DEFAULT_CONFIG_DIR", str(tmp_path))

    config = config_loader._load_default_config({"device": "nvidia"})

    assert config["device_marker"] == "nvidia"
    assert (
        config["model_deploy_config"]["llm"]["default"]["vllm"]["tool_call_parser"]
        == "device_parser"
    )


def test_unknown_hardware_uses_explicit_ascend_engine_default_config(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "nvidia_default.json",
        {"device_marker": "nvidia", "model_deploy_config": {"llm": {}}},
    )
    _write_json(
        tmp_path / "ascend_default.json",
        {"device_marker": "ascend", "model_deploy_config": {"llm": {}}},
    )
    monkeypatch.setattr(config_loader, "DEFAULT_CONFIG_DIR", str(tmp_path))

    config = config_loader._load_default_config(
        {"device": "unknown"},
        {"engine": "vllm_ascend"},
    )

    assert config["device_marker"] == "ascend"


def test_engine_fallback_defaults_are_not_loaded_when_model_type_is_missing(monkeypatch):
    assert config_loader.DEFAULT_CONFIG_FILES["nvidia"] == "nvidia_default.json"
    assert config_loader.DEFAULT_CONFIG_FILES["ascend"] == "ascend_default.json"
    assert "sglang" not in config_loader.DEFAULT_CONFIG_FILES
    assert "mindie" not in config_loader.DEFAULT_CONFIG_FILES


def test_engine_fallback_defaults_ignore_removed_engine_default_files(tmp_path, monkeypatch):
    _write_json(tmp_path / "sglang_default.json", {"legacy": "sglang"})
    _write_json(tmp_path / "mindie_default.json", {"legacy": "mindie"})
    monkeypatch.setattr(config_loader, "DEFAULT_CONFIG_DIR", str(tmp_path))

    assert config_loader._load_engine_fallback_defaults("sglang") == {}
    assert config_loader._load_engine_fallback_defaults("mindie") == {}


def test_nvidia_default_contains_function_call_model_defaults():
    config = config_loader._load_default_config({"device": "nvidia"})

    model_defaults = config["model_deploy_config"]["llm"]["DeepseekV3ForCausalLM"]
    assert model_defaults["default"]["vllm"]["tool_call_parser"] == "deepseek_v3"


def test_model_default_templates_do_not_enable_auto_tool_choice():
    for device in ("nvidia", "ascend"):
        assert not _contains_key(_model_deploy_config(device), "enable_auto_tool_choice")


def test_pd_default_templates_do_not_enable_auto_tool_choice():
    pd_path = Path(config_loader.DEFAULT_CONFIG_DIR) / config_loader.DEFAULT_CONFIG_FILES["pd_config"]
    pd_config = json.loads(pd_path.read_text(encoding="utf-8")).get("pd_config", {})

    assert not _contains_key(pd_config, "enable_auto_tool_choice")


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


def test_minimax_pro5000_reasoning_parser_exact_precision_models_are_loaded():
    config_loader._load_reasoning_parser_support.cache_clear()
    hardware = {
        "device": "nvidia",
        "count": 8,
        "details": [{"name": "G6550+RTX PRO 5000 * 8"}],
    }

    m3_config = config_loader._get_model_specific_config(
        hardware,
        {
            "engine": "vllm",
            "model_name": "MiniMax-M3-MXFP8",
            "model_path": "/models/MiniMax-M3-MXFP8",
            "model_type": "llm",
            "enable_auto_think_choice": True,
        },
        _FakeModelInfo("MiniMax-M3-MXFP8", "MiniMaxM3SparseForConditionalGeneration"),
    )
    m3_base_config = config_loader._get_model_specific_config(
        hardware,
        {
            "engine": "vllm",
            "model_name": "MiniMax-M3",
            "model_path": "/models/MiniMax-M3",
            "model_type": "llm",
            "enable_auto_think_choice": True,
        },
        _FakeModelInfo("MiniMax-M3", "MiniMaxM3SparseForConditionalGeneration"),
    )
    m25_config = config_loader._get_model_specific_config(
        hardware,
        {
            "engine": "vllm",
            "model_name": "MiniMax-M2.5-NVFP4",
            "model_path": "/models/MiniMax-M2.5-NVFP4",
            "model_type": "llm",
            "enable_auto_think_choice": True,
        },
        _FakeModelInfo("MiniMax-M2.5-NVFP4", "MiniMaxM2ForCausalLM"),
    )
    m27_config = config_loader._get_model_specific_config(
        hardware,
        {
            "engine": "vllm",
            "model_name": "MiniMax-M2.7-NVFP4",
            "model_path": "/models/MiniMax-M2.7-NVFP4",
            "model_type": "llm",
            "enable_auto_think_choice": True,
        },
        _FakeModelInfo("MiniMax-M2.7-NVFP4", "MiniMaxM2ForCausalLM"),
    )

    assert m3_config["reasoning_parser"] == "minimax_m3"
    assert "reasoning_parser" not in m3_base_config
    assert m25_config["reasoning_parser"] == "minimax_m2"
    assert m27_config["reasoning_parser"] == "minimax_m2"

    found, parser = config_loader._resolve_reasoning_parser_support(
        "MiniMaxM2ForCausalLM",
        "MiniMax-M2.7-w8a8-QuaRot",
        "vllm_ascend",
    )
    assert found is True
    assert parser == "minimax_m2_append_think"


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
        "MAX_NUM_SEQS",
        "MAX_NUM_BATCHED_TOKENS",
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytest",
            "--gpu-memory-utilization",
            "0.8",
            "--max-num-seqs",
            "48",
            "--max-num-batched-tokens",
            "8192",
        ],
    )
    _clear_removed_param_env(monkeypatch)

    params = {
        "gpu_memory_utilization": 0.92,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 16384,
    }
    engine_cmd_parameter = {
        "gpu_memory_utilization": 0.8,
        "max_num_seqs": 48,
        "max_num_batched_tokens": 8192,
    }

    config_loader._set_common_params(params, engine_cmd_parameter, _mapping_path())

    assert params["gpu_memory_utilization"] == 0.8
    assert params["max_num_seqs"] == 48
    assert params["max_num_batched_tokens"] == 8192


def test_vllm_capacity_defaults_reach_vllm_commands(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    _clear_removed_param_env(monkeypatch)
    engine_cmd_parameter = {
        "max_num_seqs": 32,
        "max_num_batched_tokens": 4096,
    }

    for engine in ("vllm", "vllm_ascend"):
        engine_config = {}
        config_loader._set_common_params(
            engine_config,
            engine_cmd_parameter,
            _mapping_path(),
        )
        command = vllm_adapter.build_start_command(
            {
                "engine": engine,
                "model_name": "Demo",
                "model_path": "/models/demo",
                "model_type": "llm",
                "engine_config": engine_config,
            }
        )

        assert "--max-num-seqs 32" in command
        assert "--max-num-batched-tokens 4096" in command


def test_vllm_capacity_model_defaults_are_preserved(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    _clear_removed_param_env(monkeypatch)
    engine_config = {
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
    }

    config_loader._set_common_params(
        engine_config,
        {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 4096,
        },
        _mapping_path(),
    )

    assert engine_config["max_num_seqs"] == 64
    assert engine_config["max_num_batched_tokens"] == 8192


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


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
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
    assert "DeepSeek-V4-Flash-A2" not in deepseek_v4
    assert "DeepSeek-V4-Flash-A3" not in deepseek_v4
    assert "DeepSeek-V4-Flash-Ascend910B" not in deepseek_v4
    assert "DeepSeek-V4-Flash-Ascend910C" not in deepseek_v4
    assert (
        deepseek_v4["DeepSeek-V4-Flash-w8a8-mtp-Ascend910B"]["vllm_ascend"]["no_enable_prefix_caching"]
        is True
    )
    assert "enable_prefix_caching" not in deepseek_v4["DeepSeek-V4-Flash-w8a8-mtp-Ascend910B"]["vllm_ascend"]
    assert "enable_prefix_caching" not in deepseek_v4["DeepSeek-V4-Flash-w8a8-mtp-Ascend910C"]["vllm_ascend"]
    assert "no_enable_prefix_caching" not in deepseek_v4["DeepSeek-V4-Flash-w8a8-mtp-Ascend910C"]["vllm_ascend"]
    _assert_engine_fields(
        deepseek_v4["DeepSeek-V4-Pro-w4a8-mtp"]["vllm_ascend_distributed"],
        {
            "max_num_seqs": 32,
            "served_model_name": "dsv4",
            "tool_call_parser": "deepseek_v4",
        },
    )
    deepseek_v4_pro = deepseek_v4["DeepSeek-V4-Pro-w4a8-mtp"]["vllm_ascend_distributed"]
    assert "trust_remote_code" not in deepseek_v4_pro
    assert "enable_chunked_prefill" not in deepseek_v4_pro
    assert "enable_prefix_caching" not in deepseek_v4_pro
    assert _as_dict(
        deepseek_v4["DeepSeek-V4-Pro-w4a8-mtp"]["vllm_ascend_distributed"]["additional_config"]
    )["multistream_overlap_shared_expert"] is True

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
        assert model_name not in qwen35_dense

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


def test_qwen_day0_910b_reuse_defaults_are_independent_copies():
    llm = _model_deploy_config("ascend")["llm"]

    dense = llm["Qwen3_5ForConditionalGeneration"]
    assert dense["Qwen3.6-27B-Ascend910B"] == dense["Qwen3.6-27B-Ascend910C"]

    moe = llm["Qwen3_5MoeForConditionalGeneration"]
    for model_name in (
        "Qwen3.5-35B-A3B",
        "Qwen3.5-122B-A10B",
        "Qwen3.6-35B-A3B",
    ):
        assert moe[f"{model_name}-Ascend910B"] == moe[f"{model_name}-Ascend910C"]

    assert "Qwen3.5-397B-A17B-w8a8-mtp-Ascend910B" in moe
    assert "Qwen3.5-397B-A17B-w8a8-mtp-Ascend910C" in moe
    assert (
        moe["Qwen3.5-397B-A17B-w8a8-mtp-Ascend910B"]
        != moe["Qwen3.5-397B-A17B-w8a8-mtp-Ascend910C"]
    )


def test_qwen35_397b_w8a8_mtp_ascend_defaults_match_day0_scripts():
    moe = _model_deploy_config("ascend")["llm"]["Qwen3_5MoeForConditionalGeneration"]

    expected_by_card = {
        "Ascend910C": 262144,
        "Ascend910B": 131072,
    }
    expected_additional = {
        "enable_cpu_binding": True,
        "ascend_compilation_config": {"enable_npugraph_ex": True},
    }
    for card_name, expected_len in expected_by_card.items():
        config = moe[f"Qwen3.5-397B-A17B-w8a8-mtp-{card_name}"]
        for engine in ("vllm_ascend", "vllm_ascend_distributed"):
            engine_config = config[engine]
            assert engine_config["use_vllm_serve"] is True
            assert engine_config["max_model_len"] == expected_len
            assert engine_config["max_num_seqs"] == 32
            assert engine_config["max_num_batched_tokens"] == 8192
            assert engine_config["gpu_memory_utilization"] == 0.95
            assert engine_config["enable_expert_parallel"] is True
            assert engine_config["async_scheduling"] is True
            assert engine_config["compilation_config"] == {"cudagraph_mode": "FULL_DECODE_ONLY"}
            assert engine_config["additional_config"] == expected_additional
            if card_name == "Ascend910B":
                # 910B3 Day0 397B-w8a8-mtp 脚本不带 --language-model-only，避免生成命令额外注入。
                assert "language_model_only" not in engine_config
            else:
                assert engine_config["language_model_only"] is True
            # TP/DP 属于运行时拓扑，不能重新固化到 defaults。
            assert "tensor_parallel_size" not in engine_config
            assert "data_parallel_size" not in engine_config


def test_qwen35_35b_a3b_ascend_defaults_match_day0_script():
    moe = _model_deploy_config("ascend")["llm"]["Qwen3_5MoeForConditionalGeneration"]
    expected_additional = {
        "enable_cpu_binding": False,
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": False,
        },
        "multistream_overlap_shared_expert": False,
    }

    for card_name in ("Ascend910C", "Ascend910B"):
        config = moe[f"Qwen3.5-35B-A3B-{card_name}"]
        for engine in ("vllm_ascend", "vllm_ascend_distributed"):
            engine_config = config[engine]
            assert engine_config["use_vllm_serve"] is True
            assert engine_config["max_model_len"] == 161072
            assert engine_config["max_num_seqs"] == 32
            assert engine_config["max_num_batched_tokens"] == 8192
            assert engine_config["gpu_memory_utilization"] == 0.9
            assert engine_config["enable_expert_parallel"] is True
            assert engine_config["language_model_only"] is True
            assert engine_config["compilation_config"] == {"cudagraph_mode": "FULL_DECODE_ONLY"}
            assert engine_config["additional_config"] == expected_additional
            assert "async_scheduling" not in engine_config
            # TP/DP 属于运行时拓扑，不能重新固化到 defaults。
            assert "tensor_parallel_size" not in engine_config
            assert "data_parallel_size" not in engine_config


def test_qwen35_397b_w8a8_mtp_uses_whitelist_driven_ascend_profile():
    moe = _model_deploy_config("ascend")["llm"]["Qwen3_5MoeForConditionalGeneration"]
    scenario = config_loader._SpecialEngineScenario()
    model_info = _FakeModelInfo(
        "Qwen/Qwen3.5-397B-A17B-w8a8-mtp",
        "Qwen3_5MoeForConditionalGeneration",
    )

    cases = [
        ("Ascend910C", 262144),
        ("Ascend910B_64G", 131072),
    ]
    for card_name, expected_len in cases:
        config = config_loader._match_model_engine_config(
            moe,
            "qwen/qwen3.5-397b-a17b-w8a8-mtp",
            "vllm_ascend",
            scenario,
            model_info,
            {"device": "ascend", "details": [{"name": card_name}]},
            "/models/qwen/qwen3.5-397b-a17b-w8a8-mtp",
        )

        assert config["max_model_len"] == expected_len
        assert config["max_num_seqs"] == 32
        assert config["max_num_batched_tokens"] == 8192


def test_qwen35_day0_single_node_topology_uses_generic_tp_rule(monkeypatch):
    monkeypatch.setattr(
        config_loader,
        "check_pcie_cards",
        lambda *_args: (False, None),
    )

    cases = [
        ("Qwen/Qwen3.5-35B-A3B", 2),
        ("Qwen/Qwen3.5-397B-A17B-w8a8-mtp", 8),
    ]
    for model_name, device_count in cases:
        model_info = _FakeModelInfo(
            model_name,
            "Qwen3_5MoeForConditionalGeneration",
        )
        params = {
            "engine": "vllm_ascend",
            "distributed": False,
            "device_count": device_count,
            "nnodes": 1,
            "model_name": model_name,
            "model_path": f"/models/{model_name.rsplit('/', 1)[-1]}",
        }
        config = config_loader._get_model_specific_config(
            {"device": "ascend", "details": [{"name": "Ascend910C"}]},
            params,
            model_info,
        )

        # 这两类 Qwen 单机场景的 TP 与卡数一致，复用 _adjust_tensor_parallelism；
        # DP=1 交给 vLLM 默认值，不为等价场景额外增加模型专属分支。
        assert config["tensor_parallel_size"] == device_count
        assert "data_parallel_size" not in config


def test_deepseek_v4_pro_parallelism_depends_on_device_count(monkeypatch):
    config = (
        _model_deploy_config("ascend")["llm"]["DeepseekV4ForCausalLM"]
        ["DeepSeek-V4-Pro-w4a8-mtp"]["vllm_ascend_distributed"]
    )
    for key in (
        "tensor_parallel_size",
        "data_parallel_size",
        "data_parallel_size_local",
    ):
        assert key not in config

    monkeypatch.setattr(
        vllm_adapter,
        "is_deepseek_v4_pro_adapted_scope",
        lambda _params: True,
    )
    for device_count in (8, 16):
        engine_config = dict(config)
        params = {
            "engine": "vllm_ascend",
            "distributed": True,
            "device_count": device_count,
            "nnodes": 2,
            "node_rank": 1,
        }
        vllm_adapter._apply_deepseek_v4_pro_engine_defaults(
            params,
            engine_config,
            explicit_keys=set(),
        )

        assert engine_config["tensor_parallel_size"] == device_count
        assert engine_config["data_parallel_size"] == 2
        assert engine_config["data_parallel_size_local"] == 1
        assert engine_config["data_parallel_start_rank"] == 1


def test_deepseek_v4_pro_config_loader_defers_dp_topology_to_adapter(monkeypatch):
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a3")
    model_info = _FakeModelInfo("DeepSeek-V4-Pro-w4a8-mtp", "DeepseekV4ForCausalLM")
    model_info.model_quantize = "w4a8"
    model_info.config = {"_name_or_path": "DeepSeek-V4-Pro-w4a8-mtp"}
    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-V4-Pro-w4a8-mtp",
        "model_path": "/models/DeepSeek-V4-Pro-w4a8-mtp",
        "model_type": "llm",
        "model_quantize": "w4a8",
        "distributed": True,
        "distributed_executor_backend": "external_launcher",
        "device_count": 16,
        "nnodes": 2,
        "node_rank": 1,
        "enable_auto_tool_choice": True,
    }

    config = config_loader._get_model_specific_config(
        {"device": "ascend", "details": [{"name": "Ascend910C"}]},
        params,
        model_info,
    )

    # config_loader 只合并模型默认和显式开关；V4-Pro DP 拓扑交给 adapter
    # 基于本机卡数推导，避免通用分布式公式提前写入 TP=32。
    assert config["served_model_name"] == "dsv4"
    assert config["enable_auto_tool_choice"] is True
    assert "tensor_parallel_size" not in config
    assert "data_parallel_size" not in config

    prepared = vllm_adapter._prepare_engine_config({**params, "engine_config": config})
    assert prepared["tensor_parallel_size"] == 16
    assert prepared["data_parallel_size"] == 2
    assert prepared["data_parallel_size_local"] == 1
    assert prepared["data_parallel_start_rank"] == 1


def test_qwen_day0_defaults_leave_parallelism_to_device_count(monkeypatch):
    llm = _model_deploy_config("ascend")["llm"]
    dense = llm["Qwen3_5ForConditionalGeneration"]
    qwen_architectures = (
        dense,
        llm["Qwen3_5MoeForConditionalGeneration"],
    )

    for architecture in qwen_architectures:
        for model_name, model_config in architecture.items():
            if not model_name.endswith(("-Ascend910B", "-Ascend910C")):
                continue
            for engine in ("vllm_ascend", "vllm_ascend_distributed"):
                assert "tensor_parallel_size" not in model_config[engine]
                assert "data_parallel_size" not in model_config[engine]

    monkeypatch.setattr(
        config_loader,
        "check_pcie_cards",
        lambda *_args: (False, None),
    )
    single_node = dict(dense["Qwen3.6-27B-Ascend910C"]["vllm_ascend"])
    config_loader._set_parallelism_params(
        single_node,
        {
            "engine": "vllm_ascend",
            "distributed": False,
            "device_count": 6,
            "nnodes": 1,
            "model_name": "Qwen3.6-27B",
            "model_path": "/models/Qwen3.6-27B",
        },
    )
    assert single_node["tensor_parallel_size"] == 6
    assert "data_parallel_size" not in single_node

    distributed = dict(dense["Qwen3.6-27B-Ascend910C"]["vllm_ascend_distributed"])
    config_loader._set_parallelism_params(
        distributed,
        {
            "engine": "vllm_ascend",
            "distributed": True,
            "device_count": 4,
            "nnodes": 2,
            "node_ips": "10.0.0.1,10.0.0.2",
            "model_name": "Qwen3.6-27B",
            "model_path": "/models/Qwen3.6-27B",
        },
    )
    assert distributed["tensor_parallel_size"] == 8
    assert "data_parallel_size" not in distributed


def test_qwen_day0_additional_config_matches_excel_baseline():
    llm = _model_deploy_config("ascend")["llm"]
    dense = llm["Qwen3_5ForConditionalGeneration"]
    moe = llm["Qwen3_5MoeForConditionalGeneration"]

    qwen35_expected = {
        "enable_cpu_binding": False,
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": False,
        },
        "multistream_overlap_shared_expert": False,
    }
    qwen36_expected = {
        "enable_cpu_binding": True,
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
        },
    }

    expected_by_model = {
        **{
            model_name: qwen35_expected
            for model_name in (
                "Qwen3.5-27B-Ascend910C",
                "Qwen3.5-27B-Ascend910B",
            )
        },
        **{
            model_name: qwen36_expected
            for model_name in (
                "Qwen3.6-27B-Ascend910C",
                "Qwen3.6-27B-Ascend910B",
                "Qwen3.6-27B-w8a8-Ascend910C",
                "Qwen3.6-27B-w8a8-Ascend910B",
            )
        },
    }
    for model_name, expected in expected_by_model.items():
        for engine in ("vllm_ascend", "vllm_ascend_distributed"):
            assert _as_dict(dense[model_name][engine]["additional_config"]) == expected

    expected_by_model = {
        **{
            model_name: qwen35_expected
            for model_name in (
                "Qwen3.5-35B-A3B-Ascend910C",
                "Qwen3.5-35B-A3B-Ascend910B",
                "Qwen3.5-122B-A10B-Ascend910C",
                "Qwen3.5-122B-A10B-Ascend910B",
            )
        },
        **{
            model_name: qwen36_expected
            for model_name in (
                "Qwen3.6-35B-A3B-Ascend910C",
                "Qwen3.6-35B-A3B-Ascend910B",
                "Qwen3.6-35B-A3B-w8a8-Ascend910C",
                "Qwen3.6-35B-A3B-w8a8-Ascend910B",
            )
        },
    }
    for model_name, expected in expected_by_model.items():
        for engine in ("vllm_ascend", "vllm_ascend_distributed"):
            assert _as_dict(moe[model_name][engine]["additional_config"]) == expected

    qwen397_expected = {
        "enable_cpu_binding": True,
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
        },
    }
    for model_name in (
        "Qwen3.5-397B-A17B-w8a8-mtp-Ascend910C",
        "Qwen3.5-397B-A17B-w8a8-mtp-Ascend910B",
    ):
        for engine in ("vllm_ascend", "vllm_ascend_distributed"):
            assert _as_dict(moe[model_name][engine]["additional_config"]) == qwen397_expected


def test_qwen35_day0_defaults_keep_language_model_only_from_excel():
    llm = _model_deploy_config("ascend")["llm"]
    dense = llm["Qwen3_5ForConditionalGeneration"]
    moe = llm["Qwen3_5MoeForConditionalGeneration"]

    qwen35_models = [
        dense["Qwen3.5-27B-Ascend910C"],
        dense["Qwen3.5-27B-Ascend910B"],
        moe["Qwen3.5-35B-A3B-Ascend910C"],
        moe["Qwen3.5-35B-A3B-Ascend910B"],
        moe["Qwen3.5-122B-A10B-Ascend910C"],
        moe["Qwen3.5-122B-A10B-Ascend910B"],
        moe["Qwen3.5-397B-A17B-w8a8-mtp-Ascend910C"],
    ]
    for config in qwen35_models:
        for engine in ("vllm_ascend", "vllm_ascend_distributed"):
            assert config[engine]["language_model_only"] is True

    qwen397_910b = moe["Qwen3.5-397B-A17B-w8a8-mtp-Ascend910B"]
    for engine in ("vllm_ascend", "vllm_ascend_distributed"):
        # 910B3 397B-w8a8-mtp 原生脚本不带该参数，保持 exact profile 不额外输出。
        assert "language_model_only" not in qwen397_910b[engine]

    qwen36_models = [
        dense["Qwen3.6-27B-Ascend910C"],
        dense["Qwen3.6-27B-Ascend910B"],
        dense["Qwen3.6-27B-w8a8-Ascend910C"],
        dense["Qwen3.6-27B-w8a8-Ascend910B"],
        moe["Qwen3.6-35B-A3B-Ascend910C"],
        moe["Qwen3.6-35B-A3B-Ascend910B"],
        moe["Qwen3.6-35B-A3B-w8a8-Ascend910C"],
        moe["Qwen3.6-35B-A3B-w8a8-Ascend910B"],
    ]
    for config in qwen36_models:
        for engine in ("vllm_ascend", "vllm_ascend_distributed"):
            assert "language_model_only" not in config[engine]


def test_kimi_k27_code_ascend_defaults_follow_official_memcache_recipe():
    llm = _model_deploy_config("ascend")["llm"]
    kimi = llm["KimiK25ForConditionalGeneration"]["Kimi-K2.7-Code"]

    for engine in ("vllm_ascend", "vllm_ascend_distributed"):
        config = kimi[engine]
        assert config["use_vllm_serve"] is True
        assert config["max_model_len"] == 81920
        assert config["max_num_seqs"] == 48
        assert config["max_num_batched_tokens"] == 4096
        assert config["gpu_memory_utilization"] == 0.9
        assert config["quantization"] == "ascend"
        assert config["async_scheduling"] is True
        assert "enable_auto_tool_choice" not in config
        assert config["tool_call_parser"] == "kimi_k2"
        assert "seed" not in config
        assert "allowed-local-media-path" not in config
        assert "no_enable_prefix_caching" not in config
        assert "mm_encoder_tp_mode" not in config
        assert "tensor_parallel_size" not in config
        assert "data_parallel_size" not in config
        additional_config = _as_dict(config["additional_config"])
        assert additional_config == {
            "enable_npugraph_ex": True,
            "fuse_muls_add": True,
            "multistream_overlap_shared_expert": True,
        }
        assert _as_dict(config["compilation_config"]) == {
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }


def test_kimi_k26_ascend_defaults_follow_memcache_dflash_recipe():
    llm = _model_deploy_config("ascend")["llm"]
    kimi = llm["KimiK25ForConditionalGeneration"]["Kimi-K2.6-W4A8"]

    for engine in ("vllm_ascend", "vllm_ascend_distributed"):
        config = kimi[engine]
        assert config["max_model_len"] == 32768
        assert config["max_num_seqs"] == 4
        assert config["max_num_batched_tokens"] == 16384
        assert config["gpu_memory_utilization"] == 0.9
        assert config["seed"] == 42
        assert config["tool_call_parser"] == "kimi_k2"
        assert "tensor_parallel_size" not in config
        assert "data_parallel_size" not in config
        assert "async_scheduling" not in config
        assert "additional_config" not in config
        assert _as_dict(config["compilation_config"]) == {
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }


def test_kimi_ascend_exact_defaults_match_document_model_names():
    kimi_arch = _model_deploy_config("ascend")["llm"]["KimiK25ForConditionalGeneration"]
    scenario = config_loader._SpecialEngineScenario()
    hardware = {"device": "ascend", "details": [{"name": "Ascend910C"}]}

    kimi26 = config_loader._match_model_engine_config(
        kimi_arch,
        "eco-tech/kimi-k2.6-w4a8",
        "vllm_ascend",
        scenario,
        _FakeModelInfo("Eco-Tech/Kimi-K2.6-W4A8", "KimiK25ForConditionalGeneration"),
        hardware,
    )
    assert kimi26["max_model_len"] == 32768
    assert kimi26["max_num_seqs"] == 4

    kimi27_code = config_loader._match_model_engine_config(
        kimi_arch,
        "kimi-k2.7-code",
        "vllm_ascend",
        scenario,
        _FakeModelInfo("Kimi-K2.7-Code", "KimiK25ForConditionalGeneration"),
        hardware,
    )
    assert kimi27_code["max_model_len"] == 81920
    assert kimi27_code["max_num_seqs"] == 48


def test_glm47_ascend_exact_defaults_use_card_specific_cudagraph_sizes():
    glm_arch = _model_deploy_config("ascend")["llm"]["Glm4MoeForCausalLM"]
    assert "card_tokens" not in glm_arch["GLM-4.7-W8A8-floatmtp-Ascend910B"]
    assert "card_tokens" not in glm_arch["GLM-4.7-W8A8-floatmtp-Ascend910C"]

    scenario = config_loader._SpecialEngineScenario()
    model_info = _FakeModelInfo(
        "Eco-Tech/GLM-4.7-W8A8-floatmtp",
        "Glm4MoeForCausalLM",
    )

    cases = [
        ("Ascend910B_64G", [1, 2, 4, 8, 16, 32]),
        ("Ascend910C", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
    ]
    for card_name, expected_sizes in cases:
        config = config_loader._match_model_engine_config(
            glm_arch,
            "eco-tech/glm-4.7-w8a8-floatmtp",
            "vllm_ascend",
            scenario,
            model_info,
            {"device": "ascend", "details": [{"name": card_name}]},
            "/models/eco-tech/glm-4.7-w8a8-floatmtp",
        )
        assert config["use_vllm_serve"] is True
        assert config["compilation_config"]["cudagraph_capture_sizes"] == expected_sizes
        # TP/DP 属于运行时拓扑，不写进 defaults；910C 单机 16 卡由 adapter recipe 接管。
        assert "tensor_parallel_size" not in config
        assert "data_parallel_size" not in config


def test_minimax_m27_quarot_ascend_defaults_use_card_specific_profiles():
    minimax_arch = _model_deploy_config("ascend")["llm"]["MiniMaxM2ForCausalLM"]
    scenario = config_loader._SpecialEngineScenario()
    model_info = _FakeModelInfo(
        "MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "MiniMaxM2ForCausalLM",
    )

    cases = ["Ascend910C", "Ascend910B_64G"]
    for card_name in cases:
        config = config_loader._match_model_engine_config(
            minimax_arch,
            "minimax/minimax-m2.7-w8a8-quarot",
            "vllm_ascend",
            scenario,
            model_info,
            {"device": "ascend", "details": [{"name": card_name}]},
            "/models/minimax/minimax-m2.7-w8a8-quarot",
        )

        assert config["use_vllm_serve"] is True
        assert config["trust_remote_code"] is True
        assert config["quantization"] == "ascend"
        assert "load_format" not in config
        assert config["async_scheduling"] is True
        assert config["no_enable_prefix_caching"] is True
        assert config["enable_expert_parallel"] is True
        # TP/DP 属于运行时拓扑，不写进 defaults；910C 单机 16 卡由 adapter recipe 接管。
        assert "tensor_parallel_size" not in config
        assert "data_parallel_size" not in config
        assert config["max_num_seqs"] == 48
        assert config["max_model_len"] == 40690
        assert config["max_num_batched_tokens"] == 16384
        assert config["gpu_memory_utilization"] == 0.9
        assert "served_model_name" not in config
        assert config["tool_call_parser"] == "minimax_m2"
        assert _as_dict(config["additional_config"]) == {
            "enable_cpu_binding": True,
            "enable_fused_mc2": True,
            "enable_flashcomm1": True,
            "weight_nz_mode": True,
        }
        assert "enforce_eager" not in config
        assert "kv_cache_memory_bytes" not in config
        assert "speculative_config" not in config


def test_ascend910c_single_node_day0_topology_is_adapter_owned(monkeypatch):
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a3")
    monkeypatch.setattr(
        config_loader,
        "check_pcie_cards",
        lambda *_args: (False, None),
    )

    cases = [
        ("GLM-4.7-W8A8-floatmtp", 8, 2),
        ("Kimi-K2.6-w4a8", 4, 4),
        ("MiniMax-M2.7-w8a8-QuaRot", 8, 2),
    ]
    for model_name, expected_tp, expected_dp in cases:
        params = {
            "engine": "vllm_ascend",
            "distributed": False,
            "device_count": 16,
            "nnodes": 1,
            "model_name": model_name,
            "model_path": f"/models/{model_name}",
            "engine_config": {},
            "_explicit_cli_keys": set(),
        }
        # config_loader 复用既有单机 TP=device_count 规则；最终由 adapter 的
        # 非显式 recipe 覆盖成 DAY0 标准命令要求的 TP/DP。
        config_loader._set_parallelism_params(params["engine_config"], params)
        assert params["engine_config"]["tensor_parallel_size"] == 16
        prepared = vllm_adapter._prepare_engine_config(params)

        assert prepared["tensor_parallel_size"] == expected_tp
        assert prepared["data_parallel_size"] == expected_dp
        assert params["engine_config"]["tensor_parallel_size"] == expected_tp
        assert params["engine_config"]["data_parallel_size"] == expected_dp


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
        "Qwen3_5MoeForConditionalGeneration": ("default",),
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
        "enable_expert_parallel": True,
    }
    _assert_engine_fields(deepseek_flash["vllm"]["default"], expected_deepseek_flash)
    _assert_engine_fields(deepseek_flash["vllm_distributed"], expected_deepseek_flash)
    assert "gpu_memory_utilization" not in deepseek_flash["vllm"]["default"]
    assert deepseek_flash["vllm_distributed"]["gpu_memory_utilization"] == 0.92
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


def test_nvidia_day0_exact_defaults_live_in_nvidia_default_json():
    config = _model_deploy_config("nvidia")
    llm = config["llm"]

    deepseek_v3 = llm["DeepseekV3ForCausalLM"]
    for model_key in ("DeepSeek-R1", "DeepSeek-V3.1"):
        assert deepseek_v3[f"{model_key}-H20-96G"]["card_tokens"] == ["h20-96"]
        assert deepseek_v3[f"{model_key}-H20-96G"]["sglang"]["mem_fraction_static"] == 0.9
        assert deepseek_v3[f"{model_key}-H20-141G"]["card_tokens"] == ["h20-141"]
        assert deepseek_v3[f"{model_key}-H20-141G"]["sglang"]["dp"] == 8
    assert llm["Glm4MoeForCausalLM"]["GLM-4.7"]["card_tokens"] == ["h20-96", "h20-141"]
    assert llm["Glm4MoeForCausalLM"]["GLM-4.7-FP8"]["card_tokens"] == ["h20-96", "h20-141"]
    assert llm["Glm4MoeForCausalLM"]["GLM4.7"]["card_tokens"] == ["h20-96", "h20-141"]
    assert llm["GlmMoeDsaForCausalLM"]["GLM-5"]["card_tokens"] == ["h20-96", "h20-141"]
    assert llm["GlmMoeDsaForCausalLM"]["GLM-5-FP8"]["card_tokens"] == ["h20-96", "h20-141"]
    assert llm["GlmMoeDsaForCausalLM"]["GLM5.1"]["card_tokens"] == ["h20-96", "h20-141"]
    assert llm["GlmMoeDsaForCausalLM"]["GLM-5.1"]["card_tokens"] == ["h20-96", "h20-141"]
    assert llm["GlmMoeDsaForCausalLM"]["GLM-5.1-FP8"]["card_tokens"] == ["h20-96", "h20-141"]
    assert llm["Qwen3_5ForConditionalGeneration"]["Qwen3.6-27B"]["card_tokens"] == [
        "l20",
        "h20-96",
        "h20-141",
    ]
    assert llm["Qwen3_5ForConditionalGeneration"]["Qwen3.5-27B"]["card_tokens"] == [
        "rtxpro5000-72",
    ]
    assert llm["Qwen3_5MoeForConditionalGeneration"]["Qwen3.5-122B-A10B"]["card_tokens"] == [
        "rtxpro5000-72",
    ]
    assert llm["Qwen3_5MoeForConditionalGeneration"]["Qwen3.5-35B-A3B"]["card_tokens"] == [
        "rtxpro5000-72",
    ]
    assert llm["Qwen3_5MoeForConditionalGeneration"]["Qwen3.6-35B-A3B"]["card_tokens"] == [
        "l20",
        "h20-96",
        "h20-141",
    ]
    assert llm["MiniMaxM2ForCausalLM"]["MiniMax-M2.5-NVFP4"]["card_tokens"] == [
        "rtxpro5000-72",
    ]
    assert llm["MiniMaxM2ForCausalLM"]["MiniMax-M2.7"]["card_tokens"] == [
        "rtxpro5000-72",
    ]
    assert llm["MiniMaxM3SparseForConditionalGeneration"]["MiniMax-M3-MXFP8"]["card_tokens"] == [
        "rtxpro5000-72",
    ]

    glm5 = llm["GlmMoeDsaForCausalLM"]["GLM-5"]["vllm"]
    assert glm5["use_vllm_serve"] is True
    assert glm5["chat_template_content_format"] == "string"
    assert glm5["extra_cli_args"] == ["-cc.pass_config.fuse_allreduce_rms=False"]
    assert "max_model_len" not in glm5
    assert "enable_expert_parallel" not in glm5

    glm47 = llm["Glm4MoeForCausalLM"]["GLM4.7"]["vllm"]
    assert glm47["tool_call_parser"] == "glm47"
    assert "max_model_len" not in glm47

    qwen27 = llm["Qwen3_5ForConditionalGeneration"]["Qwen3.6-27B"]["vllm"]
    assert "kv_cache_dtype" not in qwen27
    assert "calculate_kv_scales" not in qwen27
    assert "enable_expert_parallel" not in qwen27

    qwen35_27 = llm["Qwen3_5ForConditionalGeneration"]["Qwen3.5-27B"]["vllm"]
    assert qwen35_27["max_model_len"] == 65536
    assert qwen35_27["enable_prefix_caching"] is True
    assert qwen35_27["tool_call_parser"] == "qwen3_coder"
    assert "speculative_config" not in qwen35_27

    qwen35_122 = llm["Qwen3_5MoeForConditionalGeneration"]["Qwen3.5-122B-A10B"]["vllm"]
    assert qwen35_122["max_model_len"] == 65536
    assert qwen35_122["enable_expert_parallel"] is True
    assert qwen35_122["enable_prefix_caching"] is True
    assert qwen35_122["tool_call_parser"] == "qwen3_coder"
    assert "speculative_config" not in qwen35_122

    qwen35_35 = llm["Qwen3_5MoeForConditionalGeneration"]["Qwen3.5-35B-A3B"]["vllm"]
    assert qwen35_35["max_model_len"] == 65536
    assert qwen35_35["enable_expert_parallel"] is True
    assert qwen35_35["enable_prefix_caching"] is True
    assert qwen35_35["tool_call_parser"] == "qwen3_coder"
    assert "speculative_config" not in qwen35_35

    qwen35 = llm["Qwen3_5MoeForConditionalGeneration"]["Qwen3.6-35B-A3B"]["vllm"]
    assert qwen35["tool_call_parser"] == "qwen3_xml"
    assert "kv_cache_dtype" not in qwen35
    assert "calculate_kv_scales" not in qwen35

    qwen_agent = llm["Qwen3_5MoeForConditionalGeneration"]["Qwen-AgentWorld-35B-A3B"]
    for engine in ("vllm", "vllm_distributed"):
        engine_config = qwen_agent[engine]
        assert engine_config["use_vllm_serve"] is True
        assert engine_config["max_model_len"] == 262144
        assert engine_config["max_num_batched_tokens"] == 1024
        assert "tensor_parallel_size" not in engine_config
        assert "enable_expert_parallel" not in engine_config
        assert engine_config["enable_prefix_caching"] is True
        assert engine_config["language_model_only"] is True
        assert "served_model_name" not in engine_config
        assert engine_config["model_loader_extra_config"] == {
            "enable_multithread_load": "true",
            "num_threads": 128,
        }
        assert "speculative_config" not in engine_config

    minimax_m25 = llm["MiniMaxM2ForCausalLM"]["MiniMax-M2.5-NVFP4"]["vllm"]
    assert minimax_m25["max_model_len"] == 196608
    assert "tensor_parallel_size" not in minimax_m25
    assert minimax_m25["gpu_memory_utilization"] == 0.9
    assert minimax_m25["kv_cache_dtype"] == "fp8"
    assert minimax_m25["tool_call_parser"] == "minimax_m2"
    assert "speculative_config" not in minimax_m25

    minimax_m27 = llm["MiniMaxM2ForCausalLM"]["MiniMax-M2.7"]["vllm"]
    assert minimax_m27["kv_cache_dtype"] == "fp8"
    assert minimax_m27["moe_backend"] == "flashinfer_cutlass"
    assert "rtx_pro_5000_72G" not in minimax_m27
    assert "speculative_config" not in minimax_m27

    minimax_m3 = llm["MiniMaxM3SparseForConditionalGeneration"]["MiniMax-M3-MXFP8"]["vllm"]
    assert minimax_m3["max_model_len"] == 80000
    assert "tensor_parallel_size" not in minimax_m3
    assert minimax_m3["gpu_memory_utilization"] == 0.95
    assert minimax_m3["tool_call_parser"] == "minimax_m3"
    assert minimax_m3["model_loader_extra_config"] == {
        "enable_multithread_load": "true",
        "num_threads": 128,
    }
    assert "speculative_config" not in minimax_m3

    embedding = config["embedding"]
    assert embedding["Qwen3ForCausalLM"]["Qwen3-Embedding-0.6B"]["card_tokens"] == [
        "l20",
        "h20-96",
        "h20-141",
    ]
    assert embedding["BertModel"]["bge-large-zh-v1.5"]["card_tokens"] == [
        "l20",
        "h20-96",
        "h20-141",
    ]
    qwen_embedding = embedding["Qwen3ForCausalLM"]["Qwen3-Embedding-0.6B"]["vllm"]
    assert "kv_cache_dtype" not in qwen_embedding
    assert "calculate_kv_scales" not in qwen_embedding
    assert embedding["BertModel"]["bge-large-zh-v1.5"]["vllm"]["gpu_memory_utilization"] == 0.9

    rerank = config["rerank"]
    assert rerank["XLMRobertaForSequenceClassification"]["bge-reranker-large"]["card_tokens"] == [
        "l20",
        "h20-96",
        "h20-141",
    ]
    assert (
        rerank["XLMRobertaForSequenceClassification"]["bge-reranker-large"]["vllm"]
        ["gpu_memory_utilization"]
        == 0.9
    )


def test_nvidia_day0_exact_defaults_require_matching_card_token():
    config = _model_deploy_config("nvidia")
    scenario = config_loader._SpecialEngineScenario()
    a100 = {
        "device": "nvidia",
        "details": [{"name": "NVIDIA A100 80GB", "total_memory": 80}],
    }
    l20 = {
        "device": "nvidia",
        "details": [{"name": "NVIDIA L20 45GB", "total_memory": 45}],
    }
    h20 = {
        "device": "nvidia",
        "details": [{"name": "NVIDIA H20 141GB", "total_memory": 141}],
    }
    pro5000 = {
        "device": "nvidia",
        "details": [{"name": "G6550+RTX PRO 5000 * 8"}],
    }

    glm5_arch = config["llm"]["GlmMoeDsaForCausalLM"]
    assert config_loader._match_model_engine_config(
        glm5_arch,
        "glm-5",
        "vllm",
        scenario,
        _FakeModelInfo("GLM-5", "GlmMoeDsaForCausalLM"),
        a100,
    ) == {}
    assert config_loader._match_model_engine_config(
        glm5_arch,
        "glm-5",
        "vllm",
        scenario,
        _FakeModelInfo("GLM-5", "GlmMoeDsaForCausalLM"),
        l20,
    ) == {}
    glm5_h20 = config_loader._match_model_engine_config(
        glm5_arch,
        "glm-5",
        "vllm",
        scenario,
        _FakeModelInfo("GLM-5", "GlmMoeDsaForCausalLM"),
        h20,
    )
    assert glm5_h20["use_vllm_serve"] is True

    qwen27_arch = config["llm"]["Qwen3_5ForConditionalGeneration"]
    qwen27_a100 = config_loader._match_model_engine_config(
        qwen27_arch,
        "qwen3.6-27b",
        "vllm",
        scenario,
        _FakeModelInfo("Qwen3.6-27B", "Qwen3_5ForConditionalGeneration"),
        a100,
    )
    assert qwen27_a100 == {}
    qwen27_l20 = config_loader._match_model_engine_config(
        qwen27_arch,
        "qwen3.6-27b",
        "vllm",
        scenario,
        _FakeModelInfo("Qwen3.6-27B", "Qwen3_5ForConditionalGeneration"),
        l20,
    )
    assert "kv_cache_dtype" not in qwen27_l20
    assert qwen27_l20["mm_encoder_tp_mode"] == "data"

    qwen35_27_pro5000 = config_loader._match_model_engine_config(
        qwen27_arch,
        "qwen/qwen3.5-27b",
        "vllm",
        scenario,
        _FakeModelInfo("Qwen3.5-27B", "Qwen3_5ForConditionalGeneration"),
        pro5000,
    )
    assert qwen35_27_pro5000["max_model_len"] == 65536
    assert qwen35_27_pro5000["enable_prefix_caching"] is True

    qwen35_moe_arch = config["llm"]["Qwen3_5MoeForConditionalGeneration"]
    qwen35_122_pro5000 = config_loader._match_model_engine_config(
        qwen35_moe_arch,
        "qwen/qwen3.5-122b-a10b",
        "vllm",
        scenario,
        _FakeModelInfo("Qwen3.5-122B-A10B", "Qwen3_5MoeForConditionalGeneration"),
        pro5000,
    )
    assert qwen35_122_pro5000["enable_expert_parallel"] is True
    assert qwen35_122_pro5000["tool_call_parser"] == "qwen3_coder"
    qwen35_35_pro5000 = config_loader._match_model_engine_config(
        qwen35_moe_arch,
        "qwen/qwen3.5-35b-a3b",
        "vllm",
        scenario,
        _FakeModelInfo("Qwen3.5-35B-A3B", "Qwen3_5MoeForConditionalGeneration"),
        pro5000,
    )
    assert qwen35_35_pro5000["enable_expert_parallel"] is True
    assert qwen35_35_pro5000["tool_call_parser"] == "qwen3_coder"

    minimax_m2_arch = config["llm"]["MiniMaxM2ForCausalLM"]
    assert config_loader._match_model_engine_config(
        minimax_m2_arch,
        "minimax/minimax-m2.5-nvfp4",
        "vllm",
        scenario,
        _FakeModelInfo("MiniMax-M2.5-NVFP4", "MiniMaxM2ForCausalLM"),
        a100,
    ) == {}
    minimax_m25_pro5000 = config_loader._match_model_engine_config(
        minimax_m2_arch,
        "minimax/minimax-m2.5-nvfp4",
        "vllm",
        scenario,
        _FakeModelInfo("MiniMax-M2.5-NVFP4", "MiniMaxM2ForCausalLM"),
        pro5000,
    )
    assert minimax_m25_pro5000["max_model_len"] == 196608
    assert "tensor_parallel_size" not in minimax_m25_pro5000

    minimax_m3_arch = config["llm"]["MiniMaxM3SparseForConditionalGeneration"]
    assert config_loader._match_model_engine_config(
        minimax_m3_arch,
        "minimax/minimax-m3-mxfp8",
        "vllm",
        scenario,
        _FakeModelInfo("MiniMax-M3-MXFP8", "MiniMaxM3SparseForConditionalGeneration"),
        a100,
    ) == {}
    minimax_m3_pro5000 = config_loader._match_model_engine_config(
        minimax_m3_arch,
        "minimax/minimax-m3-mxfp8",
        "vllm",
        scenario,
        _FakeModelInfo("MiniMax-M3-MXFP8", "MiniMaxM3SparseForConditionalGeneration"),
        pro5000,
    )
    assert minimax_m3_pro5000["max_model_len"] == 80000
    assert "tensor_parallel_size" not in minimax_m3_pro5000
    assert minimax_m3_pro5000["model_loader_extra_config"] == {
        "enable_multithread_load": "true",
        "num_threads": 128,
    }

    embedding_arch = config["embedding"]["Qwen3ForCausalLM"]
    assert config_loader._match_model_engine_config(
        embedding_arch,
        "qwen3-embedding-0.6b",
        "vllm",
        scenario,
        _FakeModelInfo("Qwen3-Embedding-0.6B", "Qwen3ForCausalLM", "embedding"),
        a100,
    ) == {}

    rerank_arch = config["rerank"]["XLMRobertaForSequenceClassification"]
    assert config_loader._match_model_engine_config(
        rerank_arch,
        "bge-reranker-large",
        "vllm",
        scenario,
        _FakeModelInfo("bge-reranker-large", "XLMRobertaForSequenceClassification", "rerank"),
        a100,
    ) == {}


def test_nvidia_day0_exact_default_replaces_arch_default(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    hardware = {
        "device": "nvidia",
        "count": 8,
        "details": [{"name": "NVIDIA H20 141GB", "total_memory": 141}],
    }
    params = {
        "engine": "vllm",
        "model_name": "GLM-5",
        "model_path": "/models/GLM-5",
        "model_type": "llm",
        "device_count": 8,
        "distributed": False,
        "nnodes": 1,
        "node_ips": "",
        "gpu_usage_mode": "full",
        "distributed_executor_backend": "ray",
        "_smart_feats": [],
        "_smart_card_token": "h20-141",
    }
    model_info = _FakeModelInfo("GLM-5", "GlmMoeDsaForCausalLM")

    defaults = config_loader._get_model_specific_config(hardware, params, model_info)

    assert defaults["use_vllm_serve"] is True
    assert defaults["tensor_parallel_size"] == 8
    assert "tool_call_parser" not in defaults
    assert "enable_auto_tool_choice" not in defaults
    assert "max_model_len" not in defaults
    assert "enable_expert_parallel" not in defaults

    enabled = config_loader._get_model_specific_config(
        hardware,
        {**params, "enable_auto_tool_choice": True},
        model_info,
    )
    assert enabled["tool_call_parser"] == "glm47"
    assert enabled["enable_auto_tool_choice"] is True


def test_qwen35_397b_ascend_distributed_forces_dp_backend_and_node_tp(monkeypatch):
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.delenv("VLLM_DISTRIBUTED_PORT", raising=False)
    distributed_config = {
        "vllm_distributed": {
            "nixl_port": 27070,
            "rpc_port": 27071,
            "ray_head_port": 28020,
        }
    }
    params = {
        "engine": "vllm_ascend",
        "model_name": "Qwen/Qwen3.5-397B-A17B",
        "model_path": "/models/Qwen3.5-397B-A17B",
        "device_count": 16,
        "distributed": True,
        "nnodes": 2,
        "node_ips": "10.0.0.1,10.0.0.2",
        "distributed_executor_backend": "ray",
    }
    model_info = _FakeModelInfo(
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen3_5MoeForConditionalGeneration",
    )

    # 回归契约一：即使上层默认/环境仍给出 ray，397B 系列在 Ascend 分布式场景下
    # 必须复用 dp_deployment 出口。这里直接调用后端分派函数，避免测试依赖完整
    # 启动链路中的硬件探测、默认配置读取等无关因素。
    config_loader._handle_vllm_distributed(distributed_config, params, model_info)

    assert params["distributed_executor_backend"] == "dp_deployment"
    engine_config = {}
    # 回归契约二：397B 新增 DP 路由后不能丢失节点内 TP。DeepSeek/GLM/Kimi 的
    # TP 由 adapter 专属逻辑兜底，397B 没有那层兜底，因此必须在通用并行设置层
    # 保证最终 engine_config 至少包含 tensor_parallel_size=device_count。
    config_loader._set_parallelism_params(engine_config, params)
    assert engine_config["tensor_parallel_size"] == 16


def test_qwen35_moe_non_397b_distributed_still_uses_ray(monkeypatch):
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.delenv("VLLM_DISTRIBUTED_PORT", raising=False)
    distributed_config = {
        "vllm_distributed": {
            "nixl_port": 27070,
            "rpc_port": 27071,
            "ray_head_port": 28020,
        }
    }
    params = {
        "engine": "vllm_ascend",
        "model_name": "Qwen/Qwen3.5-35B-A3B",
        "model_path": "/models/Qwen3.5-35B-A3B",
        "device_count": 16,
        "distributed": True,
        "nnodes": 2,
        "node_ips": "10.0.0.1,10.0.0.2",
        "distributed_executor_backend": "ray",
    }
    model_info = _FakeModelInfo(
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen3_5MoeForConditionalGeneration",
    )

    # 防扩散契约：397B 与 35B/122B/AgentWorld 共享 Qwen3_5Moe 架构，路由特例
    # 必须依赖模型身份而不是 architecture 粗粒度命中；否则会把原本走 Ray 的
    # Qwen MoE 分布式场景一起切到 dp_deployment。
    config_loader._handle_vllm_distributed(distributed_config, params, model_info)

    assert params["distributed_executor_backend"] == "ray"
