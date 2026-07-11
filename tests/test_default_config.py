import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from engines import vllm_adapter  # noqa: E402


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


def test_special_nvidia_config_selector_preserves_all_card_branches():
    config = {
        "sglang": {
            "engine_marker": True,
            "H20-96G": {"selected": "h20"},
        },
        "vllm": {
            "default": {"selected": "default"},
            "rtx_pro_5000_72G": {"selected": "pro5000"},
        },
    }
    sglang_scenario = config_loader._SpecialEngineScenario(
        deepseek_sglang_nvidia=True,
    )
    flash_scenario = config_loader._SpecialEngineScenario(
        deepseek_v4_flash_vllm_nvidia=True,
    )
    minimax_scenario = config_loader._SpecialEngineScenario(
        minimax_m27_vllm_nvidia=True,
    )

    def selection(engine_key, scenario, h20_model="", card_model=""):
        return config_loader._SpecialNvidiaConfigSelection(
            model="model",
            config=config,
            engine_key=engine_key,
            scenario=scenario,
            h20_model=h20_model,
            card_model=card_model,
        )

    assert config_loader._resolve_special_nvidia_engine_config(
        selection("sglang", sglang_scenario, h20_model="H20-96G")
    ) == {"selected": "h20"}
    assert config_loader._resolve_special_nvidia_engine_config(
        selection("sglang", sglang_scenario)
    ) == config["sglang"]
    assert config_loader._resolve_special_nvidia_engine_config(
        selection("vllm", flash_scenario, card_model="rtx_pro_5000_72G")
    ) == {"selected": "pro5000"}
    assert config_loader._resolve_special_nvidia_engine_config(
        selection("vllm", flash_scenario, card_model="other")
    ) == {"selected": "default"}
    assert config_loader._resolve_special_nvidia_engine_config(
        selection("vllm", minimax_scenario, card_model="rtx_pro_5000_72G")
    ) == {"selected": "pro5000"}
    assert config_loader._resolve_special_nvidia_engine_config(
        selection("vllm", minimax_scenario, card_model="other")
    ) == {"selected": "default"}


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
    assert deepseek_v4["DeepSeek-V4-Flash-Ascend910B"]["vllm_ascend"]["no_enable_prefix_caching"] is True
    assert "enable_prefix_caching" not in deepseek_v4["DeepSeek-V4-Flash-Ascend910B"]["vllm_ascend"]
    assert "enable_prefix_caching" not in deepseek_v4["DeepSeek-V4-Flash-Ascend910C"]["vllm_ascend"]
    assert "no_enable_prefix_caching" not in deepseek_v4["DeepSeek-V4-Flash-Ascend910C"]["vllm_ascend"]
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

    assert "Qwen3.5-397B-A17B-Ascend910B" not in moe


def test_deepseek_v4_pro_parallelism_depends_on_device_count(monkeypatch):
    config = (
        _model_deploy_config("ascend")["llm"]["DeepseekV4ForCausalLM"]
        ["DeepSeek-V4-Pro"]["vllm_ascend_distributed"]
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
                "Qwen3.5-397B-A17B-Ascend910C",
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
        moe["Qwen3.5-397B-A17B-Ascend910C"],
    ]
    for config in qwen35_models:
        for engine in ("vllm_ascend", "vllm_ascend_distributed"):
            assert config[engine]["language_model_only"] is True

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
        assert config["max_model_len"] == 81920
        assert config["max_num_seqs"] == 48
        assert config["max_num_batched_tokens"] == 4096
        assert config["gpu_memory_utilization"] == 0.9
        assert config["quantization"] == "ascend"
        assert config["async_scheduling"] is True
        assert "enable_auto_tool_choice" not in config
        assert config["tool_call_parser"] == "kimi_k2"
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


def test_nvidia_day0_exact_defaults_live_in_nvidia_default_json():
    config = _model_deploy_config("nvidia")
    llm = config["llm"]

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
    assert qwen27["kv_cache_dtype"] == "fp8"
    assert "calculate_kv_scales" not in qwen27
    assert "enable_expert_parallel" not in qwen27

    qwen35 = llm["Qwen3_5MoeForConditionalGeneration"]["Qwen3.6-35B-A3B"]["vllm"]
    assert qwen35["tool_call_parser"] == "qwen3_xml"
    assert qwen35["kv_cache_dtype"] == "fp8"
    assert "calculate_kv_scales" not in qwen35

    embedding = config["embedding"]
    qwen_embedding = embedding["Qwen3ForCausalLM"]["Qwen3-Embedding-0.6B"]["vllm"]
    assert qwen_embedding["kv_cache_dtype"] == "fp8"
    assert "calculate_kv_scales" not in qwen_embedding
    assert embedding["BertModel"]["bge-large-zh-v1.5"]["vllm"]["gpu_memory_utilization"] == 0.9

    rerank = config["rerank"]
    assert (
        rerank["XLMRobertaForSequenceClassification"]["bge-reranker-large"]["vllm"]
        ["gpu_memory_utilization"]
        == 0.9
    )


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
