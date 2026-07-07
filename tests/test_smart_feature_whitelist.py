import os
import sys
import json
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from utils import model_utils  # noqa: E402
from utils.device_utils import resolve_card_token  # noqa: E402
from core import config_loader  # noqa: E402
from core.hardware_detect import detect_hardware  # noqa: E402
from core.version_util import resolve_card_model  # noqa: E402
from engines import vllm_adapter  # noqa: E402


class _FakeDeepSeekV4Info:
    model_name = "DeepSeek-V4-Flash-w8a8-mtp"
    model_path = "/usr/local/serving/models/"
    model_architecture = "DeepseekV4ForCausalLM"

    @staticmethod
    def identify_model_architecture():
        return "DeepseekV4ForCausalLM"

    @staticmethod
    def identify_model_type():
        return "llm"


class _FakeDeepSeekV4Identifier:
    model_architecture = "DeepseekV4ForCausalLM"
    model_quantize = "fp4"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


def test_default_smart_feature_whitelist_file_is_loaded():
    assert model_utils._SMART_WHITELIST_PATH.exists()

    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "910c",
    ) == frozenset({"spec", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "910b",
    ) == frozenset({"spec", "offload"})

    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "deepseek-ai/DeepSeek-V4-Flash",
        "/models/deepseek-ai/DeepSeek-V4-Flash",
        "h20-141",
    ) == frozenset({"spec"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "deepseek-ai/DeepSeek-V4-Flash",
        "/models/deepseek-ai/DeepSeek-V4-Flash",
        "rtxpro5000-72",
    ) == frozenset({"spec", "sparse"})

    whitelist = json.loads(model_utils._SMART_WHITELIST_PATH.read_text(encoding="utf-8"))
    deepseek_tokens = [
        token.lower()
        for feature in ("spec", "sparse", "offload")
        for entry in whitelist[feature]
        if entry.get("arch") == "DeepseekV4ForCausalLM"
        for token in entry.get("name_tokens", [])
    ]
    assert "deepseek-ai/deepseek-v4-flash-fp4" not in deepseek_tokens
    assert "deepseek-v4-flash-fp4" not in deepseek_tokens
    nvidia_card_tokens = [
        token.lower()
        for feature in ("spec", "sparse", "offload")
        for entry in whitelist[feature]
        if entry.get("engine") == "vllm"
        for token in entry.get("card_tokens", [])
    ]
    assert "h20-141" in nvidia_card_tokens
    assert "rtxpro5000-72" in nvidia_card_tokens
    for non_chip_token in (
        "nh02",
        "nrp0500",
        "g6550 v8",
        "g8600 v7",
        "rtx pro 5000 72",
        "rtx_pro_5000_72g",
    ):
        assert non_chip_token not in nvidia_card_tokens
    for feature in ("spec", "sparse", "offload"):
        for entry in whitelist[feature]:
            assert len(entry.get("card_tokens", [])) == 1

    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen3.5-397B-A17B-NVFP4",
        "/models/Qwen3.5-397B-A17B-NVFP4",
        "rtxpro5000-72",
    ) == frozenset({"spec", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen3.5-397B-A17B-NVFP4",
        "/models/Qwen3.5-397B-A17B-NVFP4",
        "h20-141",
    ) == frozenset()
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen/Qwen3.5-397B-A17B",
        "/models/Qwen/Qwen3.5-397B-A17B",
        "rtxpro5000-72",
    ) == frozenset({"spec", "sparse"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen3.5-397B-A17B",
        "/models/Qwen3.5-397B-A17B",
        "rtxpro5000-72",
    ) == frozenset({"spec", "sparse"})

    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "GLM-4.7",
        "/models/GLM-4.7",
        "",
    ) == frozenset()
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "ZhipuAI/GLM-4.7-FP8",
        "/models/ZhipuAI/GLM-4.7-FP8",
        "h20-141",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "ZhipuAI/GLM-4.7-FP8",
        "/models/ZhipuAI/GLM-4.7-FP8",
        "rtxpro5000-72",
    ) == frozenset()
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "MiniMax/MiniMax-M2.7",
        "/models/MiniMax/MiniMax-M2.7",
        "rtxpro5000-72",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "ZhipuAI/GLM-5.1-FP8",
        "/models/ZhipuAI/GLM-5.1-FP8",
        "h20-141",
    ) == frozenset({"sparse"})

    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Qwen3.6-27B",
        "/models/Qwen3.6-27B",
        "910c",
    ) == frozenset({"spec"})

    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Qwen3.6-35B-A3B",
        "/models/Qwen3.6-35B-A3B",
        "910c",
    ) == frozenset({"spec"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Eco-Tech/GLM-5.1-w8a8",
        "/models/Eco-Tech/GLM-5.1-w8a8",
        "910b",
    ) == frozenset({"spec", "sparse"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Eco-Tech/GLM-5.1-w8a8",
        "/models/Eco-Tech/GLM-5.1-w8a8",
        "910c",
    ) == frozenset({"spec", "sparse"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Kimi-K2.7-Code",
        "/harbor_data/Kimi-K2.7-Code",
        "910c",
    ) == frozenset({"offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Kimi-K2.7-Code",
        "/harbor_data/Kimi-K2.7-Code",
        "910b",
    ) == frozenset()


def test_deepseek_v4_flash_a3_respects_upper_smart_feature_switches_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "false")
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "false")

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "enable_sparse": False,
        "enable_speculative_decode": False,
    }

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "ascend", "details": [{"name": "Ascend910C"}]},
    )

    assert params["_allowed_smart_feats"] == ["offload", "spec"]
    assert params["_smart_feats"] == []
    assert params["_forced_smart_feats"] == []
    assert params["enable_speculative_decode"] is False
    assert params["enable_sparse"] is False
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "false"
    assert os.environ["ENABLE_KV_OFFLOAD"] == "false"
    assert os.environ["LMCACHE_OFFLOAD"] == "false"


def test_deepseek_v4_flash_kv_transfer_is_not_injected_when_upper_offload_disabled(monkeypatch):
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_VERSION", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "false")

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "distributed": False,
        "enable_speculative_decode": False,
    }
    hardware_env = {"device": "ascend", "details": [{"name": "Ascend910C"}]}

    config_loader.apply_effective_feature_enablement(params, hardware_env)
    config = config_loader._get_model_specific_config(
        hardware_env,
        params,
        _FakeDeepSeekV4Info(),
    )

    assert params["model_name"] == "DeepSeek-V4-Flash-w8a8-mtp"
    assert config["served_model_name"] == "DeepSeek-V4-Flash-w8a8-mtp"
    assert "kv_transfer_config" not in config


def test_deepseek_v4_flash_kv_transfer_reuses_enabled_upper_offload_from_upstream_model_name(monkeypatch):
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_VERSION", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "true")

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "distributed": False,
        "enable_speculative_decode": False,
    }
    hardware_env = {"device": "ascend", "details": [{"name": "Ascend910C"}]}

    config_loader.apply_effective_feature_enablement(params, hardware_env)
    config = config_loader._get_model_specific_config(
        hardware_env,
        params,
        _FakeDeepSeekV4Info(),
    )

    kv_transfer = json.loads(config["kv_transfer_config"])
    assert params["_smart_feats"] == ["offload"]
    assert params["model_name"] == "DeepSeek-V4-Flash-w8a8-mtp"
    assert config["served_model_name"] == "DeepSeek-V4-Flash-w8a8-mtp"
    assert kv_transfer == {
        "kv_connector": "LMCacheAscendConnectorV1Dynamic",
        "kv_role": "kv_both",
        "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
    }


class _FakeKimiK27CodeInfo:
    model_name = "Kimi-K2.7-Code"
    model_path = "/harbor_data/Kimi-K2.7-Code"
    model_architecture = "KimiK25ForConditionalGeneration"

    @staticmethod
    def identify_model_architecture():
        return "KimiK25ForConditionalGeneration"

    @staticmethod
    def identify_model_type():
        return "llm"


def test_kimi_k27_code_uses_memcache_ascend_store_connector(monkeypatch):
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_VERSION", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    params = {
        "engine": "vllm_ascend",
        "model_name": "Kimi-K2.7-Code",
        "model_path": "/harbor_data/Kimi-K2.7-Code",
        "model_type": "llm",
        "distributed": False,
        "enable_speculative_decode": False,
        "device_count": 16,
    }
    hardware_env = {"device": "ascend", "details": [{"name": "Ascend910C"}]}

    config_loader.apply_effective_feature_enablement(params, hardware_env)
    config = config_loader._get_model_specific_config(
        hardware_env,
        params,
        _FakeKimiK27CodeInfo(),
    )

    kv_transfer = json.loads(config["kv_transfer_config"])
    assert params["_smart_feats"] == ["offload"]
    assert kv_transfer == {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "lookup_rpc_port": "0",
            "backend": "memcache",
        },
    }


def test_deepseek_v4_flash_auto_floor_skips_lmcache_connector(monkeypatch):
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_VERSION", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "102400")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "distributed": False,
        "enable_speculative_decode": False,
        "device_count": 8,
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
    }
    hardware_env = {"device": "ascend", "details": [{"name": "Ascend910C"}]}

    config_loader.apply_effective_feature_enablement(params, hardware_env)
    config = config_loader._get_model_specific_config(
        hardware_env,
        params,
        _FakeDeepSeekV4Info(),
    )

    assert params["_smart_feats"] == ["offload"]
    assert "kv_transfer_config" not in config


def test_generic_ascend_detail_name_falls_back_to_hardware_family(monkeypatch):
    monkeypatch.setenv("ENABLE_SPARSE", "true")
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    hardware_env = {
        "device": "ascend",
        "count": 1,
        "details": [{"name": "Ascend"}],
        "hardware_family": "Ascend910B_64G",
    }
    params = {
        "engine": "vllm_ascend",
        "model_name": "GLM-5.1-w8a8",
        "model_path": "/usr/local/serving/models/",
        "enable_sparse": True,
        "enable_speculative_decode": True,
    }

    assert resolve_card_token(hardware_env) == "ascend910b_64g"

    config_loader.apply_effective_feature_enablement(params, hardware_env)

    assert params["_smart_card_token"] == "ascend910b_64g"
    assert params["_allowed_smart_feats"] == ["sparse", "spec"]
    assert params["_smart_feats"] == ["sparse", "spec"]
    assert params["enable_sparse"] is True
    assert params["enable_speculative_decode"] is True
    assert os.environ["ENABLE_SPARSE"] == "true"
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "true"
    assert os.environ["ENABLE_KV_OFFLOAD"] == "false"


def test_nvidia_card_token_only_normalizes_chip_names():
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "NVIDIA H20 141GB"}],
    }) == "h20-141"
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "h20 96 gb"}],
    }) == "h20-96"
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "RTX PRO 5000 72GB"}],
    }) == "rtxpro5000-72"
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "NVIDIA RTX PRO 5000 72GB Blackwell"}],
    }) == "rtxpro5000-72"
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "rTx-pro_5000 72 gb"}],
    }) == "rtxpro5000-72"
    assert resolve_card_token({
        "device": "nvidia",
        "hardware_family": "RTX PRO 5000 72GB",
    }) == "rtxpro5000-72"
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "NH02(141GB) / G8600 V7"}],
    }) == "nh02(141gb) / g8600 v7"
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "NRP0500(72GB) / G6550 V8"}],
    }) == "nrp0500(72gb) / g6550 v8"


def test_spec_request_without_whitelist_stays_enabled_for_suffix_fallback(monkeypatch):
    monkeypatch.setenv("ENABLE_SPARSE", "true")
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "model_path": "/usr/local/serving/models/",
        "enable_sparse": True,
        "enable_speculative_decode": True,
    }

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "ascend", "count": 1, "details": [{"name": "Ascend910B_64G"}]},
    )

    assert params["_allowed_smart_feats"] == []
    assert params["_smart_feats"] == []
    assert params["enable_sparse"] is False
    assert params["enable_speculative_decode"] is True
    assert os.environ["ENABLE_SPARSE"] == "false"
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "true"
    assert os.environ["ENABLE_KV_OFFLOAD"] == "false"


def test_detect_hardware_accepts_minimal_ascend_hardware_family_file(tmp_path, monkeypatch):
    hardware_file = tmp_path / "hardware_info.json"
    hardware_file.write_text(
        json.dumps({"device": "ascend", "hardware_family": "Ascend910B_64G", "count": 2}),
        encoding="utf-8",
    )
    monkeypatch.setenv("WINGS_HARDWARE_FILE", str(hardware_file))
    monkeypatch.setenv("WINGS_DEVICE_COUNT", "4")
    monkeypatch.setenv("WINGS_DEVICE_NAME", "Ascend910C")

    hardware = detect_hardware(device_count=8)

    assert hardware["device"] == "ascend"
    assert hardware["count"] == 8
    assert hardware["hardware_family"] == "Ascend910B_64G"
    assert hardware["details"] == [{"name": "Ascend910B_64G"}]
    assert resolve_card_token(hardware) == "ascend910b_64g"


def test_resolve_card_token_keeps_engine_version_platform_fallback(monkeypatch):
    monkeypatch.setenv("WINGS_DEVICE_NAME", "Ascend910C")
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")
    monkeypatch.setenv("ENGINE_VERSION", "wings-vllm-ascend:v0.18.0rc1-a2")

    assert resolve_card_token({"device": "ascend", "details": []}) == "ascend910b"


def test_resolve_card_model_uses_hardware_env_only(monkeypatch):
    monkeypatch.setenv("WINGS_DEVICE_NAME", "NVIDIA RTX PRO 5000 72GB Blackwell")
    monkeypatch.setenv("ENGINE_VERSION", "wings-vllm-ascend:v0.18.0rc1-a3")

    assert resolve_card_model({"hardware_family": ""}) == "a3"
    assert resolve_card_model(
        {"hardware_family": "NVIDIA RTX PRO 5000 72GB Blackwell"}
    ) == "rtx_pro_5000_72G"


def test_deepseek_v4_flash_pro5000_detection_uses_source_hardware_or_resolved_token(monkeypatch):
    monkeypatch.setenv("WINGS_DEVICE_NAME", "NVIDIA RTX PRO 5000 72GB Blackwell")

    source = {
        "engine": "vllm",
        "model_name": "deepseek-ai/DeepSeek-V4-Flash",
        "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
    }

    assert model_utils.is_deepseek_v4_flash_rtx_pro_5000(source) is False
    assert model_utils.is_deepseek_v4_flash_rtx_pro_5000({
        **source,
        "hardware_family": "NVIDIA RTX PRO 5000 72GB Blackwell",
    }) is True
    assert model_utils.is_deepseek_v4_flash_rtx_pro_5000({
        **source,
        "_smart_card_token": "rtxpro5000-72",
    }) is True


def test_deepseek_v4_flash_pro5000_token_drives_nv_parallel_defaults(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)

    command = vllm_adapter.build_start_command({
        "engine": "vllm",
        "model_name": "deepseek-ai/DeepSeek-V4-Flash",
        "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
        "model_type": "llm",
        "device_count": 8,
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": ["spec", "sparse"],
    })

    assert " --tensor-parallel-size 4" in f" {command}"
    assert " --data-parallel-size 2" in f" {command}"


def test_final_device_count_uses_explicit_launch_value_in_full_mode():
    params = {"gpu_usage_mode": "full", "device_count": 8}

    config_loader._set_final_device_count({"count": 2}, params)

    assert params["device_count"] == 8
