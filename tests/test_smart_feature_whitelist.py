import os
import sys
import json
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from utils import model_utils  # noqa: E402
from utils.device_utils import resolve_card_token  # noqa: E402
from core import config_loader  # noqa: E402


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


def test_default_smart_feature_whitelist_file_is_loaded():
    assert model_utils._SMART_WHITELIST_PATH.exists()

    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "910c",
    ) == frozenset({"spec", "offload"})

    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "deepseek-ai/DeepSeek-V4-Flash",
        "/models/deepseek-ai/DeepSeek-V4-Flash",
        "nvidia rtx pro 5000 72gb",
    ) == frozenset({"spec", "sparse", "offload"})

    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen3.5-397B-A17B-NVFP4",
        "/models/Qwen3.5-397B-A17B-NVFP4",
        "nvidia rtx pro 5000 72gb",
    ) == frozenset({"spec", "sparse", "offload"})

    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "GLM-4.7",
        "/models/GLM-4.7",
        "",
    ) == frozenset()

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

    assert params["_allowed_smart_feats"] == ["sparse"]
    assert params["_smart_feats"] == ["sparse"]
    assert params["enable_sparse"] is True
    assert params["enable_speculative_decode"] is False
    assert os.environ["ENABLE_SPARSE"] == "true"
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "false"
    assert os.environ["ENABLE_KV_OFFLOAD"] == "false"
