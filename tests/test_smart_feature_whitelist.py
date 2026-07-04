import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from utils import model_utils  # noqa: E402
from core import config_loader  # noqa: E402


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


def test_deepseek_v4_flash_a3_forces_whitelisted_smart_features(monkeypatch):
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

    assert params["_smart_feats"] == ["offload", "spec"]
    assert params["_forced_smart_feats"] == ["offload", "spec"]
    assert params["enable_speculative_decode"] is True
    assert params["enable_sparse"] is False
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "true"
    assert os.environ["ENABLE_KV_OFFLOAD"] == "true"
    assert os.environ["LMCACHE_OFFLOAD"] == "true"
