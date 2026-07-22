import os
import sys
import json
import importlib.util
from pathlib import Path

import pytest


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


class _FakeQwenDenseInfo:
    model_name = "Qwen"
    model_path = "/usr/local/serving/models/"
    model_architecture = "Qwen3_5ForConditionalGeneration"

    @staticmethod
    def identify_model_architecture():
        return "Qwen3_5ForConditionalGeneration"

    @staticmethod
    def identify_model_type():
        return "llm"


class _FakeQwenMoeInfo:
    model_name = "Qwen"
    model_path = "/usr/local/serving/models/"
    model_architecture = "Qwen3_5MoeForConditionalGeneration"

    @staticmethod
    def identify_model_architecture():
        return "Qwen3_5MoeForConditionalGeneration"

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


class _FakeDeepSeekV32Identifier:
    model_architecture = "DeepseekV32ForCausalLM"
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeQwen35Identifier:
    model_architecture = "Qwen3_5ForConditionalGeneration"
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


def _load_s_empty_verifier():
    path = (
        Path(__file__).resolve().parents[1]
        / "wings_control"
        / "docs"
        / "DAY0"
        / "verify_s_empty_day0_dry_run.py"
    )
    spec = importlib.util.spec_from_file_location("verify_s_empty_day0_dry_run_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_smart_feature_whitelist_file_is_loaded():
    assert model_utils._SMART_WHITELIST_PATH.exists()
    assert "GLM-4.7-FP8" in model_utils._LLM_MODELS["Glm4MoeForCausalLM"]

    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "910c",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "910b",
    ) == frozenset({"spec", "sparse", "offload"})

    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "deepseek-ai/DeepSeek-V4-Flash",
        "/models/deepseek-ai/DeepSeek-V4-Flash",
        "h20-141",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "deepseek-ai/DeepSeek-V4-Flash",
        "/models/deepseek-ai/DeepSeek-V4-Flash",
        "rtxpro5000-72",
    ) == frozenset({"spec", "sparse"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "deepseek-ai/DeepSeek-V3.2",
        "/models/deepseek-ai/DeepSeek-V3.2",
        "h20-96",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "deepseek-ai/DeepSeek-V3.2",
        "/models/deepseek-ai/DeepSeek-V3.2",
        "h20-141",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "deepseek-ai/DeepSeek-V3.2",
        "/models/deepseek-ai/DeepSeek-V3.2",
        "l20",
    ) == frozenset()

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
    for entry in whitelist["offload"]:
        assert entry.get("backend") in {"native", "lmcache", "memcache"}
        for field in (
            "size_source",
            "size_fallback_gb",
            "size_env_names",
            "memcache_meta_port",
            "memcache_config_port",
            "memcache_protocol",
        ):
            assert field not in entry
    for feature in ("spec", "sparse", "offload"):
        for entry in whitelist[feature]:
            assert "precision_tokens" not in entry
            assert not {
                "bfloat16",
                "fp16",
                "float16",
                "fp4",
                "w4a8",
                "int8",
            }.intersection(entry.get("exclude_name_tokens", []))
    nv023_glm_rows = [
        entry
        for feature in ("spec", "sparse", "offload")
        for entry in whitelist[feature]
        if entry.get("source") == "vllm-0.23"
        and any("glm" in token for token in entry.get("name_tokens", []))
    ]
    assert nv023_glm_rows
    nv023_glm_base_rows = [
        entry
        for entry in nv023_glm_rows
        if not any("fp8" in token for token in entry["name_tokens"])
    ]
    assert nv023_glm_base_rows == []
    assert {
        token
        for entry in nv023_glm_rows
        for token in entry["name_tokens"]
        if token.endswith("-fp8")
    } == {"glm-4.7-fp8", "glm-5-fp8", "glm-5.1-fp8"}
    assert all(
        "glm5.1" not in entry["name_tokens"]
        and "glm4.7" not in entry["name_tokens"]
        for entry in nv023_glm_rows
    )
    deepseek_h20_row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        "deepseek-ai/DeepSeek-V4-Flash",
        "/models/deepseek-ai/DeepSeek-V4-Flash",
        "h20-141",
        "spec",
    )
    assert deepseek_h20_row["source"] == "vllm-0.23"
    qwen_ascend_sources = {
        entry.get("source")
        for feature in ("spec", "offload")
        for entry in whitelist[feature]
        if entry.get("engine") == "vllm_ascend"
        and any("qwen3.5" in token or "qwen3.6" in token for token in entry.get("name_tokens", []))
        and entry.get("source") != "23.6.0"
    }
    assert qwen_ascend_sources == {"vllm-ascend-0.21"}

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
        "Qwen/Qwen3.5-122B-A10B",
        "/models/Qwen/Qwen3.5-122B-A10B",
        "rtxpro5000-72",
    ) == frozenset({"spec", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen/Qwen3.5-27B",
        "/models/Qwen/Qwen3.5-27B",
        "rtxpro5000-72",
    ) == frozenset({"spec", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen/Qwen3.5-35B-A3B",
        "/models/Qwen/Qwen3.5-35B-A3B",
        "rtxpro5000-72",
    ) == frozenset({"offload"})

    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "GLM-4.7",
        "/models/GLM-4.7",
        "",
    ) == frozenset()
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "GLM-4.7-FP8",
        "/models/zai-org/GLM-4.7",
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
        "MiniMax/MiniMax-M2.7-NVFP4",
        "/models/MiniMax/MiniMax-M2.7-NVFP4",
        "rtxpro5000-72",
    ) == frozenset({"spec", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "MiniMax/MiniMax-M3-MXFP8",
        "/models/MiniMax/MiniMax-M3-MXFP8",
        "rtxpro5000-72",
    ) == frozenset({"spec", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "MiniMax/MiniMax-M2.5-NVFP4",
        "/models/MiniMax/MiniMax-M2.5-NVFP4",
        "rtxpro5000-72",
    ) == frozenset({"spec", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "MiniMax/MiniMax-M2.7",
        "/models/MiniMax/MiniMax-M2.7",
        "rtxpro5000-72",
    ) == frozenset()
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "GLM-5.1-FP8",
        "/models/zai-org/GLM-5.1",
        "h20-141",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "GLM-5-FP8",
        "/models/zai-org/GLM-5",
        "h20-96",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "zai-org/GLM-5.1",
        "/models/zai-org/GLM-5.1",
        "h20-141",
    ) == frozenset()
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen3-Embedding-0.6B",
        "/models/Qwen3-Embedding-0.6B",
        "l20",
    ) == frozenset({"sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen3.6-27B",
        "/models/Qwen3.6-27B",
        "l20",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "Qwen3.6-35B-A3B",
        "/usr/local/serving/models/",
        "l20",
    ) == frozenset({"spec", "sparse", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        "bge-large-zh-v1.5",
        "/models/bge-large-zh-v1.5",
        "l20",
    ) == frozenset()

    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Qwen3.6-27B",
        "/models/Qwen3.6-27B",
        "910c",
    ) == frozenset({"spec", "offload"})

    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Qwen3.6-35B-A3B",
        "/models/Qwen3.6-35B-A3B",
        "910c",
    ) == frozenset({"spec", "offload"})
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
        "Eco-Tech/GLM-4.7-w8a8-floatmtp",
        "/models/Eco-Tech/GLM-4.7-w8a8-floatmtp",
        "910c",
    ) == frozenset({"spec"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "DeepSeek-V4-Pro-w4a8-mtp",
        "/models/DeepSeek-V4-Pro-w4a8-mtp",
        "910c",
    ) == frozenset({"spec", "sparse"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "DeepSeek-Coder-V2-Instruct-BF16",
        "/models/DeepSeek-Coder-V2-Instruct-BF16",
        "910c",
    ) == frozenset({"spec"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Eco-Tech/Kimi-K2.6-W4A8",
        "/models/Eco-Tech/Kimi-K2.6-W4A8",
        "910c",
    ) == frozenset({"spec", "offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Kimi-K2.7-Code",
        "/harbor_data/Kimi-K2.7-Code",
        "910c",
    ) == frozenset({"offload"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Eco-Tech/Kimi-K2.6",
        "/models/Eco-Tech/Kimi-K2.6",
        "910c",
    ) == frozenset()
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Kimi-K2.7-Code-w4a8",
        "/harbor_data/Kimi-K2.7-Code-w4a8",
        "910c",
    ) == frozenset()


def test_deepseek_coder_v2_spec_row_uses_model_config_architecture():
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        "DeepSeek-Coder-V2-Instruct-BF16",
        "/models/DeepSeek-Coder-V2-Instruct-BF16",
        "910c",
        "spec",
    )

    assert row is not None
    assert row["arch"] == "DeepseekV2ForCausalLM"
    assert row["suffix_num_speculative_tokens"] == 5
    assert "mtp_method" not in row
    assert "mtp_num_speculative_tokens" not in row


@pytest.mark.parametrize("card_token", ["l20", "h20-96", "h20-141"])
@pytest.mark.parametrize(
    ("model_name", "expected_features", "mtp_tokens", "mtp_moe_backend"),
    [
        ("Qwen/Qwen3-Embedding-0.6B", frozenset({"sparse", "offload"}), None, None),
        ("Qwen/Qwen3.6-27B", frozenset({"spec", "sparse", "offload"}), 1, None),
        (
            "Qwen/Qwen3.6-35B-A3B",
            frozenset({"spec", "sparse", "offload"}),
            3,
            "triton",
        ),
    ],
)
def test_nvidia_day0_l20_smart_features_inherit_to_h20(
    card_token,
    model_name,
    expected_features,
    mtp_tokens,
    mtp_moe_backend,
):
    model_path = f"/models/{model_name}"

    assert model_utils.resolve_feature_whitelist(
        "vllm",
        model_name,
        model_path,
        card_token,
    ) == expected_features

    offload_row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        model_name,
        model_path,
        card_token,
        "offload",
    )
    assert offload_row["source"] == "vllm-0.23"
    assert offload_row["backend"] == "native"

    spec_row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        model_name,
        model_path,
        card_token,
        "spec",
    )
    if mtp_tokens is None:
        assert spec_row is None
    else:
        assert spec_row["source"] == "vllm-0.23"
        assert spec_row["mtp_num_speculative_tokens"] == mtp_tokens
        assert spec_row.get("mtp_moe_backend") == mtp_moe_backend


@pytest.mark.parametrize("card_token", ["l20", "h20-96", "h20-141"])
@pytest.mark.parametrize("model_name", ["zai-org/GLM-5", "zai-org/GLM-4.7", "zai-org/GLM-5.1"])
def test_nvidia_day0_base_glm_names_have_no_smart_features(model_name, card_token):
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        model_name,
        f"/models/{model_name}",
        card_token,
    ) == frozenset()


@pytest.mark.parametrize(
    "model_name",
    ["bge-large-zh-v1.5", "bge-reranker-large"],
)
@pytest.mark.parametrize("card_token", ["l20", "h20-96", "h20-141"])
def test_nvidia_day0_bge_scenarios_stay_out_of_smart_feature_whitelist(
    model_name,
    card_token,
):
    assert model_utils.resolve_feature_whitelist(
        "vllm",
        model_name,
        f"/models/{model_name}",
        card_token,
    ) == frozenset()


@pytest.mark.parametrize(
    "feature,model_name,card_token",
    [
        ("spec", "Eco-Tech/Qwen3.6-27B-w8a8", "910c"),
        ("spec", "Eco-Tech/Qwen3.6-27B-w8a8", "910b"),
        ("spec", "Eco-Tech/Qwen3.6-35B-A3B-w8a8", "910c"),
        ("spec", "Eco-Tech/Qwen3.6-35B-A3B-w8a8", "910b"),
        ("offload", "Eco-Tech/Qwen3.6-27B-w8a8", "910c"),
        ("offload", "Eco-Tech/Qwen3.6-27B-w8a8", "910b"),
        ("offload", "Eco-Tech/Qwen3.6-35B-A3B-w8a8", "910c"),
        ("offload", "Eco-Tech/Qwen3.6-35B-A3B-w8a8", "910b"),
    ],
)
def test_qwen36_w8a8_scenarios_use_explicit_rows(
    feature,
    model_name,
    card_token,
):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        model_name,
        f"/models/{model_name}",
        card_token,
        feature,
    )

    assert row is not None
    assert any("w8a8" in token for token in row["name_tokens"])


@pytest.mark.parametrize(
    "feature,model_name,card_token",
    [
        ("spec", "Qwen/Qwen3.6-27B", "910c"),
        ("spec", "Qwen/Qwen3.6-27B", "910b"),
        ("spec", "Qwen/Qwen3.6-35B-A3B", "910c"),
        ("spec", "Qwen/Qwen3.6-35B-A3B", "910b"),
        ("offload", "Qwen/Qwen3.6-27B", "910c"),
        ("offload", "Qwen/Qwen3.6-35B-A3B", "910c"),
    ],
)
def test_qwen36_plain_scenarios_exclude_only_confirmed_w8a8_variant(
    feature,
    model_name,
    card_token,
):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        model_name,
        f"/models/{model_name}",
        card_token,
        feature,
    )

    assert row is not None
    assert row.get("exclude_name_tokens") == ("w8a8",)


def test_qwen36_35b_l20_open_source_name_enables_spec(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    params = {
        "engine": "vllm",
        "model_name": "Qwen3.6-35B-A3B",
        "model_path": "/usr/local/serving/models/",
        "enable_sparse": False,
        "enable_speculative_decode": True,
    }
    hardware = {"device": "nvidia", "details": [{"name": "L20_45G"}]}

    config_loader.apply_effective_feature_enablement(params, hardware)

    assert params["_allowed_smart_feats"] == ["offload", "sparse", "spec"]
    assert params["_smart_feats"] == ["spec"]
    assert params["enable_speculative_decode"] is True


def test_nvidia_day0_unlisted_spec_keeps_legacy_spec_request(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    hardware = {"device": "nvidia", "details": [{"name": "NVIDIA L20"}]}
    params = {
        "engine": "vllm",
        "model_name": "bge-large-zh-v1.5-BF16",
        "model_path": "/models/bge-large-zh-v1.5-BF16",
        "enable_sparse": True,
        "enable_speculative_decode": True,
    }

    config_loader.apply_effective_feature_enablement(params, hardware)

    assert params["enable_sparse"] is False
    assert params["enable_speculative_decode"] is True
    assert params["_smart_feats"] == []

    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "Kimi-K2.7-Code",
        "/harbor_data/Kimi-K2.7-Code",
        "910b",
    ) == frozenset()


def test_smart_feature_trace_preserves_upper_env_before_gate_rewrites(monkeypatch):
    # 这个场景故意让页面同时请求 spec/sparse/offload，但白名单只允许 offload。
    # sparse 会被 gate 改写为 false；审计日志仍必须保留改写前的 ENABLE_SPARSE=true，
    # 否则现场无法区分“页面没开”和“页面开了但被白名单收窄”。
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_SPARSE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    monkeypatch.delenv("KV_DISK_OFFLOAD_SIZE", raising=False)
    monkeypatch.setattr(
        config_loader,
        "resolve_feature_whitelist",
        lambda engine, name, path, card: {"offload"},
    )
    monkeypatch.setattr(
        config_loader,
        "resolve_forced_feature_whitelist",
        lambda engine, name, path, card: set(),
    )
    hardware = {"device": "nvidia", "details": [{"name": "NVIDIA L20"}]}
    params = {
        "engine": "vllm",
        "model_name": "custom-model",
        "model_path": "/models/custom-model",
        "model_type": "llm",
        "enable_sparse": True,
        "enable_speculative_decode": True,
    }

    config_loader.apply_effective_feature_enablement(params, hardware)

    assert params["_smart_feature_input_env"] == {
        "ENABLE_SPECULATIVE_DECODE": "true",
        "ENABLE_SPARSE": "true",
        "ENABLE_KV_OFFLOAD": "true",
        "LMCACHE_OFFLOAD": "true",
        "ENABLE_KV_MEM_OFFLOAD": "true",
        "KV_MEM_OFFLOAD_SIZE": "auto",
        "AVAILABLE_POD_MEM_SIZE": "81920",
        "ENABLE_KV_DISK_OFFLOAD": "false",
        "KV_DISK_OFFLOAD_SIZE": None,
    }
    assert os.environ["ENABLE_SPARSE"] == "false"
    assert os.environ["ENABLE_KV_OFFLOAD"] == "true"
    trace = params["_smart_feature_gate_trace"]
    assert trace["allowed"] == ["offload"]
    assert trace["effective"] == ["offload"]
    assert trace["features"]["sparse_kv"] == {
        "requested": True,
        "whitelist": False,
        "gate": False,
        "reason": "whitelist_miss",
    }
    assert trace["features"]["speculative_decode"] == {
        "requested": True,
        "whitelist": False,
        "gate": True,
        "reason": "suffix_fallback",
    }


def test_deepseek_v32_h20_whitelist_enables_smart_trio(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    hardware = {
        "device": "nvidia",
        "details": [{"name": "G8600+H20 * 8", "total_memory": 141}],
    }
    params = {
        "engine": "vllm",
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "model_path": "/models/deepseek-ai/DeepSeek-V3.2",
        "enable_sparse": True,
        "enable_speculative_decode": True,
    }

    config_loader.apply_effective_feature_enablement(params, hardware)

    assert params["_allowed_smart_feats"] == ["offload", "sparse", "spec"]
    assert params["_smart_feats"] == ["offload", "sparse", "spec"]
    assert params["enable_sparse"] is True
    assert params["enable_speculative_decode"] is True
    assert params["_smart_card_token"] == "h20-141"


def test_deepseek_v32_h20_sparse_whitelist_emits_indexcache_topk4(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV32Identifier)
    params = {
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "model_path": "/models/deepseek-ai/DeepSeek-V3.2",
        "model_type": "llm",
        "_smart_card_token": "h20-141",
        "_smart_feats": ["sparse"],
        "engine_config": {},
    }

    command = vllm_adapter._build_kv_sparse_cmd(params, "vllm")

    assert command == ' --hf-overrides \'{"use_index_cache":true,"index_topk_freq":4}\''
    assert vllm_adapter.resolve_sparse_variant(params, "vllm") == "indexcache_use_index_cache_topk4"
    assert params["engine_config"] == {}


def test_deepseek_v32_h20_spec_whitelist_emits_mtp3(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV32Identifier)
    params = {
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "model_path": "/models/deepseek-ai/DeepSeek-V3.2",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "_smart_card_token": "h20-141",
        "_smart_feats": ["spec"],
        "engine_config": {},
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert "--speculative-config" in command
    assert '"method":"mtp"' in command
    assert '"num_speculative_tokens":3' in command
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "moe_backend": None,
    }


def test_deepseek_v32_h20_offload_whitelist_emits_native_backend(monkeypatch):
    monkeypatch.delenv("CONFIG_FORCE", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "200")
    params = {
        "engine": "vllm",
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "model_path": "/models/deepseek-ai/DeepSeek-V3.2",
        "model_type": "llm",
        "_smart_card_token": "h20-141",
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter._build_kv_offload_cmd(params, "vllm") == (
        " --kv-offloading-backend native --kv-offloading-size 200"
    )
    assert vllm_adapter.resolve_offload_variant(params, "vllm") == (
        "native_kv_offloading_backend"
    )


def test_s_empty_verifier_counts_fp8_kv_cache_as_sparse():
    verifier = _load_s_empty_verifier()
    fp8 = verifier.ParsedCommand(flags={"kv-cache-dtype": "fp8"})
    auto = verifier.ParsedCommand(flags={"kv-cache-dtype": "auto"})
    indexcache = verifier.ParsedCommand(flags={"hf-overrides": {"use_index_cache": True}})

    assert verifier._reference_requests_sparse(fp8) is True
    assert verifier._reference_requests_sparse(indexcache) is True
    assert verifier._reference_requests_sparse(auto) is False
    assert verifier._skip_features_off_reference_field("kv-cache-dtype", fp8) is True
    assert verifier._skip_features_off_reference_field("kv-cache-dtype", auto) is False


def test_offload_backend_lookup_respects_effective_smart_feats(monkeypatch):
    monkeypatch.delenv("CONFIG_FORCE", raising=False)
    params = {
        "model_name": "Qwen3.6-27B",
        "model_path": "/models/Qwen3.6-27B",
        "_smart_card_token": "l20",
        "_smart_feats": [],
    }

    assert model_utils.resolve_feature_whitelist_row_from_params(
        params,
        "vllm",
        "offload",
    ) is not None
    assert model_utils.resolve_feature_whitelist_row_from_params(
        params,
        "vllm",
        "offload",
        require_enabled=True,
    ) is None
    assert model_utils.resolve_offload_whitelist_backend(params, "vllm") == ""


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_deepseek_v4_flash_ascend_offload_backend_remains_lmcache(card_token):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        card_token,
        "offload",
    )

    assert row is not None
    assert row["backend"] == "lmcache"


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

    assert params["_allowed_smart_feats"] == ["offload", "sparse", "spec"]
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
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "80")

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


class _FakeKimiK26Info:
    model_name = "Eco-Tech/Kimi-K2.6-W4A8"
    model_path = "/models/Eco-Tech/Kimi-K2.6-W4A8"
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
    assert "no_disable_hybrid_kv_cache_manager" not in config
    assert kv_transfer == {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "lookup_rpc_port": "0",
            "backend": "memcache",
        },
    }


def test_kimi_k26_uses_memcache_ascend_store_connector(monkeypatch):
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_VERSION", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/Kimi-K2.6-W4A8",
        "model_path": "/models/Eco-Tech/Kimi-K2.6-W4A8",
        "model_type": "llm",
        "distributed": False,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "device_count": 16,
    }
    hardware_env = {"device": "ascend", "details": [{"name": "Ascend910C"}]}

    config_loader.apply_effective_feature_enablement(params, hardware_env)
    config = config_loader._get_model_specific_config(
        hardware_env,
        params,
        _FakeKimiK26Info(),
    )

    kv_transfer = json.loads(config["kv_transfer_config"])
    assert params["_smart_feats"] == ["offload", "spec"]
    assert params["enable_speculative_decode"] is True
    assert "no_disable_hybrid_kv_cache_manager" not in config
    assert kv_transfer == {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "lookup_rpc_port": "0",
            "backend": "memcache",
        },
    }


def test_kimi_k26_effective_spec_allows_suffix_fallback_without_dflash(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "false")
    hardware_env = {"device": "ascend", "details": [{"name": "Ascend910C"}]}
    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/Kimi-K2.6-W4A8",
        "model_path": "/models/Eco-Tech/Kimi-K2.6-W4A8",
        "enable_sparse": False,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
    }

    config_loader.apply_effective_feature_enablement(params, hardware_env)

    assert params["_allowed_smart_feats"] == ["offload", "spec"]
    assert params["_smart_feats"] == ["spec"]
    assert params["enable_speculative_decode"] is True

    params["enable_speculative_decode"] = True
    params["speculative_decode_model_path"] = "z-lab/Kimi-K2.6-DFlash"
    config_loader.apply_effective_feature_enablement(params, hardware_env)

    assert params["_smart_feats"] == ["spec"]
    assert params["enable_speculative_decode"] is True


def test_kimi_k27_code_effective_spec_is_suppressed(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "true")
    hardware_env = {"device": "ascend", "details": [{"name": "Ascend910C"}]}
    params = {
        "engine": "vllm_ascend",
        "model_name": "Kimi-K2.7-Code",
        "model_path": "/harbor_data/Kimi-K2.7-Code",
        "enable_sparse": False,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "z-lab/Kimi-K2.6-DFlash",
    }

    config_loader.apply_effective_feature_enablement(params, hardware_env)

    assert params["_allowed_smart_feats"] == ["offload"]
    assert params["_smart_feats"] == ["offload"]
    assert params["enable_speculative_decode"] is False


@pytest.mark.parametrize(
    ("model_name", "model_info", "card_name", "device_count"),
    [
        ("Qwen/Qwen3.5-27B", _FakeQwenDenseInfo(), "Ascend910C", 2),
        ("Qwen/Qwen3.6-27B", _FakeQwenDenseInfo(), "Ascend910C", 2),
        ("Eco-Tech/Qwen3.6-27B-w8a8", _FakeQwenDenseInfo(), "Ascend910C", 2),
        ("Eco-Tech/Qwen3.6-27B-w8a8", _FakeQwenDenseInfo(), "Ascend910B_64G", 4),
        ("Qwen/Qwen3.6-35B-A3B", _FakeQwenMoeInfo(), "Ascend910C", 2),
        ("Eco-Tech/Qwen3.6-35B-A3B-w8a8", _FakeQwenMoeInfo(), "Ascend910C", 2),
        ("Eco-Tech/Qwen3.6-35B-A3B-w8a8", _FakeQwenMoeInfo(), "Ascend910B_64G", 4),
        ("Qwen/Qwen3.5-35B-A3B", _FakeQwenMoeInfo(), "Ascend910B_64G", 2),
    ],
)
def test_qwen_day0_memcache_omits_recompute_load_failure_policy(
    monkeypatch,
    model_name,
    model_info,
    card_name,
    device_count,
):
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_VERSION", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    params = {
        "engine": "vllm_ascend",
        "model_name": model_name,
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "distributed": False,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "device_count": device_count,
    }
    hardware_env = {"device": "ascend", "details": [{"name": card_name}]}

    config_loader.apply_effective_feature_enablement(params, hardware_env)
    config = config_loader._get_model_specific_config(
        hardware_env,
        params,
        model_info,
    )

    kv_transfer = json.loads(config["kv_transfer_config"])
    assert params["_smart_feats"] == ["offload", "spec"]
    assert config["no_disable_hybrid_kv_cache_manager"] is True
    assert kv_transfer == {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
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
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "20480")
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


@pytest.mark.parametrize(
    "hardware_env",
    [
        {"device": "ascend", "details": [{"name": "Ascend910"}]},
        {"device": "ascend", "details": [{"name": "Ascend 910"}]},
        {"device": "ascend", "hardware_family": "Ascend910_64G"},
    ],
)
def test_bare_ascend910_is_treated_as_910c_without_suffix_letter(hardware_env, monkeypatch):
    monkeypatch.setenv("ENABLE_SPARSE", "true")
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")

    params = {
        "engine": "vllm_ascend",
        "model_name": "GLM-5.1-w8a8",
        "model_path": "/usr/local/serving/models/",
        "enable_sparse": True,
        "enable_speculative_decode": True,
    }

    assert resolve_card_token(hardware_env) == "ascend910c"
    assert resolve_card_model(hardware_env) == "a3"
    assert config_loader._standard_ascend_card_token(hardware_env) == "910c"
    assert vllm_adapter._match_ascend_platform(_first_card_name(hardware_env)) == "a3"

    config_loader.apply_effective_feature_enablement(params, hardware_env)

    assert params["_smart_card_token"] == "ascend910c"
    assert params["_allowed_smart_feats"] == ["sparse", "spec"]
    assert params["_smart_feats"] == ["sparse", "spec"]


@pytest.mark.parametrize(
    ("card_name", "expected_platform"),
    [
        ("Ascend910B", "a2"),
        ("Ascend910B3", "a2"),
        ("Ascend910B_64G", "a2"),
        ("Ascend910A", ""),
    ],
)
def test_bare_ascend910_rule_does_not_override_letter_suffixes(card_name, expected_platform):
    hardware_env = {"device": "ascend", "details": [{"name": card_name}]}

    assert resolve_card_token(hardware_env) == card_name.lower()
    assert vllm_adapter._match_ascend_platform(card_name) == expected_platform
    if expected_platform == "a2":
        assert resolve_card_model(hardware_env) == "a2"
        assert config_loader._standard_ascend_card_token(hardware_env) == "910b"
    else:
        assert resolve_card_model(hardware_env) != "a3"
        assert config_loader._standard_ascend_card_token(hardware_env) == ""


def _first_card_name(hardware_env):
    details = hardware_env.get("details") or [{"name": hardware_env.get("hardware_family", "")}]
    return details[0]["name"]


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
        "details": [{"name": "G8600+H20 * 8", "total_memory": 141}],
    }) == "h20-141"
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "G8600+H20 * 8", "total_memory": 96}],
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
        "details": [{"name": "G6550+RTX PRO 5000 * 8"}],
    }) == "rtxpro5000-72"
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "TokenBox RTX PRO 5000 * 8"}],
    }) == "rtxpro5000-72"
    assert resolve_card_token({
        "device": "nvidia",
        "details": [{"name": "RTX PRO 5000 48GB"}],
    }) == "rtxpro5000-48"
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


@pytest.mark.parametrize(
    ("model_type", "model_name"),
    [
        ("embedding", "bge-large-zh-v1.5"),
        ("rerank", "bge-reranker-large"),
    ],
)
def test_pooling_models_suppress_spec_suffix_fallback(monkeypatch, model_type, model_name):
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("SD_ENABLE", "true")

    params = {
        "engine": "vllm",
        "model_name": model_name,
        "model_path": f"/models/{model_name}",
        "model_type": model_type,
        "_resolved_model_type": model_type,
        "enable_sparse": False,
        "enable_speculative_decode": True,
    }

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "nvidia", "count": 1, "details": [{"name": "NVIDIA L20"}]},
    )

    assert params["enable_speculative_decode"] is False
    assert "spec" not in params["_smart_feats"]
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "false"
    assert os.environ["SD_ENABLE"] == "false"

    engine_config = {
        "enable_speculative_decode": True,
        "speculative_config": {"method": "suffix", "num_speculative_tokens": 5},
    }
    config_loader._validate_embedding_rerank_params(engine_config, {"model_type": model_type})
    assert engine_config["enable_speculative_decode"] is False
    assert "speculative_config" not in engine_config


def test_qwen35_397b_a17b_ascend910b_is_not_in_day0_spec_whitelist():
    assert model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        "Qwen/Qwen3.5-397B-A17B",
        "/models/Qwen/Qwen3.5-397B-A17B",
        "910b",
        "spec",
    ) is None


def test_qwen36_27b_ascend910b_reuses_spec_but_disables_offload(monkeypatch):
    monkeypatch.setenv("ENABLE_SPARSE", "false")
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen35Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "qwen3.6-27b",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_sparse": False,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
    }

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "ascend", "count": 1, "details": [{"name": "Ascend910B_64G"}]},
    )

    assert params["_allowed_smart_feats"] == ["spec"]
    assert params["_smart_feats"] == ["spec"]
    assert os.environ["ENABLE_KV_OFFLOAD"] == "false"
    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "qwen3_5_mtp"


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-35B-A3B",
    ],
)
def test_qwen_day0_910b_unlisted_models_stay_out_of_offload_whitelist(model_name):
    assert model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        model_name,
        f"/models/{model_name}",
        "910b",
        "offload",
    ) is None


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3.5-35B-A3B",
        "Eco-Tech/Qwen3.6-27B-w8a8",
        "Eco-Tech/Qwen3.6-35B-A3B-w8a8",
    ],
)
def test_qwen_day0_910b_selected_models_reuse_memcache_offload(model_name):
    # 行 13、14、17 复用 910C 的 MemCache 卸载方式，只对明确点名的
    # 910B 模型补 offload，避免把整个 Qwen 910B 家族都放宽。
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        model_name,
        f"/models/{model_name}",
        "910b",
        "offload",
    )

    assert row is not None
    assert row["backend"] == "memcache"


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_qwen35_35b_a3b_ascend_spec_row_matches_day0_mtp_recipe(card_token):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        "Qwen/Qwen3.5-35B-A3B",
        "/models/Qwen/Qwen3.5-35B-A3B",
        card_token,
        "spec",
    )

    assert row is not None
    assert row.get("mtp_method") == "qwen3_5_mtp"
    assert row.get("mtp_num_speculative_tokens") == 1
    assert row.get("enforce_eager") is True


def test_qwen_day0_910b_reference_script_without_offload_still_has_no_memcache():
    """未补 910B offload 的参考脚本不能直接注入卸载。"""
    day0_dir = Path(__file__).resolve().parents[1] / "wings_control" / "docs" / "DAY0"
    script_names = ("Qwen3.5-27B-910b.sh",)
    forbidden = (
        "AscendStoreConnector",
        "MMC_",
        "LMCACHE_",
        "--kv-transfer-config",
        "--no-disable-hybrid-kv-cache-manager",
    )

    for script_name in script_names:
        script = (day0_dir / script_name).read_text(encoding="utf-8")
        assert "qwen3_5_mtp" in script
        assert "--tool-call-parser qwen3_coder" in script
        assert not any(token in script for token in forbidden)


@pytest.mark.parametrize(
    ("engine", "model_name", "model_path", "card_token", "expected_tokens"),
    [
        ("vllm", "Qwen3.5-397B-A17B-NVFP4", "/models/Qwen3.5-397B-A17B-NVFP4", "rtxpro5000-72", 3),
        ("vllm", "Qwen3.5-397B-A17B", "/models/Qwen/Qwen3.5-397B-A17B", "rtxpro5000-72", 3),
        ("vllm", "Qwen/Qwen3.5-122B-A10B", "/models/Qwen/Qwen3.5-122B-A10B", "rtxpro5000-72", 1),
        ("vllm", "Qwen/Qwen3.5-27B", "/models/Qwen/Qwen3.5-27B", "rtxpro5000-72", 2),
        ("vllm", "Qwen/Qwen-AgentWorld-35B-A3B", "/models/Qwen/Qwen-AgentWorld-35B-A3B", "rtxpro5000-72", 32),
        ("vllm", "MiniMax/MiniMax-M3-MXFP8", "/models/MiniMax/MiniMax-M3-MXFP8", "rtxpro5000-72", 32),
        ("vllm", "MiniMax/MiniMax-M2.5-NVFP4", "/models/MiniMax/MiniMax-M2.5-NVFP4", "rtxpro5000-72", 10),
        ("vllm", "MiniMax/MiniMax-M2.7-NVFP4", "/models/MiniMax/MiniMax-M2.7-NVFP4", "rtxpro5000-72", 10),
        ("vllm", "GLM-4.7-FP8", "/models/zai-org/GLM-4.7", "h20-141", 1),
        ("vllm", "deepseek-ai/DeepSeek-V3.2", "/models/deepseek-ai/DeepSeek-V3.2", "h20-141", 3),
        ("vllm", "deepseek-ai/DeepSeek-V4-Flash", "/models/deepseek-ai/DeepSeek-V4-Flash", "h20-141", 1),
        ("vllm", "deepseek-ai/DeepSeek-V4-Flash", "/models/deepseek-ai/DeepSeek-V4-Flash", "rtxpro5000-72", 2),
        ("vllm_ascend", "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "910c", 1),
        ("vllm_ascend", "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "910b", 1),
        ("vllm_ascend", "DeepSeek-V4-Pro-w4a8-mtp", "/models/DeepSeek-V4-Pro-w4a8-mtp", "910c", 1),
        ("vllm_ascend", "Eco-Tech/GLM-5.2-w8a8", "/models/Eco-Tech/GLM-5.2-w8a8", "910c", 3),
        ("vllm_ascend", "Eco-Tech/GLM-5.1-w8a8", "/models/Eco-Tech/GLM-5.1-w8a8", "910b", 3),
        ("vllm_ascend", "Eco-Tech/GLM-5.1-w8a8", "/models/Eco-Tech/GLM-5.1-w8a8", "910c", 3),
        ("vllm_ascend", "Eco-Tech/GLM-4.7-w8a8-floatmtp", "/models/Eco-Tech/GLM-4.7-w8a8-floatmtp", "910b", 3),
        ("vllm_ascend", "Eco-Tech/GLM-4.7-w8a8-floatmtp", "/models/Eco-Tech/GLM-4.7-w8a8-floatmtp", "910c", 3),
        ("vllm_ascend", "vllm-ascend/DeepSeek-V3.2-w8a8", "/models/vllm-ascend/DeepSeek-V3.2-w8a8", "910c", 3),
        ("vllm_ascend", "Qwen3.5-397B-A17B-w8a8-mtp", "/models/Qwen3.5-397B-A17B-w8a8-mtp", "910b", 3),
        ("vllm_ascend", "Qwen3.5-397B-A17B-w8a8-mtp", "/models/Qwen3.5-397B-A17B-w8a8-mtp", "910c", 1),
        ("vllm_ascend", "Qwen/Qwen3.5-35B-A3B", "/models/Qwen/Qwen3.5-35B-A3B", "910b", 1),
        ("vllm_ascend", "Qwen/Qwen3.5-122B-A10B", "/models/Qwen/Qwen3.5-122B-A10B", "910b", 1),
        ("vllm_ascend", "Qwen/Qwen3.6-27B", "/models/Qwen/Qwen3.6-27B", "910b", 3),
        ("vllm_ascend", "Qwen/Qwen3.6-35B-A3B", "/models/Qwen/Qwen3.6-35B-A3B", "910b", 3),
    ],
)
def test_spec_whitelist_mtp_rows_carry_mtp_tokens(
    engine,
    model_name,
    model_path,
    card_token,
    expected_tokens,
):
    row = model_utils.resolve_feature_whitelist_row(
        engine,
        model_name,
        model_path,
        card_token,
        "spec",
    )

    assert row is not None
    assert row.get("mtp_num_speculative_tokens") == expected_tokens


@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
def test_deepseek_v4_flash_h20_spec_rows_use_explicit_mtp_without_eager(card_token):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        "deepseek-ai/DeepSeek-V4-Flash",
        "/models/deepseek-ai/DeepSeek-V4-Flash",
        card_token,
        "spec",
    )

    assert row is not None
    assert row.get("mtp_method") == "mtp"
    assert row.get("mtp_num_speculative_tokens") == 1
    assert "enforce_eager" not in row


def test_deepseek_v4_flash_pro5000_spec_row_uses_explicit_mtp2():
    row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        "deepseek-ai/DeepSeek-V4-Flash",
        "/models/deepseek-ai/DeepSeek-V4-Flash",
        "rtxpro5000-72",
        "spec",
    )

    assert row is not None
    assert row.get("mtp_method") == "mtp"
    assert row.get("mtp_num_speculative_tokens") == 2
    assert "enforce_eager" not in row


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_minimax_m25_w8a8_quarot_spec_rows_use_eagle3_with_suffix3(card_token):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        "Eco-Tech/MiniMax-M2.5-w8a8-QuaRot",
        "/models/Eco-Tech/MiniMax-M2.5-w8a8-QuaRot",
        card_token,
        "spec",
    )

    assert row is not None
    assert row.get("draft_method") == "eagle3"
    assert row.get("draft_num_speculative_tokens") == 3
    assert row.get("draft_enforce_eager") is True
    assert row.get("suffix_num_speculative_tokens") == 3
    assert "mtp_method" not in row


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_minimax_m27_w8a8_quarot_spec_rows_are_draft_only(card_token):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        "MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        card_token,
        "spec",
    )

    assert row is not None
    assert row["source"] == "vllm-ascend-0.21"
    assert row.get("draft_method") == "eagle3"
    assert row.get("draft_num_speculative_tokens") == 3
    assert row.get("draft_enforce_eager") is True
    assert "mtp_method" not in row
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        card_token,
    ) == frozenset({"spec"})


def test_minimax_m27_w8a8_without_quarot_does_not_hit_day0_draft_row():
    assert model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        "MiniMax/MiniMax-M2.7-w8a8",
        "/models/MiniMax/MiniMax-M2.7-w8a8",
        "910c",
        "spec",
    ) is None


def test_qwen35_35b_pro5000_native_offload_has_no_spec_row():
    assert model_utils.resolve_feature_whitelist_row(
        "vllm",
        "Qwen/Qwen3.5-35B-A3B",
        "/models/Qwen/Qwen3.5-35B-A3B",
        "rtxpro5000-72",
        "spec",
    ) is None
    row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        "Qwen/Qwen3.5-35B-A3B",
        "/models/Qwen/Qwen3.5-35B-A3B",
        "rtxpro5000-72",
        "offload",
    )

    assert row is not None
    assert row.get("backend") == "native"


@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
def test_deepseek_v32_h20_offload_backend_is_native(card_token):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        "deepseek-ai/DeepSeek-V3.2",
        "/models/deepseek-ai/DeepSeek-V3.2",
        card_token,
        "offload",
    )

    assert row is not None
    assert row.get("backend") == "native"


@pytest.mark.parametrize(
    ("model_name", "model_path", "expected_backend"),
    [
        ("MiniMax/MiniMax-M3-MXFP8", "/models/MiniMax/MiniMax-M3-MXFP8", "native"),
        ("MiniMax/MiniMax-M2.5-NVFP4", "/models/MiniMax/MiniMax-M2.5-NVFP4", "native"),
        ("MiniMax/MiniMax-M2.7-NVFP4", "/models/MiniMax/MiniMax-M2.7-NVFP4", "lmcache"),
    ],
)
def test_minimax_pro5000_offload_backend_stays_model_specific(
    model_name,
    model_path,
    expected_backend,
):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        model_name,
        model_path,
        "rtxpro5000-72",
        "offload",
    )

    assert row is not None
    assert row.get("backend") == expected_backend


@pytest.mark.parametrize(
    ("feature", "model_name", "model_path", "card_token", "expected_source"),
    [
        ("spec", "MiniMax/MiniMax-M3-MXFP8", "/models/MiniMax/MiniMax-M3-MXFP8", "rtxpro5000-72", "vllm-minimax-m3"),
        ("offload", "MiniMax/MiniMax-M3-MXFP8", "/models/MiniMax/MiniMax-M3-MXFP8", "rtxpro5000-72", "vllm-minimax-m3"),
        ("spec", "MiniMax/MiniMax-M2.5-NVFP4", "/models/MiniMax/MiniMax-M2.5-NVFP4", "rtxpro5000-72", "vllm-0.23"),
        ("offload", "MiniMax/MiniMax-M2.5-NVFP4", "/models/MiniMax/MiniMax-M2.5-NVFP4", "rtxpro5000-72", "vllm-0.23"),
        ("spec", "MiniMax/MiniMax-M2.7-NVFP4", "/models/MiniMax/MiniMax-M2.7-NVFP4", "rtxpro5000-72", "vllm-0.23"),
        ("offload", "MiniMax/MiniMax-M2.7-NVFP4", "/models/MiniMax/MiniMax-M2.7-NVFP4", "rtxpro5000-72", "vllm-0.23"),
        ("spec", "Qwen/Qwen3.5-397B-A17B-NVFP4", "/models/Qwen/Qwen3.5-397B-A17B-NVFP4", "rtxpro5000-72", "vllm-0.23"),
        ("offload", "Qwen/Qwen3.5-397B-A17B-NVFP4", "/models/Qwen/Qwen3.5-397B-A17B-NVFP4", "rtxpro5000-72", "vllm-0.23"),
        ("spec", "Qwen/Qwen3.5-122B-A10B", "/models/Qwen/Qwen3.5-122B-A10B", "rtxpro5000-72", "vllm-0.23"),
        ("offload", "Qwen/Qwen3.5-122B-A10B", "/models/Qwen/Qwen3.5-122B-A10B", "rtxpro5000-72", "vllm-0.23"),
        ("spec", "Qwen/Qwen3.5-27B", "/models/Qwen/Qwen3.5-27B", "rtxpro5000-72", "vllm-0.23"),
        ("offload", "Qwen/Qwen3.5-27B", "/models/Qwen/Qwen3.5-27B", "rtxpro5000-72", "vllm-0.23"),
        ("offload", "Qwen/Qwen3.5-35B-A3B", "/models/Qwen/Qwen3.5-35B-A3B", "rtxpro5000-72", "vllm-0.23"),
        ("offload", "Qwen/Qwen3.5-35B-A3B", "/models/Qwen/Qwen3.5-35B-A3B", "910b", "vllm-ascend-0.21"),
        ("offload", "Eco-Tech/Qwen3.6-27B-w8a8", "/models/Eco-Tech/Qwen3.6-27B-w8a8", "910b", "vllm-ascend-0.21"),
        ("offload", "Eco-Tech/Qwen3.6-35B-A3B-w8a8", "/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8", "910b", "vllm-ascend-0.21"),
        ("spec", "Eco-Tech/GLM-4.7-w8a8-floatmtp", "/models/Eco-Tech/GLM-4.7-w8a8-floatmtp", "910b", "vllm-ascend-0.21"),
        ("spec", "Eco-Tech/GLM-4.7-w8a8-floatmtp", "/models/Eco-Tech/GLM-4.7-w8a8-floatmtp", "910c", "vllm-ascend-0.21"),
        ("spec", "DeepSeek-Coder-V2-Instruct-BF16", "/models/DeepSeek-Coder-V2-Instruct-BF16", "910c", "vllm-ascend-0.21"),
        ("spec", "Qwen/Qwen-AgentWorld-35B-A3B", "/models/Qwen/Qwen-AgentWorld-35B-A3B", "rtxpro5000-72", "vllm-0.23"),
        ("spec", "MiniMax/MiniMax-M2.7-w8a8-QuaRot", "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot", "910b", "vllm-ascend-0.21"),
        ("spec", "MiniMax/MiniMax-M2.7-w8a8-QuaRot", "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot", "910c", "vllm-ascend-0.21"),
        ("offload", "Eco-Tech/Kimi-K2.6-W4A8", "/models/Eco-Tech/Kimi-K2.6-W4A8", "910c", "vllm-ascend-0.21"),
        ("spec", "Eco-Tech/Kimi-K2.6-W4A8", "/models/Eco-Tech/Kimi-K2.6-W4A8", "910c", "vllm-ascend-0.21"),
        ("offload", "Kimi-K2.7-Code", "/harbor_data/Kimi-K2.7-Code", "910c", "vllm-ascend-0.21"),
    ],
)
def test_day0_adapted_whitelist_rows_use_engine_version_source(
    feature,
    model_name,
    model_path,
    card_token,
    expected_source,
):
    # source 是文档元信息，不参与 matcher；这里单独锁定本轮 DAY0 收编口径，
    # 防止后续把 Pro 5000 / Ascend 0.21 场景误回填成旧矩阵来源。
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend" if card_token.startswith("910") else "vllm",
        model_name,
        model_path,
        card_token,
        feature,
    )

    assert row is not None
    assert row.get("source") == expected_source


def test_detect_hardware_accepts_minimal_ascend_hardware_family_file(tmp_path, monkeypatch):
    hardware_file = tmp_path / "hardware_info.json"
    # Day0 适配基准的硬件文件只保留 device/hardware_family。
    # details 不是输入契约的一部分；detect_hardware 内部可以为了兼容旧调用链
    # 补齐 details，但卡型判断必须以 hardware_family 为准。
    hardware_file.write_text(
        json.dumps({"device": "ascend", "hardware_family": "Ascend910B_64G"}),
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
    assert resolve_card_model(
        {"hardware_family": "G6550+RTX PRO 5000 * 8"}
    ) == "rtx_pro_5000_72G"
    assert resolve_card_model(
        {"hardware_family": "NVIDIA RTX PRO 5000 48GB Blackwell"}
    ) == "rtx_pro_5000_48G"


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
        "hardware_family": "TokenBox RTX PRO 5000 * 8",
    }) is True
    assert model_utils.is_deepseek_v4_flash_rtx_pro_5000({
        **source,
        "hardware_family": "NVIDIA RTX PRO 5000 48GB Blackwell",
    }) is False
    assert model_utils.is_deepseek_v4_flash_rtx_pro_5000({
        **source,
        "_smart_card_token": "rtxpro5000-72",
    }) is True


def test_minimax_m3_pro5000_detection_is_exact_to_m3_mxfp8():
    source = {
        "engine": "vllm",
        "model_name": "MiniMax/MiniMax-M3-MXFP8",
        "model_path": "/models/MiniMax/MiniMax-M3-MXFP8",
        "_smart_card_token": "rtxpro5000-72",
    }

    assert model_utils.is_minimax_m3_rtx_pro_5000_vllm(source) is True
    assert model_utils.is_minimax_m3_rtx_pro_5000_vllm({
        **source,
        "model_name": "MiniMax/MiniMax-M3",
        "model_path": "/models/MiniMax/MiniMax-M3",
    }) is False
    assert model_utils.is_minimax_m3_rtx_pro_5000_vllm({
        **source,
        "_smart_card_token": "rtxpro5000-48",
    }) is False


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
