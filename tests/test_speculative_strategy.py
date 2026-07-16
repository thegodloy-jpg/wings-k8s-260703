import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import wings_entry  # noqa: E402
from engines import vllm_adapter  # noqa: E402


class _FakeModelIdentifier:
    model_architecture = "Qwen3_5ForConditionalGeneration"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeDeepSeekV4Identifier:
    model_architecture = "DeepseekV4ForCausalLM"
    model_quantize = "w8a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeGlm51Identifier:
    model_architecture = "GlmMoeDsaForCausalLM"
    model_quantize = "w8a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeGlm47Identifier:
    model_architecture = "Glm4MoeForCausalLM"
    model_quantize = "w8a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeKimiIdentifier:
    model_architecture = "KimiK25ForConditionalGeneration"
    model_quantize = "w4a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeQwen36MoeIdentifier:
    model_architecture = "Qwen3_5MoeForConditionalGeneration"
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeMiniMaxM2Identifier:
    model_architecture = "MiniMaxM2ForCausalLM"
    model_quantize = "w8a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeUnknownGlm51Identifier:
    model_architecture = "unknown_architecture"
    model_quantize = "w8a8"
    config = {}

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeQwen2Identifier:
    model_architecture = "Qwen2ForCausalLM"
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakePro5000Identifier:
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        text = f"{model_name} {model_path}".lower()
        if "qwen3.5-397b-a17b-nvfp4" in text:
            self.model_architecture = "Qwen3_5MoeForConditionalGeneration"
            self.model_quantize = "nvfp4"
        elif "qwen-agentworld-35b-a3b" in text:
            self.model_architecture = "Qwen3_5MoeForConditionalGeneration"
        elif "qwen3.5-122b-a10b" in text or "qwen3.5-35b-a3b" in text:
            self.model_architecture = "Qwen3_5MoeForConditionalGeneration"
        elif "qwen3.5-27b" in text:
            self.model_architecture = "Qwen3_5ForConditionalGeneration"
        elif "deepseek-v4-flash" in text:
            self.model_architecture = "DeepseekV4ForCausalLM"
            self.model_quantize = "fp4"
        elif "minimax-m3-mxfp8" in text:
            self.model_architecture = "MiniMaxM3SparseForConditionalGeneration"
        elif "minimax-m2.5-nvfp4" in text:
            self.model_architecture = "MiniMaxM2ForCausalLM"
        elif "minimax-m2.7-nvfp4" in text:
            self.model_architecture = "MiniMaxM2ForCausalLM"
        else:
            self.model_architecture = "unknown_architecture"


def test_resolve_speculative_strategy_passes_engine_to_mtp_method(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeModelIdentifier)

    strategy = vllm_adapter.resolve_speculative_strategy(
        {
            "model_name": "Qwen3.6-27B-w8a8",
            "model_path": "/usr/local/serving/models/",
            "model_type": "llm",
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec"],
        },
        "vllm_ascend",
    )

    assert strategy == "qwen3_5_mtp"


def test_qwen_day0_memcache_keeps_mtp_strategy_and_whitelist_tokens(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeModelIdentifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    command = vllm_adapter.build_speculative_cmd(
        {
            "engine": "vllm_ascend",
            "model_name": "Qwen/Qwen3.5-27B",
            "model_path": "/models/Qwen/Qwen3.5-27B",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_card_token": "910c",
            "_smart_feats": ["offload", "spec"],
        },
        "vllm_ascend",
    )

    assert '"method": "qwen3_5_mtp"' in command
    assert '"num_speculative_tokens": 1' in command
    assert '"enforce_eager": true' in command
    assert "suffix" not in command


def test_deepseek_v4_flash_ascend_speculative_config_uses_vllm_021_mtp(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)

    command = vllm_adapter.build_speculative_cmd(
        {
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec"],
        },
        "vllm_ascend",
    )

    assert '"method": "mtp"' in command
    assert '"num_speculative_tokens": 1' in command
    assert '"enforce_eager": true' in command
    assert "deepseek_mtp" not in command


def test_deepseek_v4_flash_adapter_does_not_recreate_json_owned_runtime_defaults(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")

    engine_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 16,
            "engine_config": {},
        }
    )

    for key in (
        "quantization",
        "block_size",
        "async_scheduling",
        "safetensors_load_strategy",
        "tokenizer_mode",
        "tool_call_parser",
        "enable_auto_tool_choice",
    ):
        assert key not in engine_config


def test_deepseek_v4_flash_topology_prefers_hardware_info_over_engine_version(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_IMAGE_FLAVOR", raising=False)
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a2")

    engine_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 16,
            "device_details": [{"name": "Ascend910C"}],
            "engine_config": {},
        }
    )

    assert engine_config["tensor_parallel_size"] == 4
    assert engine_config["data_parallel_size"] == 4
    assert engine_config["api_server_count"] == 1


def test_deepseek_v4_flash_topology_keeps_engine_version_platform_fallback(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_IMAGE_FLAVOR", raising=False)
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a3")

    engine_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 16,
            "engine_config": {},
        }
    )

    assert engine_config["tensor_parallel_size"] == 4
    assert engine_config["data_parallel_size"] == 4
    assert engine_config["api_server_count"] == 1


def test_deepseek_v4_flash_topology_ignores_explicit_platform_env(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a2")

    engine_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 16,
            "engine_config": {},
        }
    )

    assert engine_config["tensor_parallel_size"] == 8
    assert engine_config["data_parallel_size"] == 2
    assert "api_server_count" not in engine_config


def test_deepseek_v4_flash_pro5000_vllm_speculative_config_matches_tokenbox(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)

    command = vllm_adapter.build_speculative_cmd(
        {
            "engine": "vllm",
            "model_name": "Deepseek-v4-Flash",
            "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec", "sparse"],
            "_smart_card_token": "rtxpro5000-72",
        },
        "vllm",
    )

    assert '"method": "mtp"' in command
    assert '"num_speculative_tokens": 2' in command
    assert '"enforce_eager": true' not in command


def test_qwen35_nvfp4_native_offload_keeps_mtp_strategy(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeModelIdentifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    strategy = vllm_adapter.resolve_speculative_strategy(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "model_type": "llm",
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert strategy == "mtp"


def test_qwen35_nvfp4_pro5000_speculative_config_matches_tokenbox(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)

    command = vllm_adapter.build_speculative_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec", "offload"],
            "_smart_card_token": "rtxpro5000-72",
        },
        "vllm",
    )

    assert command == ' --speculative-config \'{"method":"mtp","num_speculative_tokens":3}\''


@pytest.mark.parametrize(
    ("model_name", "model_path", "expected_tokens"),
    [
        ("Qwen/Qwen3.5-122B-A10B", "/models/Qwen/Qwen3.5-122B-A10B", 1),
        ("Qwen/Qwen3.5-27B", "/models/Qwen/Qwen3.5-27B", 2),
    ],
)
def test_qwen35_pro5000_mtp_uses_whitelist_method_and_tokens(
    monkeypatch,
    model_name,
    model_path,
    expected_tokens,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)

    params = {
        "engine": "vllm",
        "model_name": model_name,
        "model_path": model_path,
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec", "offload"],
        "_smart_card_token": "rtxpro5000-72",
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert command == (
        " --speculative-config "
        f"'{{\"method\":\"mtp\",\"num_speculative_tokens\":{expected_tokens}}}'"
    )
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "mtp",
        "num_speculative_tokens": expected_tokens,
        "moe_backend": None,
    }


@pytest.mark.parametrize(
    ("model_name", "model_path", "expected_tokens"),
    [
        ("MiniMax/MiniMax-M3-MXFP8", "/models/MiniMax/MiniMax-M3-MXFP8", 32),
        ("MiniMax/MiniMax-M2.5-NVFP4", "/models/MiniMax/MiniMax-M2.5-NVFP4", 10),
        ("MiniMax/MiniMax-M2.7-NVFP4", "/models/MiniMax/MiniMax-M2.7-NVFP4", 10),
    ],
)
def test_minimax_pro5000_suffix_uses_whitelist_tokens(
    monkeypatch,
    model_name,
    model_path,
    expected_tokens,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)

    params = {
        "engine": "vllm",
        "model_name": model_name,
        "model_path": model_path,
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec", "offload"],
        "_smart_card_token": "rtxpro5000-72",
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert command == (
        " --speculative-config "
        f"'{{\"method\":\"suffix\",\"num_speculative_tokens\":{expected_tokens}}}'"
    )
    assert "suffix_decoding_max_cached_requests" not in command
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "suffix",
        "num_speculative_tokens": expected_tokens,
        "moe_backend": None,
    }


def test_qwen_agentworld_pro5000_suffix_uses_whitelist_32_tokens(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)

    params = {
        "engine": "vllm",
        "model_name": "Qwen/Qwen-AgentWorld-35B-A3B",
        "model_path": "/models/Qwen/Qwen-AgentWorld-35B-A3B",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec"],
        "_smart_card_token": "rtxpro5000-72",
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm") == "suffix"
    assert command == (
        " --speculative-config "
        "'{\"method\":\"suffix\",\"num_speculative_tokens\":32}'"
    )
    assert "suffix_decoding_max_cached_requests" not in command
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "suffix",
        "num_speculative_tokens": 32,
        "moe_backend": None,
    }


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_minimax_m27_w8a8_quarot_eagle3_uses_draft_path_options(
    monkeypatch,
    tmp_path,
    card_token,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Identifier)
    draft_dir = tmp_path / "Eagle3"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(
        json.dumps({"architectures": ["MiniMaxM2Eagle3ForCausalLM"]}),
        encoding="utf-8",
    )
    params = {
        "engine": "vllm_ascend",
        "model_name": "MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_path": "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": str(draft_dir),
        "_smart_feats": ["spec"],
        "_smart_card_token": card_token,
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "eagle3"
    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")
    body = command.split("'", 2)[1]
    config = json.loads(body)

    assert config["method"] == "eagle3"
    assert config["model"] == str(draft_dir)
    assert config["draft_tensor_parallel_size"] == 1
    assert config["num_speculative_tokens"] == 3
    assert config["enforce_eager"] is True


def test_minimax_m27_w8a8_quarot_without_draft_falls_back_to_suffix(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_path": "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec"],
        "_smart_card_token": "910c",
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    assert '"method" : "suffix"' in command
    assert "eagle3" not in command
    assert "draft_model" not in command


def test_qwen35_35b_pro5000_offload_only_does_not_auto_append_spec(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)
    params = {
        "engine": "vllm",
        "model_name": "Qwen/Qwen3.5-35B-A3B",
        "model_path": "/models/Qwen/Qwen3.5-35B-A3B",
        "model_type": "llm",
        "enable_speculative_decode": False,
        "_smart_feats": ["offload"],
        "_smart_card_token": "rtxpro5000-72",
    }

    script = vllm_adapter._build_vllm_single_script(
        params,
        "vllm serve /models/Qwen/Qwen3.5-35B-A3B",
        [],
        "vllm",
        "",
    )

    assert vllm_adapter.should_append_auto_speculative_config(params) is False
    assert "--speculative-config" not in script


def test_nvidia_day0_glm47_mtp_uses_whitelist_method_and_ignores_draft(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm47Identifier)

    params = {
        "engine": "vllm",
        "model_name": "zai-org/GLM-4.7",
        "model_path": "/models/zai-org/GLM-4.7",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "/models/old-draft",
        "_smart_card_token": "h20-141",
        "_smart_feats": ["spec"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm") == "mtp"
    command = vllm_adapter.build_speculative_cmd(params, "vllm")
    assert command == " --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'"
    assert "old-draft" not in command


def test_ascend_glm47_mtp_only_does_not_fall_back_to_suffix(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm47Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/GLM-4.7-w8a8-floatmtp",
        "model_path": "/models/Eco-Tech/GLM-4.7-w8a8-floatmtp",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "_smart_card_token": "910c",
        "_smart_feats": ["spec"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "mtp"
    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")
    assert command == (
        " --speculative-config "
        "'{\"method\":\"mtp\",\"num_speculative_tokens\":3,"
        "\"speculative_token_range\":\"256,512\"}'"
    )


def test_ascend_deepseek_v4_pro_mtp_uses_exact_whitelist_enforce_eager(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "_smart_card_token": "910c",
        "_smart_feats": ["spec", "sparse"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "mtp"
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == (
        " --speculative-config "
        "'{\"method\":\"mtp\",\"num_speculative_tokens\":1,\"enforce_eager\":true}'"
    )
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm_ascend") == {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "moe_backend": None,
        "enforce_eager": True,
    }


def test_kimi_k26_uses_suffix_fallback_without_dflash_draft(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeKimiIdentifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/Kimi-K2.6-W4A8",
        "model_path": "/models/Eco-Tech/Kimi-K2.6-W4A8",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_card_token": "910c",
        "_smart_feats": ["spec", "offload"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == (
        " --speculative-config "
        "'{\"method\" : \"suffix\", \"num_speculative_tokens\": 5, "
        "\"suffix_decoding_max_cached_requests\": 1000}'"
    )

    params["speculative_decode_model_path"] = "z-lab/Kimi-K2.6-NonDFlash-Draft"
    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    assert "draft_model" not in vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    params["speculative_decode_model_path"] = "z-lab/Kimi-K2.6-DFlash"
    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "dflash"
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == (
        " --speculative-config "
        "'{\"method\":\"dflash\",\"model\":\"z-lab/Kimi-K2.6-DFlash\","
        "\"num_speculative_tokens\":15}'"
    )

    params["_smart_feats"] = ["offload"]
    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == ""
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == ""


def test_kimi_k27_code_does_not_inherit_k26_dflash(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeKimiIdentifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "Kimi-K2.7-Code",
        "model_path": "/harbor_data/Kimi-K2.7-Code",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "z-lab/Kimi-K2.6-DFlash",
        "_smart_card_token": "910c",
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == ""
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == ""


def test_nvidia_day0_qwen35_mtp_uses_whitelist_moe_backend(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen36MoeIdentifier)

    params = {
        "engine": "vllm",
        "model_name": "Qwen3.6-35B-A3B",
        "model_path": "/models/Qwen3.6-35B-A3B",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "_smart_card_token": "l20",
        "_smart_feats": ["spec"],
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert command == (
        " --speculative-config "
        "'{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"moe_backend\":\"triton\"}'"
    )
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "moe_backend": "triton",
    }


def test_nvidia_day0_native_offload_uses_whitelist_backend(monkeypatch):
    monkeypatch.delenv("CONFIG_FORCE", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "20")
    params = {
        "engine": "vllm",
        "model_name": "Qwen3.6-27B",
        "model_path": "/models/Qwen3.6-27B",
        "model_type": "llm",
        "_smart_card_token": "l20",
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter._build_kv_offload_cmd(params, "vllm") == (
        " --kv-offloading-backend native --kv-offloading-size 20"
    )
    assert vllm_adapter.resolve_offload_variant(params, "vllm") == (
        "native_kv_offloading_backend"
    )
    assert vllm_adapter.resolve_effective_kv_mem_offload_size(params, "vllm") == 20


@pytest.mark.parametrize("raw_size", ["", "invalid", "0", "-1"])
def test_nvidia_day0_native_offload_discards_invalid_page_size(monkeypatch, raw_size):
    monkeypatch.delenv("CONFIG_FORCE", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", raw_size)
    params = {
        "engine": "vllm",
        "model_name": "Qwen3.6-27B",
        "model_path": "/models/Qwen3.6-27B",
        "model_type": "llm",
        "_smart_card_token": "l20",
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter._build_kv_offload_cmd(params, "vllm") == ""
    assert vllm_adapter.resolve_offload_variant(params, "vllm") == "disabled"
    assert vllm_adapter.resolve_effective_kv_mem_offload_size(params, "vllm") is None


def test_nvidia_day0_native_offload_config_force_bypasses_whitelist(monkeypatch):
    monkeypatch.setenv("CONFIG_FORCE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "20")
    params = {
        "engine": "vllm",
        "model_name": "Qwen3.6-27B",
        "model_path": "/models/Qwen3.6-27B",
        "model_type": "llm",
        "_smart_card_token": "l20",
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter._build_kv_offload_cmd(params, "vllm") == ""


def test_advanced_feature_fallback_removes_embedded_speculative_config(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)
    monkeypatch.setattr(wings_entry, "ModelIdentifier", _FakePro5000Identifier)
    monkeypatch.setattr(
        wings_entry,
        "start_engine_service",
        lambda merged: vllm_adapter.build_start_script(merged),
    )
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    merged = {
        "engine": "vllm",
        "model_name": "Qwen3.5-397B-A17B-NVFP4",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "enable_sparse": False,
        "_smart_feats": ["offload", "spec"],
        "engine_config": {
            "model": "/usr/local/serving/models/",
            "served_model_name": "Qwen3.5-397B-A17B-NVFP4",
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
            "tensor_parallel_size": 8,
        },
    }

    fallback_cmd = wings_entry._build_advanced_feature_fallback_cmd(merged)

    assert "--speculative-config" not in fallback_cmd
    assert "--kv-offloading-backend" not in fallback_cmd
    assert merged["engine_config"]["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }


def test_pro5000_spec_models_do_not_emit_ears_env_or_runtime_deps(monkeypatch):
    # 这个测试同时保护两条边界：
    # - 旧的 EARS/install-runtime-deps 补丁不能借 spec 场景回流；
    # - DeepSeek-V4-Flash + Pro5000 的新依赖安装独立于 spec/sparse/offload，命中即生成。
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)
    monkeypatch.setattr(wings_entry, "ModelIdentifier", _FakePro5000Identifier)
    monkeypatch.setenv("ENGINE_VERSION", "v0.23.0")

    scenarios = [
        {
            "engine": "vllm",
            "model_name": "deepseek-ai/DeepSeek-V4-Flash",
            "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "enable_sparse": True,
            "speculative_decode_model_path": "none",
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "sparse"],
            "installs_deepseek_packages": True,
        },
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "enable_sparse": False,
            "speculative_decode_model_path": "none",
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
            "installs_deepseek_packages": False,
        },
    ]

    for params in scenarios:
        env_commands = vllm_adapter._build_speculative_env_commands(params, "vllm")
        accel_preamble = wings_entry._build_accel_preamble("vllm", params)

        assert env_commands == []
        assert "install-runtime-deps" not in accel_preamble
        assert '"ears"' not in accel_preamble
        assert "VLLM_EARS_TOLERANCE" not in accel_preamble
        if params["installs_deepseek_packages"]:
            assert "python3 install.py --config" in accel_preamble
            assert "deepgemm:nv_dev_a6b593d" in accel_preamble
            assert "flashinfer:v0.6.12" in accel_preamble
        else:
            assert "install.py" not in accel_preamble


def test_spec_request_without_whitelist_generates_suffix_config(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": [],
    }

    assert vllm_adapter.should_append_auto_speculative_config(params) is True

    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    assert '"method" : "suffix"' in command
    assert '"num_speculative_tokens": 5' in command


@pytest.mark.parametrize("draft_path", ["none", "None", " none ", '"none"', "'none'", "null"])
def test_spec_draft_sentinel_values_do_not_generate_draft_model(monkeypatch, draft_path):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": draft_path,
        "_smart_feats": [],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    assert '"method" : "suffix"' in command
    assert '"method" : "draft_model"' not in command
    assert '"model"' not in command


def test_glm51_ascend_spec_whitelist_uses_native_mtp(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    params = {
        "engine": "vllm_ascend",
        "model_name": "GLM-5.1-w8a8",
        "model_path": "/models/GLM-5.1-w8a8",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["sparse", "spec"],
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    assert '"method": "deepseek_mtp"' in command
    assert '"num_speculative_tokens": 3' in command
    assert "suffix" not in command


@pytest.mark.parametrize(
    ("identifier_cls", "expected_strategy"),
    [
        (_FakeGlm51Identifier, "deepseek_mtp"),
        (_FakeGlm47Identifier, "glm4_moe_mtp"),
    ],
)
def test_auto_floor_discarded_offload_keeps_mtp_strategy(
    monkeypatch,
    identifier_cls,
    expected_strategy,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", identifier_cls)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "102400")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")

    strategy = vllm_adapter.resolve_speculative_strategy(
        {
            "engine": "vllm_ascend",
            "model_name": "GLM-5.1-w8a8",
            "model_path": "/models/GLM-5.1-w8a8",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["offload", "spec"],
        },
        "vllm_ascend",
    )

    assert strategy == expected_strategy


def test_auto_floor_with_disk_offload_still_uses_suffix_guard(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "102400")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "true")
    monkeypatch.setenv("KV_DISK_OFFLOAD_PATH", "/mnt/kvcache_offload")
    monkeypatch.setenv("KV_DISK_OFFLOAD_SIZE", "8")

    strategy = vllm_adapter.resolve_speculative_strategy(
        {
            "engine": "vllm_ascend",
            "model_name": "GLM-5.1-w8a8",
            "model_path": "/models/GLM-5.1-w8a8",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["offload", "spec"],
        },
        "vllm_ascend",
    )

    assert strategy == "suffix"


def test_glm51_roce_distributed_engine_config_uses_official_mtp_num3(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)
    monkeypatch.setattr(vllm_adapter, "_is_roce_distributed", lambda: True)

    params = {
        "engine": "vllm_ascend",
        "model_name": "GLM-5.1-w8a8",
        "model_path": "/models/GLM-5.1-w8a8",
        "model_type": "llm",
        "distributed": True,
        "distributed_executor_backend": "dp_deployment",
        "enable_speculative_decode": True,
        "engine_config": {
            "async_scheduling": True,
            "enable_expert_parallel": True,
        },
    }

    engine_config = vllm_adapter._prepare_engine_config(params)

    assert engine_config["speculative_config"] == {
        "num_speculative_tokens": 3,
        "method": "deepseek_mtp",
    }
    assert params["engine_config"]["speculative_config"] == {
        "num_speculative_tokens": 3,
        "method": "deepseek_mtp",
    }
    assert "async_scheduling" not in engine_config
    assert "enable_expert_parallel" not in engine_config


def test_glm51_ascend_indexcache_hf_overrides_enable_index_cache(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)

    command = vllm_adapter._build_kv_sparse_cmd(
        {
            "engine": "vllm_ascend",
            "model_name": "glm-5.1-32b-chat",
            "model_path": "/models/glm-5.1-32b-chat",
            "model_type": "llm",
            "_smart_feats": ["sparse"],
        },
        "vllm_ascend",
    )

    assert '--hf-overrides' in command
    assert '"use_index_cache": true' in command
    assert '"index_topk_freq": 8' in command

    variant = vllm_adapter.resolve_sparse_variant(
        {
            "engine": "vllm_ascend",
            "model_name": "glm-5.1-32b-chat",
            "model_path": "/models/glm-5.1-32b-chat",
            "model_type": "llm",
        },
        "vllm_ascend",
    )

    assert variant == "indexcache_use_index_cache_topk8"


def test_glm51_ascend_indexcache_uses_model_name_when_config_missing(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeUnknownGlm51Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "GLM-5.1-w8a8",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "_smart_feats": ["sparse"],
    }

    command = vllm_adapter._build_kv_sparse_cmd(params, "vllm_ascend")

    assert '--hf-overrides' in command
    assert '"use_index_cache": true' in command
    assert '"index_topk_freq": 8' in command

    variant = vllm_adapter.resolve_sparse_variant(params, "vllm_ascend")

    assert variant == "indexcache_use_index_cache_topk8"


def test_sparse_whitelist_command_and_variant_share_plan(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)
    monkeypatch.setattr(
        vllm_adapter,
        "resolve_feature_whitelist_row_from_params",
        lambda *_args: {"strategy": "indexcache"},
    )
    monkeypatch.setattr(vllm_adapter, "_resolve_sparse_topk", lambda *_args, **_kwargs: 7)
    params = {
        "model_name": "Qwen2",
        "model_path": "/models/Qwen2",
        "model_type": "llm",
    }

    command = vllm_adapter._build_kv_sparse_cmd(params, "vllm")
    variant = vllm_adapter.resolve_sparse_variant(params, "vllm")

    assert command == ' --hf-overrides \'{"use_index_cache":true,"index_topk_freq":7}\''
    assert variant == "indexcache_use_index_cache_topk7"


def test_sparse_fp8_variant_is_pure_while_command_applies_shared_plan(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)
    monkeypatch.setattr(vllm_adapter, "INDEXCACHE_ARCHS", set())
    monkeypatch.setattr(
        vllm_adapter,
        "resolve_feature_whitelist_row_from_params",
        lambda *_args: None,
    )
    monkeypatch.setattr(vllm_adapter, "_is_deepseek_v4_flash_params", lambda *_args: False)
    monkeypatch.setattr(
        vllm_adapter,
        "is_minimax_m27_rtx_pro_5000_vllm",
        lambda *_args: False,
    )
    params = {
        "model_name": "Qwen2",
        "model_path": "/models/Qwen2",
        "model_type": "llm",
        "engine_config": {},
    }

    assert vllm_adapter.resolve_sparse_variant(params, "vllm") == "fp8"
    assert params["engine_config"] == {}
    assert vllm_adapter._build_kv_sparse_cmd(params, "vllm") == ""
    assert params["engine_config"] == {
        "kv_cache_dtype": "fp8",
    }


def test_glm51_ascend_preserves_explicit_expert_parallel_in_command(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)

    command = vllm_adapter.build_start_command(
        {
            "engine": "vllm_ascend",
            "model_name": "GLM-5.1-w8a8",
            "model_path": "/models/GLM-5.1-w8a8",
            "model_type": "llm",
            "engine_config": {"enable_expert_parallel": True},
            "_explicit_cli_keys": ["enable_expert_parallel"],
        },
    )

    assert " --enable-expert-parallel" in f" {command}"
