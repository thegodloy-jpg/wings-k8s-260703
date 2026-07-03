import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

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


def test_deepseek_v4_flash_pro5000_vllm_speculative_config_uses_mtp_num2(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.setenv("WINGS_DEVICE_NAME", "NVIDIA RTX PRO 5000 72GB Blackwell")

    command = vllm_adapter.build_speculative_cmd(
        {
            "engine": "vllm",
            "model_name": "Deepseek-v4-Flash",
            "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec", "sparse"],
        },
        "vllm",
    )

    assert '"method": "mtp"' in command
    assert '"num_speculative_tokens": 2' in command
    assert '"enforce_eager": true' in command


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
