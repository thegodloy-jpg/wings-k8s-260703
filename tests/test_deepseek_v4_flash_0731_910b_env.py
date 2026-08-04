import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402


MODEL_NAME = "DeepSeek-V4-Flash-0731-w8a8"


class _FakeDeepSeekV4Identifier:
    model_architecture = "DeepseekV4ForCausalLM"
    model_quantize = "w8a8"
    config = {}

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


def _render_model_env(monkeypatch, model_name, platform):
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", platform)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    card_name = "Ascend910B" if platform == "a2" else "Ascend910C"
    return vllm_adapter._build_model_env_commands(
        {
            "engine": "vllm_ascend",
            "model_name": model_name,
            "model_path": "/var/ai-model/DeepSeek-V4-Flash-w8a8/",
            "model_type": "llm",
            "device_details": [{"name": card_name}],
        },
        "vllm_ascend",
    )


def test_0731_w8a8_910b_uses_dedicated_six_variable_recipe(monkeypatch):
    assert _render_model_env(monkeypatch, MODEL_NAME, "a2") == [
        "export OMP_NUM_THREADS=10",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        'export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"',
        "export HCCL_BUFFSIZE=1024",
        "export TASK_QUEUE_ENABLE=1",
        'export HCCL_OP_EXPANSION_MODE="AIV"',
    ]


@pytest.mark.parametrize("quantize", ["ascend", "compressed-tensors"])
def test_0731_w8a8_910b_does_not_depend_on_generic_quant_metadata(
    monkeypatch, quantize
):
    monkeypatch.setattr(_FakeDeepSeekV4Identifier, "model_quantize", quantize)

    assert _render_model_env(monkeypatch, MODEL_NAME, "a2") == [
        "export OMP_NUM_THREADS=10",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        'export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"',
        "export HCCL_BUFFSIZE=1024",
        "export TASK_QUEUE_ENABLE=1",
        'export HCCL_OP_EXPANSION_MODE="AIV"',
    ]


def test_0731_w8a8_910b_final_env_filters_public_injections(monkeypatch):
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a3")
    monkeypatch.setattr(_FakeDeepSeekV4Identifier, "model_quantize", "ascend")
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    params = {
        "engine": "vllm_ascend",
        "model_name": MODEL_NAME.lower(),
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "_smart_card_token": "910b",
    }

    env = vllm_adapter._build_vllm_common_env_cmds(params, "vllm_ascend")

    assert not any("OMP_PROC_BIND" in command for command in env)
    assert not any("VLLM_ASCEND_ENABLE_FLASHCOMM1" in command for command in env)


@pytest.mark.parametrize(
    ("model_name", "served_model_name"),
    [
        ("DeepSeek-V4-Flash-0731-w8a8-Ascend910B", None),
        ("deployment-alias", MODEL_NAME),
    ],
)
def test_0731_w8a8_910b_accepts_exact_runtime_identity_fields(
    monkeypatch, model_name, served_model_name
):
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a2")
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    params = {
        "engine": "vllm_ascend",
        "model_name": model_name,
        "served_model_name": served_model_name,
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "device_details": [{"name": "Ascend910B"}],
    }

    env = vllm_adapter._build_vllm_common_env_cmds(params, "vllm_ascend")

    assert not any("OMP_PROC_BIND" in command for command in env)
    assert not any("VLLM_ASCEND_ENABLE_FLASHCOMM1" in command for command in env)


def test_0731_w8a8_910c_keeps_generic_flashcomm_recipe(monkeypatch):
    env = _render_model_env(monkeypatch, MODEL_NAME, "a3")

    assert "export OMP_PROC_BIND=false" in env
    assert "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1" in env


@pytest.mark.parametrize(
    "model_name",
    [
        "DeepSeek-V4-Flash-w8a8-mtp",
        "DeepSeek-V4-Flash-w8a8",
    ],
)
def test_non_0731_910b_models_keep_existing_generic_recipe(monkeypatch, model_name):
    env = _render_model_env(monkeypatch, model_name, "a2")

    assert "export OMP_PROC_BIND=false" in env
    assert "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1" in env
