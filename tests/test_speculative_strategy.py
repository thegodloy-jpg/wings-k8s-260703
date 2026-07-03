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
