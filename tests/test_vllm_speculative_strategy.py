import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINGS_CONTROL = ROOT / "wings_control"
if str(WINGS_CONTROL) not in sys.path:
    sys.path.insert(0, str(WINGS_CONTROL))


from engines.vllm_adapter import resolve_speculative_strategy  # noqa: E402


def test_qwen35_vllm_ascend_none_draft_uses_engine_specific_mtp(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}),
        encoding="utf-8",
    )

    strategy = resolve_speculative_strategy(
        {
            "model_name": "Qwen3.6-27B-w8a8",
            "model_path": str(tmp_path),
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_feats": frozenset({"spec"}),
        },
        "vllm_ascend",
    )

    assert strategy == "qwen3_5_mtp"
