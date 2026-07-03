import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from utils import model_utils  # noqa: E402


def test_default_smart_feature_whitelist_file_is_loaded():
    assert model_utils._SMART_WHITELIST_PATH.exists()

    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend",
        "DeepSeek-V4-Flash",
        "/models/DeepSeek-V4-Flash",
        "910c",
    ) == frozenset({"spec", "sparse", "offload"})
