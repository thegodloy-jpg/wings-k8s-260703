import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from utils import model_utils  # noqa: E402


def _reasoning_models_by_arch() -> dict[str, set[str]]:
    """返回 reason_parser.yaml 中按架构显式登记的模型名。

    `_LLM_MODELS` 是模型识别使用的全局目录，reason_parser.yaml 是启动时
    选择 parser 使用的目录。二者是独立文件，这个测试用于防止只把模型加入
    全局目录，却遗漏同步 reasoning parser 目录。
    """
    path = ROOT / "wings_control" / "docs" / "features" / "reasoning_parser" / "reason_parser.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        entry["name"]: set((entry.get("models") or {}).keys())
        for entry in data.get("architectures", [])
    }


def test_llm_catalog_models_are_registered_in_reasoning_parser_models():
    reasoning_models = _reasoning_models_by_arch()

    for arch, expected_models in model_utils._LLM_MODELS.items():
        if arch not in reasoning_models:
            continue
        assert set(expected_models) <= reasoning_models[arch]
