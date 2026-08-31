import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core import config_loader  # noqa: E402
from utils import model_utils  # noqa: E402


REASONING_SUPPORT_PATH = (
    ROOT
    / "wings_control"
    / "docs"
    / "features"
    / "reasoning_parser"
    / "reason_parser.yaml"
)
FUNCTION_CALL_SUPPORT_PATH = (
    ROOT
    / "wings_control"
    / "docs"
    / "features"
    / "function_call"
    / "function_call_support.yaml"
)


def _load_support(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _canonical_model_name(model_name: str) -> str:
    return model_name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].casefold()


def _reasoning_models() -> set[str]:
    """返回 reason_parser.yaml 中按完整模型名显式登记的规范化索引。

    `_LLM_MODELS` 是模型识别使用的全局目录，reason_parser.yaml 是启动时
    选择 parser 使用的目录。二者是独立文件，这个测试用于防止只把模型加入
    全局目录，却遗漏同步 reasoning parser 目录。
    """
    return {
        _canonical_model_name(model_name)
        for model_name in _reasoning_model_map()
    }


def _reasoning_model_map() -> dict[str, dict]:
    data = _load_support(REASONING_SUPPORT_PATH)
    result = {}
    for architecture in data["architectures"]:
        result.update(architecture["models"])
    return result


def test_llm_catalog_models_are_registered_in_reasoning_parser_models():
    reasoning_models = _reasoning_models()
    expected_models = {
        _canonical_model_name(model_name)
        for model_names in model_utils._LLM_MODELS.values()
        for model_name in model_names
    }

    assert expected_models <= reasoning_models


def test_deepseek_coder_v2_catalog_uses_model_config_architecture():
    assert "DeepSeek-Coder-V2-Instruct" in model_utils._LLM_MODELS["DeepseekV2ForCausalLM"]
    assert "DeepSeek-Coder-V2-Instruct" not in model_utils._LLM_MODELS["DeepseekV3ForCausalLM"]

    assert _canonical_model_name("DeepSeek-Coder-V2-Instruct") in _reasoning_models()


def test_reasoning_parser_support_preserves_schema_and_uses_exact_model_index():
    support = _load_support(REASONING_SUPPORT_PATH)
    reasoning_models = _reasoning_model_map()
    expected_target_models = {
        "Qwen3.8-27B": {"vllm": "qwen3"},
        "Qwen3.8-27B-FP8": {"vllm": "qwen3"},
        "Qwen3.8-27B-w8a8": {"vllm_ascend": "qwen3"},
        "DeepSeek-V4-Pro-0813": {"vllm": "deepseek_v4"},
        "DeepSeek-V4-Pro-0813-w4a8": {"vllm_ascend": "deepseek_v4"},
        "DeepSeek-V4-Flash-0731": {"vllm": "deepseek_v4"},
        "DeepSeek-V4-Flash-0731-w8a8": {"vllm_ascend": "deepseek_v4"},
        "Kimi-K3": {"vllm": "kimi_k3", "vllm_ascend": "kimi_k3"},
        "Kimi-K3-w4a8": {"vllm_ascend": "kimi_k3"},
    }

    assert support["feature"] == "reasoning_parser"
    assert support["field"] == "reasoning_parser"
    assert support["engines"] == ["vllm", "vllm_ascend"]
    assert "models" not in support
    assert isinstance(support["architectures"], list)
    assert all(
        isinstance(architecture.get("name"), str)
        and (
            "config" not in architecture
            or isinstance(architecture.get("config"), dict)
        )
        and isinstance(architecture.get("models"), dict)
        for architecture in support["architectures"]
    )
    for model_name, expected_engines in expected_target_models.items():
        assert reasoning_models[model_name] == expected_engines

    config_loader._load_reasoning_parser_support.cache_clear()
    # 组织名前缀、大小写和 distributed 后缀可以规范化；architecture 不参与判定。
    assert config_loader._resolve_reasoning_parser_support(
        "WrongArchitecture",
        "eco-tech/DEEPSEEK-v4-PRO-0813",
        "vllm_distributed",
    ) == (True, "deepseek_v4")
    # 同架构、相似后缀也不能继承已登记模型的 parser。
    assert config_loader._resolve_reasoning_parser_support(
        "DeepseekV4ForCausalLM",
        "DeepSeek-V4-Pro-0813-custom",
        "vllm",
    ) == (False, None)
    assert config_loader._apply_reasoning_parser_support(
        {"reasoning_parser": "deepseek_v4", "max_model_len": 4096},
        "DeepseekV4ForCausalLM",
        "DeepSeek-V4-Pro-0813-custom",
        "vllm",
    ) == {"max_model_len": 4096}


def test_target_function_call_support_uses_exact_model_name_index():
    support = _load_support(FUNCTION_CALL_SUPPORT_PATH)
    expected_models = {
        "Qwen3.8-27B": {"vllm": "qwen3_coder"},
        "Qwen3.8-27B-FP8": {"vllm": "qwen3_coder"},
        "Qwen3.8-27B-w8a8": {"vllm_ascend": "qwen3_coder"},
        "DeepSeek-V4-Pro-0813": {"vllm": "deepseek_v4"},
        "DeepSeek-V4-Pro-0813-w4a8": {"vllm_ascend": "deepseek_v4"},
        "DeepSeek-V4-Flash-0731": {"vllm": "deepseek_v4"},
        "DeepSeek-V4-Flash-0731-w8a8": {"vllm_ascend": "deepseek_v4"},
        "Kimi-K3": {"vllm": "kimi_k3"},
        "Kimi-K3-w4a8": {"vllm_ascend": "kimi_k3"},
    }

    assert support["feature"] == "function_call"
    assert support["field"] == "tool_call_parser"
    assert support["engines"] == ["vllm", "vllm_ascend"]
    assert support["models"] == expected_models

    config_loader._load_function_call_support.cache_clear()
    assert config_loader._resolve_function_call_support(
        "Qwen/QWEN3.8-27B-W8A8",
        "vllm_ascend_distributed",
    ) == (True, "qwen3_coder")
    assert config_loader._resolve_function_call_support(
        "Qwen3.8-27B-w8a8-custom",
        "vllm_ascend",
    ) == (False, None)
    assert config_loader._resolve_function_call_support(
        "Qwen3.8-27B-w8a8",
        "vllm",
    ) == (False, None)
    assert config_loader._apply_function_call_support(
        {"tool_call_parser": "legacy"},
        "Qwen3.8-27B-w8a8",
        "vllm",
    ) == {}
    assert config_loader._apply_function_call_support(
        {"tool_call_parser": "legacy"},
        "Unmigrated-Legacy-Model",
        "vllm",
    ) == {"tool_call_parser": "legacy"}
