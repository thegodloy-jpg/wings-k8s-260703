import json
import shlex
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines import vllm_adapter  # noqa: E402


_MODEL_NAME = "Qwen3.8-27B-FP8"
_MODEL_PATH = "/model"
_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_RTXPRO5000_72G = "NVIDIA RTX PRO 5000 72GB Blackwell"
_RUNTIME_ENV_NAMES = (
    "PD_ROLE",
    "ENGINE",
    "WINGS_ENGINE",
    "ENGINE_PORT",
    "PROXY_PORT",
    "PORT",
    "POD_IP",
    "RANK_IP",
    "SERVED_MODEL_NAME",
    "CONFIG_FORCE",
    "CONFIG_FILE",
    "ENABLE_AUTO_TOOL_CHOICE",
    "ENABLE_AUTO_THINK_CHOICE",
    "ENABLE_SPECULATIVE_DECODE",
    "TENSOR_PARALLEL_SIZE",
    "DATA_PARALLEL_SIZE",
    "MAX_NUM_SEQS",
    "ENABLE_SPARSE",
    "ENABLE_KV_OFFLOAD",
    "LMCACHE_OFFLOAD",
    "ENGINE_VERSION",
)


class _FakeQwen38Identifier:
    config = {}
    model_architecture = _ARCHITECTURE
    model_quantize = "fp8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type

    @staticmethod
    def identify_model_architecture():
        return _ARCHITECTURE

    @staticmethod
    def identify_model_type():
        return "llm"

    @staticmethod
    def is_wings_supported():
        return True


def _nvidia_arch_defaults():
    config_path = Path(config_loader.DEFAULT_CONFIG_DIR) / "nvidia_default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["model_deploy_config"]["llm"][_ARCHITECTURE]


def _hardware(card_name=_RTXPRO5000_72G, count=1):
    return {
        "device": "nvidia",
        "count": count,
        "details": [{"name": card_name}],
        "hardware_family": card_name,
    }


def _match_defaults(
    model_name=_MODEL_NAME,
    card_name=_RTXPRO5000_72G,
    count=1,
    engine="vllm",
):
    return config_loader._match_model_engine_config(
        _nvidia_arch_defaults(),
        model_name.lower(),
        engine,
        config_loader._SpecialEngineScenario(),
        _FakeQwen38Identifier(model_name, _MODEL_PATH, "llm"),
        _hardware(card_name, count),
    )


def _clear_runtime_env(monkeypatch):
    for name in _RUNTIME_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _render_command(
    monkeypatch,
    *,
    device_count=1,
    max_num_seqs=None,
    tensor_parallel_size=None,
    model_path=_MODEL_PATH,
    host="0.0.0.0",
    port=8001,
    served_model_name="Qwen/Qwen3.8-27B",
    enable_auto_tool_choice=True,
    enable_auto_think_choice=True,
):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("SERVED_MODEL_NAME", served_model_name)
    if max_num_seqs is not None:
        monkeypatch.setenv("MAX_NUM_SEQS", str(max_num_seqs))
    if tensor_parallel_size is not None:
        monkeypatch.setenv("TENSOR_PARALLEL_SIZE", str(tensor_parallel_size))
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "_record_selected_engine", lambda *_args: None)
    monkeypatch.setattr(config_loader, "check_pcie_cards", lambda *_args: (False, []))

    argv = [
        "--model-name",
        _MODEL_NAME,
        "--model-path",
        model_path,
        "--model-type",
        "llm",
        "--engine",
        "vllm",
        "--device-count",
        str(device_count),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if enable_auto_tool_choice:
        argv.append("--enable-auto-tool-choice")
    if enable_auto_think_choice:
        argv.append("--enable-auto-think-choice")
    launch_args = parse_launch_args(argv)
    params = config_loader.load_and_merge_configs(
        _hardware(count=device_count),
        launch_args,
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    shlex.split(exec_line, posix=True)
    return params, exec_line


def test_qwen38_fp8_rtxpro5000_profile_keeps_only_required_match_metadata():
    profile = _nvidia_arch_defaults()[_MODEL_NAME]

    assert profile == {
        "exact_model_names": ["qwen3.8-27b-fp8"],
        "card_tokens": ["rtxpro5000-72"],
        "vllm": {
            "use_vllm_serve": True,
            "trust_remote_code": True,
            "max_num_seqs": 512,
            "tool_call_parser": "qwen3_coder",
            "mm_encoder_tp_mode": "data",
        },
    }
    assert "device_counts" not in profile


@pytest.mark.parametrize("model_name", [_MODEL_NAME, f"Qwen/{_MODEL_NAME}"])
@pytest.mark.parametrize(
    "card_name",
    [_RTXPRO5000_72G, "NVIDIA RTX PRO 5000 Blackwell"],
)
@pytest.mark.parametrize("device_count", [1, 2, 8])
def test_qwen38_fp8_rtxpro5000_profile_does_not_restrict_visible_gpu_count(
    model_name,
    card_name,
    device_count,
):
    assert _match_defaults(
        model_name=model_name,
        card_name=card_name,
        count=device_count,
    ) == {
        "use_vllm_serve": True,
        "trust_remote_code": True,
        "max_num_seqs": 512,
        "tool_call_parser": "qwen3_coder",
        "mm_encoder_tp_mode": "data",
    }


@pytest.mark.parametrize(
    ("model_name", "card_name", "engine"),
    [
        ("Qwen3.8-27B", _RTXPRO5000_72G, "vllm"),
        ("Qwen3.8-27B-w8a8", _RTXPRO5000_72G, "vllm"),
        (_MODEL_NAME, "NVIDIA RTX PRO 5000 48GB Blackwell", "vllm"),
        (_MODEL_NAME, "NVIDIA H20 141GB", "vllm"),
        (_MODEL_NAME, _RTXPRO5000_72G, "vllm_distributed"),
    ],
)
def test_qwen38_fp8_rtxpro5000_profile_does_not_broaden(
    model_name,
    card_name,
    engine,
):
    assert _match_defaults(model_name=model_name, card_name=card_name, engine=engine) == {}


def test_qwen38_fp8_rtxpro5000_single_gpu_renders_reference_vllm_command(
    monkeypatch,
):
    params, exec_line = _render_command(monkeypatch)

    assert params["engine_config"]["tensor_parallel_size"] == 1
    assert exec_line == (
        "exec vllm serve /model"
        " --trust-remote-code"
        " --max-num-seqs 512"
        " --tool-call-parser qwen3_coder"
        " --mm-encoder-tp-mode data"
        " --reasoning-parser qwen3"
        " --host 0.0.0.0"
        " --port 8001"
        " --served-model-name Qwen/Qwen3.8-27B"
        " --enable-auto-tool-choice"
        " --default-chat-template-kwargs '{\"enable_thinking\":true}'"
        " --tensor-parallel-size 1"
    )
    for absent in (
        "--max-model-len",
        "--max-num-batched-tokens",
        "--quantization",
        "--enable-prefix-caching",
    ):
        assert absent not in exec_line


def test_qwen38_fp8_rtxpro5000_keeps_runtime_topology_and_explicit_override(
    monkeypatch,
):
    params, exec_line = _render_command(
        monkeypatch,
        device_count=2,
        max_num_seqs=64,
        tensor_parallel_size=1,
    )

    assert params["engine_config"]["tensor_parallel_size"] == 1
    assert params["engine_config"]["max_num_seqs"] == 64
    assert "--tensor-parallel-size 1" in exec_line
    assert "--max-num-seqs 64" in exec_line
    assert "--max-num-seqs 512" not in exec_line


def test_qwen38_fp8_rtxpro5000_keeps_runtime_endpoint_and_names(monkeypatch):
    _, exec_line = _render_command(
        monkeypatch,
        model_path="/runtime/models/qwen38",
        host="127.0.0.1",
        port=18000,
        served_model_name="runtime-qwen38",
    )

    assert exec_line.startswith("exec vllm serve /runtime/models/qwen38 ")
    assert "--host 127.0.0.1" in exec_line
    assert "--port 18000" in exec_line
    assert "--served-model-name runtime-qwen38" in exec_line


def test_qwen38_fp8_rtxpro5000_keeps_feature_switches_opt_in(monkeypatch):
    params, exec_line = _render_command(
        monkeypatch,
        enable_auto_tool_choice=False,
        enable_auto_think_choice=False,
    )

    assert "tool_call_parser" not in params["engine_config"]
    assert "reasoning_parser" not in params["engine_config"]
    assert "--enable-auto-tool-choice" not in exec_line
    assert "--tool-call-parser" not in exec_line
    assert "--reasoning-parser" not in exec_line
    assert "--default-chat-template-kwargs '{\"enable_thinking\":false}'" in exec_line
