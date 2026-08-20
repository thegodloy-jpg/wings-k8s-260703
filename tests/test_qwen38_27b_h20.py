import json
import shlex
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines import vllm_adapter  # noqa: E402
from utils import model_utils  # noqa: E402


_MODEL_NAME = "Qwen3.8-27B"
_MODEL_PATH = "/model"
_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_H20_CARDS = (
    "NVIDIA H20 96GB",
    "NVIDIA H20 141GB",
)
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
    "SD_ENABLE",
    "TENSOR_PARALLEL_SIZE",
    "DATA_PARALLEL_SIZE",
    "ENABLE_SPARSE",
    "ENABLE_KV_OFFLOAD",
    "LMCACHE_OFFLOAD",
    "ENGINE_VERSION",
)


class _FakeQwen38Identifier:
    config = {}
    model_architecture = _ARCHITECTURE
    model_quantize = ""

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


def _hardware(card_name=_H20_CARDS[0], count=1):
    return {
        "device": "nvidia",
        "count": count,
        "details": [{"name": card_name}],
        "hardware_family": card_name,
    }


def _match_defaults(
    *,
    model_name=_MODEL_NAME,
    card_name=_H20_CARDS[0],
    count=1,
    engine_key="vllm",
):
    return config_loader._match_model_engine_config(
        _nvidia_arch_defaults(),
        model_name.lower(),
        engine_key,
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
    tensor_parallel_size=None,
    config_file=None,
    enable_auto_tool_choice=True,
    enable_auto_think_choice=True,
    enable_speculative_decode=True,
):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("SERVED_MODEL_NAME", "Qwen/Qwen3.8-27B")
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
        _MODEL_PATH,
        "--model-type",
        "llm",
        "--engine",
        "vllm",
        "--device-count",
        str(device_count),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    if config_file is not None:
        argv.extend(["--config-file", str(config_file)])
    if enable_auto_tool_choice:
        argv.append("--enable-auto-tool-choice")
    if enable_auto_think_choice:
        argv.append("--enable-auto-think-choice")
    if enable_speculative_decode:
        argv.append("--enable-speculative-decode")

    params = config_loader.load_and_merge_configs(
        _hardware(count=device_count),
        parse_launch_args(argv),
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    shlex.split(exec_line, posix=True)
    return params, exec_line


def test_qwen38_27b_h20_profile_is_static_and_topology_free():
    profile = _nvidia_arch_defaults()["Qwen3.8-27B-H20"]
    expected_engine_config = {
        "use_vllm_serve": True,
        "tool_call_parser": "qwen3_coder",
        "mm_encoder_tp_mode": "data",
    }

    assert profile == {
        "exact_model_names": ["qwen3.8-27b"],
        "card_tokens": ["h20-96", "h20-141"],
        "vllm": expected_engine_config,
        "vllm_distributed": expected_engine_config,
    }
    assert "device_counts" not in profile
    for engine_key in ("vllm", "vllm_distributed"):
        assert not {
            "tensor_parallel_size",
            "data_parallel_size",
            "pipeline_parallel_size",
            "distributed_executor_backend",
        } & profile[engine_key].keys()


@pytest.mark.parametrize("model_name", [_MODEL_NAME, f"Qwen/{_MODEL_NAME}"])
@pytest.mark.parametrize("card_name", _H20_CARDS)
@pytest.mark.parametrize("device_count", [1, 2, 4, 8])
@pytest.mark.parametrize("engine_key", ["vllm", "vllm_distributed"])
def test_qwen38_27b_h20_profile_does_not_restrict_topology(
    model_name,
    card_name,
    device_count,
    engine_key,
):
    assert _match_defaults(
        model_name=model_name,
        card_name=card_name,
        count=device_count,
        engine_key=engine_key,
    ) == {
        "use_vllm_serve": True,
        "tool_call_parser": "qwen3_coder",
        "mm_encoder_tp_mode": "data",
    }


@pytest.mark.parametrize(
    ("model_name", "card_name"),
    [
        ("Qwen3.8-27B-FP8", _H20_CARDS[0]),
        ("Qwen3.8-27B-w8a8", _H20_CARDS[0]),
        (_MODEL_NAME, "NVIDIA H100 80GB"),
        (_MODEL_NAME, "NVIDIA RTX PRO 5000 72GB Blackwell"),
    ],
)
def test_qwen38_27b_h20_profile_does_not_broaden(model_name, card_name):
    assert _match_defaults(model_name=model_name, card_name=card_name) == {}


@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
def test_qwen38_27b_h20_spec_row_uses_exact_mtp3_without_topology_fields(card_token):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        "Qwen/Qwen3.8-27B",
        "/models/Qwen/Qwen3.8-27B",
        card_token,
        "spec",
    )

    assert row is not None
    assert row["mtp_method"] == "mtp"
    assert row["mtp_num_speculative_tokens"] == 3
    assert "enforce_eager" not in row
    assert not {"device_counts", "tensor_parallel_size", "data_parallel_size", "nnodes"} & row.keys()


@pytest.mark.parametrize("model_name", ["Qwen3.8-27B-FP8", "Qwen3.8-27B-w8a8"])
def test_qwen38_27b_h20_spec_row_excludes_quantized_name_suffixes(model_name):
    assert model_utils.resolve_feature_whitelist_row(
        "vllm",
        model_name,
        f"/models/{model_name}",
        "h20-96",
        "spec",
    ) is None


def test_qwen38_27b_h20_single_gpu_renders_reference_command(monkeypatch):
    params, exec_line = _render_command(monkeypatch)

    assert params["engine_config"]["tensor_parallel_size"] == 1
    assert exec_line == (
        "exec vllm serve /model"
        " --tool-call-parser qwen3_coder"
        " --mm-encoder-tp-mode data"
        " --reasoning-parser qwen3"
        " --host 0.0.0.0"
        " --port 8000"
        " --served-model-name Qwen/Qwen3.8-27B"
        " --enable-auto-tool-choice"
        " --default-chat-template-kwargs '{\"enable_thinking\":true}'"
        " --tensor-parallel-size 1"
        " --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
    )
    for absent in (
        "--trust-remote-code",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--quantization",
        "--enable-prefix-caching",
        "--enforce-eager",
    ):
        assert absent not in exec_line


@pytest.mark.parametrize("device_count", [1, 2, 4, 8])
def test_qwen38_27b_h20_uses_generic_dynamic_tp(monkeypatch, device_count):
    params, exec_line = _render_command(monkeypatch, device_count=device_count)

    assert params["engine_config"]["tensor_parallel_size"] == device_count
    assert f"--tensor-parallel-size {device_count}" in exec_line
    assert "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'" in exec_line


def test_qwen38_27b_h20_preserves_explicit_tp_and_dp(monkeypatch, tmp_path):
    # data_parallel_size 没有 launcher CLI，沿用项目现有的 engine-native
    # config-file 入口；模型 profile 只声明静态能力，不得覆盖用户拓扑。
    config_file = tmp_path / "qwen38-h20-runtime.json"
    config_file.write_text(json.dumps({"data_parallel_size": 2}), encoding="utf-8")
    params, exec_line = _render_command(
        monkeypatch,
        device_count=4,
        tensor_parallel_size=2,
        config_file=config_file,
    )

    assert params["engine_config"]["tensor_parallel_size"] == 2
    assert params["engine_config"]["data_parallel_size"] == 2
    assert "--tensor-parallel-size 2" in exec_line
    assert "--data-parallel-size 2" in exec_line


def test_qwen38_27b_h20_keeps_feature_switches_opt_in(monkeypatch):
    params, exec_line = _render_command(
        monkeypatch,
        enable_auto_tool_choice=False,
        enable_auto_think_choice=False,
        enable_speculative_decode=False,
    )

    assert "tool_call_parser" not in params["engine_config"]
    assert "reasoning_parser" not in params["engine_config"]
    assert "--enable-auto-tool-choice" not in exec_line
    assert "--tool-call-parser" not in exec_line
    assert "--reasoning-parser" not in exec_line
    assert "--speculative-config" not in exec_line
    assert "--default-chat-template-kwargs '{\"enable_thinking\":false}'" in exec_line
