import json
import shlex
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from core import wings_entry  # noqa: E402
from core.port_plan import derive_port_plan  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines import vllm_adapter  # noqa: E402
from utils import model_utils  # noqa: E402


_MODEL_NAME = "Qwen3.8-27B-w8a8"
_MODEL_PATH = "/var/aispace/model/ai-storage/ai-prod/platform/Qwen3.8-27B-w8a8"
_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"


class _FakeQwen38Identifier:
    config = {}
    model_architecture = _ARCHITECTURE
    model_quantize = "w8a8"

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


def _ascend_arch_defaults():
    config_path = Path(config_loader.DEFAULT_CONFIG_DIR) / "ascend_default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["model_deploy_config"]["llm"][_ARCHITECTURE]


def _match_defaults(model_name: str, card_name: str, engine_key: str = "vllm_ascend"):
    return config_loader._match_model_engine_config(
        _ascend_arch_defaults(),
        model_name.lower(),
        engine_key,
        config_loader._SpecialEngineScenario(),
        _FakeQwen38Identifier(model_name, _MODEL_PATH, "llm"),
        {"device": "ascend", "details": [{"name": card_name}]},
        _MODEL_PATH.lower(),
    )


def test_qwen38_27b_w8a8_910b_selects_exact_single_node_profile():
    config = _match_defaults(_MODEL_NAME, "Ascend910B")

    assert config == {
        "use_vllm_serve": True,
        "trust_remote_code": True,
        "quantization": "ascend",
        "max_num_seqs": 32,
        "max_model_len": 131072,
        "max_num_batched_tokens": 16384,
        "gpu_memory_utilization": 0.85,
        "enable_prefix_caching": True,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "additional_config": {"enable_cpu_binding": True},
    }
    # TP/DP 是运行时拓扑，不能固化在模型 defaults 中。
    assert "tensor_parallel_size" not in config
    assert "data_parallel_size" not in config
    # 投机能力由精确白名单和页面开关控制，不能在模型 defaults 中强开。
    assert "speculative_config" not in config


@pytest.mark.parametrize(
    ("model_name", "card_name", "engine_key"),
    [
        ("Qwen3.8-27B", "Ascend910B", "vllm_ascend"),
        (_MODEL_NAME, "Ascend910C", "vllm_ascend"),
        (_MODEL_NAME, "Ascend910B", "vllm_ascend_distributed"),
    ],
)
def test_qwen38_27b_profile_does_not_broaden(model_name, card_name, engine_key):
    assert _match_defaults(model_name, card_name, engine_key) == {}


def test_qwen38_27b_910b_spec_whitelist_renders_exact_mtp(monkeypatch):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        _MODEL_NAME,
        _MODEL_PATH,
        "910b",
        "spec",
    )

    assert row is not None
    assert row["mtp_method"] == "qwen3_5_mtp"
    assert row["mtp_num_speculative_tokens"] == 3
    assert row["enforce_eager"] is True
    assert model_utils.resolve_feature_whitelist_row(
        "vllm_ascend", _MODEL_NAME, _MODEL_PATH, "910c", "spec"
    ) is None
    assert model_utils.resolve_feature_whitelist_row(
        "vllm_ascend", "Qwen3.8-27B", "/models/Qwen3.8-27B", "910b", "spec"
    ) is None

    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    command = vllm_adapter.build_speculative_cmd(
        {
            "engine": "vllm_ascend",
            "model_name": _MODEL_NAME,
            "model_path": _MODEL_PATH,
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec"],
            "_smart_card_token": "910b",
        },
        "vllm_ascend",
    )

    assert command == (
        " --speculative-config "
        "'{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3,\"enforce_eager\":true}'"
    )


def test_qwen38_27b_910b_uses_dedicated_minimal_env(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    commands = vllm_adapter._build_model_env_commands(
        {
            "engine": "vllm_ascend",
            "model_name": _MODEL_NAME,
            "model_path": _MODEL_PATH,
            "model_type": "llm",
            "device_count": 2,
            "nnodes": 1,
            "distributed": False,
            "_smart_card_token": "ascend910b",
        },
        "vllm_ascend",
    )

    assert commands == [
        "export HCCL_BUFFSIZE=512",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_name": "Qwen3.8-27B"},
        {"_smart_card_token": "ascend910c"},
        {"device_count": 4},
        {"distributed": True, "nnodes": 2},
    ],
)
def test_qwen38_27b_env_scope_does_not_broaden(overrides):
    params = {
        "model_name": _MODEL_NAME,
        "device_count": 2,
        "nnodes": 1,
        "distributed": False,
        "_smart_card_token": "ascend910b",
    }
    params.update(overrides)

    assert not vllm_adapter._is_qwen38_27b_w8a8_910b_single_node_env_scope(
        params,
        _FakeQwen38Identifier(_MODEL_NAME, _MODEL_PATH, "llm"),
    )


def test_qwen38_27b_production_chain_renders_target_single_node_command(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "_record_selected_engine", lambda *_args: None)
    for env_name in (
        "PD_ROLE",
        "ENGINE",
        "WINGS_ENGINE",
        "ENGINE_PORT",
        "PROXY_PORT",
        "PORT",
        "POD_IP",
        "RANK_IP",
        "ENABLE_SPECULATIVE_DECODE",
        "SD_ENABLE",
        "TENSOR_PARALLEL_SIZE",
        "DATA_PARALLEL_SIZE",
        "ENABLE_SPARSE",
        "ENABLE_KV_OFFLOAD",
        "LMCACHE_OFFLOAD",
        "ASCEND_ENFORCE_EAGER",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("SERVED_MODEL_NAME", "qwen3.8")
    monkeypatch.setenv("HCCL_BUFFSIZE", "512")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("ENGINE_PORT", "8000")
    monkeypatch.setenv("POD_IP", "10.0.0.8")

    # DP=1 由本次部署显式声明；模型 defaults 只保存静态能力参数。
    config_file = tmp_path / "qwen38-27b-runtime.json"
    config_file.write_text(json.dumps({"data_parallel_size": 1}), encoding="utf-8")
    launch_args = parse_launch_args(
        [
            "--model-name",
            _MODEL_NAME,
            "--model-path",
            _MODEL_PATH,
            "--model-type",
            "llm",
            "--engine",
            "vllm_ascend",
            "--device-count",
            "2",
            "--host",
            "0.0.0.0",
            "--port",
            "18000",
            "--config-file",
            str(config_file),
            "--enable-speculative-decode",
            "--speculative-decode-model-path",
            "none",
        ]
    )
    port_plan = derive_port_plan(
        port=launch_args.port,
        enable_reason_proxy=True,
    )
    params = wings_entry._prepare_merged_params(
        launch_args,
        port_plan,
        {
            "device": "ascend",
            "count": 2,
            "details": [{"name": "Ascend910B"}],
        },
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    shlex.split(exec_line, posix=True)

    assert params["_smart_card_token"].endswith("910b")
    assert params["_smart_feats"] == ["spec"]
    assert params["engine_config"]["tensor_parallel_size"] == 2
    assert params["engine_config"]["data_parallel_size"] == 1
    # v4 的 --port/PORT 是代理端口；真正的 vLLM 监听端口由 ENGINE_PORT 控制。
    assert port_plan.proxy_port == 18000
    assert port_plan.backend_port == 8000
    assert exec_line == (
        f"exec vllm serve {_MODEL_PATH}"
        " --trust-remote-code"
        " --quantization ascend"
        " --max-num-seqs 32"
        " --max-model-len 131072"
        " --max-num-batched-tokens 16384"
        " --gpu-memory-utilization 0.85"
        " --enable-prefix-caching"
        " --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'"
        " --additional-config '{\"enable_cpu_binding\":true}'"
        " --host 10.0.0.8"
        " --port 8000"
        " --served-model-name qwen3.8"
        " --default-chat-template-kwargs '{\"enable_thinking\":false}'"
        " --tensor-parallel-size 2"
        " --data-parallel-size 1"
        " --speculative-config "
        "'{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3,\"enforce_eager\":true}'"
    )
    # 可见卡由部署层传入；启动脚本不得把 2,3 改写成逻辑卡号或其它物理卡号。
    assert "ASCEND_RT_VISIBLE_DEVICES=" not in script
    assert [line for line in script.splitlines() if line.startswith("export ")] == [
        "export HCCL_BUFFSIZE=512",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
        "export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}",
    ]
