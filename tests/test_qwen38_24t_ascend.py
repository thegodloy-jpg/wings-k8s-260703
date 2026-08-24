import json
import shlex
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines import vllm_adapter, vllm_distributed  # noqa: E402
from utils import model_utils  # noqa: E402


_MODEL_NAME = "Qwen3.8-2.4T-A95B-w8a8"
_MODEL_PATH = f"/mnt/models/{_MODEL_NAME}"
_ARCHITECTURE = "Qwen3_5MoeForCausalLM"
_NODE_IPS = "7.6.28.240,7.6.28.241,7.6.28.242,7.6.28.243"


class _FakeQwen38AscendInfo:
    config = {
        "architectures": [_ARCHITECTURE],
        "_name_or_path": f"Eco-Tech/{_MODEL_NAME}",
    }
    model_quantize = "w8a8"
    model_architecture = _ARCHITECTURE
    model_name = _MODEL_NAME
    model_path = _MODEL_PATH

    @staticmethod
    def identify_model_architecture():
        return _ARCHITECTURE

    @staticmethod
    def identify_model_type():
        return "llm"

    @staticmethod
    def is_wings_supported():
        return True


class _FakeQwen38AscendIdentifier(_FakeQwen38AscendInfo):
    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


def _ascend_arch_defaults():
    config_path = Path(config_loader.DEFAULT_CONFIG_DIR) / "ascend_default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["model_deploy_config"]["llm"][_ARCHITECTURE]


def _route_params(**overrides):
    params = {
        "engine": "vllm_ascend",
        "model_name": _MODEL_NAME,
        "model_path": _MODEL_PATH,
        "model_type": "llm",
        "device_count": 16,
        "distributed": True,
        "nnodes": 4,
        "node_rank": 0,
        "node_ips": _NODE_IPS,
        "distributed_executor_backend": "ray",
        "_smart_card_token": "910c",
    }
    params.update(overrides)
    return params


def _clear_recipe_env(monkeypatch):
    for env_name in (
        "PD_ROLE",
        "VLLM_DISTRIBUTED_PORT",
        "VLLM_DP_RPC_PORT",
        "TENSOR_PARALLEL_SIZE",
        "DATA_PARALLEL_SIZE",
        "ENABLE_SPARSE",
        "ENABLE_KV_OFFLOAD",
        "LMCACHE_OFFLOAD",
        "HCCL_IF_IP",
        "HCCL_SOCKET_IFNAME",
        "TP_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "HCCL_BUFFSIZE",
        "HCCL_BUFFSIZE_EP",
        "HCCL_CONNECT_TIMEOUT",
        "VLLM_ENGINE_READY_TIMEOUT_S",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "OMP_PROC_BIND",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TASK_QUEUE_ENABLE",
        "PYTORCH_NPU_ALLOC_CONF",
        "ASCEND_RT_VISIBLE_DEVICES",
        "HCCL_OP_EXPANSION_MODE",
    ):
        monkeypatch.delenv(env_name, raising=False)


def test_qwen38_24t_w8a8_selects_exact_910c_distributed_defaults():
    config = config_loader._match_model_engine_config(
        _ascend_arch_defaults(),
        _MODEL_NAME.lower(),
        "vllm_ascend_distributed",
        config_loader._SpecialEngineScenario(),
        _FakeQwen38AscendInfo(),
        {"device": "ascend", "details": [{"name": "Ascend910C"}]},
        _MODEL_PATH.lower(),
    )

    assert config == {
        "use_vllm_serve": True,
        "trust_remote_code": True,
        "quantization": "ascend",
        "safetensors_load_strategy": "lazy",
        "enable_prefix_caching": True,
        "enable_expert_parallel": True,
        "max_model_len": 133120,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 16384,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "additional_config": {
            "enable_cpu_binding": True,
            "enable_flashcomm1": False,
            "enable_fused_mc2": 0,
        },
    }


def test_qwen38_24t_w8a8_auto_selects_vllm_ascend(monkeypatch):
    monkeypatch.setattr(config_loader, "get_router_env", lambda: False)

    assert config_loader._select_ascend_engine(
        "Ascend910C",
        _FakeQwen38AscendInfo(),
    ) == "vllm_ascend"


@pytest.mark.parametrize(
    ("model_name", "card_name"),
    [
        ("qwen3.8", "Ascend910C"),
        (_MODEL_NAME, "Ascend910B3"),
    ],
)
def test_qwen38_24t_w8a8_defaults_do_not_broaden(model_name, card_name):
    config = config_loader._match_model_engine_config(
        _ascend_arch_defaults(),
        model_name.lower(),
        "vllm_ascend_distributed",
        config_loader._SpecialEngineScenario(),
        _FakeQwen38AscendInfo(),
        {"device": "ascend", "details": [{"name": card_name}]},
        _MODEL_PATH.lower(),
    )

    assert config == {}


def test_qwen38_24t_w8a8_routes_to_native_dp_with_recipe_marker(monkeypatch):
    _clear_recipe_env(monkeypatch)
    params = _route_params()

    config_loader._handle_vllm_distributed(
        {"vllm_distributed": {"nixl_port": 5759, "rpc_port": 13355, "ray_head_port": 28020}},
        params,
        _FakeQwen38AscendInfo(),
    )

    assert params["distributed_executor_backend"] == "dp_deployment"
    assert params["rpc_port"] == "8002"
    assert params["_qwen38_24t_w8a8_910c_dp"] is True
    assert "_preserve_dp_worker_port" not in params
    assert "ray_head_port" not in params


@pytest.mark.parametrize(("nnodes", "device_count"), [(2, 16), (4, 8), (4, 0)])
def test_qwen38_24t_w8a8_rejects_non_recipe_topology(monkeypatch, nnodes, device_count):
    _clear_recipe_env(monkeypatch)
    params = _route_params(nnodes=nnodes, device_count=device_count)

    with pytest.raises(ValueError, match="requires exactly 4 nodes and 16 NPUs per node"):
        config_loader._handle_vllm_distributed(
            {"vllm_distributed": {"ray_head_port": 28020}},
            params,
            _FakeQwen38AscendInfo(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_name": "qwen3.8"},
        {"_smart_card_token": "910b"},
    ],
)
def test_qwen38_24t_w8a8_route_does_not_broaden(monkeypatch, overrides):
    _clear_recipe_env(monkeypatch)
    params = _route_params(**overrides)

    config_loader._handle_vllm_distributed(
        {"vllm_distributed": {"ray_head_port": 28020}},
        params,
        _FakeQwen38AscendInfo(),
    )

    assert params["distributed_executor_backend"] == "ray"
    assert params["ray_head_port"] == 28020
    assert "_qwen38_24t_w8a8_910c_dp" not in params


@pytest.mark.parametrize(
    ("platform", "overrides", "expected_error"),
    [
        ("a2", {}, "requires Ascend 910C/A3"),
        ("a3", {"device_count": 8}, "requires device_count=16"),
        ("a3", {"nnodes": 2}, "requires nnodes=4"),
        ("a3", {"node_rank": 4}, "requires node_rank in 0..3"),
    ],
)
def test_qwen38_24t_w8a8_dp_env_revalidates_runtime_scope(
    monkeypatch,
    platform,
    overrides,
    expected_error,
):
    params = _route_params(**overrides)
    monkeypatch.setattr(
        vllm_adapter,
        "ascend_platform_from_runtime",
        lambda _params: platform,
    )

    with pytest.raises(ValueError, match=expected_error):
        vllm_distributed._validate_qwen38_24t_w8a8_910c_dp_env_params(params)


def test_qwen38_24t_w8a8_registers_mtp_parser_and_always_on_thinking():
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        _MODEL_NAME,
        _MODEL_PATH,
        "910c",
        "spec",
    )
    found, parser = config_loader._resolve_reasoning_parser_support(
        _ARCHITECTURE,
        _MODEL_NAME,
        "vllm_ascend_distributed",
    )

    assert row is not None
    assert row["arch"] == _ARCHITECTURE
    assert row["mtp_method"] == "qwen3_5_mtp"
    assert row["mtp_num_speculative_tokens"] == 1
    assert (found, parser) == (True, "qwen3")
    assert config_loader._resolve_reasoning_parser_support(
        _ARCHITECTURE,
        _MODEL_NAME,
        "vllm_distributed",
    ) == (False, None)
    assert model_utils.resolve_thinking_off_policy(
        f"Eco-Tech/{_MODEL_NAME}"
    ) == (model_utils.THINKING_ALWAYS_ON, {})
    assert model_utils.resolve_thinking_off_policy(
        "Qwen3.8-27B"
    ) == (model_utils.THINKING_HYBRID, {"enable_thinking": False})


def test_qwen38_24t_w8a8_thinking_switch_only_controls_parser():
    params = {"reasoning_parser": "qwen3"}
    config_loader._set_reasoning_parser(params, {"enable_auto_think_choice": False})
    config_loader._set_thinking_default(
        params,
        {"enable_auto_think_choice": False},
        _FakeQwen38AscendInfo(),
    )

    assert "reasoning_parser" not in params
    assert "default_chat_template_kwargs" not in params


def test_qwen38_24t_w8a8_mtp_and_parser_remain_opt_in(monkeypatch):
    _clear_recipe_env(monkeypatch)
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeQwen38AscendIdentifier)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38AscendIdentifier)
    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeQwen38AscendIdentifier)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "_record_selected_engine", lambda *_args: None)
    monkeypatch.setenv("RANK_IP", "7.6.28.240")
    monkeypatch.setenv("NETWORK_INTERFACE", "enp196s0f1")

    launch_args = parse_launch_args([
        "--model-name", _MODEL_NAME,
        "--model-path", _MODEL_PATH,
        "--model-type", "llm",
        "--engine", "vllm_ascend",
        "--device-count", "16",
        "--port", "8001",
        "--distributed",
        "--nnodes", "4",
        "--node-rank", "0",
        "--node-ips", _NODE_IPS,
        "--master-ip", "7.6.28.240",
    ])
    params = config_loader.load_and_merge_configs(
        {
            "device": "ascend",
            "count": 16,
            "details": [{"name": "Ascend910C"}],
        },
        launch_args,
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert "--speculative-config" not in exec_line
    assert "--reasoning-parser" not in exec_line
    assert params["_smart_feats"] == []


@pytest.mark.parametrize("node_rank", [0, 1, 2, 3])
@pytest.mark.parametrize("enable_reasoning_parser", [False, True])
def test_qwen38_24t_w8a8_full_chain_renders_four_rank_dp_commands(
    monkeypatch,
    node_rank,
    enable_reasoning_parser,
):
    _clear_recipe_env(monkeypatch)
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeQwen38AscendIdentifier)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38AscendIdentifier)
    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeQwen38AscendIdentifier)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "_record_selected_engine", lambda *_args: None)
    monkeypatch.setenv("RANK_IP", f"7.6.28.{240 + node_rank}")
    monkeypatch.setenv("NETWORK_INTERFACE", "enp196s0f1")
    monkeypatch.setenv("SERVED_MODEL_NAME", "qwen3.8")

    argv = [
        "--model-name", _MODEL_NAME,
        "--model-path", _MODEL_PATH,
        "--model-type", "llm",
        "--engine", "vllm_ascend",
        "--device-count", "16",
        "--port", "8001",
        "--distributed",
        "--nnodes", "4",
        "--node-rank", str(node_rank),
        "--node-ips", _NODE_IPS,
        "--master-ip", "7.6.28.240",
        "--enable-speculative-decode",
        "--speculative-decode-model-path", "none",
    ]
    if enable_reasoning_parser:
        argv.append("--enable-auto-think-choice")
    launch_args = parse_launch_args(argv)
    params = config_loader.load_and_merge_configs(
        {
            "device": "ascend",
            "count": 16,
            "details": [{"name": "Ascend910C"}],
        },
        launch_args,
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    shlex.split(exec_line, posix=True)

    assert params["distributed_executor_backend"] == "dp_deployment"
    assert params["rpc_port"] == "8002"
    assert params["_smart_feats"] == ["spec"]
    assert params["engine_config"]["tensor_parallel_size"] == 16
    assert "default_chat_template_kwargs" not in params["engine_config"]
    if enable_reasoning_parser:
        assert params["engine_config"]["reasoning_parser"] == "qwen3"
        assert "--reasoning-parser qwen3" in exec_line
    else:
        assert "reasoning_parser" not in params["engine_config"]
        assert "--reasoning-parser" not in exec_line

    for expected in (
        f"exec vllm serve {_MODEL_PATH}",
        "--quantization ascend",
        "--safetensors-load-strategy lazy",
        "--tensor-parallel-size 16",
        "--enable-prefix-caching",
        "--enable-expert-parallel",
        "--max-model-len 133120",
        "--max-num-seqs 8",
        "--max-num-batched-tokens 16384",
        "--gpu-memory-utilization 0.85",
        "--enforce-eager",
        "--speculative-config '{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":1}'",
        "--data-parallel-address 7.6.28.240",
        "--data-parallel-rpc-port 8002",
        "--data-parallel-size 4",
        "--data-parallel-size-local 1",
    ):
        assert expected in exec_line
    assert '"enable_cpu_binding":true' in exec_line
    assert '"enable_flashcomm1":false' in exec_line
    assert '"enable_fused_mc2":0' in exec_line
    assert "--tokenizer" not in exec_line
    assert "--default-chat-template-kwargs" not in exec_line

    if node_rank == 0:
        assert "--host 0.0.0.0" in exec_line
        assert "--port 8001" in exec_line
        assert "--headless" not in exec_line
        assert "--data-parallel-start-rank" not in exec_line
    else:
        assert "--host" not in exec_line
        assert "--port" not in exec_line
        assert "--headless" in exec_line
        assert f"--data-parallel-start-rank {node_rank}" in exec_line

    export_lines = [line for line in script.splitlines() if line.startswith("export ")]
    expected_exports = {
        "HCCL_IF_IP": "$VLLM_HOST_IP",
        "GLOO_SOCKET_IFNAME": "enp196s0f1",
        "HCCL_SOCKET_IFNAME": "enp196s0f1",
        "TP_SOCKET_IFNAME": "enp196s0f1",
        "HCCL_BUFFSIZE": "1024",
        "HCCL_BUFFSIZE_EP": "2048",
        "HCCL_CONNECT_TIMEOUT": "600",
        "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3000",
        "OMP_PROC_BIND": "false",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(index) for index in range(16)),
    }
    for name, value in expected_exports.items():
        assert export_lines.count(f"export {name}={value}") == 1
    for omitted_name in ("OMP_NUM_THREADS", "TASK_QUEUE_ENABLE", "HCCL_OP_EXPANSION_MODE"):
        assert not any(line.startswith(f"export {omitted_name}=") for line in export_lines)
