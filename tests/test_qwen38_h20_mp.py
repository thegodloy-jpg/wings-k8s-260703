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


_MODEL_NAME = "Qwen3.8-2.4T-A95B-FP8"
_MODEL_PATH = "/models/Qwen3.8-2.4T-A95B-FP8"
_ARCHITECTURE = "Qwen3_5MoeForCausalLM"


class _FakeQwen38Info:
    config = {}
    model_quantize = "fp8"
    model_architecture = _ARCHITECTURE
    model_name = _MODEL_NAME

    @staticmethod
    def identify_model_architecture():
        return _ARCHITECTURE

    @staticmethod
    def identify_model_type():
        return "llm"


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


def _qwen38_arch_defaults():
    config_path = Path(config_loader.DEFAULT_CONFIG_DIR) / "nvidia_default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["model_deploy_config"]["llm"][_ARCHITECTURE]


def _qwen38_route_params(*, nnodes=4, device_count=8, card_token="h20-141"):
    return {
        "engine": "vllm",
        "model_name": _MODEL_NAME,
        "model_path": _MODEL_PATH,
        "device_count": device_count,
        "distributed": True,
        "nnodes": nnodes,
        "node_ips": ",".join(f"7.6.25.{57 + index}" for index in range(nnodes)),
        "distributed_executor_backend": "ray",
        "_smart_card_token": card_token,
    }


@pytest.mark.parametrize(
    "model_name",
    [_MODEL_NAME, f"Qwen/{_MODEL_NAME}"],
)
@pytest.mark.parametrize("card_name", ["NVIDIA H20 96GB", "NVIDIA H20 141GB"])
def test_qwen38_h20_selects_exact_distributed_defaults(model_name, card_name):
    config = config_loader._match_model_engine_config(
        _qwen38_arch_defaults(),
        model_name.lower(),
        "vllm_distributed",
        config_loader._SpecialEngineScenario(),
        _FakeQwen38Info(),
        {"device": "nvidia", "details": [{"name": card_name}]},
        _MODEL_PATH.lower(),
    )

    assert config == {
        "use_vllm_serve": True,
        "trust_remote_code": True,
        "max_model_len": 133000,
        "max_cudagraph_capture_size": 256,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 4096,
        "enable_expert_parallel": True,
        "tensor_parallel_size": 8,
        "data_parallel_size": 4,
        "all2all_backend": "allgather_reducescatter",
        "tool_call_parser": "qwen3_coder",
        "distributed_timeout_seconds": 7200,
        "cpu_distributed_timeout_seconds": 7200,
    }


@pytest.mark.parametrize(
    ("model_name", "model_path", "card_name"),
    [
        ("Qwen3.8-2.4T-A95B", _MODEL_PATH, "NVIDIA H20 141GB"),
        (_MODEL_NAME, _MODEL_PATH, "NVIDIA H100 80GB"),
    ],
)
def test_qwen38_profile_does_not_broaden(model_name, model_path, card_name):
    config = config_loader._match_model_engine_config(
        _qwen38_arch_defaults(),
        model_name.lower(),
        "vllm_distributed",
        config_loader._SpecialEngineScenario(),
        _FakeQwen38Info(),
        {"device": "nvidia", "details": [{"name": card_name}]},
        model_path.lower(),
    )

    assert config == {}


@pytest.mark.parametrize(
    "model_name",
    [_MODEL_NAME, f"Qwen/{_MODEL_NAME}"],
)
@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
def test_qwen38_four_node_h20_routes_to_native_mp(monkeypatch, model_name, card_token):
    monkeypatch.delenv("PD_ROLE", raising=False)
    params = _qwen38_route_params(card_token=card_token)
    params["model_name"] = model_name

    config_loader._handle_vllm_distributed(
        {"vllm_distributed": {"ray_head_port": 28020}},
        params,
        _FakeQwen38Info(),
    )

    assert params["distributed_executor_backend"] == "mp"
    assert "ray_head_port" not in params


@pytest.mark.parametrize(("nnodes", "device_count"), [(2, 8), (4, 4), (4, 0)])
def test_qwen38_h20_rejects_non_recipe_topology(monkeypatch, nnodes, device_count):
    monkeypatch.delenv("PD_ROLE", raising=False)
    params = _qwen38_route_params(nnodes=nnodes, device_count=device_count)

    with pytest.raises(
        ValueError,
        match="requires exactly 4 nodes and 8 GPUs per node",
    ):
        config_loader._handle_vllm_distributed(
            {"vllm_distributed": {"ray_head_port": 28020}},
            params,
            _FakeQwen38Info(),
        )


def test_qwen38_h20_rejects_invalid_topology_value(monkeypatch):
    monkeypatch.delenv("PD_ROLE", raising=False)
    params = _qwen38_route_params()
    params["nnodes"] = "invalid"

    with pytest.raises(
        ValueError,
        match="requires valid nnodes/device_count",
    ):
        config_loader._handle_vllm_distributed(
            {"vllm_distributed": {"ray_head_port": 28020}},
            params,
            _FakeQwen38Info(),
        )


def test_qwen38_route_does_not_broaden_to_neighbor_model(monkeypatch):
    monkeypatch.delenv("PD_ROLE", raising=False)
    params = _qwen38_route_params()
    params["model_name"] = "Qwen3.8-2.4T-A95B"

    config_loader._handle_vllm_distributed(
        {"vllm_distributed": {"ray_head_port": 28020}},
        params,
        _FakeQwen38Info(),
    )

    assert params["distributed_executor_backend"] == "ray"
    assert params["ray_head_port"] == 28020


def test_qwen38_route_requires_canonical_model_name_even_when_path_matches(monkeypatch):
    monkeypatch.delenv("PD_ROLE", raising=False)
    params = _qwen38_route_params()
    params["model_name"] = "qwen3.8"

    config_loader._handle_vllm_distributed(
        {"vllm_distributed": {"ray_head_port": 28020}},
        params,
        _FakeQwen38Info(),
    )

    assert params["distributed_executor_backend"] == "ray"
    assert params["ray_head_port"] == 28020


def test_qwen38_nvidia_pd_route_still_takes_precedence(monkeypatch):
    monkeypatch.setenv("PD_ROLE", "P")
    monkeypatch.delenv("VLLM_DISTRIBUTED_PORT", raising=False)
    params = _qwen38_route_params(nnodes=2)

    config_loader._handle_vllm_distributed(
        {
            "vllm_distributed": {
                "nixl_port": 27070,
                "rpc_port": 27071,
                "ray_head_port": 28020,
            }
        },
        params,
        _FakeQwen38Info(),
    )

    assert params["distributed_executor_backend"] == "dp_deployment"
    assert params["nixl_port"] == 27070
    assert params["rpc_port"] == 27071


def test_qwen38_reasoning_parser_is_vllm_only():
    found, parser = config_loader._resolve_reasoning_parser_support(
        _ARCHITECTURE,
        _MODEL_NAME,
        "vllm_distributed",
    )
    ascend_found, ascend_parser = config_loader._resolve_reasoning_parser_support(
        _ARCHITECTURE,
        _MODEL_NAME,
        "vllm_ascend_distributed",
    )

    assert (found, parser) == (True, "qwen3")
    assert (ascend_found, ascend_parser) == (False, None)


@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
def test_qwen38_h20_spec_whitelist_uses_mtp3_only(card_token):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        _MODEL_NAME,
        _MODEL_PATH,
        card_token,
        "spec",
    )

    assert row is not None
    assert row["arch"] == _ARCHITECTURE
    assert row["mtp_method"] == "mtp"
    assert row["mtp_num_speculative_tokens"] == 3
    assert model_utils.resolve_feature_whitelist(
        "vllm", _MODEL_NAME, _MODEL_PATH, card_token
    ) == frozenset({"spec"})
    assert model_utils.resolve_feature_whitelist(
        "vllm", "Qwen3.8-2.4T-A95B", "/models/Qwen3.8-2.4T-A95B", card_token
    ) == frozenset()


@pytest.mark.parametrize("node_rank", [0, 1, 2, 3])
def test_qwen38_full_config_chain_renders_four_rank_mp_commands(monkeypatch, node_rank):
    node_ips = "7.6.25.57,7.6.25.58,7.6.25.59,7.6.25.60"
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "_record_selected_engine", lambda *_args: None)
    for env_name in (
        "PD_ROLE",
        "TENSOR_PARALLEL_SIZE",
        "DATA_PARALLEL_SIZE",
        "ENABLE_SPARSE",
        "ENABLE_KV_OFFLOAD",
        "LMCACHE_OFFLOAD",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "MASTER_PORT",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("RANK_IP", f"7.6.25.{57 + node_rank}")
    monkeypatch.setenv("NETWORK_INTERFACE", "bond0")
    monkeypatch.setenv("SERVED_MODEL_NAME", "qwen3.8")

    launch_args = parse_launch_args([
        "--model-name", _MODEL_NAME,
        "--model-path", _MODEL_PATH,
        "--model-type", "llm",
        "--engine", "vllm",
        "--device-count", "8",
        "--port", "8001",
        "--distributed",
        "--nnodes", "4",
        "--node-rank", str(node_rank),
        "--node-ips", node_ips,
        "--master-ip", "7.6.25.57",
        "--enable-auto-tool-choice",
        "--enable-auto-think-choice",
        "--enable-speculative-decode",
        "--speculative-decode-model-path", "none",
    ])
    params = config_loader.load_and_merge_configs(
        {
            "device": "nvidia",
            "count": 8,
            "details": [{"name": "NVIDIA H20 141GB"}],
        },
        launch_args,
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    for line in script.splitlines():
        if line.strip():
            shlex.split(line, posix=True)

    assert params["model_name"] == _MODEL_NAME
    assert params["model_path"] == _MODEL_PATH
    assert params["distributed_executor_backend"] == "mp"
    assert params["_smart_feats"] == ["spec"]
    assert params["engine_config"]["tensor_parallel_size"] == 8
    assert params["engine_config"]["data_parallel_size"] == 4
    assert params["engine_config"]["max_model_len"] == 133000
    assert params["engine_config"]["tool_call_parser"] == "qwen3_coder"
    assert params["engine_config"]["reasoning_parser"] == "qwen3"
    common = (
        f"exec vllm serve {_MODEL_PATH}"
        " --trust-remote-code"
        " --max-model-len 133000"
        " --max-cudagraph-capture-size 256"
        " --gpu-memory-utilization 0.9"
        " --max-num-seqs 8"
        " --max-num-batched-tokens 4096"
        " --enable-expert-parallel"
        " --tensor-parallel-size 8"
        " --data-parallel-size 4"
        " --all2all-backend allgather_reducescatter"
    )
    topology = (
        " --distributed-executor-backend mp"
        " --nnodes 4"
        f" --node-rank {node_rank}"
        " --master-addr 7.6.25.57"
        " --master-port 29501"
    )
    if node_rank == 0:
        expected = (
            common
            + " --tool-call-parser qwen3_coder"
            + " --distributed-timeout-seconds 7200"
            + " --cpu-distributed-timeout-seconds 7200"
            + " --reasoning-parser qwen3"
            + " --host 0.0.0.0"
            + " --port 8001"
            + " --served-model-name qwen3.8"
            + " --enable-auto-tool-choice"
            + " --default-chat-template-kwargs '{\"enable_thinking\":true}'"
            + " --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
            + topology
        )
    else:
        expected = (
            common
            + " --distributed-timeout-seconds 7200"
            + " --cpu-distributed-timeout-seconds 7200"
            + " --served-model-name qwen3.8"
            + " --default-chat-template-kwargs '{\"enable_thinking\":true}'"
            + " --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
            + " --headless"
            + topology
        )
    assert exec_line == expected


@pytest.mark.parametrize("node_rank", [0, 1, 2, 3])
def test_qwen38_four_rank_final_mp_commands(monkeypatch, node_rank):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_SPARSE", "false")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "false")
    monkeypatch.setenv("NETWORK_INTERFACE", "bond0")
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    params = {
        "engine": "vllm",
        "model_name": _MODEL_NAME,
        "model_path": _MODEL_PATH,
        "model_type": "llm",
        "device_count": 8,
        "distributed": True,
        "distributed_executor_backend": "mp",
        "nnodes": 4,
        "node_rank": node_rank,
        "master_ip": "7.6.25.57",
        "master_port": 29501,
        "enable_sparse": False,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "engine_config": {
            "use_vllm_serve": True,
            "model": _MODEL_PATH,
            "host": "0.0.0.0",
            "port": 8001,
            "served_model_name": "qwen3.8",
            "trust_remote_code": True,
            "max_model_len": 133000,
            "max_cudagraph_capture_size": 256,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 4096,
            "enable_expert_parallel": True,
            "tensor_parallel_size": 8,
            "data_parallel_size": 4,
            "all2all_backend": "allgather_reducescatter",
            "enable_auto_tool_choice": True,
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "qwen3",
            "default_chat_template_kwargs": json.dumps({"enable_thinking": True}),
            "distributed_timeout_seconds": 7200,
            "cpu_distributed_timeout_seconds": 7200,
        },
    }

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "nvidia", "count": 8, "details": [{"name": "NVIDIA H20 141GB"}]},
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert params["_allowed_smart_feats"] == ["spec"]
    assert params["_smart_feats"] == ["spec"]
    assert "export VLLM_HOST_IP=" in script
    assert "export NCCL_SOCKET_IFNAME=bond0" in script
    assert "export GLOO_SOCKET_IFNAME=bond0" in script
    assert "VLLM_USE_V2_MODEL_RUNNER" not in script
    assert "VLLM_USE_RUST_FRONTEND" not in script
    assert "ray start" not in script
    for expected in (
        "--served-model-name qwen3.8",
        "--trust-remote-code",
        "--max-model-len 133000",
        "--max-cudagraph-capture-size 256",
        "--gpu-memory-utilization 0.9",
        "--max-num-seqs 8",
        "--max-num-batched-tokens 4096",
        "--enable-expert-parallel",
        "--tensor-parallel-size 8",
        "--data-parallel-size 4",
        "--all2all-backend allgather_reducescatter",
        "--default-chat-template-kwargs",
        "--distributed-timeout-seconds 7200",
        "--cpu-distributed-timeout-seconds 7200",
        "--distributed-executor-backend mp",
        "--nnodes 4",
        f"--node-rank {node_rank}",
        "--master-addr 7.6.25.57",
        "--master-port 29501",
    ):
        assert expected in exec_line
    assert '"method":"mtp"' in exec_line
    assert '"num_speculative_tokens":3' in exec_line
    assert exec_line.count("--speculative-config") == 1
    assert "--kv-cache-dtype" not in exec_line
    assert "--kv-transfer-config" not in exec_line

    if node_rank == 0:
        assert "--headless" not in exec_line
        assert "--host 0.0.0.0" in exec_line
        assert "--port 8001" in exec_line
        assert "--enable-auto-tool-choice" in exec_line
        assert "--tool-call-parser qwen3_coder" in exec_line
        assert "--reasoning-parser qwen3" in exec_line
    else:
        assert "--headless" in exec_line
        assert "--host" not in exec_line
        assert "--port" not in exec_line
        assert "--enable-auto-tool-choice" not in exec_line
        assert "--tool-call-parser" not in exec_line
        assert "--reasoning-parser" not in exec_line
