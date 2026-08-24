import json
import shlex
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader, wings_entry  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines import vllm_adapter, vllm_distributed  # noqa: E402
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
        "max_num_seqs": 16,
        "max_num_batched_tokens": 8192,
        "enable_prefix_caching": True,
        "enable_expert_parallel": True,
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

    params["model_architecture"] = _ARCHITECTURE
    engine_config = {}
    config_loader._set_parallelism_params(engine_config, params)
    assert engine_config["tensor_parallel_size"] == 8
    assert engine_config["data_parallel_size"] == 4


def test_qwen38_runtime_parallelism_keeps_explicit_env_precedence(monkeypatch):
    monkeypatch.delenv("PD_ROLE", raising=False)
    params = _qwen38_route_params()
    config_loader._handle_vllm_distributed(
        {"vllm_distributed": {"ray_head_port": 28020}},
        params,
        _FakeQwen38Info(),
    )
    params["model_architecture"] = _ARCHITECTURE

    engine_config = {}
    config_loader._set_parallelism_params(engine_config, params)
    monkeypatch.setenv("TENSOR_PARALLEL_SIZE", "4")
    monkeypatch.setenv("DATA_PARALLEL_SIZE", "8")
    config_loader._apply_cli_overrides(engine_config, {"engine": "vllm"})

    assert engine_config["tensor_parallel_size"] == 4
    assert engine_config["data_parallel_size"] == 8


@pytest.mark.parametrize("model_name", [_MODEL_NAME, f"Qwen/{_MODEL_NAME}"])
@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
def test_qwen38_four_node_h20_mp_matches_reference_env(
    monkeypatch,
    model_name,
    card_token,
):
    monkeypatch.setenv("NETWORK_INTERFACE", "bond0")
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    params = _qwen38_route_params(card_token=card_token)
    params["model_name"] = model_name

    commands = vllm_distributed._build_mp_env_commands(params)

    assert commands == [
        "export NCCL_SOCKET_IFNAME=bond0",
        "export GLOO_SOCKET_IFNAME=bond0",
        "export VLLM_ENGINE_READY_TIMEOUT_S=10800",
        "export VLLM_DP_LB_KV_CACHE_AWARE=1",
        "export VLLM_USE_V2_MODEL_RUNNER=1",
        "export VLLM_USE_RUST_FRONTEND=1",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_name": "Qwen3.8-2.4T-A95B"},
        {"_smart_card_token": "h100"},
        {"device_count": 4},
        {"nnodes": 2},
    ],
)
def test_qwen38_reference_env_does_not_broaden(overrides):
    params = _qwen38_route_params()
    params.update(overrides)

    commands = vllm_distributed._build_mp_env_commands(params)

    assert any(command.startswith("export VLLM_HOST_IP=") for command in commands)
    for env_name in (
        "VLLM_ENGINE_READY_TIMEOUT_S",
        "VLLM_DP_LB_KV_CACHE_AWARE",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_USE_RUST_FRONTEND",
    ):
        assert not any(command.startswith(f"export {env_name}=") for command in commands)


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
def test_qwen38_h20_spec_and_sparse_whitelists_are_exact(card_token):
    spec_row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        _MODEL_NAME,
        _MODEL_PATH,
        card_token,
        "spec",
    )
    sparse_row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        _MODEL_NAME,
        _MODEL_PATH,
        card_token,
        "sparse",
    )

    assert spec_row is not None
    assert spec_row["arch"] == _ARCHITECTURE
    assert spec_row["mtp_method"] == "mtp"
    assert spec_row["mtp_num_speculative_tokens"] == 3
    assert sparse_row is not None
    assert sparse_row["arch"] == _ARCHITECTURE
    assert sparse_row["strategy"] == "fp8"
    assert model_utils.resolve_feature_whitelist(
        "vllm", _MODEL_NAME, _MODEL_PATH, card_token
    ) == frozenset({"sparse", "spec"})
    assert model_utils.resolve_feature_whitelist(
        "vllm", "Qwen3.8-2.4T-A95B", "/models/Qwen3.8-2.4T-A95B", card_token
    ) == frozenset()
    assert model_utils.resolve_feature_whitelist_row(
        "vllm", _MODEL_NAME, _MODEL_PATH, "h100", "sparse"
    ) is None


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
        "--enable-sparse",
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
    assert params["_allowed_smart_feats"] == ["sparse", "spec"]
    assert params["_smart_feats"] == ["sparse", "spec"]
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
        " --max-num-seqs 16"
        " --max-num-batched-tokens 8192"
        " --enable-prefix-caching"
        " --enable-expert-parallel"
        " --tensor-parallel-size 8"
        " --data-parallel-size 4"
        " --all2all-backend allgather_reducescatter"
    )
    topology = (
        " --nnodes 4"
        f" --node-rank {node_rank}"
        " --master-addr 7.6.25.57"
    )
    if node_rank == 0:
        expected = (
            common
            + " --tool-call-parser qwen3_coder"
            + " --distributed-timeout-seconds 7200"
            + " --cpu-distributed-timeout-seconds 7200"
            + " --reasoning-parser qwen3"
            + " --port 8001"
            + " --served-model-name qwen3.8"
            + " --enable-auto-tool-choice"
            + " --kv-cache-dtype fp8"
            + " --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
            + topology
        )
    else:
        expected = (
            common
            + " --tool-call-parser qwen3_coder"
            + " --distributed-timeout-seconds 7200"
            + " --cpu-distributed-timeout-seconds 7200"
            + " --reasoning-parser qwen3"
            + " --port 8001"
            + " --served-model-name qwen3.8"
            + " --enable-auto-tool-choice"
            + " --kv-cache-dtype fp8"
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
            # 即使上层残留了显式 FP8，sparse 关闭态也必须在最终渲染前清掉。
            "kv_cache_dtype": "fp8",
            "max_num_seqs": 16,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
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

    assert params["_allowed_smart_feats"] == ["sparse", "spec"]
    assert params["_smart_feats"] == ["spec"]
    assert "export VLLM_HOST_IP=" not in script
    assert "export NCCL_SOCKET_IFNAME=bond0" in script
    assert "export GLOO_SOCKET_IFNAME=bond0" in script
    assert "export VLLM_ENGINE_READY_TIMEOUT_S=10800" in script
    assert "export VLLM_DP_LB_KV_CACHE_AWARE=1" in script
    assert "export VLLM_USE_V2_MODEL_RUNNER=1" in script
    assert "export VLLM_USE_RUST_FRONTEND=1" in script
    assert "ray start" not in script
    for expected in (
        "--served-model-name qwen3.8",
        "--trust-remote-code",
        "--max-model-len 133000",
        "--max-cudagraph-capture-size 256",
        "--gpu-memory-utilization 0.9",
        "--max-num-seqs 16",
        "--max-num-batched-tokens 8192",
        "--enable-prefix-caching",
        "--enable-expert-parallel",
        "--tensor-parallel-size 8",
        "--data-parallel-size 4",
        "--all2all-backend allgather_reducescatter",
        "--distributed-timeout-seconds 7200",
        "--cpu-distributed-timeout-seconds 7200",
        "--nnodes 4",
        f"--node-rank {node_rank}",
        "--master-addr 7.6.25.57",
    ):
        assert expected in exec_line
    assert '"method":"mtp"' in exec_line
    assert '"num_speculative_tokens":3' in exec_line
    assert exec_line.count("--speculative-config") == 1
    assert "--kv-cache-dtype" not in exec_line
    assert "kv_cache_dtype" not in params["engine_config"]
    assert "--kv-transfer-config" not in exec_line

    assert "--host" not in exec_line
    assert "--port 8001" in exec_line
    assert "--enable-auto-tool-choice" in exec_line
    assert "--tool-call-parser qwen3_coder" in exec_line
    assert "--reasoning-parser qwen3" in exec_line
    assert "--default-chat-template-kwargs" not in exec_line
    assert "--distributed-executor-backend" not in exec_line
    assert "--master-port" not in exec_line
    assert ("--headless" in exec_line) is (node_rank != 0)


def test_qwen38_sparse_fp8_status_and_fallback_are_consistent(monkeypatch):
    # FP8 KV 必须由 sparse 有效态拥有；全特性回退后不能残留在重建命令中。
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(wings_entry, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(
        wings_entry,
        "start_engine_service",
        lambda merged: vllm_adapter.build_start_script(merged),
    )
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "false")
    params = {
        "engine": "vllm",
        "model_name": _MODEL_NAME,
        "model_path": _MODEL_PATH,
        "model_type": "llm",
        "device_count": 8,
        "enable_sparse": True,
        "enable_speculative_decode": True,
        "engine_config": {
            "use_vllm_serve": True,
            "model": _MODEL_PATH,
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
        },
    }
    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "nvidia", "count": 8, "details": [{"name": "NVIDIA H20 141GB"}]},
    )

    enabled_script = vllm_adapter.build_start_script(params)
    enabled_exec_line = next(
        line for line in enabled_script.splitlines() if line.startswith("exec ")
    )
    status = wings_entry._resolve_advanced_feature_status("vllm", params)
    fallback_cmd = wings_entry._build_advanced_feature_fallback_cmd(params)

    assert enabled_exec_line.count("--kv-cache-dtype fp8") == 1
    assert status["features"]["sparse_kv"] is True
    assert status["variants"]["sparse_kv"] == "fp8"
    assert "--kv-cache-dtype" not in fallback_cmd
    assert "--speculative-config" not in fallback_cmd
    assert params["engine_config"]["kv_cache_dtype"] == "fp8"

    sparse_only = {
        "engine": "vllm",
        "model_name": _MODEL_NAME,
        "model_path": _MODEL_PATH,
        "model_type": "llm",
        "device_count": 8,
        "enable_sparse": True,
        "enable_speculative_decode": False,
        "engine_config": {"use_vllm_serve": True, "model": _MODEL_PATH},
    }
    config_loader.apply_effective_feature_enablement(
        sparse_only,
        {"device": "nvidia", "count": 8, "details": [{"name": "NVIDIA H20 141GB"}]},
    )
    vllm_adapter.build_start_script(sparse_only)

    sparse_only_fallback = wings_entry._build_advanced_feature_fallback_cmd(sparse_only)

    assert "--kv-cache-dtype" not in sparse_only_fallback
    assert sparse_only["engine_config"]["kv_cache_dtype"] == "fp8"
