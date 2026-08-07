import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_distributed  # noqa: E402
from utils.vllm_helpers import DistScriptCtx, DpDeploymentTopology  # noqa: E402


def _ctx(*, engine="vllm", node_rank=1, nnodes=2):
    return DistScriptCtx(
        engine=engine,
        cmd=(
            "vllm serve /models/test --host 0.0.0.0 --port 8000 "
            "--data-parallel-size 2 --data-parallel-size-local 1 "
            "--data-parallel-start-rank 0"
        ),
        is_ascend=engine == "vllm_ascend",
        node_rank=node_rank,
        nnodes=nnodes,
        head_addr="10.0.0.1",
        ray_port="28020",
        node_ips="10.0.0.1,10.0.0.2",
    )


def _mp_ctx(*, node_rank=0, net_ips=None):
    node_ips = net_ips or "7.6.25.57,7.6.25.58,7.6.25.59,7.6.25.60"
    return DistScriptCtx(
        engine="vllm",
        cmd=(
            "vllm serve /models/Kimi-K3 --trust-remote-code "
            "--tensor-parallel-size 8 --data-parallel-size 4 "
            "--host 0.0.0.0 --port 17000 "
            "--enable-auto-tool-choice --tool-call-parser kimi_k3 "
            "--reasoning-parser kimi_k3 --served-model-name kimi_k3"
        ),
        is_ascend=False,
        node_rank=node_rank,
        nnodes=4,
        head_addr="7.6.25.57",
        ray_port="28020",
        node_ips=node_ips,
    )


def test_kimi_k3_mp_rank0_keeps_frontend_and_native_topology(monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "enp66s0f1")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "enp66s0f1")

    commands = vllm_distributed._build_mp_commands({}, _mp_ctx(node_rank=0))
    final_command = commands[-1]

    assert "export NCCL_SOCKET_IFNAME=enp66s0f1" in commands
    assert "export GLOO_SOCKET_IFNAME=enp66s0f1" in commands
    # MP 路径只保留网卡选择，不再替 Kimi-K3 强制覆盖 NCCL/SSM 运行时策略。
    for env_name in ("NCCL_NVLS_ENABLE", "NCCL_DEBUG", "VLLM_SSM_CONV_STATE_LAYOUT"):
        assert not any(command.startswith(f"export {env_name}=") for command in commands)
    assert "--distributed-executor-backend mp" in final_command
    assert "--nnodes 4" in final_command
    assert "--node-rank 0" in final_command
    assert "--master-addr 7.6.25.57" in final_command
    assert "--master-port 29501" in final_command
    assert "--host 0.0.0.0" in final_command
    assert "--port 17000" in final_command
    assert "--enable-auto-tool-choice" in final_command
    assert "--tool-call-parser kimi_k3" in final_command
    assert "--reasoning-parser kimi_k3" in final_command
    assert "--headless" not in final_command
    assert "--data-parallel-size 4" in final_command


@pytest.mark.parametrize("node_rank", [1, 2, 3])
def test_kimi_k3_mp_worker_is_headless_and_uses_local_nic(monkeypatch, node_rank):
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ens3f3")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "ens3f3")

    commands = vllm_distributed._build_mp_commands(
        {"model_name": "Kimi-K3", "_smart_card_token": "h20-141"},
        _mp_ctx(node_rank=node_rank),
    )
    final_command = commands[-1]

    assert "export NCCL_SOCKET_IFNAME=ens3f3" in commands
    assert "export GLOO_SOCKET_IFNAME=ens3f3" in commands
    assert f"--node-rank {node_rank}" in final_command
    assert "--nnodes 4" in final_command
    assert "--master-addr 7.6.25.57" in final_command
    assert "--master-port 29501" in final_command
    assert "--headless" in final_command
    assert "--host " not in final_command
    assert "--port " not in final_command
    assert "--enable-auto-tool-choice" not in final_command
    assert "--tool-call-parser" not in final_command
    assert "--reasoning-parser" not in final_command
    assert "--served-model-name" not in final_command
    assert "--tensor-parallel-size 8" in final_command
    assert "--data-parallel-size 4" in final_command


def test_mp_worker_keeps_served_model_name_outside_kimi_k3_h20_scope(monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "29501")

    commands = vllm_distributed._build_mp_commands({}, _mp_ctx(node_rank=1))

    assert "--served-model-name kimi_k3" in commands[-1]


def test_vllm_distributed_mp_branch_does_not_fall_through_to_dp(monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "29501")
    script = vllm_distributed._build_vllm_distributed_script(
        {
            "distributed_executor_backend": "mp",
            "node_rank": 1,
            "nnodes": 4,
            "master_ip": "7.6.25.57",
            "node_ips": "7.6.25.57,7.6.25.58,7.6.25.59,7.6.25.60",
        },
        _mp_ctx(node_rank=1).cmd,
        ["export COMMON_ENV=1"],
        "vllm",
        "",
    )

    assert "export COMMON_ENV=1" in script
    assert "--distributed-executor-backend mp" in script
    assert "--node-rank 1" in script
    assert "--data-parallel-size 4" in script
    assert "ray start" not in script


@pytest.mark.parametrize("master_port", ["0", "65536"])
def test_kimi_k3_mp_rejects_out_of_range_master_port(monkeypatch, master_port):
    monkeypatch.setenv("MASTER_PORT", master_port)

    with pytest.raises(ValueError, match="range 1..65535"):
        vllm_distributed._build_mp_commands({}, _mp_ctx())


def test_dp_deployment_preserves_explicit_dp_with_user_tp(monkeypatch):
    class _FakeModelIdentifier:
        model_architecture = "DeepseekV3ForCausalLM"

        def __init__(self, *_args):
            pass

    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeModelIdentifier)
    commands = vllm_distributed._build_dp_deployment_commands(
        {
            "device_count": 8,
            "engine_config": {
                "tensor_parallel_size": 4,
                "data_parallel_size": 99,
            },
            "_explicit_cli_keys": {
                "tensor_parallel_size",
                "data_parallel_size",
            },
            "rpc_port": 12890,
        },
        _ctx(engine="vllm_ascend"),
    )
    final_command = commands[-1]

    assert "--data-parallel-size 99" in final_command
    assert "--data-parallel-size-local 2" in final_command
    assert "--data-parallel-start-rank 2" in final_command


def test_dp_deployment_keeps_automatic_topology_without_explicit_dp():
    topology = vllm_distributed._resolve_dp_deployment_topology(
        {
            "device_count": 8,
            "engine_config": {"tensor_parallel_size": 4},
        },
        _ctx(engine="vllm_ascend"),
        SimpleNamespace(model_architecture="DeepseekV3ForCausalLM"),
    )

    assert topology == DpDeploymentTopology("4", "2", "2")


def test_dp_deployment_final_command_rebuilds_explicit_dp_once(monkeypatch):
    class _FakeModelIdentifier:
        model_architecture = "OtherForCausalLM"

        def __init__(self, *_args):
            pass

    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeModelIdentifier)
    commands = vllm_distributed._build_dp_deployment_commands(
        {
            "engine_config": {"data_parallel_size": 99},
            "_explicit_cli_keys": {"data_parallel_size"},
            "rpc_port": 12890,
        },
        _ctx(node_rank=0),
    )
    final_command = commands[-1]

    assert final_command.count("--data-parallel-size ") == 1
    assert "--data-parallel-size 99" in final_command
    assert "--data-parallel-size-local 1" in final_command
    assert "--data-parallel-start-rank" not in final_command


def test_dp_deployment_reads_explicit_dp_from_environment(monkeypatch):
    class _FakeModelIdentifier:
        model_architecture = "OtherForCausalLM"

        def __init__(self, *_args):
            pass

    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeModelIdentifier)
    monkeypatch.setenv("DATA_PARALLEL_SIZE", "12")
    commands = vllm_distributed._build_dp_deployment_commands(
        {
            "engine_config": {},
            "_explicit_cli_keys": {"data_parallel_size"},
            "rpc_port": 12890,
        },
        _ctx(node_rank=0),
    )

    assert "--data-parallel-size 12" in commands[-1]
    assert "--data-parallel-size-local 1" in commands[-1]


@pytest.mark.parametrize("value", [0, -1, "invalid"])
def test_dp_deployment_rejects_invalid_explicit_dp(monkeypatch, value):
    class _FakeModelIdentifier:
        model_architecture = "OtherForCausalLM"

        def __init__(self, *_args):
            pass

    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeModelIdentifier)
    with pytest.raises(ValueError, match="requires positive data_parallel_size"):
        vllm_distributed._build_dp_deployment_commands(
            {
                "engine_config": {"data_parallel_size": value},
                "_explicit_cli_keys": {"data_parallel_size"},
                "rpc_port": 12890,
            },
            _ctx(node_rank=0),
        )


def test_dp_deployment_keeps_generic_automatic_topology(monkeypatch):
    class _FakeModelIdentifier:
        model_architecture = "OtherForCausalLM"

        def __init__(self, *_args):
            pass

    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeModelIdentifier)
    commands = vllm_distributed._build_dp_deployment_commands(
        {
            "engine_config": {"tensor_parallel_size": 2},
            "_explicit_cli_keys": {"tensor_parallel_size"},
            "rpc_port": 12890,
        },
        _ctx(node_rank=1, nnodes=2),
    )
    final_command = commands[-1]

    assert "--data-parallel-size 2" in final_command
    assert "--data-parallel-size-local 1" in final_command
    assert "--data-parallel-start-rank 1" in final_command
