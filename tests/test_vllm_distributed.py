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
