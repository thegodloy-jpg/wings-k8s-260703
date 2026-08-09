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
            "--tensor-parallel-size 32 "
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


def _kimi_k3_h20_mp_params():
    return {
        "model_name": "Kimi-K3",
        "device_count": 8,
        "nnodes": 4,
        "_smart_card_token": "h20-141",
    }


def _kimi_k3_910c_ctx(*, node_rank=0):
    return DistScriptCtx(
        engine="vllm_ascend",
        cmd=(
            "vllm serve /data/Kimi-K3-w4a8 --host 0.0.0.0 --port 18000 "
            "--served-model-name kimi-k3 --allowed-local-media-path / "
            "--trust-remote-code --tensor-parallel-size 16 --enable-prefix-caching "
            "--enable-expert-parallel --max-num-seqs 16 --max-model-len 131027 "
            "--max-num-batched-tokens 4096 --gpu-memory-utilization 0.9 "
            "--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' "
            "--profiler-config "
            "'{\"profiler\":\"torch\",\"torch_profiler_dir\":\"./vllm_profile\","
            "\"torch_profiler_with_stack\":false}' "
            "--mm-processor-cache-gb 0 "
            "--additional-config "
            "'{\"enable_cpu_binding\":true,\"enable_flashcomm1\":true,"
            "\"enable_mc2_hierarchy_comm\":true}' "
            "--mm-encoder-tp-mode data --limit-mm-per-prompt '{\"vision_chunk\":40}' "
            "--enable-auto-tool-choice --reasoning-parser kimi_k3 --tool-call-parser kimi_k3"
        ),
        is_ascend=True,
        node_rank=node_rank,
        nnodes=4,
        head_addr="7.6.28.252",
        ray_port="28020",
        node_ips="7.6.28.252,7.6.28.253,7.6.28.241,7.6.28.240",
    )


class _FakeKimiK3ModelIdentifier:
    model_architecture = "KimiK3ForConditionalGeneration"

    def __init__(self, *_args):
        pass


def _kimi_k3_910c_params(node_rank):
    return {
        "engine": "vllm_ascend",
        "model_name": "Kimi-K3-w4a8",
        "model_path": "/data/Kimi-K3-w4a8",
        "device_count": 16,
        "distributed": True,
        "nnodes": 4,
        "node_rank": node_rank,
        "rpc_port": 27777,
        "engine_config": {"tensor_parallel_size": 16},
        "_smart_card_token": "910c",
        "_kimi_k3_910c_dp": True,
        "_preserve_dp_worker_port": True,
    }


def test_kimi_k3_mp_rank0_keeps_frontend_and_native_topology(monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "enp66s0f1")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "enp66s0f1")

    commands = vllm_distributed._build_mp_commands(
        _kimi_k3_h20_mp_params(),
        _mp_ctx(node_rank=0),
    )
    final_command = commands[-1]

    assert "export NCCL_SOCKET_IFNAME=enp66s0f1" in commands
    assert "export GLOO_SOCKET_IFNAME=enp66s0f1" in commands
    assert "export VLLM_ENGINE_READY_TIMEOUT_S=3600" in commands
    assert "export VLLM_USE_V2_MODEL_RUNNER=1" in commands
    assert "export VLLM_USE_RUST_FRONTEND=1" in commands
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in commands
    assert "ulimit -l unlimited" in commands
    # 新配方只补充镜像要求的运行环境，不恢复旧 NCCL/SSM 强制策略。
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
    assert "--tensor-parallel-size 32" in final_command
    assert "--data-parallel-size" not in final_command


@pytest.mark.parametrize("node_rank", [1, 2, 3])
def test_kimi_k3_mp_worker_is_headless_and_uses_local_nic(monkeypatch, node_rank):
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ens3f3")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "ens3f3")

    commands = vllm_distributed._build_mp_commands(
        _kimi_k3_h20_mp_params(),
        _mp_ctx(node_rank=node_rank),
    )
    final_command = commands[-1]

    assert "export NCCL_SOCKET_IFNAME=ens3f3" in commands
    assert "export GLOO_SOCKET_IFNAME=ens3f3" in commands
    assert "export VLLM_ENGINE_READY_TIMEOUT_S=3600" in commands
    assert "export VLLM_USE_V2_MODEL_RUNNER=1" in commands
    assert "export VLLM_USE_RUST_FRONTEND=1" in commands
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in commands
    assert "ulimit -l unlimited" in commands
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
    assert "--served-model-name kimi_k3" in final_command
    assert "--tensor-parallel-size 32" in final_command
    assert "--data-parallel-size" not in final_command


def test_generic_mp_worker_keeps_served_model_name_without_kimi_runtime_env(monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "29501")

    commands = vllm_distributed._build_mp_commands({}, _mp_ctx(node_rank=1))

    assert "--served-model-name kimi_k3" in commands[-1]
    assert not any("VLLM_USE_V2_MODEL_RUNNER" in command for command in commands)
    assert not any("VLLM_USE_RUST_FRONTEND" in command for command in commands)
    assert "unset PYTORCH_CUDA_ALLOC_CONF" not in commands
    assert "ulimit -l unlimited" not in commands


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
    assert "--tensor-parallel-size 32" in script
    assert "--data-parallel-size" not in script
    assert "ray start" not in script


@pytest.mark.parametrize("master_port", ["0", "65536"])
def test_kimi_k3_mp_rejects_out_of_range_master_port(monkeypatch, master_port):
    monkeypatch.setenv("MASTER_PORT", master_port)

    with pytest.raises(ValueError, match="range 1..65535"):
        vllm_distributed._build_mp_commands({}, _mp_ctx())


def test_kimi_k3_w4a8_910c_rank0_matches_standard_dp_recipe(monkeypatch):
    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeKimiK3ModelIdentifier)
    monkeypatch.setenv("NETWORK_INTERFACE", "enp196s0f0")
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "enp196")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("HCCL_BUFFSIZE", "800")
    monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", "7200")

    commands = vllm_distributed._build_dp_deployment_commands(
        _kimi_k3_910c_params(0),
        _kimi_k3_910c_ctx(node_rank=0),
    )
    final_command = commands[-1]

    expected_env = {
        "export HCCL_IF_IP=$VLLM_HOST_IP",
        "export GLOO_SOCKET_IFNAME=enp196s0f0",
        "export TP_SOCKET_IFNAME=enp196s0f0",
        "export HCCL_SOCKET_IFNAME=enp196",
        "export VLLM_ENGINE_READY_TIMEOUT_S=7200",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
        "export HCCL_BUFFSIZE=800",
        "export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        "export HCCL_INTER_HCCS_DISABLE=true",
        "export HCCL_INTRA_ROCE_ENABLE=1",
        "export VLLM_ASCEND_ENABLE_FUSED_MC2=0",
        "export HCCL_LOGIC_SUPERPOD_ID=0",
    }
    assert expected_env.issubset(set(commands))
    assert not any("HCCL_WHITELIST_DISABLE" in command for command in commands)
    assert not any("HCCL_CONNECT_TIMEOUT" in command for command in commands)
    assert not any("HCCL_EXEC_TIMEOUT" in command for command in commands)
    assert "--data-parallel-address 7.6.28.252" in final_command
    assert "--data-parallel-rpc-port 27777" in final_command
    assert "--data-parallel-size 4" in final_command
    assert "--data-parallel-size-local 1" in final_command
    assert "--data-parallel-start-rank" not in final_command
    assert "--headless" not in final_command
    assert "--host 0.0.0.0" in final_command
    assert "--port 18000" in final_command
    assert "--profiler-config" in final_command
    assert "--enable-auto-tool-choice" in final_command
    assert "--reasoning-parser kimi_k3" in final_command
    assert "--tool-call-parser kimi_k3" in final_command


@pytest.mark.parametrize("node_rank", [0, 1, 2, 3])
def test_kimi_k3_w4a8_910c_full_script_drops_generic_ascend_env_defaults(
    monkeypatch,
    node_rank,
):
    vllm_adapter = vllm_distributed._import_vllm_adapter()
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeKimiK3ModelIdentifier)
    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeKimiK3ModelIdentifier)
    monkeypatch.setenv("POD_IP", "7.6.28.252")
    monkeypatch.setenv("NETWORK_INTERFACE", "enp196s0f0")
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "enp196")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("HCCL_BUFFSIZE", "800")

    params = _kimi_k3_910c_params(node_rank)
    params.update({
        "distributed_executor_backend": "dp_deployment",
        "master_ip": "7.6.28.252",
        "node_ips": "7.6.28.252,7.6.28.253,7.6.28.241,7.6.28.240",
        "host": "0.0.0.0",
        "port": 18000,
        "model_type": "llm",
    })
    script = vllm_adapter.build_start_script(params)
    exports = [line for line in script.splitlines() if line.startswith("export ")]

    expected_once = {
        "HCCL_BUFFSIZE": "export HCCL_BUFFSIZE=800",
        "OMP_PROC_BIND": "export OMP_PROC_BIND=false",
        "OMP_NUM_THREADS": "export OMP_NUM_THREADS=1",
        "TASK_QUEUE_ENABLE": "export TASK_QUEUE_ENABLE=1",
        "PYTORCH_NPU_ALLOC_CONF": "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
    }
    for env_name, expected in expected_once.items():
        assert [line for line in exports if line.startswith(f"export {env_name}=")] == [expected]
    assert not any(line.startswith("export HCCL_OP_EXPANSION_MODE=") for line in exports)
    assert f"export HCCL_LOGIC_SUPERPOD_ID={node_rank}" in exports
    assert "source /usr/local/Ascend/ascend-toolkit/set_env.sh" in script
    assert "libascend_hal.so" in script


@pytest.mark.parametrize("node_rank", [1, 2, 3])
def test_kimi_k3_w4a8_910c_workers_keep_port_and_use_ranked_superpod(
    monkeypatch,
    node_rank,
):
    monkeypatch.setattr(vllm_distributed, "ModelIdentifier", _FakeKimiK3ModelIdentifier)
    monkeypatch.setenv("NETWORK_INTERFACE", "enp196s0f1")
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "enp196")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("HCCL_BUFFSIZE", "800")

    commands = vllm_distributed._build_dp_deployment_commands(
        _kimi_k3_910c_params(node_rank),
        _kimi_k3_910c_ctx(node_rank=node_rank),
    )
    final_command = commands[-1]

    assert "export GLOO_SOCKET_IFNAME=enp196s0f1" in commands
    assert "export TP_SOCKET_IFNAME=enp196s0f1" in commands
    assert "export HCCL_SOCKET_IFNAME=enp196" in commands
    assert f"export HCCL_LOGIC_SUPERPOD_ID={node_rank}" in commands
    assert not any("VLLM_ENGINE_READY_TIMEOUT_S" in command for command in commands)
    assert "--host " not in final_command
    assert "--port 18000" in final_command
    assert "--headless" in final_command
    assert f"--data-parallel-start-rank {node_rank}" in final_command
    assert "--data-parallel-address 7.6.28.252" in final_command
    assert "--data-parallel-rpc-port 27777" in final_command
    assert "--data-parallel-size 4" in final_command
    assert "--data-parallel-size-local 1" in final_command
    assert "--served-model-name kimi-k3" in final_command
    assert "--tensor-parallel-size 16" in final_command


def test_kimi_k3_w4a8_910c_recipe_rejects_wrong_card(monkeypatch):
    params = _kimi_k3_910c_params(0)
    params["_smart_card_token"] = "910b"
    monkeypatch.setenv("NETWORK_INTERFACE", "enp196s0f0")

    with pytest.raises(ValueError, match="requires Ascend 910C/A3"):
        vllm_distributed._build_dp_env_commands(
            True,
            params,
            "KimiK3ForConditionalGeneration",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("device_count", 8, "requires device_count=16"),
        ("nnodes", 2, "requires nnodes=4"),
        ("node_rank", 4, "requires node_rank in 0..3"),
    ],
)
def test_kimi_k3_w4a8_910c_recipe_rejects_invalid_topology(
    monkeypatch,
    field,
    value,
    message,
):
    params = _kimi_k3_910c_params(0)
    params[field] = value
    monkeypatch.setenv("NETWORK_INTERFACE", "enp196s0f0")

    with pytest.raises(ValueError, match=message):
        vllm_distributed._build_dp_env_commands(
            True,
            params,
            "KimiK3ForConditionalGeneration",
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
    assert "--host " not in final_command
    assert "--port " not in final_command
