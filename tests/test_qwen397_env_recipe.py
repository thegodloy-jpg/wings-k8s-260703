import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import wings_entry  # noqa: E402
from engines import vllm_adapter  # noqa: E402


class _FakeQwen35MoeIdentifier:
    model_architecture = "Qwen3_5MoeForConditionalGeneration"
    model_quantize = ""
    config = {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}

    def __init__(self, *args, **kwargs):
        pass


def _export_by_name(commands):
    return {
        command.split(" ", 1)[1].split("=", 1)[0]: command
        for command in commands
        if command.startswith("export ") and "=" in command
    }


def test_qwen35_397b_w8a8_mtp_910c_env_matches_day0_script():
    params = {
        "engine": "vllm_ascend",
        "model_name": "Qwen/Qwen3.5-397B-A17B-w8a8-mtp",
        "model_path": "/models/Qwen3.5-397B-A17B-w8a8-mtp",
        "device_count": 8,
        "nnodes": 1,
        "device_details": [{"name": "Ascend910C"}],
    }

    commands = vllm_adapter._build_vllm_common_env_cmds(params, "vllm_ascend")
    export_by_name = _export_by_name(commands)

    assert "ASCEND_RT_VISIBLE_DEVICES" not in export_by_name
    assert export_by_name.get("HCCL_IF_IP") != "export HCCL_IF_IP=127.0.0.1"
    assert export_by_name.get("GLOO_SOCKET_IFNAME") != "export GLOO_SOCKET_IFNAME=lo"
    assert export_by_name.get("TP_SOCKET_IFNAME") != "export TP_SOCKET_IFNAME=lo"
    assert export_by_name.get("HCCL_SOCKET_IFNAME") != "export HCCL_SOCKET_IFNAME=lo"
    assert export_by_name["PYTHONHASHSEED"] == "export PYTHONHASHSEED=0"
    assert export_by_name["HCCL_BUFFSIZE"] == "export HCCL_BUFFSIZE=512"
    assert export_by_name["OMP_PROC_BIND"] == "export OMP_PROC_BIND=false"
    assert export_by_name["OMP_NUM_THREADS"] == "export OMP_NUM_THREADS=1"
    assert export_by_name["PYTORCH_NPU_ALLOC_CONF"] == (
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
    )
    assert export_by_name["VLLM_USE_V1"] == "export VLLM_USE_V1=1"
    assert export_by_name["LD_PRELOAD"] == (
        "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
    )
    assert export_by_name["TASK_QUEUE_ENABLE"] == "export TASK_QUEUE_ENABLE=1"
    assert export_by_name["HCCL_OP_EXPANSION_MODE"] == 'export HCCL_OP_EXPANSION_MODE="AIV"'

    network_params = {
        **params,
        "distributed": True,
        "distributed_executor_backend": "ray",
    }
    network_commands = vllm_adapter._build_env_commands(
        network_params,
        "10.42.0.7",
        "eth9",
        str(Path(__file__).resolve().parents[1] / "wings_control"),
    )
    network_exports = _export_by_name(network_commands)
    assert network_exports["HCCL_IF_IP"] == "export HCCL_IF_IP=10.42.0.7"
    assert network_exports["GLOO_SOCKET_IFNAME"] == "export GLOO_SOCKET_IFNAME=eth9"
    assert network_exports["TP_SOCKET_IFNAME"] == "export TP_SOCKET_IFNAME=eth9"
    assert network_exports.get("HCCL_SOCKET_IFNAME") != "export HCCL_SOCKET_IFNAME=lo"

    startup = wings_entry._assemble_startup_command(
        "vllm_ascend",
        params,
        {"device": "ascend", "details": [{"name": "Ascend910C"}]},
        "exec true\n",
        "",
    )

    assert "export PROMETHEUS_MULTIPROC_DIR=" not in startup
    assert "export PYTHONUNBUFFERED=" not in startup


def test_qwen35_122b_a10b_ascend_env_adds_runtime_flags_only(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen35MoeIdentifier)
    params = {
        "engine": "vllm_ascend",
        "model_name": "Qwen/Qwen3.5-122B-A10B",
        "model_path": "/models/Qwen3.5-122B-A10B",
        "device_count": 8,
        "nnodes": 1,
        "device_details": [{"name": "Ascend910C"}],
    }

    commands = vllm_adapter._build_vllm_common_env_cmds(params, "vllm_ascend")
    export_by_name = _export_by_name(commands)

    assert export_by_name["PYTHONHASHSEED"] == "export PYTHONHASHSEED=0"
    assert export_by_name["VLLM_USE_V1"] == "export VLLM_USE_V1=1"
    assert export_by_name["LD_PRELOAD"] == (
        "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
    )
    assert "ASCEND_RT_VISIBLE_DEVICES" not in export_by_name
    assert export_by_name.get("HCCL_IF_IP") != "export HCCL_IF_IP=127.0.0.1"
    assert export_by_name.get("GLOO_SOCKET_IFNAME") != "export GLOO_SOCKET_IFNAME=lo"
    assert export_by_name.get("TP_SOCKET_IFNAME") != "export TP_SOCKET_IFNAME=lo"
    assert export_by_name.get("HCCL_SOCKET_IFNAME") != "export HCCL_SOCKET_IFNAME=lo"

    network_params = {
        **params,
        "distributed": True,
        "distributed_executor_backend": "ray",
    }
    network_commands = vllm_adapter._build_env_commands(
        network_params,
        "10.42.0.7",
        "eth9",
        str(Path(__file__).resolve().parents[1] / "wings_control"),
    )
    network_exports = _export_by_name(network_commands)
    assert network_exports["HCCL_IF_IP"] == "export HCCL_IF_IP=10.42.0.7"
    assert network_exports["GLOO_SOCKET_IFNAME"] == "export GLOO_SOCKET_IFNAME=eth9"
    assert network_exports["TP_SOCKET_IFNAME"] == "export TP_SOCKET_IFNAME=eth9"

    startup = wings_entry._assemble_startup_command(
        "vllm_ascend",
        params,
        {"device": "ascend", "details": [{"name": "Ascend910C"}]},
        "exec true\n",
        "",
    )

    assert "export PROMETHEUS_MULTIPROC_DIR=" in startup
    assert "export PYTHONUNBUFFERED=1" in startup
