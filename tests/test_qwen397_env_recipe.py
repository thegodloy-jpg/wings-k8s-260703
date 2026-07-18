import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import wings_entry  # noqa: E402
from engines import vllm_adapter  # noqa: E402


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
    exports = [command for command in commands if command.startswith("export ")]

    # 397B w8a8-mtp 的 910C 标准脚本要求 env 集合精确对齐；
    # 通用 Qwen-MoE / base env 里的额外 export 不能继续透传。
    assert commands[0] == "unset ASCEND_RT_VISIBLE_DEVICES"
    assert exports == [
        "export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}",
        "export HCCL_IF_IP=127.0.0.1",
        "export GLOO_SOCKET_IFNAME=lo",
        "export TP_SOCKET_IFNAME=lo",
        "export HCCL_SOCKET_IFNAME=lo",
        "export PYTHONHASHSEED=0",
        "export HCCL_BUFFSIZE=512",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export VLLM_USE_V1=1",
        "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}",
        "export TASK_QUEUE_ENABLE=1",
        'export HCCL_OP_EXPANSION_MODE="AIV"',
    ]
    assert "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true" in commands
    assert "source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true" in commands
    assert all("LD_LIBRARY_PATH" not in command for command in exports)

    startup = wings_entry._assemble_startup_command(
        "vllm_ascend",
        params,
        {"device": "ascend", "details": [{"name": "Ascend910C"}]},
        "exec true\n",
        "",
    )

    assert "export PROMETHEUS_MULTIPROC_DIR=" not in startup
    assert "export PYTHONUNBUFFERED=" not in startup
