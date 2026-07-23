import json
import logging
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from core.engine_manager import start_engine_service  # noqa: E402
from core.port_plan import PortPlan  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from core.wings_entry import _prepare_merged_params  # noqa: E402
from engines import vllm_adapter  # noqa: E402


_CLEAR_ENV = (
    "ASCEND_PLATFORM",
    "BLOCK_SIZE",
    "CONFIG_FILE",
    "DEVICE_COUNT",
    "DP_SIZE",
    "DP_SIZE_LOCAL",
    "DTYPE",
    "DISTRIBUTED_EXECUTOR_BACKEND",
    "ENABLE_AUTO_THINK_CHOICE",
    "ENABLE_AUTO_TOOL_CHOICE",
    "ENABLE_CHUNKED_PREFILL",
    "ENABLE_EXPERT_PARALLEL",
    "ENABLE_PREFIX_CACHING",
    "ENABLE_RAG_ACC",
    "ENABLE_SPARSE",
    "ENABLE_SPECULATIVE_DECODE",
    "ENGINE",
    "ENGINE_PORT",
    "GLOO_SOCKET_IFNAME",
    "GPU_MEMORY_UTILIZATION",
    "HCCL_BUFFSIZE",
    "HOST_IP",
    "INPUT_LENGTH",
    "KV_CACHE_DTYPE",
    "MASTER_IP",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_NUM_SEQS",
    "MODEL_NAME",
    "MODEL_PATH",
    "MODEL_TYPE",
    "NETWORK_INTERFACE",
    "NO_ENABLE_PREFIX_CACHING",
    "NODE_IPS",
    "OMP_NUM_THREADS",
    "OUTPUT_LENGTH",
    "PD_DECODE_DP_SIZE",
    "PD_DECODE_TP_SIZE",
    "PD_DP_ADDRESS",
    "PD_DP_RANK_START",
    "PD_DP_SIZE",
    "PD_DP_SIZE_LOCAL",
    "PD_INDEX",
    "PD_PREFILL_DP_SIZE",
    "PD_PREFILL_TP_SIZE",
    "PD_ROLE",
    "PD_TP_SIZE",
    "POD_IP",
    "PORT",
    "QUANTIZATION",
    "RANK_IP",
    "SEED",
    "SERVED_MODEL_NAME",
    "SPECULATIVE_DECODE_MODEL_PATH",
    "TP_SIZE",
    "TRUST_REMOTE_CODE",
    "VLLM_LLMDD_RPC_PORT",
    "VLLM_MOONCAKE_BOOTSTRAP_PORT",
    "WINGS_ASCEND_PLATFORM",
    "WINGS_ENGINE",
)

_PREFILL_IP = "10.254.124.131"
_DECODE_IP = "10.254.124.182"
_VLLM_START_PORT = 7100


def test_pd_external_lb_1p1d_missing_tp_uses_device_count(monkeypatch):
    for name in _CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PD_ROLE", "P")

    ext = config_loader._get_pd_external_lb_params(device_count=2)

    assert ext["tp_size"] == 2
    assert ext["dp_size"] == 1
    assert ext["dp_size_local"] == 1


def _write_deepseek_v4_config(model_dir: Path) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": 61,
                "quantization_config": {"quant_method": "w8a8"},
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )


def _extract_exports(script: str) -> dict[str, str]:
    exports = {}
    for line in script.splitlines():
        stripped = line.strip()
        match = re.match(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", stripped)
        if match:
            exports[match.group(1)] = stripped
    return exports


def _export_names(script: str) -> list[str]:
    names = []
    for line in script.splitlines():
        match = re.match(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=", line)
        if match:
            names.append(match.group(1))
    return names


def _assert_no_duplicate_export_names(script: str) -> None:
    names = _export_names(script)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == []


def _render_pd_deepseek_v4_script(tmp_path, monkeypatch, role: str, local_ip: str, pd_index: int) -> str:
    for name in _CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setenv("PD_ROLE", role)
    monkeypatch.setenv("PD_INDEX", str(pd_index))
    monkeypatch.setenv("PD_PREFILL_DP_SIZE", "2")
    monkeypatch.setenv("PD_PREFILL_TP_SIZE", "4")
    monkeypatch.setenv("PD_DECODE_DP_SIZE", "8")
    monkeypatch.setenv("PD_DECODE_TP_SIZE", "1")
    monkeypatch.setenv("DP_SIZE_LOCAL", "1" if role == "P" else "8")
    monkeypatch.setenv("MASTER_IP", local_ip)
    monkeypatch.setenv("RANK_IP", local_ip)
    monkeypatch.setenv("HOST_IP", local_ip)
    monkeypatch.setenv("POD_IP", local_ip)
    monkeypatch.setenv("NODE_IPS", local_ip)
    monkeypatch.setenv("NETWORK_INTERFACE", "xxxx")
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")

    model_dir = tmp_path / f"deepseek-v4-{role}"
    _write_deepseek_v4_config(model_dir)
    launch_args = parse_launch_args(
        [
            "--model-name",
            "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "--model-path",
            str(model_dir),
            "--model-type",
            "llm",
            "--engine",
            "vllm_ascend",
            "--device-count",
            "8",
            "--trust-remote-code",
        ]
    )
    monkeypatch.setattr(sys, "argv", ["pytest"])
    merged = _prepare_merged_params(
        launch_args,
        PortPlan(
            enable_proxy=True,
            backend_port=_VLLM_START_PORT,
            proxy_port=18000,
            health_port=19000,
        ),
        {"device": "ascend", "count": 8, "details": [{"name": "Ascend"}] * 8},
    )
    return start_engine_service(merged)


def _assert_user_pd_topology(script: str, role: str) -> None:
    compact = re.sub(r"\s+", "", script)
    assert '"prefill":{"dp_size":2,"tp_size":4' in compact
    assert '"decode":{"dp_size":8,"tp_size":1' in compact

    if role == "P":
        assert "for i in $(seq 0 0); do" in script
        assert "RANK=$((0 + i)); PORT=$((7100 + i))" in script
        assert "--tensor-parallel-size 4 --data-parallel-size 2" in script
        assert "--data-parallel-address 10.254.124.131" in script
    else:
        assert "for i in $(seq 0 7); do" in script
        assert "RANK=$((0 + i)); PORT=$((7100 + i))" in script
        assert "--tensor-parallel-size 1 --data-parallel-size 8" in script
        assert "--data-parallel-address 10.254.124.182" in script


def _assert_common_official_deepseek_v4_pd_env(exports: dict[str, str]) -> None:
    assert exports["VLLM_RPC_TIMEOUT"] == "export VLLM_RPC_TIMEOUT=3600000"
    assert exports["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == (
        "export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000"
    )
    assert exports["HCCL_EXEC_TIMEOUT"] == "export HCCL_EXEC_TIMEOUT=204"
    assert exports["OMP_PROC_BIND"] == "export OMP_PROC_BIND=false"
    assert exports["OMP_NUM_THREADS"] == "export OMP_NUM_THREADS=10"
    assert exports["PYTORCH_NPU_ALLOC_CONF"] == (
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
    )
    assert exports["TASK_QUEUE_ENABLE"] == "export TASK_QUEUE_ENABLE=1"
    assert exports["HCCL_OP_EXPANSION_MODE"] == "export HCCL_OP_EXPANSION_MODE=AIV"

    for unwanted in (
        "CLOSE_MATMUL_K_SHIFT",
        "HCCL_DETERMINISTIC",
        "LCCL_DETERMINISTIC",
        "LD_LIBRARY_PATH",
        "VLLM_LLMDD_RPC_PORT",
        "VLLM_MOONCAKE_BOOTSTRAP_PORT",
        "VLLM_USE_V1",
    ):
        assert unwanted not in exports


def test_deepseek_v4_pd_prefill_env_matches_official_recipe(tmp_path, monkeypatch):
    script = _render_pd_deepseek_v4_script(tmp_path, monkeypatch, "P", _PREFILL_IP, 0)
    exports = _extract_exports(script)

    _assert_no_duplicate_export_names(script)
    _assert_user_pd_topology(script, "P")
    _assert_common_official_deepseek_v4_pd_env(exports)
    assert exports["HCCL_IF_IP"] == f"export HCCL_IF_IP={_PREFILL_IP}"
    assert exports["GLOO_SOCKET_IFNAME"] == "export GLOO_SOCKET_IFNAME=xxxx"
    assert exports["TP_SOCKET_IFNAME"] == "export TP_SOCKET_IFNAME=xxxx"
    assert exports["HCCL_SOCKET_IFNAME"] == "export HCCL_SOCKET_IFNAME=xxxx"
    assert exports["HCCL_BUFFSIZE"] == "export HCCL_BUFFSIZE=2560"
    assert exports["HCCL_CONNECT_TIMEOUT"] == "export HCCL_CONNECT_TIMEOUT=120"
    assert exports["VLLM_ASCEND_ENABLE_FLASHCOMM1"] == (
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1"
    )
    assert "VLLM_ASCEND_ENABLE_FUSED_MC2" not in exports
    assert "VLLM_ASCEND_ENABLE_MLAPO" not in exports
    assert "VLLM_MOONCAKE_BOOTSTRAP_PORT=" not in script


def test_deepseek_v4_pd_decode_env_matches_official_recipe(tmp_path, monkeypatch):
    script = _render_pd_deepseek_v4_script(tmp_path, monkeypatch, "D", _DECODE_IP, 1)
    exports = _extract_exports(script)

    _assert_no_duplicate_export_names(script)
    _assert_user_pd_topology(script, "D")
    _assert_common_official_deepseek_v4_pd_env(exports)
    assert exports["HCCL_IF_IP"] == f"export HCCL_IF_IP={_DECODE_IP}"
    assert exports["GLOO_SOCKET_IFNAME"] == "export GLOO_SOCKET_IFNAME=xxxx"
    assert exports["TP_SOCKET_IFNAME"] == "export TP_SOCKET_IFNAME=xxxx"
    assert exports["HCCL_SOCKET_IFNAME"] == "export HCCL_SOCKET_IFNAME=xxxx"
    assert exports["HCCL_BUFFSIZE"] == "export HCCL_BUFFSIZE=1024"
    assert exports["HCCL_CONNECT_TIMEOUT"] == "export HCCL_CONNECT_TIMEOUT=1200"
    assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in exports
    assert "VLLM_ASCEND_ENABLE_FUSED_MC2" not in exports
    assert "VLLM_ASCEND_ENABLE_MLAPO" not in exports
    assert "VLLM_MOONCAKE_BOOTSTRAP_PORT=" not in script


def test_deepseek_v4_pd_refreshes_mooncake_linker_cache_without_env_leak(tmp_path, monkeypatch):
    script = _render_pd_deepseek_v4_script(tmp_path, monkeypatch, "D", _DECODE_IP, 1)
    exports = _extract_exports(script)

    assert "export LD_LIBRARY_PATH" not in script
    assert "LD_LIBRARY_PATH" not in exports
    assert "ldconfig /usr/local/lib >/dev/null 2>&1 || true" in script
    assert (
        "LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-} "
        "ASCEND_RT_VISIBLE_DEVICES=$CARDS"
    ) in script


def test_deepseek_v4_pd_env_does_not_use_generic_env_builders(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vllm_adapter,
        "_build_pd_role_env_commands",
        lambda *args, **kwargs: ["export SHOULD_NOT_LEAK_FROM_GENERIC_PD=1"],
    )
    monkeypatch.setattr(
        vllm_adapter,
        "_build_model_env_commands",
        lambda *args, **kwargs: ["export SHOULD_NOT_LEAK_FROM_MODEL_ENV=1"],
    )

    script = _render_pd_deepseek_v4_script(tmp_path, monkeypatch, "P", _PREFILL_IP, 0)

    assert "SHOULD_NOT_LEAK_FROM_GENERIC_PD" not in script
    assert "SHOULD_NOT_LEAK_FROM_MODEL_ENV" not in script
    assert "export HCCL_IF_IP=10.254.124.131" in script
    assert "export VLLM_RPC_TIMEOUT=3600000" in script


def test_deepseek_v4_pd_logs_env_trigger_path(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    _render_pd_deepseek_v4_script(tmp_path, monkeypatch, "D", _DECODE_IP, 1)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "[PD external-lb trigger]" in logs
    assert "[vllm_adapter.env_path] selected=pd_external_lb_isolated" in logs
    assert "[PD external-lb env] isolated_builder=True" in logs
    assert "[PD external-lb env merge]" in logs
    assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" in logs
    assert "stripped_env=[]" in logs


def test_nvidia_pd_does_not_use_ascend_external_lb_mooncake_registry(tmp_path, monkeypatch):
    for name in _CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setenv("PD_ROLE", "P")
    monkeypatch.setenv("PD_INDEX", "0")
    monkeypatch.setenv("PD_PREFILL_DP_SIZE", "2")
    monkeypatch.setenv("PD_PREFILL_TP_SIZE", "4")
    monkeypatch.setenv("PD_DECODE_DP_SIZE", "2")
    monkeypatch.setenv("PD_DECODE_TP_SIZE", "4")
    monkeypatch.setenv("DP_SIZE_LOCAL", "1")
    monkeypatch.setenv("MASTER_IP", _PREFILL_IP)
    monkeypatch.setenv("RANK_IP", _PREFILL_IP)
    monkeypatch.setenv("HOST_IP", _PREFILL_IP)
    monkeypatch.setenv("POD_IP", _PREFILL_IP)
    monkeypatch.setenv("NODE_IPS", _PREFILL_IP)

    model_dir = tmp_path / "deepseek-v4-nvidia"
    _write_deepseek_v4_config(model_dir)
    launch_args = parse_launch_args(
        [
            "--model-name",
            "deepseek-ai/DeepSeek-V4-Flash",
            "--model-path",
            str(model_dir),
            "--model-type",
            "llm",
            "--engine",
            "vllm",
            "--device-count",
            "8",
            "--trust-remote-code",
        ]
    )
    monkeypatch.setattr(sys, "argv", ["pytest"])
    merged = _prepare_merged_params(
        launch_args,
        PortPlan(
            enable_proxy=True,
            backend_port=_VLLM_START_PORT,
            proxy_port=18000,
            health_port=19000,
        ),
        {"device": "nvidia", "count": 8, "details": [{"name": "H20"}] * 8},
    )

    assert "_pd_external_lb" not in merged
    engine_config = merged["engine_config"]
    assert engine_config.get("quantization") != "ascend"
    kv_transfer = json.loads(engine_config["kv_transfer_config"])
    assert kv_transfer["kv_connector"] == "NixlConnector"
    assert "Mooncake" not in kv_transfer["kv_connector"]
