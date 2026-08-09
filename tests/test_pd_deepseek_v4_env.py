import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from core.engine_manager import start_engine_service  # noqa: E402
from core.port_plan import PortPlan  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from core.wings_entry import _prepare_merged_params  # noqa: E402
from engines import vllm_adapter  # noqa: E402


_CLEAR_ENV = (
    "ASCEND_A3_ENABLE",
    "ASCEND_PLATFORM",
    "BLOCK_SIZE",
    "CONFIG_FILE",
    "DATA_PARALLEL_SIZE",
    "DEVICE_COUNT",
    "DP_SIZE",
    "DP_SIZE_LOCAL",
    "DTYPE",
    "DISTRIBUTED_EXECUTOR_BACKEND",
    "ENGINE_IMAGE_FLAVOR",
    "ENGINE_VERSION",
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
    "TENSOR_PARALLEL_SIZE",
    "VLLM_LLMDD_RPC_PORT",
    "VLLM_MOONCAKE_BOOTSTRAP_PORT",
    "VLLM_SSM_CONV_STATE_LAYOUT",
    "WINGS_ASCEND_PLATFORM",
    "WINGS_ENGINE",
)

_PREFILL_IP = "10.254.124.131"
_DECODE_IP = "10.254.124.182"
_VLLM_START_PORT = 7100


def _write_arch_config(model_dir: Path, architecture: str) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": [architecture]}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("role", ("P", "D"))
def test_pd_standalone_kv_topology_uses_local_device_count(monkeypatch, role):
    for name in _CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PD_ROLE", role)

    config = config_loader._get_pd_config(
        {"device": "ascend", "device_count": 2},
        role,
    )

    extra = config["kv_connector_extra_config"]
    expected = {"tp_size": 2, "dp_size": 1, "pp_size": 1}
    assert extra["prefill"] == expected
    assert extra["decode"] == expected


@pytest.mark.parametrize("role", ("P", "D"))
@pytest.mark.parametrize("device_count", (1, 2, 8))
@pytest.mark.parametrize(
    ("device", "engine"),
    (("nvidia", "vllm"), ("ascend", "vllm_ascend")),
)
def test_pd_1p1d_without_any_tp_dp_keeps_standalone_recipe_and_local_tp(
    monkeypatch,
    tmp_path,
    role,
    device_count,
    device,
    engine,
):
    for name in _CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PD_ROLE", role)
    architecture = (
        "Qwen3_5MoeForConditionalGeneration"
        if device == "nvidia"
        else "UnknownArchitecture"
    )
    model_dir = tmp_path / "qwen35-moe"
    _write_arch_config(model_dir, architecture)
    model_info = SimpleNamespace(model_architecture=architecture)
    base_kv = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
    params = {
        "engine": engine,
        "device_count": device_count,
        "distributed": False,
        "nnodes": 1,
        "node_rank": 0,
        "model_name": "test-model",
        "model_path": str(model_dir),
        "model_type": "llm",
        "engine_config": {
            "use_vllm_serve": True,
            "model": str(model_dir),
            "port": 17000,
            "kv_transfer_config": json.dumps(base_kv),
        },
    }

    config_loader._apply_pd_external_lb(params, model_info, {"device": device})

    assert params["engine_config"]["tensor_parallel_size"] == device_count
    assert params["engine_config"]["data_parallel_size"] == 1
    assert params["_pd_engine_overrides"]["tensor_parallel_size"] == device_count
    assert params["_pd_engine_overrides"]["data_parallel_size"] == 1
    assert "_pd_external_lb" not in params
    assert json.loads(params["engine_config"]["kv_transfer_config"]) == base_kv
    script = vllm_adapter.build_start_script(params)
    exec_line = next(
        line for line in script.splitlines()
        if "vllm serve" in line and not line.lstrip().startswith("echo ")
    )
    assert f"--tensor-parallel-size {device_count}" in exec_line
    assert "--data-parallel-size 1" in exec_line
    assert "Mooncake" not in exec_line

    if device == "nvidia":
        assert "export VLLM_SSM_CONV_STATE_LAYOUT=DS" in script
        assert "ASCEND_RT_VISIBLE_DEVICES" not in exec_line
    else:
        assert "VLLM_SSM_CONV_STATE_LAYOUT" not in script
        assert "--data-parallel-external-lb" not in exec_line


@pytest.mark.parametrize(
    "architecture",
    ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"),
)
@pytest.mark.parametrize("role", ("P", "D"))
def test_nvidia_qwen35_pd_sets_ssm_conv_state_layout(
    tmp_path, monkeypatch, architecture, role
):
    model_dir = tmp_path / architecture
    _write_arch_config(model_dir, architecture)
    monkeypatch.setenv("PD_ROLE", role)

    commands = vllm_adapter._build_model_env_commands(
        {
            "model_name": "qwen3.5",
            "model_path": str(model_dir),
            "model_type": "llm",
        },
        "vllm",
    )

    assert "export VLLM_SSM_CONV_STATE_LAYOUT=DS" in commands


def test_nvidia_ssm_conv_state_layout_is_model_and_pd_scoped(tmp_path, monkeypatch):
    model_dir = tmp_path / "qwen35-moe"
    _write_arch_config(model_dir, "Qwen3_5MoeForConditionalGeneration")
    params = {
        "model_name": "qwen3.5-moe",
        "model_path": str(model_dir),
        "model_type": "llm",
    }
    monkeypatch.delenv("PD_ROLE", raising=False)
    assert "export VLLM_SSM_CONV_STATE_LAYOUT=DS" not in (
        vllm_adapter._build_model_env_commands(params, "vllm")
    )

    monkeypatch.setenv("PD_ROLE", "P")
    assert "export VLLM_SSM_CONV_STATE_LAYOUT=DS" not in (
        vllm_adapter._build_model_env_commands(params, "vllm_ascend")
    )

    other_dir = tmp_path / "other-model"
    _write_arch_config(other_dir, "UnknownArchitecture")
    params["model_path"] = str(other_dir)

    assert "export VLLM_SSM_CONV_STATE_LAYOUT=DS" not in (
        vllm_adapter._build_model_env_commands(params, "vllm")
    )


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


def _render_pd_deepseek_v4_script(
    tmp_path,
    monkeypatch,
    role: str,
    local_ip: str,
    pd_index: int,
    *,
    platform: str = "a3",
    prefill_dp: int = 2,
    prefill_tp: int = 4,
    decode_dp: int = 8,
    decode_tp: int = 1,
    dp_size_local: int | None = None,
    device_count: int = 8,
    hardware_name: str = "Ascend910C",
) -> str:
    for name in _CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setenv("PD_ROLE", role)
    monkeypatch.setenv("PD_INDEX", str(pd_index))
    monkeypatch.setenv("PD_PREFILL_DP_SIZE", str(prefill_dp))
    monkeypatch.setenv("PD_PREFILL_TP_SIZE", str(prefill_tp))
    monkeypatch.setenv("PD_DECODE_DP_SIZE", str(decode_dp))
    monkeypatch.setenv("PD_DECODE_TP_SIZE", str(decode_tp))
    if dp_size_local is None:
        dp_size_local = 1 if role == "P" else decode_dp
    monkeypatch.setenv("DP_SIZE_LOCAL", str(dp_size_local))
    monkeypatch.setenv("MASTER_IP", local_ip)
    monkeypatch.setenv("RANK_IP", local_ip)
    monkeypatch.setenv("HOST_IP", local_ip)
    monkeypatch.setenv("POD_IP", local_ip)
    monkeypatch.setenv("NODE_IPS", local_ip)
    monkeypatch.setenv("NETWORK_INTERFACE", "xxxx")

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
            str(device_count),
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
        {
            "device": "ascend",
            "count": device_count,
            "details": [{"name": hardware_name}] * device_count,
        },
    )
    return start_engine_service(merged)


def _extract_vllm_exec_line(script: str) -> str:
    return next(
        line.strip()
        for line in script.splitlines()
        if "vllm serve" in line and not line.lstrip().startswith("echo ")
    )


@pytest.mark.parametrize("role", ("P", "D"))
def test_pd_large_ep_does_not_load_ascend_defaults(tmp_path, monkeypatch, role):
    monkeypatch.setattr(
        config_loader, "_load_default_config",
        lambda *_args, **_kwargs: pytest.fail("PD large-EP loaded ascend_default.json"),
    )
    script = _render_pd_deepseek_v4_script(
        tmp_path, monkeypatch, role,
        _PREFILL_IP if role == "P" else _DECODE_IP, 0,
    )
    assert "--data-parallel-external-lb" in _extract_vllm_exec_line(script)


def test_pd_large_ep_rejects_unregistered_architecture_without_default_fallback(monkeypatch):
    for name in _CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PD_ROLE", "P")
    monkeypatch.setenv("PD_PREFILL_DP_SIZE", "2")
    monkeypatch.setenv("PD_PREFILL_TP_SIZE", "4")

    params = {
        "engine": "vllm_ascend",
        "device_count": 8,
        "engine_config": {},
    }
    model_info = SimpleNamespace(model_architecture="UnknownArchitecture")

    with pytest.raises(ValueError, match="no registered profile"):
        config_loader._apply_pd_external_lb(params, model_info, {"device": "ascend"})


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
    assert exports["HCCL_INTER_HCCS_DISABLE"] == "export HCCL_INTER_HCCS_DISABLE=true"
    assert exports["HCCL_INTRA_ROCE_ENABLE"] == "export HCCL_INTRA_ROCE_ENABLE=1"

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


def _assert_a3_only_deepseek_v4_pd_fields(
    exports: dict[str, str], script: str, pd_index: int
) -> None:
    """A3 独占 MC2/SuperPod 配置，A2 不得从公共层继承。"""
    assert exports["VLLM_ASCEND_ENABLE_FUSED_MC2"] == (
        "export VLLM_ASCEND_ENABLE_FUSED_MC2=0"
    )
    assert exports["HCCL_LOGIC_SUPERPOD_ID"] == (
        f"export HCCL_LOGIC_SUPERPOD_ID={pd_index}"
    )
    assert '"enable_mc2_hierarchy_comm":true' in script


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
    _assert_a3_only_deepseek_v4_pd_fields(exports, script, 0)
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
    _assert_a3_only_deepseek_v4_pd_fields(exports, script, 1)
    assert "VLLM_ASCEND_ENABLE_MLAPO" not in exports
    assert "VLLM_MOONCAKE_BOOTSTRAP_PORT=" not in script


@pytest.mark.parametrize(
    (
        "platform",
        "role",
        "pd_index",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "dp_size",
        "tp_size",
        "dp_size_local",
        "connect_timeout",
        "hccl_buffsize",
        "prefix_flag",
        "has_flashcomm",
        "has_dsa_cp",
    ),
    (
        ("a3", "P", 0, 1048576, 8192, 16, 4, 4, 4, 120, 2560,
         "--no-enable-prefix-caching", True, True),
        ("a3", "D", 1, 1048576, 120, 60, 16, 1, 16, 1200, 1024,
         "--no-enable-prefix-caching", False, False),
        ("a2", "P", 0, 135000, 4096, 16, 8, 1, 8, 1200, 1024,
         "--enable-prefix-caching", False, False),
        ("a2", "D", 4, 135000, 60, 30, 32, 1, 8, 1200, 1024,
         "--no-enable-prefix-caching", False, False),
    ),
)
def test_deepseek_v4_pd_final_command_matches_v023_profile_with_no_async_override(
    tmp_path,
    monkeypatch,
    platform,
    role,
    pd_index,
    max_model_len,
    max_num_batched_tokens,
    max_num_seqs,
    dp_size,
    tp_size,
    dp_size_local,
    connect_timeout,
    hccl_buffsize,
    prefix_flag,
    has_flashcomm,
    has_dsa_cp,
):
    # 官方 A3 是 P=DP4/TP4、D=DP16/TP1；A2 是 P=DP8/TP1、D=DP32/TP1。
    prefill_dp, prefill_tp = ((4, 4) if platform == "a3" else (8, 1))
    decode_dp, decode_tp = ((16, 1) if platform == "a3" else (32, 1))
    device_count = 16 if platform == "a3" else 8
    local_ip = _PREFILL_IP if role == "P" else _DECODE_IP

    script = _render_pd_deepseek_v4_script(
        tmp_path,
        monkeypatch,
        role,
        local_ip,
        pd_index,
        platform=platform,
        prefill_dp=prefill_dp,
        prefill_tp=prefill_tp,
        decode_dp=decode_dp,
        decode_tp=decode_tp,
        dp_size_local=dp_size_local,
        device_count=device_count,
        hardware_name="Ascend910C" if platform == "a3" else "Ascend910B_64G",
    )
    exports = _extract_exports(script)
    exec_line = _extract_vllm_exec_line(script)
    compact = re.sub(r"\s+", "", script)

    assert f"--max-model-len {max_model_len}" in exec_line
    assert f"--max-num-batched-tokens {max_num_batched_tokens}" in exec_line
    assert f"--max-num-seqs {max_num_seqs}" in exec_line
    assert f"--tensor-parallel-size {tp_size} --data-parallel-size {dp_size}" in exec_line
    assert f"for i in $(seq 0 {dp_size_local - 1}); do" in script
    assert f'"prefill":{{"dp_size":{prefill_dp},"tp_size":{prefill_tp}' in compact
    assert f'"decode":{{"dp_size":{decode_dp},"tp_size":{decode_tp}' in compact
    assert prefix_flag in exec_line
    assert exports["HCCL_CONNECT_TIMEOUT"] == (
        f"export HCCL_CONNECT_TIMEOUT={connect_timeout}"
    )
    assert exports["HCCL_BUFFSIZE"] == f"export HCCL_BUFFSIZE={hccl_buffsize}"
    assert ("VLLM_ASCEND_ENABLE_FLASHCOMM1" in exports) is has_flashcomm
    assert ('"enable_dsa_cp":true' in exec_line) is has_dsa_cp
    has_a3_mc2 = platform == "a3"
    assert ("VLLM_ASCEND_ENABLE_FUSED_MC2" in exports) is has_a3_mc2
    assert ("HCCL_LOGIC_SUPERPOD_ID" in exports) is has_a3_mc2
    assert ('"enable_mc2_hierarchy_comm":true' in exec_line) is has_a3_mc2
    if has_a3_mc2:
        _assert_a3_only_deepseek_v4_pd_fields(exports, exec_line, pd_index)

    if role == "P":
        assert "--enforce-eager" in exec_line
    else:
        assert "--enforce-eager" not in exec_line
    assert "--async-scheduling" not in exec_line
    assert "--no-async-scheduling" in exec_line
    if platform == "a2" and role == "P":
        assert "--no-enable-prefix-caching" not in exec_line


@pytest.mark.parametrize(
    ("env_name", "env_value", "hardware_name", "expected"),
    (
        ("WINGS_ASCEND_PLATFORM", "a3", "Ascend910B_64G", "a2"),
        ("ASCEND_PLATFORM", "a3", "Ascend910B_64G", "a2"),
        ("ENGINE_IMAGE_FLAVOR", "a3", "Ascend910B_64G", "a2"),
        ("ENGINE_VERSION", "0.23.0-a3", "Ascend910B_64G", "a2"),
        ("ASCEND_A3_ENABLE", "1", "Ascend910B_64G", "a2"),
        ("WINGS_ASCEND_PLATFORM", "a2", "Ascend910C", "a3"),
    ),
)
def test_pd_platform_uses_only_hardware_info(
    monkeypatch, env_name, env_value, hardware_name, expected
):
    for name in _CLEAR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, env_value)
    assert config_loader._resolve_ascend_platform(
        {"device": "ascend", "details": [{"name": hardware_name}]}
    ) == expected


def test_pd_platform_falls_back_to_hardware_family_for_generic_detail():
    assert config_loader._resolve_ascend_platform(
        {
            "device": "ascend",
            "details": [{"name": "Ascend"}],
            "hardware_family": "Ascend910C",
        }
    ) == "a3"


def test_pd_large_ep_rejects_unknown_hardware_platform(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="cannot resolve Ascend platform"):
        _render_pd_deepseek_v4_script(
            tmp_path,
            monkeypatch,
            "P",
            _PREFILL_IP,
            0,
            hardware_name="Ascend",
        )


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
