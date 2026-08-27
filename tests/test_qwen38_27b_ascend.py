import json
import shlex
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from core import wings_entry  # noqa: E402
from core.port_plan import derive_port_plan  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines import vllm_adapter  # noqa: E402
from utils import model_utils  # noqa: E402


_MODEL_NAME = "Qwen3.8-27B-w8a8"
_MODEL_PATH = "/var/aispace/model/ai-storage/ai-prod/platform/Qwen3.8-27B-w8a8"
_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_RUNTIME_ENV_NAMES = (
    "PD_ROLE",
    "ENGINE",
    "WINGS_ENGINE",
    "ENGINE_PORT",
    "PROXY_PORT",
    "PORT",
    "POD_IP",
    "RANK_IP",
    "SERVED_MODEL_NAME",
    "DISTRIBUTED_EXECUTOR_BACKEND",
    "CONFIG_FORCE",
    "ENABLE_AUTO_THINK_CHOICE",
    "ENABLE_SPECULATIVE_DECODE",
    "SD_ENABLE",
    "TENSOR_PARALLEL_SIZE",
    "DATA_PARALLEL_SIZE",
    "ENABLE_SPARSE",
    "ENABLE_KV_OFFLOAD",
    "LMCACHE_OFFLOAD",
    "ENABLE_KV_MEM_OFFLOAD",
    "KV_MEM_OFFLOAD_SIZE",
    "ENGINE_VERSION",
    "ASCEND_ENFORCE_EAGER",
    "ASCEND_RT_VISIBLE_DEVICES",
    "VLLM_USE_MODELSCOPE",
    "PYTORCH_NPU_ALLOC_CONF",
    "TASK_QUEUE_ENABLE",
    "HCCL_BUFFSIZE",
    "HCCL_OP_EXPANSION_MODE",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "VLLM_USE_SIMPLE_KV_OFFLOAD",
)


def _clear_runtime_env(monkeypatch):
    for env_name in _RUNTIME_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)


class _FakeQwen38Identifier:
    config = {}
    model_architecture = _ARCHITECTURE
    model_quantize = "w8a8"

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

    @staticmethod
    def is_wings_supported():
        return True


def _ascend_arch_defaults():
    config_path = Path(config_loader.DEFAULT_CONFIG_DIR) / "ascend_default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["model_deploy_config"]["llm"][_ARCHITECTURE]


def _match_defaults(
    model_name: str,
    card_name: str,
    engine_key: str = "vllm_ascend",
    device_count: int = 2,
):
    return config_loader._match_model_engine_config(
        _ascend_arch_defaults(),
        model_name.lower(),
        engine_key,
        config_loader._SpecialEngineScenario(),
        _FakeQwen38Identifier(model_name, _MODEL_PATH, "llm"),
        {
            "device": "ascend",
            "count": device_count,
            "details": [{"name": card_name}],
        },
        _MODEL_PATH.lower(),
    )


def test_qwen38_27b_w8a8_910b_selects_exact_single_node_profile():
    config = _match_defaults(_MODEL_NAME, "Ascend910B")

    assert config == {
        "use_vllm_serve": True,
        "trust_remote_code": True,
        "quantization": "ascend",
        "max_num_seqs": 64,
        "max_model_len": 131072,
        "max_num_batched_tokens": 16384,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": True,
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64],
        },
        "additional_config": {"enable_cpu_binding": True},
    }
    # TP/DP 是运行时拓扑，不能固化在模型 defaults 中。
    assert "tensor_parallel_size" not in config
    assert "data_parallel_size" not in config
    # 投机能力由精确白名单和页面开关控制，不能在模型 defaults 中强开。
    assert "speculative_config" not in config


def test_qwen38_27b_w8a8_910c_selects_exact_profile():
    config = _match_defaults(_MODEL_NAME, "Ascend910C")

    assert config == {
        "use_vllm_serve": True,
        "trust_remote_code": True,
        "quantization": "ascend",
        "max_num_seqs": 64,
        "max_model_len": 133072,
        "max_num_batched_tokens": 16384,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": True,
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64],
        },
        "additional_config": {"enable_cpu_binding": True},
    }
    # TP/DP 属于运行时拓扑，不固化到模型 defaults；前缀缓存仍是静态配方。
    assert "tensor_parallel_size" not in config
    assert "data_parallel_size" not in config
    assert config["enable_prefix_caching"] is True
    # 投机能力仍由精确白名单和页面开关控制，不在模型 defaults 中强开。
    assert "speculative_config" not in config

    profile = _ascend_arch_defaults()["Qwen3.8-27B-w8a8-910C"]
    assert "Qwen3.8-27B-w8a8-910C-single-node" not in _ascend_arch_defaults()
    assert "device_counts" not in profile


@pytest.mark.parametrize(
    ("model_name", "card_name", "engine_key"),
    [
        ("Qwen3.8-27B", "Ascend910B", "vllm_ascend"),
        ("Qwen3.8-27B", "Ascend910C", "vllm_ascend"),
        (_MODEL_NAME, "Ascend310P", "vllm_ascend"),
        (_MODEL_NAME, "Ascend910B", "vllm_ascend_distributed"),
        (_MODEL_NAME, "Ascend910C", "vllm_ascend_distributed"),
    ],
)
def test_qwen38_27b_profile_does_not_broaden(model_name, card_name, engine_key):
    assert _match_defaults(model_name, card_name, engine_key) == {}


@pytest.mark.parametrize("device_count", [1, 2, 4, 8])
def test_qwen38_27b_910c_profile_does_not_restrict_local_device_count(
    device_count,
):
    assert _match_defaults(
        _MODEL_NAME,
        "Ascend910C",
        device_count=device_count,
    ) == _match_defaults(_MODEL_NAME, "Ascend910C", device_count=2)


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_qwen38_27b_ascend_spec_whitelist_renders_exact_mtp(
    monkeypatch,
    card_token,
):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        _MODEL_NAME,
        _MODEL_PATH,
        card_token,
        "spec",
    )

    assert row is not None
    assert row["exact_model_names"] == ("qwen3.8-27b-w8a8",)
    assert row["mtp_method"] == "qwen3_5_mtp"
    assert row["mtp_num_speculative_tokens"] == 2
    assert row["enforce_eager"] is True
    assert model_utils.resolve_feature_whitelist_row(
        "vllm_ascend", "Qwen3.8-27B", "/models/Qwen3.8-27B", card_token, "spec"
    ) is None

    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    command = vllm_adapter.build_speculative_cmd(
        {
            "engine": "vllm_ascend",
            "model_name": _MODEL_NAME,
            "model_path": _MODEL_PATH,
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec"],
            "_smart_card_token": card_token,
        },
        "vllm_ascend",
    )

    assert command == (
        " --speculative-config "
        "'{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":2,\"enforce_eager\":true}'"
    )


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_qwen38_27b_ascend_native_offload_whitelist_is_exact(card_token):
    row = model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        _MODEL_NAME,
        _MODEL_PATH,
        card_token,
        "offload",
    )

    assert row is not None
    assert row["exact_model_names"] == ("qwen3.8-27b-w8a8",)
    assert row["backend"] == "native"
    assert row["source"] == "vllm-ascend-0.23"
    assert model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        "Qwen3.8-27B",
        "/models/Qwen3.8-27B",
        card_token,
        "offload",
    ) is None


def test_exact_model_names_remains_optional_for_existing_rows():
    row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        "GLM-5-FP8",
        "/models/GLM-5-FP8",
        "h20-141",
        "spec",
    )

    assert row is not None
    assert "exact_model_names" not in row


@pytest.mark.parametrize("feature", ["spec", "offload"])
@pytest.mark.parametrize("card_token", ["910b", "910c"])
@pytest.mark.parametrize(
    ("model_name", "model_path"),
    [
        ("Qwen3.8-27B-w8a8-custom", "/models/Qwen3.8-27B-w8a8-custom"),
        ("Other-27B", "/models/archive/Qwen3.8-27B-w8a8"),
    ],
)
def test_qwen38_27b_smart_feature_rows_reject_alias_and_path_substrings(
    feature,
    card_token,
    model_name,
    model_path,
):
    assert model_utils.resolve_feature_whitelist_row(
        "vllm_ascend",
        model_name,
        model_path,
        card_token,
        feature,
    ) is None


def test_qwen38_27b_alias_offload_request_is_suppressed_end_to_end(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("ENGINE_VERSION", "v0.23.0-a3")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "100")
    params = {
        "engine": "vllm_ascend",
        "model_name": "Qwen3.8-27B-w8a8-custom",
        "model_path": "/models/Qwen3.8-27B-w8a8-custom",
        "model_type": "llm",
        "_resolved_model_type": "llm",
        "device_count": 2,
        "nnodes": 1,
        "distributed": False,
    }

    config_loader.apply_effective_feature_enablement(
        params,
        {
            "device": "ascend",
            "count": 2,
            "details": [{"name": "Ascend910C"}],
        },
    )

    assert params["_smart_feats"] == []
    assert vllm_adapter.build_kv_offload_cmd(params, "vllm_ascend") == ""
    assert vllm_adapter.resolve_kv_offload_effective_state(
        params,
        "vllm_ascend",
    )[0] is False


@pytest.mark.parametrize(
    ("card_token", "engine_version"),
    [("ascend910b", "v0.23.0"), ("ascend910c", "v0.23.0-a3")],
)
def test_qwen38_27b_ascend_native_offload_renders_for_023_golden_image(
    monkeypatch,
    card_token,
    engine_version,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("ENGINE_VERSION", engine_version)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "100")
    params = {
        "engine": "vllm_ascend",
        "model_name": _MODEL_NAME,
        "model_path": _MODEL_PATH,
        "model_type": "llm",
        "device_count": 2,
        "nnodes": 1,
        "distributed": False,
        "_smart_card_token": card_token,
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter.build_kv_offload_cmd(params, "vllm_ascend") == (
        " --kv-offloading-backend native --kv-offloading-size 100"
    )
    assert vllm_adapter.resolve_kv_offload_effective_state(
        params,
        "vllm_ascend",
    ) == (True, "native_kv_offloading_backend")
    aligned = vllm_adapter._align_qwen38_27b_w8a8_910c_env(
        ["export VLLM_USE_SIMPLE_KV_OFFLOAD=0"],
        params,
        "vllm_ascend",
    )
    assert aligned == ["export VLLM_USE_SIMPLE_KV_OFFLOAD=1"]


@pytest.mark.parametrize(
    ("card_token", "engine_version", "mem_enabled", "size"),
    [
        ("ascend910b", "v0.22.0", "true", "100"),
        ("ascend910b", "v0.23.0-a3", "true", "100"),
        ("ascend910c", "v0.23.0", "true", "100"),
        ("ascend910c", "v0.24.0-a3", "true", "100"),
        ("ascend910c", "v0.23.0-a3", "false", "100"),
        ("ascend910c", "v0.23.0-a3", "true", "0"),
    ],
)
def test_qwen38_27b_ascend_native_offload_rejects_wrong_image_or_size(
    monkeypatch,
    card_token,
    engine_version,
    mem_enabled,
    size,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("ENGINE_VERSION", engine_version)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", mem_enabled)
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", size)
    params = {
        "engine": "vllm_ascend",
        "model_name": _MODEL_NAME,
        "model_path": _MODEL_PATH,
        "model_type": "llm",
        "device_count": 2,
        "nnodes": 1,
        "distributed": False,
        "_smart_card_token": card_token,
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter.build_kv_offload_cmd(params, "vllm_ascend") == ""
    assert vllm_adapter.resolve_kv_offload_effective_state(
        params,
        "vllm_ascend",
    )[0] is False
    aligned = vllm_adapter._align_qwen38_27b_w8a8_910c_env(
        ["export VLLM_USE_SIMPLE_KV_OFFLOAD=0"],
        params,
        "vllm_ascend",
    )
    assert "export VLLM_USE_SIMPLE_KV_OFFLOAD=1" not in aligned


def test_qwen38_27b_910b_uses_dedicated_minimal_env(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    commands = vllm_adapter._build_model_env_commands(
        {
            "engine": "vllm_ascend",
            "model_name": _MODEL_NAME,
            "model_path": _MODEL_PATH,
            "model_type": "llm",
            "device_count": 2,
            "nnodes": 1,
            "distributed": False,
            "_smart_card_token": "ascend910b",
        },
        "vllm_ascend",
    )

    assert commands == [
        "export VLLM_USE_MODELSCOPE=True",
        "export HCCL_BUFFSIZE=512",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
    ]


@pytest.mark.parametrize("device_count", [1, 2, 4, 8])
def test_qwen38_27b_910c_uses_dedicated_minimal_env(monkeypatch, device_count):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    commands = vllm_adapter._build_model_env_commands(
        {
            "engine": "vllm_ascend",
            "model_name": _MODEL_NAME,
            "model_path": "/usr/local/serving/models",
            "model_type": "llm",
            "device_count": device_count,
            "nnodes": 1,
            "distributed": False,
            "_smart_card_token": "ascend910c",
        },
        "vllm_ascend",
    )

    assert commands == [
        "export VLLM_USE_MODELSCOPE=True",
        "export HCCL_BUFFSIZE=512",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_name": "Qwen3.8-27B"},
        {"_smart_card_token": "ascend910c"},
        {"device_count": 4},
        {"distributed": True, "nnodes": 2},
    ],
)
def test_qwen38_27b_env_scope_does_not_broaden(overrides):
    params = {
        "model_name": _MODEL_NAME,
        "device_count": 2,
        "nnodes": 1,
        "distributed": False,
        "_smart_card_token": "ascend910b",
    }
    params.update(overrides)

    assert not vllm_adapter._is_qwen38_27b_w8a8_910b_single_node_env_scope(
        params,
        _FakeQwen38Identifier(_MODEL_NAME, _MODEL_PATH, "llm"),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_name": "Qwen3.8-27B"},
        {"_smart_card_token": "ascend910b"},
        {"distributed": True, "nnodes": 2},
    ],
)
def test_qwen38_27b_910c_local_scope_does_not_broaden(overrides):
    params = {
        "engine": "vllm_ascend",
        "model_name": _MODEL_NAME,
        "device_count": 2,
        "nnodes": 1,
        "distributed": False,
        "_smart_card_token": "ascend910c",
    }
    params.update(overrides)

    assert not vllm_adapter._is_qwen38_27b_w8a8_910c_local_scope(
        params,
        _FakeQwen38Identifier(_MODEL_NAME, "/usr/local/serving/models", "llm"),
    )


@pytest.mark.parametrize("device_count", [1, 2, 4, 8])
def test_qwen38_27b_910c_local_scope_accepts_visible_device_count(device_count):
    params = {
        "engine": "vllm_ascend",
        "model_name": _MODEL_NAME,
        "device_count": device_count,
        "nnodes": 1,
        "distributed": False,
        "_smart_card_token": "ascend910c",
    }

    assert vllm_adapter._is_qwen38_27b_w8a8_910c_local_scope(
        params,
        _FakeQwen38Identifier(_MODEL_NAME, "/usr/local/serving/models", "llm"),
    )


@pytest.mark.parametrize("device_count", [1, 2, 4, 8])
def test_qwen38_27b_910c_final_command_derives_tp_from_local_devices(
    monkeypatch,
    device_count,
):
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "_record_selected_engine", lambda *_args: None)
    _clear_runtime_env(monkeypatch)

    launch_args = parse_launch_args(
        [
            "--model-name",
            _MODEL_NAME,
            "--model-path",
            "/usr/local/serving/models",
            "--model-type",
            "llm",
            "--engine",
            "vllm_ascend",
            "--device-count",
            str(device_count),
            "--host",
            "0.0.0.0",
            "--port",
            "18000",
        ]
    )
    params = config_loader.load_and_merge_configs(
        {
            "device": "ascend",
            "count": device_count,
            "details": [{"name": "Ascend910C"}],
        },
        launch_args,
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    shlex.split(exec_line, posix=True)

    assert params["engine_config"]["tensor_parallel_size"] == device_count
    assert "data_parallel_size" not in params["engine_config"]
    assert "distributed_executor_backend" not in params["engine_config"]
    assert "--distributed-executor-backend" not in exec_line
    assert f"--tensor-parallel-size {device_count}" in exec_line
    assert "--data-parallel-size" not in exec_line
    assert "export VLLM_USE_MODELSCOPE=True" in script
    assert "export HCCL_BUFFSIZE=512" in script
    assert "export OMP_PROC_BIND=" not in script
    assert "export OMP_NUM_THREADS=" not in script
    assert "export TASK_QUEUE_ENABLE=" not in script
    assert "export HCCL_OP_EXPANSION_MODE=" not in script
    assert "export VLLM_USE_SIMPLE_KV_OFFLOAD=" not in script
    assert "--kv-offloading-backend" not in exec_line
    assert "--kv-offloading-size" not in exec_line


@pytest.mark.parametrize(
    "overrides",
    [
        {"_smart_card_token": "ascend910b", "device_count": 4},
        {"distributed": True, "nnodes": 2},
        {"model_name": "Qwen3.8-27B"},
        {"engine": "vllm"},
    ],
)
def test_qwen38_27b_910c_env_alignment_does_not_broaden(monkeypatch, overrides):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    params = {
        "engine": "vllm_ascend",
        "model_name": _MODEL_NAME,
        "model_path": "/usr/local/serving/models",
        "model_type": "llm",
        "device_count": 2,
        "nnodes": 1,
        "distributed": False,
        "_smart_card_token": "ascend910c",
    }
    params.update(overrides)
    commands = [
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=1",
        "export TASK_QUEUE_ENABLE=1",
        "export HCCL_OP_EXPANSION_MODE=AIV",
    ]

    assert vllm_adapter._align_qwen38_27b_w8a8_910c_env(
        commands, params, params["engine"]
    ) == commands


def test_qwen38_27b_production_chain_renders_target_single_node_command(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "_record_selected_engine", lambda *_args: None)
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("SERVED_MODEL_NAME", "qwen3.8")
    monkeypatch.setenv("HCCL_BUFFSIZE", "512")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("ENGINE_PORT", "8000")
    monkeypatch.setenv("POD_IP", "10.0.0.8")
    monkeypatch.setenv("ENGINE_VERSION", "v0.23.0")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "100")

    # DP=1 由本次部署显式声明；模型 defaults 只保存静态能力参数。
    config_file = tmp_path / "qwen38-27b-runtime.json"
    config_file.write_text(json.dumps({"data_parallel_size": 1}), encoding="utf-8")
    launch_args = parse_launch_args(
        [
            "--model-name",
            _MODEL_NAME,
            "--model-path",
            _MODEL_PATH,
            "--model-type",
            "llm",
            "--engine",
            "vllm_ascend",
            "--device-count",
            "2",
            "--host",
            "0.0.0.0",
            "--port",
            "18000",
            "--config-file",
            str(config_file),
            "--enable-speculative-decode",
            "--speculative-decode-model-path",
            "none",
        ]
    )
    port_plan = derive_port_plan(
        port=launch_args.port,
        enable_reason_proxy=True,
    )
    params = wings_entry._prepare_merged_params(
        launch_args,
        port_plan,
        {
            "device": "ascend",
            "count": 2,
            "details": [{"name": "Ascend910B"}],
        },
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    shlex.split(exec_line, posix=True)

    assert params["_smart_card_token"].endswith("910b")
    assert params["_smart_feats"] == ["offload", "spec"]
    assert params["engine_config"]["tensor_parallel_size"] == 2
    assert params["engine_config"]["data_parallel_size"] == 1
    # v4 的 --port/PORT 是代理端口；真正的 vLLM 监听端口由 ENGINE_PORT 控制。
    assert port_plan.proxy_port == 18000
    assert port_plan.backend_port == 8000
    assert exec_line == (
        f"exec vllm serve {_MODEL_PATH}"
        " --trust-remote-code"
        " --quantization ascend"
        " --max-num-seqs 64"
        " --max-model-len 131072"
        " --max-num-batched-tokens 16384"
        " --gpu-memory-utilization 0.9"
        " --enable-prefix-caching"
        " --compilation-config "
        "'{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2,4,8,16,24,32,40,48,56,64]}'"
        " --additional-config '{\"enable_cpu_binding\":true}'"
        " --host 10.0.0.8"
        " --port 8000"
        " --served-model-name qwen3.8"
        " --default-chat-template-kwargs '{\"enable_thinking\":false}'"
        " --tensor-parallel-size 2"
        " --data-parallel-size 1"
        " --speculative-config "
        "'{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":2,\"enforce_eager\":true}'"
        " --kv-offloading-backend native"
        " --kv-offloading-size 100"
    )
    # 可见卡由部署层传入；启动脚本不得把 2,3 改写成逻辑卡号或其它物理卡号。
    assert "ASCEND_RT_VISIBLE_DEVICES=" not in script
    assert [line for line in script.splitlines() if line.startswith("export ")] == [
        "export VLLM_USE_MODELSCOPE=True",
        "export HCCL_BUFFSIZE=512",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export VLLM_USE_SIMPLE_KV_OFFLOAD=1",
    ]


def test_qwen38_27b_910c_production_chain_renders_target_command(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen38Identifier)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "_record_selected_engine", lambda *_args: None)
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    # 该画像承载的是目标启动配方，外层遗留值不能让最终脚本漂移。
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "False")
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "max_split_size_mb:64")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    monkeypatch.setenv("HCCL_BUFFSIZE", "1024")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "NONE")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("OMP_PROC_BIND", "true")
    monkeypatch.setenv("SERVED_MODEL_NAME", "qwen3.8")
    monkeypatch.setenv("ENGINE_PORT", "8000")
    monkeypatch.setenv("POD_IP", "10.0.0.9")
    monkeypatch.setenv("ENGINE_VERSION", "v0.23.0-a3")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "100")

    # DP=1 由本次部署显式声明；模型 defaults 不持有运行时拓扑。
    config_file = tmp_path / "qwen38-27b-910c-runtime.json"
    config_file.write_text(json.dumps({"data_parallel_size": 1}), encoding="utf-8")
    launch_args = parse_launch_args(
        [
            "--model-name",
            _MODEL_NAME,
            "--model-path",
            "/usr/local/serving/models",
            "--model-type",
            "llm",
            "--engine",
            "vllm_ascend",
            "--device-count",
            "2",
            "--host",
            "0.0.0.0",
            "--port",
            "18000",
            "--config-file",
            str(config_file),
            "--enable-speculative-decode",
            "--speculative-decode-model-path",
            "none",
        ]
    )
    port_plan = derive_port_plan(
        port=launch_args.port,
        enable_reason_proxy=True,
    )
    params = wings_entry._prepare_merged_params(
        launch_args,
        port_plan,
        {
            "device": "ascend",
            "count": 2,
            "details": [{"name": "Ascend910C"}],
        },
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    shlex.split(exec_line, posix=True)

    assert params["_smart_card_token"].endswith("910c")
    assert params["_smart_feats"] == ["offload", "spec"]
    assert "distributed_executor_backend" not in params["engine_config"]
    assert params["engine_config"]["tensor_parallel_size"] == 2
    assert params["engine_config"]["data_parallel_size"] == 1
    assert port_plan.proxy_port == 18000
    assert port_plan.backend_port == 8000
    assert exec_line == (
        "exec vllm serve /usr/local/serving/models"
        " --trust-remote-code"
        " --quantization ascend"
        " --max-num-seqs 64"
        " --max-model-len 133072"
        " --max-num-batched-tokens 16384"
        " --gpu-memory-utilization 0.9"
        " --enable-prefix-caching"
        " --compilation-config "
        "'{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2,4,8,16,24,32,40,48,56,64]}'"
        " --additional-config '{\"enable_cpu_binding\":true}'"
        " --host 10.0.0.9"
        " --port 8000"
        " --served-model-name qwen3.8"
        " --default-chat-template-kwargs '{\"enable_thinking\":false}'"
        " --tensor-parallel-size 2"
        " --data-parallel-size 1"
        " --speculative-config "
        "'{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":2,\"enforce_eager\":true}'"
        " --kv-offloading-backend native"
        " --kv-offloading-size 100"
    )
    assert "--distributed-executor-backend" not in exec_line
    assert "--async-scheduling" not in exec_line
    assert "ASCEND_RT_VISIBLE_DEVICES=" not in script
    assert [line for line in script.splitlines() if line.startswith("export ")] == [
        "export VLLM_USE_MODELSCOPE=True",
        "export HCCL_BUFFSIZE=512",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export VLLM_USE_SIMPLE_KV_OFFLOAD=1",
    ]
