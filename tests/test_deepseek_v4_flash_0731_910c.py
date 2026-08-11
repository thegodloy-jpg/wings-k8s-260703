import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines import vllm_adapter  # noqa: E402
from utils import model_utils  # noqa: E402


MODEL_NAME = "DeepSeek-V4-Flash-0731-w8a8"
MODEL_PATH = f"/root/.cache/modelscope/hub/models/vllm-ascend/{MODEL_NAME}"
LEGACY_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp"
)


class _FakeDeepSeekV4Info:
    config = {}
    model_architecture = "DeepseekV4ForCausalLM"
    model_quantize = "w8a8"

    def __init__(self, model_name=MODEL_NAME, model_path=MODEL_PATH, model_type="llm"):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type

    @staticmethod
    def identify_model_architecture():
        return "DeepseekV4ForCausalLM"

    @staticmethod
    def identify_model_type():
        return "llm"

    @staticmethod
    def is_wings_supported():
        return True


def _ascend_defaults():
    path = (
        Path(config_loader.DEFAULT_CONFIG_DIR)
        / config_loader.DEFAULT_CONFIG_FILES["ascend"]
    )
    return json.loads(path.read_text(encoding="utf-8"))["model_deploy_config"]


def _as_dict(value):
    return json.loads(value) if isinstance(value, str) else value


@pytest.mark.parametrize("distributed", [False, True])
def test_0731_w8a8_910c_selects_exact_profile(monkeypatch, distributed):
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")

    config = config_loader._get_model_specific_config(
        {"device": "ascend", "count": 16, "details": [{"name": "Ascend910C"}]},
        {
            "engine": "vllm_ascend",
            "model_name": MODEL_NAME,
            "model_path": MODEL_PATH,
            "model_type": "llm",
            "distributed": distributed,
            "enable_auto_tool_choice": True,
            "enable_auto_think_choice": True,
        },
        _FakeDeepSeekV4Info(),
    )

    assert config["max_num_batched_tokens"] == 10240
    assert config["gpu_memory_utilization"] == 0.9
    assert config["api_server_count"] == 1
    assert config["max_num_seqs"] == 64
    assert config["enable_expert_parallel"] is True
    assert config["quantization"] == "ascend"
    if distributed:
        assert config["max_model_len"] == 1048576
        assert config["block_size"] == 128
        assert config["async_scheduling"] is True
        assert config["safetensors_load_strategy"] == "prefetch"
        assert _as_dict(config["model_loader_extra_config"]) == {
            "enable_multithread_load": "true",
            "num_threads": 128,
        }
        expected_additional_config = {
            "ascend_compilation_config": {
                "enable_npugraph_ex": True,
                "enable_static_kernel": False,
            },
            "enable_cpu_binding": True,
            "multistream_overlap_shared_expert": True,
        }
    else:
        assert config["max_model_len"] == 136192
        assert config["block_size"] == 32
        assert "async_scheduling" not in config
        assert "safetensors_load_strategy" not in config
        assert _as_dict(config["model_loader_extra_config"]) == {
            "enable_multithread_load": True,
            "num_threads": 128,
        }
        expected_additional_config = {
            "ascend_compilation_config": {
                "enable_npugraph_ex": True,
                "enable_static_kernel": False,
            },
            "enable_cpu_binding": True,
            "enable_dsa_cp": True,
            "enable_flashcomm1": True,
            "multistream_overlap_shared_expert": True,
        }
    assert _as_dict(config["compilation_config"]) == {"cudagraph_mode": "FULL_DECODE_ONLY"}
    assert _as_dict(config["additional_config"]) == expected_additional_config
    assert config["tokenizer_mode"] == "deepseek_v4"
    assert config["tool_call_parser"] == "deepseek_v4"
    assert config["reasoning_parser"] == "deepseek_v4"
    assert config["served_model_name"] == MODEL_NAME
    # 0731 精确 profile 不继承旧 w8a8-mtp 的额外启动开关，避免模型配方互相漂移。
    assert "trust_remote_code" not in config
    assert "no_disable_hybrid_kv_cache_manager" not in config
    assert "tensor_parallel_size" not in config
    assert "data_parallel_size" not in config


def test_0731_w8a8_910c_keeps_legacy_profile_and_features_isolated():
    deepseek_v4 = _ascend_defaults()["llm"]["DeepseekV4ForCausalLM"]
    legacy = deepseek_v4["DeepSeek-V4-Flash-w8a8-mtp-Ascend910C"]["vllm_ascend"]

    assert legacy["trust_remote_code"] is True
    assert legacy["no_disable_hybrid_kv_cache_manager"] is True
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend", MODEL_NAME, MODEL_PATH, "910c"
    ) == frozenset({"spec", "sparse"})
    assert model_utils.resolve_feature_whitelist(
        "vllm_ascend", MODEL_NAME, LEGACY_MODEL_PATH, "910c"
    ) == frozenset({"spec", "sparse", "offload"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"distributed": True},
        {"device_count": 8},
        {"nnodes": 2},
    ],
)
@pytest.mark.parametrize("pd_role", [None, "P", "D"])
def test_0731_w8a8_910c_env_scope_ignores_topology(monkeypatch, overrides, pd_role):
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")
    if pd_role:
        monkeypatch.setenv("PD_ROLE", pd_role)
    else:
        monkeypatch.delenv("PD_ROLE", raising=False)
    params = {
        "engine": "vllm_ascend",
        "model_name": MODEL_NAME,
        "device_count": 16,
        "nnodes": 1,
        "distributed": False,
        "_smart_card_token": "910c",
    }
    params.update(overrides)

    assert vllm_adapter._is_deepseek_v4_flash_0731_w8a8_910c_env_scope(
        params, _FakeDeepSeekV4Info()
    )


@pytest.mark.parametrize(
    ("overrides", "platform"),
    [
        ({"model_name": "DeepSeek-V4-Flash-w8a8-mtp"}, "a3"),
        ({"engine": "vllm"}, "a3"),
        ({}, "a2"),
    ],
)
def test_0731_w8a8_910c_env_scope_keeps_model_hardware_boundary(
    monkeypatch, overrides, platform
):
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", platform)
    params = {
        "engine": "vllm_ascend",
        "model_name": MODEL_NAME,
        "_smart_card_token": "910c" if platform == "a3" else "910b",
    }
    params.update(overrides)

    assert not vllm_adapter._is_deepseek_v4_flash_0731_w8a8_910c_env_scope(
        params, _FakeDeepSeekV4Info()
    )


@pytest.mark.parametrize(("device_count", "expected_dp"), [(8, 2), (16, 4)])
@pytest.mark.parametrize("spec_enabled", [True, False])
@pytest.mark.parametrize("sparse_enabled", [True, False])
def test_0731_w8a8_910c_final_command_matches_recipe(
    monkeypatch, spec_enabled, sparse_enabled, device_count, expected_dp
):
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")
    monkeypatch.setenv("SERVED_MODEL_NAME", "dsv4")
    for env_name in (
        "ENABLE_KV_OFFLOAD",
        "ENABLE_SPARSE",
        "ENABLE_SPECULATIVE_DECODE",
        "SD_ENABLE",
        "TENSOR_PARALLEL_SIZE",
        "DATA_PARALLEL_SIZE",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeDeepSeekV4Info)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Info)

    argv = [
        "--model-name", MODEL_NAME,
        "--model-path", MODEL_PATH,
        "--model-type", "llm",
        "--engine", "vllm_ascend",
        "--device-count", str(device_count),
        "--port", "8900",
        "--enable-auto-tool-choice",
        "--enable-auto-think-choice",
        "--speculative-decode-model-path", "none",
    ]
    if spec_enabled:
        argv.append("--enable-speculative-decode")
    if sparse_enabled:
        argv.append("--enable-sparse")
    params = config_loader.load_and_merge_configs(
        {
            "device": "ascend",
            "count": device_count,
            "details": [{"name": "Ascend910C"}],
        },
        parse_launch_args(argv),
    )

    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    expected_effective = set()
    if spec_enabled:
        expected_effective.add("spec")
    if sparse_enabled:
        expected_effective.add("sparse")
    assert set(params["_allowed_smart_feats"]) == {"spec", "sparse"}
    assert set(params["_smart_feats"]) == expected_effective

    for export in (
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=10",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        'export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"',
        "export HCCL_BUFFSIZE=1024",
        "export TASK_QUEUE_ENABLE=1",
        'export HCCL_OP_EXPANSION_MODE="AIV"',
        "export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096",
        "export VLLM_ENGINE_READY_TIMEOUT_S=3600",
    ):
        assert script.count(export) == 1
    assert "export VLLM_ASCEND_ENABLE_FLASHCOMM1=" not in script

    assert exec_line.startswith(f"exec vllm serve {MODEL_PATH} ")
    for flag in (
        "--max-model-len 136192",
        "--max-num-batched-tokens 10240",
        "--served-model-name dsv4",
        "--gpu-memory-utilization 0.9",
        "--api-server-count 1",
        "--max-num-seqs 64",
        f"--tensor-parallel-size 4 --data-parallel-size {expected_dp}",
        "--enable-expert-parallel",
        "--tokenizer-mode deepseek_v4",
        "--tool-call-parser deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser deepseek_v4",
        "--model-loader-extra-config '" + '{"enable_multithread_load":true,"num_threads":128}' + "'",
        "--quantization ascend",
        "--port 8900",
        "--block-size 32",
    ):
        assert flag in exec_line
    assert "--async-scheduling" not in exec_line
    assert "--safetensors-load-strategy" not in exec_line
    assert '"cudagraph_mode":"FULL_DECODE_ONLY"' in exec_line
    assert '"enable_npugraph_ex":true' in exec_line
    assert '"enable_dsa_cp":true' in exec_line
    assert '"enable_flashcomm1":true' in exec_line
    assert '"multistream_overlap_shared_expert":true' in exec_line
    assert "--trust-remote-code" not in exec_line
    assert "--no-disable-hybrid-kv-cache-manager" not in exec_line
    assert "--kv-cache-dtype" not in exec_line
    assert "--kv-offloading-backend" not in exec_line

    if sparse_enabled:
        assert (
            "'" + '{"use_index_cache":true,"index_topk_freq":8}' + "'"
            in exec_line
        )
    else:
        assert "--hf-overrides" not in exec_line

    if spec_enabled:
        # 0731-W8A8 在 910C 上使用用户确认的 DSpark7 配方，且只能生成一次。
        assert exec_line.count("--speculative-config") == 1
        assert (
            "'" + '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}' + "'"
            in exec_line
        )
        assert '"method":"mtp"' not in exec_line
        assert '"num_speculative_tokens":1' not in exec_line
    else:
        assert "--speculative-config" not in exec_line
