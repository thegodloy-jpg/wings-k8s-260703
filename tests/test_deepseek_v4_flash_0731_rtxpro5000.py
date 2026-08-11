import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wings_control"))

from core import config_loader, wings_entry  # noqa: E402
from engines import vllm_adapter  # noqa: E402
from utils import model_utils  # noqa: E402


MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_PATH = "/model/DeepSeek-V4-Flash-0731"
RTXPRO5000_HARDWARE = {
    "device": "nvidia",
    "count": 4,
    "details": [{"name": "TokenBox RTX PRO 5000 * 4"}],
    "hardware_family": "TokenBox RTX PRO 5000 * 4",
}


class _FakeDeepSeekV4Info:
    model_architecture = "DeepseekV4ForCausalLM"
    model_quantize = ""
    model_type = "llm"
    config = {}

    def __init__(self, model_name=MODEL_NAME, model_path=MODEL_PATH, model_type="llm"):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type

    def identify_model_architecture(self):
        return self.model_architecture

    def identify_model_type(self):
        return self.model_type

    def is_wings_supported(self):
        return True


def _deepseek_v4_defaults():
    path = ROOT / "wings_control" / "config" / "defaults" / "nvidia_default.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["model_deploy_config"]["llm"]["DeepseekV4ForCausalLM"]


def _match_defaults(model_name, hardware):
    return config_loader._match_model_engine_config(
        _deepseek_v4_defaults(),
        model_name.lower(),
        "vllm",
        config_loader._SpecialEngineScenario(
            deepseek_v4_flash_vllm_nvidia=True,
        ),
        _FakeDeepSeekV4Info(model_name=model_name),
        hardware,
        MODEL_PATH.lower(),
    )


@pytest.mark.parametrize(
    "model_name",
    ["DeepSeek-V4-Flash-0731", MODEL_NAME],
)
def test_0731_rtxpro5000_selects_independent_exact_profile(model_name):
    assert _match_defaults(model_name, RTXPRO5000_HARDWARE) == {
        "use_vllm_serve": True,
        "max_model_len": 1048576,
        "block_size": 256,
        "enable_expert_parallel": True,
        "tokenizer_mode": "deepseek_v4",
        "tool_call_parser": "deepseek_v4",
        "gpu_memory_utilization": 0.8,
        "enable_prefix_caching": True,
        "compilation_config": {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "custom_ops": ["all"],
        },
        "async_scheduling": True,
        "attention_backend": "FLASHINFER_MLA_SPARSE_DSV4",
        "enable_flashinfer_autotune": True,
        "all2all_backend": "deepep_high_throughput",
    }


def test_0731_rtxpro5000_profile_does_not_replace_existing_nvidia_profiles():
    base = _match_defaults("deepseek-ai/DeepSeek-V4-Flash", RTXPRO5000_HARDWARE)
    h20 = _match_defaults(
        MODEL_NAME,
        {"device": "nvidia", "details": [{"name": "NVIDIA H20 96GB"}]},
    )
    rtxpro5000_48 = _match_defaults(
        MODEL_NAME,
        {
            "device": "nvidia",
            "details": [{"name": "NVIDIA RTX PRO 5000 48GB Blackwell"}],
        },
    )

    assert base["max_model_len"] == 140000
    assert base["attention_backend"] == "FLASHINFER_MLA_SPARSE_SM120_DSV4"
    assert base["enable_eplb"] is True
    assert h20["max_model_len"] == 200000
    assert h20["disable_custom_all_reduce"] is True
    assert "kv_cache_dtype" not in h20
    assert rtxpro5000_48["max_model_len"] == 4096
    assert rtxpro5000_48["kv_cache_dtype"] == "fp8"


def test_0731_rtxpro5000_whitelist_selects_dspark5_and_fp8_indexcache():
    allowed = model_utils.resolve_feature_whitelist(
        "vllm",
        MODEL_NAME,
        MODEL_PATH,
        "rtxpro5000-72",
    )
    sparse_row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        MODEL_NAME,
        MODEL_PATH,
        "rtxpro5000-72",
        "sparse",
    )
    spec_row = model_utils.resolve_feature_whitelist_row(
        "vllm",
        MODEL_NAME,
        MODEL_PATH,
        "rtxpro5000-72",
        "spec",
    )

    assert allowed == {"sparse", "spec"}
    assert sparse_row is not None
    assert sparse_row["strategy"] == "indexcache"
    assert sparse_row["kv_cache_dtype"] == "fp8"
    assert sparse_row["topk"] == {"accuracy_first": 4}
    assert spec_row is not None
    assert spec_row["mtp_method"] == "dspark"
    assert spec_row["mtp_num_speculative_tokens"] == 5
    assert spec_row["draft_sample_method"] == "greedy"

    base_allowed = model_utils.resolve_feature_whitelist(
        "vllm",
        "deepseek-ai/DeepSeek-V4-Flash",
        "/model/DeepSeek-V4-Flash",
        "rtxpro5000-72",
    )
    assert {"spec", "sparse"}.issubset(base_allowed)


@pytest.mark.parametrize(
    ("engine", "model_name", "card_token", "expected"),
    [
        ("vllm", MODEL_NAME, "rtxpro5000-72", True),
        ("vllm", "DeepSeek-V4-Flash-0731", "rtxpro5000-72", True),
        ("vllm", "deepseek-ai/DeepSeek-V4-Flash", "rtxpro5000-72", False),
        ("vllm", MODEL_NAME, "rtxpro5000-48", False),
        ("vllm", MODEL_NAME, "h20-141", False),
        ("vllm_ascend", "DeepSeek-V4-Flash-0731-w8a8", "910c", False),
    ],
)
def test_0731_rtxpro5000_profile_scope_is_exact(
    engine,
    model_name,
    card_token,
    expected,
):
    params = {
        "engine": engine,
        "model_name": model_name,
        "model_path": MODEL_PATH,
        "_smart_card_token": card_token,
        # API 别名不能代替真实模型身份触发精确 profile。
        "served_model_name": "DeepSeek-V4-Flash-0731",
    }

    assert vllm_adapter.is_deepseek_v4_flash_0731_rtx_pro_5000_scope(
        params,
        engine,
    ) is expected


def _build_rtxpro5000_params(enable_sparse):
    engine_config = _match_defaults(MODEL_NAME, RTXPRO5000_HARDWARE)
    engine_config.update(
        {
            "model": MODEL_PATH,
            "host": "0.0.0.0",
            "port": 5685,
            "served_model_name": "DeepSeek-V4-Flash",
            "enable_auto_tool_choice": True,
            "reasoning_parser": "deepseek_v4",
        }
    )
    return {
        "engine": "vllm",
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "model_type": "llm",
        "device_count": 4,
        "enable_sparse": enable_sparse,
        "enable_speculative_decode": False,
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": ["sparse"] if enable_sparse else [],
        "engine_config": engine_config,
    }


def test_0731_rtxpro5000_final_command_contains_static_and_sparse_recipe(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Info)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "false")
    params = _build_rtxpro5000_params(enable_sparse=True)
    config_loader.apply_effective_feature_enablement(params, RTXPRO5000_HARDWARE)

    assert params["_allowed_smart_feats"] == ["sparse", "spec"]
    assert params["_smart_feats"] == ["sparse"]

    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert exec_line.startswith(f"exec vllm serve {MODEL_PATH} ")
    assert exec_line.count("--kv-cache-dtype fp8") == 1
    assert "--block-size 256" in exec_line
    assert "--tensor-parallel-size 4" in exec_line
    assert "--data-parallel-size 1" in exec_line
    assert "--max-model-len 1048576" in exec_line
    assert "--enable-expert-parallel" in exec_line
    assert "--gpu-memory-utilization 0.8" in exec_line
    assert "--enable-prefix-caching" in exec_line
    assert "--async-scheduling" in exec_line
    assert "--attention-backend FLASHINFER_MLA_SPARSE_DSV4" in exec_line
    assert "--enable-flashinfer-autotune" in exec_line
    assert "--all2all-backend deepep_high_throughput" in exec_line
    assert '"cudagraph_mode":"FULL_AND_PIECEWISE"' in exec_line
    assert '"custom_ops":["all"]' in exec_line
    assert "--hf-overrides" in exec_line
    assert '"use_index_cache":true' in exec_line
    assert '"index_topk_freq":4' in exec_line
    assert "--speculative-config" not in exec_line
    assert "--kv-offloading-backend" not in exec_line
    assert "--kv-offloading-size" not in exec_line
    assert (
        vllm_adapter.resolve_sparse_variant(params, "vllm")
        == "fp8_indexcache_use_index_cache_topk4"
    )

    status = wings_entry._resolve_advanced_feature_status("vllm", params)
    assert status["features"]["sparse_kv"] is True
    assert status["features"]["speculative_decode"] is False
    assert status["features"]["kv_offload"] is False


def test_0731_rtxpro5000_block_size_uses_default_but_keeps_explicit_override(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Info)
    params = _build_rtxpro5000_params(enable_sparse=True)

    assert params["engine_config"]["block_size"] == 256

    params["_explicit_cli_keys"] = ["block_size"]
    params["engine_config"]["block_size"] = 128
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert "--block-size 128" in exec_line
    assert "--block-size 256" not in exec_line


def test_0731_rtxpro5000_page_spec_request_generates_dspark5(
    monkeypatch,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Info)
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    params = _build_rtxpro5000_params(enable_sparse=True)
    params.update({
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
    })

    config_loader.apply_effective_feature_enablement(params, RTXPRO5000_HARDWARE)
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert params["_allowed_smart_feats"] == ["sparse", "spec"]
    assert params["_smart_feats"] == ["sparse", "spec"]
    assert params["enable_speculative_decode"] is True
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "true"
    assert os.environ["SD_ENABLE"] == "true"
    assert vllm_adapter.should_append_auto_speculative_config(params) is True
    assert (
        vllm_adapter.build_speculative_cmd(params, "vllm")
        == " --speculative-config "
        "'{\"method\":\"dspark\",\"num_speculative_tokens\":5,"
        "\"draft_sample_method\":\"greedy\"}'"
    )
    assert exec_line.count("--speculative-config") == 1
    assert '"method":"dspark"' in exec_line
    assert '"num_speculative_tokens":5' in exec_line
    assert '"draft_sample_method":"greedy"' in exec_line
    assert '"method" : "suffix"' not in exec_line
    assert exec_line.count("--kv-cache-dtype fp8") == 1
    assert '"use_index_cache":true' in exec_line

    status = wings_entry._resolve_advanced_feature_status("vllm", params)
    assert status["features"]["speculative_decode"] is True
    assert status["features"]["sparse_kv"] is True


def test_0731_rtxpro5000_spec_off_removes_explicit_speculative_config(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Info)
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    params = _build_rtxpro5000_params(enable_sparse=True)
    params["engine_config"]["speculative_config"] = {
        "method": "dspark",
        "num_speculative_tokens": 5,
    }

    prepared = vllm_adapter._prepare_engine_config(params)
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert "speculative_config" not in prepared
    assert "speculative_config" not in params["engine_config"]
    assert params["enable_speculative_decode"] is False
    assert params["_smart_feats"] == ["sparse"]
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "false"
    assert os.environ["SD_ENABLE"] == "false"
    assert vllm_adapter.build_speculative_cmd(params, "vllm") == ""
    assert "--speculative-config" not in exec_line
    assert exec_line.count("--kv-cache-dtype fp8") == 1


def test_0731_rtxpro5000_sparse_off_does_not_leave_fp8(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Info)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "false")
    params = _build_rtxpro5000_params(enable_sparse=False)
    config_loader.apply_effective_feature_enablement(params, RTXPRO5000_HARDWARE)

    assert params["_allowed_smart_feats"] == ["sparse", "spec"]
    assert params["_smart_feats"] == []

    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert "--kv-cache-dtype" not in exec_line
    assert "--hf-overrides" not in exec_line
    assert "--speculative-config" not in exec_line
    assert "kv_cache_dtype" not in params["engine_config"]


def test_0731_rtxpro5000_keeps_existing_accel_package_path(monkeypatch):
    monkeypatch.setenv("ENGINE_VERSION", "v0.23.0")
    params = _build_rtxpro5000_params(enable_sparse=True)

    assert wings_entry._should_install_deepseek_v4_flash_pro5000_packages(
        "vllm",
        params,
    ) is True
    preamble = wings_entry._build_accel_preamble("vllm", params)
    assert preamble.count("python3 install.py --config") == 1
    assert "deepgemm:nv_dev_a6b593d" in preamble
    assert "flashinfer:v0.6.12" in preamble
