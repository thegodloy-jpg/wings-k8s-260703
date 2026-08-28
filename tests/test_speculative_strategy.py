import json
import logging
import os
import shlex
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from core import config_loader, wings_entry  # noqa: E402
from core.port_plan import derive_port_plan  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from engines import vllm_adapter  # noqa: E402


class _FakeModelIdentifier:
    model_architecture = "Qwen3_5ForConditionalGeneration"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeDeepSeekV4Identifier:
    config = {}
    model_architecture = "DeepseekV4ForCausalLM"
    model_quantize = "w8a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type

    @staticmethod
    def identify_model_architecture():
        return "DeepseekV4ForCausalLM"

    @staticmethod
    def identify_model_type():
        return "llm"


class _FakeKimiK3Identifier:
    model_architecture = "KimiK3ForConditionalGeneration"
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeDeepSeekV2Identifier:
    model_architecture = "DeepseekV2ForCausalLM"
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeGlm51Identifier:
    model_architecture = "GlmMoeDsaForCausalLM"
    model_quantize = "w8a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeGlm47Identifier:
    model_architecture = "Glm4MoeForCausalLM"
    model_quantize = "w8a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeKimiIdentifier:
    model_architecture = "KimiK25ForConditionalGeneration"
    model_quantize = "w4a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeQwen36MoeIdentifier:
    model_architecture = "Qwen3_5MoeForConditionalGeneration"
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeMiniMaxM2Identifier:
    model_architecture = "MiniMaxM2ForCausalLM"
    model_quantize = "w8a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeUnknownGlm51Identifier:
    model_architecture = "unknown_architecture"
    model_quantize = "w8a8"
    config = {}

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeQwen2Identifier:
    model_architecture = "Qwen2ForCausalLM"
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakePro5000Identifier:
    model_quantize = ""

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        text = f"{model_name} {model_path}".lower()
        if "qwen3.5-397b-a17b-nvfp4" in text:
            self.model_architecture = "Qwen3_5MoeForConditionalGeneration"
            self.model_quantize = "nvfp4"
        elif "qwen-agentworld-35b-a3b" in text:
            self.model_architecture = "Qwen3_5MoeForConditionalGeneration"
        elif "qwen3.5-122b-a10b" in text or "qwen3.5-35b-a3b" in text:
            self.model_architecture = "Qwen3_5MoeForConditionalGeneration"
        elif "qwen3.5-27b" in text:
            self.model_architecture = "Qwen3_5ForConditionalGeneration"
        elif "deepseek-v4-flash" in text:
            self.model_architecture = "DeepseekV4ForCausalLM"
            self.model_quantize = "fp4"
        elif "minimax-m3-mxfp8" in text:
            self.model_architecture = "MiniMaxM3SparseForConditionalGeneration"
        elif "minimax-m2.5-nvfp4" in text:
            self.model_architecture = "MiniMaxM2ForCausalLM"
        elif "minimax-m2.7-nvfp4" in text:
            self.model_architecture = "MiniMaxM2ForCausalLM"
        else:
            self.model_architecture = "unknown_architecture"


def _export_by_name(commands):
    return {
        command.split(" ", 1)[1].split("=", 1)[0]: command
        for command in commands
        if command.startswith("export ") and "=" in command
    }


def test_resolve_speculative_strategy_passes_engine_to_mtp_method(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeModelIdentifier)

    strategy = vllm_adapter.resolve_speculative_strategy(
        {
            "model_name": "Qwen3.6-27B-w8a8",
            "model_path": "/usr/local/serving/models/",
            "model_type": "llm",
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec"],
        },
        "vllm_ascend",
    )

    assert strategy == "qwen3_5_mtp"


def test_qwen_day0_memcache_keeps_mtp_strategy_and_whitelist_tokens(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeModelIdentifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    command = vllm_adapter.build_speculative_cmd(
        {
            "engine": "vllm_ascend",
            "model_name": "Qwen/Qwen3.5-27B",
            "model_path": "/models/Qwen/Qwen3.5-27B",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_card_token": "910c",
            "_smart_feats": ["offload", "spec"],
        },
        "vllm_ascend",
    )

    assert '"method": "qwen3_5_mtp"' in command
    assert '"num_speculative_tokens": 1' in command
    assert '"enforce_eager": true' in command
    assert "suffix" not in command


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_qwen35_35b_a3b_ascend_mtp_matches_day0_recipe(monkeypatch, card_token):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen36MoeIdentifier)

    command = vllm_adapter.build_speculative_cmd(
        {
            "engine": "vllm_ascend",
            "model_name": "Qwen/Qwen3.5-35B-A3B",
            "model_path": "/models/Qwen/Qwen3.5-35B-A3B",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_card_token": card_token,
            "_smart_feats": ["spec"],
        },
        "vllm_ascend",
    )

    assert command == (
        " --speculative-config "
        "'{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":1,\"enforce_eager\":true}'"
    )


def test_deepseek_coder_v2_ascend_uses_whitelist_suffix_default(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV2Identifier)
    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-Coder-V2-Instruct-BF16",
        "model_path": "/models/DeepSeek-Coder-V2-Instruct-BF16",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_card_token": "910c",
        "_smart_feats": ["spec"],
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")
    config = json.loads(command.split("'", 2)[1])

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    assert config == {
        "method": "suffix",
        "num_speculative_tokens": 5,
        "suffix_decoding_max_cached_requests": 1000,
    }


def test_deepseek_v4_flash_ascend_speculative_config_uses_vllm_021_mtp(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)

    command = vllm_adapter.build_speculative_cmd(
        {
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec"],
        },
        "vllm_ascend",
    )

    assert '"method": "mtp"' in command
    assert '"num_speculative_tokens": 1' in command
    assert '"enforce_eager": true' in command
    assert "deepseek_mtp" not in command


def test_deepseek_v4_flash_adapter_does_not_recreate_json_owned_runtime_defaults(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")

    engine_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 16,
            "engine_config": {},
        }
    )

    for key in (
        "quantization",
        "block_size",
        "async_scheduling",
        "safetensors_load_strategy",
        "tokenizer_mode",
        "tool_call_parser",
        "enable_auto_tool_choice",
    ):
        assert key not in engine_config


def test_deepseek_v4_flash_topology_prefers_hardware_info_over_engine_version(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_IMAGE_FLAVOR", raising=False)
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a2")

    engine_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 16,
            "device_details": [{"name": "Ascend910C"}],
            "engine_config": {},
        }
    )

    assert engine_config["tensor_parallel_size"] == 4
    assert engine_config["data_parallel_size"] == 4
    assert engine_config["api_server_count"] == 1


def test_deepseek_v4_flash_topology_keeps_engine_version_platform_fallback(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.delenv("WINGS_ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ASCEND_PLATFORM", raising=False)
    monkeypatch.delenv("ENGINE_IMAGE_FLAVOR", raising=False)
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a3")

    engine_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 16,
            "engine_config": {},
        }
    )

    assert engine_config["tensor_parallel_size"] == 4
    assert engine_config["data_parallel_size"] == 4
    assert engine_config["api_server_count"] == 1


def test_deepseek_v4_flash_topology_ignores_explicit_platform_env(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a2")

    engine_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 16,
            "engine_config": {},
        }
    )

    assert engine_config["tensor_parallel_size"] == 8
    assert engine_config["data_parallel_size"] == 2
    assert "api_server_count" not in engine_config


def test_deepseek_v4_flash_pro5000_vllm_speculative_config_matches_tokenbox(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)

    params = {
        "engine": "vllm",
        "model_name": "Deepseek-v4-Flash",
        "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec", "sparse"],
        "_smart_card_token": "rtxpro5000-72",
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert command == (
        " --speculative-config "
        "'{\"method\":\"mtp\",\"num_speculative_tokens\":2}'"
    )
    assert '"enforce_eager": true' not in command
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "mtp",
        "num_speculative_tokens": 2,
        "moe_backend": None,
    }


def test_deepseek_v4_flash_h20_vllm_speculative_config_matches_day0_recipe(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)

    params = {
        "engine": "vllm",
        "model_name": "deepseek-ai/DeepSeek-V4-Flash",
        "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec", "sparse", "offload"],
        "_smart_card_token": "h20-141",
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm") == "mtp"
    assert vllm_adapter.build_speculative_cmd(params, "vllm") == (
        " --speculative-config "
        "'{\"method\":\"mtp\",\"num_speculative_tokens\":1}'"
    )
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "moe_backend": None,
    }


@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
@pytest.mark.parametrize("offload_size", [80, 200])
def test_deepseek_v4_flash_0731_h20_final_command_contains_complete_recipe(
    monkeypatch,
    card_token,
    offload_size,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", str(offload_size))
    model_path = "/var/ai-model/LocalStorage/DeepSeek-V4-Flash-0731"
    params = {
        "engine": "vllm",
        "model_name": "DeepSeek-V4-Flash-0731",
        "model_path": model_path,
        "model_type": "llm",
        "device_count": 8,
        "enable_sparse": True,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_card_token": card_token,
        "engine_config": {
            "use_vllm_serve": True,
            "model": model_path,
            "trust_remote_code": True,
            "block_size": 256,
            "enable_expert_parallel": True,
            "tensor_parallel_size": 8,
            "tokenizer_mode": "deepseek_v4",
            "tool_call_parser": "deepseek_v4",
            "enable_auto_tool_choice": True,
            "reasoning_parser": "deepseek_v4",
            "max_model_len": 200000,
            "disable_custom_all_reduce": True,
            "served_model_name": "DeepSeek-V4-Flash-0731",
            "port": 18000,
        },
    }

    card_name = "NVIDIA H20 96GB" if card_token == "h20-96" else "NVIDIA H20 141GB"
    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "nvidia", "count": 8, "details": [{"name": card_name}]},
    )
    assert params["_allowed_smart_feats"] == ["offload", "sparse", "spec"]
    assert params["_smart_feats"] == ["offload", "sparse", "spec"]
    assert params["_smart_feature_gate_trace"]["features"]["kv_offload"] == {
        "requested": True,
        "whitelist": True,
        "gate": True,
        "reason": "enabled",
    }
    assert os.environ["ENABLE_KV_OFFLOAD"] == "true"
    assert os.environ["LMCACHE_OFFLOAD"] == "true"

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm") == "dspark"
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "dspark",
        "num_speculative_tokens": 2,
        "moe_backend": None,
        "draft_sample_method": "probabilistic",
    }

    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert exec_line.startswith(f"exec vllm serve {model_path} ")
    assert exec_line.count("--kv-cache-dtype fp8") == 1
    assert exec_line.count("--hf-overrides") == 1
    assert "'{}'" not in exec_line
    assert "'" + '{"use_index_cache":true,"index_topk_freq":4}' + "'" in exec_line
    expected_spec_config = (
        "'"
        + '{"method":"dspark","num_speculative_tokens":2,'
        '"draft_sample_method":"probabilistic"}'
        + "'"
    )
    assert expected_spec_config in exec_line
    for expected_arg in (
        "--trust-remote-code",
        "--block-size 256",
        "--enable-expert-parallel",
        "--tensor-parallel-size 8",
        "--tokenizer-mode deepseek_v4",
        "--tool-call-parser deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser deepseek_v4",
        "--max-model-len 200000",
        "--disable-custom-all-reduce",
        "--served-model-name DeepSeek-V4-Flash-0731",
        "--port 18000",
    ):
        assert expected_arg in exec_line
    assert "--kv-offloading-backend native" in exec_line
    assert f"--kv-offloading-size {offload_size}" in exec_line
    assert wings_entry._should_install_nvidia_native_offload_packages(
        "vllm", params
    ) is True
    feature_status = wings_entry._resolve_advanced_feature_status("vllm", params)
    assert feature_status["features"]["kv_offload"] is True
    assert feature_status["variants"]["kv_offload"] == "native_kv_offloading_backend"
    assert feature_status["others"]["kv_mem_offload_size"] == offload_size
    assert vllm_adapter.resolve_sparse_variant(params, "vllm") == (
        "fp8_indexcache_use_index_cache_topk4"
    )
    params["enable_sparse"] = False
    params["_smart_feats"] = ["offload", "spec"]
    sparse_off_script = vllm_adapter.build_start_script(params)
    assert "--kv-cache-dtype" not in sparse_off_script
    assert "--hf-overrides" not in sparse_off_script
    assert f"--kv-offloading-size {offload_size}" in sparse_off_script
    assert "kv_cache_dtype" not in params["engine_config"]

    # 页面关闭内存卸载后，命令、状态和安装判定必须使用同一最终解析结果。
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "false")
    mem_offload_disabled_script = vllm_adapter.build_start_script(params)
    assert "--kv-offloading-backend" not in mem_offload_disabled_script
    assert "--kv-offloading-size" not in mem_offload_disabled_script
    assert wings_entry._should_install_nvidia_native_offload_packages(
        "vllm", params
    ) is False
    feature_status = wings_entry._resolve_advanced_feature_status("vllm", params)
    assert feature_status["features"]["kv_offload"] is False
    assert feature_status["variants"]["kv_offload"] == "disabled"
    assert feature_status["others"]["kv_mem_offload_size"] is None


@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
def test_deepseek_v4_pro_0813_h20_final_mp_command_matches_recipe(
    monkeypatch,
    card_token,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_SPARSE", "false")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    monkeypatch.setenv("NETWORK_INTERFACE", "bond0")
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    model_path = "/models/DeepSeek-V4-Pro-0813"
    params = {
        "engine": "vllm",
        "model_name": "DeepSeek-V4-Pro-0813",
        "model_path": model_path,
        "model_type": "llm",
        "device_count": 8,
        "distributed": True,
        "distributed_executor_backend": "mp",
        "nnodes": 4,
        "node_rank": 0,
        "master_ip": "7.6.25.57",
        "master_port": 29501,
        "enable_sparse": False,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "engine_config": {
            "use_vllm_serve": True,
            "model": model_path,
            "trust_remote_code": True,
            "kv_cache_dtype": "fp8",
            "block_size": 256,
            "enable_expert_parallel": True,
            "tensor_parallel_size": 32,
            "max_model_len": 200000,
            "gpu_memory_utilization": 0.95,
            "max_num_seqs": 16,
            "no_enable_flashinfer_autotune": True,
            "disable_custom_all_reduce": True,
            "compilation_config": json.dumps({
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
            }),
            "tokenizer_mode": "deepseek_v4",
            "enable_auto_tool_choice": True,
            "tool_call_parser": "deepseek_v4",
            "reasoning_parser": "deepseek_v4",
        },
    }
    card_name = "NVIDIA H20 96GB" if card_token == "h20-96" else "NVIDIA H20 141GB"

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "nvidia", "count": 8, "details": [{"name": card_name}]},
    )
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert params["_allowed_smart_feats"] == ["offload", "sparse", "spec"]
    assert params["_smart_feats"] == ["spec"]
    assert "export VLLM_HOST_IP=" in script
    assert "export NCCL_SOCKET_IFNAME=bond0" in script
    assert "export GLOO_SOCKET_IFNAME=bond0" in script
    for expected in (
        "--trust-remote-code",
        "--block-size 256",
        "--enable-expert-parallel",
        "--tensor-parallel-size 32",
        "--max-model-len 200000",
        "--gpu-memory-utilization 0.95",
        "--max-num-seqs 16",
        "--no-enable-flashinfer-autotune",
        "--disable-custom-all-reduce",
        "--compilation-config",
        "--tokenizer-mode deepseek_v4",
        "--enable-auto-tool-choice",
        "--tool-call-parser deepseek_v4",
        "--reasoning-parser deepseek_v4",
        "--distributed-executor-backend mp",
        "--nnodes 4",
        "--node-rank 0",
        "--master-addr 7.6.25.57",
        "--master-port 29501",
    ):
        assert expected in exec_line
    assert '"method":"dspark"' in exec_line
    assert '"num_speculative_tokens":5' in exec_line
    assert '"draft_sample_method":"greedy"' in exec_line
    assert '"cudagraph_mode":"FULL_DECODE_ONLY"' in exec_line
    assert exec_line.count("--speculative-config") == 1
    # Pro-0813 H20 的 FP8 KV 与 IndexCache 同属 sparse 能力包；关闭 sparse
    # 时两者都不能从静态 profile 残留到最终命令。
    assert "--kv-cache-dtype fp8" not in exec_line
    assert "--hf-overrides" not in exec_line
    assert "--headless" not in exec_line
    assert "--data-parallel-size" not in exec_line
    assert "--kv-transfer-config" not in exec_line


@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
@pytest.mark.parametrize("node_rank", [0, 1])
def test_deepseek_v4_pro_0813_dual_h20_matches_tuned_three_feature_recipe(
    monkeypatch,
    card_token,
    node_rank,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_SPARSE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "512")
    monkeypatch.setenv("NETWORK_INTERFACE", "bond0")
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    model_path = "/models/DeepSeek-V4-Pro-0813"
    params = {
        "engine": "vllm",
        "model_name": "DeepSeek-V4-Pro-0813",
        "model_path": model_path,
        "model_type": "llm",
        "device": "nvidia",
        "device_count": 8,
        "distributed": True,
        "distributed_executor_backend": "mp",
        "nnodes": 2,
        "node_rank": node_rank,
        "master_ip": "7.6.25.59",
        "master_port": 29501,
        "enable_sparse": True,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "engine_config": {
            "use_vllm_serve": True,
            "model": model_path,
            "trust_remote_code": True,
            "kv_cache_dtype": "fp8",
            "block_size": 256,
            # adapter 单测直接消费已经由 nvidia_default.json 合并好的引擎配置。
            "enable_prefix_caching": True,
            "enable_expert_parallel": True,
            "enable_ep_weight_filter": True,
            "tensor_parallel_size": 16,
            "max_model_len": 133000,
            "gpu_memory_utilization": 0.92,
            "max_num_seqs": 32,
            "max_num_batched_tokens": 8192,
            "no_enable_flashinfer_autotune": True,
            "disable_custom_all_reduce": True,
            "cpu_distributed_timeout_seconds": 7200,
            "compilation_config": {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
            "tokenizer_mode": "deepseek_v4",
            "enable_auto_tool_choice": True,
            "tool_call_parser": "deepseek_v4",
            "reasoning_parser": "deepseek_v4",
        },
    }
    card_name = "NVIDIA H20 96GB" if card_token == "h20-96" else "NVIDIA H20 141GB"

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "nvidia", "count": 8, "details": [{"name": card_name}]},
    )
    assert params["_allowed_smart_feats"] == ["offload", "sparse", "spec"]
    assert params["_smart_feats"] == ["offload", "sparse", "spec"]

    config_loader._set_kv_cache_config(
        params["engine_config"],
        params,
        _FakeDeepSeekV4Identifier(params["model_name"], model_path, "llm"),
    )
    connector = json.loads(params["engine_config"]["kv_transfer_config"])
    assert connector == {
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use_per_rank": 68719476736,
            "lazy_offload": False,
        },
    }

    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    for expected in (
        "--kv-cache-dtype fp8",
        "--block-size 256",
        "--enable-prefix-caching",
        "--enable-expert-parallel",
        "--enable-ep-weight-filter",
        "--tensor-parallel-size 16",
        "--max-model-len 133000",
        "--gpu-memory-utilization 0.92",
        "--max-num-seqs 32",
        "--max-num-batched-tokens 8192",
        "--cpu-distributed-timeout-seconds 7200",
        "--no-enable-flashinfer-autotune",
        "--disable-custom-all-reduce",
        "--nnodes 2",
        f"--node-rank {node_rank}",
        "--master-addr 7.6.25.59",
        "--master-port 29501",
    ):
        assert expected in exec_line
    assert exec_line.count("--speculative-config") == 1
    assert '"method":"dspark"' in exec_line
    assert '"num_speculative_tokens":5' in exec_line
    assert '"draft_sample_method":"greedy"' in exec_line
    assert exec_line.count("--hf-overrides") == 1
    assert '"use_index_cache":true,"index_topk_freq":8' in exec_line
    assert exec_line.count("SimpleCPUOffloadConnector") == 1
    assert exec_line.count("68719476736") == 1
    assert '"lazy_offload":false' in exec_line
    assert "--kv-offloading-backend" not in exec_line
    assert "LMCacheConnector" not in exec_line
    assert "LMCACHE_" not in script
    assert ("--headless" in exec_line) is (node_rank == 1)

    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "moe_backend": None,
        "draft_sample_method": "greedy",
    }
    assert vllm_adapter.resolve_sparse_variant(params, "vllm") == (
        "fp8_indexcache_use_index_cache_topk8"
    )
    feature_status = wings_entry._resolve_advanced_feature_status("vllm", params)
    assert feature_status["features"]["kv_offload"] is True
    assert feature_status["variants"]["kv_offload"] == (
        "simple_cpu_offload_connector+custom"
    )
    assert feature_status["others"]["kv_mem_offload_size"] == 512


@pytest.mark.parametrize(
    "topology",
    [
        {
            "distributed": True,
            "distributed_executor_backend": "mp",
            "nnodes": 2,
            "engine_config": {"tensor_parallel_size": 16},
        },
        {
            "distributed": True,
            "distributed_executor_backend": "mp",
            "nnodes": 4,
            "engine_config": {"tensor_parallel_size": 32},
        },
        {
            "distributed": False,
            "distributed_executor_backend": "ray",
            "nnodes": 1,
            "engine_config": {
                "tensor_parallel_size": 4,
                "data_parallel_size": 2,
            },
        },
    ],
)
def test_deepseek_v4_pro_0813_h20_features_do_not_depend_on_topology(
    monkeypatch,
    topology,
):
    """Pro-0813 H20 的三项特性仅由白名单和页面开关控制。"""
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_SPARSE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    params = {
        "engine": "vllm",
        "model_name": "DeepSeek-V4-Pro-0813",
        "model_path": "/models/DeepSeek-V4-Pro-0813",
        "device_count": 8,
        "enable_sparse": True,
        "enable_speculative_decode": True,
    }
    params.update(topology)

    config_loader.apply_effective_feature_enablement(
        params,
        {
            "device": "nvidia",
            "count": 8,
            "details": [{"name": "NVIDIA H20 96GB"}],
        },
    )

    assert params["_allowed_smart_feats"] == ["offload", "sparse", "spec"]
    assert params["_smart_feats"] == ["offload", "sparse", "spec"]


@pytest.mark.parametrize(
    ("device_count", "expected_per_rank_bytes"),
    [
        (4, 137438953472),
        (8, 68719476736),
    ],
)
def test_deepseek_v4_pro_0813_h20_simple_cpu_uses_local_device_count(
    monkeypatch,
    device_count,
    expected_per_rank_bytes,
):
    """节点容量按实际本机 worker 数均分，不依赖固定 TP/DP recipe。"""
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "512")
    params = {
        "engine": "vllm",
        "model_name": "DeepSeek-V4-Pro-0813",
        "model_path": "/models/DeepSeek-V4-Pro-0813",
        "_smart_card_token": "h20-96",
        "_smart_feats": ["offload"],
        "device_count": device_count,
        "engine_config": {"enable_prefix_caching": True},
    }

    config = vllm_adapter.resolve_topology_free_simple_cpu_offload_config(
        params,
        "vllm",
    )

    assert config is not None
    assert config["kv_connector_extra_config"]["cpu_bytes_to_use_per_rank"] == (
        expected_per_rank_bytes
    )


@pytest.mark.parametrize(
    ("card_name", "card_token"),
    [
        ("NVIDIA H20 96GB", "h20-96"),
        ("NVIDIA H20 141GB", "h20-141"),
    ],
)
@pytest.mark.parametrize(
    ("node_rank", "local_ip"),
    [(0, "7.6.25.59"), (1, "7.6.25.95")],
)
@pytest.mark.parametrize("nnodes", [2, 4])
def test_deepseek_v4_pro_0813_h20_production_config_chain(
    monkeypatch,
    tmp_path,
    card_name,
    card_token,
    node_rank,
    local_ip,
    nnodes,
):
    """从生产入口验证 H20 三项特性在双机和四机拓扑保持一致。"""
    for env_name in (
        "PD_ROLE",
        "CONFIG_FORCE",
        "TENSOR_PARALLEL_SIZE",
        "DATA_PARALLEL_SIZE",
        "PIPELINE_PARALLEL_SIZE",
        "MASTER_PORT",
        "VLLM_DISTRIBUTED_PORT",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "LMCACHE_OFFLOAD",
        "ENABLE_KV_DISK_OFFLOAD",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_SPARSE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "512")
    monkeypatch.setenv("NETWORK_INTERFACE", "bond0")
    monkeypatch.setenv("RANK_IP", local_ip)

    hardware_file = tmp_path / f"hardware_rank{node_rank}_{card_token}.json"
    hardware_file.write_text(
        json.dumps({
            "device": "nvidia",
            "count": 8,
            "details": [
                {
                    "device_id": index,
                    "name": card_name,
                    "total_memory": 96 if card_token == "h20-96" else 141,
                    "free_memory": 90 if card_token == "h20-96" else 135,
                    "used_memory": 6,
                }
                for index in range(8)
            ],
            "units": "GB",
        }),
        encoding="utf-8",
    )
    status_file = tmp_path / f"advanced_features_rank{node_rank}_{card_token}.json"
    monkeypatch.setenv("WINGS_HARDWARE_FILE", str(hardware_file))
    monkeypatch.setattr(wings_entry, "_ADVANCED_FEATURES_FILE", str(status_file))
    monkeypatch.setattr(config_loader, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.setattr(wings_entry, "ModelIdentifier", _FakeDeepSeekV4Identifier)
    monkeypatch.setattr(config_loader, "_check_vram_requirements", lambda *_args: None)
    monkeypatch.setattr(config_loader, "_record_selected_engine", lambda *_args: None)
    node_ips = ",".join(
        ["7.6.25.59", "7.6.25.95", "7.6.25.96", "7.6.25.97"][:nnodes]
    )

    launch_args = parse_launch_args([
        "--model-name", "DeepSeek-V4-Pro-0813",
        "--model-path", "/mnt/models/DeepSeek-V4-Pro-0813",
        "--model-type", "llm",
        "--engine", "vllm",
        "--device-count", "8",
        "--port", "8000",
        "--distributed",
        "--nnodes", str(nnodes),
        "--node-rank", str(node_rank),
        "--node-ips", node_ips,
        "--nodes", node_ips,
        "--master-ip", "7.6.25.59",
        "--head-node-addr", "7.6.25.59",
        "--enable-auto-tool-choice",
        "--enable-auto-think-choice",
        "--enable-speculative-decode",
        "--enable-sparse",
        "--speculative-decode-model-path", "none",
    ])
    plan = wings_entry.build_launcher_plan(
        launch_args,
        derive_port_plan(
            port=launch_args.port,
            enable_reason_proxy=False,
            health_port=19000,
        ),
    )
    params = plan.merged_params
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    shlex.split(exec_line, posix=True)
    prepared_config = vllm_adapter._prepare_engine_config(params)

    assert params["_smart_card_token"] == card_token
    assert params["distributed_executor_backend"] == "mp"
    expected_features = ["offload", "sparse", "spec"]
    assert params["_allowed_smart_feats"] == expected_features
    assert params["_smart_feats"] == expected_features
    assert params["engine_config"]["tensor_parallel_size"] == 8 * nnodes
    for key, expected in (
        ("max_model_len", 133000),
        ("gpu_memory_utilization", 0.92),
        ("max_num_seqs", 32),
        ("max_num_batched_tokens", 8192),
        ("enable_prefix_caching", True),
        ("enable_ep_weight_filter", True),
        ("cpu_distributed_timeout_seconds", 7200),
    ):
        # 这些值必须在 adapter 运行前就由 nvidia defaults 选配完成。
        assert params["engine_config"][key] == expected
        assert prepared_config[key] == expected
    assert json.loads(params["engine_config"]["kv_transfer_config"]) == {
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use_per_rank": 68719476736,
            "lazy_offload": False,
        },
    }

    assert "export VLLM_HOST_IP=${POD_IP:-${RANK_IP:-" in script
    assert os.environ["RANK_IP"] == local_ip
    assert "export NCCL_SOCKET_IFNAME=bond0" in script
    assert "export GLOO_SOCKET_IFNAME=bond0" in script
    for expected in (
        "--trust-remote-code",
        "--kv-cache-dtype fp8",
        "--block-size 256",
        "--enable-expert-parallel",
        f"--tensor-parallel-size {8 * nnodes}",
        f"--nnodes {nnodes}",
        f"--node-rank {node_rank}",
        "--master-addr 7.6.25.59",
        "--master-port 29501",
        "--max-model-len 133000",
        "--gpu-memory-utilization 0.92",
        "--max-num-seqs 32",
        "--max-num-batched-tokens 8192",
        "--enable-prefix-caching",
        "--enable-ep-weight-filter",
        "--cpu-distributed-timeout-seconds 7200",
        "--no-enable-flashinfer-autotune",
        "--disable-custom-all-reduce",
        "--tokenizer-mode deepseek_v4",
        "--distributed-executor-backend mp",
    ):
        assert expected in exec_line
    assert '"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"' in exec_line
    assert exec_line.count("--speculative-config") == 1
    assert '"method":"dspark"' in exec_line
    assert '"num_speculative_tokens":5' in exec_line
    assert '"draft_sample_method":"greedy"' in exec_line
    assert exec_line.count("--hf-overrides") == 1
    assert '"use_index_cache":true,"index_topk_freq":8' in exec_line
    assert exec_line.count("SimpleCPUOffloadConnector") == 1
    assert exec_line.count("68719476736") == 1
    assert ("--headless" in exec_line) is (node_rank != 0)
    # headless worker 沿用框架既有 API 参数裁剪；这些字段只保留在 rank0 API server。
    for api_flag in (
        "--host 7.6.25.59",
        "--port 8000",
        "--enable-auto-tool-choice",
        "--tool-call-parser deepseek_v4",
        "--reasoning-parser deepseek_v4",
    ):
        assert (api_flag in exec_line) is (node_rank == 0)

    status = json.loads(status_file.read_text(encoding="utf-8"))
    assert status["features"] == {
        "speculative_decode": True,
        "sparse_kv": True,
        "kv_offload": True,
        "rag_acc": False,
    }
    assert status["variants"] == {
        "speculative_decode": "dspark",
        "sparse_kv": "fp8_indexcache_use_index_cache_topk8",
        "kv_offload": "simple_cpu_offload_connector+custom",
    }
    assert status["others"]["kv_mem_offload_size"] == 512
    assert status["others"]["speculative_decode"] == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "moe_backend": None,
        "draft_sample_method": "greedy",
    }
    assert "SimpleCPUOffloadConnector" in plan.command
    assert "index_topk_freq" in plan.command
    primary_lines = [
        line.strip()
        for line in plan.command.splitlines()
        if line.strip().startswith("vllm serve ") and "--speculative-config" in line
    ]
    fallback_lines = [
        line.strip()
        for line in plan.command.splitlines()
        if line.strip().startswith("vllm serve ") and "--speculative-config" not in line
    ]
    assert primary_lines
    assert all("--kv-cache-dtype fp8" in line for line in primary_lines)
    assert fallback_lines
    assert all("--kv-cache-dtype fp8" not in line for line in fallback_lines)


@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
def test_kimi_k3_h20_simple_cpu_offload_final_command_matches_tuned_recipe(
    monkeypatch,
    card_token,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeKimiK3Identifier)
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth0")
    model_path = "/usr/local/serving/models"
    params = {
        "engine": "vllm",
        "model_name": "Kimi-K3",
        "model_path": model_path,
        "model_type": "llm",
        "device_count": 8,
        "distributed": True,
        "distributed_executor_backend": "mp",
        "nnodes": 4,
        "node_rank": 0,
        "master_ip": "10.254.114.238",
        "master_port": 29501,
        "enable_sparse": True,
        "enable_speculative_decode": True,
        "_smart_card_token": card_token,
        "engine_config": {
            "use_vllm_serve": True,
            "model": model_path,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.98,
            "tensor_parallel_size": 32,
            "no_enable_flashinfer_autotune": True,
            "extra_cli_args": ["-cc.pass_config.fuse_allreduce_rms=False"],
            "moe_backend": "marlin",
            "disable_custom_all_reduce": True,
            "distributed_timeout_seconds": 1200,
            "tool_call_parser": "kimi_k3",
            "enable_auto_tool_choice": True,
            "reasoning_parser": "kimi_k3",
            "served_model_name": "kimi_k3",
            "host": "0.0.0.0",
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 10,
            "max_model_len": 32768,
            "attention_backend": "FLASHMLA",
            "enable_prefix_caching": True,
            "port": 18000,
        },
    }

    card_name = "NVIDIA H20 96GB" if card_token == "h20-96" else "NVIDIA H20 141GB"
    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "nvidia", "count": 8, "details": [{"name": card_name}]},
    )
    assert params["_allowed_smart_feats"] == ["offload"]
    assert params["_smart_feats"] == ["offload"]
    assert params["enable_sparse"] is False
    assert params["enable_speculative_decode"] is False
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "false"
    assert os.environ["SD_ENABLE"] == "false"

    config_loader._set_kv_cache_config(
        params["engine_config"],
        params,
        _FakeKimiK3Identifier(params["model_name"], model_path, "llm"),
    )
    connector = json.loads(params["engine_config"]["kv_transfer_config"])
    assert connector == {
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use_per_rank": 5368709120,
            "lazy_offload": False,
        },
    }

    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert "export NCCL_SOCKET_IFNAME=eth0" in script
    assert "export GLOO_SOCKET_IFNAME=eth0" in script
    assert "export VLLM_ENGINE_READY_TIMEOUT_S=3600" in script
    assert "export VLLM_USE_V2_MODEL_RUNNER=1" in script
    assert "export VLLM_USE_RUST_FRONTEND=1" in script
    assert "unset PYTORCH_CUDA_ALLOC_CONF" not in script
    assert "ulimit -l unlimited" not in script
    assert exec_line.startswith(f"exec vllm serve {model_path} ")
    for expected in (
        "--trust-remote-code",
        "--gpu-memory-utilization 0.98",
        "--tensor-parallel-size 32",
        "--no-enable-flashinfer-autotune",
        "-cc.pass_config.fuse_allreduce_rms=False",
        "--moe-backend marlin",
        "--disable-custom-all-reduce",
        "--distributed-timeout-seconds 1200",
        "--max-num-batched-tokens 4096",
        "--max-num-seqs 10",
        "--max-model-len 32768",
        "--attention-backend FLASHMLA",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser kimi_k3",
        "--reasoning-parser kimi_k3",
        "--served-model-name kimi_k3",
        "--host 0.0.0.0",
        "--port 18000",
        "--distributed-executor-backend mp",
        "--nnodes 4",
        "--node-rank 0",
        "--master-addr 10.254.114.238",
        "--master-port 29501",
    ):
        assert expected in exec_line
    assert "--data-parallel-size" not in exec_line
    assert "--enable-expert-parallel" not in exec_line
    assert exec_line.count("SimpleCPUOffloadConnector") == 1
    assert exec_line.count("5368709120") == 1
    assert '"lazy_offload":false' in exec_line
    assert "--kv-offloading-backend" not in exec_line
    assert "LMCacheConnector" not in exec_line
    assert "LMCACHE_" not in script
    assert "--kv-cache-dtype" not in exec_line
    assert "--hf-overrides" not in exec_line
    assert "--speculative-config" not in exec_line
    assert "suffix" not in exec_line
    assert wings_entry._detect_offload_command_emitted(script) is True

    feature_status = wings_entry._resolve_advanced_feature_status("vllm", params)
    assert feature_status["features"]["kv_offload"] is True
    assert feature_status["variants"]["kv_offload"] == (
        "simple_cpu_offload_connector+custom"
    )
    assert feature_status["others"]["kv_mem_offload_size"] == 40


def test_kimi_k3_h20_non_tuned_topology_keeps_suffix_fallback(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    params = {
        "engine": "vllm",
        "model_name": "Kimi-K3",
        "model_path": "/models/Kimi-K3",
        "model_type": "llm",
        "device_count": 8,
        "distributed": True,
        "distributed_executor_backend": "mp",
        "nnodes": 2,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
    }

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "nvidia", "count": 8, "details": [{"name": "NVIDIA H20 141GB"}]},
    )

    assert params["enable_speculative_decode"] is True
    assert vllm_adapter.resolve_speculative_strategy(params, "vllm") == "suffix"
    assert '"method" : "suffix"' in vllm_adapter.build_speculative_cmd(params, "vllm")


def test_kimi_k3_w4a8_910c_suppresses_suffix_without_offload(monkeypatch):
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "false")
    params = {
        "engine": "vllm_ascend",
        "model_name": "Kimi-K3-w4a8",
        "model_path": "/data/Kimi-K3-w4a8",
        "model_type": "llm",
        "device_count": 16,
        "distributed": True,
        "distributed_executor_backend": "dp_deployment",
        "nnodes": 4,
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_kimi_k3_910c_dp": True,
    }

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "ascend", "count": 16, "details": [{"name": "Ascend910C"}]},
    )

    assert params["_allowed_smart_feats"] == ["offload"]
    assert params["_smart_feats"] == []
    assert params["enable_speculative_decode"] is False
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "false"
    assert os.environ["SD_ENABLE"] == "false"
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == ""


def test_kimi_k3_w4a8_910c_removes_explicit_suffix_config(monkeypatch):
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    params = {
        "engine": "vllm_ascend",
        "model_name": "Kimi-K3-w4a8",
        "model_path": "/data/Kimi-K3-w4a8",
        "model_type": "llm",
        "device_count": 16,
        "distributed": True,
        "distributed_executor_backend": "dp_deployment",
        "nnodes": 4,
        "enable_speculative_decode": True,
        "_smart_card_token": "910c",
        "_smart_feats": ["spec"],
        "_kimi_k3_910c_dp": True,
        "engine_config": {
            "tensor_parallel_size": 16,
            "speculative_config": {"method": "suffix", "num_speculative_tokens": 5},
        },
    }

    engine_config = vllm_adapter._prepare_engine_config(params)

    assert "speculative_config" not in engine_config
    assert "speculative_config" not in params["engine_config"]
    assert params["enable_speculative_decode"] is False
    assert params["_smart_feats"] == []
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "false"
    assert os.environ["SD_ENABLE"] == "false"


@pytest.mark.parametrize(
    ("param_path", "invalid_value"),
    [
        (("nnodes",), 2),
        (("engine_config", "data_parallel_size"), 2),
        (("engine_config", "tensor_parallel_size"), 4),
        (("engine_config", "enable_prefix_caching"), False),
    ],
)
def test_kimi_k3_h20_simple_cpu_offload_rejects_other_runtime_scopes(
    monkeypatch,
    param_path,
    invalid_value,
):
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    params = {
        "engine": "vllm",
        "model_name": "Kimi-K3",
        "model_path": "/models/Kimi-K3",
        "model_type": "llm",
        "device_count": 8,
        "distributed": True,
        "nnodes": 4,
        "_smart_card_token": "h20-141",
        "_smart_feats": ["offload"],
        "engine_config": {
            "tensor_parallel_size": 32,
        },
    }
    target = params
    for key in param_path[:-1]:
        target = target[key]
    target[param_path[-1]] = invalid_value

    assert vllm_adapter.resolve_kimi_k3_h20_simple_cpu_config(
        params,
        "vllm",
    ) is None
    assert vllm_adapter.resolve_kv_offload_effective_state(params, "vllm") == (
        False,
        "disabled",
    )
    # Kimi 固定配方失配后不能回落到 topology-free resolver。
    assert vllm_adapter.resolve_simple_cpu_offload_config(params, "vllm") is None


@pytest.mark.parametrize("card_token", ["h20-96", "h20-141"])
def test_kimi_k3_h20_simple_cpu_offload_resolves_auto_size(monkeypatch, card_token):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeKimiK3Identifier)
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", str(652 * 1024))
    params = {
        "engine": "vllm",
        "model_name": "Kimi-K3",
        "model_path": "/models/Kimi-K3",
        "model_type": "llm",
        "device_count": 8,
        "distributed": True,
        "distributed_executor_backend": "mp",
        "nnodes": 4,
        "node_rank": 0,
        "master_ip": "7.6.25.57",
        "master_port": 29501,
        "enable_sparse": False,
        "enable_speculative_decode": False,
        "engine_config": {
            "use_vllm_serve": True,
            "model": "/models/Kimi-K3",
            "tensor_parallel_size": 32,
        },
    }

    card_name = "NVIDIA H20 96GB" if card_token == "h20-96" else "NVIDIA H20 141GB"
    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "nvidia", "count": 8, "details": [{"name": card_name}]},
    )
    assert params["_allowed_smart_feats"] == ["offload"]
    assert params["_smart_feats"] == ["offload"]

    config = vllm_adapter.resolve_kimi_k3_h20_simple_cpu_config(
        params,
        "vllm",
    )
    # 652GiB 扣除 TP8×DP4 引擎预留和 10% 安全垫后为 359GiB，
    # SimpleCPU 再按本地 8 rank 向下对齐为 352GiB，每 rank 44GiB。
    assert config == {
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "cpu_bytes_to_use_per_rank": 44 * 1024 ** 3,
            "lazy_offload": False,
        },
    }
    config_loader._set_kv_cache_config(
        params["engine_config"],
        params,
        _FakeKimiK3Identifier(params["model_name"], params["model_path"], "llm"),
    )
    # 真实 launcher 的最终参数会携带该默认值；它不能覆盖已经终定的引擎语义。
    params["enable_prefix_caching"] = False
    config_loader._enforce_simple_cpu_offload_kv_transfer_config(
        params["engine_config"],
        params,
    )
    assert json.loads(params["engine_config"]["kv_transfer_config"]) == config
    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    assert "--kv-transfer-config" in exec_line
    assert "47244640256" in exec_line
    assert '"lazy_offload":false' in exec_line
    assert vllm_adapter.resolve_kv_offload_effective_state(params, "vllm") == (
        True,
        "simple_cpu_offload_connector+custom",
    )
    assert vllm_adapter.resolve_effective_kv_mem_offload_size(params, "vllm") == 352


def test_kimi_k3_h20_simple_cpu_offload_rejects_auto_without_capacity(monkeypatch):
    monkeypatch.delenv("PD_ROLE", raising=False)
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.delenv("AVAILABLE_POD_MEM_SIZE", raising=False)
    params = {
        "engine": "vllm",
        "model_name": "Kimi-K3",
        "model_path": "/models/Kimi-K3",
        "model_type": "llm",
        "device_count": 8,
        "distributed": True,
        "nnodes": 4,
        "_smart_card_token": "h20-141",
        "_smart_feats": ["offload"],
        "engine_config": {
            "tensor_parallel_size": 32,
        },
    }

    assert vllm_adapter.resolve_kimi_k3_h20_simple_cpu_config(
        params,
        "vllm",
    ) is None


def test_kimi_k3_h20_simple_cpu_offload_preserves_pd_connector(
    monkeypatch,
):
    monkeypatch.setenv("PD_ROLE", "P")
    monkeypatch.setenv("LMCACHE_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    ctx = {
        "engine": "vllm",
        "device": "nvidia",
        "model_name": "Kimi-K3",
        "model_path": "/models/Kimi-K3",
        "model_type": "llm",
        "device_count": 8,
        "distributed": True,
        "nnodes": 4,
        "_smart_card_token": "h20-141",
        "_smart_feats": ["offload"],
    }
    engine_config = {
        "tensor_parallel_size": 32,
    }

    config_loader._set_kv_cache_config(engine_config, ctx)
    pd_config = json.loads(engine_config["kv_transfer_config"])
    assert pd_config == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
    }
    config_loader._enforce_simple_cpu_offload_kv_transfer_config(
        engine_config,
        ctx,
    )
    assert json.loads(engine_config["kv_transfer_config"]) == pd_config
    assert vllm_adapter.resolve_kv_offload_effective_state(ctx, "vllm") == (
        False,
        "disabled",
    )


def test_qwen35_nvfp4_native_offload_keeps_mtp_strategy(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeModelIdentifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    strategy = vllm_adapter.resolve_speculative_strategy(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "model_type": "llm",
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert strategy == "mtp"


def test_qwen35_nvfp4_pro5000_speculative_config_matches_tokenbox(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)

    command = vllm_adapter.build_speculative_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "_smart_feats": ["spec", "offload"],
            "_smart_card_token": "rtxpro5000-72",
        },
        "vllm",
    )

    assert command == ' --speculative-config \'{"method":"mtp","num_speculative_tokens":3}\''


@pytest.mark.parametrize(
    ("model_name", "model_path", "expected_tokens"),
    [
        ("Qwen/Qwen3.5-122B-A10B", "/models/Qwen/Qwen3.5-122B-A10B", 1),
        ("Qwen/Qwen3.5-27B", "/models/Qwen/Qwen3.5-27B", 2),
    ],
)
def test_qwen35_pro5000_mtp_uses_whitelist_method_and_tokens(
    monkeypatch,
    model_name,
    model_path,
    expected_tokens,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)

    params = {
        "engine": "vllm",
        "model_name": model_name,
        "model_path": model_path,
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec", "offload"],
        "_smart_card_token": "rtxpro5000-72",
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert command == (
        " --speculative-config "
        f"'{{\"method\":\"mtp\",\"num_speculative_tokens\":{expected_tokens}}}'"
    )
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "mtp",
        "num_speculative_tokens": expected_tokens,
        "moe_backend": None,
    }


@pytest.mark.parametrize(
    ("model_name", "model_path", "expected_tokens"),
    [
        ("MiniMax/MiniMax-M3-MXFP8", "/models/MiniMax/MiniMax-M3-MXFP8", 32),
        ("MiniMax/MiniMax-M2.5-NVFP4", "/models/MiniMax/MiniMax-M2.5-NVFP4", 10),
        ("MiniMax/MiniMax-M2.7-NVFP4", "/models/MiniMax/MiniMax-M2.7-NVFP4", 10),
    ],
)
def test_minimax_pro5000_suffix_uses_whitelist_tokens(
    monkeypatch,
    model_name,
    model_path,
    expected_tokens,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)

    params = {
        "engine": "vllm",
        "model_name": model_name,
        "model_path": model_path,
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec", "offload"],
        "_smart_card_token": "rtxpro5000-72",
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert command == (
        " --speculative-config "
        f"'{{\"method\":\"suffix\",\"num_speculative_tokens\":{expected_tokens}}}'"
    )
    assert "suffix_decoding_max_cached_requests" not in command
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "suffix",
        "num_speculative_tokens": expected_tokens,
        "moe_backend": None,
    }


def test_qwen_agentworld_pro5000_suffix_uses_whitelist_32_tokens(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)

    params = {
        "engine": "vllm",
        "model_name": "Qwen/Qwen-AgentWorld-35B-A3B",
        "model_path": "/models/Qwen/Qwen-AgentWorld-35B-A3B",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec"],
        "_smart_card_token": "rtxpro5000-72",
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm") == "suffix"
    assert command == (
        " --speculative-config "
        "'{\"method\":\"suffix\",\"num_speculative_tokens\":32}'"
    )
    assert "suffix_decoding_max_cached_requests" not in command
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "suffix",
        "num_speculative_tokens": 32,
        "moe_backend": None,
    }


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_minimax_m25_w8a8_quarot_eagle3_reuses_draft_path(monkeypatch, card_token):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Identifier)
    draft_path = "/path/to/weight/Eagle3/"
    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/MiniMax-M2.5-w8a8-QuaRot",
        "model_path": "/models/Eco-Tech/MiniMax-M2.5-w8a8-QuaRot",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": draft_path,
        "_smart_feats": ["spec"],
        "_smart_card_token": card_token,
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")
    config = json.loads(command.split("'", 2)[1])

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "eagle3"
    assert config == {
        "method": "eagle3",
        "model": draft_path,
        "num_speculative_tokens": 3,
        "enforce_eager": True,
    }


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_minimax_m25_w8a8_quarot_without_eagle_uses_suffix3(monkeypatch, card_token):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Identifier)
    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/MiniMax-M2.5-w8a8-QuaRot",
        "model_path": "/models/Eco-Tech/MiniMax-M2.5-w8a8-QuaRot",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec"],
        "_smart_card_token": card_token,
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")
    config = json.loads(command.split("'", 2)[1])

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    assert config == {
        "method": "suffix",
        "num_speculative_tokens": 3,
        "suffix_decoding_max_cached_requests": 1000,
    }


@pytest.mark.parametrize("card_token", ["910b", "910c"])
def test_minimax_m27_w8a8_quarot_eagle3_uses_draft_path_options(
    monkeypatch,
    tmp_path,
    card_token,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Identifier)
    draft_dir = tmp_path / "Eagle3"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(
        json.dumps({"architectures": ["MiniMaxM2Eagle3ForCausalLM"]}),
        encoding="utf-8",
    )
    params = {
        "engine": "vllm_ascend",
        "model_name": "MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_path": "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": str(draft_dir),
        "_smart_feats": ["spec"],
        "_smart_card_token": card_token,
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "eagle3"
    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")
    body = command.split("'", 2)[1]
    config = json.loads(body)

    assert config["method"] == "eagle3"
    assert config["model"] == str(draft_dir)
    assert "draft_tensor_parallel_size" not in config
    assert config["num_speculative_tokens"] == 3
    assert config["enforce_eager"] is True


def test_minimax_m27_w8a8_quarot_910b_env_matches_day0_script_without_deploy_pinning(
    monkeypatch,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Identifier)
    params = {
        "engine": "vllm_ascend",
        "model_name": "MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_path": "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_type": "llm",
        "device_count": 8,
        "nnodes": 1,
        "device_details": [{"name": "Ascend910B_64G"}],
    }

    commands = vllm_adapter._build_vllm_common_env_cmds(params, "vllm_ascend")
    export_by_name = _export_by_name(commands)

    assert "ASCEND_RT_VISIBLE_DEVICES" not in export_by_name
    assert "MOONCAKE_CONFIG_PATH" not in export_by_name
    assert export_by_name["VLLM_LOG_LEVEL"] == "export VLLM_LOG_LEVEL=DEBUG"
    assert export_by_name["HCCL_OP_EXPANSION_MODE"] == 'export HCCL_OP_EXPANSION_MODE="AIV"'
    assert export_by_name["HCCL_BUFFSIZE"] == "export HCCL_BUFFSIZE=1024"
    assert export_by_name["PYTORCH_NPU_ALLOC_CONF"] == (
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
    )
    assert export_by_name["OMP_NUM_THREADS"] == "export OMP_NUM_THREADS=1"
    assert export_by_name["LD_PRELOAD"] == (
        "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
    )
    assert export_by_name["TASK_QUEUE_ENABLE"] == "export TASK_QUEUE_ENABLE=1"
    assert export_by_name["ASCEND_BUFFER_POOL"] == "export ASCEND_BUFFER_POOL=4:8"
    assert export_by_name["VLLM_ASCEND_BALANCE_SCHEDULING"] == (
        "export VLLM_ASCEND_BALANCE_SCHEDULING=0"
    )
    assert export_by_name["LD_LIBRARY_PATH"] == (
        "export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:"
        "/usr/local/lib:${LD_LIBRARY_PATH:-}"
    )
    assert export_by_name["PYTHONHASHSEED"] == "export PYTHONHASHSEED=0"
    assert export_by_name["ACL_OP_INIT_MODE"] == "export ACL_OP_INIT_MODE=1"

    for removed_name in (
        "OMP_PROC_BIND",
        "VLLM_USE_GRAPH",
        "VLLM_USE_V1",
        "VLLM_ASCEND_ENABLE_FUSED_MC2",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1",
        "VLLM_TORCH_COMPILE",
    ):
        assert removed_name not in export_by_name


def test_minimax_m27_w8a8_quarot_910b_startup_includes_top_level_eager(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Identifier)
    defaults_path = (
        Path(__file__).resolve().parents[1]
        / "wings_control"
        / "config"
        / "defaults"
        / "ascend_default.json"
    )
    profile = json.loads(defaults_path.read_text(encoding="utf-8"))[
        "model_deploy_config"
    ]["llm"]["MiniMaxM2ForCausalLM"]["MiniMax-M2.7-w8a8-QuaRot-Ascend910B"][
        "vllm_ascend"
    ]
    draft_dir = tmp_path / "Eagle3"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(
        json.dumps({"architectures": ["MiniMaxM2Eagle3ForCausalLM"]}),
        encoding="utf-8",
    )
    params = {
        "engine": "vllm_ascend",
        "model_name": "MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_path": "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_type": "llm",
        "device_count": 8,
        "nnodes": 1,
        "device_details": [{"name": "Ascend910B_64G"}],
        "enable_speculative_decode": True,
        "speculative_decode_model_path": str(draft_dir),
        "_smart_feats": ["spec"],
        "_smart_card_token": "910b",
        "engine_config": json.loads(json.dumps(profile)),
        "_explicit_cli_keys": set(),
    }

    script = vllm_adapter.build_start_script(params)
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    assert " --enforce-eager " in f" {exec_line} "
    assert '"enforce_eager": true' in exec_line
    assert "draft_tensor_parallel_size" not in exec_line


def test_minimax_m27_w8a8_quarot_without_draft_falls_back_to_suffix(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_path": "/models/MiniMax/MiniMax-M2.7-w8a8-QuaRot",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["spec"],
        "_smart_card_token": "910c",
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    assert '"method" : "suffix"' in command
    assert "eagle3" not in command
    assert "draft_model" not in command


def test_qwen35_35b_pro5000_offload_only_does_not_auto_append_spec(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)
    params = {
        "engine": "vllm",
        "model_name": "Qwen/Qwen3.5-35B-A3B",
        "model_path": "/models/Qwen/Qwen3.5-35B-A3B",
        "model_type": "llm",
        "enable_speculative_decode": False,
        "_smart_feats": ["offload"],
        "_smart_card_token": "rtxpro5000-72",
    }

    script = vllm_adapter._build_vllm_single_script(
        params,
        "vllm serve /models/Qwen/Qwen3.5-35B-A3B",
        [],
        "vllm",
        "",
    )

    assert vllm_adapter.should_append_auto_speculative_config(params) is False
    assert "--speculative-config" not in script


def test_nvidia_day0_glm47_mtp_uses_whitelist_method_and_ignores_draft(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm47Identifier)

    params = {
        "engine": "vllm",
        "model_name": "GLM-4.7-FP8",
        "model_path": "/models/zai-org/GLM-4.7",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "/models/old-draft",
        "_smart_card_token": "h20-141",
        "_smart_feats": ["spec"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm") == "mtp"
    command = vllm_adapter.build_speculative_cmd(params, "vllm")
    assert command == " --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'"
    assert "old-draft" not in command


def test_ascend_glm47_mtp_only_does_not_fall_back_to_suffix(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm47Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/GLM-4.7-w8a8-floatmtp",
        "model_path": "/models/Eco-Tech/GLM-4.7-w8a8-floatmtp",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "_smart_card_token": "910c",
        "_smart_feats": ["spec"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "mtp"
    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")
    assert command == (
        " --speculative-config "
        "'{\"method\":\"mtp\",\"num_speculative_tokens\":3,"
        "\"speculative_token_range\":\"256,512\"}'"
    )


def test_ascend_deepseek_v4_pro_mtp_uses_exact_whitelist_enforce_eager(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-V4-Pro-w4a8-mtp",
        "model_path": "/models/DeepSeek-V4-Pro-w4a8-mtp",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "_smart_card_token": "910c",
        "_smart_feats": ["spec", "sparse"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "mtp"
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == (
        " --speculative-config "
        "'{\"method\":\"mtp\",\"num_speculative_tokens\":1,\"enforce_eager\":true}'"
    )
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm_ascend") == {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "moe_backend": None,
        "enforce_eager": True,
    }


def test_kimi_k26_uses_suffix_fallback_without_dflash_draft(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeKimiIdentifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/Kimi-K2.6-W4A8",
        "model_path": "/models/Eco-Tech/Kimi-K2.6-W4A8",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_card_token": "910c",
        "_smart_feats": ["spec", "offload"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == (
        " --speculative-config "
        "'{\"method\" : \"suffix\", \"num_speculative_tokens\": 5, "
        "\"suffix_decoding_max_cached_requests\": 1000}'"
    )

    params["speculative_decode_model_path"] = "z-lab/Kimi-K2.6-NonDFlash-Draft"
    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    assert "draft_model" not in vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    params["speculative_decode_model_path"] = "z-lab/Kimi-K2.6-DFlash"
    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "dflash"
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == (
        " --speculative-config "
        "'{\"method\":\"dflash\",\"model\":\"z-lab/Kimi-K2.6-DFlash\","
        "\"num_speculative_tokens\":15}'"
    )

    params["_smart_feats"] = ["offload"]
    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == ""
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == ""


def test_kimi_k27_code_does_not_inherit_k26_dflash(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeKimiIdentifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "Kimi-K2.7-Code",
        "model_path": "/harbor_data/Kimi-K2.7-Code",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "z-lab/Kimi-K2.6-DFlash",
        "_smart_card_token": "910c",
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == ""
    assert vllm_adapter.build_speculative_cmd(params, "vllm_ascend") == ""


def test_nvidia_day0_qwen35_mtp_uses_whitelist_moe_backend(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen36MoeIdentifier)

    params = {
        "engine": "vllm",
        "model_name": "Qwen3.6-35B-A3B",
        "model_path": "/models/Qwen3.6-35B-A3B",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "_smart_card_token": "l20",
        "_smart_feats": ["spec"],
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm")

    assert command == (
        " --speculative-config "
        "'{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"moe_backend\":\"triton\"}'"
    )
    assert vllm_adapter.resolve_effective_speculative_details(params, "vllm") == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "moe_backend": "triton",
    }


def test_nvidia_day0_native_offload_uses_whitelist_backend(monkeypatch):
    monkeypatch.delenv("CONFIG_FORCE", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "20")
    params = {
        "engine": "vllm",
        "model_name": "Qwen3.6-27B",
        "model_path": "/models/Qwen3.6-27B",
        "model_type": "llm",
        "_smart_card_token": "l20",
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter._build_kv_offload_cmd(params, "vllm") == (
        " --kv-offloading-backend native --kv-offloading-size 20"
    )
    assert vllm_adapter.resolve_offload_variant(params, "vllm") == (
        "native_kv_offloading_backend"
    )
    assert vllm_adapter.resolve_effective_kv_mem_offload_size(params, "vllm") == 20


@pytest.mark.parametrize("raw_size", ["", "invalid", "0", "-1"])
def test_nvidia_day0_native_offload_discards_invalid_page_size(monkeypatch, raw_size):
    monkeypatch.delenv("CONFIG_FORCE", raising=False)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", raw_size)
    params = {
        "engine": "vllm",
        "model_name": "Qwen3.6-27B",
        "model_path": "/models/Qwen3.6-27B",
        "model_type": "llm",
        "_smart_card_token": "l20",
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter._build_kv_offload_cmd(params, "vllm") == ""
    assert vllm_adapter.resolve_offload_variant(params, "vllm") == "disabled"
    assert vllm_adapter.resolve_effective_kv_mem_offload_size(params, "vllm") is None


def test_nvidia_day0_native_offload_config_force_bypasses_whitelist(monkeypatch):
    monkeypatch.setenv("CONFIG_FORCE", "true")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "20")
    params = {
        "engine": "vllm",
        "model_name": "Qwen3.6-27B",
        "model_path": "/models/Qwen3.6-27B",
        "model_type": "llm",
        "_smart_card_token": "l20",
        "_smart_feats": ["offload"],
    }

    assert vllm_adapter._build_kv_offload_cmd(params, "vllm") == ""


def test_advanced_feature_fallback_removes_embedded_speculative_config(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)
    monkeypatch.setattr(wings_entry, "ModelIdentifier", _FakePro5000Identifier)
    monkeypatch.setattr(
        wings_entry,
        "start_engine_service",
        lambda merged: vllm_adapter.build_start_script(merged),
    )
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    merged = {
        "engine": "vllm",
        "model_name": "Qwen3.5-397B-A17B-NVFP4",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "enable_sparse": False,
        "_smart_feats": ["offload", "spec"],
        "engine_config": {
            "model": "/usr/local/serving/models/",
            "served_model_name": "Qwen3.5-397B-A17B-NVFP4",
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
            "tensor_parallel_size": 8,
        },
    }

    fallback_cmd = wings_entry._build_advanced_feature_fallback_cmd(merged)

    assert "--speculative-config" not in fallback_cmd
    assert "--kv-offloading-backend" not in fallback_cmd
    assert merged["engine_config"]["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }


def test_pro5000_spec_models_preserve_registered_accel_install_scenes(
    monkeypatch,
):
    # 这个测试同时保护两条边界：
    # - 旧的 EARS/install-runtime-deps 补丁不能借 spec 场景回流；
    # - 原有 DeepSeek+Pro5000 与新增 effective native offload 都能独立安装依赖。
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakePro5000Identifier)
    monkeypatch.setattr(wings_entry, "ModelIdentifier", _FakePro5000Identifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    monkeypatch.setenv("ENGINE_VERSION", "v0.23.0")

    scenarios = [
        {
            "engine": "vllm",
            "model_name": "deepseek-ai/DeepSeek-V4-Flash",
            "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "enable_sparse": True,
            "speculative_decode_model_path": "none",
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "sparse"],
            "installs_nvidia_packages": True,
        },
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "enable_sparse": False,
            "speculative_decode_model_path": "none",
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
            "installs_nvidia_packages": True,
        },
    ]

    for params in scenarios:
        env_commands = vllm_adapter._build_speculative_env_commands(params, "vllm")
        accel_preamble = wings_entry._build_accel_preamble("vllm", params)

        assert env_commands == []
        assert "install-runtime-deps" not in accel_preamble
        assert '"ears"' not in accel_preamble
        assert "VLLM_EARS_TOLERANCE" not in accel_preamble
        if params["installs_nvidia_packages"]:
            assert "python3 install.py --config" in accel_preamble
            assert "deepgemm:nv_dev_a6b593d" in accel_preamble
            assert "flashinfer:v0.6.12" in accel_preamble
        else:
            assert "install.py" not in accel_preamble


def test_spec_request_without_whitelist_generates_suffix_config(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": [],
    }

    assert vllm_adapter.should_append_auto_speculative_config(params) is True

    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    assert '"method" : "suffix"' in command
    assert '"num_speculative_tokens": 5' in command


def test_async_scheduling_auto_suffix_disables_spec_and_keeps_scheduling(
    monkeypatch, caplog,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("SD_ENABLE", "true")

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": [],
        "engine_config": {
            "async_scheduling": True,
            "scheduling_policy": "priority",
        },
    }

    with caplog.at_level(logging.INFO):
        script = vllm_adapter.build_start_script(params)

    assert "--async-scheduling" in script
    assert "--scheduling-policy priority" in script
    assert "--speculative-config" not in script
    assert params["enable_speculative_decode"] is False
    assert params["_smart_feats"] == []
    assert params["engine_config"]["async_scheduling"] is True
    assert "speculative_config" not in params["engine_config"]
    assert "Keeping scheduling enabled, disabling speculative decoding" in caplog.text
    assert "Async/suffix conflict guard applied" in caplog.text
    assert os.environ["ENABLE_SPECULATIVE_DECODE"] == "false"
    assert os.environ["SD_ENABLE"] == "false"


def test_async_scheduling_explicit_suffix_config_is_removed(monkeypatch, caplog):
    monkeypatch.setenv("ENABLE_SPECULATIVE_DECODE", "true")
    monkeypatch.setenv("SD_ENABLE", "true")
    params = {
        "engine": "vllm_ascend",
        "model_name": "Qwen3.6-27B-w8a8",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "_smart_feats": ["spec"],
        "engine_config": {
            "async_scheduling": True,
            "speculative_config": {
                "method": "suffix",
                "num_speculative_tokens": 5,
            },
        },
    }

    with caplog.at_level(logging.INFO):
        engine_config = vllm_adapter._prepare_engine_config(params)

    assert engine_config["async_scheduling"] is True
    assert "speculative_config" not in engine_config
    assert "speculative_config" not in params["engine_config"]
    assert params["enable_speculative_decode"] is False
    assert params["_smart_feats"] == []
    assert "source=engine_config.speculative_config" in caplog.text
    assert "removed_speculative_config" in caplog.text


def test_async_scheduling_mtp_speculative_config_is_kept():
    params = {
        "engine": "vllm_ascend",
        "model_name": "Qwen3.6-27B-w8a8",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "_smart_feats": ["spec"],
        "engine_config": {
            "async_scheduling": True,
            "speculative_config": {
                "method": "qwen3_5_mtp",
                "num_speculative_tokens": 3,
            },
        },
    }

    engine_config = vllm_adapter._prepare_engine_config(params)

    assert engine_config["speculative_config"] == {
        "method": "qwen3_5_mtp",
        "num_speculative_tokens": 3,
    }
    assert params["enable_speculative_decode"] is True
    assert params["_smart_feats"] == ["spec"]


def test_async_suffix_veto_updates_startup_accel_status(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)
    monkeypatch.setattr(
        wings_entry,
        "_ADVANCED_FEATURES_FILE",
        str(tmp_path / "advanced_features.json"),
    )
    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": [],
        "engine_config": {
            "async_scheduling": "true",
        },
    }

    vllm_adapter.prepare_params_for_startup_status(params)
    wings_entry._write_advanced_features_json("vllm_ascend", params)

    data = json.loads((tmp_path / "advanced_features.json").read_text(encoding="utf-8"))
    assert data["features"]["speculative_decode"] is False
    assert data["variants"]["speculative_decode"] is None
    assert data["others"]["speculative_decode"] is None
    assert wings_entry._collect_active_feature_names(params, "vllm_ascend") == []


@pytest.mark.parametrize("draft_path", ["none", "None", " none ", '"none"', "'none'", "null"])
def test_spec_draft_sentinel_values_do_not_generate_draft_model(monkeypatch, draft_path):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": draft_path,
        "_smart_feats": [],
    }

    assert vllm_adapter.resolve_speculative_strategy(params, "vllm_ascend") == "suffix"
    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    assert '"method" : "suffix"' in command
    assert '"method" : "draft_model"' not in command
    assert '"model"' not in command


def test_glm51_ascend_spec_whitelist_uses_native_mtp(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    params = {
        "engine": "vllm_ascend",
        "model_name": "GLM-5.1-w8a8",
        "model_path": "/models/GLM-5.1-w8a8",
        "model_type": "llm",
        "enable_speculative_decode": True,
        "speculative_decode_model_path": "none",
        "_smart_feats": ["sparse", "spec"],
    }

    command = vllm_adapter.build_speculative_cmd(params, "vllm_ascend")

    assert '"method": "deepseek_mtp"' in command
    assert '"num_speculative_tokens": 3' in command
    assert "suffix" not in command


@pytest.mark.parametrize(
    ("identifier_cls", "expected_strategy"),
    [
        (_FakeGlm51Identifier, "deepseek_mtp"),
        (_FakeGlm47Identifier, "glm4_moe_mtp"),
    ],
)
def test_auto_floor_discarded_offload_keeps_mtp_strategy(
    monkeypatch,
    identifier_cls,
    expected_strategy,
):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", identifier_cls)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")

    strategy = vllm_adapter.resolve_speculative_strategy(
        {
            "engine": "vllm_ascend",
            "model_name": "GLM-5.1-w8a8",
            "model_path": "/models/GLM-5.1-w8a8",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["offload", "spec"],
        },
        "vllm_ascend",
    )

    assert strategy == expected_strategy


def test_auto_floor_with_disk_offload_still_uses_suffix_guard(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "true")
    monkeypatch.setenv("KV_DISK_OFFLOAD_PATH", "/mnt/kvcache_offload")
    monkeypatch.setenv("KV_DISK_OFFLOAD_SIZE", "8")

    strategy = vllm_adapter.resolve_speculative_strategy(
        {
            "engine": "vllm_ascend",
            "model_name": "GLM-5.1-w8a8",
            "model_path": "/models/GLM-5.1-w8a8",
            "model_type": "llm",
            "enable_speculative_decode": True,
            "speculative_decode_model_path": "none",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["offload", "spec"],
        },
        "vllm_ascend",
    )

    assert strategy == "suffix"


def test_glm51_roce_distributed_engine_config_uses_official_mtp_num3(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)
    monkeypatch.setattr(vllm_adapter, "_is_roce_distributed", lambda: True)

    params = {
        "engine": "vllm_ascend",
        "model_name": "GLM-5.1-w8a8",
        "model_path": "/models/GLM-5.1-w8a8",
        "model_type": "llm",
        "distributed": True,
        "distributed_executor_backend": "dp_deployment",
        "enable_speculative_decode": True,
        "engine_config": {
            "async_scheduling": True,
            "enable_expert_parallel": True,
        },
    }

    engine_config = vllm_adapter._prepare_engine_config(params)

    assert engine_config["speculative_config"] == {
        "num_speculative_tokens": 3,
        "method": "deepseek_mtp",
    }
    assert params["engine_config"]["speculative_config"] == {
        "num_speculative_tokens": 3,
        "method": "deepseek_mtp",
    }
    assert "async_scheduling" not in engine_config
    assert "enable_expert_parallel" not in engine_config


def test_glm51_ascend_indexcache_hf_overrides_enable_index_cache(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)

    command = vllm_adapter._build_kv_sparse_cmd(
        {
            "engine": "vllm_ascend",
            "model_name": "glm-5.1-32b-chat",
            "model_path": "/models/glm-5.1-32b-chat",
            "model_type": "llm",
            "_smart_feats": ["sparse"],
        },
        "vllm_ascend",
    )

    assert '--hf-overrides' in command
    assert '"use_index_cache": true' in command
    assert '"index_topk_freq": 8' in command

    variant = vllm_adapter.resolve_sparse_variant(
        {
            "engine": "vllm_ascend",
            "model_name": "glm-5.1-32b-chat",
            "model_path": "/models/glm-5.1-32b-chat",
            "model_type": "llm",
        },
        "vllm_ascend",
    )

    assert variant == "indexcache_use_index_cache_topk8"


def test_glm51_ascend_indexcache_uses_model_name_when_config_missing(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeUnknownGlm51Identifier)

    params = {
        "engine": "vllm_ascend",
        "model_name": "GLM-5.1-w8a8",
        "model_path": "/usr/local/serving/models/",
        "model_type": "llm",
        "_smart_feats": ["sparse"],
    }

    command = vllm_adapter._build_kv_sparse_cmd(params, "vllm_ascend")

    assert '--hf-overrides' in command
    assert '"use_index_cache": true' in command
    assert '"index_topk_freq": 8' in command

    variant = vllm_adapter.resolve_sparse_variant(params, "vllm_ascend")

    assert variant == "indexcache_use_index_cache_topk8"


def test_sparse_whitelist_command_and_variant_share_plan(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)
    monkeypatch.setattr(
        vllm_adapter,
        "resolve_feature_whitelist_row_from_params",
        lambda *_args: {"strategy": "indexcache"},
    )
    monkeypatch.setattr(vllm_adapter, "_resolve_sparse_topk", lambda *_args, **_kwargs: 7)
    params = {
        "model_name": "Qwen2",
        "model_path": "/models/Qwen2",
        "model_type": "llm",
    }

    command = vllm_adapter._build_kv_sparse_cmd(params, "vllm")
    variant = vllm_adapter.resolve_sparse_variant(params, "vllm")

    assert command == ' --hf-overrides \'{"use_index_cache":true,"index_topk_freq":7}\''
    assert variant == "indexcache_use_index_cache_topk7"


def test_sparse_fp8_variant_is_pure_while_command_applies_shared_plan(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen2Identifier)
    monkeypatch.setattr(vllm_adapter, "INDEXCACHE_ARCHS", set())
    monkeypatch.setattr(
        vllm_adapter,
        "resolve_feature_whitelist_row_from_params",
        lambda *_args: None,
    )
    monkeypatch.setattr(vllm_adapter, "_is_deepseek_v4_flash_params", lambda *_args: False)
    monkeypatch.setattr(
        vllm_adapter,
        "is_minimax_m27_rtx_pro_5000_vllm",
        lambda *_args: False,
    )
    params = {
        "model_name": "Qwen2",
        "model_path": "/models/Qwen2",
        "model_type": "llm",
        "engine_config": {},
    }

    assert vllm_adapter.resolve_sparse_variant(params, "vllm") == "fp8"
    assert params["engine_config"] == {}
    assert vllm_adapter._build_kv_sparse_cmd(params, "vllm") == ""
    assert params["engine_config"] == {
        "kv_cache_dtype": "fp8",
    }


def test_glm51_ascend_preserves_explicit_expert_parallel_in_command(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeGlm51Identifier)

    command = vllm_adapter.build_start_command(
        {
            "engine": "vllm_ascend",
            "model_name": "GLM-5.1-w8a8",
            "model_path": "/models/GLM-5.1-w8a8",
            "model_type": "llm",
            "engine_config": {"enable_expert_parallel": True},
            "_explicit_cli_keys": ["enable_expert_parallel"],
        },
    )

    assert " --enable-expert-parallel" in f" {command}"
