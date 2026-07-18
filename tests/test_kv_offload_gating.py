import sys
import json
import logging
import inspect
import os
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402
from engines import vllm_distributed  # noqa: E402
from core import wings_entry  # noqa: E402
from core import config_loader  # noqa: E402
from core.port_plan import derive_port_plan  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from features.kv_offload import memcache as memcache_package  # noqa: E402
from features.kv_offload.memcache import hybrid as memcache_hybrid  # noqa: E402


def test_memcache_helpers_are_not_owned_by_vllm_adapter():
    assert not hasattr(vllm_adapter, "build_memcache_ascend_store_config")
    assert not hasattr(vllm_adapter, "is_kimi_k27_code_memcache_params")


def test_memcache_transfer_config_is_owned_by_config_loader():
    assert not hasattr(memcache_hybrid, "build_memcache_ascend_store_config")
    assert hasattr(config_loader, "_build_memcache_ascend_store_config")


def test_memcache_package_exports_all_declared_helpers():
    for name in memcache_package.__all__:
        assert hasattr(memcache_package, name)


def test_memcache_fragment_is_rendered_from_shell_templates(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    fragment = memcache_hybrid.build_memcache_hybrid_fragment(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
            "_smart_feats": ["offload"],
        },
    )

    source = inspect.getsource(memcache_hybrid.build_memcache_hybrid_fragment)
    template_dir = Path(memcache_hybrid.__file__).resolve().parent

    assert fragment["enabled"] is True
    assert (template_dir / "memcache_engine_prelude.sh").exists()
    assert (template_dir / "memcache_master.sh").exists()
    assert (template_dir / "memcache_engine_prelude.sh").read_text(
        encoding="utf-8"
    ).startswith("#!/bin/bash\n")
    assert fragment["engine_prelude"].startswith("#!/bin/bash\n")
    assert "ock.mmc.local_service.dram.size = ${WINGS_MEMCACHE_DRAM_GB}GB" in fragment["engine_prelude"]
    assert "Starting local MetaService for ConfigStore" in fragment["engine_prelude"]
    assert '"${WINGS_MEMCACHE_DIR}/start_memcache_master.sh"' in fragment["engine_prelude"]
    assert "_wings_memcache_config_store_ready" in fragment["engine_prelude"]
    assert "ConfigStore ${WINGS_MEMCACHE_CONFIG_STORE_URL} is ready" in fragment["engine_prelude"]
    assert "Effective mmc_local.conf" in fragment["engine_prelude"]
    assert "MetaService startup log" in fragment["engine_prelude"]
    assert "Effective mmc_meta.conf" in fragment["master_script"]
    assert "Executing: python -c" in fragment["master_script"]
    assert "ock.mmc.local_service.dram.size" not in source


def test_prepare_params_for_startup_status_consumes_prepared_topology(monkeypatch):
    prepared = {
        "tensor_parallel_size": 4,
        "data_parallel_size": 2,
        "data_parallel_size_local": 1,
        "data_parallel_start_rank": 0,
    }
    monkeypatch.setattr(vllm_adapter, "_prepare_engine_config", lambda _params: prepared)
    params = {"engine": "vllm_ascend", "engine_config": {}}

    vllm_adapter.prepare_params_for_startup_status(params)

    assert params["engine_config"] == prepared


def test_prepare_params_for_startup_status_logs_and_reraises(monkeypatch, caplog):
    def _raise_prepare_error(_params):
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(vllm_adapter, "_prepare_engine_config", _raise_prepare_error)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="prepare failed"):
        vllm_adapter.prepare_params_for_startup_status({"engine": "vllm"})

    assert "Failed to prepare final engine config for status reporting" in caplog.text


def test_prepare_params_for_startup_status_rejects_invalid_return(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "_prepare_engine_config", lambda _params: None)

    with pytest.raises(TypeError, match="must return a dict"):
        vllm_adapter.prepare_params_for_startup_status({"engine": "vllm"})


def _clear_deepseek_v4_flash_lmcache_env(monkeypatch):
    for key in (
        "LMCACHE_LOCAL_CPU",
        "LMCACHE_MAX_LOCAL_CPU_SIZE",
        "LMCACHE_TRACK_USAGE",
        "LMCACHE_USE_LAYERWISE",
        "LMCACHE_NUMA_MODE",
        "LMCACHE_EXTRA_CONFIG",
        "LMCACHE_LOOKUP_SERVER_WORKER_IDS",
        "AVAILABLE_POD_MEM_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)


class _FakeDeepSeekV4ProIdentifier:
    model_architecture = "DeepseekV4ForCausalLM"
    model_quantize = "w4a8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeQwen35Nvfp4Identifier:
    model_architecture = "Qwen3_5MoeForConditionalGeneration"
    model_quantize = "nvfp4"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeMiniMaxM2Identifier:
    model_architecture = "MiniMaxM2ForCausalLM"
    model_quantize = "nvfp4"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


class _FakeMiniMaxM3Identifier:
    model_architecture = "MiniMaxM3SparseForConditionalGeneration"
    model_quantize = "mxfp8"

    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type


def test_lmcache_env_exports_are_dropped_when_l2_child_switches_are_off(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "false")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "200")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    monkeypatch.setenv("KV_DISK_OFFLOAD_PATH", "/mnt/kv")
    monkeypatch.setenv("KV_DISK_OFFLOAD_SIZE", "500")

    commands = vllm_adapter._build_cache_env_commands(
        "vllm",
        {
            "engine": "vllm",
            "model_name": "GLM-4.7-FP8",
            "model_path": "/models/ZhipuAI/GLM-4.7-FP8",
            "model_type": "llm",
            "device_count": 1,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert rendered == ""
    assert "KV_MEM_OFFLOAD_SIZE" not in rendered
    assert "KV_DISK_OFFLOAD_PATH" not in rendered
    assert "KV_DISK_OFFLOAD_SIZE" not in rendered


def test_deepseek_v4_flash_ascend_lmcache_env_uses_021_local_cpu_switches(monkeypatch):
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a3")
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)

    commands = vllm_adapter._build_cache_env_commands(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 8,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export LMCACHE_LOCAL_CPU=True" in rendered
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=5" in rendered
    assert "export LMCACHE_TRACK_USAGE=false" in rendered
    assert "export LMCACHE_USE_LAYERWISE=False" in rendered
    assert "export LMCACHE_NUMA_MODE=auto" in rendered
    assert "export LMCACHE_CHUNK_SIZE=1024" in rendered
    assert "export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3" in rendered
    assert "CPUOffloadingConnector" not in rendered


def test_lmcache_custom_size_is_computed_per_card(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "200")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    monkeypatch.setattr(
        vllm_adapter,
        "_write_lmcache_config_yaml",
        lambda engine, max_cpu_size=None, local_cpu_enabled=None: None,
    )

    commands = vllm_adapter._build_cache_env_commands(
        "vllm",
        {
            "engine": "vllm",
            "model_name": "GLM-4.7-FP8",
            "model_path": "/models/ZhipuAI/GLM-4.7-FP8",
            "model_type": "llm",
            "device_count": 8,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export KV_MEM_OFFLOAD_SIZE=25" in rendered


def test_deepseek_v4_flash_ascend_lmcache_worker_ids_follow_tensor_parallel_size(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)

    commands = vllm_adapter._build_cache_env_commands(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3,4,5,6,7" in rendered


def test_deepseek_v4_flash_ascend_lmcache_worker_ids_keep_explicit_env(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.setenv("LMCACHE_LOOKUP_SERVER_WORKER_IDS", "1,3")

    commands = vllm_adapter._build_cache_env_commands(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export LMCACHE_LOOKUP_SERVER_WORKER_IDS=1,3" in rendered
    assert "export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3,4,5,6,7" not in rendered


def test_deepseek_v4_flash_ascend_lmcache_drops_cpu_pool_without_page_size(monkeypatch):
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.delenv("ENABLE_KV_MEM_OFFLOAD", raising=False)
    monkeypatch.delenv("KV_MEM_OFFLOAD_SIZE", raising=False)
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")

    commands = vllm_adapter._build_cache_env_commands(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 8,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export LMCACHE_LOCAL_CPU" not in rendered
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE" not in rendered


def test_deepseek_v4_flash_ascend_lmcache_auto_size_is_computed_per_card(monkeypatch):
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "204800")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")

    commands = vllm_adapter._build_cache_env_commands(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export LMCACHE_LOCAL_CPU=True" in rendered
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=15" in rendered
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=auto" not in rendered


def test_deepseek_v4_flash_ascend_lmcache_auto_size_treats_available_pod_mem_as_mb(monkeypatch):
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "241664")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")

    commands = vllm_adapter._build_cache_env_commands(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=19" in rendered
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=27179" not in rendered


def test_deepseek_v4_flash_ascend_lmcache_auto_floor_disables_cpu_pool(monkeypatch):
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")

    commands = vllm_adapter._build_cache_env_commands(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export LMCACHE_LOCAL_CPU" not in rendered
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE" not in rendered
    assert "export KV_MEM_OFFLOAD_AUTO_FLOOR_DISABLED=true" in rendered


def test_lmcache_auto_floor_omits_generic_cpu_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    monkeypatch.setattr(vllm_adapter, "_LMCACHE_SHARED_VOLUME", str(tmp_path))

    commands = vllm_adapter._build_cache_env_commands(
        "vllm",
        {
            "engine": "vllm",
            "model_name": "GLM-4.7-FP8",
            "model_path": "/models/GLM-4.7-FP8",
            "model_type": "llm",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export ENABLE_KV_MEM_OFFLOAD" not in rendered
    assert "export KV_MEM_OFFLOAD_SIZE" not in rendered
    assert "export KV_MEM_OFFLOAD_AUTO_FLOOR_DISABLED=true" in rendered
    assert "export LMCACHE_CONFIG_FILE" not in rendered
    assert not (tmp_path / "lmcache_config.yaml").exists()


def test_lmcache_auto_floor_reports_inactive_status_and_skips_patch(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    monkeypatch.delenv("ENABLE_KV_QAT", raising=False)
    monkeypatch.delenv("ENABLE_COLD_START", raising=False)
    monkeypatch.setattr(
        wings_entry,
        "_ADVANCED_FEATURES_FILE",
        str(tmp_path / "advanced_features.json"),
    )

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_type": "llm",
        "device_count": 8,
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "_smart_feats": ["offload"],
    }

    should_install = wings_entry._should_install_deepseek_v4_flash_ascend_lmcache(
        "vllm_ascend", params
    )
    snippet = wings_entry._build_deepseek_v4_flash_ascend_lmcache_install_snippet(
        "vllm_ascend", params
    )
    wings_entry._write_advanced_features_json("vllm_ascend", params)

    data = json.loads((tmp_path / "advanced_features.json").read_text(encoding="utf-8"))
    assert should_install is False
    assert snippet == ""
    assert data["features"]["kv_offload"] is False
    assert data["variants"]["kv_offload"] == "lmcache_cpu+auto+floor_disabled"
    assert data["others"]["kv_mem_offload_size"] == 0
    assert wings_entry._collect_active_feature_names(params) == []
    assert wings_entry._has_advanced_features(params) is False


def test_deepseek_v4_flash_ascend_auto_reports_effective_lmcache_size(monkeypatch, tmp_path):
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "204800")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    monkeypatch.setattr(
        wings_entry,
        "_ADVANCED_FEATURES_FILE",
        str(tmp_path / "advanced_features.json"),
    )

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_type": "llm",
        "device_count": 8,
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "_smart_feats": ["offload"],
    }

    wings_entry._write_advanced_features_json("vllm_ascend", params)

    data = json.loads((tmp_path / "advanced_features.json").read_text(encoding="utf-8"))
    assert data["features"]["kv_offload"] is True
    assert data["variants"]["kv_offload"] == "lmcache_cpu+auto"
    assert data["others"]["kv_mem_offload_size"] == 15


def test_launcher_plan_reports_lmcache_auto_size_from_generated_command(monkeypatch, tmp_path):
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "204800")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    state_file = tmp_path / "advanced_features.json"
    monkeypatch.setattr(wings_entry, "_ADVANCED_FEATURES_FILE", str(state_file))

    hardware_file = tmp_path / "hardware_info.json"
    hardware_file.write_text(
        json.dumps(
            {
                "device": "ascend",
                "count": 8,
                "hardware_family": "Ascend910B_64G",
                "details": [
                    {
                        "device_id": index,
                        "name": "Ascend910B3",
                        "total_memory": 64,
                        "free_memory": 60,
                        "used_memory": 4,
                    }
                    for index in range(8)
                ],
                "units": "GB",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WINGS_HARDWARE_FILE", str(hardware_file))

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "torch_dtype": "bfloat16",
                "quantization_config": {"quant_method": "ascend"},
            }
        ),
        encoding="utf-8",
    )
    launch_args = parse_launch_args(
        [
            "--model-name", "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "--model-path", str(model_dir),
            "--engine", "vllm_ascend",
            "--device-count", "8",
            "--trust-remote-code",
            "--port", "18000",
            "--node-rank", "0",
        ]
    )
    port_plan = derive_port_plan(
        port=launch_args.port,
        enable_reason_proxy=True,
        health_port=19000,
    )

    plan = wings_entry.build_launcher_plan(launch_args, port_plan)

    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=15" in plan.command
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["features"]["kv_offload"] is True
    assert data["variants"]["kv_offload"] == "lmcache_cpu+auto"
    assert data["others"]["kv_mem_offload_size"] == 15


def test_qwen35_nvfp4_auto_floor_reports_inactive_offload_status(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeQwen35Nvfp4Identifier)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    monkeypatch.delenv("ENABLE_KV_QAT", raising=False)
    monkeypatch.delenv("ENABLE_COLD_START", raising=False)
    monkeypatch.setattr(
        wings_entry,
        "_ADVANCED_FEATURES_FILE",
        str(tmp_path / "advanced_features.json"),
    )

    params = {
        "engine": "vllm",
        "model_name": "Qwen3.5-397B-A17B-NVFP4",
        "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
        "model_type": "llm",
        "device_count": 8,
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "enable_speculative_decode": True,
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": ["spec", "offload"],
    }

    wings_entry._write_advanced_features_json("vllm", params)

    data = json.loads((tmp_path / "advanced_features.json").read_text(encoding="utf-8"))
    assert data["features"]["speculative_decode"] is True
    assert data["features"]["kv_offload"] is False
    assert data["variants"]["speculative_decode"] == "mtp"
    assert data["variants"]["kv_offload"] == "native_kv_offloading_backend+auto+floor_disabled"
    assert data["others"]["kv_mem_offload_size"] == 0
    assert wings_entry._collect_active_feature_names(params) == ["speculative_decode"]

    with caplog.at_level(logging.INFO):
        wings_entry._log_advanced_feature_config("vllm", params, True)

    assert "active features = speculative_decode" in caplog.text
    assert (
        "[lmcache_offload] inactive "
        "(variant=native_kv_offloading_backend+auto+floor_disabled)"
    ) in caplog.text
    assert "kv_transfer_config = " not in caplog.text


def test_qwen35_nvfp4_native_reports_effective_kv_mem_size(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "80")
    monkeypatch.setattr(
        wings_entry,
        "_ADVANCED_FEATURES_FILE",
        str(tmp_path / "advanced_features.json"),
    )

    params = {
        "engine": "vllm",
        "model_name": "Qwen3.5-397B-A17B-NVFP4",
        "model_path": "/models/Qwen3.5-397B-NVFP4",
        "model_type": "llm",
        "device_count": 8,
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": ["spec", "offload"],
    }

    wings_entry._write_advanced_features_json("vllm", params)

    data = json.loads((tmp_path / "advanced_features.json").read_text(encoding="utf-8"))
    assert data["features"]["kv_offload"] is True
    assert data["variants"]["kv_offload"] == "native_kv_offloading_backend"
    assert data["others"]["kv_mem_offload_size"] == 80


def test_lmcache_auto_floor_with_disk_keeps_patch_and_disk_variant(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "true")
    monkeypatch.setenv("KV_DISK_OFFLOAD_PATH", "/mnt/kvcache_offload")
    monkeypatch.setenv("KV_DISK_OFFLOAD_SIZE", "8")

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_type": "llm",
        "device_count": 8,
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "_smart_feats": ["offload"],
    }

    should_install = wings_entry._should_install_deepseek_v4_flash_ascend_lmcache(
        "vllm_ascend", params
    )
    variant = vllm_adapter.resolve_offload_variant(params, "vllm_ascend")

    assert should_install is True
    assert variant == "lmcache_disk+cpu_auto_floor_disabled"


def test_deepseek_v4_flash_ascend_custom_kv_mem_size_wins_over_lmcache_size(monkeypatch):
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "80")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "21")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "262144")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "true")

    commands = vllm_adapter._build_cache_env_commands(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["offload"],
        },
    )

    rendered = "\n".join(commands)
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=10" in rendered
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=21" not in rendered


def test_deepseek_v4_flash_ascend_custom_reports_effective_lmcache_size(monkeypatch, tmp_path):
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "80")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "21")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    monkeypatch.setattr(
        wings_entry,
        "_ADVANCED_FEATURES_FILE",
        str(tmp_path / "advanced_features.json"),
    )

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_type": "llm",
        "device_count": 8,
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "_smart_feats": ["offload"],
    }

    wings_entry._write_advanced_features_json("vllm_ascend", params)

    data = json.loads((tmp_path / "advanced_features.json").read_text(encoding="utf-8"))
    assert data["features"]["kv_offload"] is True
    assert data["variants"]["kv_offload"] == "lmcache_cpu+custom"
    assert data["others"]["kv_mem_offload_size"] == 10


def test_deepseek_v4_flash_ascend_ld_preload_is_safe_under_set_u(monkeypatch):
    env_commands = vllm_adapter._build_deepseek_v4_flash_env(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
        },
    )

    ld_preload = next(cmd for cmd in env_commands if cmd.startswith("export LD_PRELOAD="))
    assert "${LD_PRELOAD:+" in ld_preload
    assert "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" not in ld_preload


def test_kimi_k27_code_ascend_env_includes_jemalloc_and_pythonhashseed():
    env_commands = vllm_adapter._build_kimik25_ascend_env(
        "KimiK25ForConditionalGeneration"
    )

    rendered = "\n".join(env_commands)
    assert "export PYTHONHASHSEED=0" in rendered
    ld_preload = next(cmd for cmd in env_commands if cmd.startswith("export LD_PRELOAD="))
    assert "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2" in ld_preload
    assert "${LD_PRELOAD:+" in ld_preload
    assert "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" not in ld_preload


def test_kimi_k27_code_ascend_env_matches_day0_memcache_script(monkeypatch):
    monkeypatch.delenv("DISTRIBUTED", raising=False)

    env_commands = vllm_adapter._build_ascend_arch_model_env_commands(
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
        },
        "KimiK25ForConditionalGeneration",
    )
    rendered = "\n".join(env_commands)

    assert "export OMP_NUM_THREADS=10" in rendered
    assert "export HCCL_BUFFSIZE=1024" in rendered
    assert "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1" in rendered
    assert "export VLLM_ASCEND_ENABLE_MLAPO=1" in rendered
    assert "export VLLM_ASCEND_BALANCE_SCHEDULING=1" in rendered
    assert "export VLLM_ENGINE_READY_TIMEOUT_S=3600" not in rendered


def test_deepseek_v4_flash_ascend_v021_uses_lmcache_package_config(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_type": "llm",
        "_smart_feats": ["offload"],
    }

    for engine_version in ("v0.21.0-a2", "v0.21.0-a3"):
        monkeypatch.setenv("ENGINE_VERSION", engine_version)

        should_install = wings_entry._should_install_deepseek_v4_flash_ascend_lmcache(
            "vllm_ascend", params
        )
        snippet = wings_entry._build_deepseek_v4_flash_ascend_lmcache_install_snippet(
            "vllm_ascend", params
        )

        assert should_install is True
        assert (
            "python install.py --config "
            "'{\"packages\": [\"lmcache-ascend:v0.4.5\"]}'"
        ) in snippet
        assert 'cd "/accel-volume"' in snippet


def test_deepseek_v4_flash_ascend_future_version_keeps_lmcache_patch_hook(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENGINE_VERSION", "v0.22.0-a2")

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_type": "llm",
        "_smart_feats": ["offload"],
    }
    should_install = wings_entry._should_install_deepseek_v4_flash_ascend_lmcache(
        "vllm_ascend",
        params,
    )
    snippet = wings_entry._build_deepseek_v4_flash_ascend_lmcache_install_snippet(
        "vllm_ascend", params
    )

    assert should_install is True
    assert "lmcache-ascend:v0.4.5" in snippet


def test_deepseek_v4_flash_lmcache_legacy_env_alias_enables_patch(monkeypatch):
    monkeypatch.delenv("ENABLE_KV_OFFLOAD", raising=False)
    monkeypatch.setenv("LMCACHE_OFFLOAD", "true")

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_type": "llm",
    }

    config_loader.apply_effective_feature_enablement(
        params,
        {"device": "ascend", "details": [{"name": "Ascend910C"}]},
    )

    assert os.environ["ENABLE_KV_OFFLOAD"] == "true"
    assert params["_allowed_smart_feats"] == ["offload", "sparse", "spec"]
    assert params["_smart_feats"] == ["offload"]
    assert (
        wings_entry._should_install_deepseek_v4_flash_ascend_lmcache(
            "vllm_ascend", params
        )
        is True
    )


def test_lmcache_patch_uses_effective_smart_feats_when_env_not_synced(monkeypatch):
    monkeypatch.delenv("ENABLE_KV_OFFLOAD", raising=False)
    monkeypatch.delenv("LMCACHE_OFFLOAD", raising=False)

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_type": "llm",
        "_smart_feats": ["offload"],
    }

    assert (
        wings_entry._should_install_deepseek_v4_flash_ascend_lmcache(
            "vllm_ascend", params
        )
        is True
    )


def test_lmcache_env_exports_use_effective_smart_feats_when_env_not_synced(monkeypatch):
    _clear_deepseek_v4_flash_lmcache_env(monkeypatch)
    monkeypatch.delenv("ENABLE_KV_OFFLOAD", raising=False)
    monkeypatch.delenv("LMCACHE_OFFLOAD", raising=False)
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "80")

    params = {
        "engine": "vllm_ascend",
        "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
        "model_type": "llm",
        "device_count": 8,
        "_smart_feats": ["offload"],
    }

    commands = vllm_adapter._build_cache_env_commands("vllm_ascend", params)
    rendered = "\n".join(commands)
    variant = vllm_adapter.resolve_offload_variant(params, "vllm_ascend")
    resolved_size = vllm_adapter.resolve_effective_kv_mem_offload_size(
        params,
        "vllm_ascend",
        variant,
    )

    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=10" in rendered
    assert variant == "lmcache_cpu+custom"
    assert resolved_size == 10


def test_memcache_fragment_uses_effective_smart_feats_when_env_not_synced(monkeypatch):
    monkeypatch.delenv("ENABLE_KV_OFFLOAD", raising=False)
    monkeypatch.delenv("LMCACHE_OFFLOAD", raising=False)
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    fragment = memcache_hybrid.build_memcache_hybrid_fragment(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
            "device_count": 16,
            "_smart_feats": ["offload"],
        },
    )

    assert fragment["enabled"] is True
    assert 'export WINGS_MEMCACHE_DRAM_GB="2"' in fragment["engine_prelude"]


def test_native_kv_offload_uses_effective_smart_feats_when_env_not_synced(monkeypatch):
    monkeypatch.delenv("ENABLE_KV_OFFLOAD", raising=False)
    monkeypatch.delenv("LMCACHE_OFFLOAD", raising=False)
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


def test_fallback_uses_effective_smart_feats_for_kv_offload(monkeypatch):
    monkeypatch.delenv("ENABLE_KV_OFFLOAD", raising=False)
    monkeypatch.delenv("LMCACHE_OFFLOAD", raising=False)
    captured = {}

    def _capture_start_engine_service(merged):
        captured.update(merged)
        return "exec engine\n"

    monkeypatch.setattr(wings_entry, "start_engine_service", _capture_start_engine_service)

    wings_entry._build_advanced_feature_fallback_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.6-27B",
            "model_path": "/models/Qwen3.6-27B",
            "model_type": "llm",
            "_smart_card_token": "l20",
            "_smart_feats": ["offload"],
            "enable_speculative_decode": False,
            "enable_sparse": False,
            "engine_config": {
                "model": "/models/Qwen3.6-27B",
                "kv_transfer_config": {"kv_connector": "LMCacheConnectorV1"},
            },
        }
    )

    assert "kv_transfer_config" not in captured["engine_config"]
    assert captured["_wings_fallback_no_kv_offload"] is True


def test_nvidia_vllm_lmcache_skips_patch_install_target():
    should_install = wings_entry._should_install_deepseek_v4_flash_ascend_lmcache(
        "vllm",
        {
            "engine": "vllm",
            "model_name": "MiniMax/MiniMax-M2.7-NVFP4",
            "model_path": "/models/MiniMax/MiniMax-M2.7-NVFP4",
            "model_type": "llm",
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
        },
    )

    assert should_install is False


def test_kimi_k27_code_memcache_skips_lmcache_patch(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    should_install = wings_entry._should_install_deepseek_v4_flash_ascend_lmcache(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
            "_smart_feats": ["offload"],
        },
    )

    assert should_install is False


def test_kimi_k27_code_memcache_skips_lmcache_env(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    commands = vllm_adapter._build_cache_env_commands(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
            "device_count": 16,
            "_smart_feats": ["offload"],
        },
    )

    assert commands == []


def test_kimi_memcache_model_matcher_uses_document_model_name():
    assert memcache_hybrid.is_kimi_k27_code_memcache_params(
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
        },
        "vllm_ascend",
    ) is True
    assert memcache_hybrid.is_kimi_k26_memcache_params(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/Kimi-K2.6",
            "model_path": "/models/Eco-Tech/Kimi-K2.6",
        },
        "vllm_ascend",
    ) is False
    assert memcache_hybrid.is_kimi_k27_code_memcache_params(
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code-w4a8",
            "model_path": "/harbor_data/Kimi-K2.7-Code-w4a8",
        },
        "vllm_ascend",
    ) is False
    assert memcache_hybrid.is_kimi_k26_memcache_params(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/Kimi-K2.6-W4A8",
            "model_path": "/models/Eco-Tech/Kimi-K2.6-W4A8",
        },
        "vllm_ascend",
    ) is True


def test_kimi_k27_code_memcache_engine_prelude_uses_per_card_page_offload_memory(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    fragment = memcache_hybrid.build_memcache_hybrid_fragment(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
            "device_count": 16,
            "_smart_feats": ["offload"],
        },
    )

    assert fragment["enabled"] is True
    # 页面下发 40G 是节点总容量；16 卡与 LMCache 一样均分为每卡 2G。
    assert 'export WINGS_MEMCACHE_DRAM_GB="2"' in fragment["engine_prelude"]
    assert "tcp://127.0.0.1:5000" in fragment["engine_prelude"]
    assert "tcp://127.0.0.1:6000" in fragment["engine_prelude"]
    assert 'export WINGS_MEMCACHE_PROTOCOL="${WINGS_MEMCACHE_PROTOCOL:-device_sdma}"' in fragment["engine_prelude"]
    assert "ock.mmc.local_service.dram.size = ${WINGS_MEMCACHE_DRAM_GB}GB" in fragment["engine_prelude"]
    assert "MMC_LOCAL_CONFIG_PATH" in fragment["engine_prelude"]
    assert "MMC_META_CONFIG_PATH" in fragment["master_script"]


def test_kimi_k26_memcache_engine_prelude_uses_per_card_page_offload_memory(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    fragment = memcache_hybrid.build_memcache_hybrid_fragment(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/Kimi-K2.6-W4A8",
            "model_path": "/models/Eco-Tech/Kimi-K2.6-W4A8",
            "model_type": "llm",
            "device_count": 16,
            "_smart_feats": ["offload"],
        },
    )

    assert fragment["enabled"] is True
    assert 'export WINGS_MEMCACHE_DRAM_GB="2"' in fragment["engine_prelude"]
    assert "AscendStore" not in fragment["engine_prelude"]


def test_kimi_k26_and_k27_code_topology_defaults_are_separate():
    kimi26_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/Kimi-K2.6-W4A8",
            "model_path": "/models/Eco-Tech/Kimi-K2.6-W4A8",
            "model_type": "llm",
            "device_count": 16,
            "engine_config": {"tensor_parallel_size": 16},
        }
    )
    assert kimi26_config["tensor_parallel_size"] == 4
    assert kimi26_config["data_parallel_size"] == 4

    kimi27_config = vllm_adapter._prepare_engine_config(
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
            "device_count": 16,
            "engine_config": {},
        }
    )
    assert kimi27_config["tensor_parallel_size"] == 16
    assert "data_parallel_size" not in kimi27_config


def test_memcache_auto_memory_is_evenly_split_per_card(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "204800")

    # 200G 容器内存、TP=2 时，节点可卸载容量为 163G；4 卡向下均分为每卡 40G。
    assert memcache_hybrid.resolve_memcache_dram_gb(
        {
            "device_count": 4,
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
        }
    ) == 40


@pytest.mark.parametrize(
    (
        "model_name",
        "card_token",
        "expected_meta_port",
        "expected_config_port",
        "expected_protocol",
    ),
    [
        ("Qwen/Qwen3.5-27B", "910c", 50051, 50061, "device_sdma"),
        ("Qwen/Qwen3.6-27B", "910c", 50071, 50081, "device_rdma"),
        ("Eco-Tech/Qwen3.6-27B-w8a8", "910c", 50071, 50081, "device_rdma"),
        ("Qwen/Qwen3.6-35B-A3B", "910c", 50071, 50081, "device_rdma"),
        ("Eco-Tech/Qwen3.6-35B-A3B-w8a8", "910c", 50071, 50081, "device_rdma"),
    ],
)
def test_qwen_day0_memcache_profile_uses_static_scene_defaults(
    monkeypatch,
    model_name,
    card_token,
    expected_meta_port,
    expected_config_port,
    expected_protocol,
):
    """Qwen Day0 MemCache 端口和协议来自代码侧场景默认。

    页面仍然可以通过 WINGS_MEMCACHE_META_SERVICE_URL 和
    WINGS_MEMCACHE_CONFIG_STORE_URL、WINGS_MEMCACHE_PROTOCOL 覆盖最终值。
    页面未覆盖时，不能用全局 RDMA 默认值覆盖 Qwen3.5 的 SDMA 标准。
    白名单只负责命中 memcache backend，不承载端口和协议。
    """
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    fragment = memcache_hybrid.build_memcache_hybrid_fragment(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": model_name,
            "model_path": f"/models/{model_name}",
            "model_type": "llm",
            "_smart_card_token": card_token,
            "_smart_feats": ["offload"],
        },
    )

    assert fragment["enabled"] is True
    assert f"tcp://127.0.0.1:{expected_meta_port}" in fragment["engine_prelude"]
    assert f"tcp://127.0.0.1:{expected_config_port}" in fragment["engine_prelude"]
    assert f"tcp://127.0.0.1:{expected_meta_port}" in fragment["master_script"]
    assert f"tcp://127.0.0.1:{expected_config_port}" in fragment["master_script"]
    assert (
        f'export WINGS_MEMCACHE_PROTOCOL="${{WINGS_MEMCACHE_PROTOCOL:-{expected_protocol}}}"'
        in fragment["engine_prelude"]
    )
    assert (
        "ock.mmc.local_service.protocol = ${WINGS_MEMCACHE_PROTOCOL}"
        in fragment["engine_prelude"]
    )


@pytest.mark.parametrize(
    ("model_name", "expected_protocol"),
    [
        ("Qwen/Qwen3.5-27B", None),
        ("Qwen/Qwen3.5-35B-A3B", "device_sdma"),
        ("Qwen/Qwen3.5-122B-A10B", None),
        ("Qwen/Qwen3.6-27B", None),
        ("Eco-Tech/Qwen3.6-27B-w8a8", "device_rdma"),
        ("Qwen/Qwen3.6-35B-A3B", None),
        ("Eco-Tech/Qwen3.6-35B-A3B-w8a8", "device_rdma"),
    ],
)
def test_qwen_day0_910b_memcache_follows_whitelist(monkeypatch, model_name, expected_protocol):
    """910B 只允许白名单补充的 Qwen Day0 MemCache 场景，其余 Qwen 仍保持关闭。"""
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")

    params = {
        "engine": "vllm_ascend",
        "model_name": model_name,
        "model_path": f"/models/{model_name}",
        "model_type": "llm",
        "device_count": 4,
        "_smart_card_token": "910b",
        "_smart_feats": ["offload", "spec"],
    }

    assert (
        memcache_hybrid.is_qwen_day0_memcache_params(params, "vllm_ascend")
        is bool(expected_protocol)
    )
    fragment = memcache_hybrid.build_memcache_hybrid_fragment(
        "vllm_ascend",
        params,
    )
    if expected_protocol:
        assert fragment["enabled"] is True
        assert (
            f'export WINGS_MEMCACHE_PROTOCOL="${{WINGS_MEMCACHE_PROTOCOL:-{expected_protocol}}}"'
            in fragment["engine_prelude"]
        )
    else:
        assert fragment == memcache_hybrid.empty_memcache_hybrid_fragment()


def test_memcache_backend_respects_effective_offload_switch(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    params = {
        "engine": "vllm_ascend",
        "model_name": "Qwen3.6-27B",
        "model_path": "/models/Qwen3.6-27B",
        "model_type": "llm",
        "device_count": 2,
        "_smart_card_token": "910c",
        "_smart_feats": [],
    }

    assert memcache_hybrid.is_memcache_hybrid_params(params, "vllm_ascend") is False
    assert memcache_hybrid.build_memcache_hybrid_fragment(
        "vllm_ascend",
        params,
    ) == memcache_hybrid.empty_memcache_hybrid_fragment()


def test_qwen_day0_memcache_auto_memory_below_floor_is_disabled(monkeypatch):
    """Qwen Day0 MemCache 复用通用 auto floor，容量不足时应关闭 offload。"""
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.delenv("LMCACHE_MAX_LOCAL_CPU_SIZE", raising=False)
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "40960")

    params = {
        "engine": "vllm_ascend",
        "model_name": "Qwen3.6-27B",
        "model_path": "/usr/local/serving/models",
        "model_type": "llm",
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "_smart_card_token": "910c",
        "_smart_feats": ["offload", "spec"],
    }

    fragment = memcache_hybrid.build_memcache_hybrid_fragment("vllm_ascend", params)
    engine_config = {}
    config_loader._set_kv_cache_config(engine_config, params)
    variant = vllm_adapter.resolve_offload_variant(params, "vllm_ascend")
    resolved_size = vllm_adapter.resolve_effective_kv_mem_offload_size(
        params,
        "vllm_ascend",
        variant,
    )

    assert memcache_hybrid.resolve_memcache_dram_gb(params) is None
    assert fragment == {
        "enabled": False,
        "engine_prelude": "",
        "fallback_cleanup": "",
        "master_script": "",
        "env": {},
    }
    assert "kv_transfer_config" not in engine_config
    assert variant == "disabled"
    assert resolved_size is None


def test_kimi_k27_code_memcache_without_page_memory_is_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.delenv("KV_MEM_OFFLOAD_SIZE", raising=False)
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "40")

    fragment = memcache_hybrid.build_memcache_hybrid_fragment(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
            "device_count": 16,
            "_smart_feats": ["offload"],
        },
    )

    assert fragment == {
        "enabled": False,
        "engine_prelude": "",
        "fallback_cleanup": "",
        "master_script": "",
        "env": {},
    }


def test_kimi_k27_code_memcache_prelude_is_assembled_before_engine_body(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    monkeypatch.delenv("ENABLE_ACCEL", raising=False)

    command = wings_entry._assemble_startup_command(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
            "device_count": 16,
            "_smart_feats": ["offload"],
        },
        {"device": "ascend", "details": [{"name": "Ascend910C"}]},
        'exec vllm serve /harbor_data/Kimi-K2.7-Code\n',
        "",
    )

    prelude_index = command.index("# --- wings-memcache: engine prelude ---")
    master_start_index = command.index("Starting local MetaService for ConfigStore")
    config_ready_index = command.index("ConfigStore ${WINGS_MEMCACHE_CONFIG_STORE_URL} is ready")
    engine_index = command.index("exec vllm serve /harbor_data/Kimi-K2.7-Code")
    assert prelude_index < master_start_index < config_ready_index < engine_index
    assert 'export WINGS_MEMCACHE_DRAM_GB="2"' in command
    assert "ock.mmc.local_service.dram.size = ${WINGS_MEMCACHE_DRAM_GB}GB" in command
    assert "start_memcache_master.sh" in command


def test_kimi_k27_code_memcache_reports_active_variant(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    monkeypatch.setattr(
        wings_entry,
        "_ADVANCED_FEATURES_FILE",
        str(tmp_path / "advanced_features.json"),
    )

    params = {
        "engine": "vllm_ascend",
        "model_name": "Kimi-K2.7-Code",
        "model_path": "/harbor_data/Kimi-K2.7-Code",
        "model_type": "llm",
        "device_count": 16,
        "_smart_feats": ["offload"],
    }

    wings_entry._write_advanced_features_json("vllm_ascend", params)

    data = json.loads((tmp_path / "advanced_features.json").read_text(encoding="utf-8"))
    assert data["features"]["kv_offload"] is True
    assert data["variants"]["kv_offload"] == "memcache"
    assert data["others"]["kv_mem_offload_size"] == 2


def test_deepseek_v4_pro_is_not_cpu_offloading_connector_special_case(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeDeepSeekV4ProIdentifier)

    special = vllm_adapter._classify_offload_special_case(
        {
            "engine": "vllm_ascend",
            "model_name": "DeepSeek-V4-Pro-w4a8-mtp",
            "model_path": "/models/DeepSeek-V4-Pro-w4a8-mtp",
            "model_type": "llm",
        },
        "vllm_ascend",
    )

    assert special == ""


def test_deepseek_v4_pro_dp_env_matches_reference_script(monkeypatch):
    monkeypatch.setenv("ENGINE_VERSION", "0.21.0-a3")
    params = {
        "engine": "vllm_ascend",
        "model_name": "DeepSeek-V4-Pro-w4a8-mtp",
        "model_path": "/models/DeepSeek-V4-Pro-w4a8-mtp",
        "model_type": "llm",
        "distributed": True,
        "distributed_executor_backend": "external_launcher",
        "nnodes": 2,
    }

    env_commands = vllm_distributed._build_ascend_dp_env_commands(params, "eth0")

    # 参考脚本要求变量集合完全一致，额外的 multi-block/multi-groups/FUSED_MC2/whitelist 均不能出现。
    assert env_commands == [
        'export HCCL_OP_EXPANSION_MODE="AIV"',
        "export HCCL_IF_IP=${POD_IP:-${RANK_IP:-$(hostname -i | awk '{print $1}')}}",
        "export GLOO_SOCKET_IFNAME=eth0",
        "export TP_SOCKET_IFNAME=eth0",
        "export HCCL_SOCKET_IFNAME=eth0",
        "export HCCL_BUFFSIZE=2048",
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "export OMP_PROC_BIND=false",
        "export OMP_NUM_THREADS=10",
        "export TASK_QUEUE_ENABLE=1",
        "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD",
        "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1",
    ]
    assert vllm_adapter._build_deepseek_v4_pro_env(params) == []


def test_qwen35_nvfp4_native_offload_drops_when_page_size_missing(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.delenv("LMCACHE_MAX_LOCAL_CPU_SIZE", raising=False)

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == ""


def test_qwen35_nvfp4_native_offload_ignores_legacy_lmcache_size_input(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "200")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == ""


def test_qwen35_nvfp4_native_offload_reuses_page_size_without_per_card_scaling(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "80")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "200")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == " --kv-offloading-backend native --kv-offloading-size 80"


@pytest.mark.parametrize(
    ("model_name", "model_path", "smart_feats"),
    [
        (
            "Qwen/Qwen3.5-122B-A10B",
            "/models/Qwen/Qwen3.5-122B-A10B",
            ["spec", "offload"],
        ),
        (
            "Qwen/Qwen3.5-27B",
            "/models/Qwen/Qwen3.5-27B",
            ["spec", "offload"],
        ),
        (
            "Qwen/Qwen3.5-35B-A3B",
            "/models/Qwen/Qwen3.5-35B-A3B",
            ["offload"],
        ),
        (
            "MiniMax/MiniMax-M3-MXFP8",
            "/models/MiniMax/MiniMax-M3-MXFP8",
            ["spec", "offload"],
        ),
        (
            "MiniMax/MiniMax-M2.5-NVFP4",
            "/models/MiniMax/MiniMax-M2.5-NVFP4",
            ["spec", "offload"],
        ),
    ],
)
def test_pro5000_native_offload_uses_kv_mem_size_and_skips_lmcache_env(
    monkeypatch,
    model_name,
    model_path,
    smart_feats,
):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    params = {
        "engine": "vllm",
        "model_name": model_name,
        "model_path": model_path,
        "device_count": 4,
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": smart_feats,
    }

    assert vllm_adapter._build_kv_offload_cmd(params, "vllm") == (
        " --kv-offloading-backend native --kv-offloading-size 40"
    )
    assert vllm_adapter._build_cache_env_commands("vllm", params) == []


def test_minimax_m25_pro5000_env_uses_v1_without_lmcache(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM2Identifier)
    params = {
        "engine": "vllm",
        "model_name": "MiniMax/MiniMax-M2.5-NVFP4",
        "model_path": "/models/MiniMax/MiniMax-M2.5-NVFP4",
        "model_type": "llm",
        "device_count": 4,
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": ["spec", "offload"],
    }

    commands = vllm_adapter._build_model_env_commands(params, "vllm")
    common_commands = vllm_adapter._build_vllm_common_env_cmds(params, "vllm")

    assert commands == ["export VLLM_USE_V1=1"]
    assert "export VLLM_USE_V1=1" in common_commands
    assert not any("LMCACHE_" in command for command in commands)


def test_minimax_m3_pro5000_env_uses_v1_without_lmcache(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "ModelIdentifier", _FakeMiniMaxM3Identifier)
    params = {
        "engine": "vllm",
        "model_name": "MiniMax/MiniMax-M3-MXFP8",
        "model_path": "/models/MiniMax/MiniMax-M3-MXFP8",
        "model_type": "llm",
        "device_count": 8,
        "_smart_card_token": "rtxpro5000-72",
    }

    commands = vllm_adapter._build_model_env_commands(params, "vllm")

    assert commands == [
        "export VLLM_USE_V1=1",
        "export PYTHONHASHSEED=0",
    ]
    assert not any("LMCACHE_" in command for command in commands)


def test_qwen35_nvfp4_native_offload_auto_uses_kv_mem_formula_without_per_card_scaling(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.delenv("LMCACHE_MAX_LOCAL_CPU_SIZE", raising=False)
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "204800")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == " --kv-offloading-backend native --kv-offloading-size 121"


def test_qwen35_nvfp4_native_offload_auto_reuses_kv_mem_formula_floor(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.delenv("LMCACHE_MAX_LOCAL_CPU_SIZE", raising=False)
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == ""


def test_qwen35_nvfp4_native_offload_legacy_auto_input_is_ignored(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "204800")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == ""


def test_qwen35_nvfp4_native_offload_legacy_auto_input_variant_is_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.delenv("KV_MEM_OFFLOAD_SIZE", raising=False)
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "102400")

    params = {
        "engine": "vllm",
        "model_name": "Qwen3.5-397B-A17B-NVFP4",
        "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
        "device_count": 8,
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": ["spec", "offload"],
    }

    command = vllm_adapter._build_kv_offload_cmd(params, "vllm")
    variant = vllm_adapter.resolve_offload_variant(params, "vllm")

    assert command == ""
    assert variant == "disabled"


def test_deepseek_v4_flash_native_offload_reuses_page_size_without_per_card_scaling(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "200")
    monkeypatch.delenv("AVAILABLE_POD_MEM_SIZE", raising=False)

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "deepseek-ai/DeepSeek-V4-Flash",
            "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
            "device_count": 8,
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == " --kv-offloading-backend native --kv-offloading-size 200"


def test_deepseek_v4_flash_native_offload_auto_uses_formula_without_per_card_scaling(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "204800")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "deepseek-ai/DeepSeek-V4-Flash",
            "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == " --kv-offloading-backend native --kv-offloading-size 121"


def test_deepseek_v4_flash_native_offload_auto_reuses_formula_floor(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "auto")
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "81920")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "deepseek-ai/DeepSeek-V4-Flash",
            "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == ""


def test_qwen35_nvfp4_native_offload_skips_lmcache_patch(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    should_install = wings_entry._should_install_deepseek_v4_flash_ascend_lmcache(
        "vllm",
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["spec", "offload"],
        },
    )

    assert should_install is False


def test_deepseek_v4_flash_pro5000_installs_packages_without_feature_enablement(monkeypatch):
    # Pro5000 的 deepgemm/flashinfer 安装是运行时依赖补齐，不是 spec/sparse/offload
    # 任一高级特性的副作用。因此 _smart_feats 为空时仍必须生成 accel preamble。
    monkeypatch.setenv("ENGINE_VERSION", "v0.23.0")
    params = {
        "engine": "vllm",
        "model_name": "deepseek-ai/DeepSeek-V4-Flash",
        "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": [],
    }

    should_install = wings_entry._should_install_deepseek_v4_flash_pro5000_packages(
        "vllm", params
    )
    snippet = wings_entry._build_deepseek_v4_flash_pro5000_package_install_snippet(
        "vllm", params
    )
    accel_preamble = wings_entry._build_accel_preamble("vllm", params)

    assert should_install is True
    assert accel_preamble == snippet
    assert 'cd "/accel-volume"' in snippet
    assert (
        "python3 install.py --config "
        '\'{"engine": {"name": "vllm", "version": "v0.23.0"}, '
        '"packages": ["deepgemm:nv_dev_a6b593d", "flashinfer:v0.6.12"]}\''
    ) in snippet


def test_deepseek_v4_flash_pro5000_install_payload_uses_upstream_engine_version(monkeypatch):
    # 上层通过 ENGINE_VERSION 传递当前 vLLM 镜像版本；install.py payload 必须跟随它，
    # 不能回退到旧的硬编码 v0.23.0，否则镜像升级后会安装错误版本的依赖包。
    monkeypatch.setenv("ENGINE_VERSION", "0.23.1-rtxpro5000")
    params = {
        "engine": "vllm",
        "model_name": "deepseek-ai/DeepSeek-V4-Flash",
        "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": [],
    }

    snippet = wings_entry._build_deepseek_v4_flash_pro5000_package_install_snippet(
        "vllm", params
    )

    assert '"name": "vllm"' in snippet
    assert '"version": "v0.23.1"' in snippet
    assert '"version": "v0.23.0"' not in snippet


def test_deepseek_v4_flash_pro5000_install_skips_without_upstream_engine_version(monkeypatch):
    # 版本不可解析时宁可跳过安装，也不要猜测版本。这个回归用例保护
    # “版本由上层联动传入”的契约，避免后续又引入隐式默认版本。
    monkeypatch.delenv("ENGINE_VERSION", raising=False)
    params = {
        "engine": "vllm",
        "model_name": "deepseek-ai/DeepSeek-V4-Flash",
        "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": [],
    }

    assert (
        wings_entry._build_deepseek_v4_flash_pro5000_package_install_snippet(
            "vllm", params
        )
        == ""
    )


def test_qwen35_nvfp4_pro5000_skips_deepseek_package_install():
    # 芯片命中 Pro5000 不等于所有模型都要安装 DSV4 专属包；
    # 模型身份仍是触发条件的一部分，Qwen 路径必须保持无 install.py 片段。
    params = {
        "engine": "vllm",
        "model_name": "Qwen3.5-397B-A17B-NVFP4",
        "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
        "_smart_card_token": "rtxpro5000-72",
        "_smart_feats": ["spec", "offload"],
    }

    assert (
        wings_entry._should_install_deepseek_v4_flash_pro5000_packages("vllm", params)
        is False
    )
    assert (
        wings_entry._build_deepseek_v4_flash_pro5000_package_install_snippet(
            "vllm", params
        )
        == ""
    )


def test_deepseek_v4_flash_pro5000_without_offload_whitelist_omits_native_kv_offload_cli(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.delenv("LMCACHE_MAX_LOCAL_CPU_SIZE", raising=False)

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "deepseek-ai/DeepSeek-V4-Flash",
            "model_path": "/models/deepseek-ai/DeepSeek-V4-Flash",
            "device_count": 8,
            "_smart_feats": ["spec", "sparse"],
        },
        "vllm",
    )

    assert command == ""


def test_qwen35_nvfp4_native_offload_skips_lmcache_env(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    commands = vllm_adapter._build_cache_env_commands(
        "vllm",
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "_smart_card_token": "rtxpro5000-72",
            "_smart_feats": ["offload"],
        },
    )

    assert commands == []


def test_qwen_memcache_capability_follows_offload_whitelist_card_token():
    """Qwen MemCache 能力必须来自 offload 白名单。

    Day0 Qwen 矩阵对芯片敏感。局部模型 token 列表只能判断“是不是
    Qwen”，不能判断“这个 Qwen 在当前芯片上是否允许 offload”。这个回归
    测试确保 MemCache 与特性 gating 和 dry-run 命令生成使用同一份白名单行。
    """
    assert memcache_hybrid.is_qwen_day0_memcache_params(
        {
            "engine": "vllm_ascend",
            "model_name": "Qwen/Qwen3.5-27B",
            "model_path": "/models/Qwen/Qwen3.5-27B",
            "_smart_card_token": "910c",
        },
        "vllm_ascend",
    ) is True
    assert memcache_hybrid.is_qwen_day0_memcache_params(
        {
            "engine": "vllm_ascend",
            "model_name": "Qwen/Qwen3.5-27B",
            "model_path": "/models/Qwen/Qwen3.5-27B",
            "_smart_card_token": "910b",
        },
        "vllm_ascend",
    ) is False
