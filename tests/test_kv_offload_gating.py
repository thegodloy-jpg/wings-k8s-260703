import sys
import json
import logging
import inspect
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402
from core import wings_entry  # noqa: E402
from core import config_loader  # noqa: E402
from core.port_plan import derive_port_plan  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from features.kv_offload.memcache import hybrid as memcache_hybrid  # noqa: E402


def test_memcache_helpers_are_not_owned_by_vllm_adapter():
    assert not hasattr(vllm_adapter, "build_memcache_ascend_store_config")
    assert not hasattr(vllm_adapter, "is_kimi_k27_code_memcache_params")


def test_memcache_transfer_config_is_owned_by_config_loader():
    assert not hasattr(memcache_hybrid, "build_memcache_ascend_store_config")
    assert hasattr(config_loader, "_build_memcache_ascend_store_config")


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
    assert "ock.mmc.local_service.dram.size = ${WINGS_MEMCACHE_DRAM_GB}GB" in fragment["engine_prelude"]
    assert "ock.mmc.local_service.dram.size" not in source


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


def test_lmcache_env_exports_do_not_leak_l2_child_values_when_l2_switches_are_off(monkeypatch):
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
    assert "export ENABLE_KV_OFFLOAD=true" in rendered
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


def test_deepseek_v4_flash_ascend_lmcache_defaults_cpu_pool_when_whitelist_forces_offload(monkeypatch):
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
    assert "export LMCACHE_LOCAL_CPU=True" in rendered
    assert "export LMCACHE_MAX_LOCAL_CPU_SIZE=40" in rendered


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
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "102400")
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
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "102400")
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
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "102400")
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

    target = wings_entry._resolve_lmcache_install_target("vllm_ascend", params)
    snippet = wings_entry._build_lmcache_install_snippet("vllm_ascend", params)
    wings_entry._write_advanced_features_json("vllm_ascend", params)

    data = json.loads((tmp_path / "advanced_features.json").read_text(encoding="utf-8"))
    assert target is None
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
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "153600")
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
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "80")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "200")
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
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "102400")
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

    target = wings_entry._resolve_lmcache_install_target("vllm_ascend", params)
    variant = vllm_adapter.resolve_offload_variant(params, "vllm_ascend")

    assert target == "ascend-arm"
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

        target = wings_entry._resolve_lmcache_install_target("vllm_ascend", params)
        snippet = wings_entry._build_lmcache_install_snippet("vllm_ascend", params)

        assert target == "ascend-arm"
        assert (
            "python install.py --config "
            "'{\"packages\": [\"lmcache-ascend:v0.4.5\"]}'"
        ) in snippet
        assert 'cd "/accel-volume"' in snippet
        assert "--lmcache-target ascend-arm" not in snippet


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
    target = wings_entry._resolve_lmcache_install_target(
        "vllm_ascend",
        params,
    )
    snippet = wings_entry._build_lmcache_install_snippet("vllm_ascend", params)

    assert target == "ascend-arm"
    assert "lmcache-ascend:v0.4.5" in snippet
    assert "--lmcache-target ascend-arm" not in snippet


def test_nvidia_lmcache_keeps_target_install_command():
    snippet = wings_entry._render_lmcache_install_snippet("nvidia-x86")

    assert (
        "python3 /accel-volume/install.py --lmcache-target nvidia-x86"
        in snippet
    )
    assert "lmcache-ascend:v0.4.5" not in snippet


def test_kimi_k27_code_memcache_skips_lmcache_patch(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

    target = wings_entry._resolve_lmcache_install_target(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Kimi-K2.7-Code",
            "model_path": "/harbor_data/Kimi-K2.7-Code",
            "model_type": "llm",
            "_smart_feats": ["offload"],
        },
    )

    assert target is None


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
    assert "ock.mmc.local_service.dram.size = ${WINGS_MEMCACHE_DRAM_GB}GB" in fragment["engine_prelude"]
    assert "MMC_LOCAL_CONFIG_PATH" in fragment["engine_prelude"]
    assert "MMC_META_CONFIG_PATH" in fragment["master_script"]


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
    ("model_name", "card_token", "expected_meta_port", "expected_config_port"),
    [
        ("Qwen/Qwen3.5-27B", "910c", 50051, 50061),
        ("Qwen/Qwen3.6-27B", "910c", 50071, 50081),
        ("Eco-Tech/Qwen3.6-27B-w8a8", "910c", 50071, 50081),
        ("Qwen/Qwen3.6-35B-A3B", "910c", 50071, 50081),
        ("Eco-Tech/Qwen3.6-35B-A3B-w8a8", "910c", 50071, 50081),
    ],
)
def test_qwen_day0_memcache_ports_follow_offload_whitelist(
    monkeypatch,
    model_name,
    card_token,
    expected_meta_port,
    expected_config_port,
):
    """Qwen Day0 MemCache 默认端口属于场景数据。

    页面仍然可以通过 WINGS_MEMCACHE_META_SERVICE_URL 和
    WINGS_MEMCACHE_CONFIG_STORE_URL 覆盖渲染后的 URL。页面未覆盖时，
    Qwen 必须使用同一条 offload 白名单场景行中的模型+芯片端口。
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


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.6-27B",
        "Eco-Tech/Qwen3.6-27B-w8a8",
        "Qwen/Qwen3.6-35B-A3B",
        "Eco-Tech/Qwen3.6-35B-A3B-w8a8",
    ],
)
def test_qwen_day0_910b_never_enables_memcache(monkeypatch, model_name):
    """当前 Qwen3.5/3.6 910B 场景只保留 MTP，不允许 MemCache 卸载。"""
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

    assert memcache_hybrid.is_qwen_day0_memcache_params(params, "vllm_ascend") is False
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
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "45056")

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
    monkeypatch.delenv("LMCACHE_MAX_LOCAL_CPU_SIZE", raising=False)

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
    engine_index = command.index("exec vllm serve /harbor_data/Kimi-K2.7-Code")
    assert prelude_index < engine_index
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


def test_qwen35_nvfp4_uses_native_kv_offload_cli(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.delenv("LMCACHE_MAX_LOCAL_CPU_SIZE", raising=False)

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == " --kv-offloading-backend native --kv-offloading-size 200"


def test_qwen35_nvfp4_native_offload_reuses_page_size_without_per_card_scaling(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "200")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == " --kv-offloading-backend native --kv-offloading-size 200"


def test_qwen35_nvfp4_native_offload_prefers_kv_mem_size_without_per_card_scaling(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "80")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "200")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == " --kv-offloading-backend native --kv-offloading-size 80"


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
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "180224")

    command = vllm_adapter._build_kv_offload_cmd(
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "device_count": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == ""


def test_qwen35_nvfp4_native_offload_auto_uses_formula_without_per_card_scaling(monkeypatch):
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
            "_smart_feats": ["spec", "offload"],
        },
        "vllm",
    )

    assert command == " --kv-offloading-backend native --kv-offloading-size 121"


def test_qwen35_nvfp4_native_offload_legacy_auto_floor_variant_matches_cli(monkeypatch):
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
        "_smart_feats": ["spec", "offload"],
    }

    command = vllm_adapter._build_kv_offload_cmd(params, "vllm")
    variant = vllm_adapter.resolve_offload_variant(params, "vllm")

    assert command == ""
    assert variant == "native_kv_offloading_backend+auto+floor_disabled"


def test_deepseek_v4_flash_native_offload_reuses_page_size_without_per_card_scaling(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
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
    monkeypatch.setenv("AVAILABLE_POD_MEM_SIZE", "102400")

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

    target = wings_entry._resolve_lmcache_install_target(
        "vllm",
        {
            "engine": "vllm",
            "model_name": "Qwen3.5-397B-A17B-NVFP4",
            "model_path": "/models/Qwen3.5-397B-A17B-NVFP4",
            "_smart_feats": ["spec", "offload"],
        },
    )

    assert target is None


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
