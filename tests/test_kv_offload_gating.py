import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402
from core import wings_entry  # noqa: E402


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
    assert "export LMCACHE_CONFIG_FILE" not in rendered
    assert not (tmp_path / "lmcache_config.yaml").exists()


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


def test_deepseek_v4_flash_ascend_v021_uses_lmcache_patch(monkeypatch):
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
        assert "python3 /accel-volume/install.py --lmcache-target ascend-arm" in snippet


def test_deepseek_v4_flash_ascend_future_version_keeps_lmcache_patch_hook(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENGINE_VERSION", "v0.22.0-a2")

    target = wings_entry._resolve_lmcache_install_target(
        "vllm_ascend",
        {
            "engine": "vllm_ascend",
            "model_name": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_path": "/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
            "model_type": "llm",
            "_smart_feats": ["offload"],
        },
    )

    assert target == "ascend-arm"


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

    assert command == " --kv-offloading-backend native --kv-offloading-size 0"


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
