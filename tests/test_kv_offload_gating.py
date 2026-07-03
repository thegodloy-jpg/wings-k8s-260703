import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402
from core import wings_entry  # noqa: E402


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
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")
    monkeypatch.setenv("ENABLE_KV_MEM_OFFLOAD", "true")
    monkeypatch.setenv("KV_MEM_OFFLOAD_SIZE", "40")
    monkeypatch.setenv("ENABLE_KV_DISK_OFFLOAD", "false")
    for key in (
        "LMCACHE_LOCAL_CPU",
        "LMCACHE_MAX_LOCAL_CPU_SIZE",
        "LMCACHE_TRACK_USAGE",
        "LMCACHE_USE_LAYERWISE",
        "LMCACHE_NUMA_MODE",
        "LMCACHE_EXTRA_CONFIG",
        "LMCACHE_LOOKUP_SERVER_WORKER_IDS",
    ):
        monkeypatch.delenv(key, raising=False)

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
    assert "export LMCACHE_TRACK_USAGE=false" in rendered
    assert "export LMCACHE_USE_LAYERWISE=False" in rendered
    assert "export LMCACHE_NUMA_MODE=auto" in rendered
    assert "export LMCACHE_CHUNK_SIZE=1024" in rendered
    assert "export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3" in rendered
    assert "CPUOffloadingConnector" not in rendered


def test_deepseek_v4_flash_ascend_installs_lmcache_patch(monkeypatch):
    monkeypatch.setenv("ENABLE_KV_OFFLOAD", "true")

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
