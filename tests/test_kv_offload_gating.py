import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402


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
            "model_name": "glm-4.7",
            "model_path": "/models/glm-4.7",
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
