import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from distributed import worker  # noqa: E402


def test_worker_uses_local_model_path_when_mounts_differ(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/var/ai-model/Kimi-K3")
    launch_kwargs = {
        "model_path": "/var/aispace/model/Kimi-K3",
        "node_rank": 1,
        "distributed_executor_backend": "mp",
    }

    worker._apply_worker_local_model_path(launch_kwargs)

    assert launch_kwargs["model_path"] == "/var/ai-model/Kimi-K3"
    assert launch_kwargs["node_rank"] == 1


def test_worker_keeps_dispatched_model_path_without_local_override(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    launch_kwargs = {
        "model_path": "/models/Kimi-K3",
        "distributed_executor_backend": "mp",
    }

    worker._apply_worker_local_model_path(launch_kwargs)

    assert launch_kwargs["model_path"] == "/models/Kimi-K3"


def test_worker_does_not_change_non_mp_model_path(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/worker/local/model")
    launch_kwargs = {
        "model_path": "/master/dispatched/model",
        "distributed_executor_backend": "ray",
    }

    worker._apply_worker_local_model_path(launch_kwargs)

    assert launch_kwargs["model_path"] == "/master/dispatched/model"
