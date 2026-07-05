import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from proxy import health_service  # noqa: E402


def test_startup_accel_route_is_served_before_proxy_catch_all(monkeypatch, tmp_path):
    state_file = tmp_path / "advanced_features.json"
    state_file.write_text(
        json.dumps(
            {
                "engine": "vllm",
                "features": {
                    "speculative_decode": True,
                    "lmcache_offload": False,
                },
                "variants": {
                    "speculative_decode": "mtp",
                    "lmcache_offload": None,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health_service.settings, "ADVANCED_FEATURES_FILE", str(state_file))

    client = TestClient(health_service.app)
    response = client.get("/v1/startup/accel")

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "msg": "",
        "data": {
            "engine": "vllm",
            "features": {
                "speculative_decode": True,
                "lmcache_offload": False,
            },
            "variants": {
                "speculative_decode": "mtp",
                "lmcache_offload": None,
            },
        },
    }
