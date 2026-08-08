import sys
from pathlib import Path
from types import SimpleNamespace


WINGS_ROOT = Path(__file__).resolve().parents[1] / "wings_control"
if str(WINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(WINGS_ROOT))

from core.port_plan import PortPlan  # noqa: E402
from core.start_args_compat import parse_launch_args  # noqa: E402
from core import wings_entry  # noqa: E402
from wings_control import wings_control as launcher  # noqa: E402


def test_rank0_propagates_resolved_backend_to_worker_dispatch(monkeypatch):
    launch_args = parse_launch_args(
        [
            "--model-name",
            "Kimi-K3",
            "--model-path",
            "/var/aispace/model/Kimi-K3",
            "--engine",
            "vllm",
        ]
    )
    monkeypatch.setattr(
        launcher,
        "build_launcher_plan",
        lambda *_args: SimpleNamespace(
            command="rank0-script",
            merged_params={"distributed_executor_backend": "mp"},
        ),
    )
    monkeypatch.setattr(launcher, "_write_start_command", lambda command: command)

    master_args = launcher._generate_rank0_script(
        launch_args,
        PortPlan(True, 17000, 8000, 19000),
        ["7.6.25.57", "7.6.25.58", "7.6.25.59", "7.6.25.60"],
        "7.6.25.57",
        "7.6.25.57",
    )

    assert launch_args.distributed_executor_backend == "ray"
    assert master_args.distributed_executor_backend == "mp"
    assert master_args.nnodes == 4
    assert master_args.node_rank == 0
    assert master_args.node_ips == "7.6.25.57,7.6.25.58,7.6.25.59,7.6.25.60"


def test_kimi_k3_910c_worker_merge_preserves_backend_port_only(monkeypatch):
    launch_args = SimpleNamespace(
        model_name="Kimi-K3-w4a8",
        model_path="/data/Kimi-K3-w4a8",
        distributed=True,
        nnodes=4,
        node_rank=2,
        head_node_addr="7.6.28.252",
        distributed_executor_backend="dp_deployment",
        node_ips="7.6.28.252,7.6.28.253,7.6.28.241,7.6.28.240",
        master_ip="7.6.28.252",
        to_namespace=lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        wings_entry,
        "load_and_merge_configs",
        lambda **_kwargs: {
            "engine": "vllm_ascend",
            "host": "0.0.0.0",
            "port": 8000,
            "_preserve_dp_worker_port": True,
            "engine_config": {
                "host": "0.0.0.0",
                "port": 8000,
            },
        },
    )

    merged = wings_entry._prepare_merged_params(
        launch_args,
        PortPlan(True, 18000, 8000, 19000),
        {"device": "ascend"},
    )

    assert "host" not in merged
    assert merged["port"] == 18000
    assert "host" not in merged["engine_config"]
    assert merged["engine_config"]["port"] == 18000
