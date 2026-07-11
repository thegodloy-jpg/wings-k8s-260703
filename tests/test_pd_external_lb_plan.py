import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


_ROOT = Path(__file__).resolve().parents[1]
_CONTROL_DIR = _ROOT / "wings_control"
sys.path.insert(0, str(_CONTROL_DIR))

from core import config_loader, pd_external_lb  # noqa: E402

# 这里不能直接 ``import wings_control as launcher``：
# 多数历史测试会把 ``.../wings_control`` 插到 sys.path[0]，直接 import 会把
# ``wings_control.py`` 注册为顶层模块 ``wings_control``，后续再按 package 形式导入
# ``wings_control.log_analyzer`` 时会失败。用独立模块名加载可以测试 _determine_role，
# 同时不污染 package 名称。
_LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "_pd_external_lb_launcher",
    _CONTROL_DIR / "wings_control.py",
)
launcher = importlib.util.module_from_spec(_LAUNCHER_SPEC)
assert _LAUNCHER_SPEC.loader is not None
sys.modules[_LAUNCHER_SPEC.name] = launcher
_LAUNCHER_SPEC.loader.exec_module(launcher)


_PD_ENV_KEYS = (
    "ASCEND_A3_ENABLE",
    "ASCEND_PLATFORM",
    "DP_SIZE",
    "DP_SIZE_LOCAL",
    "ENGINE_IMAGE_FLAVOR",
    "ENGINE_VERSION",
    "HOST_IP",
    "MASTER_IP",
    "NODE_IPS",
    "PD_DECODE_DP_SIZE",
    "PD_DECODE_TP_SIZE",
    "PD_DP_ADDRESS",
    "PD_DP_RANK_START",
    "PD_DP_SIZE",
    "PD_DP_SIZE_LOCAL",
    "PD_INDEX",
    "PD_PREFILL_DP_SIZE",
    "PD_PREFILL_TP_SIZE",
    "PD_ROLE",
    "PD_TP_SIZE",
    "RANK_IP",
    "TP_SIZE",
    "VLLM_MOONCAKE_BOOTSTRAP_PORT",
    "WINGS_ASCEND_PLATFORM",
)


def _clear_pd_env(monkeypatch):
    for key in _PD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _set_prefill_env(monkeypatch):
    _clear_pd_env(monkeypatch)
    monkeypatch.setenv("PD_ROLE", "P")
    monkeypatch.setenv("PD_INDEX", "3")
    monkeypatch.setenv("PD_PREFILL_DP_SIZE", "2")
    monkeypatch.setenv("PD_PREFILL_TP_SIZE", "4")
    monkeypatch.setenv("PD_DECODE_DP_SIZE", "8")
    monkeypatch.setenv("PD_DECODE_TP_SIZE", "1")
    monkeypatch.setenv("DP_SIZE_LOCAL", "1")
    monkeypatch.setenv("MASTER_IP", "10.0.0.10")
    monkeypatch.setenv("RANK_IP", "10.0.0.10")
    monkeypatch.setenv("HOST_IP", "10.0.0.10")
    monkeypatch.setenv("NODE_IPS", "10.0.0.10,10.0.0.11")
    monkeypatch.setenv("WINGS_ASCEND_PLATFORM", "a2")


def _registry():
    """构造最小 PD registry，用于验证 plan 合并语义而不是依赖真实 pd_config.json。"""

    return {
        "FakeForCausalLM": {
            "connector": "MooncakeConnectorV1",
            "kv_port": {"P": "30010", "D": "30110"},
            "common": {
                "enable_prefix_caching": False,
                "max_model_len": 1024,
                "compilation_config": {"level": 1},
            },
            "prefill": {
                "engine": {
                    "enable_prefix_caching": True,
                    "max_num_seqs": 16,
                },
                "env": {"PREFILL_ONLY": "1"},
                "strip_env": ["PREFILL_STRIP"],
            },
            "decode": {
                "engine": {"max_num_seqs": 8},
                "env": {"DECODE_ONLY": "1"},
            },
            "common_env": {"COMMON_ENV": "1"},
            "strip_env": ["COMMON_STRIP"],
            "extra_config": {"use_ascend_direct": True},
            "platform_overrides": {
                "a2": {
                    "common": {"compilation_config": {"a2": True}},
                    "common_env": {"PLATFORM_ENV": "a2"},
                }
            },
        }
    }


def test_pd_external_lb_plan_merges_registry_and_preserves_explicit_keys(monkeypatch):
    """plan 层应同时保护 registry 合并、显式 key 规则和 PD 拓扑优先级。"""

    _set_prefill_env(monkeypatch)
    cmd = {
        "engine_config": {},
        "_explicit_cli_keys": [
            "data_parallel_size",
            "max_model_len",
            "tensor_parallel_size",
        ],
    }
    model_info = SimpleNamespace(model_architecture="FakeForCausalLM")

    plan = pd_external_lb.build_pd_external_lb_plan(cmd, model_info, _registry())

    assert plan is not None
    assert plan.role == "P"
    assert plan.platform == "a2"
    assert plan.connector == "MooncakeConnectorV1"
    assert plan.ext["dp_size"] == 2
    assert plan.ext["tp_size"] == 4
    assert plan.ext["dp_rank_start"] == 0
    assert plan.ext["pd_index_base"] == 3
    assert plan.ext["kv_port_base"] == 30010
    assert plan.ext["bootstrap_base"] == 23000

    assert plan.engine_overrides["tensor_parallel_size"] == 4
    assert plan.engine_overrides["data_parallel_size"] == 2
    assert plan.engine_overrides["enable_prefix_caching"] is True
    assert plan.engine_overrides["compilation_config"] == {"level": 1, "a2": True}
    assert "max_model_len" not in plan.engine_overrides

    assert plan.env == {
        "COMMON_ENV": "1",
        "PLATFORM_ENV": "a2",
        "PREFILL_ONLY": "1",
    }
    assert plan.strip_env == ["COMMON_STRIP", "PREFILL_STRIP"]
    assert plan.kv_transfer_config["kv_connector"] == "MooncakeConnectorV1"
    assert plan.kv_transfer_config["kv_role"] == "kv_producer"
    assert plan.kv_transfer_config["engine_id"] == "__PD_INDEX__"
    assert plan.kv_transfer_config["kv_connector_extra_config"]["prefill"] == {
        "dp_size": 2,
        "tp_size": 4,
    }
    assert plan.kv_transfer_config["kv_connector_extra_config"]["decode"] == {
        "dp_size": 8,
        "tp_size": 1,
    }


def test_config_loader_apply_pd_external_lb_keeps_legacy_runtime_fields(monkeypatch):
    """config_loader 适配层必须继续写出 adapter 消费的 legacy _pd_* 字段。"""

    _set_prefill_env(monkeypatch)
    monkeypatch.setattr(config_loader, "_load_pd_config", _registry)

    params = {
        "distributed": True,
        "engine_config": {"max_model_len": 4096},
        "_explicit_cli_keys": ["max_model_len"],
    }
    model_info = SimpleNamespace(model_architecture="FakeForCausalLM")

    config_loader._apply_pd_external_lb(params, model_info)

    engine_config = params["engine_config"]
    kv_config = json.loads(engine_config["kv_transfer_config"])
    assert params["distributed"] is False
    assert params["_pd_external_lb"]["connector"] == "MooncakeConnectorV1"
    assert params["_pd_external_lb"]["kv_port_base"] == 30010
    assert params["_pd_engine_overrides"]["tensor_parallel_size"] == 4
    assert params["_pd_engine_overrides"]["data_parallel_size"] == 2
    assert params["_pd_env"]["PREFILL_ONLY"] == "1"
    assert params["_pd_strip_env"] == ["COMMON_STRIP", "PREFILL_STRIP"]
    assert engine_config["max_model_len"] == 4096
    assert engine_config["tensor_parallel_size"] == 4
    assert engine_config["data_parallel_size"] == 2
    assert kv_config["kv_role"] == "kv_producer"
    assert kv_config["engine_id"] == "__PD_INDEX__"


def test_pd_external_lb_role_gate_does_not_import_config_loader(monkeypatch):
    """launcher role gate 只允许依赖轻量 helper，不能重新耦合到 config_loader。"""

    _set_prefill_env(monkeypatch)
    monkeypatch.setenv("DISTRIBUTED", "true")

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "core.config_loader":
            raise AssertionError("role gate must not import core.config_loader")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert launcher._determine_role() == "standalone"
