"""PD external-lb 启动路径的共享规划模块。

本文件是 PD external-lb 的“规则所有者”，刻意放在 ``core`` 而不是
``engines/vllm_adapter.py``：

1. 这里负责解析上层下发的 ``PD_*`` / ``TP_SIZE`` / ``DP_SIZE`` 环境契约；
2. 这里负责把 ``pd_config.json`` registry 条目合并成一个可执行 plan；
3. adapter 侧只消费 plan 产物并渲染 shell，不再自行解释 PD 拓扑；
4. ``config_loader.py`` 里保留若干旧私有函数名作为兼容 wrapper，避免外部测试或
   临时脚本直接调用旧函数时立刻断裂。

维护原则：
- PD external-lb 不是一个新 engine，而是 vLLM/vLLM-Ascend 的特殊启动拓扑；
- 新增模型/connector/env recipe 时优先改 ``pd_config.json``，不要新增 Python 分支；
- 所有涉及 PD_ROLE、TP/DP、DP_SIZE_LOCAL、PD_INDEX 的解析应优先收敛到本文件。
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from wings_control.utils.env_utils import (
        get_local_ip,
        get_master_ip,
        get_node_ips,
        get_pd_role_env,
    )
    from wings_control.utils.file_utils import load_json_config
except ImportError:
    from utils.env_utils import (  # noqa: F401
        get_local_ip,
        get_master_ip,
        get_node_ips,
        get_pd_role_env,
    )
    from utils.file_utils import load_json_config  # noqa: F401


logger = logging.getLogger(__name__)

DEFAULT_PD_CONFIG_FILE = "pd_config.json"

# PD 场景里 TP/DP 是上层 PD 拓扑的权威来源。即使用户或通用配置链路显式写入了
# tensor_parallel_size/data_parallel_size，也必须被当前 PD 角色的 env 拓扑覆盖。
# 这一点和普通单机/分布式推理不同，后续不要把它改回“用户显式值永远优先”。
PD_TOPOLOGY_KEYS = {"tensor_parallel_size", "data_parallel_size"}

# 按 (config_dir, filename) 缓存 registry。测试会通过传入临时 config_dir 或直接传
# registry 来隔离状态；生产路径只读取默认 pd_config.json。
_PD_CONFIG_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


@dataclass(frozen=True)
class PDParallelEnv:
    """当前 PD 角色的原始 TP/DP env 视图。

    这里保留 raw 值和来源 env 名称，而不是只返回 int：
    - fallback/final-guard 需要在日志中指出到底命中了哪个 env；
    - 非法值需要由调用方决定如何告警和是否跳过；
    - 解析优先级只能有一份，避免多个函数各自重写一套 ``os.getenv`` 链。
    """

    role: str
    role_prefix: str
    raw_tp: Optional[str]
    raw_dp: Optional[str]
    tp_source: Optional[str]
    dp_source: Optional[str]


@dataclass(frozen=True)
class PDExternalLbPlan:
    """PD external-lb 的跨模块稳定产物。

    ``config_loader`` 负责把 plan 写回 legacy params 字段，``vllm_adapter`` 仍按
    既有字段消费，避免一次重构同时改动 shell 渲染层。字段含义：

    - ``ext``：fork 脚本需要的拓扑与端口基准，最终写入 ``_pd_external_lb``；
    - ``engine_overrides``：registry 解析后的 engine 参数，最终写入
      ``engine_config`` 和 ``_pd_engine_overrides``，供 adapter 在模型默认注入后重申；
    - ``kv_transfer_config``：带 ``__PD_INDEX__`` / ``__PD_KVPORT__`` 占位符的 KV 配置；
    - ``env`` / ``strip_env``：PD isolated env builder 消费的角色环境和剔除列表。
    """

    arch: str
    role: str
    platform: str
    connector: str
    ext: Dict[str, Any]
    engine_overrides: Dict[str, Any]
    kv_transfer_config: Dict[str, Any]
    env: Dict[str, Any]
    strip_env: list[str]


def resolve_default_config_dir() -> str:
    """解析默认配置目录，语义与 ``config_loader`` 的默认目录保持一致。"""

    env_dir = os.getenv("WINGS_CONFIG_DIR", "").strip()
    if env_dir:
        return env_dir
    bundled_dir = Path(__file__).resolve().parents[1] / "config" / "defaults"
    if bundled_dir.exists():
        return str(bundled_dir)
    return "wings/config"


def reset_pd_config_cache() -> None:
    """清空 pd_config 缓存，供需要隔离 registry 状态的测试使用。"""

    _PD_CONFIG_CACHE.clear()


def load_pd_config(
    config_dir: Optional[str] = None,
    filename: str = DEFAULT_PD_CONFIG_FILE,
) -> Dict[str, Any]:
    """加载 ``pd_config.json`` 中的 ``pd_config`` registry 段。

    返回值按模型 architecture 做 key，通常包含一个 ``default`` 兜底条目。读取失败、
    文件缺失或格式不对时返回空 dict；调用方会自动降级到非 external-lb 的 PD fallback。
    """

    resolved_dir = config_dir or resolve_default_config_dir()
    key = (resolved_dir, filename)
    if key not in _PD_CONFIG_CACHE:
        cfg = load_json_config(os.path.join(resolved_dir, filename))
        _PD_CONFIG_CACHE[key] = cfg.get("pd_config", {}) if isinstance(cfg, dict) else {}
    return _PD_CONFIG_CACHE[key]


def _first_env_with_name(*names: str) -> Tuple[Optional[str], Optional[str]]:
    """按优先级返回第一个非空 env 值及其变量名。

    PD 拓扑存在通用别名和角色专属变量，例如 ``TP_SIZE``、``PD_TP_SIZE``、
    ``PD_PREFILL_TP_SIZE``。保留来源变量名可以让诊断日志解释“为什么是这个值”。
    """

    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value, name
    return None, None


def _first_env(*names: str) -> Optional[str]:
    return _first_env_with_name(*names)[0]


def _int(default: str | int, *names: str) -> int:
    """从一组 env 中解析 int；非法或缺失时回退到默认值。"""

    value = _first_env(*names)
    try:
        return int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def get_pd_role_prefix(pd_role: str) -> str:
    """把短角色 P/D 映射为 env 后缀 PREFILL/DECODE。"""

    return "PREFILL" if pd_role == "P" else "DECODE"


def read_pd_parallel_env(pd_role: Optional[str] = None) -> Optional[PDParallelEnv]:
    """读取当前 PD 角色的 TP/DP 原始 env。

    优先级固定为：
    - TP: ``TP_SIZE`` > ``PD_TP_SIZE`` > ``PD_{PREFILL|DECODE}_TP_SIZE``
    - DP: ``DP_SIZE`` > ``PD_DP_SIZE`` > ``PD_{PREFILL|DECODE}_DP_SIZE``

    注意本函数只读取、不解释为 int。这样 fallback、final-guard、日志诊断能共用同一套
    优先级，同时保留各自对非法值的处理方式。
    """

    role = pd_role or get_pd_role_env()
    if not role:
        return None
    role_prefix = get_pd_role_prefix(role)
    raw_tp, tp_source = _first_env_with_name(
        "TP_SIZE",
        "PD_TP_SIZE",
        f"PD_{role_prefix}_TP_SIZE",
    )
    raw_dp, dp_source = _first_env_with_name(
        "DP_SIZE",
        "PD_DP_SIZE",
        f"PD_{role_prefix}_DP_SIZE",
    )
    return PDParallelEnv(
        role=role,
        role_prefix=role_prefix,
        raw_tp=raw_tp,
        raw_dp=raw_dp,
        tp_source=tp_source,
        dp_source=dp_source,
    )


def resolve_pd_parallel_overrides(
    pd_role: Optional[str] = None,
) -> Tuple[Optional[PDParallelEnv], Dict[str, Any], Dict[str, str]]:
    """把 PD TP/DP env 转为 engine_config overrides。

    返回三元组：
    - ``PDParallelEnv``：原始读取结果；
    - ``overrides``：解析成功的 ``tensor_parallel_size`` / ``data_parallel_size``；
    - ``invalid``：解析失败的 env 名和值，调用方负责写日志。

    该函数用于 ``_apply_pd_topology_fallback``，也可作为后续新增 guard 的唯一入口。
    """

    parallel_env = read_pd_parallel_env(pd_role)
    if parallel_env is None:
        return None, {}, {}

    overrides: Dict[str, Any] = {}
    invalid: Dict[str, str] = {}
    if parallel_env.raw_tp is not None:
        try:
            overrides["tensor_parallel_size"] = int(parallel_env.raw_tp)
        except (TypeError, ValueError):
            invalid[parallel_env.tp_source or "TP_SIZE"] = parallel_env.raw_tp
    if parallel_env.raw_dp is not None:
        try:
            overrides["data_parallel_size"] = int(parallel_env.raw_dp)
        except (TypeError, ValueError):
            invalid[parallel_env.dp_source or "DP_SIZE"] = parallel_env.raw_dp
    return parallel_env, overrides, invalid


def resolve_pd_external_lb_params() -> Optional[Dict[str, Any]]:
    """解析 external-lb fork 脚本需要的 PD 拓扑参数。

    触发条件是存在 ``PD_ROLE`` 且当前角色 DP 合法。未显式设置 DP 时默认 1，
    这用于覆盖 1P1D 场景：即使没有 ``DP_SIZE``，也应走 registry 管理的
    external-lb recipe，而不是退回 standalone PD 的老逻辑。

    返回值会直接进入 ``_pd_external_lb``，字段名需要保持向后兼容；adapter 中的
    shell 渲染依赖这些 key。
    """

    role = get_pd_role_env()
    if not role:
        return None

    role_prefix = get_pd_role_prefix(role)

    # DP_SIZE/PD_DP_SIZE 优先表达“本角色全局 DP”。没有显式值时默认 1，保证
    # 1P1D 也能命中 pd_config registry；只有小于 1 的值才视为不启用 external-lb。
    raw = _first_env("DP_SIZE", "PD_DP_SIZE", f"PD_{role_prefix}_DP_SIZE")
    if raw is None:
        dp_size = 1
    else:
        try:
            dp_size = int(raw)
        except (TypeError, ValueError):
            dp_size = 1
    if dp_size < 1:
        return None

    # TP 是单个 engine 实例的张量并行度，仍由 PD env 契约决定，不能回退到
    # device_count；device_count 属于普通单机推理的兜底逻辑。
    tp_size = _int("1", "TP_SIZE", "PD_TP_SIZE", f"PD_{role_prefix}_TP_SIZE")

    # DP_SIZE_LOCAL 表示当前 pod 内要 fork 的 engine 实例数。缺省时保守为 1，
    # 不根据设备数量自动推导，避免上层没有明确拓扑时本地擅自多 fork。
    dp_local_raw = _first_env("DP_SIZE_LOCAL", "PD_DP_SIZE_LOCAL")
    if dp_local_raw is not None:
        try:
            dp_size_local = int(dp_local_raw)
        except (TypeError, ValueError):
            dp_size_local = 1
    else:
        dp_size_local = 1
        logger.warning(
            "[PD external-lb] DP_SIZE_LOCAL/PD_DP_SIZE_LOCAL not set (role=%s); "
            "defaulting dp_size_local=1. Set DP_SIZE_LOCAL to match the number "
            "of engine instances forked in this pod.",
            role,
        )
    dp_size_local = min(dp_size_local, dp_size)

    # data-parallel-address 代表当前角色 DP 域 head。优先信任平台显式透传，
    # 否则回退到本 pod IP，保证本地/单 pod 调试仍可生成可运行脚本。
    dp_address = (
        _first_env("Master_IP", "MASTER_IP", "PD_DP_ADDRESS")
        or get_master_ip()
        or _first_env("RANK_IP", "HOST_IP")
        or get_local_ip()
        or ""
    )

    # rpc-port 按角色硬编码而不是读取 env。这样同角色每个 pod 都能得到一致端口，
    # P/D 又天然分离，避免平台动态端口导致 DP 域 rendezvous 不一致。
    rpc_port = "12890" if role == "P" else "12777"

    # NODE_IPS 是“当前角色域”的 pod IP 列表，顺序即 DP rank 派生顺序。这里刻意
    # 优先使用 RANK_IP：HOST_IP 在同宿主机多 pod 场景可能相同，会造成 rank 撞车。
    node_ips = [ip.strip() for ip in (get_node_ips() or "").split(",") if ip.strip()]
    explicit_start = _first_env("PD_DP_RANK_START")
    if explicit_start is not None:
        dp_rank_start = _int("0", "PD_DP_RANK_START")
    else:
        host_ip = (_first_env("RANK_IP", "HOST_IP") or get_local_ip() or "").strip()
        if host_ip in node_ips:
            node_rank = node_ips.index(host_ip)
        else:
            node_rank = 0
            if len(node_ips) > 1:
                logger.error(
                    "[PD external-lb] local IP %r is not in NODE_IPS %r; "
                    "falling back dp_rank_start=0. Multi-node PD ranks may collide.",
                    host_ip,
                    node_ips,
                )
        dp_rank_start = node_rank * dp_size_local

    # PD_INDEX 是跨 P/D 的全局实例编号。上层显式传递时完全信任；未传递时只给
    # 1P1D 默认值，复杂多实例场景应由上层负责下发，wings 不在这里推导。
    try:
        pd_index_base = int(os.getenv("PD_INDEX", ""))
    except (TypeError, ValueError):
        pd_index_base = 0 if role == "P" else 1

    result = {
        "role": role,
        "dp_size": dp_size,
        "tp_size": tp_size,
        "dp_size_local": dp_size_local,
        "dp_rank_start": dp_rank_start,
        "dp_address": dp_address,
        "rpc_port": str(rpc_port),
        "pd_index_base": pd_index_base,
    }
    logger.info(
        "[PD external-lb params] role=%s tp_size=%d dp_size=%d dp_size_local=%d "
        "dp_rank_start=%d dp_address=%s rpc_port=%s pd_index_base=%d "
        "(PD_DECODE_TP_SIZE=%s, PD_PREFILL_TP_SIZE=%s, PD_DECODE_DP_SIZE=%s, "
        "PD_PREFILL_DP_SIZE=%s)",
        result["role"],
        result["tp_size"],
        result["dp_size"],
        result["dp_size_local"],
        result["dp_rank_start"],
        result["dp_address"],
        result["rpc_port"],
        result["pd_index_base"],
        os.getenv("PD_DECODE_TP_SIZE", "<unset>"),
        os.getenv("PD_PREFILL_TP_SIZE", "<unset>"),
        os.getenv("PD_DECODE_DP_SIZE", "<unset>"),
        os.getenv("PD_PREFILL_DP_SIZE", "<unset>"),
    )
    return result


def is_pd_external_lb_active() -> bool:
    """供 launcher role gate 使用的稳定入口。

    该函数刻意放在本模块，避免 ``wings_control.py`` 为了判定角色而 import
    ``config_loader``，从而触发配置加载链、模型识别或其它副作用。
    """

    return resolve_pd_external_lb_params() is not None


def _peer_topology(prefix: str) -> Optional[Dict[str, int]]:
    """读取对端角色拓扑，用于填充 KV connector 的全局 prefill/decode 映射。"""

    dp = os.getenv(f"PD_{prefix}_DP_SIZE")
    tp = os.getenv(f"PD_{prefix}_TP_SIZE")
    if dp and tp:
        try:
            return {"dp_size": int(dp), "tp_size": int(tp)}
        except (TypeError, ValueError):
            return None
    return None


def build_pd_external_lb_kv(entry: Dict[str, Any], ext: Dict[str, Any]) -> Dict[str, Any]:
    """根据 registry entry 和本角色拓扑构造 ``kv_transfer_config``。

    这里不直接写具体端口和 engine_id，而是保留占位符：
    - ``__PD_KVPORT__``：由 fork 脚本按 ``PD_INDEX`` 替换为全局唯一 KV port；
    - ``__PD_INDEX__``：由 shell 运行时展开，确保 P/D 多实例不会撞 engine_id。

    这样 registry 描述的是“模型/connector recipe”，实例编号和端口分配仍由启动脚本
    在实际 fork 时决定。
    """

    role = ext["role"]
    kv_role = "kv_producer" if role == "P" else "kv_consumer"
    me = {"dp_size": ext["dp_size"], "tp_size": ext["tp_size"]}

    if role == "P":
        prefill, decode = me, _peer_topology("DECODE")
        if decode is None:
            logger.warning(
                "[PD external-lb] peer(decode) topology unknown "
                "(set PD_DECODE_DP_SIZE/PD_DECODE_TP_SIZE); KV mapping may be wrong"
            )
            decode = me
    else:
        decode, prefill = me, _peer_topology("PREFILL")
        if prefill is None:
            logger.warning(
                "[PD external-lb] peer(prefill) topology unknown "
                "(set PD_PREFILL_DP_SIZE/PD_PREFILL_TP_SIZE); KV mapping may be wrong"
            )
            prefill = me

    extra = {"prefill": prefill, "decode": decode}

    # extra_config 合并顺序：全局模型 extra < 角色 extra。角色 extra 用于处理
    # decode/prefill 单侧差异，例如 consumer 专属 buffer device 或 connector flag。
    model_extra = entry.get("extra_config") or {}
    if model_extra:
        extra.update(model_extra)
    role_key = "prefill" if role == "P" else "decode"
    role_extra = (entry.get(role_key) or {}).get("extra_config") or {}
    if role_extra:
        extra.update(role_extra)
    if (
        os.getenv("PD_DISABLE_ASCEND_DIRECT", "").strip().lower()
        in ("1", "true", "yes", "on")
        and "use_ascend_direct" in extra
    ):
        # 这是部署级临时规避开关，不写回 registry。默认保持官方 ADXL 行为，
        # 只有显式设置该 env 的部署才移除 use_ascend_direct。
        extra.pop("use_ascend_direct", None)
        logger.warning(
            "[PD external-lb] PD_DISABLE_ASCEND_DIRECT is active; removed "
            "use_ascend_direct from kv_connector_extra_config"
        )

    cfg = {
        "kv_connector": entry["connector"],
        "kv_role": kv_role,
        "kv_port": "__PD_KVPORT__",
        "kv_connector_extra_config": extra,
    }
    module_path = entry.get("connector_module_path")
    if module_path:
        cfg["kv_connector_module_path"] = module_path
    kv_buffer_device = entry.get("kv_buffer_device")
    if kv_buffer_device:
        cfg["kv_buffer_device"] = kv_buffer_device
    cfg["engine_id"] = "__PD_INDEX__"
    return cfg


def resolve_ascend_platform() -> str:
    """解析 Ascend 平台标识，供 registry 的 ``platform_overrides`` 使用。

    显式信号优先：``WINGS_ASCEND_PLATFORM`` / ``ASCEND_PLATFORM`` /
    ``ENGINE_IMAGE_FLAVOR``。无显式信号时才读取 ``ENGINE_VERSION=-a3`` 或
    ``ASCEND_A3_ENABLE``。解析不到返回空串，表示“不应用平台 overlay”。
    """

    val = (
        os.getenv("WINGS_ASCEND_PLATFORM")
        or os.getenv("ASCEND_PLATFORM")
        or os.getenv("ENGINE_IMAGE_FLAVOR")
        or ""
    ).strip().lower()
    if val in {"a3", "atlas-a3", "atlas_a3", "910c"}:
        return "a3"
    if val in {"a2", "atlas-a2", "atlas_a2", "910b"}:
        return "a2"
    if os.getenv("ENGINE_VERSION", "").strip().lower().endswith("-a3") or os.getenv(
        "ASCEND_A3_ENABLE",
        "",
    ).strip().lower() in {"1", "true", "yes"}:
        return "a3"
    return ""


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深合并 registry 配置，并确保不污染缓存中的原始 entry。"""

    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _select_registry_entry(
    registry: Dict[str, Any],
    arch: str,
) -> Optional[Dict[str, Any]]:
    """按模型 architecture 选择 registry entry，找不到时回退 default。"""

    entry = registry.get(arch) or registry.get("default")
    return copy.deepcopy(entry) if entry else None


def _apply_platform_overrides(entry: Dict[str, Any], arch: str) -> Tuple[Dict[str, Any], str]:
    """应用平台 overlay。

    ``default_platform`` 是条目级 opt-in：只在没有显式平台信号时生效，用于那些
    “漏设平台时应该按某个平台 recipe 运行”的模型。没有 default_platform 的条目仍保持
    老语义：解析不到平台就不应用 overlay。
    """

    platform = resolve_ascend_platform()
    overrides = entry.pop("platform_overrides", None)
    default_platform = entry.pop("default_platform", None)
    if not platform and default_platform:
        platform = default_platform
        logger.info(
            "[PD external-lb] no explicit platform signal; using entry default "
            "platform '%s' (arch=%s)",
            platform,
            arch,
        )
    if overrides and platform and platform in overrides:
        entry = _merge_configs(entry, overrides[platform])
        logger.info("[PD external-lb] applied platform_overrides[%s] for arch=%s", platform, arch)
    return entry, platform


def build_pd_external_lb_plan(
    cmd_known_params: Dict[str, Any],
    model_info: Any,
    registry: Optional[Dict[str, Any]] = None,
) -> Optional[PDExternalLbPlan]:
    """生成 PD external-lb plan。

    这是本模块的主入口。它只做“解析和规划”，不直接修改 ``cmd_known_params``；
    ``config_loader`` 负责把 plan 写回 legacy 字段。这样的边界有两个好处：
    - 单元测试可以直接验证 plan，不必跑完整配置合并链；
    - 后续如果 adapter 改为直接消费 plan，也不需要再迁移 env/registry 解析逻辑。
    """

    ext = resolve_pd_external_lb_params()
    if not ext:
        return None

    registry = registry if registry is not None else load_pd_config()
    arch = getattr(model_info, "model_architecture", None) or ""
    entry = _select_registry_entry(registry, arch)
    if not entry:
        logger.warning(
            "[PD external-lb] no registry entry for arch=%s and no default; "
            "fall back to standalone PD",
            arch,
        )
        return None

    entry, platform = _apply_platform_overrides(entry, arch)
    role = ext["role"]
    role_key = "prefill" if role == "P" else "decode"
    explicit = set(cmd_known_params.get("_explicit_cli_keys") or [])

    # engine 参数优先级：用户显式非拓扑参数 > registry > 基础默认。
    # 例外是 TP/DP：在 PD 场景中它们永远以 PD env 拓扑为准。
    merged_engine = {
        **entry.get("common", {}),
        **(entry.get(role_key) or {}).get("engine", {}),
    }
    merged_engine["tensor_parallel_size"] = ext["tp_size"]
    merged_engine["data_parallel_size"] = ext["dp_size"]
    engine_overrides = {
        key: copy.deepcopy(value)
        for key, value in merged_engine.items()
        if key not in explicit or key in PD_TOPOLOGY_KEYS
    }

    planned_ext = copy.deepcopy(ext)

    # kv_port_base/bootstrap_base 是脚本渲染需要的基准值。V1/Hybrid connector 最终
    # 仍按 PD_INDEX 派生全局唯一端口，这里保留字段是为了兼容现有 adapter 消费结构。
    try:
        planned_ext["kv_port_base"] = int(entry["kv_port"][role])
    except (KeyError, TypeError, ValueError):
        planned_ext["kv_port_base"] = 30000 if role == "P" else 30100
    planned_ext["bootstrap_base"] = int(
        os.getenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", "23000" if role == "P" else "23100")
    )
    planned_ext["connector"] = entry["connector"]

    # env 合并顺序与 engine 一致：common < 当前角色。strip_env 同时叠加 common 和角色级，
    # 由 isolated env builder 在最终 shell 片段中剔除多余变量。
    env = {
        **entry.get("common_env", {}),
        **(entry.get(role_key) or {}).get("env", {}),
    }
    strip_env = list(entry.get("strip_env", [])) + list(
        (entry.get(role_key) or {}).get("strip_env", [])
    )

    return PDExternalLbPlan(
        arch=arch,
        role=role,
        platform=platform,
        connector=entry["connector"],
        ext=planned_ext,
        engine_overrides=engine_overrides,
        kv_transfer_config=build_pd_external_lb_kv(entry, planned_ext),
        env=env,
        strip_env=strip_env,
    )
