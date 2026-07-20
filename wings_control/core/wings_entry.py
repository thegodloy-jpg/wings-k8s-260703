
"""将 launcher 参数转换成 engine 启动计划。

它是 launcher 控制链路里的中枢桥接层：
- 上游拿到的是 CLI/环境变量；
- 下游需要的是一段可执行的 shell 脚本；
- 中间还要结合硬件探测、默认配置、用户配置和端口规划。

最终产物 `LauncherPlan.command` 会被写入共享卷，供 engine 容器执行。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings
from core.config_loader import load_and_merge_configs
from utils.file_utils import safe_write_file, WriteOptions
from core.engine_manager import start_engine_service
from core.hardware_detect import detect_hardware
from core.port_plan import PortPlan
from core.start_args_compat import LaunchArgs
from engines.vllm_adapter import (
    resolve_speculative_strategy,
    resolve_sparse_variant,
    resolve_kv_offload_effective_state,
    resolve_effective_kv_mem_offload_size,
    resolve_effective_speculative_details,
    prepare_params_for_startup_status,
    lmcache_auto_floor_disables_all_backends,
    _is_deepseek_v4_flash_params,
    _is_qwen35_397b_w8a8_mtp_ascend910c_single_node_8,
    _inject_env_echo,
    _need_triton_patch,
)
from features.kv_offload.memcache import (
    build_memcache_hybrid_fragment,
)
from utils.vllm_helpers import (
    build_modelslim_quarot_patch_preamble,
    build_triton_patch_preamble,
)
from utils.env_utils import (
    OFFLOAD_MIN_GB,
    get_local_ip,
    get_lmcache_env,
    get_master_ip,
    validate_ip,
)
from utils.device_utils import resolve_card_token
from utils.model_utils import (
    ModelIdentifier,
    INDEXCACHE_ARCHS,
    feature_allowed,
    is_deepseek_v4_flash_rtx_pro_5000,
)

logger = logging.getLogger(__name__)

# install.py 现在只服务于明确登记的少数场景。这里保留两个边界：
# 1) Ascend LMCache：仍然跟随 offload 的有效状态，失败时还要回写 kv_offload=false；
# 2) RTX PRO 5000 + DeepSeek-V4-Flash：属于运行时依赖补齐，和 spec/sparse/offload
#    等高级特性开关解耦，只要模型与芯片命中就应在 engine 启动前先执行。
# 旧的通用 patch/features 入口已经删除，后续新增场景也必须在这里显式登记。
_LMCACHE_ASCEND_PACKAGE_CONFIG = '{"packages": ["lmcache-ascend:v0.4.5"]}'
# Pro5000 场景只有 packages 固定；engine.name/version 必须从本次启动上下文动态生成，
# 避免 launcher 升级后仍向 install.py 传递过期的 vLLM 版本。
_DEEPSEEK_V4_FLASH_PRO5000_PACKAGES = [
    "deepgemm:nv_dev_a6b593d",
    "flashinfer:v0.6.12",
]
# 上层的 ENGINE_VERSION 可能带 v 前缀、镜像后缀或卡型后缀，例如
# "v0.23.0" / "0.23.1-rtxpro5000"。install.py 只需要规范的 vX.Y.Z。
_ENGINE_VERSION_FOR_INSTALL_RE = re.compile(r"v?(\d+)\.(\d+)(?:\.(\d+))?", re.IGNORECASE)


def _shell_escape_single_quote(value: str) -> str:
    """对字符串中的单引号进行 shell 安全转义。"""
    return value.replace("'", "'\"'\"'")


def _inject_legacy_distributed_aliases(merged: dict, launch_args: LaunchArgs) -> None:
    """Preserve legacy distributed top-level fields across launcher and worker hops."""
    topology_csv = (
        getattr(launch_args, "node_ips", "")
        or getattr(launch_args, "nodes", "")
        or merged.get("node_ips", "")
        or merged.get("nodes", "")
    )
    if topology_csv:
        merged["node_ips"] = topology_csv
        merged["nodes"] = topology_csv

    master_ip = (
        getattr(launch_args, "master_ip", "")
        or merged.get("master_ip", "")
        or get_master_ip()
        or merged.get("head_node_addr", "")
    )
    if master_ip:
        if merged.get("distributed") and (
            not merged.get("head_node_addr") or merged.get("head_node_addr") == "127.0.0.1"
        ):
            merged["head_node_addr"] = master_ip
        merged["master_ip"] = master_ip
        merged["ray_head_ip"] = (
            getattr(launch_args, "ray_head_ip", "")
            or merged.get("ray_head_ip", "")
            or master_ip
        )
        if merged.get("engine") == "mindie":
            merged["mindie_master_addr"] = merged.get("mindie_master_addr") or master_ip


def _resolve_engine_service_host() -> str:
    """Return the concrete Pod IP used by the engine API listener."""
    for env_key in ("POD_IP", "RANK_IP"):
        candidate = os.getenv(env_key, "").strip()
        if validate_ip(candidate) and candidate != "0.0.0.0":
            return candidate

    local_ip = get_local_ip()
    if validate_ip(local_ip) and local_ip != "0.0.0.0":
        return local_ip

    logger.warning(
        "Unable to resolve Pod IP for engine listener; falling back to 127.0.0.1"
    )
    return "127.0.0.1"


@dataclass(frozen=True)
class LauncherPlan:
    """launcher 生成的最终计划。

    Attributes:
        command:       完整的 bash 启动脚本内容（含 shebang + set -euo pipefail），
                       将被写入 /shared-volume/start_command.sh 供 engine 容器执行。
        merged_params: 多层合并后的完整参数字典，便于日志审计和调试。
        hardware_env:  硬件探测结果（device/count/details），便于下游判断。
    """

    command: str
    merged_params: dict
    hardware_env: dict


def _prepare_merged_params(launch_args: LaunchArgs, port_plan: PortPlan, hardware: dict) -> dict:
    """配置合并、分布式参数注入与 host/port 分配，返回可直接传入 adapter 的 merged 字典。"""
    known_args = launch_args.to_namespace()
    merged = load_and_merge_configs(hardware_env=hardware, known_args=known_args)
    merged["model_name"] = launch_args.model_name
    merged["model_path"] = launch_args.model_path
    is_distributed = getattr(launch_args, "distributed", False)
    node_rank = getattr(launch_args, "node_rank", 0)
    merged["distributed"] = is_distributed
    merged["nnodes"] = getattr(launch_args, "nnodes", 1)
    merged["node_rank"] = node_rank
    merged["head_node_addr"] = getattr(launch_args, "head_node_addr", "127.0.0.1")
    # load_and_merge_configs() may auto-select a safer backend for specific
    # model/hardware combinations (for example Ascend DeepSeek uses
    # dp_deployment instead of Ray). Do not overwrite that decision with the
    # LaunchArgs parser default of "ray" unless the merge layer left it unset.
    merged.setdefault(
        "distributed_executor_backend",
        getattr(launch_args, "distributed_executor_backend", "ray"),
    )
    _inject_legacy_distributed_aliases(merged, launch_args)
    engine_cfg = dict(merged.get("engine_config", {}))
    # rank0 或单机场景需要显式注入 host/port，让 backend engine 真正提供服务。
    if not is_distributed or node_rank == 0:
        engine_host = _resolve_engine_service_host()
        merged["host"] = engine_host
        merged["port"] = port_plan.backend_port
        engine_cfg["host"] = engine_host
        engine_cfg["port"] = port_plan.backend_port
        if merged.get("engine") == "mindie":
            engine_cfg["ipAddress"] = engine_host
        logger.info(
            "Engine listener host resolved: host=%s port=%s "
            "(distributed=%s node_rank=%s POD_IP=%s RANK_IP=%s)",
            engine_host,
            port_plan.backend_port,
            is_distributed,
            node_rank,
            os.getenv("POD_IP", ""),
            os.getenv("RANK_IP", ""),
        )
    else:
        # 非 0 号节点一般只承担计算，不直接对外提供 engine 监听地址。
        merged.pop("host", None)
        merged.pop("port", None)
        engine_cfg.pop("host", None)
        engine_cfg.pop("port", None)
        engine_cfg.pop("ipAddress", None)
    merged["engine_config"] = engine_cfg
    return merged


def _is_lmcache_offload_allowed(engine: str, merged: dict | None) -> bool:
    """返回当前模型是否通过 offload 白名单门控。"""
    if not merged:
        return True
    smart_feats = merged.get("_smart_feats")
    if smart_feats is not None:
        return "offload" in smart_feats
    return feature_allowed(
        engine,
        merged.get("model_name"),
        merged.get("model_path"),
        merged.get("_smart_card_token") or resolve_card_token(),
        "offload",
    )


def _should_install_deepseek_v4_flash_ascend_lmcache(
    engine: str,
    merged: dict | None,
) -> bool:
    """仅在 Ascend DeepSeek-V4-Flash 真正启用 LMCache/offload 时安装 LMCache 包。"""
    if engine != "vllm_ascend":
        return False
    if not merged or not _is_deepseek_v4_flash_params(merged):
        return False
    if not _is_kv_offload_requested(merged):
        return False
    if not _is_lmcache_offload_allowed(engine, merged):
        logger.info(
            "[SmartFeature] offload suppressed by whitelist; "
            "skipping DeepSeek-V4-Flash Ascend LMCache package install."
        )
        return False
    if lmcache_auto_floor_disables_all_backends(merged):
        logger.info(
            "[KVCache Offload] auto memory offload capacity below floor and no "
            "disk/QAT/cold-start backend is active; skipping DeepSeek-V4-Flash "
            "Ascend LMCache package install."
        )
        return False
    return True


def _should_install_deepseek_v4_flash_pro5000_packages(
    engine: str,
    merged: dict | None,
) -> bool:
    """RTX PRO 5000 补丁只看模型/芯片/引擎身份，不读取高级特性开关。"""
    return is_deepseek_v4_flash_rtx_pro_5000(merged, engine)


def _resolve_engine_version_for_install(merged: dict | None = None) -> str:
    """解析传给 install.py 的 vLLM 版本。

    版本来源优先级：
    - merged["engine_version"]：预留给未来显式透传；
    - merged["engine"]["version"]：兼容上层若把 engine 当结构体透传的形态；
    - ENGINE_VERSION：当前标准化校验与 K8s 环境实际使用的入口。

    找不到版本时返回空字符串，由调用方跳过安装，避免退回硬编码版本。
    """
    candidates = []
    if isinstance(merged, dict):
        engine_config = merged.get("engine")
        candidates.extend(
            [
                merged.get("engine_version"),
                engine_config.get("version") if isinstance(engine_config, dict) else None,
            ]
        )
    candidates.append(os.getenv("ENGINE_VERSION", ""))

    for raw in candidates:
        text = str(raw or "").strip()
        if not text:
            continue
        match = _ENGINE_VERSION_FOR_INSTALL_RE.search(text)
        if not match:
            continue
        major, minor, patch = match.groups()
        return f"v{int(major)}.{int(minor)}.{int(patch or 0)}"
    return ""


def _build_deepseek_v4_flash_pro5000_package_config(
    engine: str,
    merged: dict | None = None,
) -> str:
    """构造 Pro5000 install.py 的 JSON payload。

    packages 固定来自适配需求；engine.name/version 必须跟随本次启动参数。
    这里用 json.dumps 生成 JSON，避免手写字符串在版本联动后变成隐性拼接错误。
    """
    version = _resolve_engine_version_for_install(merged)
    if not version:
        return ""
    return json.dumps(
        {
            "engine": {
                "name": engine,
                "version": version,
            },
            "packages": _DEEPSEEK_V4_FLASH_PRO5000_PACKAGES,
        }
    )


def _render_deepseek_v4_flash_pro5000_package_install_snippet(package_config: str) -> str:
    """渲染 Pro5000 运行时依赖安装片段。

    片段仍固定在 WINGS_ACCEL_DIR 下执行，和 Ascend LMCache 片段保持同一安装位置。
    set +e 保证补丁安装失败不会阻断主服务启动；这类依赖只影响加速能力。
    """
    accel_dir = settings.WINGS_ACCEL_DIR.rstrip("/")
    return (
        "# --- wings-accel: install DeepSeek-V4-Flash RTX PRO 5000 packages (fault-tolerant) ---\n"
        f"if [ -f \"{accel_dir}/install.py\" ]; then\n"
        "    echo '[wings-accel] Installing DeepSeek-V4-Flash RTX PRO 5000 packages...'\n"
        "    set +e\n"
        f"    (cd \"{accel_dir}\" && python3 install.py --config "
        f"'{_shell_escape_single_quote(package_config)}')\n"
        "    PRO5000_RC=$?\n"
        "    set -e\n"
        "    if [ $PRO5000_RC -ne 0 ]; then\n"
        '        echo "[wings-accel] WARNING: DeepSeek-V4-Flash RTX PRO 5000 package install failed'
        ' (exit=$PRO5000_RC), skipping. Service will continue without RTX PRO 5000 package install."\n'
        "    else\n"
        "        echo '[wings-accel] DeepSeek-V4-Flash RTX PRO 5000 packages installed successfully.'\n"
        "    fi\n"
        "else\n"
        f"    echo '[wings-accel] WARNING: {accel_dir}/install.py not found, "
        "skipping DeepSeek-V4-Flash RTX PRO 5000 package install.'\n"
        "fi\n"
    )


def _build_deepseek_v4_flash_pro5000_package_install_snippet(
    engine: str,
    merged: dict | None = None,
) -> str:
    """按场景构建 Pro5000 安装片段。

    先判断模型/芯片身份，再判断上层是否给出可解析的 ENGINE_VERSION。
    这样既满足“命中场景就先安装”，又避免在版本未知时向 install.py 传过期版本。
    """
    if not _should_install_deepseek_v4_flash_pro5000_packages(engine, merged):
        return ""
    package_config = _build_deepseek_v4_flash_pro5000_package_config(engine, merged)
    if not package_config:
        logger.warning(
            "ENGINE_VERSION is missing or unrecognized; skipping DeepSeek-V4-Flash "
            "RTX PRO 5000 package install."
        )
        return ""
    return _render_deepseek_v4_flash_pro5000_package_install_snippet(package_config)


def _render_deepseek_v4_flash_ascend_lmcache_install_snippet() -> str:
    """渲染 Ascend LMCache 包安装片段。

    该片段和 Pro5000 片段不同：它属于 offload 能力的一部分，失败后需要同步把
    advanced_features.json 中的 kv_offload 标记为 false，避免页面继续展示已启用。
    """
    accel_dir = settings.WINGS_ACCEL_DIR.rstrip("/")
    update_json = _shell_update_feature_json("kv_offload", False)
    return (
        "# --- wings-accel: install DeepSeek-V4-Flash Ascend LMCache package (fault-tolerant) ---\n"
        f"if [ -f \"{accel_dir}/install.py\" ]; then\n"
        "    echo '[wings-accel] Installing DeepSeek-V4-Flash Ascend LMCache package...'\n"
        "    set +e\n"
        f"    (cd \"{accel_dir}\" && python install.py --config "
        f"'{_LMCACHE_ASCEND_PACKAGE_CONFIG}')\n"
        "    LMCACHE_RC=$?\n"
        "    set -e\n"
        "    if [ $LMCACHE_RC -ne 0 ]; then\n"
        '        echo "[wings-accel] WARNING: DeepSeek-V4-Flash Ascend LMCache package install failed'
        ' (exit=$LMCACHE_RC), skipping. Service will continue without LMCache package install."\n'
        + update_json
        + "    else\n"
        "        echo '[wings-accel] DeepSeek-V4-Flash Ascend LMCache package installed successfully.'\n"
        "    fi\n"
        "else\n"
        f"    echo '[wings-accel] WARNING: {accel_dir}/install.py not found, "
        "skipping DeepSeek-V4-Flash Ascend LMCache package install.'\n"
        "fi\n"
    )


def _build_deepseek_v4_flash_ascend_lmcache_install_snippet(
    engine: str,
    merged: dict | None = None,
) -> str:
    if not _should_install_deepseek_v4_flash_ascend_lmcache(engine, merged):
        return ""
    return _render_deepseek_v4_flash_ascend_lmcache_install_snippet()


def _build_accel_preamble(engine: str, merged: dict) -> str:
    """生成所有受支持的 install.py 前置片段。

    顺序有业务含义：
    - Pro5000 依赖补丁是基础运行时补齐，只要命中场景就先执行；
    - Ascend LMCache 安装仍跟随 offload 的有效启用状态。

    两者都位于 engine 启动脚本之前，保持补丁位置固定。
    """
    if not settings.ENABLE_ACCEL:
        logger.debug(
            "Accel disabled: skipping DeepSeek-V4-Flash package installs"
        )
        return ""

    install_pro5000 = _should_install_deepseek_v4_flash_pro5000_packages(engine, merged)
    install_ascend_lmcache = _should_install_deepseek_v4_flash_ascend_lmcache(engine, merged)
    snippets = []
    if install_pro5000:
        pro5000_snippet = _build_deepseek_v4_flash_pro5000_package_install_snippet(
            engine, merged
        )
        if pro5000_snippet:
            snippets.append(pro5000_snippet)
            logger.info("Accel: injecting DeepSeek-V4-Flash RTX PRO 5000 package install")
    if install_ascend_lmcache:
        snippets.append(_render_deepseek_v4_flash_ascend_lmcache_install_snippet())
        logger.info("Accel: injecting DeepSeek-V4-Flash Ascend LMCache package install")
    return "".join(snippets)

def _is_env_override_file(path: Path) -> bool:
    """Return True if *path* is a valid env-override file (not hidden, not README)."""
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.name.upper() != "README.MD"
    )


_ENV_KEY_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _parse_env_file(fpath: Path) -> list[str]:
    """Parse a KEY=VALUE .env file and return a list of 'export KEY=VALUE' shell lines."""
    export_lines: list[str] = []
    try:
        content = fpath.read_text(encoding="utf-8")
        for lineno, raw_line in enumerate(content.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                logger.warning(
                    "Skipping invalid line %d in %s: no '=' found",
                    lineno, fpath.name,
                )
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not _ENV_KEY_RE.match(key):
                logger.warning(
                    "Skipping line %d in %s: invalid variable name %r",
                    lineno, fpath.name, key,
                )
                continue
            # 去掉可选的引号包裹
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            export_lines.append(f"export {key}={shlex.quote(value)}")
            logger.debug("  env: %s=%s", key, value)
    except Exception as e:
        logger.error("Failed to parse env file %s: %s", fpath, e)
    return export_lines


def _build_env_overrides_preamble() -> str:
    """读取 env_overrides 目录下的 .env/.sh 文件，生成注入到 start_command.sh 的环境变量前置片段。

    - .env 文件: 逐行解析 KEY=VALUE，生成 export 语句
    - .sh 文件: 通过 source 命令执行

    文件按名称字母序排列，隐藏文件（.开头）和 README 被忽略。
    """
    env_dir = Path(settings.ENV_OVERRIDES_DIR)
    if not env_dir.is_absolute():
        # 相对路径基于工作目录（通常是 /opt/wings-control/wings_control）
        env_dir = Path(os.getcwd()) / env_dir

    if not env_dir.is_dir():
        logger.debug("env_overrides directory not found: %s, skipping", env_dir)
        return ""

    lines: list[str] = []
    files = sorted(f for f in env_dir.iterdir() if _is_env_override_file(f))

    if not files:
        logger.debug("No env override files in %s", env_dir)
        return ""

    lines.append("# --- wings: user env overrides ---")
    for fpath in files:
        suffix = fpath.suffix.lower()
        logger.info("Loading env override: %s", fpath.name)

        if suffix == ".env":
            export_lines = _parse_env_file(fpath)
            lines.extend(export_lines)

        elif suffix == ".sh":
            # shell 脚本在全局 set -u 下执行时，允许引用暂未定义的变量。
            quoted_path = shlex.quote(str(fpath))
            quoted_label = shlex.quote(fpath.name)
            lines.append("set +u")
            lines.append(
                "if command -v wings_source_env_with_diff >/dev/null 2>&1; then "
                f"wings_source_env_with_diff {quoted_path} {quoted_label}; "
                f"else source {quoted_path}; fi"
            )
            lines.append("set -u")

        else:
            logger.debug("Ignoring unsupported file type: %s", fpath.name)

    if len(lines) <= 1:  # 只有注释头
        return ""

    lines.append("# --- end env overrides ---\n")
    preamble = "\n".join(lines) + "\n"
    logger.info("Injecting %d env override entries into start_command.sh", len(lines) - 2)
    return preamble


def _build_faulthandler_patch_preamble(engine: str) -> str:
    """为 SGLang 引擎注入 faulthandler.enable() 安全补丁。

    SGLang ≤ 0.5.10 的 scheduler.py 无保护地调用 ``faulthandler.enable()``，
    在 K8s 容器中因 /dev/shm (tmpfs) 计入 cgroup 内存限制，可能触发
    ``OSError: [Errno 12] Cannot allocate memory``。

    本函数通过 ``sitecustomize.py`` 注入 monkey-patch，用 try/except
    包裹原始 ``faulthandler.enable``，使其在 OOM 时静默降级而非崩溃。
    仅对 SGLang 引擎生效。

    Args:
        engine: 引擎类型

    Returns:
        str: shell 脚本片段；非 sglang 引擎返回空字符串。
    """
    if engine != "sglang":
        return ""

    patch_dir = "/tmp/wings_sitecustomize"
    # sitecustomize.py 在 Python 解释器启动时自动加载（早于任何用户代码），
    # 因此能在 SGLang import 链之前完成 monkey-patch。
    return (
        f"# --- wings: SGLang faulthandler.enable() OOM workaround ---\n"
        f"mkdir -p {patch_dir}\n"
        f"cat > {patch_dir}/sitecustomize.py << 'WINGS_FAULTHANDLER_PATCH'\n"
        f"import faulthandler as _fh\n"
        f"_original_enable = _fh.enable\n"
        f"def _safe_enable(*args, **kwargs):\n"
        f"    try:\n"
        f"        return _original_enable(*args, **kwargs)\n"
        f"    except OSError:\n"
        f"        pass  # /dev/shm tmpfs counted against cgroup memory limit\n"
        f"_fh.enable = _safe_enable\n"
        f"WINGS_FAULTHANDLER_PATCH\n"
        f'export PYTHONPATH="{patch_dir}:${{PYTHONPATH:-}}"\n'
        f"echo \"[wings] Injected faulthandler.enable() OOM patch for SGLang\"\n"
        f"# --- end faulthandler patch ---\n"
    )


def _build_analyzer_preamble(engine: str, merged: dict, hardware: dict) -> str:
    """生成 log_analyzer 进度监控的 shell 片段（仅 master 节点）。
    
    Args:
        engine: 引擎类型（vllm/vllm_ascend/sglang/mindie 等）
        merged: 合并后的参数字典
        hardware: 硬件探测结果
        
    Returns:
        str: log_analyzer 启动脚本片段。以下情况返回空字符串：
             - worker 节点 (node_rank > 0)
    """

    
    is_distributed = merged.get("distributed", False)
    node_rank = merged.get("node_rank", 0)
    
    if not is_distributed or node_rank == 0:
        # 构建分析器配置
        analyzer_config = {
            "engine": engine,
            "deployment_mode": "distributed" if is_distributed else "single",
            "hardware": hardware.get("device", "nvidia"),
            "nnodes": merged.get("nnodes", 1),
            "node_rank": node_rank,
            "distributed_backend": merged.get("distributed_executor_backend", "ray"),
            "tensor_parallel_size": merged.get("device_count", 1),
            "model_name": merged.get("model_name", ""),
            "model_path": merged.get("model_path", ""),
            "backend_port": merged.get("port", 17000)
        }

        analyzer_preamble = f"""
# --- log_analyzer: 启动部署进度监控（仅master节点） ---
# 清空旧的日志文件，确保 log_analyzer 只分析新的日志（避免残留内容触发误判）
rm -f /var/log/wings/engine.log
rm -f /var/log/wings/engine-full.log
rm -f {settings.PROGRESS_FILE}

# 记录脚本开始时间（用于计算耗时）
SCRIPT_START_EPOCH=$(date +%s)

ANALYZER_CONFIG='{_shell_escape_single_quote(json.dumps(analyzer_config))}'
echo "[log_analyzer] 配置信息: $ANALYZER_CONFIG"

# 启动日志分析器（后台）
# 清除旧 __pycache__，防止跨 Python 版本的 pyc magic number 不匹配
find {settings.SHARED_VOLUME_PATH}/log_analyzer -name '__pycache__' -type d -exec rm -rf {{}} + 2>/dev/null || true
cd {settings.SHARED_VOLUME_PATH} && python3 -B -m log_analyzer.log_analyzer \\
    --config "$ANALYZER_CONFIG" \\
    --log-file /var/log/wings/engine.log \\
    --progress-file {settings.PROGRESS_FILE} &
LOG_ANALYZER_PID=$!
echo "[log_analyzer] 分析器PID: $LOG_ANALYZER_PID"

# 注册清理函数（等待分析器完全退出）
cleanup_analyzer() {{
    local exit_code=$?
    echo "[log_analyzer] 停止分析器..."
    if [ -n "$LOG_ANALYZER_PID" ]; then
        kill $LOG_ANALYZER_PID 2>/dev/null || true
        # 等待分析器进程完全退出，确保完成收尾工作
        wait $LOG_ANALYZER_PID 2>/dev/null || true
    fi

    if [ -n "${{ENGINE_PID:-}}" ]; then
        echo "[cleanup] 发送 SIGTERM 给引擎进程..."
        kill -TERM "$ENGINE_PID" 2>/dev/null || true
    else
        # ENGINE_PID 未设置说明引擎启动前脚本就失败了（如 ray: command not found）
        # 写入失败进度，让上层感知到部署失败
        if [ "$exit_code" -ne 0 ]; then
            echo "[cleanup] 引擎启动前脚本异常退出，退出码: $exit_code"
            local curr_time
            curr_time=$(date -Iseconds)
            local start_time
            start_time=$(date -Iseconds -d "@${{SCRIPT_START_EPOCH}}")
            local elapsed
            elapsed=$(( $(date +%s) - SCRIPT_START_EPOCH ))
            cat >> "{settings.PROGRESS_FILE}" <<EARLY_FAIL_EOF
{{"progress": 0, "phase_code": "script_error", "phase_name": "启动脚本执行失败", "status": "failed", "key_log": "引擎启动前脚本异常退出，退出码: $exit_code", "curr_time": "$curr_time", "start_time": "$start_time", "elapsed_time_s": $elapsed}}
EARLY_FAIL_EOF
        fi
    fi
}}
trap cleanup_analyzer EXIT  SIGTERM SIGINT

"""
        logger.info("Injected log_analyzer for master node (node_rank=%d)", node_rank)
        return analyzer_preamble
    else:
        # Worker节点：不运行分析器，但确保共享卷目录存在
        analyzer_preamble = """
# --- log_analyzer: Worker节点不运行分析器 ---
echo "[log_analyzer] Worker节点(node_rank > 0)，跳过分析器启动"
# 确保共享卷目录存在
mkdir -p /shared-volume

"""
        logger.info("Skipped log_analyzer for worker node (node_rank=%d)", node_rank)
        return analyzer_preamble


def _build_env_echo_helpers_preamble() -> str:
    """生成环境变量打印辅助函数，供 engine 端 source 外部脚本时使用。

    wings_source_env_with_diff 会在 source 前后采集 env 快照，并打印新增或变化
    的 KEY=VALUE 行，用于覆盖 Ascend/MindIE 官方 set_env.sh 内部设置的变量。

    Returns:
        str: 可注入 start_command.sh 前置区域的 shell 函数定义

    注意:
        - 只打印 source 后仍然存在且新增/变化的变量
        - 不做脱敏，变量值会按实际内容输出到 engine 日志
    """
    return r'''# --- wings: env echo helpers ---
wings_source_env_with_diff() {
    local script_path="$1"
    local label="${2:-$1}"
    if [ "$#" -ge 2 ]; then
        shift 2
    else
        shift 1
    fi
    if [ ! -f "$script_path" ]; then
        echo "[wings-env-source] WARN: $label not found: $script_path"
        return 0
    fi

    local before_file after_file
    before_file="$(mktemp)"
    after_file="$(mktemp)"
    env | sort > "$before_file" || true

    set +u
    # shellcheck disable=SC1090
    source "$script_path" "$@"
    local source_rc=$?
    set -u

    env | sort > "$after_file" || true
    comm -13 "$before_file" "$after_file" | sed "s|^|[wings-env-source] $label |" || true
    rm -f "$before_file" "$after_file"
    return "$source_rc"
}
# --- end wings env echo helpers ---
'''


# ── 高级特性状态 JSON ──
# 在 advanced_features.json 中记录 4 个高级特性的使能状态 + 变体，
# 供 health 接口（/v1/startup/accel）和外部监控查询。初始值在 Python 层写入
# （build_launcher_plan），补丁安装失败时在 shell 层通过 python3 -c 单行脚本更新对应字段为 false。
# 它是加速特性使能状态的单一真相源（/v1/startup/accel 仅读此文件）。

_ADVANCED_FEATURES_FILE = settings.ADVANCED_FEATURES_FILE


def _is_kv_offload_requested(merged: dict | None = None) -> bool:
    smart_feats = merged.get("_smart_feats") if isinstance(merged, dict) else None
    if smart_feats is not None:
        return "offload" in smart_feats
    return get_lmcache_env()


def _resolve_kv_offload_state(engine: str, merged: dict) -> tuple[bool, str | None]:
    """Resolve effective KV offload state and diagnostic variant."""
    return resolve_kv_offload_effective_state(merged, engine)


def _resolve_advanced_feature_status(engine: str, merged: dict) -> dict:
    """统一计算 advanced_features.json 与 SmartFeature 日志共用的最终状态。

    这里是“最终承载形式”的单一入口：features 只回答 true/false，
    variants/others 才承载具体策略和容量。日志块复用同一个结果，避免出现
    advanced_features.json 已经是 false、但日志还按旧分支展示 enabled 的漂移。
    """
    kv_offload_effective, kv_offload_variant = _resolve_kv_offload_state(engine, merged)
    features = {
        "speculative_decode": bool(merged.get("enable_speculative_decode")),
        "sparse_kv": bool(merged.get("enable_sparse")),
        "kv_offload": kv_offload_effective,
        "rag_acc": os.getenv("RAG_ACC_ENABLED", "").strip().lower() == "true",
    }
    variants = {
        "speculative_decode": (resolve_speculative_strategy(merged, engine) or "none")
        if features["speculative_decode"] else None,
        "sparse_kv": resolve_sparse_variant(merged, engine) if features["sparse_kv"] else None,
        "kv_offload": kv_offload_variant,
    }
    others = {
        "kv_mem_offload_size": resolve_effective_kv_mem_offload_size(
            merged,
            engine,
            kv_offload_variant,
        ),
        "speculative_decode": resolve_effective_speculative_details(merged, engine),
    }
    return {"engine": engine, "features": features, "variants": variants, "others": others}


def _write_advanced_features_json(engine: str, merged: dict) -> None:
    """写入高级特性初始状态 JSON 到共享卷。

    4 个 bool 字段（features，类型不变，旧消费者继续读）：
      - speculative_decode: 投机推理
      - sparse_kv: KV 稀疏
      - kv_offload: LMCache KV 卸载
      - rag_acc: RAG 加速

    variants 段（需求一 §4，纯新增）：bool 旁挂「具体走哪种变体」细粒度，仅
    features[x]=true 有意义，false 给 null。变体由产出口同源纯函数推导
    （resolve_speculative_strategy / resolve_sparse_variant / resolve_offload_variant）；
    因本写入早于产出口在脚本生成阶段运行，故独立按 merged/env 推导，不依赖产出口先跑。
    下游消费者＝health 接口 /v1/startup/accel（读 settings.ADVANCED_FEATURES_FILE 并透出
    features + variants 给页面，见 proxy/health_service.py）；改字段须同步该端点。
    """
    data = _resolve_advanced_feature_status(engine, merged)
    ok = safe_write_file(
        _ADVANCED_FEATURES_FILE, data, is_json=True,
        options=WriteOptions(is_json=True, atomic=True),
    )
    if ok:
        logger.info(
            "Wrote advanced_features.json: features=%s variants=%s others=%s",
            data["features"], data["variants"], data["others"],
        )
    else:
        logger.error("Failed to write advanced_features.json")


def _shell_update_feature_json(feature_key: str, value: bool = False) -> str:
    """生成 shell 单行脚本，用于在补丁安装失败时更新 advanced_features.json 中的指定字段。"""
    json_path = _ADVANCED_FEATURES_FILE
    val_str = "True" if value else "False"
    return (
        f"        python3 -c \""
        f"import json, os; "
        f"p='{json_path}'; "
        f"d=json.load(open(p)) if os.path.exists(p) else {{'engine':'','features':{{}}}}; "
        f"d.setdefault('features',{{}})['{feature_key}']={val_str}; "
        f"f=open(p+'.tmp','w'); json.dump(d,f,indent=4); f.close(); "
        f"os.replace(p+'.tmp',p)"
        f"\"\n"
    )


# ── 高级特性回退策略 ──
# 启用高级特性（投机解码/KV稀疏/KV卸载）时，若引擎崩溃则无条件禁用所有高级特性重试一次。
# 采用一刀切策略：不区分启动阶段或运行阶段，崩溃即回退。
# 后续打补丁机制会通过特性状态码实现更精细的回退控制。


def _has_advanced_features(merged: dict, engine: str | None = None) -> bool:
    """判断是否启用了任何高级特性（投机解码、KV 稀疏、KV 卸载）。"""
    if merged.get("enable_speculative_decode"):
        return True
    if merged.get("enable_sparse"):
        return True
    effective, _ = _resolve_kv_offload_state(engine or merged.get("engine", ""), merged)
    if effective:
        return True
    return False


def _collect_active_feature_names(merged: dict, engine: str | None = None) -> list[str]:
    """收集当前激活的高级特性名称列表（用于日志）。"""
    names: list[str] = []
    if merged.get("enable_speculative_decode"):
        names.append("speculative_decode")
    if merged.get("enable_sparse"):
        names.append("sparse_kv")
    effective, _ = _resolve_kv_offload_state(engine or merged.get("engine", ""), merged)
    if effective:
        names.append("lmcache_offload")
    return names


# SmartFeature 独立日志块只展示页面上层下发的主字段。
# 不展示 SD_ENABLE、SPARSE_ENABLE、SPECULATIVE_DECODE_MODEL_PATH 等派生/历史字段，
# 是为了让 [Input Env] 保持“原始请求口径”；这些派生字段可能已经被 gate helper
# 改写，放在同一组里反而会干扰判断。
_SMART_FEATURE_AUDIT_ENV_KEYS = (
    "ENABLE_SPECULATIVE_DECODE",
    "ENABLE_SPARSE",
    "ENABLE_KV_OFFLOAD",
    "LMCACHE_OFFLOAD",
    "ENABLE_KV_MEM_OFFLOAD",
    "KV_MEM_OFFLOAD_SIZE",
    "AVAILABLE_POD_MEM_SIZE",
    "ENABLE_KV_DISK_OFFLOAD",
    "KV_DISK_OFFLOAD_SIZE",
)


def _format_bool(value) -> str:
    return "true" if bool(value) else "false"


def _format_list(values) -> str:
    if not values:
        return "[]"
    return "[" + ",".join(str(v) for v in values) + "]"


def _format_raw_env_value(value) -> str:
    if value is None:
        return "<unset>"
    text = str(value).strip()
    return text if text else "<empty>"


def _format_size_env_gb(value) -> str:
    """格式化卸载容量环境变量。

    KV_MEM_OFFLOAD_SIZE / KV_DISK_OFFLOAD_SIZE 的数值语义本来就是 GB，
    因此只补 GB 后缀；auto、空值和非法输入原样保留，避免日志误报。
    """
    text = _format_raw_env_value(value)
    if text in {"<unset>", "<empty>"} or text.lower() == "auto":
        return text
    try:
        return f"{int(text)}GB"
    except (TypeError, ValueError):
        return text


def _format_available_pod_mem_gb(value, *, with_unit: bool) -> str:
    """把 AVAILABLE_POD_MEM_SIZE 从 MiB 口径转成 GB 展示。

    上层传入的 AVAILABLE_POD_MEM_SIZE 是 MiB 数字，例如 81920 表示 80GB。
    日志统一转成 GB，是为了和 OFFLOAD_MIN_GB、resolved_node_size_gb 放在同一单位下比较。
    """
    text = _format_raw_env_value(value)
    if text in {"<unset>", "<empty>"}:
        return text
    try:
        formatted = f"{float(text) / 1024.0:.2f}"
    except (TypeError, ValueError):
        return text
    return f"{formatted}GB" if with_unit else formatted


def _format_smart_feature_env_value(name: str, value) -> str:
    if name == "AVAILABLE_POD_MEM_SIZE":
        return _format_available_pod_mem_gb(value, with_unit=True)
    if name in {"KV_MEM_OFFLOAD_SIZE", "KV_DISK_OFFLOAD_SIZE"}:
        return _format_size_env_gb(value)
    return _format_raw_env_value(value)


def _get_smart_feature_input_env(merged: dict) -> dict:
    """优先使用 config_loader 保存的收口前快照。

    正常链路中 ``_smart_feature_input_env`` 一定存在；fallback 只用于老路径、
    单测直接调用或异常情况下的降级日志，不能反向影响实际使能逻辑。
    """
    snapshot = merged.get("_smart_feature_input_env")
    if isinstance(snapshot, dict):
        return snapshot
    return {name: os.getenv(name) for name in _SMART_FEATURE_AUDIT_ENV_KEYS}


def _get_smart_feature_gate_trace(merged: dict) -> dict:
    trace = merged.get("_smart_feature_gate_trace")
    return trace if isinstance(trace, dict) else {}


def _get_engine_or_merged_value(merged: dict, key: str, default=None):
    value = merged.get(key)
    if value not in (None, ""):
        return value
    engine_config = merged.get("engine_config")
    if isinstance(engine_config, dict):
        value = engine_config.get(key)
        if value not in (None, ""):
            return value
    return default


def _fallback_gate_row(merged: dict, feature_key: str, smart_name: str) -> dict:
    """在缺少 gate trace 时按历史字段补一行可读的 gate 状态。

    这不是新的判定入口，只是日志降级策略。正常启动路径应消费
    ``config_loader.apply_effective_feature_enablement`` 写入的
    ``_smart_feature_gate_trace``，从而保留 requested/whitelist/gate/reason 的完整链路。
    """
    allowed = set(merged.get("_allowed_smart_feats") or [])
    forced = set(merged.get("_forced_smart_feats") or [])
    smart_feats = set(merged.get("_smart_feats") or [])
    if feature_key == "speculative_decode":
        requested = bool(merged.get("enable_speculative_decode"))
        gate = requested
    elif feature_key == "sparse_kv":
        requested = bool(merged.get("enable_sparse"))
        gate = requested
    else:
        requested = _is_kv_offload_requested(merged)
        gate = smart_name in smart_feats if merged.get("_smart_feats") is not None else requested
    whitelisted = smart_name in allowed or smart_name in forced
    if gate:
        reason = "enabled"
    elif not requested:
        reason = "request_off"
    elif not whitelisted:
        reason = "whitelist_miss"
    else:
        reason = "suppressed"
    return {
        "requested": requested,
        "whitelist": whitelisted,
        "gate": gate,
        "reason": reason,
    }


def _get_gate_row(merged: dict, feature_key: str, smart_name: str) -> dict:
    trace = _get_smart_feature_gate_trace(merged)
    features = trace.get("features") if isinstance(trace.get("features"), dict) else {}
    row = features.get(feature_key)
    return row if isinstance(row, dict) else _fallback_gate_row(merged, feature_key, smart_name)


def _detect_speculative_command_emitted(command: str) -> bool:
    """基于最终 start_command.sh 判断投机字段是否真的注入。"""
    return "--speculative-config" in command or "speculative_config" in command


def _detect_sparse_command_emitted(command: str, variant: str | None) -> bool:
    """基于最终 start_command.sh 判断稀疏相关字段是否真的注入。"""
    variant = variant or ""
    if variant == "noop":
        return False
    if "indexcache" in variant:
        return "index_topk_freq" in command or "use_index_cache" in command
    if variant == "fp8":
        return "--kv-cache-dtype fp8" in command or '"kv_cache_dtype": "fp8"' in command
    return any(marker in command for marker in ("index_topk_freq", "use_index_cache"))


def _detect_offload_command_emitted(command: str) -> bool:
    """基于最终 start_command.sh 判断卸载相关字段是否真的注入。

    这里故意不只看 gate/json 状态：KV offload 的 auto 场景可能先命中白名单，
    后续因可用内存低于下限被丢弃。只有检查真实命令，才能发现
    kv-transfer / LMCache / native offload 字段是否还被错误追加。
    """
    markers = (
        "--kv-offloading-backend",
        "LMCacheConnector",
        "LMCacheAscendConnector",
        "AscendStoreConnector",
        "WINGS_MEMCACHE_DRAM_GB",
        "LMCACHE_",
    )
    return any(marker in command for marker in markers)


def _resolve_smart_feature_final_reason(
    feature_key: str,
    *,
    final: bool,
    variant: str | None,
    gate_reason: str,
) -> str:
    """合并 gate reason 与 final resolve reason。

    gate reason 解释“页面请求经过白名单后是否允许”；final reason 解释
    “最终 JSON/命令为什么是这个状态”。当 offload auto 低于内存下限时，
    final reason 必须覆盖 gate 的 enabled，避免日志误导为已经启用。
    """
    variant = variant or ""
    if feature_key == "kv_offload":
        if "floor_disabled" in variant:
            return "auto_floor_disabled"
        if "auto+unavailable" in variant:
            return "auto_unavailable"
        if variant == "disabled":
            return "disabled"
    if final:
        if feature_key == "speculative_decode" and gate_reason == "suffix_fallback":
            return "suffix_fallback"
        if feature_key == "sparse_kv" and variant == "noop":
            return "noop"
        return "enabled"
    return gate_reason or "disabled"


def _build_smart_feature_final_rows(engine: str, merged: dict, command: str) -> dict:
    """把最终 JSON 状态、变体和真实命令注入状态合成三特性行。

    这一层是日志的关键对齐点：
    - final 来自 advanced_features.json 同源计算；
    - variant/size_gb 来自各特性最终解析；
    - emitted 来自已经拼好的 start_command.sh；
    - reason 优先说明 final 阶段的丢弃原因。
    """
    status = _resolve_advanced_feature_status(engine, merged)
    features = status["features"]
    variants = status["variants"]
    spec_gate = _get_gate_row(merged, "speculative_decode", "spec")
    sparse_gate = _get_gate_row(merged, "sparse_kv", "sparse")
    offload_gate = _get_gate_row(merged, "kv_offload", "offload")
    rows = {
        "speculative_decode": {
            "final": bool(features.get("speculative_decode")),
            "variant": variants.get("speculative_decode") or "none",
            "emitted": _detect_speculative_command_emitted(command),
            "gate_reason": str(spec_gate.get("reason") or ""),
        },
        "sparse_kv": {
            "final": bool(features.get("sparse_kv")),
            "variant": variants.get("sparse_kv") or "none",
            "emitted": _detect_sparse_command_emitted(command, variants.get("sparse_kv")),
            "gate_reason": str(sparse_gate.get("reason") or ""),
        },
        "kv_offload": {
            "final": bool(features.get("kv_offload")),
            "variant": variants.get("kv_offload") or "none",
            "emitted": _detect_offload_command_emitted(command),
            "gate_reason": str(offload_gate.get("reason") or ""),
            "size_gb": status["others"].get("kv_mem_offload_size"),
        },
    }
    for key, row in rows.items():
        row["reason"] = _resolve_smart_feature_final_reason(
            key,
            final=row["final"],
            variant=row.get("variant"),
            gate_reason=row.get("gate_reason", ""),
        )
    return {"status": status, "rows": rows}


def _format_feature_gate_line(feature_key: str, row: dict) -> str:
    return (
        f"feature={feature_key} "
        f"requested={_format_bool(row.get('requested'))} "
        f"whitelist={_format_bool(row.get('whitelist'))} "
        f"gate={_format_bool(row.get('gate'))} "
        f"reason={row.get('reason') or 'unknown'}"
    )


def _format_feature_final_line(feature_key: str, row: dict) -> str:
    pieces = [
        f"feature={feature_key}",
        f"final={_format_bool(row.get('final'))}",
        f"variant={row.get('variant') or 'none'}",
    ]
    if feature_key == "kv_offload":
        pieces.append(f"size_gb={row.get('size_gb')}")
    pieces.extend([
        f"emitted={_format_bool(row.get('emitted'))}",
        f"reason={row.get('reason') or 'unknown'}",
    ])
    return " ".join(pieces)


def _format_json_features(features: dict) -> str:
    return (
        "{"
        f"speculative_decode:{_format_bool(features.get('speculative_decode'))},"
        f"sparse_kv:{_format_bool(features.get('sparse_kv'))},"
        f"kv_offload:{_format_bool(features.get('kv_offload'))}"
        "}"
    )


def _format_command_emitted(rows: dict) -> str:
    return (
        "{"
        f"speculative_decode:{_format_bool(rows['speculative_decode'].get('emitted'))},"
        f"sparse_kv:{_format_bool(rows['sparse_kv'].get('emitted'))},"
        f"kv_offload:{_format_bool(rows['kv_offload'].get('emitted'))}"
        "}"
    )


def _build_smart_feature_enablement_log_block(
    engine: str,
    merged: dict,
    hardware: dict,
    command: str,
) -> str:
    """构造独立、可 grep 的 SmartFeature 使能日志块。

    日志块按照“上层输入 -> 白名单收口 -> 最终解析 -> 输出承载”排列。
    运行时仍走项目统一 LOG_FORMAT；调用方用一次 logger.info 输出整块文本，
    这样第一行带有组件/函数/行号，后续换行保留块状结构，kubectl 中更容易定位。
    """
    input_env = _get_smart_feature_input_env(merged)
    gate_trace = _get_smart_feature_gate_trace(merged)
    final_payload = _build_smart_feature_final_rows(engine, merged, command)
    status = final_payload["status"]
    rows = final_payload["rows"]
    gate_rows = {
        "speculative_decode": _get_gate_row(merged, "speculative_decode", "spec"),
        "sparse_kv": _get_gate_row(merged, "sparse_kv", "sparse"),
        "kv_offload": _get_gate_row(merged, "kv_offload", "offload"),
    }
    separator = "=" * 80
    available_pod_mem_gb = _format_available_pod_mem_gb(
        input_env.get("AVAILABLE_POD_MEM_SIZE"),
        with_unit=False,
    )
    kv_size = status["others"].get("kv_mem_offload_size")
    kv_reason = rows["kv_offload"].get("reason") or "unknown"
    lines = [
        "[SmartFeature]",
        separator,
        "=                         SMART FEATURE ENABLEMENT                              =",
        separator,
        "",
        "[Context]",
        f"engine={engine}",
        f"card={merged.get('_smart_card_token') or resolve_card_token(hardware) or '(empty)'}",
        f"model={merged.get('model_name') or merged.get('model_path') or '<unset>'}",
        f"model_type={merged.get('_resolved_model_type') or merged.get('model_type') or '<unset>'}",
        f"device_count={merged.get('device_count', '<unset>')}",
        f"tp={_get_engine_or_merged_value(merged, 'tensor_parallel_size', '<unset>')}",
        f"dp={_get_engine_or_merged_value(merged, 'data_parallel_size', '<unset>')}",
        "",
        "[Input Env]",
    ]
    lines.extend(
        f"{name}={_format_smart_feature_env_value(name, input_env.get(name))}"
        for name in _SMART_FEATURE_AUDIT_ENV_KEYS
    )
    lines.extend([
        "",
        "[Whitelist Gate]",
        f"allowed={_format_list(gate_trace.get('allowed') or merged.get('_allowed_smart_feats'))}",
        f"forced={_format_list(gate_trace.get('forced') or merged.get('_forced_smart_feats'))}",
        f"effective={_format_list(gate_trace.get('effective') or merged.get('_smart_feats'))}",
        "",
        _format_feature_gate_line("speculative_decode", gate_rows["speculative_decode"]),
        _format_feature_gate_line("sparse_kv", gate_rows["sparse_kv"]),
        _format_feature_gate_line("kv_offload", gate_rows["kv_offload"]),
        "",
        "[Final Resolve]",
        "offload_capacity "
        f"mode={_format_raw_env_value(input_env.get('KV_MEM_OFFLOAD_SIZE'))} "
        f"available_pod_mem_gb={available_pod_mem_gb} "
        f"floor_gb={OFFLOAD_MIN_GB} "
        f"resolved_node_size_gb={kv_size} "
        f"reason={kv_reason}",
        "",
        _format_feature_final_line("speculative_decode", rows["speculative_decode"]),
        _format_feature_final_line("sparse_kv", rows["sparse_kv"]),
        _format_feature_final_line("kv_offload", rows["kv_offload"]),
        "",
        "[Output]",
        f"json features={_format_json_features(status['features'])}",
        f"command emitted={_format_command_emitted(rows)}",
        "",
        separator,
        "=                       SMART FEATURE ENABLEMENT END                            =",
        separator,
    ])
    return "\n".join(lines)


def _log_smart_feature_enablement(
    engine: str,
    merged: dict,
    hardware: dict,
    command: str,
) -> None:
    """在启动命令生成后打印 SmartFeature 审计日志。

    必须放在 _assemble_startup_command 之后，因为 emitted 字段要读取真实命令。
    如果提前打印，只能预测是否注入，无法覆盖 auto floor disabled 这类 late discard。
    """
    logger.info(
        "\n%s",
        _build_smart_feature_enablement_log_block(engine, merged, hardware, command),
    )


def _build_monitor_script(
    fallback_cmd: str = "",
    retry_cmd: str = "",
    active_features: str = "",
    engine: str = "",
    rag_configured: bool = False,
) -> str:
    """生成引擎进程等待和异常处理的 shell 片段。

    Args:
        fallback_cmd:    当高级特性导致引擎快速失败时的回退启动命令（含 & 和 ENGINE_PID 赋值）。
                         如果为空字符串，则不生成高级特性回退逻辑。
        retry_cmd:       当默认模式引擎崩溃时的重试启动命令（与原始命令相同）。
                         如果为空字符串，则不生成重试逻辑。优先级低于 fallback_cmd。
        active_features: 当前激活的高级特性名称（逗号分隔），用于日志。
        engine:          引擎名称，用于 fallback 时写入 advanced_features.json。
        rag_configured:  RAG 加速是否已配置，fallback 时保留此值。

    Returns:
        str: 进程监控脚本片段
    """
    progress_file = settings.PROGRESS_FILE

    # ── 公共片段：清理 analyzer + 写失败进度 ──
    cleanup_analyzer = (
        '  echo "[引擎] 停止日志解析进程..."\n'
        '  [ -n "${LOG_ANALYZER_PID:-}" ] && kill "$LOG_ANALYZER_PID" 2>/dev/null || true\n'
        '  trap - EXIT'
    )
    write_progress = (
        '  CURR_TIME=$(date -Iseconds)\n'
        '  SCRIPT_START_EPOCH="${SCRIPT_START_EPOCH:-$(date +%s)}"\n'
        '  START_TIME=$(date -Iseconds -d "@${SCRIPT_START_EPOCH}")\n'
        '  ELAPSED_TIME=$(( $(date +%s) - SCRIPT_START_EPOCH ))\n'
        '\n'
        f'  cat >> "{progress_file}" <<EOF\n'
        '{"progress": 0, "phase_code": "engine_crash", "phase_name": "引擎进程异常退出", '
        '"status": "failed", "key_log": "引擎进程异常退出，退出码: $EXIT_CODE", '
        '"curr_time": "$CURR_TIME", "start_time": "$START_TIME", "elapsed_time_s": $ELAPSED_TIME}\n'
        'EOF'
    )

    if not fallback_cmd and not retry_cmd:
        # ── 基础版：无回退 / 无重试 ──
        return f"""
# --- 引擎进程等待和异常处理 ---
if wait "$ENGINE_PID"; then
  echo "[引擎] 引擎进程正常退出"
{cleanup_analyzer}
else
  EXIT_CODE=$?
  echo "[引擎] 引擎进程异常退出，退出码: $EXIT_CODE"

{write_progress}

{cleanup_analyzer}

  exit "$EXIT_CODE"
fi

"""

    if not fallback_cmd and retry_cmd:
        # ── 默认模式重试版：崩溃后用相同参数重试一次 ──
        # 不要为 retry_cmd 行加缩进：内嵌 heredoc 的关闭标记必须位于列 0，
        # 否则 'bash -n' 报 'here-document delimited by end-of-file'。
        indented_retry = retry_cmd.rstrip("\n")
        cleanup_4 = cleanup_analyzer.replace("  ", "      ")
        write_progress_4 = write_progress.replace("  ", "      ")

        return f"""
# --- Engine process wait and exception handling (with crash retry) ---
echo "[Engine] Engine process monitor started, PID=$ENGINE_PID"
if wait "$ENGINE_PID"; then
  echo "[Engine] Engine process exited normally"
{cleanup_analyzer}
else
  EXIT_CODE=$?
  ENGINE_DURATION=$(( $(date +%s) - ENGINE_START_EPOCH ))
  echo "[Engine] Engine process exited abnormally, exit_code=$EXIT_CODE, runtime=${{ENGINE_DURATION}}s"
  echo "[Engine] ┌── Engine Crash Retry ──"
  echo "[Engine] │ Reason: Engine crashed (exit_code=$EXIT_CODE, runtime=${{ENGINE_DURATION}}s)"
  echo "[Engine] │ Action: Retrying engine startup with same parameters (attempt 2/2)"
  echo "[Engine] └── Retry command about to execute..."
  # 清理上一次启动残留：ray head/worker 进程 + 端口占用
  if command -v ray >/dev/null 2>&1; then
    echo "[Engine] Stopping leftover Ray cluster before retry..."
    ray stop --force >/dev/null 2>&1 || true
  fi
  # 兜底：杀掉残留的 vLLM EngineCore / WorkerProc（父进程已死但子进程可能还在）
  pkill -9 -f 'vllm.*EngineCore' 2>/dev/null || true
  pkill -9 -f 'vllm.*WorkerProc' 2>/dev/null || true
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  echo "[Engine] Waiting 5s for port release before retry..."
  sleep 5
  ENGINE_START_EPOCH=$(date +%s)
{indented_retry}
  echo "[Engine] Retry engine started, waiting for process exit..."
  if wait "$ENGINE_PID"; then
    echo "[Engine] Engine process exited normally (retry mode)"
{cleanup_4}
  else
    EXIT_CODE=$?
    echo "[Engine] Retry also failed, exit_code=$EXIT_CODE — unrecoverable"

{write_progress_4}

{cleanup_4}

    exit "$EXIT_CODE"
  fi
fi

"""

    # ── 增强版：高级特性快速失败回退 ──
    # 不要为 fallback_cmd 行加缩进：内嵌 heredoc 的关闭标记必须位于列 0，
    # 否则 'bash -n' 报 'here-document delimited by end-of-file'。
    indented_fallback = fallback_cmd.rstrip("\n")
    # 缩进 cleanup 和 progress 到回退 if/else 内部
    cleanup_4 = cleanup_analyzer.replace("  ", "    ")  # 4-space indent (回退逻辑减少了一层if嵌套)
    write_progress_4 = write_progress.replace("  ", "    ")
    feat_label = active_features or "advanced_features"
    rag_val = "true" if rag_configured else "false"
    json_path = _ADVANCED_FEATURES_FILE

    return f"""
# --- Engine process wait and exception handling (with advanced feature fallback) ---
echo "[AdvFeature] Engine process monitor started, PID=$ENGINE_PID"
echo "[AdvFeature] Active advanced features: {feat_label}"
if wait "$ENGINE_PID"; then
  echo "[Engine] Engine process exited normally"
{cleanup_analyzer}
else
  EXIT_CODE=$?
  ENGINE_DURATION=$(( $(date +%s) - ENGINE_START_EPOCH ))
  echo "[Engine] Engine process exited abnormally, exit_code=$EXIT_CODE, runtime=${{ENGINE_DURATION}}s"

  # 一刀切策略：高级特性启用时崩溃 → 无条件禁用所有高级特性重试一次
  echo "[AdvFeature] ┌── Advanced Feature Fallback Triggered ──"
  echo "[AdvFeature] │ Reason: Engine crashed (exit_code=$EXIT_CODE, runtime=${{ENGINE_DURATION}}s)"
  echo "[AdvFeature] │ Features disabled: {feat_label}"
  echo "[AdvFeature] │ Action: Restarting engine without advanced features"
  echo "[AdvFeature] └── Fallback command about to execute..."
  echo "[Engine] Falling back to basic mode (disabled: {feat_label})..."
  # 更新 advanced_features.json：引擎级特性全部置 false，RAG 保持不变
  cat > "{json_path}" <<'FEATURES_EOF'
{{
    "engine": "{engine}",
    "features": {{
        "speculative_decode": false,
        "sparse_kv": false,
        "kv_offload": false,
        "rag_acc": {rag_val}
    }},
    "variants": {{
        "speculative_decode": null,
        "sparse_kv": null,
        "kv_offload": null
    }},
    "others": {{
        "kv_mem_offload_size": 0,
        "speculative_decode": null
    }}
}}
FEATURES_EOF
  echo "[AdvFeature] Updated advanced_features.json: all engine features disabled"
  # 清理上一次启动残留：ray head/worker 进程 + 端口占用
  # （fallback 会重新执行 ray start --head，若旧 head 仍在则会因端口冲突失败）
  if command -v ray >/dev/null 2>&1; then
    echo "[AdvFeature] Stopping leftover Ray cluster before fallback restart..."
    ray stop --force >/dev/null 2>&1 || true
  fi
  # 兜底：杀掉残留的 vLLM EngineCore / WorkerProc（父进程已死但子进程可能还在）
  pkill -9 -f 'vllm.*EngineCore' 2>/dev/null || true
  pkill -9 -f 'vllm.*WorkerProc' 2>/dev/null || true
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  echo "[Engine] Waiting 5s for port release before restart..."
  sleep 5
  ENGINE_START_EPOCH=$(date +%s)
{indented_fallback}
  echo "[AdvFeature] Fallback-mode engine started, waiting for process exit..."
  if wait "$ENGINE_PID"; then
    echo "[Engine] Engine process exited normally (fallback mode)"
    echo "[AdvFeature] Fallback-mode engine exited normally"
{cleanup_4}
  else
    EXIT_CODE=$?
    echo "[Engine] Fallback mode also exited abnormally, exit_code=$EXIT_CODE"
    echo "[AdvFeature] ✗ Fallback mode also failed, exit_code=$EXIT_CODE — unrecoverable"

{write_progress_4}

{cleanup_4}

    exit "$EXIT_CODE"
  fi
fi

"""


def _strip_exec_and_backgroundify(script_body: str) -> str:
    """将引擎脚本从 'exec cmd' 格式转换为 'cmd &' 后台运行格式。

    逐行从末尾扫描，找到最后一个非空行：
    - 若以 'exec ' 开头：剔除 exec 前缀并追加 ' &'
    - 否则：直接追加 ' &'
    """
    lines = script_body.rstrip("\n").split("\n")
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].lstrip()
        if stripped.startswith("exec "):
            indent = lines[i][: len(lines[i]) - len(stripped)]
            lines[i] = indent + stripped[5:] + " &"
            break
        if stripped:
            lines[i] = lines[i] + " &"
            break
    return "\n".join(lines) + "\n"


def _log_advanced_feature_config(
    engine: str, merged: dict, has_advanced_feature: bool,
) -> None:
    """记录所有高级特性（投机解码 / KV 稀疏 / KV 卸载）的配置日志。"""
    if not has_advanced_feature:
        logger.info("[AdvFeature] No advanced features enabled")
        return
    feature_names = _collect_active_feature_names(merged, engine)
    logger.info("[AdvFeature] ┌── Advanced Feature Config ──")
    logger.info("[AdvFeature] │ engine = %s", engine)
    logger.info("[AdvFeature] │ active features = %s", ", ".join(feature_names))
    # 投机解码
    if merged.get("enable_speculative_decode"):
        spec_model_path = merged.get("speculative_decode_model_path", "")
        logger.info("[AdvFeature] │ [speculative_decode]")
        logger.info("[AdvFeature] │   model_path = %s",
                    spec_model_path or "(none, using auto strategy)")
    # KV 稀疏（IndexCache / FP8 KV CACHE）
    if merged.get("enable_sparse"):
        model_info = ModelIdentifier(
            merged.get("model_name"), merged.get("model_path"), merged.get("model_type"),
        )
        arch = model_info.model_architecture
        _ic_archs = INDEXCACHE_ARCHS
        strategy = "IndexCache" if arch in _ic_archs else "FP8 KV CACHE"
        logger.info("[AdvFeature] │ [sparse_kv]")
        logger.info("[AdvFeature] │   strategy = %s (arch=%s)", strategy, arch)
    # KV 卸载
    kv_offload_effective, kv_offload_variant = _resolve_kv_offload_state(engine, merged)
    if kv_offload_effective:
        logger.info("[AdvFeature] │ [lmcache_offload]")
        logger.info("[AdvFeature] │   kv_transfer_config = %s",
                    merged.get("kv_transfer_config", "(not set)"))
    elif kv_offload_variant:
        logger.info(
            "[AdvFeature] │ [lmcache_offload] inactive (variant=%s)",
            kv_offload_variant,
        )
    logger.info("[AdvFeature] │ fallback_strategy = all-or-nothing (crash triggers full rollback)")
    logger.info("[AdvFeature] └── Advanced features injected into engine start command")


def _build_advanced_feature_fallback_cmd(merged: dict) -> str:
    """生成禁用所有高级特性的引擎回退启动命令。

    禁用的特性：
    - 投机解码（--speculative-config）
    - KV 稀疏（--sparse-config + sparse --kv-transfer-config）
    - KV 卸载（LMCache --kv-transfer-config）

    Returns:
        回退命令字符串（已后台化 + ENGINE_PID 记录）。
    """
    feature_names = _collect_active_feature_names(merged)
    feature_label = ", ".join(feature_names)
    merged_no_features = dict(merged)
    merged_no_features["enable_speculative_decode"] = False
    merged_no_features["enable_sparse"] = False
    kv_offload_requested = _is_kv_offload_requested(merged)
    original_ec = merged_no_features.get("engine_config", {})
    if isinstance(original_ec, dict):
        ec_copy = dict(original_ec)
        if "speculative_config" in ec_copy:
            ec_copy.pop("speculative_config", None)
            logger.info(
                "[AdvFeature] Removed speculative_config "
                "from engine_config for fallback (Speculative Decode was enabled)"
            )
        if (
            kv_offload_requested
            and "kv_transfer_config" in ec_copy
        ):
            ec_copy.pop("kv_transfer_config", None)
            logger.info(
                "[AdvFeature] Removed kv_transfer_config "
                "from engine_config for fallback (LMCache Offload was enabled)"
            )
        if ec_copy != original_ec:
            merged_no_features["engine_config"] = ec_copy
    # kv_transfer_config 由 config_loader._set_kv_cache_config() 注入到
    # engine_config 嵌套字典中，需要从正确的层级移除。
    # 使用浅拷贝 engine_config 避免污染原始 merged 数据。
    # （PD 分离的 kv_transfer_config 也会一并移除，但在崩溃回退场景下可接受）
    if kv_offload_requested:
        original_ec = merged_no_features.get("engine_config", {})
        if isinstance(original_ec, dict) and "kv_transfer_config" in original_ec:
            ec_copy = dict(original_ec)
            ec_copy.pop("kv_transfer_config", None)
            merged_no_features["engine_config"] = ec_copy
            logger.info(
                "[AdvFeature] Removed kv_transfer_config "
                "from engine_config for fallback (LMCache Offload was enabled)"
            )
        # [V4-Flash-NV-Day0] NV V4-Flash native 卸载是构建期 CLI flag（非 kv_transfer_config），
        # 需显式抑制，否则 fallback 重建命令仍会带上 --kv-offloading-backend。
        merged_no_features["_wings_fallback_no_kv_offload"] = True
    fallback_body = start_engine_service(merged_no_features)
    fallback_cmd = _strip_exec_and_backgroundify(fallback_body)
    memcache_fragment = build_memcache_hybrid_fragment(
        merged.get("engine", ""),
        merged,
    )
    if memcache_fragment["enabled"]:
        fallback_cmd = memcache_fragment["fallback_cleanup"] + fallback_cmd
    fallback_cmd += "ENGINE_PID=$!\n"
    fallback_cmd += (
        f'echo "[Engine] Engine PID: $ENGINE_PID '
        f'(advanced features disabled: {feature_label}, fallback mode)"\n'
    )
    logger.info(
        "[AdvFeature] Generated fallback command (disabled: %s) for fast-fail recovery",
        feature_label,
    )
    return fallback_cmd


def _build_pid_tracked_script(script_body: str, has_advanced_feature: bool) -> str:
    """将引擎启动脚本转换为后台模式并注入 ENGINE_PID / ENGINE_START_EPOCH 跟踪。

    始终注入 ENGINE_START_EPOCH，用于高级特性快速失败检测和默认模式崩溃重试。

    Args:
        script_body:          原始引擎启动脚本体
        has_advanced_feature: 是否启用了高级特性（决定日志标签）

    Returns:
        修改后的脚本体字符串
    """
    body = _strip_exec_and_backgroundify(script_body)
    body += "ENGINE_PID=$!\n"
    if has_advanced_feature:
        body += 'echo "[Engine] Engine PID: $ENGINE_PID (advanced features enabled)"\n'
    else:
        body += 'echo "[Engine] Engine PID: $ENGINE_PID"\n'
    # 始终注入 ENGINE_START_EPOCH，用于崩溃重试的运行时长统计
    body = "ENGINE_START_EPOCH=$(date +%s)\n" + body
    return body


def _build_engine_retry_cmd(merged: dict) -> str:
    """生成引擎重试启动命令（与原始命令相同，用于默认参数崩溃后的一次重试）。

    Returns:
        重试命令字符串（已后台化 + ENGINE_PID 记录）。
    """
    retry_body = start_engine_service(merged)
    retry_cmd = _strip_exec_and_backgroundify(retry_body)
    retry_cmd += "ENGINE_PID=$!\n"
    retry_cmd += 'echo "[Engine] Engine PID: $ENGINE_PID (retry mode)"\n'
    logger.info("[Engine] Generated retry command for default-mode crash recovery")
    return retry_cmd


def _resolve_engine_and_features(
    merged: dict, launch_args: LaunchArgs,
) -> tuple[str, bool, str]:
    """确定引擎类型、判断高级特性状态，并写入状态 JSON / 记录日志。

    Returns:
        (engine, has_advanced_feature, active_features_label)
    """
    # engine 已在 load_and_merge_configs 中经过 _auto_select_engine 的
    # 自动选择、校验和升级（如 vllm → vllm_ascend），不可用原始值覆盖。
    engine = merged.get("engine", launch_args.engine)
    has_advanced_feature = _has_advanced_features(merged, engine)
    active_feature_names = _collect_active_feature_names(merged, engine)
    active_features_label = ", ".join(active_feature_names)

    _write_advanced_features_json(engine, merged)
    _log_advanced_feature_config(engine, merged, has_advanced_feature)
    return engine, has_advanced_feature, active_features_label


def _build_engine_and_monitor_scripts(
    engine: str,
    merged: dict,
    has_advanced_feature: bool,
    active_features_label: str,
) -> tuple[str, str]:
    """生成引擎启动脚本体（含 PID 跟踪）和进程监控脚本。

    根据是否启用高级特性，生成不同的回退/重试策略：
    - 高级特性启用 → 崩溃时禁用全部高级特性回退
    - 默认模式 → 崩溃时用相同参数重试一次

    Returns:
        (script_body, monitor_script)
    """
    fallback_cmd = _build_advanced_feature_fallback_cmd(merged) if has_advanced_feature else ""
    retry_cmd = _build_engine_retry_cmd(merged) if not has_advanced_feature else ""
    script_body = _build_pid_tracked_script(start_engine_service(merged), has_advanced_feature)

    rag_configured = os.getenv("RAG_ACC_ENABLED", "").strip().lower() == "true"
    monitor_script = _build_monitor_script(
        fallback_cmd=fallback_cmd, retry_cmd=retry_cmd,
        active_features=active_features_label,
        engine=engine, rag_configured=rag_configured,
    )
    return script_body, monitor_script


def _assemble_startup_command(
    engine: str,
    merged: dict,
    hardware: dict,
    script_body: str,
    monitor_script: str,
) -> str:
    """收集所有前置脚本片段，与引擎脚本体和监控脚本组装成完整的 bash 启动脚本。

    组装顺序（每层职责）：
      1. shebang + 安全选项 + 日志目录 + Prometheus metrics 目录
      2. log_analyzer 部署进度监控
      3. 标准输出无缓冲 + 日志过滤 tee 管道
      4. faulthandler / triton / modelslim 补丁
      5. 用户 env_overrides
      6. accel 加速包安装
      7. 引擎启动脚本体（含 PID 跟踪）
      8. 进程监控（等待 + 回退/重试逻辑）
    """
    analyzer_preamble = _build_analyzer_preamble(engine, merged, hardware)
    faulthandler_patch = _build_faulthandler_patch_preamble(engine)
    triton_patch = build_triton_patch_preamble(engine, _need_triton_patch)
    modelslim_patch = build_modelslim_quarot_patch_preamble(engine)
    accel_preamble = _build_accel_preamble(engine, merged)
    memcache_fragment = build_memcache_hybrid_fragment(engine, merged)
    env_overrides = _build_env_overrides_preamble()
    env_echo_helpers = _build_env_echo_helpers_preamble()
    qwen397_env_exact = _is_qwen35_397b_w8a8_mtp_ascend910c_single_node_8(merged, engine)
    prometheus_export = "" if qwen397_env_exact else (
        "export PROMETHEUS_MULTIPROC_DIR=/var/log/wings/prometheus_multiproc\n"
    )
    python_unbuffered_export = "" if qwen397_env_exact else "export PYTHONUNBUFFERED=1\n"

    full_script = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "mkdir -p /var/log/wings\n"
        # Prometheus multi-process metrics directory: ensure a clean dir
        # on every engine restart so stale worker .db files don't pollute
        # /metrics output.  Located under the shared log volume so that
        # both the engine container and the sidecar proxy can reach it.
        "rm -rf /var/log/wings/prometheus_multiproc\n"
        "mkdir -p /var/log/wings/prometheus_multiproc\n"
        + env_echo_helpers
        # Qwen3.5-397B w8a8-mtp 的 910C 标准脚本要求显式 env 集合精确对齐；
        # 该场景仍保留目录准备和日志管道，但不额外注入全局 export。
        + prometheus_export
        + analyzer_preamble
        # Disable Python stdout full-buffering so that engine ready
        # messages (e.g. "Starting vLLM server on") reach engine.log
        # immediately rather than being stuck in an 8 KB buffer.
        + python_unbuffered_export
        # Filter engine noise from console output and engine.log:
        #   1) /health and /metrics access logs (uvicorn)
        #   2) "Prefill batch" / "Decode batch" scheduler metrics (SGLang)
        # Complete unfiltered logs are saved to engine-full.log for debugging.
        + "exec > >(tee -a /var/log/wings/engine-full.log"
        " | grep --line-buffered -vE"
        " '\"GET\\s+/(health|metrics)\\s|\\b(Prefill|Decode) batch\\b'"
        " | tee -a /var/log/wings/engine.log) 2>&1\n"
        + faulthandler_patch
        + triton_patch
        + modelslim_patch
        + env_overrides
        + accel_preamble
        + memcache_fragment["engine_prelude"]
        + script_body
        + monitor_script
    )
    return _inject_env_echo(full_script)


def build_launcher_plan(launch_args: LaunchArgs, port_plan: PortPlan) -> LauncherPlan:
    """根据启动参数、硬件信息和端口规划生成完整启动脚本。

    编排流程：
    1. 硬件探测 + 配置合并 → merged 参数字典
    2. 引擎/高级特性解析 → engine, 特性状态
    3. 引擎启动脚本 + 进程监控脚本生成
    4. 全部 preamble + 脚本体 + 监控组装成完整 bash 命令

    Args:
        launch_args: 标准化的启动参数（来自 parse_launch_args）
        port_plan:   三层端口分配方案（来自 derive_port_plan）

    Returns:
        LauncherPlan: 包含完整 shell 脚本、合并参数和硬件信息
    """
    hardware = detect_hardware(launch_args.device_count)
    merged = _prepare_merged_params(launch_args, port_plan, hardware)
    prepare_params_for_startup_status(merged)

    engine, has_advanced_feature, active_features_label = _resolve_engine_and_features(
        merged, launch_args,
    )
    script_body, monitor_script = _build_engine_and_monitor_scripts(
        engine, merged, has_advanced_feature, active_features_label,
    )
    command = _assemble_startup_command(engine, merged, hardware, script_body, monitor_script)

    _log_smart_feature_enablement(engine, merged, hardware, command)
    logger.info("Generated start_command.sh (%d bytes)", len(command))
    logger.debug(
        "start_command.sh content:\n"
        "╔══════════════ start_command.sh ══════════════╗\n%s\n"
        "╚══════════════ end start_command.sh ══════════╝",
        command,
    )
    return LauncherPlan(command=command, merged_params=merged, hardware_env=hardware)
