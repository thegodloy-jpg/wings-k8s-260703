# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""
SGLang 引擎适配器。

在 sidecar launcher 模式下，本模块将统一参数转换为 SGLang 启动脚本。
与 vllm-engine 类似，将 start_command.sh 写入共享卷供引擎容器执行。

核心接口:
  - build_start_command(params) -> str : 生成 SGLang 启动命令
  - build_start_script(params)  -> str : 生成完整 bash 脚本（不含 shebang）
"""

import ast
import json
import logging
import os
import shlex
from typing import Dict, Any, List, Optional

from utils.env_utils import get_local_ip, validate_ip


# 日志记录器
logger = logging.getLogger(__name__)

# 模块根目录：用于定位环境脚本文件
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_sglang_node_host() -> str:
    """Return the concrete node/Pod IP for SGLang distributed communication."""
    for env_key in ("POD_IP", "RANK_IP"):
        candidate = os.getenv(env_key, "").strip()
        if validate_ip(candidate) and candidate != "0.0.0.0":
            return candidate

    local_ip = get_local_ip()
    if validate_ip(local_ip) and local_ip != "0.0.0.0":
        return local_ip

    logger.warning(
        "[SGLang] Unable to resolve Pod IP for --host; falling back to 127.0.0.1"
    )
    return "127.0.0.1"


def _build_base_env_commands(params: Dict[str, Any], root: str) -> List[str]:
    """构建 SGLang 基础环境设置命令。

    查找并加载项目中的 SGLang 环境设置脚本（如果存在）。
    脚本路径: <root>/wings/config/set_sglang_env.sh
    若全局 helper 已注入，则通过 wings_source_env_with_diff 打印 source 前后
    新增/变化的环境变量；否则回退到普通 source。

    Args:
        params: 参数字典（当前未使用，保留为扩展点）
        root:   项目根目录路径

    Returns:
        List[str]: 环境设置命令列表，可能为空

    注意:
        - 脚本不存在时记录警告并返回空列表
        - 不会导致启动失败，仅影响特定特性
        - set +u / set -u 用于兼容脚本中引用未定义变量的场景
    """
    env_script = os.path.join(root, "wings", "config", "set_sglang_env.sh")
    if os.path.exists(env_script):
        quoted_script = shlex.quote(env_script)
        return [
            "set +u",
            "if command -v wings_source_env_with_diff >/dev/null 2>&1; then "
            f"wings_source_env_with_diff {quoted_script} set_sglang_env.sh; "
            f"else source {quoted_script}; fi",
            "set -u",
        ]
    logger.debug("SGLang env script not found at %s; starting without sourcing env script", env_script)
    return []


def _build_sglang_cmd_parts(params: Dict[str, Any]) -> str:
    """构建 SGLang 核心启动命令字符串。

    将 engine_config 字典转换为 sglang.launch_server CLI 参数格式：
    python3 -m sglang.launch_server --arg1 value1 --arg2 value2 ...

    参数转换规则:
    - 参数名: snake_case → kebab-case (如 tp_size → --tp-size)
    - 布尔值: True → 仅输出 flag (如 --disable-log-stats)
    - 布尔值: False → 跳过，不输出任何内容
    - 空字符串: 跳过，避免生成空参数（如 --model-path '')
    - JSON 字典: 用单引号包裹，确保 shell 正确解析
    - 其他值: 使用 shlex.quote 安全转义

    Args:
        params: 参数字典，必须包含 engine_config 子字典

    Returns:
        str: 完整的 SGLang 启动命令字符串

    示例输出:
        python3 -m sglang.launch_server \\
            --model-path /weights --host 0.0.0.0 --port 17000 \\
            --tp-size 4 --trust-remote-code
    """
    engine_config = params.get("engine_config", {})

    # python3  /usr/bin/python
    cmd_parts = ["python3", "-m", "sglang.launch_server"]

    for arg, value in engine_config.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            # Skip empty string args to avoid passing blank CLI flags
            continue

        arg_name = f"--{arg.replace('_', '-')}"
        # 参数去重：跳过已存在的 CLI 参数，防止分布式参数与 engine_config 冲突
        if arg_name in cmd_parts:
            continue

        if isinstance(value, bool):
            if value:
                cmd_parts.append(arg_name)
        elif isinstance(value, dict):
            # dict 透传：紧凑 JSON + shlex.quote，避免 str(dict) 产生单引号 key
            cmd_parts.extend([arg_name, shlex.quote(json.dumps(value, ensure_ascii=False, separators=(',', ':')))])
        elif isinstance(value, str) and (
            (value.strip().startswith('{') and value.strip().endswith('}')) or
            (value.strip().startswith('[') and value.strip().endswith(']'))
        ):
            stripped = value.strip()
            normalized: Optional[str] = None
            try:
                parsed = json.loads(stripped)
                normalized = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed = ast.literal_eval(stripped)
                    normalized = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
                    logger.warning(
                        "[sglang] %s value is Python-repr str (single-quoted keys); "
                        "auto-normalized to JSON.",
                        arg_name,
                    )
                except (ValueError, SyntaxError):
                    pass
            if normalized is not None:
                cmd_parts.extend([arg_name, shlex.quote(normalized)])
            else:
                cmd_parts.extend([arg_name, shlex.quote(value)])
        else:
            cmd_parts.extend([arg_name, shlex.quote(str(value))])

    return " ".join(cmd_parts)


def build_start_command(params: Dict[str, Any]) -> str:
    """为 launcher 生成 SGLang 启动命令字符串（不含 shebang 和环境设置）。

    此函数仅进行命令拼装，不启动任何子进程。

    单机模式:
        python3 -m sglang.launch_server --model-path ... --host ... --port ...

    分布式模式 (多节点, nnodes > 1):
        python3 -m sglang.launch_server ... \\
            --nnodes <n> --node-rank <rank> --dist-init-addr <addr:port>

    Args:
        params: 参数字典，包含以下关键字段:
            - engine_config: SGLang 启动参数
            - distributed:   是否分布式模式
            - nnodes:        总节点数
            - node_rank:     当前节点编号
            - head_node_addr: 主节点地址 (可包含端口号)

    Returns:
        str: SGLang 启动命令字符串

    环境变量:
        - SGLANG_DIST_PORT: 分布式通信端口，默认 28030
    """
    cmd = _build_sglang_cmd_parts(params)
    is_distributed = params.get("distributed", False)
    nnodes = params.get("nnodes", 1)
    node_rank = params.get("node_rank", 0)
    head_node_addr = params.get("head_node_addr") or params.get("master_ip") or "127.0.0.1"

    if is_distributed and nnodes > 1:
        cmd += f" --nnodes {nnodes} --node-rank {node_rank}"
        if ":" in head_node_addr:
            cmd += f" --dist-init-addr {shlex.quote(head_node_addr)}"
        else:
            # dist_port: params 优先（config_loader 从 distributed_config.json 注入），其次环境变量
            sglang_dist_port = str(
                params.get(
                    "dist_port",
                    os.getenv("SGLANG_DISTRIBUTED_PORT", os.getenv("SGLANG_DIST_PORT", "28030")),
                )
            )
            cmd += f" --dist-init-addr {shlex.quote(head_node_addr)}:{sglang_dist_port}"
        # 非 master 节点需要绑定所有地址（对齐 A）
        # 说明：rank=0 的 --host 由 engine_config 注入（通常已设为 Pod IP），
        # rank>0 显式追加以确保 master 可达 worker 的内部通信端口。
        # SGLang 对 worker 节点的 --host 参数不影响 HTTP API（仅 rank=0 启动 HTTP），
        # 但部分版本用它绑定 gRPC/NCCL 通信，因此保留此逻辑。
        if node_rank != 0:
            cmd += f" --host {shlex.quote(_resolve_sglang_node_host())}"

    return cmd


def _build_sglang_trace_env_commands(params: Dict[str, Any]) -> List[str]:
    """构建 SGLang OTLP trace 相关的环境变量命令。

    仅在 enable_otlp_traces 为 True 时生效。
    """
    if not params.get("enable_otlp_traces"):
        return []

    logger.info("[AdvFeature-EnableTrace] SGLang OTLP trace env commands injected")
    return [
        'export OTEL_SERVICE_NAME="sglang"',
        'export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true',
        'export OTEL_RESOURCE_ATTRIBUTES="service.instance.id=$(hostname),k8s.pod.name=$(hostname)"',
        'export SGLANG_TRACE_LEVEL=2',
    ]


def build_start_script(params: Dict[str, Any]) -> str:
    """生成完整的 bash 启动脚本体（start_command.sh 内容，不含 shebang）。

    这是 SGLang 适配器的主要入口，生成的脚本结构：

        [source set_sglang_env.sh]   # 环境设置（可选）
        export GLOO/TP/NCCL_SOCKET_IFNAME=...  # 分布式模式
        exec python3 -m sglang.launch_server --model-path ... --host ... --port ...

    使用 exec 确保引擎进程替换 shell 成为 PID 1，正确接收容器信号。
    脚本出口处统一调用 _inject_env_echo，与 vllm/mindie 适配器对齐：
    所有 export 和启动命令均自动注入 [wings-env] / [wings-cmd] echo 行。

    Args:
        params: 参数字典，传递给 build_start_command()

    Returns:
        str: 完整的 bash 脚本体（以换行符结尾）
    """
    from engines.vllm_adapter import _inject_env_echo  # 延迟导入，避免循环
    env_cmds = _build_base_env_commands(params, root_dir)
    core_cmd = build_start_command(params)

    lines: List[str] = []
    lines.extend(env_cmds)

    # 分布式通信环境变量（对齐 A：GLOO/TP/NCCL_SOCKET_IFNAME）
    is_distributed = params.get("distributed", False)
    nnodes = params.get("nnodes", 1)
    if is_distributed and nnodes > 1:
        net_if = os.getenv("NETWORK_INTERFACE", os.getenv("GLOO_SOCKET_IFNAME", "eth0"))
        lines.append(f"export GLOO_SOCKET_IFNAME={net_if}")
        lines.append(f"export TP_SOCKET_IFNAME={net_if}")
        lines.append(f"export NCCL_SOCKET_IFNAME={net_if}")

    # OTLP trace 环境变量
    lines.extend(_build_sglang_trace_env_commands(params))

    lines.append(f"exec {core_cmd}")
    script = "\n".join(lines) + "\n"
    return _inject_env_echo(script)


def start_engine(params: Dict[str, Any]):
    """旧版兼容接口（sidecar launcher 模式中已禁用）。

    在 sidecar 架构中，适配器不允许直接启动推理进程。
    应使用 build_start_script() 或 build_start_command()
    生成脚本并写入共享卷。

    Raises:
        RuntimeError: 始终抛出，阻止意外调用
    """
    raise RuntimeError(
        "start_engine is disabled in launcher mode. "
        "Use build_start_command() / build_start_script() and write to shared volume instead."
    )
