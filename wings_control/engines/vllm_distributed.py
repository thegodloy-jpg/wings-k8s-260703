# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""vLLM distributed script building utilities for Ray and dp_deployment backends."""

import logging
import os
import re
import shlex
from typing import Dict, Any, List

from utils.model_utils import ModelIdentifier, is_glm52_model
from utils.vllm_helpers import (
    _strip_cli_flag, _safe_int, DistScriptCtx, DpDeploymentTopology,
    _transform_dp_cmd, _SH_VLLM_HOST, _SH_IF_DETECT,
)

logger = logging.getLogger(__name__)


def _import_vllm_adapter():
    """Import vllm_adapter in both package and script-style layouts."""
    try:
        from engines import vllm_adapter
    except ImportError:
        import vllm_adapter  # type: ignore
    return vllm_adapter


def _build_comm_env_commands(is_ascend: bool) -> List[str]:
    """返回 Ray 分布式模式的 HCCL/NCCL 通信环境变量设置命令。"""
    if not is_ascend:
        nccl_if = os.getenv('NCCL_SOCKET_IFNAME', 'eth0')
        return [f"export NCCL_SOCKET_IFNAME={nccl_if}", f"export TP_SOCKET_IFNAME={nccl_if}"]
    return [
        "export HCCL_WHITELIST_DISABLE=1",
        "export HCCL_IF_IP=$VLLM_HOST_IP",
        "export HCCL_SOCKET_IFNAME=" + _SH_IF_DETECT,
        "export TP_SOCKET_IFNAME=" + _SH_IF_DETECT,
        "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
        "export ASCEND_PROCESS_LOG_PATH=/tmp/ray_vllm010",
        f"export HCCL_CONNECT_TIMEOUT={os.getenv('HCCL_CONNECT_TIMEOUT', '1800')}",
        f"export HCCL_EXEC_TIMEOUT={os.getenv('HCCL_EXEC_TIMEOUT', '7200')}",
        f"export RAY_CGRAPH_get_timeout={os.getenv('RAY_CGRAPH_get_timeout', '3600')}",
    ]


def _build_ray_wait_loop(nnodes: int) -> List[str]:
    """返回等待所有 Ray 节点加入的 shell 循环命令。

    head 节点最多等待 60*5s。超时后直接 exit 1，让 K8s 明确重启/报错，
    避免 vLLM 在 Ray 集群未就绪时进入更难定位的编译或初始化卡死。
    """
    return [
        "RAY_WAIT_OK=0",
        "for i in $(seq 1 60); do",
        "  COUNT=$(python3 -c \"import ray; ray.init(address='auto',ignore_reinit_error=True); "
        "print(len([n for n in ray.nodes() if n['alive']])); ray.shutdown()\" 2>/dev/null || echo 0)",
        f"  if [ \"$COUNT\" -ge \"{nnodes}\" ]; then RAY_WAIT_OK=1; break; fi",
        f"  echo \"[ray-wait] iter=$i count=$COUNT expected={nnodes}, sleep 5s...\"",
        "  sleep 5",
        "done",
        "if [ \"$RAY_WAIT_OK\" != \"1\" ]; then",
        f"  echo \"[ray-wait] FATAL: only $COUNT/{nnodes} ray nodes joined after 300s. "
        "Check worker pod status / network / RAY_PORT reachability.\" >&2",
        "  exit 1",
        "fi",
        "echo \"[ray-wait] OK: $COUNT ray nodes joined.\"\n",
    ]


def _build_ray_head_start_commands(params: Dict[str, Any], ctx: DistScriptCtx) -> List[str]:
    """构造 Ray head 启动命令和对应 echo 日志。"""
    vllm_adapter = _import_vllm_adapter()
    ray_head_cmd = (
        f"ray start --head --port={ctx.ray_port} --node-ip-address=$VLLM_HOST_IP "
        f"{vllm_adapter.get_ray_resource_flag(ctx.engine, params)} --dashboard-host=$VLLM_HOST_IP\n"
    )
    logger.info("[ray] head start command: %s", ray_head_cmd.strip())
    return [f'echo "[ray] head start command: {ray_head_cmd.strip()}"', ray_head_cmd]


def _build_ray_parallel_overrides(params: Dict[str, Any], ctx: DistScriptCtx) -> tuple[str, str]:
    """为 Ray MoE 架构覆盖 TP/PP。

    Qwen3/MiniMax 等 MoE Ray 路径需要按节点数推 pipeline_parallel_size，并把
    tensor_parallel_size 固定为本机 device_count。普通架构返回原始命令和空追加参数。
    """
    model_info_ray = ModelIdentifier(params.get("model_name"), params.get("model_path"), params.get("model_type"))
    ray_auto_pp_archs = {"Qwen3MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration", "MiniMaxM2ForCausalLM"}
    if getattr(model_info_ray, "model_architecture", None) not in ray_auto_pp_archs:
        return ctx.cmd, ""
    num_nodes = len(ctx.node_ips.split(",")) if ctx.node_ips else 1
    cmd_for_exec = _strip_cli_flag(_strip_cli_flag(ctx.cmd, "--tensor-parallel-size"), "--pipeline-parallel-size")
    logger.info("[vllm_ascend ray] Set parallel parameters: pipeline_parallel_size=%s, tensor_parallel_size=%s",
                num_nodes, params.get("device_count", 1))
    return cmd_for_exec, f" --pipeline-parallel-size {num_nodes} --tensor-parallel-size {params.get('device_count', 1)}"


def _build_ray_head_exec_command(params: Dict[str, Any], ctx: DistScriptCtx, sparse_args: str) -> str:
    """构造 Ray head 节点最终 exec 命令。"""
    vllm_adapter = _import_vllm_adapter()
    eager_flag = " --enforce-eager" if vllm_adapter.need_enforce_eager(ctx.engine) else ""
    speculative_extra = ""
    if vllm_adapter.should_append_auto_speculative_config(params):
        speculative_extra = vllm_adapter.build_speculative_cmd(params, ctx.engine)
    cmd_for_exec, ray_pp_extra = _build_ray_parallel_overrides(params, ctx)
    if params.get("distributed_executor_backend", "ray") == "ray":
        cmd_for_exec = _strip_cli_flag(cmd_for_exec, "--distributed-executor-backend")
        backend_extra = " --distributed-executor-backend ray"
    else:
        backend_extra = ""
    return f"exec {cmd_for_exec}{eager_flag}{speculative_extra}{sparse_args}{ray_pp_extra}{backend_extra}"


def _build_ray_head_commands(params: Dict[str, Any], ctx: DistScriptCtx, sparse_args: str) -> List[str]:
    """组装 Ray head 节点脚本片段：host IP、通信环境、Ray start、节点等待和 exec。"""
    parts: List[str] = [_SH_VLLM_HOST]
    parts.extend(_build_comm_env_commands(ctx.is_ascend))
    parts.append("export GLOO_SOCKET_IFNAME=" + _SH_IF_DETECT + "\n")
    parts.extend(_build_ray_head_start_commands(params, ctx))
    parts.extend(_build_ray_wait_loop(ctx.nnodes))
    parts.append(_build_ray_head_exec_command(params, ctx, sparse_args))
    return parts


def _build_ascend_ray_worker_env(ray_port: str, node_ips: str, head_addr: str = "") -> List[str]:
    """构造 Ascend Ray worker 的 head 探测与通信环境。

    有 master/head_addr 时优先探测该地址，再扫描 node_ips；这样兼容 K8s service
    DNS、上层显式 IP 和多节点裸 IP 三种部署方式。
    """
    if head_addr:
        ip_list_expr = (f"KNOWN_HEAD=\"{head_addr}\"\nNODE_IPS_LIST=\"{node_ips}\"\n"
                        "# 优先尝试已知 head，再扫描其余节点\n"
                        "CANDIDATE_IPS=\"$KNOWN_HEAD $(echo $NODE_IPS_LIST | tr ',' ' ' | grep -v \"^$KNOWN_HEAD$\")\"")
    else:
        ip_list_expr = f"NODE_IPS_LIST=\"{node_ips}\"\nCANDIDATE_IPS=\"$(echo $NODE_IPS_LIST | tr ',' ' ')\""
    return [
        "export HCCL_WHITELIST_DISABLE=1",
        _SH_VLLM_HOST,
        ip_list_expr,
        "HEAD_IP=\"\"",
        f"echo \"[worker] Scanning for Ray head on port {ray_port}...\"",
        "for attempt in $(seq 1 120); do",
        "  for ip in $CANDIDATE_IPS; do",
        f"    if python3 -c \"import socket; s=socket.socket(); s.settimeout(2); "
        f"s.connect(('$ip',{ray_port})); s.close()\" 2>/dev/null; then",
        "      HEAD_IP=$ip",
        f"      echo \"[worker] Found Ray head at $HEAD_IP:{ray_port}\"",
        "      break 2",
        "    fi",
        "  done",
        "  sleep 5",
        "done",
        "if [ -z \"$HEAD_IP\" ]; then echo '[worker] ERROR: Could not find Ray head'; exit 1; fi\n",
        "export HCCL_IF_IP=$VLLM_HOST_IP",
        "export HCCL_SOCKET_IFNAME=" + _SH_IF_DETECT,
        "export TP_SOCKET_IFNAME=" + _SH_IF_DETECT,
        "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
        "export ASCEND_PROCESS_LOG_PATH=/tmp/ray_vllm010",
        "export RAY_CGRAPH_get_timeout=" + os.getenv('RAY_CGRAPH_get_timeout', '3600'),
    ]


def _build_ray_worker_commands(params: Dict[str, Any], ctx: DistScriptCtx) -> List[str]:
    """构造 Ray worker 节点脚本片段。"""
    vllm_adapter = _import_vllm_adapter()
    if ctx.is_ascend:
        parts = _build_ascend_ray_worker_env(ctx.ray_port, ctx.node_ips, ctx.head_addr)
    else:
        nccl_if = os.getenv('NCCL_SOCKET_IFNAME', 'eth0')
        parts = [
            f"export NCCL_SOCKET_IFNAME={nccl_if}", f"export TP_SOCKET_IFNAME={nccl_if}", _SH_VLLM_HOST,
            "for i in $(seq 1 60); do",
            f"  python3 -c \"import socket; s=socket.socket(); s.settimeout(2); "
            f"s.connect(('{ctx.head_addr}',{ctx.ray_port})); s.close()\" 2>/dev/null && break",
            "  sleep 5", "done", f"HEAD_IP=\"{ctx.head_addr}\"",
        ]
    parts.append("export GLOO_SOCKET_IFNAME=" + _SH_IF_DETECT + "\n")
    ray_worker_cmd = (
        f"exec ray start --address=$HEAD_IP:{ctx.ray_port} --node-ip-address=$VLLM_HOST_IP "
        f"{vllm_adapter.get_ray_resource_flag(ctx.engine, params)} --block"
    )
    logger.info("[ray] worker start command: %s", ray_worker_cmd)
    parts.extend([f'echo "[ray] worker start command: {ray_worker_cmd}"', ray_worker_cmd])
    return parts


def _build_ascend_dp_env_commands(params: Dict[str, Any], net_if: str) -> List[str]:
    """构造 Ascend dp_deployment 通信环境。

    GLM-5 / DeepSeek-V4-Pro / 通用 DeepSeek DP 三档默认值不同：
    - GLM-5 DP   : OMP=1     / BUFFSIZE=200  / TIMEOUT=1800
    - V4-Pro     : OMP=10    / BUFFSIZE=2048 / TIMEOUT=7200（与 V4-Pro 模型 env 对齐）
    - 通用 DeepSeek DP: OMP=100 / BUFFSIZE=1024 / TIMEOUT=1800

    该函数在 ``common_env_cmds`` 之后输出，会覆盖前序值，故必须在此处用正确默认值，
    否则 V4-Pro 模型 env 设置的 HCCL_BUFFSIZE=2048 / OMP=10 / TIMEOUT=7200 都会被
    硬编码默认覆盖回去。``os.getenv`` 仍然允许调用方通过环境变量进一步覆盖。
    """
    vllm_adapter = _import_vllm_adapter()
    dp_arch = vllm_adapter.get_deepseek_ascend_dp_model_architecture(params)
    is_glm5_dp = dp_arch == "GlmMoeDsaForCausalLM"
    # GLM-5.2 双机仅对齐 BALANCE_SCHEDULING=0 / +FLASHCOMM1=1（HCCL_BUFFSIZE 维持 1024，
    # 不按模型硬编码——由平台 HCCL_BUFFSIZE env 覆盖）；GLM-5/5.1 维持 BALANCE=1 / 无 FLASHCOMM1。
    is_glm52_dp = is_glm5_dp and is_glm52_model(params.get("model_name"), params.get("model_path"))
    is_v4_pro_dp = vllm_adapter.is_deepseek_v4_pro_adapted_scope(params)
    if is_glm5_dp:
        omp_default, buffsize_default, connect_timeout_default = "1", "1024", "1800"
    elif is_v4_pro_dp:
        omp_default, buffsize_default, connect_timeout_default = "10", "2048", "7200"
    else:
        omp_default, buffsize_default, connect_timeout_default = "100", "1024", "1800"
    env_commands = [
        _SH_VLLM_HOST,
        "export HCCL_WHITELIST_DISABLE=1",
        "export HCCL_IF_IP=$VLLM_HOST_IP",
        f"export GLOO_SOCKET_IFNAME={net_if}",
        f"export TP_SOCKET_IFNAME={net_if}",
        f"export HCCL_SOCKET_IFNAME={net_if}",
        f"export HCCL_CONNECT_TIMEOUT={os.getenv('HCCL_CONNECT_TIMEOUT', connect_timeout_default)}",
        f"export HCCL_EXEC_TIMEOUT={os.getenv('HCCL_EXEC_TIMEOUT', '7200')}",
        "export OMP_PROC_BIND=false",
        f"export OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', omp_default)}",
        f"export HCCL_BUFFSIZE={os.getenv('HCCL_BUFFSIZE', buffsize_default)}",
        'echo "[wings-env] final HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-}"',
        "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
    ]
    if dp_arch in ("DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"):
        env_commands.extend([
            "export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/deepseek-v32/"
            "vendors/customize:${ASCEND_CUSTOM_OPP_PATH:-}",
            "export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/"
            "op_api/lib/:${LD_LIBRARY_PATH:-}",
        ])
    if is_glm5_dp:
        # RoCE 互联场景不注入 HCCL_OP_EXPANSION_MODE=AIV
        if not (params.get("distributed") and vllm_adapter.is_roce_distributed()):
            env_commands.append("export HCCL_OP_EXPANSION_MODE=AIV")
        env_commands.append(
            f"export VLLM_ASCEND_BALANCE_SCHEDULING={'0' if is_glm52_dp else '1'}")
        if is_glm52_dp:
            env_commands.append("export VLLM_ASCEND_ENABLE_FLASHCOMM1=1")
    if vllm_adapter.is_deepseek_ascend_dp_deployment(params):
        env_commands.append(f"export VLLM_ENGINE_READY_TIMEOUT_S={os.getenv('VLLM_ENGINE_READY_TIMEOUT_S', '7200')}")
    return env_commands


def _build_nvidia_dp_env_commands(params: Dict[str, Any], net_if: str) -> List[str]:
    """构造 NVIDIA dp_deployment 通信环境。"""
    nixl_port = params.get("nixl_port", os.getenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "12345"))
    return [
        f"export GLOO_SOCKET_IFNAME={net_if}",
        f"export TP_SOCKET_IFNAME={net_if}",
        f"export NCCL_SOCKET_IFNAME={net_if}",
        f"export VLLM_NIXL_SIDE_CHANNEL_PORT={nixl_port}",
        "export NCCL_IB_DISABLE=0",
        "export NCCL_CUMEM_ENABLE=0",
        "export NCCL_NET_GDR_LEVEL=SYS",
    ]


def _build_dp_env_commands(is_ascend: bool, params: Dict[str, Any]) -> List[str]:
    """返回 dp_deployment 模式的分布式通信环境变量命令。"""
    net_if = os.getenv("NETWORK_INTERFACE", os.getenv("GLOO_SOCKET_IFNAME", "eth0"))
    return _build_ascend_dp_env_commands(params, net_if) if is_ascend else _build_nvidia_dp_env_commands(params, net_if)


def _strip_dp_cli_flags(cmd: str) -> str:
    """移除基础命令中已有的 DP 参数，避免追加拓扑时产生重复 CLI。"""
    for flag in ("--data-parallel-address", "--data-parallel-rpc-port", "--data-parallel-size",
                 "--data-parallel-size-local", "--data-parallel-rank", "--data-parallel-start-rank"):
        cmd = _strip_cli_flag(cmd, flag)
    return re.sub(r"\s+--headless\b", "", re.sub(r"\s+--data-parallel-external-lb\b", "", cmd))


def _resolve_dp_deployment_topology(
    params: Dict[str, Any],
    ctx: DistScriptCtx,
    model_info: ModelIdentifier,
) -> DpDeploymentTopology:
    """Resolve dp_deployment topology.

    普通架构按"每节点 1 个 DP rank"处理；DeepSeek Ascend DP 按
    device_count / tensor_parallel_size 推导本机 DP 数，再乘 nnodes 得到全局 DP。
    V4-Flash/Pro 的 TP/DP 已在 ``_prepare_engine_config`` 中写回 params，
    因此这里读取 engine_config 即可得到一致拓扑。
    """
    vllm_adapter = _import_vllm_adapter()
    dp_arch = (
        vllm_adapter.get_deepseek_ascend_dp_model_architecture(params)
        or model_info.model_architecture
        or ""
    )
    if not (vllm_adapter.is_deepseek_ascend_dp_architecture(dp_arch) and ctx.engine == "vllm_ascend"):
        return DpDeploymentTopology(str(ctx.nnodes), "1", str(ctx.node_rank))
    device_count = _safe_int(params.get("device_count"))
    engine_config = params.get("engine_config") or {}
    tp_size = (
        _safe_int(engine_config.get("tensor_parallel_size"))
        or vllm_adapter.default_deepseek_ascend_dp_tensor_parallel_size(
            dp_arch,
            device_count,
        )
    )
    if not device_count or device_count <= 0:
        raise ValueError("DeepSeek Ascend DP requires a positive device_count to compute DP topology")
    if not tp_size or tp_size <= 0:
        raise ValueError("DeepSeek Ascend DP requires a positive tensor_parallel_size")
    if device_count % tp_size != 0:
        raise ValueError("DeepSeek Ascend DP requires device_count to be divisible by tensor_parallel_size: "
                         f"device_count={device_count}, tensor_parallel_size={tp_size}")
    dp_size_local = device_count // tp_size
    return DpDeploymentTopology(
        str(dp_size_local * int(ctx.nnodes)),
        str(dp_size_local),
        str(int(ctx.node_rank) * dp_size_local),
    )


def _build_dp_exec_command(
    ctx: DistScriptCtx,
    dp_cmd: str,
    dp_rpc_port: str,
    topology: DpDeploymentTopology,
    include_rank0_start_rank: bool = False,
) -> str:
    """根据 node_rank 构造 dp_deployment head/worker 的最终 exec 行。"""
    common = (
        f" --data-parallel-address {shlex.quote(ctx.head_addr)}"
        f" --data-parallel-rpc-port {dp_rpc_port}"
        f" --data-parallel-size {topology.dp_size}"
        f" --data-parallel-size-local {topology.dp_size_local}"
    )
    if ctx.node_rank == 0:
        rank0_start_rank = (
            f" --data-parallel-start-rank {topology.dp_start_rank}"
            if include_rank0_start_rank
            else ""
        )
        return f"exec {dp_cmd}{common}{rank0_start_rank}"
    dp_cmd_headless = re.sub(r"\s*--port\s+(?:'[^']*'|\S+)", "", re.sub(r"\s*--host\s+(?:'[^']*'|\S+)", "", dp_cmd))
    return f"exec {dp_cmd_headless}{common} --headless --data-parallel-start-rank {topology.dp_start_rank}"


def _build_dp_deployment_commands(params: Dict[str, Any], ctx: DistScriptCtx, sparse_args: str = "") -> List[str]:
    """组装 dp_deployment 脚本片段：环境、命令转换、speculative/sparse 参数和拓扑 CLI。"""
    vllm_adapter = _import_vllm_adapter()
    dp_rpc_port = str(params.get("rpc_port", os.getenv('VLLM_DP_RPC_PORT', '13355')))
    model_info = ModelIdentifier(params.get("model_name"), params.get("model_path"), params.get("model_type"))
    topology = _resolve_dp_deployment_topology(params, ctx, model_info)
    dp_cmd = _strip_dp_cli_flags(_transform_dp_cmd(ctx.cmd))
    speculative_extra = ""
    if vllm_adapter.should_append_auto_speculative_config(params):
        speculative_extra = vllm_adapter.build_speculative_cmd(params, ctx.engine)
    parts = _build_dp_env_commands(ctx.is_ascend, params)
    include_rank0_start_rank = bool(params.get("_force_data_parallel_start_rank_on_rank0"))
    parts.append(_build_dp_exec_command(ctx, f"{dp_cmd}{speculative_extra}{sparse_args}",
                                       dp_rpc_port, topology, include_rank0_start_rank))
    return parts


def _resolve_vllm_dist_params(params: Dict[str, Any]) -> tuple[str, str, str]:
    """解析分布式 head 地址、节点列表和 Ray 端口。"""
    head_addr = params.get("ray_head_ip") or params.get("master_ip") or params.get("head_node_addr", "infer-0.infer-hl")
    node_ips = params.get("node_ips") or params.get("nodes") or os.getenv("NODE_IPS", head_addr)
    return head_addr, node_ips, str(params.get("ray_head_port", os.getenv("RAY_PORT", "28020")))


def _build_vllm_distributed_script(params: Dict[str, Any], cmd: str, common_env_cmds: List[str],
                                   engine: str, sparse_args: str) -> str:
    """组装分布式脚本主体。

    ``common_env_cmds`` 已包含基础环境、KV offload、QAT、模型专属 env 等公共片段；
    本函数只根据 backend 分派 Ray 或 dp_deployment 分支，最后统一拼成 start script。
    """
    node_rank = params.get("node_rank", 0)
    head_addr, node_ips, ray_port = _resolve_vllm_dist_params(params)
    ctx = DistScriptCtx(engine=engine, cmd=cmd, is_ascend=(engine == "vllm_ascend"), node_rank=node_rank,
                        nnodes=params.get("nnodes", 1), head_addr=head_addr, ray_port=ray_port, node_ips=node_ips)
    script_parts = list(common_env_cmds)
    if params.get("distributed_executor_backend", "ray") == "ray":
        script_parts.extend(_build_ray_head_commands(params, ctx, sparse_args) if node_rank == 0
                            else _build_ray_worker_commands(params, ctx))
    else:
        script_parts.extend(_build_dp_deployment_commands(params, ctx, sparse_args))
    return "\n".join(script_parts) + "\n"