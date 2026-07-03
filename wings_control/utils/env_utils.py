"""环境变量解析辅助函数，供 adapter 和 launcher 配置解析使用。

复用自 wings 项目的环境工具模块。

功能概述:
    本模块集中管理环境变量的解析，避免各模块重复实现。
    主要分类:
    - IP 地址获取  : get_master_ip(), get_local_ip(), get_node_ips()
    - 端口获取     : get_server_port(), get_master_port(), get_vllm_distributed_port() 等
    - 特性开关     : get_lmcache_env(), get_pd_role_env(), get_router_env() 等
    - 配置路径     : get_router_nats_path_env() 等

Sidecar 架构契约:
    - 集中环境变量解析，避免漂移
    - 默认值明确且对 sidecar 场景安全
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import socket
import logging

logger = logging.getLogger(__name__)


def validate_ip(ip_str):
    """校验 IP 地址格式是否合法（仅支持 IPv4）。

    Args:
        ip_str (str): IP 地址字符串

    Returns:
        bool: 合法的 IPv4 地址返回 True，否则返回 False
    """
    if not ip_str:
        return False

    # IPv4
    try:
        socket.inet_aton(ip_str)
        return True
    except socket.error:
        return False


def get_master_ip():
    """获取分布式集群的 master IP 地址。

    从 MASTER_IP 环境变量读取，用于 Ray/HCCL 等分布式通信初始化。

    Returns:
        str | None: master IP 地址，未设置时返回 None
    """
    master_ip = os.getenv('MASTER_IP', None)
    return master_ip


def get_local_ip():
    """获取本节点的 IP 地址。

    优先从 RANK_IP 环境变量读取，若未设置则调用 socket 获取主机名对应的 IP。

    Returns:
        str: 本机 IP 地址
    """
    host_ip = os.getenv('RANK_IP', None)
    if not host_ip:
        hostname = socket.gethostname()
        try:
            host_ip = socket.gethostbyname(hostname)
        except socket.gaierror as e:
            logger.warning("get_local_ip: hostname '%s' lookup failed: %s; falling back to 127.0.0.1", hostname, e)
            host_ip = "127.0.0.1"
    return host_ip


def get_node_ips():
    """获取分布式推理集群中所有节点的 IP 地址列表。

    从 NODE_IPS 环境变量读取，若值带有方括号则自动去除，
    返回逗号分隔的 IP 字符串，供多节点分布式通信使用。

    Returns:
        str | None: 逗号分隔的节点 IP 字符串，未设置时返回 None
    """
    node_ips = os.getenv('NODE_IPS')
    if node_ips and "[" in node_ips:
        node_ips = node_ips.replace("[", "").replace("]", "")
    return node_ips


def get_server_port():
    """获取推理服务监听端口。

    从 SERVER_PORT 环境变量读取，将其转换为整数并返回。
    若未设置或值非法（无法转为整数），则记录警告并返回 None。

    Returns:
        int | None: 服务端口号，未设置或格式非法时返回 None
    """
    port = os.getenv('SERVER_PORT')
    if port:
        try:
            return int(port)
        except ValueError:
            logger.warning("Invalid SERVER_PORT value %r, ignoring", port)
    return None


def get_master_port():
    """获取分布式训练/推理主节点（Master）通信端口。

    从 MASTER_PORT 环境变量读取，用于 Ray/HCCL 等分布式框架
    初始化时指定主节点的通信端口。若未设置或值非法则返回 None。

    Returns:
        int | None: 主节点端口号，未设置或格式非法时返回 None
    """
    port = os.getenv('MASTER_PORT')
    if port:
        try:
            return int(port)
        except ValueError:
            logger.warning("Invalid MASTER_PORT value %r, ignoring", port)
    return None


def get_worker_port():
    """获取工作节点（Worker）通信端口。

    从 WORKER_PORT 环境变量读取，用于分布式推理场景下
    Worker 节点的服务监听端口。若未设置或值非法则返回 None。

    Returns:
        int | None: 工作节点端口号，未设置或格式非法时返回 None
    """
    port = os.getenv('WORKER_PORT')
    if port:
        try:
            return int(port)
        except ValueError:
            logger.warning("Invalid WORKER_PORT value %r, ignoring", port)
    return None


def get_vllm_distributed_port():
    """获取 vLLM 分布式推理的 Ray Worker 通信端口。

    从 VLLM_DISTRIBUTED_PORT 环境变量读取，用于 vLLM 多卡/多节点场景下
    Ray Worker 之间的内部通信。若未设置或值非法则返回 None。

    Returns:
        int | None: vLLM 分布式通信端口号，未设置或格式非法时返回 None
    """
    port = os.getenv('VLLM_DISTRIBUTED_PORT')
    if port:
        try:
            return int(port)
        except ValueError:
            logger.warning("Invalid VLLM_DISTRIBUTED_PORT value %r, ignoring", port)
    return None


def get_sglang_distributed_port():
    """获取 SGLang 分布式推理通信端口。

    从 SGLANG_DISTRIBUTED_PORT 环境变量读取，用于 SGLang 多节点分布式推理
    场景下各节点之间的通信。若未设置或值非法则返回 None。

    Returns:
        int | None: SGLang 分布式通信端口号，未设置或格式非法时返回 None
    """
    port = os.getenv('SGLANG_DISTRIBUTED_PORT')
    if port:
        try:
            return int(port)
        except ValueError:
            logger.warning("Invalid SGLANG_DISTRIBUTED_PORT value %r, ignoring", port)
    return None


def get_lmcache_env():
    """检查 KVCache Offload（卸载到 CPU/磁盘）功能是否启用。

    从 ENABLE_KV_OFFLOAD 环境变量读取。
    判断是否将 KVCache 卸载到 CPU 内存或本地磁盘以节省 GPU 显存。

    Returns:
        bool: 启用返回 True，未设置或为 'false' 时返回 False
    """
    return os.getenv('ENABLE_KV_OFFLOAD', 'false').lower() == 'true'


def get_qat_env():
    """检查 QAT（Quick Assist Technology）压缩功能是否启用。

    从 ENABLE_KV_QAT 环境变量读取。
    L3 门控：仅在 L2 磁盘开关 ENABLE_KV_DISK_OFFLOAD=true 时生效（需求一 §A.1）。

    Returns:
        bool: 启用返回 True，否则返回 False
    """
    # L3 门控：需 L2 磁盘开关为 true
    if os.getenv('ENABLE_KV_DISK_OFFLOAD', 'false').lower() != 'true':
        return False
    return os.getenv('ENABLE_KV_QAT', 'false').lower() == 'true'


def get_cold_start_env():
    """检查 KVCache 冷启动预热（Cold Start Pre-caching）功能是否启用。

    从 LMCACHE_COLD_START 环境变量读取，判断是否在引擎启动时
    对高频 Prompt 的 KVCache 进行预热（pre-caching），减少首次推理延迟。
    需配合 LMCache Offload 功能一起使用。

    Returns:
        bool: 启用返回 True，未设置或为 'false' 时返回 False
    """
    cold_start = os.getenv('LMCACHE_COLD_START', 'false')
    return cold_start.lower() == 'true'


def get_pd_role_env():
    """获取 PD 分离式推理（Disaggregated Inference）中的角色。

    从 PD_ROLE 环境变量读取，返回当前节点在 Prefill-Decode 分离架构中的角色：
    - "P"：Prefill 节点，负责预填充阶段
    - "D"：Decode 节点，负责解码阶段
    若未设置或值不合法则返回空字符串，表示未启用 PD 分离推理。

    Returns:
        str: 角色标识 "P" 或 "D"，未启用时返回空字符串
    """
    pd_role = os.getenv('PD_ROLE', '')
    if pd_role and pd_role not in ("P", "D"):
        logger.warning("PD_ROLE id not P or D, PD is not enabled")
        pd_role = ''
    return pd_role


def get_router_env():
    """检查 Wings Router（基于 NATS 的负载均衡路由）是否启用。

    从 WINGS_ROUTE_ENABLE 环境变量读取，判断是否启用 Wings 平台的
    推理请求路由功能，启用后请求将通过 NATS 进行负载均衡分发。

    Returns:
        bool: 启用返回 True，未设置或为 'false' 时返回 False
    """
    router = os.getenv('WINGS_ROUTE_ENABLE', 'false')
    router = router.lower() == 'true'
    return router


def get_router_instance_group_name_env():
    """获取 Wings Router 中当前实例所属的分组名称。

    从 WINGS_ROUTE_INSTANCE_GROUP_NAME 环境变量读取，用于在 Wings Router
    中标识该推理实例所属的逻辑分组，便于路由策略按组分发请求。

    Returns:
        str: 实例分组名称，未设置时返回空字符串
    """
    env_name = "WINGS_ROUTE_INSTANCE_GROUP_NAME"
    router_instance_group_name = os.getenv(env_name, '')
    return router_instance_group_name


def get_router_instance_name_env():
    """获取 Wings Router 中当前推理实例的唯一标识名称。

    从 WINGS_ROUTE_INSTANCE_NAME 环境变量读取，用于在 Wings Router
    路由表中唯一标识当前推理服务实例。

    Returns:
        str: 实例名称，未设置时返回空字符串
    """
    env_name = "WINGS_ROUTE_INSTANCE_NAME"
    router_instance_name = os.getenv(env_name, '')
    return router_instance_name


def get_router_nats_path_env():
    """获取 NATS 服务器的连接端点地址。

    从 WINGS_ROUTE_NATS_PATH 环境变量读取，返回 NATS 消息服务器的
    连接 URL，供 Wings Router 建立消息通信使用。

    Returns:
        str: NATS 服务器连接地址，未设置时返回空字符串
    """
    env_name = "WINGS_ROUTE_NATS_PATH"
    router_nats_path = os.getenv(env_name, '')
    return router_nats_path


def get_config_force_env():
    """检查是否强制使用用户提供的配置覆盖所有默认值。

    从 CONFIG_FORCE 环境变量读取，当启用时，用户提供的推理引擎配置
    将完全覆盖系统自动生成的默认配置参数。

    Returns:
        bool: 启用返回 True，未设置或为 'false' 时返回 False
    """
    config_force = os.getenv('CONFIG_FORCE', 'false')
    config_force = config_force.lower() == 'true'
    return config_force


def get_speculative_decoding_env():
    """获取推测解码是否开启环境变量。

    Returns:
        bool: 返回 SD_ENABLE 环境变量的值，如果未设置则返回 False
    """
    speculative_enable = os.getenv('SD_ENABLE', 'false')
    speculative_enable = speculative_enable.lower() == 'true'
    return speculative_enable


def get_sparse_env():
    """获取稀疏 KV 是否开启环境变量。

    Returns:
        bool: 返回 SPARSE_ENABLE 环境变量的值，如果未设置则返回 False
    """
    sparse_enable = os.getenv('SPARSE_ENABLE', 'false')
    sparse_enable = sparse_enable.lower() == 'true'
    return sparse_enable


# SmartKVSparse 精度/性能档位取值（需求一 §2.4）
SPARSE_LEVEL_ACCURACY_FIRST = 'accuracy_first'      # 精度优先
SPARSE_LEVEL_PERFORMANCE_FIRST = 'performance_first'  # 性能优先


def get_sparse_level_env():
    """读取 SmartKVSparse 请求档位（accuracy_first / performance_first）。需求一 §2.4。

    取值语义：
        - ``accuracy_first``（精度优先）：使用 sparse 表 accuracy topk；
        - ``performance_first``（性能优先）：优先使用 sparse 表 performance topk，缺失则回退本行 accuracy topk。
    缺省（未下发）或非法取值一律回落 ``accuracy_first``。
    本函数只返回「请求档位」（已小写规整），具体 topk 由 sparse 白名单表解析。

    Returns:
        str: ``accuracy_first`` 或 ``performance_first``
    """
    raw = os.getenv('SPARSE_LEVEL', '').strip().lower()
    if raw in (SPARSE_LEVEL_ACCURACY_FIRST, SPARSE_LEVEL_PERFORMANCE_FIRST):
        return raw
    return SPARSE_LEVEL_ACCURACY_FIRST


def log_kvcache_offload_config(lmcache_offload_enabled, qat_enabled):
    """记录 KVCache Offload 相关配置信息，用于调试和运维排查。

    当 KVCache Offload 功能启用时，打印 CPU 内存、本地磁盘路径及大小限制
    等配置项。若 QAT 压缩也启用，还会打印 QAT 损失级别和实例数等参数。

    Args:
        lmcache_offload_enabled (bool): KVCache Offload 是否启用
        qat_enabled (bool): QAT 压缩是否启用
    """
    if not lmcache_offload_enabled:
        return

    logger.info("[KVCache Offload] KVCache Offload feature is enabled: %s", lmcache_offload_enabled)
    logger.info("[KVCache Offload] Local memory is enabled: %s", os.getenv('ENABLE_KV_MEM_OFFLOAD', 'Not set'))
    logger.info("[KVCache Offload] Local memory max size: %s", os.getenv('KV_MEM_OFFLOAD_SIZE', 'Not set'))
    logger.info("[KVCache Offload] Local disk path: %s", os.getenv('KV_DISK_OFFLOAD_PATH', 'Not set'))
    logger.info("[KVCache Offload] Local disk max size: %s", os.getenv('KV_DISK_OFFLOAD_SIZE', 'Not set'))

    logger.info("[KVCache Offload] QAT Compression feature is enabled: %s", qat_enabled)
    if not qat_enabled:
        return

    logger.info("[KVCache Offload] QAT Loss Level: %s", os.getenv('KV_QAT_COMPRESS_LEVEL', 'Not set'))
    logger.info("[KVCache Offload] QAT Instance Number: %s", os.getenv('KV_QAT_INSTANCE_NUM', 'Not set'))


def check_env():
    """校验环境变量配置的一致性与合法性。

    检查各环境变量之间的依赖关系是否满足，例如：
    - 启用 QAT 压缩时必须同时启用 LMCache Offload
    - 启用 QAT 时必须配置本地磁盘路径和大小
    - 配置了 Wings Router 实例分组时必须同时设置实例名和 NATS 路径
    校验通过前会先调用 log_kvcache_offload_config 记录配置详情。

    Raises:
        ValueError: 当环境变量配置存在冲突或缺少必要依赖项时抛出

    Returns:
        bool: 校验通过返回 True
    """
    qat = get_qat_env()
    lmcache_offload = get_lmcache_env()

    log_kvcache_offload_config(lmcache_offload, qat)

    if qat:
        if not lmcache_offload:
            raise ValueError("QAT is enabled but KV offload (ENABLE_KV_OFFLOAD) is not configured")
        elif not os.getenv("KV_DISK_OFFLOAD_PATH") or not os.getenv("KV_DISK_OFFLOAD_SIZE"):
            raise ValueError("QAT is enabled but KV_DISK_OFFLOAD_PATH or KV_DISK_OFFLOAD_SIZE is not configured")

    router_instance_group_name = get_router_instance_group_name_env()
    if router_instance_group_name:
        if not get_router_instance_name_env():
            raise ValueError("Wings Router is enabled but instance name is not set")
        if not get_router_nats_path_env():
            raise ValueError("Wings Router enabled but nats path is not set")
    return True