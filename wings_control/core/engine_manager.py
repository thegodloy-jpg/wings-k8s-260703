# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""引擎适配器调度器。

它只做一件事：根据 `params["engine"]` 找到对应 adapter，并让 adapter 生成
`start_command.sh` 的脚本内容。

映射关系如下：
- `vllm` -> `engines.vllm_adapter`
- `vllm_ascend` -> 仍复用 `vllm_adapter`，只是参数和环境不同
- `sglang` -> `engines.sglang_adapter`
- `mindie` -> `engines.mindie_adapter`
"""

import importlib
import logging
import re as _re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# adapter 模块所在的 Python 包路径；运行时通过 importlib 拼接模块全名加载
ENGINE_ADAPTER_PACKAGE = "engines"

# 引擎别名映射表：将逻辑引擎名映射到真实 adapter 文件名。
# `vllm_ascend` 不是独立 adapter 文件，而是复用 vllm 的命令拼装逻辑，
# 差异通过参数（如 device=ascend）在 adapter 内部分支处理。
ENGINE_ADAPTER_ALIASES: Dict[str, str] = {
    "vllm_ascend": "vllm",
}


def start_engine_service(params: Dict[str, Any]) -> str:
    """根据 engine 类型加载 adapter 并生成启动脚本。

    执行流程：
    1. 从 params['engine'] 取得引擎名（如 vllm、sglang、mindie）
    2. 通过 ENGINE_ADAPTER_ALIASES 解析别名（vllm_ascend → vllm）
    3. 拼接模块名 engines.<adapter>_adapter 并用 importlib 动态导入
    4. 优先调用 build_start_script()（完整脚本），回退到 build_start_command()（单行命令）

    Args:
        params: 合并后的完整参数字典，至少包含 'engine' 键

    Returns:
        str: shell 脚本内容，将被写入 /shared-volume/start_command.sh

    Raises:
        ValueError: 缺少 engine 键
        ImportError: adapter 模块不存在（引擎名拼写错误或缺少实现文件）
        AttributeError: adapter 既没有 build_start_script 也没有 build_start_command
    """
    engine_name = params.get("engine")
    if not engine_name:
        raise ValueError("Missing 'engine' key in params dict.")

    adapter_key = ENGINE_ADAPTER_ALIASES.get(engine_name, engine_name)
    # 白名单校验：只允许字母数字和下划线，防止 importlib 加载任意模块
    if not _re.match(r'^[a-zA-Z0-9_]+$', adapter_key):
        raise ValueError(f"Invalid engine adapter key: '{adapter_key}'")
    logger.info("Loading adapter for engine: %s (adapter: %s)", engine_name, adapter_key)

    adapter_module_name = f"{ENGINE_ADAPTER_PACKAGE}.{adapter_key}_adapter"
    try:
        adapter_module = importlib.import_module(adapter_module_name)
    except ImportError as e:
        logger.error(
            "Failed to import adapter '%s' for engine '%s'.",
            adapter_module_name,
            engine_name,
            exc_info=True,
        )
        raise ImportError(
            f"Adapter for engine '{engine_name}' not found: {adapter_module_name}.py"
        ) from e

    # 优先使用更完整的脚本生成接口。
    # build_start_script 支持多行脚本（如 export 环境变量 + exec 命令），
    # 是推荐的 adapter 实现方式。
    if hasattr(adapter_module, "build_start_script"):
        logger.info("Using build_start_script from %s", adapter_module_name)
        return adapter_module.build_start_script(params)

    # 兼容旧 adapter：如果只有单条命令，则自动包装成 `exec ...`。
    # exec 确保 engine 进程替换 shell 成为 PID 1，正确接收信号。
    if hasattr(adapter_module, "build_start_command"):
        logger.info(
            "build_start_script not found; falling back to build_start_command from %s",
            adapter_module_name,
        )
        cmd = adapter_module.build_start_command(params)
        return f"exec {cmd}\n"

    raise AttributeError(
        f"Adapter '{adapter_module_name}' implements neither "
        f"build_start_script nor build_start_command."
    )
