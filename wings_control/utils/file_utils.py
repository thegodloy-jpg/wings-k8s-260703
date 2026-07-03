"""安全文件 I/O 辅助函数，用于共享卷工件和配置文件操作。

复用自 wings 项目的文件工具模块。

功能概述:
    本模块提供安全的文件操作函数，主要功能:
    - get_directory_size()    : 计算目录总大小（字节）
    - safe_write_file()       : 安全写入文件（支持 JSON/文本，指定权限）
    - check_permission_640()  : 检查文件权限是否为 640
    - check_torch_dtype()     : 检查模型 config.json 中 torch_dtype 是否支持
    - load_json_config()      : 安全加载 JSON 配置文件

Sidecar 架构契约:
    - 保持命令/状态工件写入的可靠性
    - 权限和 JSON 写入语义保持稳定
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
import stat
import json
from dataclasses import dataclass, field
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

INDENT = 4

# 默认文件打开标志和权限
_DEFAULT_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
_DEFAULT_MODES = stat.S_IRUSR | stat.S_IWUSR


@dataclass
class WriteOptions:
    """安全写入文件的选项参数封装。

    Attributes:
        is_json: 是否以 JSON 格式写入
        flags:   文件打开标志（默认 O_WRONLY|O_CREAT|O_TRUNC）
        modes:   文件权限模式（默认 600）
        atomic:  是否使用原子写入（先写临时文件再 rename）
    """
    is_json: bool = False
    flags: int = field(default_factory=lambda: _DEFAULT_FLAGS)
    modes: int = field(default_factory=lambda: _DEFAULT_MODES)
    atomic: bool = False


def _write_content(f, content: Any, is_json: bool) -> None:
    """将内容写入已打开的文件对象。"""
    if is_json:
        json.dump(content, f, indent=INDENT)
    else:
        f.write(content)


def safe_write_file(file_path: str,
                   content: Any,
                   is_json: bool = False,
                   options: WriteOptions = None) -> bool:
    """安全写入文件，支持 JSON 序列化和文本写入。

    使用 os.open() + os.fdopen() 确保文件权限在创建时即被设置，
    避免竞态条件下的权限空窗期。

    Args:
        file_path: 目标文件路径
        content:   要写入的内容（字符串或可 JSON 序列化对象）
        is_json:   是否以 JSON 格式写入（便捷参数，等价于 options.is_json）
        options:   高级写入选项（flags/modes/atomic），默认使用安全默认值

    Returns:
        bool: 写入成功返回 True，失败返回 False
    """
    if options is None:
        options = WriteOptions(is_json=is_json)
    elif is_json:
        options.is_json = is_json

    flags = options.flags
    modes = options.modes

    try:
        # 符号链接检查：拒绝写入符号链接目标，防止路径劫持
        if os.path.islink(file_path):
            logger.error("Refusing to write to symlink: %s", file_path)
            return False

        if options.atomic:
            # 原子写入：先写临时文件，再 os.replace() 原子重命名。
            # 避免其他进程在写入过程中读到截断的中间状态。
            tmp_path = file_path + ".tmp"
            with os.fdopen(os.open(tmp_path, flags, modes), 'w', encoding='utf-8') as f:
                _write_content(f, content, options.is_json)
            os.replace(tmp_path, file_path)
        else:
            with os.fdopen(os.open(file_path, flags, modes), 'w', encoding='utf-8') as f:
                _write_content(f, content, options.is_json)
        return True
    except Exception as e:
        logger.error("Failed to write file %s: %s", file_path, e, exc_info=True)
        return False


def get_directory_size(path: str) -> int:
    """计算目录及其子文件的总大小（字节）。

    递归遍历目录下所有文件，累加文件大小（不包括符号链接）。

    Args:
        path: 目录路径

    Returns:
        int: 目录总大小（字节）
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def check_permission_640(file_path):
    """检查文件权限是否为 640。

    Args:
        file_path (str): 要检查权限的文件路径。

    Returns:
        bool: 如果文件权限为 640 则返回 True，否则返回 False。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        PermissionError: 无权访问文件时抛出。
        OSError: 其他操作系统错误。
    """
    try:
        #
        stat_info = os.stat(file_path)

        # 0o777
        file_permissions = stat_info.st_mode & 0o777

        # 6400o640
        if file_permissions == 0o640:
            message = f"File '{file_path}' has correct permissions: 640"
            logger.info(message)
            return True
        else:
            message = f"File '{file_path}' has incorrect permissions. " \
                      f"Current permissions: octal {oct(file_permissions)}, " \
                      f"please change to permission 640!"
            logger.info(message)
            return False

    except FileNotFoundError:
        logger.error("Error: File '%s' does not exist", file_path)
        raise
    except PermissionError:
        logger.error("Error: No permission to access file '%s'", file_path)
        raise
    except OSError as e:
        logger.error("OS error occurred while checking permissions: %s", e)
        raise OSError(f"Failed to check file permissions: {e}") from e


def check_torch_dtype(json_file_path):
    """检查模型 config.json 中 torch_dtype 字段是否为 bfloat16。

    Ascend 310 不支持 bfloat16，若检测到该配置则抛出 ValueError 提示用户修改。

    Args:
        json_file_path (str): 模型 config.json 文件的路径。

    Returns:
        bool: torch_dtype 不为 bfloat16 时返回 True。

    Raises:
        ValueError: torch_dtype 为 bfloat16 时抛出，提示修改为 float16。
        FileNotFoundError: 文件不存在时抛出。
        json.JSONDecodeError: JSON 格式错误时抛出。
        IOError: I/O 读取错误时抛出。
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # torch_dtype
        if data.get('torch_dtype') == 'bfloat16':
            error_msg = "Ascend310 does not support bfloat16. Please modify the config.json " \
            "under the model weight path and change torch_dtype to float16"
            raise ValueError(error_msg)

        return True

    except FileNotFoundError:
        logger.error("The file %s was not found", json_file_path)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON format in file: %s", e)
        raise
    except IOError as e:
        logger.error("An error occurred while reading the file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error checking torch dtype: %s", e)
        raise RuntimeError(f"Failed to check torch dtype: {e}") from e


def load_json_config(file_path: str) -> Dict[str, Any]:
    """加载并解析 JSON 配置文件。

    文件不存在时返回空字典并记录警告；解析失败时返回空字典并记录错误。

    Args:
        file_path (str): JSON 配置文件路径。

    Returns:
        Dict[str, Any]: 解析后的配置字典，失败时返回 {}。
    """
    if not os.path.exists(file_path):
        logger.warning("Config file not found: %s", file_path)
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            config_data = json.load(f)
            logger.info("Successfully loaded config file: %s", file_path)
            return config_data
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON config file: %s", file_path, exc_info=True)
        return {}
    except Exception as _:
        logger.error("Unknown error loading config file: %s", file_path, exc_info=True)
        return {}
