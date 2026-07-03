# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

"""vLLM adapter helper functions and utilities."""

import ast
import json
import logging
import os
import re
import shlex
from dataclasses import dataclass
from typing import Dict, Any, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


# ── Shell script constants ──────────────────────────────────────────────────

_SH_DETECT_IP = (
    "$(python3 -c \"import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
    "s.connect(('8.8.8.8',80));print(s.getsockname()[0]);s.close()\""
    " 2>/dev/null || hostname -i)"
)
_SH_VLLM_HOST = "export VLLM_HOST_IP=${POD_IP:-${RANK_IP:-" + _SH_DETECT_IP + "}}"
_SH_IF_DETECT = "$(awk '$2==\"00000000\"{print $1;exit}' /proc/net/route 2>/dev/null || echo eth0)"


# ── Distributed script classes ──────────────────────────────────────────────

@dataclass
class DistScriptCtx:
    """分布式脚本生成共用上下文。"""
    engine: str
    cmd: str
    is_ascend: bool
    node_rank: int
    nnodes: int
    head_addr: str
    ray_port: str
    node_ips: str


class DpDeploymentTopology(NamedTuple):
    """dp_deployment 拓扑参数。"""
    dp_size: str
    dp_size_local: str
    dp_start_rank: str


class Glm47DefaultMergeResult:
    """Result of merging a GLM-4.7 W8A8 dict default."""
    def __init__(self, value: Optional[Any], action: str):
        self.value = value
        self.action = action


class Glm47InjectionStats:
    """Accumulated GLM-4.7 W8A8 injection statistics."""
    def __init__(self):
        self.injected: List[str] = []
        self.deep_merged: List[str] = []
        self.skipped: List[str] = []


# ── CLI utilities ──────────────────────────────────────────────────────────

def _strip_cli_flag(cmd: str, flag: str) -> str:
    """从已构建的 vLLM CLI 命令字符串中移除指定的 ``--xxx <value>`` 片段。"""
    tokens = cmd.split()
    out: List[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == flag and i + 1 < len(tokens):
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


def _is_json_like_str(s: str) -> bool:
    """检查字符串是否看起来像 JSON（以 { } 或 [ ] 包围）。"""
    return (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']'))


def _try_parse_json_like_str(stripped: str, arg_name: str) -> Optional[str]:
    """尝试解析 JSON 或 Python literal 格式的字符串，返回规范化的 JSON 字符串。"""
    try:
        parsed = json.loads(stripped)
        return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
    except ValueError:
        pass

    try:
        parsed = ast.literal_eval(stripped)
        logger.warning(
            "[vLLM] %s value is Python-repr str (single-quoted keys); "
            "auto-normalized to JSON. Upstream stringification suspected.",
            arg_name,
        )
        return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
    except (ValueError, SyntaxError):
        logger.warning(
            "[vLLM] %s value looks like dict/list but neither JSON nor "
            "Python literal; passing through as-is: %r",
            arg_name, stripped[:120],
        )
        return None


def _format_cli_arg(arg_name: str, value) -> List[str]:
    """将单个引擎参数值格式化为 CLI 参数片段。"""
    if isinstance(value, bool):
        return [arg_name] if value else []
    if isinstance(value, list):
        str_items = [shlex.quote(str(item)) for item in value]
        return [arg_name] + str_items
    if isinstance(value, dict):
        return [arg_name, shlex.quote(json.dumps(value, ensure_ascii=False, separators=(',', ':')))]
    if isinstance(value, str):
        stripped = value.strip()
        if _is_json_like_str(stripped):
            normalized = _try_parse_json_like_str(stripped, arg_name)
            if normalized is not None:
                return [arg_name, shlex.quote(normalized)]
            return [arg_name, shlex.quote(value)]
    return [arg_name, shlex.quote(str(value))]


# ── Quantization detection ──────────────────────────────────────────────────

_W8A8_QUANT_METHOD_ALIASES = {
    "w8a8", "w8a8_int8", "w8a8int8",
    "smoothquant", "smooth_quant",
    "ascend_w8a8", "ascend-w8a8",
}

_W4A8_QUANT_METHOD_ALIASES = {
    "w4a8", "w4a8_int8", "w4a8int8",
    "ascend_w4a8", "ascend-w4a8",
}


def _is_w8a8_quantize(quantize: Optional[str]) -> bool:
    """判定模型是否为 W8A8 量化变体（容忍命名差异）。"""
    if not quantize:
        return False
    q = str(quantize).strip().lower()
    if not q:
        return False
    if q in _W8A8_QUANT_METHOD_ALIASES:
        return True
    return "w8a8" in q


def _is_w4a8_quantize(quantize: Optional[str]) -> bool:
    """判定模型是否为 W4A8 量化变体（容忍命名差异）。"""
    if not quantize:
        return False
    q = str(quantize).strip().lower()
    if not q:
        return False
    if q in _W4A8_QUANT_METHOD_ALIASES:
        return True
    return "w4a8" in q


# ── Dict merging utilities ──────────────────────────────────────────────────

def _deep_merge_user_priority(user: Any, default: Any) -> Any:
    """递归深合并：user 有则保留 user，user 没有的 sub-key 用 default 填充。"""
    if not isinstance(user, dict) or not isinstance(default, dict):
        return user if user is not None else default
    merged = dict(user)
    for k, v in default.items():
        if k not in merged or merged[k] is None:
            merged[k] = v
        else:
            merged[k] = _deep_merge_user_priority(merged[k], v)
    return merged


def _is_empty_engine_config_value(value: Any) -> bool:
    """Return True when an engine_config value should be treated as unset."""
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, (dict, list)) and len(value) == 0)
    )


def _parse_dict_like_config(value: Any) -> Optional[Dict[str, Any]]:
    """Parse dict-like string config values without changing invalid user input."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, dict) else None


# ── Integer parsing ────────────────────────────────────────────────────────

def _safe_int(value: Any) -> Optional[int]:
    """Safe integer conversion."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ── Shell script generation ────────────────────────────────────────────────

_TRITON_PATCH_LINES = (
    "# Patch triton driver.py: Ascend NPU has no Triton backend, return dummy driver",
    "python3 << 'TRITON_PATCH_EOF'",
    "try:",
    "    import triton.runtime, os",
    "    drv_path = os.path.join(os.path.dirname(triton.runtime.__file__), 'driver.py')",
    "    with open(drv_path) as f:",
    "        src = f.read()",
    "    if 'raise RuntimeError' in src and 'PATCHED_NPU' not in src:",
    "        patch = '''",
    "        # PATCHED_NPU: Ascend NPU has no Triton backend, provide dummy driver",
    "        class _NpuDummyDrv:",
    "            def get_current_target(self):",
    "                import types; return types.SimpleNamespace(backend='npu', arch='Ascend910B', warp_size=0)",
    "            def get_current_device(self): return 0",
    "            def get_device_capability(self, *a): return (0, 0)",
    "            def get_device_properties(self, device=0):",
    "                try:",
    "                    import torch_npu; n = torch_npu.npu.get_device_name(device); "
    "c = 20 if '910B' in str(n) else 30",
    "                except Exception: c = 20",
    "                return {'num_aicore': c, 'num_vectorcore': c}",
    "            def __getattr__(self, name): return _NpuDummyDrv()",
    "            def __call__(self, *a, **k): return self",
    "            def __repr__(self): return '<NpuDummy>'",
    "            def __int__(self): return 0",
    "            def __bool__(self): return False",
    "        return _NpuDummyDrv()'''",
    "        src = src.replace(",
    '            \'raise RuntimeError(f"{len(active_drivers)} active drivers ({active_drivers}). '
    'There should only be one.")\',',
    "            patch.strip()",
    "        )",
    "        with open(drv_path, 'w') as f:",
    "            f.write(src)",
    "        print('[triton-patch] Patched', drv_path, 'for Ascend NPU')",
    "    else:",
    "        print('[triton-patch] Already patched or not needed')",
    "except Exception as e:",
    "    print(f'[triton-patch] Skip: {e}')",
    "TRITON_PATCH_EOF",
)


def build_triton_patch_preamble(engine: str, _need_triton_patch) -> str:
    """返回 Triton NPU 补丁的 shell 脚本片段。"""
    return "" if not _need_triton_patch(engine) else "\n".join(_TRITON_PATCH_LINES) + "\n"


def build_modelslim_quarot_patch_preamble(engine: str) -> str:
    """为 QuaRot 等非 modelslim 量化格式注入 modelslim_config.py 兼容性补丁。"""
    if engine != "vllm_ascend":
        return ""
    return (
        "# --- wings: modelslim_config.py QuaRot compatibility patch ---\n"
        "python3 << 'MODELSLIM_PATCH_EOF'\n"
        "try:\n"
        "    import importlib.util, pathlib\n"
        "    spec = importlib.util.find_spec('vllm_ascend.quantization.modelslim_config')\n"
        "    if spec and spec.origin:\n"
        "        p = pathlib.Path(spec.origin)\n"
        "        txt = p.read_text()\n"
        "        old = 'self.quant_description[shard_prefix + ' + '\"' + '.weight' + '\"' + ']'\n"
        "        new = 'self.quant_description.get(shard_prefix + ' + '\"' + '.weight' + '\"' + ')'\n"
        "        if old in txt:\n"
        "            p.write_text(txt.replace(old, new))\n"
        "            print('[modelslim-patch] Patched modelslim_config.py: dict[] -> dict.get() "
        "for QuaRot compatibility')\n"
        "        else:\n"
        "            print('[modelslim-patch] Already patched or pattern not found')\n"
        "    else:\n"
        "        print('[modelslim-patch] modelslim_config module not found, skipping')\n"
        "except Exception as e:\n"
        "    print(f'[modelslim-patch] Skip: {e}')\n"
        "MODELSLIM_PATCH_EOF\n"
        "# --- end modelslim patch ---\n"
    )


def _transform_dp_cmd(cmd: str) -> str:
    """把 OpenAI api_server 命令转换为 dp_deployment 需要的 ``vllm serve`` 命令。"""
    _model_match = re.search(r"--model\s+('(?:[^']*)'|\S+)", cmd)
    if not _model_match:
        return cmd
    dp_cmd = re.sub(r"\s*--model\s+(?:'[^']*'|\S+)", "", cmd)
    return re.sub(r"^python3\s+-m\s+vllm\.entrypoints\.openai\.api_server",
                  f"vllm serve {_model_match.group(1)}", dp_cmd)
