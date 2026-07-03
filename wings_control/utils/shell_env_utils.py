"""生成 start_command.sh 时的 shell 环境变量（export 语句）文本处理工具。

与 ``env_utils.py`` 区分：
    - ``env_utils.py``      : 运行时**读取** os.environ（``get_*_env()``）。
    - 本模块 shell_env_utils : 构建期处理拼接出来的 shell ``export`` 命令列表
                               （去重等），不读取真实环境。

当前能力:
    - dedupe_env_exports(): 多个 builder 重复导出同名变量时收口去重，保证每个变量
      最终只剩一条 export 生效（等价最终值），且不破坏累加型与块内导出。
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
from __future__ import annotations

import re
from typing import Dict, List

# 顶格（无缩进）的简单 `export VAR=value` 行；缩进的（如 if/else 块内）不匹配，天然跳过。
_EXPORT_LINE_RE = re.compile(r"^export ([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def classify_env_export(var: str, rhs: str) -> str:
    """判定 ``export VAR=RHS`` 的类型：'plain' | 'soft_default' | 'accumulation'。

    - plain：RHS 不引用自身（纯赋值，覆盖式）。
    - soft_default：整个 RHS 恰好是对自身的一次展开，如 ``${VAR:-x}`` / ``${VAR}``
      （仅在未设置时取默认，可被外部环境覆盖）。
    - accumulation：RHS 在更大的串里引用了自身，如 ``"a:${VAR:-}"``（追加/累加，绝不能去重）。
    """
    if not re.search(r"\$\{?" + re.escape(var) + r"(?![A-Za-z0-9_])", rhs):
        return "plain"
    stripped = rhs.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        stripped = stripped[1:-1]
    if re.fullmatch(r"\$\{" + re.escape(var) + r"(:?[-=+?][^}]*)?\}", stripped):
        return "soft_default"
    return "accumulation"


def dedupe_env_exports(cmds: List[str]) -> List[str]:
    """对顶层 ``export VAR=value`` 去重，保证每个变量最终只有一条 export 生效。

    规则（保持与去重前完全等价的最终值）：
      * 仅处理顶格（无缩进）的简单 ``export VAR=...`` 行，绝不触碰 if/else/for 块内的缩进导出；
      * 累加型（如 ``LD_LIBRARY_PATH="...:${LD_LIBRARY_PATH:-}"``）整条跳过，不参与去重；
      * 同名变量若存在「纯赋值」occurrence：保留**最后一个纯赋值**（bash 中最后赋值生效），
        丢弃更早的纯赋值与其后的 ``${VAR:-默认}`` 软默认（后者此时已是空操作）；
      * 同名变量若**只有软默认**：保留第一个（保留可被外部环境覆盖的语义）；
      * 单次出现的变量、非 export 行一律原样保留。
    """
    accumulation_vars = set()
    occ: Dict[str, List[tuple]] = {}
    for idx, line in enumerate(cmds):
        m = _EXPORT_LINE_RE.match(line)
        if not m:
            continue
        var, rhs = m.group(1), m.group(2)
        kind = classify_env_export(var, rhs)
        if kind == "accumulation":
            accumulation_vars.add(var)
            continue
        occ.setdefault(var, []).append((idx, kind))

    drop_idx = set()
    for var, lst in occ.items():
        if var in accumulation_vars or len(lst) <= 1:
            continue
        plain_idxs = [i for i, k in lst if k == "plain"]
        keep = plain_idxs[-1] if plain_idxs else lst[0][0]
        drop_idx.update(i for i, _k in lst if i != keep)

    if not drop_idx:
        return cmds
    return [line for idx, line in enumerate(cmds) if idx not in drop_idx]
