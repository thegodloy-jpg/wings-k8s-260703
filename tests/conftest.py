"""pytest 收集入口的路径稳定器。"""

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

# 多个历史测试会把 ``wings_control/`` 目录插到 sys.path[0]，用于兼容
# ``from engines import ...`` 这类顶层导入。全量收集时这会让
# ``import wings_control`` 误命中 ``wings_control/wings_control.py``。
# 这里先按仓库根路径加载包，后续测试即使调整 sys.path 也会复用 sys.modules
# 中的包对象，保证 ``wings_control.log_analyzer`` 等包导入稳定。
importlib.import_module("wings_control")
