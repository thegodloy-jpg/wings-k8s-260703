# 启动命令标准化校验工具

本目录包含设计文档和第一版可执行工具。工具接受正式可用命令，通过当前仓库的生产链路生成同场景 Wings 命令，并执行结构化语义比较。

## 文件

```text
启动命令标准化校验/
├── startup_command_standard.py       # 命令行工具
├── test_startup_command_standard.py  # 自动化测试
├── examples/                         # 场景输入示例
├── README.md
└── 启动命令标准化对比设计.md
```

## 环境

- Python 3.10 或更高版本；
- 在仓库根目录执行；
- `verify` 使用项目现有 Python 依赖；
- 工具本身只使用 Python 标准库。

## 1. 整理非标准命令

```powershell
python "wings_control/docs/启动命令标准化校验/startup_command_standard.py" draft `
  --input raw_command.txt `
  --output build/command-standard/candidate
```

同时提供场景上下文：

```powershell
python "wings_control/docs/启动命令标准化校验/startup_command_standard.py" draft `
  --input raw_command.txt `
  --scenario scenario.json `
  --output build/command-standard/candidate
```

输出：

- `raw_input.txt`：原始输入；
- `normalized_command.sh`：规范化命令；
- `normalized_command.json`：结构化命令；
- `scenario.candidate.json`：候选场景；
- `normalization_report.json`：修复、警告和缺失信息。

## 2. 比较两份已有命令

```powershell
python "wings_control/docs/启动命令标准化校验/startup_command_standard.py" compare `
  --standard standard_command.sh `
  --actual actual_start_command.sh `
  --scenario scenario.json `
  --output build/command-standard/compare
```

## 3. 生成当前项目命令并比较

以下命令使用现有 Qwen DAY0 dry-run 命令作为正式命令，调用当前项目 `build_launcher_plan()` 生成同场景实际命令：

```powershell
python "wings_control/docs/启动命令标准化校验/startup_command_standard.py" verify `
  --standard "wings_control/docs/DAY0/dry_run/qwen_day0_current/Qwen3.5-27B-910B/exec_command.txt" `
  --scenario "wings_control/docs/启动命令标准化校验/examples/qwen35-27b-910b.scenario.json" `
  --output build/command-standard/qwen35-27b-910b
```

主要输出：

- `actual_start_command.sh`：当前项目生成的完整脚本；
- `merged_params.json`：当前项目合并后的有效配置；
- `normalized_standard.json`：标准命令解析结果；
- `normalized_actual.json`：实际命令解析结果；
- `comparison.json`：机器可读报告；
- `comparison.md`：人工可读报告。

## 退出码

| 退出码 | 含义 |
|---:|---|
| 0 | PASS |
| 1 | FAIL，存在标准差异 |
| 2 | ERROR，输入无效或项目生成失败 |
| 3 | REVIEW，可以解析但需要人工确认 |

## 比较策略

- CLI 顺序、Shell 引号和 JSON 键顺序不参与比较；
- 标准命令中的环境变量和 CLI 参数默认必须在实际命令中存在且值相等；
- 实际命令额外 CLI 默认失败；
- 实际脚本额外环境变量默认只报告；
- 动态字段必须在 `scenario.json` 中声明关系规则；
- 实际命令中的重复 CLI 或重复环境变量失败；
- `forbidden` 可声明禁止出现的 CLI、环境变量和脚本片段；
- 标准无法解析或项目生成异常返回 ERROR，不会伪造对比成功。

## 自测

```powershell
python -m pytest -q "wings_control/docs/启动命令标准化校验/test_startup_command_standard.py"
```
