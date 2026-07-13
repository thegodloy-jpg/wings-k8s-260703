#!/usr/bin/env python3
"""启动命令标准化、语义比较和 Wings 生产链路验证工具。

该工具不实现命令生成逻辑。``verify`` 子命令使用当前仓库的
``core.wings_entry.build_launcher_plan`` 生成实际 ``start_command.sh``，
然后把正式场景命令与实际命令解析为结构化数据后比较。
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import json
import os
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[2]
WINGS_CONTROL_ROOT = REPO_ROOT / "wings_control"

ENGINE_MODULE = "vllm.entrypoints.openai.api_server"
BOOL_LITERALS = {"true": True, "false": False, "none": None, "null": None}
JSON_VALUE_FLAGS = {
    "additional-config",
    "compilation-config",
    "default-chat-template-kwargs",
    "hf-overrides",
    "kv-transfer-config",
    "sparse-config",
    "speculative-config",
}
LOG_PREFIX_RE = re.compile(r"^(?:\[[^\]]+\]\s*)+")
EXPORT_RE = re.compile(r"^(?:export\s+)([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
NUMBER_INT_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
NUMBER_FLOAT_RE = re.compile(r"^-?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


MANAGED_ENV_KEYS = {
    "ASCEND_PLATFORM",
    "AVAILABLE_POD_MEM_SIZE",
    "DEVICE_COUNT",
    "DISTRIBUTED",
    "DISTRIBUTED_EXECUTOR_BACKEND",
    "ENABLE_AUTO_THINK_CHOICE",
    "ENABLE_AUTO_TOOL_CHOICE",
    "ENABLE_CHUNKED_PREFILL",
    "ENABLE_EXPERT_PARALLEL",
    "ENABLE_KV_DISK_OFFLOAD",
    "ENABLE_KV_MEM_OFFLOAD",
    "ENABLE_KV_OFFLOAD",
    "ENABLE_PREFIX_CACHING",
    "ENABLE_REASON_PROXY",
    "ENABLE_SPARSE",
    "ENABLE_SPECULATIVE_DECODE",
    "ENGINE",
    "ENGINE_IMAGE_FLAVOR",
    "ENGINE_PORT",
    "ENGINE_VERSION",
    "GLOO_SOCKET_IFNAME",
    "HEAD_NODE_ADDR",
    "HOST",
    "INPUT_LENGTH",
    "KV_MEM_OFFLOAD_SIZE",
    "LMCACHE_OFFLOAD",
    "MASTER_IP",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_NUM_SEQS",
    "MODEL_NAME",
    "MODEL_PATH",
    "NETWORK_INTERFACE",
    "NNODES",
    "NODE_IPS",
    "NODE_RANK",
    "NODES",
    "OUTPUT_LENGTH",
    "POD_IP",
    "PORT",
    "PROXY_PORT",
    "RANK_IP",
    "SAVE_PATH",
    "SD_ENABLE",
    "SPARSE_ENABLE",
    "SPECULATIVE_DECODE_MODEL_PATH",
    "TRUST_REMOTE_CODE",
    "WINGS_ASCEND_PLATFORM",
    "WINGS_HARDWARE_FILE",
}

LAUNCH_VALUE_FLAGS = {
    "host",
    "port",
    "input_length",
    "output_length",
    "model_type",
    "dtype",
    "kv_cache_dtype",
    "quantization",
    "quantization_param_path",
    "gpu_memory_utilization",
    "block_size",
    "max_num_seqs",
    "seed",
    "max_num_batched_tokens",
    "speculative_decode_model_path",
    "nnodes",
    "node_rank",
    "head_node_addr",
    "distributed_executor_backend",
    "node_ips",
    "nodes",
    "master_ip",
    "ray_head_ip",
}

LAUNCH_BOOL_FLAGS = {
    "trust_remote_code",
    "enable_chunked_prefill",
    "enable_expert_parallel",
    "enable_prefix_caching",
    "enable_speculative_decode",
    "enable_rag_acc",
    "enable_auto_tool_choice",
    "enable_auto_think_choice",
    "enable_sparse",
    "enable_smartqos",
    "enable_otlp_traces",
    "distributed",
}


@dataclass
class Notice:
    code: str
    message: str
    line: int | None = None
    original: str | None = None
    normalized: str | None = None


@dataclass
class NormalizedCommand:
    raw: str
    cleaned_script: str
    entrypoint: str | None
    entrypoint_kind: str | None
    environment: dict[str, Any]
    flags: dict[str, Any]
    positional_args: list[Any]
    pre_commands: list[str]
    duplicate_environment: dict[str, list[Any]] = field(default_factory=dict)
    duplicate_flags: dict[str, list[Any]] = field(default_factory=dict)
    repairs: list[Notice] = field(default_factory=list)
    warnings: list[Notice] = field(default_factory=list)
    errors: list[Notice] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.entrypoint is not None and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Difference:
    scope: str
    path: str
    rule: str
    expected: Any
    actual: Any
    result: str
    severity: str
    message: str = ""


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, _json_dump(value))


def _read_text(path_text: str) -> str:
    if path_text == "-":
        return sys.stdin.read()
    return Path(path_text).read_text(encoding="utf-8-sig")


def _strip_log_prefix(line: str, line_no: int, repairs: list[Notice]) -> str:
    stripped = line.strip()
    if not stripped:
        return ""

    # 生产脚本会在真实命令前插入 echo '[wings-cmd] >>> ...'。这只是日志，
    # 不能再次被当作可执行命令提取，否则带有 Shell 转义的引号会造成误解析。
    if stripped.startswith("echo ") and ("[wings-cmd]" in stripped or "[wings-env]" in stripped):
        return ""

    if "[wings-cmd] >>>" in stripped:
        value = stripped.split("[wings-cmd] >>>", 1)[1].strip()
        repairs.append(Notice("remove_wings_log_prefix", "移除 [wings-cmd] 日志前缀", line_no, stripped, value))
        return value
    if "[wings-env]" in stripped:
        value = stripped.split("[wings-env]", 1)[1].strip()
        repairs.append(Notice("remove_wings_log_prefix", "移除 [wings-env] 日志前缀", line_no, stripped, value))
        return value

    candidate = LOG_PREFIX_RE.sub("", stripped).strip()
    if candidate != stripped and (
        candidate.startswith("export ")
        or "vllm serve" in candidate
        or ENGINE_MODULE in candidate
    ):
        repairs.append(Notice("remove_log_prefix", "移除通用日志前缀", line_no, stripped, candidate))
        return candidate
    return stripped


def _logical_lines(text: str, repairs: list[Notice]) -> list[tuple[int, str]]:
    cleaned: list[tuple[int, str]] = []
    for line_no, raw in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        value = _strip_log_prefix(raw, line_no, repairs)
        if value:
            cleaned.append((line_no, value))

    result: list[tuple[int, str]] = []
    buffer = ""
    start_line = 0
    for line_no, line in cleaned:
        if buffer:
            buffer += line
        else:
            buffer = line
            start_line = line_no
        if buffer.rstrip().endswith("\\"):
            buffer = buffer.rstrip()[:-1].rstrip() + " "
            repairs.append(Notice("join_shell_continuation", "合并 Shell 续行", line_no))
            continue
        result.append((start_line, buffer.strip()))
        buffer = ""

    if buffer:
        result.append((start_line, buffer.strip()))

    # 日志复制时长命令可能被自动折行，安全地合并以 -- 开头的后续行。
    joined: list[tuple[int, str]] = []
    for line_no, line in result:
        if joined and line.startswith("--") and _is_engine_line(joined[-1][1]):
            previous_no, previous = joined[-1]
            joined[-1] = (previous_no, previous + " " + line)
            repairs.append(Notice("join_wrapped_cli_line", "合并以 -- 开头的日志折行", line_no, line, joined[-1][1]))
        elif joined and (joined[-1][1].endswith(" -m") or joined[-1][1].endswith(" -m ")) and line.startswith(ENGINE_MODULE):
            previous_no, previous = joined[-1]
            joined[-1] = (previous_no, previous + " " + line)
            repairs.append(Notice("join_wrapped_module", "合并被拆开的 Python 模块入口", line_no))
        else:
            joined.append((line_no, line))
    return joined


def _is_engine_line(line: str) -> bool:
    value = line.strip()
    if value.startswith("echo ") and "[wings-cmd]" in value:
        return False
    return bool(re.search(r"(?:^|\s)vllm\s+serve(?:\s|$)", value)) or ENGINE_MODULE in value


def _unquote_env(value: str) -> Any:
    value = value.strip()
    try:
        parts = shlex.split(value, posix=True)
    except ValueError:
        return value
    return parts[0] if len(parts) == 1 else value


def _normalize_scalar(value: str, repairs: list[Notice], field_name: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in BOOL_LITERALS:
        return BOOL_LITERALS[lowered]
    if NUMBER_INT_RE.fullmatch(stripped):
        try:
            return int(stripped)
        except ValueError:
            pass
    if NUMBER_FLOAT_RE.fullmatch(stripped):
        try:
            return float(stripped)
        except ValueError:
            pass
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return stripped
            if isinstance(parsed, (dict, list)):
                repairs.append(Notice("normalize_python_literal", f"将 {field_name} 的 Python 字面量转成 JSON 语义"))
                return parsed
    return stripped


def _parse_cli(command: str, repairs: list[Notice], errors: list[Notice]) -> tuple[str | None, str | None, dict[str, Any], list[Any], dict[str, list[Any]]]:
    value = command.strip()
    if value.endswith("&"):
        value = value[:-1].rstrip()
        repairs.append(Notice("remove_background_marker", "移除后台执行符号 &"))
    try:
        tokens = shlex.split(value, posix=True)
    except ValueError as exc:
        errors.append(Notice("shell_parse_error", f"无法解析引擎命令: {exc}", original=command))
        return None, None, {}, [], {}

    while tokens and tokens[0] in {"exec", "nohup", "env"}:
        repairs.append(Notice("remove_command_wrapper", f"规范化命令包装器 {tokens[0]}"))
        tokens.pop(0)

    entrypoint = None
    entrypoint_kind = None
    remaining: list[str] = []
    positional: list[Any] = []

    if len(tokens) >= 2 and tokens[0] == "vllm" and tokens[1] == "serve":
        entrypoint = ENGINE_MODULE
        entrypoint_kind = "vllm_serve"
        remaining = tokens[2:]
        if remaining and not remaining[0].startswith("--"):
            positional_model = remaining.pop(0)
            positional.append(positional_model)
    elif len(tokens) >= 3 and tokens[0] in {"python", "python3"} and tokens[1] == "-m" and tokens[2] == ENGINE_MODULE:
        entrypoint = ENGINE_MODULE
        entrypoint_kind = "python_module"
        remaining = tokens[3:]
    else:
        errors.append(Notice("unsupported_entrypoint", "只支持 vllm serve 或 vLLM OpenAI Python 模块入口", original=command))
        return None, None, {}, [], {}

    flags: dict[str, Any] = {}
    duplicates: dict[str, list[Any]] = {}
    index = 0
    while index < len(remaining):
        token = remaining[index]
        if not token.startswith("--"):
            positional.append(_normalize_scalar(token, repairs, "positional"))
            index += 1
            continue

        raw_flag = token[2:]
        if "=" in raw_flag:
            raw_key, raw_value = raw_flag.split("=", 1)
            index += 1
            parsed_value: Any = _normalize_scalar(raw_value, repairs, raw_key)
        else:
            raw_key = raw_flag
            if index + 1 < len(remaining) and not remaining[index + 1].startswith("--"):
                parsed_value = _normalize_scalar(remaining[index + 1], repairs, raw_key)
                index += 2
            else:
                parsed_value = True
                index += 1

        key = raw_key.strip().lower().replace("_", "-")
        if key in flags:
            duplicates.setdefault(key, [flags[key]]).append(parsed_value)
        flags[key] = parsed_value

    if positional and "model" not in flags and entrypoint_kind == "vllm_serve":
        flags["model"] = positional.pop(0)
    return entrypoint, entrypoint_kind, flags, positional, duplicates


def normalize_command(text: str) -> NormalizedCommand:
    repairs: list[Notice] = []
    warnings: list[Notice] = []
    errors: list[Notice] = []
    lines = _logical_lines(text, repairs)

    environment: dict[str, Any] = {}
    duplicate_env: dict[str, list[Any]] = {}
    pre_commands: list[str] = []
    engine_lines: list[tuple[int, str]] = []

    for line_no, line in lines:
        if line.startswith("#"):
            continue
        match = EXPORT_RE.match(line)
        if match:
            key, raw_value = match.groups()
            value = _normalize_scalar(str(_unquote_env(raw_value)), repairs, f"env.{key}")
            if key in environment:
                duplicate_env.setdefault(key, [environment[key]]).append(value)
            environment[key] = value
            continue
        if _is_engine_line(line):
            engine_lines.append((line_no, line))
            continue
        if not line.startswith("echo "):
            pre_commands.append(line)

    unique_engine_lines: list[tuple[int, str]] = []
    for item in engine_lines:
        if item[1] not in [existing[1] for existing in unique_engine_lines]:
            unique_engine_lines.append(item)
    if not unique_engine_lines:
        errors.append(Notice("engine_command_missing", "未找到 vLLM 核心启动命令"))
        entrypoint = entrypoint_kind = None
        flags: dict[str, Any] = {}
        positional: list[Any] = []
        duplicate_flags: dict[str, list[Any]] = {}
    else:
        if len(unique_engine_lines) > 1:
            warnings.append(Notice(
                "multiple_engine_commands",
                "脚本包含主命令和回退/重试命令；当前按出现顺序选择第一条主命令",
                original="\n".join(line for _, line in unique_engine_lines),
            ))
        selected = unique_engine_lines[0][1]
        entrypoint, entrypoint_kind, flags, positional, duplicate_flags = _parse_cli(selected, repairs, errors)
        for key in sorted(JSON_VALUE_FLAGS & set(flags)):
            if isinstance(flags[key], str):
                errors.append(Notice(
                    "json_flag_parse_error",
                    f"--{key} 应为合法 JSON，但无法解析",
                    original=flags[key],
                ))

    for key, values in duplicate_env.items():
        warnings.append(Notice("duplicate_environment", f"环境变量 {key} 被重复赋值: {values}"))
    for key, values in duplicate_flags.items():
        warnings.append(Notice("duplicate_flag", f"CLI 参数 --{key} 重复出现: {values}"))

    cleaned_script = "\n".join(line for _, line in lines)
    return NormalizedCommand(
        raw=text,
        cleaned_script=cleaned_script,
        entrypoint=entrypoint,
        entrypoint_kind=entrypoint_kind,
        environment=environment,
        flags=flags,
        positional_args=positional,
        pre_commands=pre_commands,
        duplicate_environment=duplicate_env,
        duplicate_flags=duplicate_flags,
        repairs=repairs,
        warnings=warnings,
        errors=errors,
    )


def _render_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return shlex.quote(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if value is None:
        return "null"
    return shlex.quote(str(value))


def render_normalized_shell(command: NormalizedCommand) -> str:
    lines: list[str] = []
    for key in sorted(command.environment):
        lines.append(f"export {key}={_render_value(command.environment[key])}")
    if lines:
        lines.append("")
    if command.entrypoint:
        parts = ["python3", "-m", command.entrypoint]
        for key in sorted(command.flags):
            value = command.flags[key]
            parts.append(f"--{key}")
            if value is not True:
                parts.append(_render_value(value))
        parts.extend(_render_value(item) for item in command.positional_args)
        lines.append(" \\\n  ".join(parts))
    return "\n".join(lines).rstrip() + "\n"


def render_engine_command(command: NormalizedCommand) -> str:
    """渲染不含环境变量的规范化核心引擎命令。"""
    engine_only = copy.copy(command)
    engine_only.environment = {}
    return render_normalized_shell(engine_only)


def _nested_get(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _dynamic_rule(config: dict[str, Any], scope: str, path: str) -> dict[str, Any] | None:
    value = config.get("dynamic_fields", {}).get(f"{scope}.{path}")
    if value is None:
        return None
    return {"rule": value} if isinstance(value, str) else dict(value)


def _append_diff(differences: list[Difference], scope: str, path: str, rule: str, expected: Any, actual: Any, result: str, severity: str, message: str = "") -> None:
    differences.append(Difference(scope, path, rule, expected, actual, result, severity, message))


def _compare_dynamic(rule_config: dict[str, Any], scope: str, path: str, expected: Any, actual: Any, scenario: dict[str, Any], runtime: dict[str, Any], differences: list[Difference]) -> None:
    rule = rule_config.get("rule", "any_nonempty")
    ok = False
    target: Any = expected
    if rule in {"any_nonempty", "ignore_value"}:
        ok = actual not in (None, "")
        target = "<nonempty>"
    elif rule == "model_path":
        target = runtime.get("model_path")
        ok = bool(expected) and actual == target
    elif rule == "equals_launch_input":
        input_path = rule_config.get("input", f"launch.{path}")
        target = _nested_get(scenario, input_path)
        ok = actual == target
    elif rule == "runtime_host_policy":
        allowed = rule_config.get("allowed") or [
            _nested_get(scenario, "launch.host"),
            _nested_get(scenario, "environment.POD_IP"),
            "0.0.0.0",
            "127.0.0.1",
        ]
        allowed = [value for value in allowed if value not in (None, "")]
        target = allowed
        ok = actual in allowed
    else:
        _append_diff(differences, scope, path, rule, expected, actual, "REVIEW", "warning", "未知动态比较规则")
        return
    _append_diff(differences, scope, path, rule, target, actual, "PASS" if ok else "FAIL", "info" if ok else "error")


def _compare_value(scope: str, path: str, expected: Any, actual: Any, differences: list[Difference]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child_path = f"{path}.{key}" if path else key
            if key not in expected:
                _append_diff(differences, scope, child_path, "exact", None, actual[key], "FAIL", "error", "实际值包含标准中不存在的字段")
            elif key not in actual:
                _append_diff(differences, scope, child_path, "exact", expected[key], None, "FAIL", "error", "实际值缺少字段")
            else:
                _compare_value(scope, child_path, expected[key], actual[key], differences)
        return
    ok = expected == actual
    _append_diff(differences, scope, path, "exact", expected, actual, "PASS" if ok else "FAIL", "info" if ok else "error")


def compare_commands(standard: NormalizedCommand, actual: NormalizedCommand, scenario: dict[str, Any] | None = None, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = scenario or {}
    runtime = runtime or {}
    config = scenario.get("comparison", {})
    differences: list[Difference] = []

    if not standard.valid or not actual.valid:
        return {
            "schema_version": 1,
            "scenario_id": scenario.get("scenario_id", "adhoc"),
            "result": "ERROR",
            "summary": {"passed": 0, "failed": 0, "review": 0, "errors": 1},
            "differences": [],
            "parse_errors": {
                "standard": [asdict(item) for item in standard.errors],
                "actual": [asdict(item) for item in actual.errors],
            },
        }

    _compare_value("entrypoint", "module", standard.entrypoint, actual.entrypoint, differences)

    ignore = config.get("ignore", {}) if isinstance(config.get("ignore", {}), dict) else {}
    ignored_cli = set(ignore.get("cli", []))
    ignored_env = set(ignore.get("environment", []))

    for key, expected in sorted(standard.environment.items()):
        if key in ignored_env:
            continue
        actual_value = actual.environment.get(key)
        dynamic = _dynamic_rule(config, "environment", key)
        if dynamic:
            _compare_dynamic(dynamic, "environment", key, expected, actual_value, scenario, runtime, differences)
        elif key not in actual.environment:
            _append_diff(differences, "environment", key, "exact", expected, None, "FAIL", "error", "实际脚本缺少标准环境变量")
        else:
            _compare_value("environment", key, expected, actual_value, differences)

    extra_env_policy = config.get("extra_environment_variable", "report")
    for key in sorted(set(actual.environment) - set(standard.environment) - ignored_env):
        if extra_env_policy == "ignore":
            continue
        result = "FAIL" if extra_env_policy == "fail" else "INFO"
        severity = "error" if result == "FAIL" else "info"
        _append_diff(differences, "environment", key, "extra", None, actual.environment[key], result, severity, "实际脚本额外环境变量")

    for key, expected in sorted(standard.flags.items()):
        if key in ignored_cli:
            continue
        actual_value = actual.flags.get(key)
        dynamic = _dynamic_rule(config, "cli", key)
        if dynamic:
            _compare_dynamic(dynamic, "cli", key, expected, actual_value, scenario, runtime, differences)
        elif key not in actual.flags:
            _append_diff(differences, "cli", key, "exact", expected, None, "FAIL", "error", "实际命令缺少标准参数")
        else:
            _compare_value("cli", key, expected, actual_value, differences)

    extra_cli_policy = config.get("extra_cli_argument", "fail")
    for key in sorted(set(actual.flags) - set(standard.flags) - ignored_cli):
        if extra_cli_policy == "ignore":
            continue
        result = "FAIL" if extra_cli_policy == "fail" else "REVIEW"
        severity = "error" if result == "FAIL" else "warning"
        _append_diff(differences, "cli", key, "extra", None, actual.flags[key], result, severity, "实际命令额外 CLI 参数")

    forbidden = config.get("forbidden", {})
    for key in forbidden.get("cli", []):
        normalized_key = key.lstrip("-").replace("_", "-")
        if normalized_key in actual.flags:
            _append_diff(differences, "cli", normalized_key, "absent", None, actual.flags[normalized_key], "FAIL", "error", "命中禁止 CLI 参数")
    for key in forbidden.get("environment", []):
        if key in actual.environment:
            _append_diff(differences, "environment", key, "absent", None, actual.environment[key], "FAIL", "error", "命中禁止环境变量")

    for key, values in actual.duplicate_flags.items():
        _append_diff(differences, "cli", key, "unique", "single value", values, "FAIL", "error", "实际命令包含重复 CLI 参数")
    for key, values in actual.duplicate_environment.items():
        distinct = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values}
        if len(distinct) == 1:
            _append_diff(
                differences,
                "environment",
                key,
                "consistent_across_branches",
                values[0],
                values,
                "INFO",
                "info",
                "主命令与回退分支重复导出相同环境变量",
            )
        else:
            _append_diff(differences, "environment", key, "unique", "single value", values, "FAIL", "error", "实际脚本包含取值冲突的重复环境变量")
    for key, values in standard.duplicate_flags.items():
        _append_diff(differences, "standard", f"cli.{key}", "valid_standard", "single value", values, "REVIEW", "warning", "标准命令自身包含重复参数")

    actual_script = actual.raw
    for fragment in config.get("required_script_fragments", []):
        ok = fragment in actual_script
        _append_diff(differences, "script", fragment, "contains", fragment, fragment if ok else None, "PASS" if ok else "FAIL", "info" if ok else "error")
    for fragment in config.get("forbidden_script_fragments", []):
        present = fragment in actual_script
        _append_diff(differences, "script", fragment, "absent", None, fragment if present else None, "FAIL" if present else "PASS", "error" if present else "info")

    failed = sum(item.result == "FAIL" for item in differences)
    review = sum(item.result == "REVIEW" for item in differences)
    passed = sum(item.result == "PASS" for item in differences)
    info = sum(item.result == "INFO" for item in differences)
    result = "FAIL" if failed else "REVIEW" if review else "PASS"
    return {
        "schema_version": 1,
        "scenario_id": scenario.get("scenario_id", "adhoc"),
        "result": result,
        "summary": {"passed": passed, "failed": failed, "review": review, "info": info, "errors": 0},
        "differences": [asdict(item) for item in differences],
        "parse_warnings": {
            "standard": [asdict(item) for item in standard.warnings],
            "actual": [asdict(item) for item in actual.warnings],
        },
    }


def comparison_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        f"# 启动命令对比报告：{report.get('scenario_id', 'adhoc')}",
        "",
        f"- 结果：**{report.get('result', 'ERROR')}**",
        f"- 通过：{summary.get('passed', 0)}",
        f"- 失败：{summary.get('failed', 0)}",
        f"- 待确认：{summary.get('review', 0)}",
        f"- 信息项：{summary.get('info', 0)}",
        "",
        "## 差异",
        "",
        "| 结果 | 范围 | 字段 | 规则 | 标准值 | 实际值 | 说明 |",
        "|---|---|---|---|---|---|---|",
    ]
    emitted = 0
    for item in report.get("differences", []):
        if item["result"] in {"PASS", "INFO"}:
            continue
        expected = json.dumps(item.get("expected"), ensure_ascii=False, sort_keys=True).replace("|", "\\|")
        actual = json.dumps(item.get("actual"), ensure_ascii=False, sort_keys=True).replace("|", "\\|")
        message = str(item.get("message", "")).replace("|", "\\|")
        lines.append(f"| {item['result']} | {item['scope']} | `{item['path']}` | {item['rule']} | `{expected}` | `{actual}` | {message} |")
        emitted += 1
    if emitted == 0:
        lines.append("| PASS | - | - | - | - | - | 无失败或待确认差异 |")
    return "\n".join(lines) + "\n"


def _candidate_scenario(command: NormalizedCommand, supplied: dict[str, Any] | None = None) -> dict[str, Any]:
    supplied = copy.deepcopy(supplied or {})
    supplied.setdefault("schema_version", 1)
    supplied.setdefault("scenario_id", "candidate")
    supplied.setdefault("status", "candidate")
    supplied.setdefault("comparison", {})
    supplied["comparison"].setdefault("extra_cli_argument", "fail")
    supplied["comparison"].setdefault("extra_environment_variable", "report")
    missing = []
    for path in ("model.name", "model.architecture", "hardware.family", "hardware.device_count", "engine.name", "engine.version"):
        if _nested_get(supplied, path) in (None, ""):
            missing.append(path)
    if not command.valid:
        status = "INVALID_INPUT"
    elif missing or command.warnings:
        status = "REVIEW"
    else:
        status = "READY"
    supplied["candidate_analysis"] = {
        "entrypoint": command.entrypoint,
        "inferred_model_path": command.flags.get("model"),
        "inferred_port": command.flags.get("port"),
        "missing_required_context": missing,
        "status": status,
    }
    return supplied


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return value


def _validate_scenario(scenario: dict[str, Any]) -> list[str]:
    required = (
        "scenario_id",
        "model.name",
        "model.architecture",
        "hardware.device",
        "hardware.family",
        "hardware.device_count",
        "engine.name",
        "engine.version",
    )
    return [path for path in required if _nested_get(scenario, path) in (None, "")]


def _model_config(scenario: dict[str, Any]) -> dict[str, Any]:
    model = scenario["model"]
    config = copy.deepcopy(model.get("config", {}))
    config.setdefault("architectures", [model["architecture"]])
    config.setdefault("model_type", model.get("model_type", "auto"))
    config.setdefault("torch_dtype", model.get("torch_dtype", "bfloat16"))
    config.setdefault("num_hidden_layers", model.get("num_hidden_layers", 64))
    quantization = model.get("quantization")
    if quantization and "quantization_config" not in config:
        config["quantization_config"] = {"quant_method": quantization}
    return config


@contextlib.contextmanager
def _patched_environment(values: dict[str, str]) -> Iterator[None]:
    keys = MANAGED_ENV_KEYS | set(values)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update({key: str(value) for key, value in values.items()})
        yield
    finally:
        for key in keys:
            os.environ.pop(key, None)
        for key, value in previous.items():
            if value is not None:
                os.environ[key] = value


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _build_launch_argv(scenario: dict[str, Any], model_path: Path, output_dir: Path) -> list[str]:
    launch = scenario.get("launch", {})
    argv = [
        "--model-name", str(scenario["model"]["name"]),
        "--model-path", model_path.as_posix(),
        "--engine", str(scenario["engine"]["name"]),
        "--device-count", str(scenario["hardware"]["device_count"]),
        "--port", str(launch.get("port", 18000)),
        "--save-path", output_dir.as_posix(),
    ]
    for key in sorted(LAUNCH_VALUE_FLAGS):
        if key not in launch or key in {"port"}:
            continue
        value = launch[key]
        if value in (None, ""):
            continue
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    for key in sorted(LAUNCH_BOOL_FLAGS):
        if bool(launch.get(key, False)):
            argv.append(f"--{key.replace('_', '-')}")
    return argv


def _generation_environment(scenario: dict[str, Any], model_path: Path, hardware_file: Path, output_dir: Path) -> dict[str, str]:
    launch = scenario.get("launch", {})
    values = {
        "SHARED_VOLUME_PATH": (output_dir / "shared").as_posix(),
        "ENABLE_REASON_PROXY": "false",
        "POD_IP": str(scenario.get("environment", {}).get("POD_IP", "127.0.0.1")),
        "RANK_IP": str(scenario.get("environment", {}).get("RANK_IP", "127.0.0.1")),
        "NETWORK_INTERFACE": str(scenario.get("environment", {}).get("NETWORK_INTERFACE", "lo")),
        "DISTRIBUTED_EXECUTOR_BACKEND": str(launch.get("distributed_executor_backend", "mp")),
        "MODEL_NAME": str(scenario["model"]["name"]),
        "MODEL_PATH": model_path.as_posix(),
        "ENGINE": str(scenario["engine"]["name"]),
        "ENGINE_VERSION": str(scenario["engine"].get("version", "")),
        "DEVICE_COUNT": str(scenario["hardware"]["device_count"]),
        "PORT": str(launch.get("port", 18000)),
        "SAVE_PATH": output_dir.as_posix(),
        "WINGS_HARDWARE_FILE": hardware_file.as_posix(),
        "TRUST_REMOTE_CODE": _bool_text(launch.get("trust_remote_code", False)),
        "ENABLE_AUTO_TOOL_CHOICE": _bool_text(launch.get("enable_auto_tool_choice", False)),
        "ENABLE_AUTO_THINK_CHOICE": _bool_text(launch.get("enable_auto_think_choice", False)),
        "ENABLE_SPECULATIVE_DECODE": _bool_text(launch.get("enable_speculative_decode", False)),
        "SPECULATIVE_DECODE_MODEL_PATH": str(launch.get("speculative_decode_model_path", "")),
        "ENABLE_SPARSE": _bool_text(launch.get("enable_sparse", False)),
        "ENABLE_KV_DISK_OFFLOAD": _bool_text(launch.get("enable_kv_disk_offload", False)),
        "ENABLE_KV_OFFLOAD": _bool_text(launch.get("enable_kv_offload", False)),
        "ENABLE_KV_MEM_OFFLOAD": _bool_text(launch.get("enable_kv_offload", False)),
        "LMCACHE_OFFLOAD": _bool_text(launch.get("enable_kv_offload", False)),
        "KV_MEM_OFFLOAD_SIZE": str(launch.get("kv_mem_offload_size", 40)),
        "AVAILABLE_POD_MEM_SIZE": str(launch.get("available_pod_mem_size", 262144)),
    }
    values.update({key: str(value) for key, value in scenario.get("environment", {}).items()})
    return values


def generate_wings_command(scenario: dict[str, Any], output_dir: Path) -> tuple[str, dict[str, Any]]:
    missing = _validate_scenario(scenario)
    if missing:
        raise ValueError(f"场景缺少必填字段: {', '.join(missing)}")

    generated_dir = output_dir / "generated"
    model_path = generated_dir / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    _write_json(model_path / "config.json", _model_config(scenario))
    _write_text(model_path / "wings.txt", str(scenario["model"]["name"]))
    hardware_file = generated_dir / "hardware_info.json"
    _write_json(hardware_file, {
        "device": scenario["hardware"]["device"],
        "hardware_family": scenario["hardware"]["family"],
    })

    sys.dont_write_bytecode = True
    if str(WINGS_CONTROL_ROOT) not in sys.path:
        sys.path.insert(0, str(WINGS_CONTROL_ROOT))
    from config.settings import settings
    from core.port_plan import derive_port_plan
    from core.start_args_compat import parse_launch_args
    from core.wings_entry import build_launcher_plan

    argv = _build_launch_argv(scenario, model_path, output_dir)
    env = _generation_environment(scenario, model_path, hardware_file, output_dir)
    with _patched_environment(env):
        launch_args = parse_launch_args(argv)
        port_plan = derive_port_plan(
            port=launch_args.port,
            enable_reason_proxy=False,
            health_port=settings.HEALTH_PORT,
        )
        plan = build_launcher_plan(launch_args, port_plan)

    _write_text(output_dir / "actual_start_command.sh", plan.command)
    _write_json(output_dir / "merged_params.json", plan.merged_params)
    return plan.command, {
        "model_path": model_path.as_posix(),
        "argv": argv,
        "hardware": plan.hardware_env,
        "active_smart_features": plan.merged_params.get("_smart_feats", []),
        "allowed_smart_features": plan.merged_params.get("_allowed_smart_feats", []),
    }


def _save_comparison(output_dir: Path, standard: NormalizedCommand, actual: NormalizedCommand, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "normalized_standard.json", standard.to_dict())
    _write_json(output_dir / "normalized_actual.json", actual.to_dict())
    _write_text(output_dir / "normalized_standard.sh", render_normalized_shell(standard))
    _write_text(output_dir / "normalized_actual.sh", render_normalized_shell(actual))
    _write_text(output_dir / "actual_exec_command.txt", render_engine_command(actual))
    _write_json(output_dir / "comparison.json", report)
    _write_text(output_dir / "comparison.md", comparison_markdown(report))


def _exit_code(result: str) -> int:
    return {"PASS": 0, "FAIL": 1, "ERROR": 2, "REVIEW": 3}.get(result, 2)


def command_draft(args: argparse.Namespace) -> int:
    raw = _read_text(args.input)
    normalized = normalize_command(raw)
    supplied = _load_json(Path(args.scenario)) if args.scenario else {}
    candidate = _candidate_scenario(normalized, supplied)
    output = Path(args.output)
    _write_text(output / "raw_input.txt", raw)
    _write_text(output / "normalized_command.sh", render_normalized_shell(normalized))
    _write_json(output / "normalized_command.json", normalized.to_dict())
    _write_json(output / "scenario.candidate.json", candidate)
    report = {
        "status": candidate["candidate_analysis"]["status"],
        "repairs": [asdict(item) for item in normalized.repairs],
        "warnings": [asdict(item) for item in normalized.warnings],
        "errors": [asdict(item) for item in normalized.errors],
        "missing_required_context": candidate["candidate_analysis"]["missing_required_context"],
    }
    _write_json(output / "normalization_report.json", report)
    print(f"{report['status']} {output}")
    return 2 if report["status"] == "INVALID_INPUT" else 3 if report["status"] == "REVIEW" else 0


def command_compare(args: argparse.Namespace) -> int:
    standard = normalize_command(_read_text(args.standard))
    actual = normalize_command(_read_text(args.actual))
    scenario = _load_json(Path(args.scenario)) if args.scenario else {}
    report = compare_commands(standard, actual, scenario)
    output = Path(args.output)
    _save_comparison(output, standard, actual, report)
    print(f"{report['result']} {output / 'comparison.md'}")
    return _exit_code(report["result"])


def command_verify(args: argparse.Namespace) -> int:
    scenario = _load_json(Path(args.scenario))
    output = Path(args.output)
    standard = normalize_command(_read_text(args.standard))
    try:
        actual_script, runtime = generate_wings_command(scenario, output)
        actual = normalize_command(actual_script)
        report = compare_commands(standard, actual, scenario, runtime)
        report["runtime"] = runtime
    except Exception as exc:  # 需要把项目生成异常写入稳定报告后交给调用方。
        actual = normalize_command("")
        report = {
            "schema_version": 1,
            "scenario_id": scenario.get("scenario_id", "unknown"),
            "result": "ERROR",
            "summary": {"passed": 0, "failed": 0, "review": 0, "info": 0, "errors": 1},
            "differences": [],
            "generation_error": {"type": type(exc).__name__, "message": str(exc)},
        }
    _save_comparison(output, standard, actual, report)
    print(f"{report['result']} {output / 'comparison.md'}")
    return _exit_code(report["result"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动命令标准化与 Wings 同场景语义对比")
    subparsers = parser.add_subparsers(dest="command", required=True)

    draft = subparsers.add_parser("draft", help="把非标准输入整理为候选标准")
    draft.add_argument("--input", required=True, help="原始命令文件；使用 - 从 stdin 读取")
    draft.add_argument("--scenario", help="可选的场景 JSON，用于补充上下文")
    draft.add_argument("--output", required=True, help="候选标准输出目录")
    draft.set_defaults(func=command_draft)

    compare = subparsers.add_parser("compare", help="比较两份已有命令")
    compare.add_argument("--standard", required=True, help="正式可用的标准命令")
    compare.add_argument("--actual", required=True, help="待比较的实际命令或完整脚本")
    compare.add_argument("--scenario", help="可选的场景和比较策略 JSON")
    compare.add_argument("--output", required=True, help="报告输出目录")
    compare.set_defaults(func=command_compare)

    verify = subparsers.add_parser("verify", help="生成当前项目同场景命令并与标准比较")
    verify.add_argument("--standard", required=True, help="正式可用的标准命令")
    verify.add_argument("--scenario", required=True, help="场景 JSON")
    verify.add_argument("--output", required=True, help="报告和生成产物输出目录")
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
