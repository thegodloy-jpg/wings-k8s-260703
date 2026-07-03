# -*- coding: utf-8 -*-
"""Anthropic Messages API ↔ OpenAI Chat Completions API 双向转换器。

在 wings-control proxy 层实现协议转换，使后端只需支持 OpenAI 协议。
同时支持在转换时注入 priority 参数，解决 Anthropic 协议不支持 priority 调度的问题。

转换逻辑与 vLLM 的 AnthropicServingMessages 对齐，支持:
- text / image / thinking / tool_use / tool_result 内容块
- system 消息（含 x-anthropic-billing-header 剥离）
- tool_choice 映射（auto/any/none/tool）
- tools 转换（含 defer_loading）
- output_config（json_schema / reasoning_effort）
- streaming_options（include_usage）
- priority 注入

参考:
    vLLM: vllm/entrypoints/anthropic/serving.py
    SGLang: anthropic/serving.py
"""
from __future__ import annotations
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

# =============================================================================
# 常量映射
# =============================================================================

FINISH_REASON_MAP: Dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


# =============================================================================
# 请求体转换：Anthropic → OpenAI
# =============================================================================


def convert_anthropic_request_to_openai(
    anthropic_body: Dict[str, Any],
    priority: Optional[int] = None,
) -> Dict[str, Any]:
    """将 Anthropic /v1/messages 请求体转换为 OpenAI /v1/chat/completions 格式。

    与 vLLM `_convert_anthropic_to_openai_request()` 逻辑对齐。

    Args:
        anthropic_body: 原始 Anthropic 请求体字典。
        priority: 可选的 priority 值，用于优先级调度。

    Returns:
        OpenAI 格式的请求体字典。
    """
    openai_messages: List[Dict[str, Any]] = []

    # 1. 处理 system 消息
    _convert_system_message(anthropic_body, openai_messages)

    # 2. 处理 messages
    _convert_messages(anthropic_body.get("messages", []), openai_messages)

    # 3. 构建基础请求
    openai_body: Dict[str, Any] = {
        "model": anthropic_body.get("model", ""),
        "messages": openai_messages,
        "max_tokens": anthropic_body.get("max_tokens", 4096),
        "max_completion_tokens": anthropic_body.get("max_tokens", 4096),
        "stream": anthropic_body.get("stream", False),
    }

    # 4. 可选参数
    if "stop_sequences" in anthropic_body:
        openai_body["stop"] = anthropic_body["stop_sequences"]
    if anthropic_body.get("temperature") is not None:
        openai_body["temperature"] = anthropic_body["temperature"]
    if anthropic_body.get("top_p") is not None:
        openai_body["top_p"] = anthropic_body["top_p"]
    if anthropic_body.get("top_k") is not None:
        openai_body["top_k"] = anthropic_body["top_k"]

    # 5. 处理 streaming_options（与 vLLM 一致：流式时设置 include_usage）
    if anthropic_body.get("stream"):
        openai_body["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True,
        }

    # 6. 处理 output_config（json_schema / reasoning_effort）
    _handle_output_config(anthropic_body, openai_body)

    # 7. 转换 tools
    _convert_tools(anthropic_body, openai_body)

    # 8. 转换 tool_choice
    _convert_tool_choice(anthropic_body, openai_body)

    # 9. 注入 priority（关键：解决 Anthropic 协议不支持 priority 调度的问题）
    if priority is not None:
        openai_body["priority"] = priority
    elif "priority" in anthropic_body:
        openai_body["priority"] = anthropic_body["priority"]
    elif anthropic_body.get("metadata") and isinstance(anthropic_body["metadata"], dict):
        meta_priority = anthropic_body["metadata"].get("priority")
        if meta_priority is not None:
            openai_body["priority"] = meta_priority

    return openai_body


def _convert_system_message(
    anthropic_body: Dict[str, Any],
    openai_messages: List[Dict[str, Any]],
) -> None:
    """转换 Anthropic system 消息为 OpenAI 格式。

    与 vLLM `_convert_system_message()` 逻辑对齐:
    - string 类型直接添加
    - list 类型提取 text 块，剥离 x-anthropic-billing-header
    """
    system = anthropic_body.get("system")
    if not system:
        return

    if isinstance(system, str):
        openai_messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        system_prompt = _extract_system_prompt(system)
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})


def _extract_system_prompt(system: List[Dict[str, Any]]) -> str:
    """从 system 列表中提取文本内容，剥离 billing header。

    Args:
        system: Anthropic system 消息列表。

    Returns:
        提取的 system prompt 文本。
    """
    system_prompt = ""
    for block in system:
        if not isinstance(block, dict):
            continue

        if block.get("type") == "text" and block.get("text"):
            text = block["text"]
            # 剥离 Claude Code 的 attribution header（含 per-request hash，破坏 prefix caching）
            if text.startswith("x-anthropic-billing-header"):
                continue
            system_prompt += text
    return system_prompt




def _convert_messages(
    messages: List[Dict[str, Any]],
    openai_messages: List[Dict[str, Any]],
) -> None:
    """转换 Anthropic messages 为 OpenAI 格式。

    与 vLLM `_convert_messages()` 逻辑对齐。
    """
    for msg in messages:
        role = msg.get("role", "user")
        openai_msg: Dict[str, Any] = {"role": role}
        content = msg.get("content", "")

        if isinstance(content, str):
            openai_msg["content"] = content
        elif isinstance(content, list):
            _convert_message_content(content, role, openai_msg, openai_messages)

        # 跳过空的 user 消息（vLLM 逻辑）
        if not (role == "user" and "content" not in openai_msg and "tool_calls" not in openai_msg):
            openai_messages.append(openai_msg)


class _BlockConverterContext:
    """Block 转换上下文，封装 _convert_block 的多个参数。"""

    def __init__(
        self,
        role: str,
        content_parts: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        reasoning_parts: List[str],
        openai_messages: List[Dict[str, Any]],
    ) -> None:
        self.role = role
        self.content_parts = content_parts
        self.tool_calls = tool_calls
        self.reasoning_parts = reasoning_parts
        self.openai_messages = openai_messages


def _convert_message_content(
    blocks: List[Dict[str, Any]],
    role: str,
    openai_msg: Dict[str, Any],
    openai_messages: List[Dict[str, Any]],
) -> None:
    """转换复杂的 content blocks。

    与 vLLM `_convert_message_content()` 逻辑对齐。
    """
    content_parts: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    reasoning_parts: List[str] = []

    ctx = _BlockConverterContext(role, content_parts, tool_calls, reasoning_parts, openai_messages)
    for block in blocks:
        _convert_block(block, ctx)

    if reasoning_parts:
        openai_msg["reasoning"] = "".join(reasoning_parts)

    if tool_calls:
        openai_msg["tool_calls"] = tool_calls

    if content_parts:
        if len(content_parts) == 1 and content_parts[0]["type"] == "text":
            openai_msg["content"] = content_parts[0]["text"]
        else:
            openai_msg["content"] = content_parts
    elif not tool_calls and not reasoning_parts:
        return


def _convert_block(
    block: Dict[str, Any],
    ctx: _BlockConverterContext,
) -> None:
    """转换单个 content block。

    与 vLLM `_convert_block()` 逻辑对齐，支持:
    - text / image / thinking / redacted_thinking / tool_use / tool_result / tool_reference
    """
    block_type = block.get("type", "")

    if block_type == "text" and block.get("text"):
        ctx.content_parts.append({"type": "text", "text": block["text"]})

    elif block_type == "image" and block.get("source"):
        image_url = _convert_image_source_to_url(block["source"])
        ctx.content_parts.append({"type": "image_url", "image_url": {"url": image_url}})

    elif block_type == "thinking" and block.get("thinking") is not None:
        ctx.reasoning_parts.append(block["thinking"])

    elif block_type == "redacted_thinking":
        # redacted_thinking 包含安全过滤后的推理内容（opaque base64 'data' 字段）
        # 跳过以避免客户端回显完整 assistant 消息时的校验错误
        pass

    elif block_type == "tool_use":
        _convert_tool_use_block(block, ctx.tool_calls)

    elif block_type == "tool_result":
        _convert_tool_result_block(block, ctx.role, ctx.openai_messages, ctx.content_parts)

    elif block_type == "tool_reference":
        # tool_reference 在 tool_result 处理时展开
        pass


def _convert_image_source_to_url(source: Dict[str, Any]) -> str:
    """转换 Anthropic image source 为 OpenAI 兼容的 URL。

    与 vLLM `_convert_image_source_to_url()` 逻辑对齐:
    - base64: {"type": "base64", "media_type": "image/jpeg", "data": "..."}
      → data:image/jpeg;base64,...
    - url: {"type": "url", "url": "https://..."}
      → https://...
    """
    source_type = source.get("type")
    if source_type == "url":
        return source.get("url", "")
    # 默认按 base64 处理
    media_type = source.get("media_type", "image/jpeg")
    data = source.get("data", "")
    return f"data:{media_type};base64,{data}"


def _convert_tool_use_block(
    block: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
) -> None:
    """转换 tool_use block 为 OpenAI function call 格式。

    与 vLLM `_convert_tool_use_block()` 逻辑对齐。
    """
    tool_call = {
        "id": block.get("id") or f"call_{int(time.time())}",
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
        },
    }
    tool_calls.append(tool_call)


def _convert_tool_result_block(
    block: Dict[str, Any],
    role: str,
    openai_messages: List[Dict[str, Any]],
    content_parts: List[Dict[str, Any]],
) -> None:
    """转换 tool_result block 为 OpenAI 格式。

    与 vLLM `_convert_tool_result_block()` 逻辑对齐:
    - user role: 调用 _convert_user_tool_result 处理文本+图片+tool_reference
    - assistant role: 简化为文本
    """
    if role == "user":
        _convert_user_tool_result(block, openai_messages)
    else:
        tool_result_text = str(block.get("content", "")) if block.get("content") else ""
        content_parts.append({"type": "text", "text": f"Tool result: {tool_result_text}"})


def _convert_user_tool_result(
    block: Dict[str, Any],
    openai_messages: List[Dict[str, Any]],
) -> None:
    """转换 user tool_result，支持文本和图片。

    与 vLLM `_convert_user_tool_result()` 逻辑对齐:
    - 文本内容 → role=tool 消息
    - 图片内容 → 后续 role=user 消息
    - tool_reference → 后续 role=tool 消息
    """
    tool_text = ""
    tool_image_urls: List[str] = []
    tool_references: List[Dict[str, Any]] = []

    content = block.get("content", "")
    if isinstance(content, str):
        tool_text = content
    elif isinstance(content, list):
        tool_text, tool_image_urls, tool_references = _parse_tool_result_content(content)

    # 添加 tool 消息
    openai_messages.append({
        "role": "tool",
        "tool_call_id": block.get("tool_use_id", ""),
        "content": tool_text or "",
    })

    # 图片内容作为后续 user 消息
    if tool_image_urls:
        openai_messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img}}
                for img in tool_image_urls
            ],
        })

    # tool_reference 作为后续 tool 消息
    if tool_references:
        openai_messages.append({
            "role": "tool",
            "tool_call_id": block.get("tool_use_id", ""),
            "content": tool_references,
        })


def _parse_tool_result_content(
    content: List[Dict[str, Any]],
) -> tuple[str, List[str], List[Dict[str, Any]]]:
    """解析 tool_result 的 content 列表。

    Returns:
        (tool_text, tool_image_urls, tool_references)
    """
    text_parts: List[str] = []
    tool_image_urls: List[str] = []
    tool_references: List[Dict[str, Any]] = []

    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            text_parts.append(item.get("text", ""))
        elif item_type == "image":
            source = item.get("source", {})
            url = _convert_image_source_to_url(source)
            if url:
                tool_image_urls.append(url)
        elif item_type == "tool_reference":
            ref_name = item.get("tool_name") or item.get("name")
            if ref_name:
                tool_references.append({"type": "tool_reference", "name": ref_name})

    tool_text = "\n".join(text_parts)
    return tool_text, tool_image_urls, tool_references




def _handle_output_config(
    anthropic_body: Dict[str, Any],
    openai_body: Dict[str, Any],
) -> None:
    """处理 output_config（json_schema / reasoning_effort）。

    与 vLLM `_handle_output_config()` 逻辑对齐。
    """
    output_config = anthropic_body.get("output_config")
    if not output_config or not isinstance(output_config, dict):
        return

    fmt = output_config.get("format")
    if fmt and isinstance(fmt, dict) and fmt.get("json_schema"):
        openai_body["response_format"] = {
            "type": fmt.get("type", "json_schema"),
            "json_schema": {
                "schema": fmt["json_schema"],
                "name": fmt.get("type", "json_schema"),
            },
        }

    effort = output_config.get("effort")
    if effort is not None:
        openai_body["reasoning_effort"] = effort


def _convert_tools(
    anthropic_body: Dict[str, Any],
    openai_body: Dict[str, Any],
) -> None:
    """转换 Anthropic tools 为 OpenAI tools 格式。

    与 vLLM `_convert_tools()` 逻辑对齐:
    Anthropic: {"name": "...", "description": "...", "input_schema": {...}, "defer_loading": ...}
    OpenAI:   {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    tools = anthropic_body.get("tools")
    if not tools:
        return

    openai_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        openai_tool: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
        # 保留 defer_loading（SGLang 也支持此属性）
        if "defer_loading" in tool:
            openai_tool["function"]["defer_loading"] = tool["defer_loading"]
        openai_tools.append(openai_tool)

    if openai_tools:
        openai_body["tools"] = openai_tools
        # 如果未设置 tool_choice，默认为 auto（与 vLLM 一致）
        if "tool_choice" not in openai_body or openai_body.get("tool_choice") is None:
            openai_body["tool_choice"] = "auto"


def _convert_tool_choice(
    anthropic_body: Dict[str, Any],
    openai_body: Dict[str, Any],
) -> None:
    """转换 Anthropic tool_choice 为 OpenAI 格式。

    与 vLLM `_convert_tool_choice()` 逻辑对齐:
    Anthropic → OpenAI:
        "auto"     → "auto"
        "any"      → "required"
        "none"     → "none"
        {"type": "tool", "name": "x"} → {"type": "function", "function": {"name": "x"}}
    """
    tool_choice = anthropic_body.get("tool_choice")
    if tool_choice is None:
        return

    if isinstance(tool_choice, str):
        mapping = {"auto": "auto", "any": "required", "none": "none"}
        openai_body["tool_choice"] = mapping.get(tool_choice, "auto")
    elif isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "auto")
        if tc_type == "auto":
            openai_body["tool_choice"] = "auto"
        elif tc_type == "any":
            openai_body["tool_choice"] = "required"
        elif tc_type == "none":
            openai_body["tool_choice"] = "none"
        elif tc_type == "tool":
            openai_body["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }


# =============================================================================
# 响应体转换：OpenAI → Anthropic（非流式）
# =============================================================================


def convert_openai_response_to_anthropic(
    openai_body: Dict[str, Any],
) -> Dict[str, Any]:
    """将 OpenAI /v1/chat/completions 响应体转换为 Anthropic /v1/messages 格式。

    与 vLLM `messages_full_converter()` 逻辑对齐。

    Args:
        openai_body: OpenAI 格式的响应体字典。

    Returns:
        Anthropic 格式的响应体字典。
    """
    choice = openai_body.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = openai_body.get("usage", {})

    # 构建 content blocks
    content: List[Dict[str, Any]] = []

    # 1. reasoning → thinking（vLLM 支持的特性）
    reasoning = message.get("reasoning")
    if reasoning:
        content.append({
            "type": "thinking",
            "thinking": reasoning,
        })

    # 2. text content
    msg_content = message.get("content")
    if msg_content:
        content.append({
            "type": "text",
            "text": msg_content,
        })

    # 3. tool_calls → tool_use
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                arguments = {}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": tc["function"]["name"],
                "input": arguments,
            })

    # 4. stop_reason 映射
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = FINISH_REASON_MAP.get(finish_reason, finish_reason)

    return {
        "id": openai_body.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": openai_body.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# =============================================================================
# 流式响应转换：OpenAI SSE → Anthropic SSE
# =============================================================================


class _ActiveBlockState:
    """流式转换状态跟踪。

    与 vLLM `message_stream_converter()` 中的 `_ActiveBlockState` 逻辑对齐。
    管理当前活跃的 content block 类型和索引，确保 Anthropic SSE 事件序列的正确性。
    """

    def __init__(self) -> None:
        self.content_block_index: int = 0
        self.block_type: Optional[str] = None
        self.block_index: Optional[int] = None
        self.tool_use_id: Optional[str] = None

    def reset(self) -> None:
        self.block_type = None
        self.block_index = None
        self.tool_use_id = None

    def start(self, block_type: str, tool_id: Optional[str] = None) -> None:
        self.block_type = block_type
        self.block_index = self.content_block_index
        self.tool_use_id = tool_id if block_type == "tool_use" else None


def _sse_event(event: str, data: dict) -> str:
    """构建 Anthropic SSE 事件字符串。

    Anthropic SSE 格式:
        event: {event_name}\n
        data: {json_data}\n\n
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _StreamConverter:
    """流式转换器，封装 OpenAI SSE 到 Anthropic SSE 的转换逻辑。"""

    def __init__(self) -> None:
        self.state = _ActiveBlockState()
        self.first_item = True
        self.finish_reason: Optional[str] = None
        self.tool_index_to_id: Dict[int, str] = {}
        self.buffer = ""

    def stop_active_block(self) -> str:
        """关闭当前活跃的 content block，返回 SSE 事件。"""
        if self.state.block_type is None:
            return ""
        events = _sse_event("content_block_stop", {
            "index": self.state.block_index,
            "type": "content_block_stop",
        })
        self.state.reset()
        self.state.content_block_index += 1
        return events

    def start_block(self, block_type: str, **kwargs) -> str:
        """开启新的 content block，返回 SSE 事件。"""
        block: Dict[str, Any] = {"type": block_type}
        block.update(kwargs)
        event_data = {
            "index": self.state.content_block_index,
            "type": "content_block_start",
            "content_block": block,
        }
        self.state.start(block_type, tool_id=kwargs.get("id"))
        return _sse_event("content_block_start", event_data)

    def get_block_index(self) -> int:
        """获取当前 block 索引，优先使用 block_index。"""
        return self.state.block_index if self.state.block_index is not None else self.state.content_block_index

    async def convert(self, openai_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """执行流式转换。

        Args:
            openai_stream: OpenAI 格式的 SSE 字节流异步迭代器。

        Yields:
            Anthropic 格式的 SSE 事件字节。
        """
        async for chunk in openai_stream:
            if not chunk:
                continue

            text = chunk.decode("utf-8", errors="replace")
            self.buffer += text

            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                line = line.strip()

                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    yield _sse_event("message_stop", {"type": "message_stop"}).encode("utf-8")
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])

                # 1. message_start（首个 chunk）
                if self.first_item:
                    yield self._handle_message_start(data)
                    continue

                # 2. 最后一个 chunk（含 usage，choices 为空）
                if len(choices) == 0:
                    yield self._handle_final_chunk(data)
                    continue

                # 3. 记录 finish_reason
                if choices[0].get("finish_reason") is not None:
                    self.finish_reason = choices[0]["finish_reason"]

                delta = choices[0].get("delta", {})

                # 4. reasoning → thinking
                reasoning_delta = delta.get("reasoning")
                if reasoning_delta is not None and reasoning_delta != "":
                    yield self._handle_reasoning_delta(reasoning_delta)

                # 5. text delta
                text_delta = delta.get("content")
                if text_delta is not None and text_delta != "":
                    yield self._handle_text_delta(text_delta)

                # 6. tool_calls delta
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    yield self._handle_tool_calls_delta(tool_calls)

    def _handle_message_start(self, data: Dict[str, Any]) -> bytes:
        """处理首个 chunk，生成 message_start 事件。"""
        self.first_item = False
        msg_start = {
            "type": "message",
            "id": data.get("id", ""),
            "content": [],
            "model": data.get("model", ""),
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": data.get("usage", {}).get("prompt_tokens", 0) if data.get("usage") else 0,
                "output_tokens": 0,
            },
        }
        return _sse_event("message_start", {
            "type": "message_start",
            "message": msg_start,
        }).encode("utf-8")

    def _handle_final_chunk(self, data: Dict[str, Any]) -> bytes:
        """处理最后一个 chunk（含 usage，choices 为空），生成 message_delta 事件。"""
        stop_events = self.stop_active_block()
        result = stop_events.encode("utf-8") if stop_events else b""

        stop_reason_str = FINISH_REASON_MAP.get(self.finish_reason or "stop", self.finish_reason or "stop")
        usage = data.get("usage", {})
        delta_event = _sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason_str},
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            } if usage else None,
        }).encode("utf-8")
        return result + delta_event

    def _handle_reasoning_delta(self, reasoning_delta: str) -> bytes:
        """处理 reasoning delta，生成 thinking 相关事件。"""
        if self.state.block_type != "thinking":
            stop_events = self.stop_active_block()
            result = stop_events.encode("utf-8") if stop_events else b""
            result += self.start_block("thinking", thinking="").encode("utf-8")
        else:
            result = b""

        block_idx = self.get_block_index()
        result += _sse_event("content_block_delta", {
            "index": block_idx,
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": reasoning_delta},
        }).encode("utf-8")
        return result

    def _handle_text_delta(self, text_delta: str) -> bytes:
        """处理 text delta，生成 text 相关事件。"""
        if self.state.block_type != "text":
            stop_events = self.stop_active_block()
            result = stop_events.encode("utf-8") if stop_events else b""
            result += self.start_block("text", text="").encode("utf-8")
        else:
            result = b""

        block_idx = self.get_block_index()
        result += _sse_event("content_block_delta", {
            "index": block_idx,
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text_delta},
        }).encode("utf-8")
        return result

    def _handle_tool_calls_delta(self, tool_calls: List[Dict[str, Any]]) -> bytes:
        """处理 tool_calls delta，生成 tool_use 相关事件。"""
        result = b""
        for tc in tool_calls:
            idx = tc.get("index", 0)
            tc_id = tc.get("id")
            func = tc.get("function", {}) or {}

            if tc_id is not None:
                result += self._handle_tool_call_with_id(idx, tc_id, func)
            else:
                result += self._handle_tool_call_incremental(idx, func)
        return result

    def _handle_tool_call_with_id(self, idx: int, tc_id: str, func: Dict[str, Any]) -> bytes:
        """处理带 id 的 tool_call（新 tool call 或初始 arguments）。"""
        result = b""
        self.tool_index_to_id[idx] = tc_id
        tool_name = func.get("name")

        if self.state.tool_use_id != tc_id and tool_name is not None:
            # 新的 tool call → 开启新的 content block
            stop_events = self.stop_active_block()
            if stop_events:
                result += stop_events.encode("utf-8")
            result += self.start_block("tool_use", id=tc_id, name=tool_name, input={}).encode("utf-8")

        # 处理初始 arguments
        if func.get("arguments") and self.state.tool_use_id == tc_id:
            result += self._emit_input_json_delta(func["arguments"])
        return result

    def _handle_tool_call_incremental(self, idx: int, func: Dict[str, Any]) -> bytes:
        """处理增量 tool_call（通过 index 查找 tool_use_id）。"""
        tool_use_id = self.tool_index_to_id.get(idx)
        if not (tool_use_id and func.get("arguments") and self.state.tool_use_id == tool_use_id):
            return b""
        return self._emit_input_json_delta(func["arguments"])

    def _emit_input_json_delta(self, partial_json: str) -> bytes:
        """生成 input_json_delta 事件。"""
        block_idx = self.get_block_index()
        return _sse_event("content_block_delta", {
            "index": block_idx,
            "type": "content_block_delta",
            "delta": {
                "type": "input_json_delta",
                "partial_json": partial_json,
            },
        }).encode("utf-8")


async def convert_openai_stream_to_anthropic(
    openai_stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """将 OpenAI SSE 流式响应转换为 Anthropic SSE 流式响应。

    与 vLLM `message_stream_converter()` 逻辑对齐。

    事件映射:
    | OpenAI SSE 事件                          | Anthropic 事件                           |
    |------------------------------------------|------------------------------------------|
    | 首个 chunk                               | event: message_start                     |
    | delta.reasoning                          | event: content_block_start (thinking)    |
    |                                          |   + content_block_delta (thinking_delta) |
    | delta.content                            | event: content_block_start (text)        |
    |                                          |   + content_block_delta (text_delta)     |
    | delta.tool_calls (有 id + name)          | event: content_block_start (tool_use)    |
    | delta.tool_calls (增量 arguments)        | event: content_block_delta (input_json)  |
    | finish_reason                            | event: content_block_stop                |
    |                                          |   + event: message_delta (stop_reason)   |
    | usage (choices 为空)                     | event: message_delta (usage)             |
    | [DONE]                                   | event: message_stop                      |

    Args:
        openai_stream: OpenAI 格式的 SSE 字节流异步迭代器。

    Yields:
        Anthropic 格式的 SSE 事件字节。
    """
    converter = _StreamConverter()
    async for chunk in converter.convert(openai_stream):
        yield chunk
