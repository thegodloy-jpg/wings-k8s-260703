import json
from contextlib import asynccontextmanager

import httpx

from proxy.proxy_config import logger
from rag_acc.prompt_manager import generate_map_prompt, generate_combine_prompt

REQUEST_TIMEOUT = 600
_TIMEOUT = httpx.Timeout(REQUEST_TIMEOUT)


def _build_request_data(input_data, messages=None, is_for_kvcache_preparation=False):
    msg_list = messages if messages is not None else input_data.messages
    data = {
        "model": input_data.model,
        "messages": msg_list,
        "stream": input_data.stream if input_data.stream is not None else True,
    }
    optional_params = [
        ("temperature", input_data.temperature),
        ("top_p", input_data.top_p),
        ("top_k", input_data.top_k),
        ("n", input_data.n),
        ("max_tokens", input_data.max_tokens),
        ("stop", input_data.stop),
        ("presence_penalty", input_data.presence_penalty),
        ("frequency_penalty", input_data.frequency_penalty),
    ]
    for param_name, param_value in optional_params:
        if param_value is not None:
            data[param_name] = param_value
    if is_for_kvcache_preparation:
        data["max_tokens"] = 1
    return data


def create_simple_request(input_data, extra_headers, backend_url):
    async def simple_request():
        data = _build_request_data(input_data)
        data_json = json.dumps(data)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream(
                "POST", backend_url,
                content=data_json,
                headers={"Content-type": "application/json", **extra_headers},
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
    return simple_request


def create_chunk_request(input_data, query, chunks, extra_headers, backend_url):
    @asynccontextmanager
    async def chunk_request(index):
        chunk = chunks[index]
        messages = [
            {"role": "user", "content": generate_map_prompt(query, chunk)}]
        is_for_kvcache_preparation = query == "<query_warm_up>"
        data = _build_request_data(input_data, messages, is_for_kvcache_preparation)
        data_json = json.dumps(data)
        logger.debug("chunk inputs: %s", data)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream(
                "POST", backend_url,
                content=data_json,
                headers={"Content-type": "application/json", **extra_headers},
            ) as resp:
                yield resp
    return chunk_request


def create_combine_request(input_data, query, chunks, extra_headers, backend_url):
    @asynccontextmanager
    async def combine_request(preliminary_analysis):
        if preliminary_analysis.endswith("<|preparation|>"):
            half_chunks = chunks[:len(chunks)//2]
            messages = [
                {"role": "user", "content": generate_combine_prompt(query, half_chunks, preliminary_analysis)}]
        else:
            messages = [
                {"role": "user", "content": generate_combine_prompt(query, chunks, preliminary_analysis)}]
        is_for_kvcache_preparation = preliminary_analysis.endswith("<|preparation|>") or query == "<query_warm_up>"
        data = _build_request_data(input_data, messages, is_for_kvcache_preparation)
        data_json = json.dumps(data)
        logger.debug("combine_request: inputs: %s", data)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream(
                "POST", backend_url,
                content=data_json,
                headers={"Content-type": "application/json", **extra_headers},
            ) as resp:
                yield resp
    return combine_request
