from fastapi import Request
from fastapi.responses import StreamingResponse
from proxy.proxy_config import logger

from rag_acc.stream_collector import StreamCollector
from rag_acc.request_handlers import create_simple_request, create_chunk_request, create_combine_request
from rag_acc.document_processor import parse_document_chunks
from rag_acc.extract_dify_info import is_dify_scenario, extract_dify_info


MIN_CONTENT_LENGTH = 2048
MIN_DOC_BLOCKS = 3


def is_warmup_scenario(content: str) -> bool:
    return "/rag_acc_warm_up" in content


def is_rag_scenario(chat_input, request: Request) -> bool:
    if len(chat_input.messages) > 1:
        msg = chat_input.messages[-1]
    else:
        msg = chat_input.messages[0]

    content = ""
    try:
        if hasattr(msg, 'content'):
            content = str(msg.content or "")
        elif isinstance(msg, dict):
            content = str(msg.get("content", "") or "")
        else:
            content = str(msg)
    except Exception as _:
        content = str(msg)

    if is_warmup_scenario(content):
        return True

    if '<|doc_start|>' not in content or '<|doc_end|>' not in content:
        return False

    if len(content) < MIN_CONTENT_LENGTH:
        logger.info(
            f"RAG scenario detected, but content length is less than 2K. "
            f"content_length: {len(content)}")
        return False
    
    doc_start_count = content.count('<|doc_start|>')
    doc_end_count = content.count('<|doc_end|>')

    if doc_start_count < MIN_DOC_BLOCKS or doc_end_count < MIN_DOC_BLOCKS:
        logger.info(
            f"RAG scenario detected, but the number of blocks is too small to perform acceleration. "
            f"doc_start_count: {doc_start_count}, doc_end_count: {doc_end_count}")
        return False
    
    return True


def extract_last_message_and_log(chat_input):
    logger.debug("input: %s", chat_input)
    messages = []

    if len(chat_input.messages) > 1:
        messages = chat_input.messages[:-1]
        msg = chat_input.messages[-1]
    else:
        msg = chat_input.messages[0]

    role = ""
    content = ""
    try:
        if hasattr(msg, 'role'):
            role = str(msg.role or "user")
        elif isinstance(msg, dict):
            role = str(msg.get("role", "user") or "user")
        else:
            role = "user"
    except Exception as _:
        role = "user"
        
    try:
        if hasattr(msg, 'content'):
            content = str(msg.content or "")
        elif isinstance(msg, dict):
            content = str(msg.get("content", "") or "")
        else:
            content = str(msg)
    except Exception as _:
        content = str(msg)

    logger.debug("msg content: %s, %s, %s", content, messages, role)
    return messages, msg, role, content


async def rag_acc_chat(chat_input, request: Request, backend_url: str = ""):
    _, _, _, content = extract_last_message_and_log(chat_input)

    is_rag = is_rag_scenario(chat_input, request)
    is_dify = is_dify_scenario(chat_input)
    if not is_rag and not is_dify:
        simple_request = create_simple_request(chat_input, {}, backend_url)
        return StreamingResponse(simple_request())

    if is_rag:
        prefix, postfix, query, chunks = parse_document_chunks(content)
        logger.debug("prefix length: %d, postfix length: %d, query length: %d", len(prefix), len(postfix), len(query))
        for i, c in enumerate(chunks):
            logger.debug("chunk[%d] length: %d", i, len(c))

    elif is_dify:
        result = extract_dify_info(chat_input)
        query = result["user_question"]
        chunks = result["rag_documents"]

    if is_warmup_scenario(content):
        query = "<query_warm_up>"
        chunks = ["<query_warm_up>"]

    chunks = sorted(chunks, key=len)

    chunk_request = create_chunk_request(
        chat_input, query, chunks, {}, backend_url)
    combine_request = create_combine_request(
        chat_input, query, chunks, {}, backend_url)

    collector = StreamCollector(len(chunks), chunk_request, combine_request)
    return StreamingResponse(collector.collect_and_stream(), media_type="text/event-stream")
