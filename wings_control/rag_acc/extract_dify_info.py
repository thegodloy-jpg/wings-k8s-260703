import re
from proxy.proxy_config import logger


def _get_msg_field(msg, field: str, default=""):
    """Pydantic 模型 / dict 兼容的字段提取。"""
    if isinstance(msg, dict):
        return msg.get(field, default)
    return getattr(msg, field, default)


def is_dify_scenario(chat_input) -> bool:
    if len(chat_input.messages) != 2:
        return False
    system_msg = None
    for msg in chat_input.messages:
        if _get_msg_field(msg, "role") == "system":
            system_msg = msg
    if system_msg:
        content = _get_msg_field(system_msg, "content", "")
        if "<context>" in content and "</context>" in content:
            return True
    return False


def extract_dify_info(chat_input):
    if not is_dify_scenario(chat_input):
        return None
    system_prompt = ""
    user_question = ""
    rag_documents = []
    for msg in chat_input.messages:
        if _get_msg_field(msg, "role") == "system":
            system_prompt = _get_msg_field(msg, "content", "")
            content = system_prompt
            logger.debug("Dify system prompt length: %d", len(content))
            if isinstance(content, str):
                context_match = re.search(r'<context>\n(.*?)\n</context>', content, re.DOTALL)
                if context_match:
                    context_content = context_match.group(1)
                    doc_chunks = context_content.split('\n')
                    logger.debug("Extracted context chunks: %d", len(doc_chunks))
                    doc_chunks = [item.strip() for item in doc_chunks if item.strip()]
                    rag_documents.extend(doc_chunks)
                else:
                    logger.info("No <context> tag found in content.")
        elif _get_msg_field(msg, "role") == "user":
            user_question = _get_msg_field(msg, "content", "")
    return {
        "rag_documents": rag_documents,
        "system_prompt": system_prompt,
        "user_question": user_question
    }
