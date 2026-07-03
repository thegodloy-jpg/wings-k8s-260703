import re


def parse_document_chunks(content):
    chunks = re.split(r'(<\|doc_start\|>|<\|doc_end\|>)', content)
    chunks = [item for item in chunks if item is not None and item.strip()]

    prefix = ""
    postfix = ""

    if chunks and not chunks[0].startswith('<|doc_start|>'):
        prefix = chunks[0]
        chunks = chunks[1:]

    if chunks and not chunks[-1].startswith('<|doc_end|>'):
        postfix = chunks[-1]
        chunks = chunks[:-1]

    chunks = [item for item in chunks if item not in ['<|doc_start|>', '<|doc_end|>']]

    query = _extract_query_from_prefix_postfix(prefix, postfix)

    return prefix, postfix, query, chunks


def _extract_query_from_prefix_postfix(prefix, postfix):
    match = re.search(r'(Question:|Question：|问题：|问题:)(.*)', prefix+postfix, re.IGNORECASE)
    if match:
        query = match.group(2).strip()
    else:
        query = postfix
    return query
