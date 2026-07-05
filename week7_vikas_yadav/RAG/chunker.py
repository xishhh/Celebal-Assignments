import re
from typing import Dict, List

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    result = []
    counter: Dict[str, int] = {}

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        page = doc["page"]

        chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_text in chunks:
            key = f"{source}_p{page}" if page else f"{source}"
            counter[key] = counter.get(key, 0) + 1
            chunk_id = f"{key}_chunk{counter[key]}"

            result.append({
                "text": chunk_text,
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
            })

    return result


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    units = _split_into_units(text, chunk_size)
    if not units:
        return []
    return _merge_into_chunks(units, chunk_size, chunk_overlap)


def _split_into_units(text: str, chunk_size: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        result = []
        for p in paragraphs:
            if len(p) > chunk_size:
                result.extend(_split_into_units(p, chunk_size))
            else:
                result.append(p)
        return result

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        result = []
        for s in sentences:
            if len(s) > chunk_size:
                result.extend(_split_into_units(s, chunk_size))
            else:
                result.append(s)
        return result

    words = text.split()
    result = []
    current = []
    current_len = 0
    for w in words:
        w_len = len(w)
        sep = 1 if current else 0
        if current_len + sep + w_len > chunk_size:
            result.append(" ".join(current))
            current = [w]
            current_len = w_len
        else:
            current.append(w)
            current_len += sep + w_len
    if current:
        result.append(" ".join(current))
    return result


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _merge_into_chunks(units: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    if not units:
        return []

    chunks = []
    start = 0

    while start < len(units):
        current = [units[start]]
        current_len = len(units[start])
        end = start + 1

        while end < len(units):
            sep = "\n\n"
            if current_len + len(sep) + len(units[end]) <= chunk_size:
                current.append(units[end])
                current_len += len(sep) + len(units[end])
                end += 1
            else:
                break

        chunks.append("\n\n".join(current))

        if end >= len(units):
            break

        next_start = end
        cum_len = 0
        for k in range(end - 1, start - 1, -1):
            cum_len += len(units[k]) + (2 if k < end - 1 else 0)
            if cum_len >= chunk_overlap:
                next_start = k
                break

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks
