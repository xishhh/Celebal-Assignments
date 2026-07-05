import logging
from typing import Dict, List

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based solely on the provided context.

Rules:
1. Answer ONLY using the information in the "Context" section below. You may synthesize and summarize across multiple chunks, but do not add facts not present in the context.
2. If the context does not contain enough information to answer the question, say "I don't have enough information to answer this question." Do NOT make up or infer answers.
3. Always mention which source(s) you used (include the source filename and page number where available).
4. Be concise and direct."""


def build_prompt(question: str, chunks: List[Dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        page_str = f", page {page}" if page is not None else ""
        label = f"Source [{i}]: {source}{page_str}"
        context_blocks.append(f"### {label}\n{chunk['text']}")

    context = "\n\n".join(context_blocks)

    prompt = f"{context}\n\nQuestion: {question}"
    return prompt


def _call_groq(messages: List[Dict]) -> str:
    if not GROQ_API_KEY:
        return (
            "[WARNING] GROQ_API_KEY is not set. "
            "Set it in your environment or switch LLM_PROVIDER to 'openai' in config.py."
        )
    import groq

    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        reply = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        return reply.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Groq API call failed: %s", e)
        return f"Sorry, I encountered an error while generating the answer: {e}"


def _call_openai(messages: List[Dict]) -> str:
    if not OPENAI_API_KEY:
        return (
            "[WARNING] OPENAI_API_KEY is not set. "
            "Set it in your environment or switch LLM_PROVIDER to 'groq' in config.py."
        )
    import openai

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        reply = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        return reply.choices[0].message.content.strip()
    except Exception as e:
        logger.error("OpenAI API call failed: %s", e)
        return f"Sorry, I encountered an error while generating the answer: {e}"


def generate_answer(question: str, chunks: List[Dict]) -> str:
    if not chunks:
        return "I couldn't find relevant information."

    prompt = build_prompt(question, chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if LLM_PROVIDER == "groq":
        return _call_groq(messages)
    elif LLM_PROVIDER == "openai":
        return _call_openai(messages)
    else:
        return f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Set it to 'groq' or 'openai' in config.py."
