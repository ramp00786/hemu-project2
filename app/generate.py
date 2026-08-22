from __future__ import annotations

import json
import os
import re

import httpx

from app.retrieve import RetrievedChunk

# Prefer the smallest available Groq chat model for low latency.
# llama-3.1-8b-instant was shut down 16 Aug 2026 (Groq free/dev tier).
DEFAULT_GROQ_MODEL = "allam-2-7b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Fewer / shorter passages → fewer input tokens → faster Groq TTFT.
DEFAULT_LLM_TOP_K = 2
DEFAULT_LLM_CHUNK_CHARS = 320
DEFAULT_LLM_MAX_TOKENS = 80

_http: httpx.AsyncClient | None = None


def _env(name: str) -> str:
    """Read env and drop inline comments like `KEY=value # note`."""
    raw = os.getenv(name, "") or ""
    return raw.split("#", 1)[0].strip().strip('"').strip("'")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def llm_context_chunks(chunks: list[RetrievedChunk], question: str = "") -> list[RetrievedChunk]:
    """Pick a small, question-overlapping slice for the LLM prompt."""
    top_k = _env_int("LLM_TOP_K", DEFAULT_LLM_TOP_K)
    if not chunks:
        return []
    if not question:
        return list(chunks[:top_k])

    q_toks = {
        t
        for t in re.findall(r"[A-Za-z0-9_\u0900-\u097F]{2,}", question.lower())
        if t
        not in {
            "the",
            "a",
            "an",
            "is",
            "are",
            "to",
            "of",
            "in",
            "on",
            "for",
            "and",
            "or",
            "what",
            "how",
            "why",
            "who",
            "when",
            "where",
            "does",
            "did",
            "do",
            "with",
            "from",
            "that",
            "this",
            "it",
            "क्या",
            "है",
            "हैं",
            "का",
            "की",
            "के",
            "में",
            "से",
            "को",
            "पर",
            "और",
            "या",
            "एक",
            "यह",
            "वह",
            "लिए",
            "कैसे",
            "क्यों",
            "कब",
            "कहाँ",
            "कितनी",
            "कितना",
            "करता",
            "करती",
        }
    }
    if not q_toks:
        return list(chunks[:top_k])

    def _cover(c: RetrievedChunk) -> tuple[float, float]:
        text_toks = set(re.findall(r"[A-Za-z0-9_\u0900-\u097F]{2,}", (c.text or "").lower()))
        overlap = q_toks & text_toks
        return (len(overlap) / max(1, len(q_toks)), c.score)

    ranked = sorted(chunks, key=_cover, reverse=True)
    return list(ranked[:top_k])


def _trim_chunk_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return (cut or text[:limit]) + "…"


def _grounded_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    char_limit = _env_int("LLM_CHUNK_CHARS", DEFAULT_LLM_CHUNK_CHARS)
    context = "\n\n".join(
        f"[{i + 1}] {_trim_chunk_text(c.text, char_limit)}" for i, c in enumerate(chunks)
    )
    return (
        "Answer using only the passages. "
        "Write ONE short sentence in English (max ~20 words). "
        "Use Hindi only if the question is clearly in Hindi. "
        "If a detail is missing, state the closest supported fact. "
        "Output NO_ANSWER only if passages are a completely different topic.\n\n"
        f"Q: {question}\n\n"
        f"Passages:\n{context}\n"
    )


def _clean_completion(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # Drop chain-of-thought prefixes that gpt-oss sometimes leaks into content.
    markers = (
        "final answer:",
        "answer:",
        "the answer is:",
    )
    lowered = text.lower()
    cut = -1
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx >= 0:
            cut = max(cut, idx + len(marker))
    if cut > 0:
        text = text[cut:].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    skip_pfx = ("we need to", "the question", "the context", "use only", "looking at")
    kept = [ln for ln in lines if not ln.lower().startswith(skip_pfx)]
    if kept:
        text = " ".join(kept)
    return " ".join(text.split())


def llm_enabled() -> bool:
    """LLM card on/off. LLM_FALLBACK=0 skips Groq; still need GROQ_API_KEY when on."""
    flag = (_env("LLM_FALLBACK") or "1").lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(_env("GROQ_API_KEY"))


def shorten_answer(text: str, limit: int = 220) -> str:
    """Keep UI answers brief even when falling back to extractive text."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:।")
    return (cut or text[:limit]) + "…"


def _build_payload(question: str, chunks: list[RetrievedChunk], *, stream: bool) -> dict:
    model = _env("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    llm_chunks = llm_context_chunks(chunks, question=question)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "One short grounded sentence in English; Hindi only if the question is clearly Hindi.",
            },
            {"role": "user", "content": _grounded_prompt(question, llm_chunks)},
        ],
        "temperature": 0.1,
        "max_tokens": _env_int("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS),
        "stream": stream,
    }
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"
    return payload


async def _client(timeout_s: float) -> httpx.AsyncClient:
    """Reuse one client so TLS/DNS is not paid on every LLM call."""
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=8.0),
            headers={"Content-Type": "application/json"},
        )
    return _http


async def generate_answer(question: str, chunks: list[RetrievedChunk], timeout_s: float = 12.0) -> str:
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")

    payload = _build_payload(question, chunks, stream=False)
    headers = {"Authorization": f"Bearer {api_key}"}
    client = await _client(timeout_s)
    resp = await client.post(GROQ_URL, headers=headers, json=payload)
    if resp.status_code >= 400 and "reasoning_effort" in payload:
        payload.pop("reasoning_effort", None)
        resp = await client.post(GROQ_URL, headers=headers, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Groq response did not include choices.")

    message = choices[0].get("message") or {}
    content = str(message.get("content") or "").strip()
    if not content:
        content = _clean_completion(str(message.get("reasoning") or ""))
    else:
        content = _clean_completion(content)
    return shorten_answer(content, 240)


async def generate_answer_stream(
    question: str,
    chunks: list[RetrievedChunk],
    timeout_s: float = 12.0,
):
    """Yield raw token deltas from Groq (OpenAI-compatible SSE)."""
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")

    payload = _build_payload(question, chunks, stream=True)
    headers = {"Authorization": f"Bearer {api_key}"}
    client = await _client(timeout_s)

    async def _iter(active_payload: dict):
        async with client.stream("POST", GROQ_URL, headers=headers, json=active_payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"Groq HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield str(piece)

    try:
        async for piece in _iter(payload):
            yield piece
    except RuntimeError:
        if "reasoning_effort" in payload:
            payload.pop("reasoning_effort", None)
            async for piece in _iter(payload):
                yield piece
        else:
            raise
