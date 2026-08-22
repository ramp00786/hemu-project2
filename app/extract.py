"""Extractive answerer: grounded 1-2 sentences from retrieved chunks.

This is the hosted latency path. No remote LLM call. Answers are copied
from retrieved text (slightly trimmed), so they stay grounded by construction.
"""

from __future__ import annotations

import re

from app.retrieve import RetrievedChunk

_SENT_RE = re.compile(r"(?<=[.!?।])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u0900-\u097F]{2,}")

STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "what",
        "which",
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
        "था",
        "थे",
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
    }
)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _content_tokens(text: str) -> set[str]:
    toks = _tokens(text)
    kept = {t for t in toks if t not in STOP}
    return kept or set(toks)


def split_sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    return parts or [text]


def _cap(text: str, limit: int = 260) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "…"


def extract_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, float]:
    """Return (answer, confidence). Confidence is 0..~1.2, not a probability."""
    if not chunks:
        return "", 0.0

    q_toks = _content_tokens(question)
    scored: list[tuple[float, str]] = []

    for rank, chunk in enumerate(chunks):
        sents = split_sentences(chunk.text)
        for i, sent in enumerate(sents):
            if len(sent) < 25:
                continue
            overlap = len(q_toks & _content_tokens(sent)) / max(1, len(q_toks))
            score = overlap
            score += 0.08 * max(0.0, min(1.0, float(chunk.score)))
            if chunk.metadata.get("strategy") == "passage_full":
                score += 0.04
            if chunk.metadata.get("is_selected") in (1, True, "1"):
                score += 0.05
            if rank == 0 and i < 2:
                score += 0.12
            low = sent.lower()
            if any(cue in low for cue in (" is ", " are ", " means ", " definition", " है", " हैं", " मतलब")):
                score += 0.03
            scored.append((score, sent))

    if not scored:
        snippet = _cap(chunks[0].text.strip(), 420)
        return snippet, 0.18 if snippet else 0.0

    scored.sort(key=lambda row: row[0], reverse=True)
    picked: list[str] = []
    seen: set[str] = set()
    best_conf = scored[0][0]
    for score, sent in scored:
        key = sent[:90]
        if key in seen:
            continue
        seen.add(key)
        picked.append(sent)
        if len(picked) >= 2:
            break

    return _cap(" ".join(picked), 260), float(best_conf)
