from __future__ import annotations

import re
from collections import Counter

from app.retrieve import RetrievedChunk

UNSAFE_PATTERNS = [
    r"\bkill\b",
    r"\bsuicide\b",
    r"\bbomb\b",
    r"\bterror\b",
]


def input_guard(question: str) -> tuple[bool, str | None]:
    q = (question or "").strip()
    if not q:
        return False, "Question is empty."
    lowered = q.lower()
    for pat in UNSAFE_PATTERNS:
        if re.search(pat, lowered):
            return False, "I cannot help with unsafe requests."
    return True, None


def is_off_topic(chunks: list[RetrievedChunk], min_score: float = 0.35) -> bool:
    if not chunks:
        return True
    best = max(c.score for c in chunks)
    return best < min_score


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\u0900-\u097F]+", (text or "").lower())


def output_guard(answer: str, chunks: list[RetrievedChunk]) -> tuple[bool, str | None]:
    a = (answer or "").strip()
    if not a:
        return False, "I could not produce a grounded answer."

    compact = re.sub(r"\s+", " ", a).strip().lower().strip(" .")
    if compact in {"i don't know", "i do not know", "no_answer", "not enough context"}:
        return False, "I do not have enough grounded context to answer."
    # Model sometimes narrates the prompt instead of answering.
    if compact.startswith("passage ") or "does not provide" in compact or "passages do not" in compact:
        return False, "I do not have enough grounded context to answer."

    # Lightweight grounding check: answer tokens should overlap retrieved context.
    answer_tokens = _tokenize(a)
    if not answer_tokens:
        return False, "I could not produce a grounded answer."

    context_text = " ".join(c.text for c in chunks)
    context_tokens = set(_tokenize(context_text))

    freq = Counter(answer_tokens)
    kept = [t for t, n in freq.items() for _ in range(n) if t in context_tokens]
    overlap_ratio = len(kept) / max(1, len(answer_tokens))

    # Paraphrased LLM answers (especially EN over HI passages) overlap less.
    if overlap_ratio < 0.08:
        return False, "The generated response appears ungrounded in retrieved context."

    return True, None
