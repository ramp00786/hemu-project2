"""Chunking strategies for MSMARCO-XI passages.

The brief requires more than one naive fixed-size split. Each function
returns a list of (text, extra_metadata) pairs. Metadata always includes
`strategy` so retrieval can tell how a chunk was produced.
"""

from __future__ import annotations

import re

# Sentence-ish split: Latin punctuation plus Devanagari danda.
_SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+")

# Character windows (not tokens) — fast, language-agnostic, with overlap.
FIXED_SIZE = 400
FIXED_OVERLAP = 80

# Group a few sentences so chunks are not tiny fragments.
SENTENCES_PER_CHUNK = 2


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation. Falls back to the whole string."""
    text = " ".join(text.split())
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    return parts or [text]


def chunk_passage_full(text: str) -> list[tuple[str, dict]]:
    """Strategy 1 — metadata-aware unit: one MSMARCO passage = one chunk."""
    text = text.strip()
    if not text:
        return []
    return [(text, {"strategy": "passage_full"})]


def chunk_fixed_overlap(
    text: str,
    size: int = FIXED_SIZE,
    overlap: int = FIXED_OVERLAP,
) -> list[tuple[str, dict]]:
    """Strategy 2 — fixed-size character windows with overlap.

    Skip if the passage already fits in one window (passage_full covers that).
    """
    text = " ".join(text.strip().split())
    if len(text) <= size:
        return []
    step = max(1, size - overlap)
    out: list[tuple[str, dict]] = []
    start = 0
    window_i = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            out.append(
                (
                    piece,
                    {
                        "strategy": "fixed_overlap",
                        "window_index": window_i,
                        "char_start": start,
                        "char_end": start + len(piece),
                    },
                )
            )
            window_i += 1
        if start + size >= len(text):
            break
        start += step
    return out


def chunk_sentence_group(
    text: str,
    n: int = SENTENCES_PER_CHUNK,
) -> list[tuple[str, dict]]:
    """Strategy 3 — semantic-ish: consecutive sentences grouped together.

    Not a neural semantic splitter (keeps this step CPU-fast). Still not a
    single naive character window.
    """
    sents = split_sentences(text)
    if len(sents) <= 1:
        return []
    out: list[tuple[str, dict]] = []
    for i in range(0, len(sents), n):
        group = sents[i : i + n]
        piece = " ".join(group).strip()
        if piece:
            out.append(
                (
                    piece,
                    {
                        "strategy": "sentence_group",
                        "sentence_start": i,
                        "sentence_end": i + len(group) - 1,
                    },
                )
            )
    return out


def chunks_for_passage(text: str) -> list[tuple[str, dict]]:
    """Apply all strategies to one passage and drop empty / duplicate texts."""
    seen: set[str] = set()
    combined: list[tuple[str, dict]] = []
    for fn in (chunk_passage_full, chunk_fixed_overlap, chunk_sentence_group):
        for piece, meta in fn(text):
            key = piece.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            combined.append((key, meta))
    return combined
