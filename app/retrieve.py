from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u0900-\u097F]{2,}")
# Queries/passages often disagree on nukta (बाज़ vs बाज).
_NUKTA_RE = re.compile("\u093c")
_LEX_STOP = frozenset(
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
        "कितनी",
        "कितना",
        "कितने",
        "करता",
        "करती",
        "करते",
    }
)


def _norm_text(text: str) -> str:
    import unicodedata

    return _NUKTA_RE.sub("", unicodedata.normalize("NFC", text or ""))


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_norm_text(text).lower())


def _env(name: str) -> str:
    raw = os.getenv(name, "") or ""
    return raw.split("#", 1)[0].strip().strip('"').strip("'")


def _embedding_device() -> str:
    """Pick query-encoder device: auto uses CUDA when available."""
    pref = (_env("EMBED_DEVICE") or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref.startswith("cuda"):
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _query_script_bias(question: str) -> str | None:
    """Return preferred passage language: en, hi, or None (neutral)."""
    text = question or ""
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if devanagari >= 3 and devanagari >= latin:
        return "hi"
    if latin >= 3 and latin > devanagari:
        return "en"
    return None


@dataclass
class RetrievedChunk:
    chunk_id: int
    score: float
    text: str
    metadata: dict


class RetrievalEngine:
    """Loads FAISS + chunks once and serves top-k retrieval."""

    def __init__(self, project_root: Path):
        self.index_dir = project_root / "indexes"
        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.json"
        self.meta_path = self.index_dir / "index_meta.json"

        if not self.index_path.exists() or not self.chunks_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                "Missing index files. Expected indexes/index.faiss, indexes/chunks.json, indexes/index_meta.json"
            )

        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.embedding_model_name = self.meta["embedding_model"]
        self.embed_device = _embedding_device()

        self.index = faiss.read_index(str(self.index_path))
        self.chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Index/chunks mismatch: index.ntotal={self.index.ntotal} vs chunks={len(self.chunks)}"
            )

        self.encoder = SentenceTransformer(self.embedding_model_name, device=self.embed_device)
        logger.info("Embedding device: %s (model=%s)", self.embed_device, self.embedding_model_name)
        self._inv: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(self.chunks):
            if item.get("strategy") != "passage_full":
                continue
            for tok in set(_tokens(item.get("text", ""))):
                if tok in _LEX_STOP:
                    continue
                self._inv[tok].append(i)
        self._configure_runtime()
        self._warmup()

    @staticmethod
    def _configure_runtime() -> None:
        # Small single-query encodes oversubscribe if every CPU thread wakes up.
        try:
            import torch

            torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        except Exception:
            pass
        try:
            faiss.omp_set_num_threads(4)
        except Exception:
            pass

    def _warmup(self) -> None:
        """Move first-query PyTorch/FAISS kernel cost off the user request."""
        self.search("what is a corporation?", k=5)

    def _from_item(self, idx: int, score: float) -> RetrievedChunk:
        item = self.chunks[idx]
        return RetrievedChunk(
            chunk_id=int(item.get("id", idx)),
            score=float(score),
            text=item.get("text", ""),
            metadata={k2: v for k2, v in item.items() if k2 not in {"id", "text"}},
        )

    def _lexical_candidates(self, question: str, limit: int = 40) -> list[RetrievedChunk]:
        q_tokens = [t for t in dict.fromkeys(_tokens(question)) if t not in _LEX_STOP]
        if not q_tokens:
            q_tokens = list(dict.fromkeys(_tokens(question)))
        if not q_tokens:
            return []

        # IDF so rare terms (बाज़) beat common ones (तेजी / यात्रा).
        n_docs = max(1, sum(1 for c in self.chunks if c.get("strategy") == "passage_full"))
        idf = {
            tok: float(np.log((n_docs + 1) / (len(self._inv.get(tok, ())) + 1)) + 1.0)
            for tok in q_tokens
        }
        weight_denom = sum(idf.values()) or 1.0

        weighted: dict[int, float] = defaultdict(float)
        hit_counts: dict[int, int] = defaultdict(int)
        for tok in q_tokens:
            w = idf[tok]
            for idx in self._inv.get(tok, ()):
                weighted[idx] += w
                hit_counts[idx] += 1
        if not weighted:
            return []

        ranked = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        max_q_idf = max(idf.values()) if idf else 1.0
        out: list[RetrievedChunk] = []
        for idx, wsum in ranked:
            coverage = wsum / weight_denom
            # Which matched tokens? Recompute max matched idf for this doc.
            item_text_toks = set(_tokens(self.chunks[idx].get("text", "")))
            matched = [t for t in q_tokens if t in item_text_toks]
            max_matched = max((idf[t] for t in matched), default=0.0)
            rare_ratio = max_matched / max_q_idf
            # Rare entity hits (बाज़) must beat piles of common terms (तेजी/यात्रा).
            lexical_score = 0.42 + 0.28 * coverage + 0.38 * rare_ratio
            if hit_counts[idx] >= 2:
                lexical_score += 0.02
            out.append(self._from_item(idx, min(0.99, lexical_score)))
        return out

    def search(self, question: str, k: int = 5) -> list[RetrievedChunk]:
        vec = self.encoder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vec = np.asarray(vec, dtype="float32")

        # Over-fetch then rerank: MiniLM on Hindi often ranks short
        # sentence_group fragments above the actual passage_full text.
        fetch_k = max(k * 12, 48)
        scores, idxs = self.index.search(vec, fetch_k)
        candidates: list[RetrievedChunk] = []
        seen_ids: set[int] = set()
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            candidates.append(self._from_item(int(idx), float(score)))
            seen_ids.add(int(idx))
        for lex in self._lexical_candidates(question):
            if lex.chunk_id in seen_ids:
                # Keep the stronger of vector vs lexical.
                for i, c in enumerate(candidates):
                    if c.chunk_id == lex.chunk_id:
                        if lex.score > c.score:
                            candidates[i] = lex
                        break
            else:
                candidates.append(lex)
                seen_ids.add(lex.chunk_id)
        return self._rerank(candidates, k, question=question)

    @staticmethod
    def _rerank(
        candidates: list[RetrievedChunk],
        k: int,
        question: str = "",
    ) -> list[RetrievedChunk]:
        q_content = {
            t
            for t in _tokens(question)
            if t not in _LEX_STOP and len(t) >= 2
        }
        lang_bias = _query_script_bias(question)
        seen: set[tuple] = set()
        scored: list[tuple[float, RetrievedChunk]] = []
        for c in candidates:
            text = (c.text or "").strip()
            if len(text) < 40:
                continue
            key = (
                c.metadata.get("query_id"),
                c.metadata.get("language"),
                c.metadata.get("passage_index"),
                c.metadata.get("strategy"),
            )
            if key in seen:
                continue
            seen.add(key)

            boost = 0.0
            strategy = c.metadata.get("strategy")
            if strategy == "passage_full":
                boost += 0.12
            elif strategy == "fixed_overlap":
                boost += 0.04
            if c.metadata.get("is_selected") in (1, True, "1"):
                boost += 0.06
            if len(text) >= 220:
                boost += 0.04
            if len(text) < 80:
                boost -= 0.10

            chunk_lang = c.metadata.get("language")
            if lang_bias and chunk_lang == lang_bias:
                boost += 0.06

            if q_content:
                text_toks = set(_tokens(text))
                overlap = q_content & text_toks
                if overlap:
                    boost += 0.08 * (len(overlap) / max(1, len(q_content)))
                    # Distinctive single-term hits (e.g. entity name) still matter.
                    if any(len(t) >= 3 for t in overlap):
                        boost += 0.04

            scored.append((c.score + boost, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]
