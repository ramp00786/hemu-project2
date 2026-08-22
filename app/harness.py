from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from app.extract import extract_answer
from app.generate import (
    _clean_completion,
    generate_answer,
    generate_answer_stream,
    llm_context_chunks,
    llm_enabled,
    shorten_answer,
)
from app.guardrails import input_guard, is_off_topic, output_guard
from app.query_cache import query_cache
from app.retrieve import RetrievalEngine
from app.stt_sarvam import transcribe_audio


def _arm(answer: str, refused: bool, reason: str | None, **extra) -> dict:
    payload = {"answer": answer, "refused": refused, "reason": reason}
    payload.update(extra)
    return payload


@dataclass
class HarnessResult:
    answer: str
    refused: bool
    reason: str | None
    transcript: str
    stt_ms: int
    retrieve_ms: int
    generate_ms: int
    latency_ms: int
    path: str
    chunks: list[dict]
    extractive: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    cached: bool = False


def _cache_payload(
    *,
    answer: str,
    refused: bool,
    reason: str | None,
    path: str,
    chunks: list[dict],
    extractive: dict,
    llm: dict,
) -> dict:
    return {
        "answer": answer,
        "refused": refused,
        "reason": reason,
        "path": path,
        "chunks": chunks,
        "extractive": extractive,
        "llm": llm,
    }


def _result_from_cache(transcript: str, entry: dict, *, lookup_ms: int = 1) -> HarnessResult:
    return HarnessResult(
        answer=entry["answer"],
        refused=entry["refused"],
        reason=entry.get("reason"),
        transcript=transcript,
        stt_ms=0,
        retrieve_ms=0,
        generate_ms=0,
        latency_ms=lookup_ms,
        path=entry["path"],
        chunks=entry.get("chunks") or [],
        extractive=entry.get("extractive") or {},
        llm=entry.get("llm") or {},
        cached=True,
    )


def _store_cache(transcript: str, result: HarnessResult) -> None:
    query_cache.set(
        transcript,
        _cache_payload(
            answer=result.answer,
            refused=result.refused,
            reason=result.reason,
            path=result.path,
            chunks=result.chunks,
            extractive=result.extractive,
            llm=result.llm,
        ),
    )


async def _retry_once(coro_factory):
    try:
        return await coro_factory()
    except Exception:
        return await coro_factory()


def _to_chunk_payload(chunks):
    return [
        {
            "id": c.chunk_id,
            "score": c.score,
            "language": c.metadata.get("language"),
            "strategy": c.metadata.get("strategy"),
            "text": c.text,
        }
        for c in chunks
    ]


def _blocked(
    *,
    answer: str,
    reason: str | None,
    transcript: str,
    latency_ms: int,
    retrieve_ms: int = 0,
    generate_ms: int = 0,
    chunks: list | None = None,
    path: str = "refused",
) -> HarnessResult:
    arm = _arm(answer, True, reason, latency_ms=latency_ms, generate_ms=generate_ms)
    return HarnessResult(
        answer=answer,
        refused=True,
        reason=reason,
        transcript=transcript,
        stt_ms=0,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        latency_ms=latency_ms,
        path=path,
        chunks=chunks or [],
        extractive=dict(arm),
        llm=dict(arm),
    )


def _is_no_answer(text: str) -> bool:
    compact = " ".join((text or "").split()).strip().lower().strip(" .")
    return compact in {
        "no_answer",
        "i don't know",
        "i do not know",
        "not enough context",
    } or compact.startswith("passage ")


async def _llm_arm(transcript: str, retrieved) -> dict:
    """Groq generation on a small top-k slice of retrieved chunks. Never raises."""
    if not llm_enabled():
        return _arm("LLM card is disabled.", True, "llm_disabled", generate_ms=0)

    llm_chunks = llm_context_chunks(retrieved, question=transcript)
    gen_t0 = perf_counter()
    try:
        # No automatic retry: a second Groq round-trip roughly doubles generate_ms.
        llm_answer = await generate_answer(transcript, llm_chunks)
    except Exception:
        llm_answer = ""
    generate_ms = int((perf_counter() - gen_t0) * 1000)

    # Small models often refuse when the numeric detail is missing; keep a
    # grounded extractive snippet from the same LLM chunks instead.
    if (not llm_answer) or _is_no_answer(llm_answer):
        fallback, _conf = extract_answer(transcript, llm_chunks)
        if fallback:
            pass_fb, _ = output_guard(fallback, llm_chunks)
            if pass_fb:
                return _arm(
                    shorten_answer(fallback, 220),
                    False,
                    "llm_extractive_fallback",
                    generate_ms=generate_ms,
                )
        if not llm_answer:
            return _arm(
                "The language model is temporarily unavailable.",
                True,
                "llm_unavailable",
                generate_ms=generate_ms,
            )

    pass_llm, llm_reason = output_guard(llm_answer, llm_chunks)
    if not pass_llm:
        fallback, _conf = extract_answer(transcript, llm_chunks)
        if fallback:
            pass_fb, _ = output_guard(fallback, llm_chunks)
            if pass_fb:
                return _arm(
                    shorten_answer(fallback, 220),
                    False,
                    "llm_extractive_fallback",
                    generate_ms=generate_ms,
                )
        return _arm(
            llm_reason or "I cannot provide a grounded LLM answer.",
            True,
            llm_reason,
            generate_ms=generate_ms,
        )
    return _arm(shorten_answer(llm_answer, 240), False, None, generate_ms=generate_ms)


async def run_text_pipeline(question: str, retriever: RetrievalEngine, top_k: int = 5) -> HarnessResult:
    transcript = (question or "").strip()
    ok, reason = input_guard(transcript)
    if not ok:
        return _blocked(
            answer=reason or "Request blocked.",
            reason=reason,
            transcript=transcript,
            latency_ms=0,
            path="refused",
        )

    cache_t0 = perf_counter()
    cached_entry = query_cache.get(transcript)
    if cached_entry is not None:
        lookup_ms = max(1, int((perf_counter() - cache_t0) * 1000))
        return _result_from_cache(transcript, cached_entry, lookup_ms=lookup_ms)

    t0 = perf_counter()
    retrieved = retriever.search(transcript, k=top_k)
    retrieve_ms = int((perf_counter() - t0) * 1000)

    if is_off_topic(retrieved):
        latency = int((perf_counter() - t0) * 1000)
        result = _blocked(
            answer="I do not have enough relevant context for this question.",
            reason="off_topic_or_low_relevance",
            transcript=transcript,
            latency_ms=latency,
            retrieve_ms=retrieve_ms,
            chunks=_to_chunk_payload(retrieved),
            path="refused",
        )
        _store_cache(transcript, result)
        return result

    extracted, _conf = extract_answer(transcript, retrieved)
    pass_ext, ext_reason = output_guard(extracted, retrieved) if extracted else (False, "empty extractive")
    extractive_ok = bool(extracted) and pass_ext

    latency_ms = int((perf_counter() - t0) * 1000)

    if extractive_ok:
        extractive = _arm(extracted, False, None, latency_ms=latency_ms)
        primary_answer = extracted
        primary_refused = False
        primary_reason = None
        path = "extractive"
    else:
        msg = ext_reason or "I cannot provide a grounded answer."
        extractive = _arm(msg, True, ext_reason, latency_ms=latency_ms)
        primary_answer = msg
        primary_refused = True
        primary_reason = ext_reason
        path = "refused"

    llm = await _llm_arm(transcript, retrieved)
    generate_ms = int(llm.get("generate_ms") or 0)

    result = HarnessResult(
        answer=primary_answer,
        refused=primary_refused,
        reason=primary_reason,
        transcript=transcript,
        stt_ms=0,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        latency_ms=latency_ms,
        path=path,
        chunks=_to_chunk_payload(retrieved),
        extractive=extractive,
        llm=llm,
    )
    _store_cache(transcript, result)
    return result


async def stream_text_pipeline(question: str, retriever: RetrievalEngine, top_k: int = 5):
    """Yield NDJSON-ready dict events: extractive first, then llm_delta*, then llm/done."""
    transcript = (question or "").strip()
    ok, reason = input_guard(transcript)
    if not ok:
        yield {
            "type": "error",
            "error": reason or "Request blocked.",
            "refused": True,
        }
        return

    cache_t0 = perf_counter()
    cached_entry = query_cache.get(transcript)
    if cached_entry is not None:
        lookup_ms = max(1, int((perf_counter() - cache_t0) * 1000))
        extractive = cached_entry.get("extractive") or {}
        llm = cached_entry.get("llm") or {}
        yield {
            "type": "extractive",
            "answer": cached_entry["answer"],
            "refused": cached_entry["refused"],
            "reason": cached_entry.get("reason"),
            "retrieve_ms": 0,
            "latency_ms": lookup_ms,
            "path": cached_entry["path"],
            "chunks": cached_entry.get("chunks") or [],
            "transcript": transcript,
            "extractive": extractive,
            "cached": True,
        }
        yield {
            "type": "llm",
            "answer": llm.get("answer") or cached_entry["answer"],
            "refused": llm.get("refused", cached_entry["refused"]),
            "reason": llm.get("reason"),
            "generate_ms": 0,
            "llm": llm,
            "cached": True,
        }
        yield {
            "type": "done",
            "generate_ms": 0,
            "retrieve_ms": 0,
            "latency_ms": lookup_ms,
            "path": cached_entry["path"],
            "cached": True,
        }
        return

    t0 = perf_counter()
    retrieved = retriever.search(transcript, k=top_k)
    retrieve_ms = int((perf_counter() - t0) * 1000)
    chunk_payload = _to_chunk_payload(retrieved)

    if is_off_topic(retrieved):
        latency = int((perf_counter() - t0) * 1000)
        msg = "I do not have enough relevant context for this question."
        extractive = _arm(msg, True, "off_topic_or_low_relevance", latency_ms=latency)
        llm = _arm(msg, True, "off_topic_or_low_relevance", generate_ms=0)
        yield {
            "type": "extractive",
            "answer": msg,
            "refused": True,
            "reason": "off_topic_or_low_relevance",
            "retrieve_ms": retrieve_ms,
            "latency_ms": latency,
            "path": "refused",
            "chunks": chunk_payload,
            "transcript": transcript,
        }
        yield {
            "type": "llm",
            "answer": msg,
            "refused": True,
            "reason": "off_topic_or_low_relevance",
            "generate_ms": 0,
        }
        yield {"type": "done"}
        query_cache.set(
            transcript,
            _cache_payload(
                answer=msg,
                refused=True,
                reason="off_topic_or_low_relevance",
                path="refused",
                chunks=chunk_payload,
                extractive=extractive,
                llm=llm,
            ),
        )
        return

    extracted, _conf = extract_answer(transcript, retrieved)
    pass_ext, ext_reason = output_guard(extracted, retrieved) if extracted else (False, "empty extractive")
    extractive_ok = bool(extracted) and pass_ext
    latency_ms = int((perf_counter() - t0) * 1000)

    if extractive_ok:
        extractive = _arm(extracted, False, None, latency_ms=latency_ms)
        path = "extractive"
    else:
        msg = ext_reason or "I cannot provide a grounded answer."
        extractive = _arm(msg, True, ext_reason, latency_ms=latency_ms)
        path = "refused"

    yield {
        "type": "extractive",
        "answer": extractive["answer"],
        "refused": extractive["refused"],
        "reason": extractive.get("reason"),
        "retrieve_ms": retrieve_ms,
        "latency_ms": latency_ms,
        "path": path,
        "chunks": chunk_payload,
        "transcript": transcript,
        "extractive": extractive,
    }

    if not llm_enabled():
        llm = _arm("LLM card is disabled.", True, "llm_disabled", generate_ms=0)
        yield {
            "type": "llm",
            "answer": llm["answer"],
            "refused": llm["refused"],
            "reason": llm["reason"],
            "generate_ms": 0,
        }
        yield {"type": "done", "generate_ms": 0, "retrieve_ms": retrieve_ms, "latency_ms": latency_ms}
        query_cache.set(
            transcript,
            _cache_payload(
                answer=extractive["answer"],
                refused=extractive["refused"],
                reason=extractive.get("reason"),
                path=path,
                chunks=chunk_payload,
                extractive=extractive,
                llm=llm,
            ),
        )
        return

    llm_chunks = llm_context_chunks(retrieved, question=transcript)
    gen_t0 = perf_counter()
    parts: list[str] = []
    try:
        async for delta in generate_answer_stream(transcript, llm_chunks):
            parts.append(delta)
            yield {"type": "llm_delta", "text": delta}
        llm_answer = _clean_completion("".join(parts))
    except Exception:
        llm_answer = ""
    generate_ms = int((perf_counter() - gen_t0) * 1000)

    llm = _finalize_llm(transcript, llm_chunks, llm_answer, generate_ms)
    yield {
        "type": "llm",
        "answer": llm["answer"],
        "refused": llm["refused"],
        "reason": llm.get("reason"),
        "generate_ms": generate_ms,
        "llm": llm,
    }
    yield {
        "type": "done",
        "generate_ms": generate_ms,
        "retrieve_ms": retrieve_ms,
        "latency_ms": latency_ms,
        "path": path,
    }
    query_cache.set(
        transcript,
        _cache_payload(
            answer=extractive["answer"],
            refused=extractive["refused"],
            reason=extractive.get("reason"),
            path=path,
            chunks=chunk_payload,
            extractive=extractive,
            llm=llm,
        ),
    )


def _finalize_llm(transcript: str, llm_chunks, llm_answer: str, generate_ms: int) -> dict:
    if (not llm_answer) or _is_no_answer(llm_answer):
        fallback, _conf = extract_answer(transcript, llm_chunks)
        if fallback:
            pass_fb, _ = output_guard(fallback, llm_chunks)
            if pass_fb:
                return _arm(
                    shorten_answer(fallback, 220),
                    False,
                    "llm_extractive_fallback",
                    generate_ms=generate_ms,
                )
        if not llm_answer:
            return _arm(
                "The language model is temporarily unavailable.",
                True,
                "llm_unavailable",
                generate_ms=generate_ms,
            )

    pass_llm, llm_reason = output_guard(llm_answer, llm_chunks)
    if not pass_llm:
        fallback, _conf = extract_answer(transcript, llm_chunks)
        if fallback:
            pass_fb, _ = output_guard(fallback, llm_chunks)
            if pass_fb:
                return _arm(
                    shorten_answer(fallback, 220),
                    False,
                    "llm_extractive_fallback",
                    generate_ms=generate_ms,
                )
        return _arm(
            llm_reason or "I cannot provide a grounded LLM answer.",
            True,
            llm_reason,
            generate_ms=generate_ms,
        )
    return _arm(shorten_answer(llm_answer, 240), False, None, generate_ms=generate_ms)


async def run_audio_pipeline(audio_bytes: bytes, filename: str, retriever: RetrievalEngine, top_k: int = 5) -> HarnessResult:
    stt_t0 = perf_counter()
    transcript = await _retry_once(lambda: transcribe_audio(audio_bytes, filename))
    stt_ms = int((perf_counter() - stt_t0) * 1000)

    result = await run_text_pipeline(transcript, retriever, top_k=top_k)
    result.stt_ms = stt_ms
    return result
