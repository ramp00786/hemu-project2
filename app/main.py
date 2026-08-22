from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.harness import run_audio_pipeline, run_text_pipeline, stream_text_pipeline
from app.latency_report import build_report
from app.query_cache import query_cache
from app.retrieve import RetrievalEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
JSONL_DIR = PROJECT_ROOT / "one-time" / "download_dataset" / "data"
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="Voice RAG App", version="0.1.0")

# Local dev convenience. Production same-origin still works.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever: RetrievalEngine | None = None
sample_questions: list[dict] = []


class AskTextRequest(BaseModel):
    question: str


def _query_text_for_ids(needed: set[int]) -> dict[int, dict]:
    """Resolve Hindi/English questions for query_ids that appear in chunks.json."""
    lookup: dict[int, dict] = {}
    if not needed:
        return lookup

    sample_path = PROJECT_ROOT / "indexes" / "sample_queries.json"
    if sample_path.exists():
        for row in json.loads(sample_path.read_text(encoding="utf-8")):
            qid = row.get("query_id")
            if qid is None:
                continue
            qid = int(qid)
            if qid not in needed:
                continue
            lookup[qid] = {
                "query_hi": (row.get("query_hi") or "").strip(),
                "query_en": (row.get("query_en") or "").strip(),
            }

    missing = needed - set(lookup)
    if not missing or not JSONL_DIR.is_dir():
        return lookup
    files = sorted(JSONL_DIR.glob("msmarco_xi_hi_*.jsonl"))
    if not files:
        return lookup
    with files[-1].open(encoding="utf-8") as fh:
        for line in fh:
            if not missing:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("query_id")
            if qid is None:
                continue
            qid = int(qid)
            if qid not in missing:
                continue
            lookup[qid] = {
                "query_hi": (row.get("query") or "").strip(),
                "query_en": (row.get("Eng_Query") or "").strip(),
            }
            missing.remove(qid)
    return lookup


def _clean_en_query(text: str) -> str:
    """Normalize English sample strings (e.g. '. what is a corporation?')."""
    return (text or "").strip().lstrip(". ").strip()


def _chunk_id_by_query(chunks: list[dict]) -> dict[int, str | None]:
    """Map query_id -> chunk id for selected English passage_full rows."""
    lookup: dict[int, str | None] = {}
    for item in chunks:
        if item.get("strategy") != "passage_full" or item.get("language") != "en":
            continue
        try:
            selected = int(item.get("is_selected") or 0)
        except (TypeError, ValueError):
            selected = 0
        if selected != 1:
            continue
        qid = item.get("query_id")
        if qid is None:
            continue
        qid = int(qid)
        if qid not in lookup:
            lookup[qid] = item.get("id")
    return lookup


def _build_sample_questions(chunks: list[dict], limit: int = 18) -> list[dict]:
    """Sample questions for UI and latency bench.

    Prefer indexes/sample_queries.json (shipped in git) so clone setups work
    without the offline JSONL slice. Supplement from chunks + JSONL only if needed.
    """
    sample_path = PROJECT_ROOT / "indexes" / "sample_queries.json"
    chunk_ids = _chunk_id_by_query(chunks)
    out: list[dict] = []
    seen: set[int] = set()

    if sample_path.exists():
        for row in json.loads(sample_path.read_text(encoding="utf-8")):
            if len(out) >= limit:
                break
            qid = row.get("query_id")
            if qid is None:
                continue
            qid = int(qid)
            if qid in seen:
                continue
            query_en = _clean_en_query(row.get("query_en") or "")
            query_hi = (row.get("query_hi") or "").strip()
            question = query_en or query_hi
            if not question:
                continue
            seen.add(qid)
            out.append(
                {
                    "query_id": qid,
                    "question": question,
                    "query_hi": query_hi,
                    "query_en": query_en,
                    "query_type": "",
                    "chunk_id": chunk_ids.get(qid),
                }
            )

    if len(out) >= limit:
        return out[:limit]

    # Extra rows when sample_queries.json is short: scan index + optional JSONL text.
    qids: dict[int, dict] = {row["query_id"]: row for row in out}
    for item in chunks:
        if len(qids) >= limit:
            break
        if item.get("strategy") != "passage_full" or item.get("language") != "en":
            continue
        try:
            selected = int(item.get("is_selected") or 0)
        except (TypeError, ValueError):
            selected = 0
        if selected != 1:
            continue
        qid = item.get("query_id")
        if qid is None:
            continue
        qid = int(qid)
        if qid in qids:
            continue
        qids[qid] = {
            "query_id": qid,
            "query_type": item.get("query_type") or "",
            "chunk_id": item.get("id"),
        }

    texts = _query_text_for_ids(set(qids) - seen)
    for qid, meta in qids.items():
        if qid in seen:
            continue
        row = texts.get(qid) or {}
        query_en = _clean_en_query(row.get("query_en") or "")
        query_hi = (row.get("query_hi") or "").strip()
        question = query_en or query_hi
        if not question:
            continue
        seen.add(qid)
        out.append(
            {
                "query_id": qid,
                "question": question,
                "query_hi": query_hi,
                "query_en": query_en,
                "query_type": meta.get("query_type") or "",
                "chunk_id": meta.get("chunk_id"),
            }
        )
        if len(out) >= limit:
            break

    return out[:limit]


def _result_payload(result) -> dict:
    return {
        "answer": result.answer,
        "refused": result.refused,
        "reason": result.reason,
        "transcript": result.transcript,
        "stt_ms": result.stt_ms,
        "retrieve_ms": result.retrieve_ms,
        "generate_ms": result.generate_ms,
        "latency_ms": result.latency_ms,
        "path": result.path,
        "extractive": result.extractive or None,
        "llm": result.llm or None,
        "chunks": result.chunks[:5],
        "cached": result.cached,
    }


@app.on_event("startup")
def _startup() -> None:
    global retriever, sample_questions
    retriever = RetrievalEngine(PROJECT_ROOT)
    sample_questions = _build_sample_questions(retriever.chunks)


@app.get("/")
def home() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_file)


@app.get("/sample-question")
def sample_question_page() -> FileResponse:
    page = STATIC_DIR / "sample-question.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="sample-question.html not found")
    return FileResponse(page)


@app.get("/latency-report")
def latency_report_page() -> FileResponse:
    page = STATIC_DIR / "latency-report.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="latency-report.html not found")
    return FileResponse(page)


def _latency_report_response(max_queries: int, *, force: bool) -> dict:
    base_url = "http://127.0.0.1:8000"
    try:
        report, was_cached = build_report(base_url=base_url, max_queries=max_queries, force=force)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "report_cached": was_cached and not force, "report": report}


@app.get("/api/latency-report")
def api_latency_report(max_queries: int = Query(default=100, ge=1, le=100)) -> dict:
    return _latency_report_response(max_queries, force=False)


@app.post("/api/latency-report/recalculate")
def api_latency_report_recalculate(max_queries: int = Query(default=100, ge=1, le=100)) -> dict:
    return _latency_report_response(max_queries, force=True)


@app.get("/api/sample-questions")
def api_sample_questions(limit: int = Query(default=18, ge=1, le=100)) -> dict:
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever unavailable")
    rows = _build_sample_questions(retriever.chunks, limit=limit)
    return {"ok": True, "count": len(rows), "questions": rows}


@app.get("/health")
def health() -> dict:
    if retriever is None:
        return {"ok": False, "detail": "Retriever not loaded"}
    return {
        "ok": True,
        "embedding_model": retriever.embedding_model_name,
        "embed_device": retriever.embed_device,
        "num_chunks": len(retriever.chunks),
        "query_cache": query_cache.stats(),
    }


@app.post("/ask-text")
async def ask_text(payload: AskTextRequest) -> dict:
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever unavailable")

    try:
        result = await run_text_pipeline(payload.question, retriever)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _result_payload(result)


@app.post("/ask-text-stream")
async def ask_text_stream(payload: AskTextRequest):
    """NDJSON stream: extractive first, then llm_delta tokens, then llm/done."""
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever unavailable")

    async def event_gen():
        try:
            async for event in stream_text_pipeline(payload.question, retriever):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_gen(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ask")
async def ask_audio(audio: UploadFile = File(...)) -> dict:
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever unavailable")

    filename = audio.filename or "audio.webm"
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        result = await run_audio_pipeline(data, filename, retriever)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _result_payload(result)
