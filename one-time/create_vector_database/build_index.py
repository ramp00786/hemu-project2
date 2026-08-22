"""Build FAISS + chunks.json from the Hindi JSONL produced by download.py.

Offline only. Query-time retrieval lives in app/ and must use the SAME
embedding model name written into indexes/index_meta.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chunking import chunks_for_passage

# Must match Person B's query-time encoder.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HERE = Path(__file__).resolve().parent
PROJECT2 = HERE.parent.parent
DOWNLOAD_DATA = HERE.parent / "download_dataset" / "data"
INDEX_DIR = PROJECT2 / "indexes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk MSMARCO-XI JSONL, embed, and write FAISS under indexes/."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSONL from download.py. Default: newest file in download_dataset/data/.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Use only the first N JSONL rows (0 = all). Useful for a trial index.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=INDEX_DIR,
        help="Where to write index.faiss, chunks.json, index_meta.json.",
    )
    return parser.parse_args()


def find_default_jsonl() -> Path | None:
    """Pick the newest msmarco_xi_hi_*.jsonl from the download step."""
    if not DOWNLOAD_DATA.is_dir():
        return None
    files = sorted(
        DOWNLOAD_DATA.glob("msmarco_xi_hi_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def load_jsonl(path: Path, max_rows: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def iter_passages(row: dict):
    """Yield (language, passage_index, text, is_selected) for Hindi and English."""
    selected = row.get("is_selected") or []
    pairs = (
        ("hi", row.get("Translated_passages") or []),
        ("en", row.get("English_passages") or []),
    )
    for lang, passages in pairs:
        for idx, text in enumerate(passages):
            if not text or not str(text).strip():
                continue
            flag = selected[idx] if idx < len(selected) else 0
            try:
                flag_i = int(flag)
            except (TypeError, ValueError):
                flag_i = 0
            yield lang, idx, str(text), flag_i


def build_chunk_records(rows: list[dict]) -> list[dict]:
    """Flatten dataset rows into chunk dicts with stable integer ids."""
    records: list[dict] = []
    chunk_id = 0
    for row in rows:
        query_id = row.get("query_id")
        query_type = row.get("query_type")
        for lang, p_idx, text, is_selected in iter_passages(row):
            for piece, extra in chunks_for_passage(text):
                rec = {
                    "id": chunk_id,
                    "text": piece,
                    "language": lang,
                    "query_id": query_id,
                    "query_type": query_type,
                    "passage_index": p_idx,
                    "is_selected": is_selected,
                    "source": "msmarco-xi-hi",
                }
                rec.update(extra)
                records.append(rec)
                chunk_id += 1
    return records


def embed_texts(texts: list[str]):
    """Encode with MiniLM; L2-normalize so FAISS inner product = cosine."""
    import os

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    pref = (os.getenv("EMBED_DEVICE") or "auto").lower()
    if pref == "cpu":
        device = "cpu"
    elif pref.startswith("cuda"):
        device = pref
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading embedding model {EMBED_MODEL} on {device}...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32")


def write_faiss(vectors, out_path: Path) -> None:
    import faiss

    dim = vectors.shape[1]
    # Inner-product index on normalized vectors = cosine nearest neighbour.
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(out_path))


def write_sample_queries(rows: list[dict], out_path: Path, n: int = 15) -> None:
    """A few dataset questions Person B can use to sanity-check retrieval."""
    samples = []
    for row in rows:
        if len(samples) >= n:
            break
        q_hi = (row.get("query") or "").strip()
        q_en = (row.get("Eng_Query") or "").strip()
        if not q_hi and not q_en:
            continue
        samples.append(
            {
                "query_id": row.get("query_id"),
                "query_hi": q_hi,
                "query_en": q_en,
            }
        )
    out_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    src = args.input or find_default_jsonl()
    if src is None or not src.is_file():
        print(
            "No JSONL found. Run download.py first:\n"
            "  cd ../download_dataset && python download.py",
            file=sys.stderr,
        )
        return 2

    print(f"Reading {src}")
    rows = load_jsonl(src, args.max_rows)
    if not rows:
        print("JSONL is empty.", file=sys.stderr)
        return 2

    records = build_chunk_records(rows)
    if not records:
        print("No chunks produced. Check Translated_passages / English_passages.", file=sys.stderr)
        return 2

    strategies = sorted({r.get("strategy") for r in records})
    print(
        f"Rows={len(rows)} chunks={len(records)} strategies={strategies} "
        f"hi={sum(1 for r in records if r['language']=='hi')} "
        f"en={sum(1 for r in records if r['language']=='en')}"
    )

    try:
        vectors = embed_texts([r["text"] for r in records])
        import faiss  # noqa: F401 — imported here so missing dep fails with a clear path
    except ImportError:
        print("Install deps: pip install -r requirements.txt", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = args.out_dir / "index.faiss"
    chunks_path = args.out_dir / "chunks.json"
    meta_path = args.out_dir / "index_meta.json"
    sample_path = args.out_dir / "sample_queries.json"

    write_faiss(vectors, faiss_path)
    chunks_path.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = {
        "embedding_model": EMBED_MODEL,
        "metric": "cosine",
        "faiss_index": "IndexFlatIP",
        "dim": int(vectors.shape[1]),
        "num_chunks": len(records),
        "num_source_rows": len(rows),
        "source_jsonl": str(src),
        "strategies": strategies,
        "languages": ["hi", "en"],
        "notes": (
            "Query encoder MUST be the same embedding_model. "
            "Normalize embeddings, then index.search(query_vec, k)."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_sample_queries(rows, sample_path)

    print(f"Wrote {faiss_path}")
    print(f"Wrote {chunks_path} ({len(records)} chunks)")
    print(f"Wrote {meta_path}")
    print(f"Wrote {sample_path} (test questions for Person B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
