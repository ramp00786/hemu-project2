"""One-time download of MSMARCO-XI Hindi (`hi`) rows to local JSONL.

This script is Person A's first offline step:
  Hugging Face (`hi` only) -> normalized JSONL in ./data/

It does NOT:
  - build embeddings or FAISS
  - load other language configs (bn, ta, ...)
  - download the full 55.6 GB multilingual dump

Next step: ../create_vector_database reads the JSONL produced here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Hugging Face dataset id. The Hub builder only exposes config "default";
# Indic languages are separate parquet files (hinval / hintrain), not a "hi" config.
DATASET_ID = "ai4bharat/MSMARCO-XI"
ALLOWED_CONFIG = "hi"
ALLOWED_SPLITS = ("validation", "train")
HINDI_PARQUET = {
    "validation": "validation/hinval.parquet",
    "train": "train/hintrain.parquet",
}

# README default: start small so a laptop can inspect fields before indexing.
DEFAULT_SPLIT = "validation"
DEFAULT_LIMIT = 5000

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def parse_args() -> argparse.Namespace:
    """CLI: --config hi (required), --split, --limit (0 = entire split)."""
    parser = argparse.ArgumentParser(
        description="Download MSMARCO-XI Hindi config and write JSONL for indexing."
    )
    parser.add_argument(
        "--config",
        default=ALLOWED_CONFIG,
        help="Hugging Face config name. Only 'hi' is allowed.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=ALLOWED_SPLITS,
        help="Dataset split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max rows (Hugging Face slice). Use 0 for the whole split.",
    )
    return parser.parse_args()


def _first_present(row: dict, keys: tuple[str, ...], default=None):
    """Return the first non-null value among alternate Hugging Face field names."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _as_list(value) -> list:
    """Passages / is_selected may be a list, a single string, or missing."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def extract_row(row: dict) -> dict:
    """Normalize one HF row into a stable JSONL record.

    The dataset card is inconsistent (`Answer` vs `answers`, nested `passages`).
    Downstream indexing should only depend on the keys written here.
    """
    passages = row.get("passages") or {}
    if not isinstance(passages, dict):
        passages = {}

    # Hindi + English passage lists live under `passages`, not as top-level columns.
    hindi_passages = _as_list(
        passages.get("Translated_passages")
        or passages.get("translated_passages")
        or passages.get("passage_text")
    )
    english_passages = _as_list(
        passages.get("English_passages")
        or passages.get("english_passages")
        or passages.get("eng_passages")
    )
    # Used later for metadata-aware chunking (which passage was marked relevant).
    is_selected = _as_list(
        passages.get("is_selected") or row.get("is_selected")
    )

    answer = _first_present(row, ("Answer", "answers", "answer"), "")
    if isinstance(answer, list):
        answer = answer[0] if answer else ""

    eng_answer = _first_present(row, ("Eng_Answer", "eng_answer", "Eng_Answers"), "")
    if isinstance(eng_answer, list):
        eng_answer = eng_answer[0] if eng_answer else ""

    return {
        "query_id": row.get("query_id"),
        "query_type": row.get("query_type"),
        "query": row.get("query") or "",
        "Answer": answer or "",
        "Eng_Query": _first_present(row, ("Eng_Query", "eng_query"), "") or "",
        "Eng_Answer": eng_answer or "",
        "Translated_passages": [str(p) for p in hindi_passages if p],
        "English_passages": [str(p) for p in english_passages if p],
        "is_selected": is_selected,
        "source_lang": row.get("source_lang"),
        "target_lang": row.get("target_lang"),
    }


def split_spec(split: str, limit: int) -> str:
    """Hugging Face slice syntax, e.g. validation[:5000]. limit=0 means the full split."""
    if limit and limit > 0:
        return f"{split}[:{limit}]"
    return split


def output_path(split: str, n_rows: int) -> Path:
    """Filename includes split and actual row count so reruns do not overwrite blindly."""
    return DATA_DIR / f"msmarco_xi_hi_{split}_{n_rows}.jsonl"


def _console(text: str) -> str:
    """Windows cp1252 consoles cannot print Hindi; replace unencodable chars."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(text).encode(enc, errors="replace").decode(enc, errors="replace")


def print_sample(record: dict) -> None:
    """Print a short first-row preview so a failed extract is obvious immediately."""
    print("\n--- first row sample ---")
    print("query_id:", record.get("query_id"))
    print("Hindi query:", _console((record.get("query") or "")[:200]))
    print("English query:", _console((record.get("Eng_Query") or "")[:200]))
    hi_p = record.get("Translated_passages") or []
    en_p = record.get("English_passages") or []
    print("Hindi passage[0]:", _console(hi_p[0][:240] if hi_p else "(none)"))
    print("English passage[0]:", _console(en_p[0][:240] if en_p else "(none)"))
    print("------------------------\n")


def main() -> int:
    args = parse_args()

    # Guardrails: never pull a non-Hindi config by accident.
    if args.config != ALLOWED_CONFIG:
        print(
            f"Only config '{ALLOWED_CONFIG}' is allowed (got '{args.config}').",
            file=sys.stderr,
        )
        return 2
    if args.limit < 0:
        print("--limit must be >= 0 (0 = whole split).", file=sys.stderr)
        return 2
    if args.split == "train" and args.limit == 0:
        print(
            "Warning: loading full Hindi train with --limit 0 is large. "
            "Prefer --limit 5000 first.",
            file=sys.stderr,
        )

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install deps: pip install -r requirements.txt", file=sys.stderr)
        return 1

    spec = split_spec(args.split, args.limit)
    parquet_file = HINDI_PARQUET[args.split]
    print(
        f"Loading {DATASET_ID} Hindi parquet={parquet_file} split={spec} "
        f"(Hub config is 'default'; Hindi is selected by file, not config='hi')"
    )
    # First run caches parquet under ~/.cache/huggingface/; later runs reuse it.
    ds = load_dataset(
        DATASET_ID,
        data_files={args.split: parquet_file},
        split=spec,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = len(ds)
    out = output_path(args.split, n)

    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            record = extract_row(dict(row))
            # ensure_ascii=False keeps Hindi script readable in the JSONL file.
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i == 0:
                print_sample(record)
            written += 1

    print(f"Wrote {written} rows -> {out}")
    print("Next: create_vector_database reads this JSONL. Do not query Hugging Face at runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
