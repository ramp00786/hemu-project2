from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import httpx

from app.latency_report import (
    SAMPLE_PATH,
    format_stats_line,
    run_single,
    summarize,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure P50/P70/P100 using /ask-text")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Run each question twice; report cold vs cached pass separately",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"Missing sample queries: {SAMPLE_PATH}")

    rows = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    questions = []
    for row in rows:
        q = (row.get("query_en") or row.get("query_hi") or "").strip()
        if q.startswith("."):
            q = q.lstrip(". ").strip()
        if q:
            questions.append(q)
    questions = questions[: args.max_queries]

    if not questions:
        raise SystemExit("No usable sample questions.")

    total_ms: list[float] = []
    path_ms: list[float] = []
    retrieve_ms: list[float] = []
    extractive_ms: list[float] = []
    generate_ms: list[float] = []
    paths: Counter[str] = Counter()
    cached_count = 0
    cold_total_ms: list[float] = []
    warm_total_ms: list[float] = []

    with httpx.Client(timeout=60.0) as client:
        for q in questions:
            if args.warm_cache:
                cold = run_single(client, args.base_url, q)
                if cold is None:
                    continue
                cold_total_ms.append(cold["http_ms"])
                time.sleep(0.05)
                warm = run_single(client, args.base_url, q)
                if warm is None:
                    continue
                warm_total_ms.append(warm["http_ms"])
                if warm["cached"]:
                    cached_count += 1
                total_ms.append(warm["http_ms"])
                lat = warm["latency_ms"]
                path_ms.append(lat)
                retrieve_ms.append(warm["retrieve_ms"])
                path = warm["path"]
                paths[path] += 1
                if path == "extractive":
                    extractive_ms.append(lat)
                generate_ms.append(warm["generate_ms"])
                time.sleep(0.05)
                continue

            result = run_single(client, args.base_url, q)
            if result is None:
                continue
            total_ms.append(result["http_ms"])
            if result["cached"]:
                cached_count += 1
            lat = result["latency_ms"]
            path_ms.append(lat)
            retrieve_ms.append(result["retrieve_ms"])
            path = result["path"]
            paths[path] += 1
            if path == "extractive":
                extractive_ms.append(lat)
            time.sleep(0.05)
            generate_ms.append(result["generate_ms"])

    if not total_ms:
        raise SystemExit("No successful bench queries.")

    print(f"Queries: {len(total_ms)}")
    print(f"Cached responses: {cached_count}")
    print(f"Paths: {dict(paths)}")
    if args.warm_cache and cold_total_ms:
        print("--- pass-1 cold HTTP ms ---")
        print(format_stats_line("cold", summarize(cold_total_ms)))
        print("--- pass-2 cached HTTP ms ---")
        print(format_stats_line("cached", summarize(warm_total_ms)))
    print("--- end-to-end HTTP ms ---")
    print(format_stats_line("HTTP", summarize(total_ms)))
    print("--- retrieve_ms ---")
    print(format_stats_line("retrieve", summarize(retrieve_ms)))
    print("--- latency_ms (retrieve + answer + guards, STT excluded) ---")
    print(format_stats_line("all paths", summarize(path_ms)))
    print("--- extractive path only (200ms target) ---")
    print(format_stats_line("extractive", summarize(extractive_ms)))
    print("--- generate_ms (LLM card; not part of 200ms claim) ---")
    print(format_stats_line("llm", summarize(generate_ms)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
