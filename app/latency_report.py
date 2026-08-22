from __future__ import annotations

import argparse
import json
import statistics
import time
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "indexes" / "sample_queries.json"

_report_cache: dict[str, Any] | None = None
_report_cache_meta: dict[str, Any] = {"max_queries": 100, "base_url": "http://127.0.0.1:8000"}


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int((p / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p100": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "p50": round(percentile(values, 50), 2),
        "p70": round(percentile(values, 70), 2),
        "p100": round(percentile(values, 100), 2),
        "avg": round(statistics.mean(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def format_stats_line(label: str, stats: dict[str, float]) -> str:
    return (
        f"{label}: P50={stats['p50']:.2f} P70={stats['p70']:.2f} "
        f"P100={stats['p100']:.2f} avg={stats['avg']:.2f}"
    )


def get_report() -> dict[str, Any] | None:
    return _report_cache


def clear_report() -> None:
    global _report_cache
    _report_cache = None


def _clean_question(text: str) -> str:
    q = (text or "").strip()
    if q.startswith("."):
        q = q.lstrip(". ").strip()
    return q


def fetch_questions(client: httpx.Client, base_url: str, limit: int = 100) -> list[str]:
    try:
        r = client.get(f"{base_url.rstrip('/')}/api/sample-questions", params={"limit": limit})
        if r.status_code < 400:
            rows = r.json().get("questions") or []
            out: list[str] = []
            for row in rows:
                q = _clean_question(row.get("question") or row.get("query_en") or row.get("query_hi") or "")
                if q:
                    out.append(q)
            if out:
                return out[:limit]
    except httpx.HTTPError:
        pass

    if not SAMPLE_PATH.exists():
        return []
    rows = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        q = _clean_question(row.get("query_en") or row.get("query_hi") or "")
        if q:
            out.append(q)
    return out[:limit]


def run_single(client: httpx.Client, base_url: str, question: str) -> dict[str, Any] | None:
    t0 = time.perf_counter()
    try:
        r = client.post(f"{base_url.rstrip('/')}/ask-text", json={"question": question})
    except httpx.HTTPError:
        return None
    http_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if r.status_code >= 400:
        return None
    data = r.json()
    return {
        "http_ms": http_ms,
        "retrieve_ms": float(data.get("retrieve_ms") or 0),
        "latency_ms": float(data.get("latency_ms") or 0),
        "generate_ms": float(data.get("generate_ms") or 0),
        "path": str(data.get("path") or "unknown"),
        "cached": bool(data.get("cached")),
        "refused": bool(data.get("refused")),
    }


def _fetch_health(client: httpx.Client, base_url: str) -> dict[str, Any]:
    try:
        r = client.get(f"{base_url.rstrip('/')}/health")
        if r.status_code < 400:
            data = r.json()
            if data.get("ok"):
                return {
                    "ok": True,
                    "embedding_model": data.get("embedding_model"),
                    "embed_device": data.get("embed_device"),
                    "num_chunks": data.get("num_chunks"),
                    "query_cache": data.get("query_cache") or {},
                }
    except httpx.HTTPError:
        pass
    return {"ok": False}


def build_report(
    base_url: str = "http://127.0.0.1:8000",
    max_queries: int = 100,
    *,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Return (report, was_cached). Raises RuntimeError if server unreachable."""
    global _report_cache, _report_cache_meta

    if _report_cache is not None and not force:
        if _report_cache_meta.get("max_queries") == max_queries and _report_cache_meta.get("base_url") == base_url:
            return _report_cache, True

    with httpx.Client(timeout=600.0) as client:
        health = _fetch_health(client, base_url)
        if not health.get("ok"):
            raise RuntimeError("Server unreachable or retriever not loaded")

        questions = fetch_questions(client, base_url, limit=max_queries)
        if not questions:
            raise RuntimeError("No sample questions available")

        rows: list[dict[str, Any]] = []
        cold_http: list[float] = []
        warm_http: list[float] = []
        retrieve_vals: list[float] = []
        extractive_vals: list[float] = []
        generate_vals: list[float] = []
        paths: Counter[str] = Counter()

        for q in questions:
            cold = run_single(client, base_url, q)
            if cold is None:
                continue
            time.sleep(0.05)
            warm = run_single(client, base_url, q)
            if warm is None:
                continue

            cold_http.append(cold["http_ms"])
            warm_http.append(warm["http_ms"])
            retrieve_vals.append(cold["retrieve_ms"])
            path = cold["path"]
            paths[path] += 1
            if path == "extractive":
                extractive_vals.append(cold["latency_ms"])
            gen = cold.get("generate_ms")
            if gen is not None:
                generate_vals.append(float(gen))

            rows.append(
                {
                    "question": q,
                    "path": path,
                    "cold_http_ms": cold["http_ms"],
                    "warm_http_ms": warm["http_ms"],
                    "retrieve_ms": cold["retrieve_ms"],
                    "latency_ms": cold["latency_ms"],
                    "generate_ms": cold["generate_ms"],
                    "warm_cached": warm["cached"],
                    "refused": cold["refused"],
                }
            )
            time.sleep(0.05)

        if not rows:
            raise RuntimeError("No successful benchmark queries")

        health = _fetch_health(client, base_url)

        try:
            tz = ZoneInfo("Asia/Kolkata")
            generated_at = datetime.now(tz).isoformat()
        except Exception:
            generated_at = datetime.now(timezone.utc).isoformat()

        report = {
            "generated_at": generated_at,
            "base_url": base_url,
            "query_count": len(rows),
            "health": health,
            "summary": {
                "cold_http": summarize(cold_http),
                "warm_http": summarize(warm_http),
                "retrieve_ms": summarize(retrieve_vals),
                "extractive_ms": summarize(extractive_vals),
                "generate_ms": summarize(generate_vals),
            },
            "paths": dict(paths),
            "queries": rows,
        }

    _report_cache = report
    _report_cache_meta = {"max_queries": max_queries, "base_url": base_url}
    return report, False


def print_report(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    print(f"Generated: {report.get('generated_at')}")
    print(f"Queries: {report.get('query_count')}")
    print(f"Paths: {report.get('paths')}")
    print("--- cold HTTP ms ---")
    print(format_stats_line("cold", summary.get("cold_http") or {}))
    print("--- warm HTTP ms ---")
    print(format_stats_line("warm", summary.get("warm_http") or {}))
    print("--- retrieve_ms ---")
    print(format_stats_line("retrieve", summary.get("retrieve_ms") or {}))
    print("--- extractive latency_ms ---")
    print(format_stats_line("extractive", summary.get("extractive_ms") or {}))
    print("--- generate_ms ---")
    print(format_stats_line("llm", summary.get("generate_ms") or {}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency report builder and CLI")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--print", dest="print_report", action="store_true", help="Print summary to stdout")
    parser.add_argument("--open-browser", action="store_true", help="Open /latency-report in browser")
    parser.add_argument("--force", action="store_true", help="Ignore cached report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.open_browser:
        webbrowser.open(f"{args.base_url.rstrip('/')}/latency-report")
        return 0
    if args.print_report or not args.open_browser:
        try:
            report, cached = build_report(
                base_url=args.base_url,
                max_queries=args.max_queries,
                force=args.force,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if cached:
            print("(cached report)")
        print_report(report)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
