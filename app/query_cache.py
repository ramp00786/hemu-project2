from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Any


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw.strip()))
    except ValueError:
        return default


def normalize_query(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def cache_key(text: str) -> str:
    normalized = normalize_query(text)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


class QueryCache:
    def __init__(self, *, enabled: bool | None = None, max_size: int | None = None) -> None:
        self.enabled = _env_flag("QUERY_CACHE", True) if enabled is None else enabled
        self.max_size = _env_int("QUERY_CACHE_MAX", 512) if max_size is None else max_size
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        key = cache_key(text)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return dict(entry)

    def set(self, text: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        key = cache_key(text)
        self._store[key] = dict(payload)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._store),
            "max_size": self.max_size,
        }


query_cache = QueryCache()
