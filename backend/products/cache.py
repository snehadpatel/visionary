"""
Visionary Products — Filesystem-based product cache.
Caches scraped results for 24 hours to avoid hammering sites.
"""
import json
import hashlib
import os
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "product_cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_TTL = 86400  # 24 hours


def cache_key(query: str, site: str) -> str:
    """Generate a unique cache key for a query+site combination."""
    return hashlib.md5(f"{site}:{query}".encode()).hexdigest()


def read_cache(key: str) -> list[dict] | None:
    """Read cached results if they exist and haven't expired."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
                if time.time() - data.get("timestamp", 0) < CACHE_TTL:
                    return data["results"]
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def write_cache(key: str, results: list[dict]):
    """Write results to cache with timestamp."""
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump({"timestamp": time.time(), "results": results}, f, indent=2)


def clear_cache():
    """Remove all cached results."""
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
    print("[Cache] All product cache cleared.")
