"""Thin, polite HTTP client for the official FPL API.

Retries with backoff, a real User-Agent, and a tiny on-disk cache so a
backfill that dies halfway does not re-hammer the endpoint on restart.

v8: a 4xx other than 429 is an answer, not a transient failure. v7 retried
every error four times with growing sleeps, so a mistyped team ID took ~15
seconds to fail and then surfaced as "could not reach the FPL API". It now
raises FPLHTTPError immediately with the status code, so callers can turn a
404 into "no such entry". The disk cache also expires: v7's `--cache`
reused a response forever, so a regular `--cache` run silently predicted
from the first day's data.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from fplagent.config import FPL_BASE, FPL_TIMEOUT, FPL_USER_AGENT, ROOT

log = logging.getLogger(__name__)
CACHE_DIR = ROOT / ".cache"
CACHE_TTL_SECONDS = 6 * 3600


class FPLHTTPError(RuntimeError):
    """A definitive HTTP error from the FPL API (not worth retrying)."""

    def __init__(self, path: str, status: int, message: str = "") -> None:
        super().__init__(f"FPL API returned {status} for {path}"
                         + (f": {message}" if message else ""))
        self.path = path
        self.status = status


class FPLClient:
    def __init__(self, base: str = FPL_BASE, use_cache: bool = False,
                 max_retries: int = 4, cache_ttl: float = CACHE_TTL_SECONDS) -> None:
        self.base = base.rstrip("/")
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": FPL_USER_AGENT,
                                     "Accept": "application/json"})
        if use_cache:
            CACHE_DIR.mkdir(exist_ok=True)

    def _cache_path(self, path: str) -> Path:
        return CACHE_DIR / (path.strip("/").replace("/", "_") + ".json")

    def get(self, path: str) -> Any:
        if self.use_cache:
            cp = self._cache_path(path)
            if cp.exists() and (time.time() - cp.stat().st_mtime) < self.cache_ttl:
                return json.loads(cp.read_text())

        url = f"{self.base}/{path.lstrip('/')}"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, timeout=FPL_TIMEOUT)
            except requests.RequestException as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 429:
                wait = 2 ** attempt * 3
                log.warning("rate limited on %s, sleeping %ss", path, wait)
                time.sleep(wait)
                last = FPLHTTPError(path, 429, "rate limited")
                continue
            if 400 <= r.status_code < 500:
                raise FPLHTTPError(path, r.status_code, r.text[:200])
            if r.status_code >= 500:
                last = FPLHTTPError(path, r.status_code, "server error")
                time.sleep(1.5 * (attempt + 1))
                continue
            try:
                data = r.json()
            except ValueError as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
                continue
            if self.use_cache:
                self._cache_path(path).write_text(json.dumps(data))
            return data
        raise RuntimeError(f"FPL API failed for {path}: {last}")

    def bootstrap(self) -> dict:
        return self.get("bootstrap-static/")

    def fixtures(self) -> list[dict]:
        return self.get("fixtures/")

    def element_summary(self, player_id: int) -> dict:
        return self.get(f"element-summary/{player_id}/")

    def live(self, gw: int) -> dict:
        return self.get(f"event/{gw}/live/")
