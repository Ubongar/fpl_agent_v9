"""`fplagent doctor` -- diagnose the environment before anything else.

Written specifically because the v5 setup session burned hours on a Postgres
connection problem that was diagnosed three different wrong ways. Every check
here reports what it found AND what to do about it, because a diagnostic that
only says "failed" just moves the guessing somewhere else.
"""
from __future__ import annotations

import importlib
import os
import platform
import socket
import sys
from urllib.parse import urlparse

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}    {msg}")


def _warn(msg: str, fix: str = "") -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}")
    if fix:
        print(f"        -> {fix}")


def _fail(msg: str, fix: str = "") -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")
    if fix:
        print(f"        -> {fix}")


def diagnose() -> int:
    problems = 0
    print(f"\nFPL Agent v9 doctor -- {platform.system()} {platform.release()}, "
          f"Python {sys.version.split()[0]}\n")

    print("Dependencies:")
    required = ["numpy", "scipy", "pandas", "sqlalchemy", "requests", "pulp"]
    optional = {"sklearn": "learned minutes model (falls back to heuristic)",
                "streamlit": "the dashboard",
                "fastapi": "the API server",
                "openai": "LLM explanations",
                "psycopg2": "Postgres driver"}
    for mod in required:
        try:
            importlib.import_module(mod)
            _ok(f"{mod}")
        except ImportError:
            _fail(f"{mod} missing", "pip install -r requirements.txt")
            problems += 1
    for mod, why in optional.items():
        try:
            importlib.import_module(mod)
            _ok(f"{mod} (optional: {why})")
        except ImportError:
            _warn(f"{mod} not installed -- {why} unavailable")

    print("\nDatabase:")
    from fplagent.config import DATABASE_URL, IS_SQLITE
    parsed = urlparse(DATABASE_URL)
    shown = DATABASE_URL if IS_SQLITE else f"{parsed.scheme}://{parsed.hostname}:{parsed.port}{parsed.path}"
    print(f"  URL: {shown}")

    if IS_SQLITE:
        _ok("SQLite -- zero setup required, everything works out of the box")
    else:
        host, port = parsed.hostname or "localhost", parsed.port or 5432
        # This is the exact check that would have resolved the v5 setup
        # session in one step instead of six: is anything listening at all,
        # and is it the thing you think it is?
        try:
            with socket.create_connection((host, port), timeout=3):
                _ok(f"something is listening on {host}:{port}")
        except OSError as e:
            _fail(f"nothing reachable on {host}:{port} ({e})",
                  "If Postgres runs in WSL and you are calling from Windows, "
                  "'localhost' may resolve to ::1 (IPv6) and miss it entirely. "
                  "Run `wsl hostname -I` and use that IP, and make sure "
                  "listen_addresses='*' is set in postgresql.conf.")
            problems += 1
        if host in ("localhost", "127.0.0.1"):
            _warn("using localhost with a non-SQLite database",
                  "If you also have a native Windows Postgres installed, it may "
                  "be occupying port 5432 and silently answering instead of the "
                  "one you meant. Use the explicit WSL IP to be certain which "
                  "server you are talking to.")

    try:
        from sqlalchemy import text
        from fplagent.db.session import session_scope
        with session_scope() as s:
            s.execute(text("SELECT 1"))
        _ok("connection succeeded")
    except Exception as e:      # noqa: BLE001
        _fail(f"connection failed: {type(e).__name__}: {e}")
        problems += 1

    try:
        from sqlalchemy import select
        from fplagent.db import tables as T
        from fplagent.db.repository import read_sql
        for name, table in (("players", T.players), ("fixtures", T.fixtures),
                            ("player_fixture_stats", T.player_fixture_stats),
                            ("predictions", T.predictions)):
            try:
                n = len(read_sql(select(table)))
                (_ok if n else _warn)(f"{name}: {n} rows")
            except Exception:
                _warn(f"{name}: table not created yet", "fplagent init-db")
    except Exception:
        pass

    print("\nConfiguration:")
    from fplagent.config import CURRENT_SEASON, LLM_API_KEY, ODDS_API_KEY
    _ok(f"season {CURRENT_SEASON}/{str(CURRENT_SEASON + 1)[-2:]}")
    (_ok if LLM_API_KEY else _warn)(
        "LLM_API_KEY set" if LLM_API_KEY
        else "LLM_API_KEY not set -- explanations fall back to raw output "
             "(everything analytical still works)")
    (_ok if ODDS_API_KEY else _warn)(
        "ODDS_API_KEY set" if ODDS_API_KEY
        else "ODDS_API_KEY not set -- market blending disabled")

    print()
    if problems:
        print(f"{RED}{problems} problem(s) found.{RESET} Fix the FAIL lines above, "
              f"then run `fplagent doctor` again.\n")
        return 1
    print(f"{GREEN}Everything looks good.{RESET} Next: `fplagent demo` for an "
          f"offline run, or `fplagent run` against the live API.\n")
    return 0
