"""Bulk read/write helpers.

Two things v5 got wrong that are fixed here:

* Per-row session.execute() in ingestion loops -- 659 players x ~10 round
  trips per pull. Everything here uses executemany-style batching, which
  matters enormously once multi-season backfill lands.
* No point-in-time discipline. read_fixture_stats_before() is the ONLY way
  training code is allowed to read performance data, and it requires a
  cutoff timestamp. Making the safe path the easy path is the whole trick.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Sequence

import pandas as pd
from sqlalchemy import Table, and_, delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement


class _Least(FunctionElement):
    """Portable two-argument minimum.

    Postgres spells it LEAST(a, b); SQLite spells the same thing MIN(a, b)
    when given two arguments, and rejects LEAST outright. SQLAlchemy has no
    built-in for this, so it gets one compiler hook here rather than an
    `if IS_SQLITE` at the one call site that needs it.
    """
    name = "least"
    inherit_cache = True


@compiles(_Least)
def _compile_least(element, compiler, **kw):
    return "least(%s)" % compiler.process(element.clauses, **kw)


@compiles(_Least, "sqlite")
def _compile_least_sqlite(element, compiler, **kw):
    return "min(%s)" % compiler.process(element.clauses, **kw)

from fplagent.db import tables as T
from fplagent.db.session import is_sqlite, session_scope


# Columns that record WHEN a fact became knowable. On conflict these must
# keep their earliest value: the whole walk-forward backtest reads
# `known_at < deadline`, so refreshing them on every re-ingest silently
# rewrites history into the present and makes every past prediction
# unscoreable. This is the single most damaging class of bug in the system
# and it is fixed here, once, for every table, rather than by remembering to
# omit the column at each call site.
FIRST_WRITE_WINS = ("known_at", "created_at", "first_seen_at")


def _upsert_stmt(table: Table, rows: Sequence[dict], conflict_cols: Sequence[str],
                 preserve: Sequence[str] = FIRST_WRITE_WINS):
    """Dialect-appropriate INSERT ... ON CONFLICT DO UPDATE.

    Every non-conflict column present in the payload is refreshed, so
    re-ingesting a gameweek corrects existing rows rather than duplicating
    them, EXCEPT for the columns in `preserve`. Those take the earlier of the
    existing and incoming value, so a row's recorded knowability never moves
    forward in time just because we pulled it again.
    """
    # Decided per statement from the ACTIVE database, not from a flag bound
    # at import time: the process can be switched to another database (the
    # demo and the test suite both do), and a Postgres INSERT compiled
    # against a SQLite engine is not the same statement.
    ins = sqlite_insert(table) if is_sqlite() else pg_insert(table)
    stmt = ins.values(list(rows))
    payload_cols = set()
    for r in rows:
        payload_cols |= set(r.keys())

    preserve_set = {c for c in preserve if c in table.c}
    update_cols = {}
    for name in payload_cols:
        if name in set(conflict_cols) or name not in table.c:
            continue
        if name in preserve_set:
            # LEAST() is the portable "earliest wins". Both SQLite and
            # Postgres have it (Postgres spells it LEAST, SQLite MIN with
            # two args, and SQLAlchemy's least() renders correctly for each).
            update_cols[name] = _Least(
                table.c[name], getattr(stmt.excluded, name))
        else:
            update_cols[name] = getattr(stmt.excluded, name)

    if not update_cols:
        return stmt.on_conflict_do_nothing(index_elements=list(conflict_cols))
    return stmt.on_conflict_do_update(
        index_elements=list(conflict_cols), set_=update_cols)


def upsert(table: Table, rows: Iterable[dict], conflict_cols: Sequence[str],
           batch_size: int = 500,
           preserve: Sequence[str] = FIRST_WRITE_WINS) -> int:
    rows = [r for r in rows if r]
    if not rows:
        return 0
    written = 0
    with session_scope() as s:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            s.execute(_upsert_stmt(table, chunk, conflict_cols, preserve))
            written += len(chunk)
    return written


def insert_rows(table: Table, rows: Iterable[dict], batch_size: int = 500) -> int:
    rows = [r for r in rows if r]
    if not rows:
        return 0
    with session_scope() as s:
        for i in range(0, len(rows), batch_size):
            s.execute(insert(table), rows[i:i + batch_size])
    return len(rows)


def read_sql(stmt) -> pd.DataFrame:
    with session_scope() as s:
        result = s.execute(stmt)
        return pd.DataFrame(result.mappings().all())


# ---------------------------------------------------------------------------
# Point-in-time reads
# ---------------------------------------------------------------------------

def read_fixture_stats_before(cutoff: datetime, season: int | None = None,
                              player_ids: Sequence[int] | None = None) -> pd.DataFrame:
    """Every per-fixture row that was KNOWN before `cutoff`.

    This is the only sanctioned read path for model training and backtesting.
    Filtering on known_at (not on gw <= n) is what makes the walk-forward
    backtest honest: a row ingested late, or corrected after the fact, cannot
    leak backwards into a prediction that was made before the correction.
    """
    conds = [T.player_fixture_stats.c.known_at < cutoff]
    if season is not None:
        conds.append(T.player_fixture_stats.c.season <= season)
    if player_ids is not None:
        conds.append(T.player_fixture_stats.c.player_id.in_(list(player_ids)))
    stmt = select(T.player_fixture_stats).where(and_(*conds))
    return read_sql(stmt)


def read_finished_fixtures(season: int, before_gw: int | None = None) -> pd.DataFrame:
    conds = [T.fixtures.c.season <= season, T.fixtures.c.finished.is_(True)]
    if before_gw is not None:
        conds.append(
            (T.fixtures.c.season < season) | (T.fixtures.c.gw < before_gw)
        )
    return read_sql(select(T.fixtures).where(and_(*conds)))


def read_gameweek(season: int, gw: int) -> dict[str, Any] | None:
    df = read_sql(select(T.gameweeks).where(
        and_(T.gameweeks.c.season == season, T.gameweeks.c.gw == gw)))
    return None if df.empty else df.iloc[0].to_dict()


def resolve_target_gw(season: int) -> int | None:
    """The next unfinished gameweek, straight from stored FPL events."""
    df = read_sql(select(T.gameweeks).where(T.gameweeks.c.season == season))
    if df.empty:
        return None
    nxt = df[df["is_next"] == True]  # noqa: E712
    if not nxt.empty:
        return int(nxt.iloc[0]["gw"])
    unfinished = df[df["finished"] == False].sort_values("gw")  # noqa: E712
    return int(unfinished.iloc[0]["gw"]) if not unfinished.empty else None


def latest_snapshot_per_player(season: int, before: datetime | None = None) -> pd.DataFrame:
    """The most recent bootstrap snapshot per player, optionally restricted
    to snapshots pulled strictly before `before` (point-in-time).

    Selects only the latest row per player in SQL. One snapshot row is
    inserted per player per ingest, so reading the whole table and sorting
    it in pandas -- what v7 did on every API request -- grows without bound
    over a season.
    """
    s = T.player_season_snapshots
    conds = [s.c.season == season]
    if before is not None:
        conds.append(s.c.pulled_at < before)
    latest = (select(s.c.player_id, func.max(s.c.pulled_at).label("max_pulled"))
              .where(and_(*conds))
              .group_by(s.c.player_id)
              .subquery())
    stmt = (select(s)
            .join(latest, and_(s.c.player_id == latest.c.player_id,
                               s.c.pulled_at == latest.c.max_pulled))
            .where(s.c.season == season))
    df = read_sql(stmt)
    if df.empty:
        return df
    # Two snapshots sharing a timestamp: keep the later insert.
    return (df.sort_values("snapshot_id")
              .groupby("player_id", as_index=False)
              .tail(1)
              .reset_index(drop=True))


def log_quality_event(stage: str, check_name: str, severity: str, detail: dict) -> None:
    insert_rows(T.data_quality_events, [{
        "stage": stage, "check_name": check_name,
        "severity": severity, "detail": detail,
    }])
