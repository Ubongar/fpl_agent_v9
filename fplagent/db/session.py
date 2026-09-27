"""Engine + session handling.

v5 opened sessions with get_session() and closed them manually with no
try/finally, so any exception leaked a connection out of the pool. Every
database interaction goes through a context manager.

v8 additions:

* `use_database(url)` switches the whole process to another database and
  disposes the old engine. v7 had no way to do that short of editing an
  environment variable before import, which is why `fplagent demo` could
  only ever run against -- and drop -- the configured production database,
  and why tests that set DATABASE_URL mid-run silently kept writing to the
  real one.
* JSON columns are written through `serialize.json_dumps`, which turns NaN
  and Infinity into null. PostgreSQL rejects bare NaN tokens in json, and a
  single NaN in an evaluation result used to fail the whole insert.
* `init_db()` adds columns that exist in the schema but not yet in the
  database. `create_all` only ever creates missing TABLES, so every new
  column added to an existing Postgres database needed a manual ALTER; this
  covers the additive case, which is the only one the project needs.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from fplagent.config import DATABASE_URL
from fplagent.db.serialize import json_dumps
from fplagent.db.tables import metadata

log = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None
_active_url: str = DATABASE_URL


def active_url() -> str:
    """The database URL every session currently uses."""
    return _active_url


def is_sqlite(url: str | None = None) -> bool:
    return (url or _active_url).startswith("sqlite")


def dispose_engine() -> None:
    """Close every pooled connection. Needed before deleting a SQLite file
    on Windows, which refuses to delete a file that is still open."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


def use_database(url: str) -> str:
    """Point every subsequent session at `url`. Returns the previous URL so
    the caller can switch back."""
    global _active_url
    previous = _active_url
    dispose_engine()
    _active_url = url
    return previous


def get_engine() -> Engine:
    global _engine, _SessionFactory
    if _engine is None:
        kwargs: dict = {"future": True, "json_serializer": json_dumps}
        if is_sqlite():
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs["pool_pre_ping"] = True
        _engine = create_engine(_active_url, **kwargs)
        if is_sqlite():
            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _):
                dbapi_conn.execute("PRAGMA journal_mode=WAL")
                dbapi_conn.execute("PRAGMA synchronous=NORMAL")
        _SessionFactory = sessionmaker(bind=_engine, future=True)
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception,
    and always returns the connection to the pool."""
    get_engine()
    assert _SessionFactory is not None
    s = _SessionFactory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def add_missing_columns(engine: Engine | None = None) -> list[str]:
    """ALTER TABLE ... ADD COLUMN for every nullable schema column missing
    from an existing table. Additive only: nothing is ever dropped, renamed
    or retyped. Returns "table.column" for each column added."""
    engine = engine or get_engine()
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    preparer = engine.dialect.identifier_preparer
    added: list[str] = []
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                if col.primary_key or not col.nullable:
                    log.warning("cannot add non-nullable column %s.%s automatically; "
                                "migrate it by hand", table.name, col.name)
                    continue
                coltype = col.type.compile(dialect=engine.dialect)
                conn.execute(text(
                    f"ALTER TABLE {preparer.quote(table.name)} "
                    f"ADD COLUMN {preparer.quote(col.name)} {coltype}"))
                added.append(f"{table.name}.{col.name}")
    if added:
        log.info("schema: added columns %s", added)
    return added


def init_db(drop: bool = False) -> list[str]:
    """Create every table and add any missing columns. Portable across
    SQLite and Postgres.

    v5's init_schema() read schema.sql and split it on ';', which breaks the
    first time you add a function, trigger or DO $$ block. Defining the schema
    in SQLAlchemy Core removes the string-splitting entirely.
    """
    engine = get_engine()
    if drop:
        metadata.drop_all(engine)
    metadata.create_all(engine)
    add_missing_columns(engine)
    return sorted(metadata.tables.keys())
