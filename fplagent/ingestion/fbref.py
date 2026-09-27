"""Penalty-taker enrichment from a MANUALLY EXPORTED FBref CSV.

WHY THIS IS A FILE READER, NOT A SCRAPER
-----------------------------------------
FBref (a Sports Reference site) states in its own Terms of Use that
automated scraping/spidering is not permitted, and backs that up with
Cloudflare bot-detection that returns 403 to non-browser requests. That is a
real, stated policy, not just a technical inconvenience -- so this module
does not send FBref any automated HTTP requests at all, and never has since
this rewrite.

What it does instead is the thing FBref's own UI is built for: every stats
table has a "Share & Export -> Get table as CSV (for Excel)" link. Clicking
that is an ordinary, human, manual action -- no different from downloading a
spreadsheet. This module reads the file you saved from doing that.

HOW TO GET THE CSV
-------------------
1. Open https://fbref.com/en/comps/9/stats/Premier-League-Stats in a browser
2. Find the "Standard Stats" table
3. Click "Share & Export" (top-right of the table) -> "Get table as CSV
   (for Excel)" -- this copies CSV text to your clipboard
4. Paste it into a text file, e.g. data/fbref_standard_stats.csv
5. Run: python -m fplagent.cli ingest --fbref-csv data/fbref_standard_stats.csv
   (or call ingest_penalty_duties(season, csv_path=...) directly)

This is a five-minute manual step you repeat every few gameweeks -- the
primary-penalty-taker list barely changes in-season, so this does not need
to be automated to be useful. scripts/seed_penalty_duties.py remains the
zero-dependency alternative if you'd rather type in the ~20 names directly
and skip FBref entirely.

FBref's CSV export repeats its header row periodically (same as the HTML
table) and sometimes uses a tab rather than a comma depending on browser/OS
clipboard behaviour, both handled below.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from fplagent.db import tables as T
from fplagent.db.repository import read_sql
from fplagent.ingestion.fbref_common import read_fbref_csv

log = logging.getLogger(__name__)

MIN_ATTEMPTS_FOR_CONFIDENCE = 4   # attempts at which "primary taker" counts as solid evidence


def load_penalties_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """Parse a manually-exported FBref 'Standard Stats' CSV.

    Returns an empty DataFrame (never raises) if the file is missing or
    doesn't look like the expected export -- this is optional enrichment
    and must never take down the pipeline around it.
    """
    path = Path(csv_path)
    df = read_fbref_csv(path)
    if df is None:
        log.info("no usable FBref CSV at %s; skipping penalty-duty enrichment "
                 "(this is optional -- see ingestion/fbref.py for how to "
                 "export one, or use scripts/seed_penalty_duties.py instead)",
                 path)
        return pd.DataFrame()

    if not {"Player", "Squad", "PKatt"}.issubset(df.columns):
        log.warning("%s is missing expected columns (Player, Squad, PKatt); "
                    "got: %s. Make sure this is the 'Standard Stats' table "
                    "export, not a different one.", path, list(df.columns))
        return pd.DataFrame()

    df = df[["Player", "Squad", "PKatt"]].copy()
    df["PKatt"] = pd.to_numeric(df["PKatt"], errors="coerce").fillna(0)
    df = df[df["PKatt"] > 0]
    if df.empty:
        log.info("%s parsed but no players with penalty attempts found", path)
        return df

    df["order"] = df.groupby("Squad")["PKatt"].rank(method="first", ascending=False).astype(int)
    return df.rename(columns={"Player": "player_name", "Squad": "team"})[
        ["team", "player_name", "order", "PKatt"]]


def get_matched_penalty_duties(season: int, csv_path: str | Path) -> list[dict]:
    """Fuzzy-match names from the exported CSV to our player IDs and return
    rows shaped for `set_piece_duties`. Empty list (never raises) on any
    failure -- missing file, missing dependency, or no confident matches."""
    fbref_df = load_penalties_from_csv(csv_path)
    if fbref_df.empty:
        return []

    try:
        from rapidfuzz import fuzz, process  # noqa: F401
    except ImportError:
        log.warning("rapidfuzz not installed; skipping penalty-duty matching "
                    "(pip install rapidfuzz to enable it)")
        return []
    from fplagent.ingestion.name_matching import resolve_player_id, resolve_team_id
    from fplagent.models.setpieces import order_to_share

    players_df = read_sql(select(
        T.players.c.player_id, T.players.c.web_name, T.players.c.team_id,
        T.players.c.first_name, T.players.c.second_name,
    ).where(T.players.c.season == season))
    teams_df = read_sql(select(
        T.teams.c.team_id, T.teams.c.name, T.teams.c.short_name
    ).where(T.teams.c.season == season))
    if players_df.empty or teams_df.empty:
        log.warning("no players/teams in the database for season %s; run "
                    "`fplagent ingest` before matching penalty duties", season)
        return []

    records: list[dict] = []
    seen: set[int] = set()
    now = datetime.utcnow()

    for _, row in fbref_df.iterrows():
        # Scope player matching to the resolved club first -- surnames
        # collide across clubs, and an unscoped global match risks wiring a
        # duty to the wrong player entirely. FBref spells clubs "Manchester
        # City" / "Newcastle Utd" / "Tottenham" against FPL's "Man City" /
        # "Newcastle" / "Spurs", which a raw fuzzy score never bridged; the
        # shared resolver's alias table does.
        team_id = resolve_team_id(row["team"], teams_df)
        if team_id is None:
            log.info("could not confidently match FBref team '%s'; skipping %s",
                     row["team"], row["player_name"])
            continue

        # FBref gives full names ("Erling Haaland"); FPL's web_name is
        # usually the surname alone, which a raw token_sort_ratio scores ~67.
        player_id = resolve_player_id(row["player_name"], players_df, team_id=team_id)
        if player_id is None:
            log.info("could not confidently match FBref player '%s' (%s)",
                     row["player_name"], row["team"])
            continue
        if player_id in seen:
            continue      # one row per (player, season, duty) in one upsert
        seen.add(player_id)

        attempts = float(row["PKatt"])
        order = int(row["order"])
        confidence = min(1.0, attempts / MIN_ATTEMPTS_FOR_CONFIDENCE)
        if order > 1:
            confidence *= 0.5   # a backup taker is inherently less certain to keep the role

        records.append({
            "player_id": player_id, "season": season, "duty": "penalties",
            "order": order, "share": order_to_share("penalties", order),
            "confidence": round(confidence, 3),
            "known_at": now, "source": "fbref_csv",
            # Overwriting a row the bootstrap ingest had retired must
            # un-retire it, or this declared duty is never read.
            "retired_at": None,
        })

    return records
