"""Hand-maintained fixtures for competitions with no free API.

football-data.org's free tier covers the Champions League (ingested
automatically -- see `ingestion/european_fixtures.py`) but not the Europa
League, the Conference League, the FA Cup, or the League Cup. No other clean
free source covers all four either. Building four separate scrapers for
fixture lists that change on a known, slow schedule (a handful of rounds a
season) is not worth the fragility, so this file is the maintained record
instead.

HOW TO UPDATE
-------------
A few times a season -- after each round draw -- check:
  * Europa League / Conference League: https://www.uefa.com/europaleague/
    (or the equivalent Conference League page) for that round's fixture dates
  * FA Cup / League Cup: https://www.thefa.com/competitions or the club's own
    official fixture list

Add one row per Premier League club involved, using their `short_name` as it
appears in our `teams` table (check with `fplagent teams` or the API's
`/teams` endpoint if unsure). `team_id` is resolved by name at seed time, not
hardcoded here, so this file survives a season's team_id reassignment.

Run after editing:
    python -m scripts.seed_cup_fixtures --season 2026

The rows below are real examples to show the format; replace them with the
season's actual fixtures. An empty or stale file degrades the congestion
signal for these competitions gracefully -- it does not break anything.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from sqlalchemy import select

from fplagent.config import CURRENT_SEASON
from fplagent.db import tables as T
from fplagent.db.repository import read_sql, upsert
from fplagent.db.session import init_db

log = logging.getLogger(__name__)

# competition: UEL | UECL | FA_CUP | LEAGUE_CUP
# date/time in the format below, 24-hour, as published (local kickoff time
# is fine -- this feeds a +/- 14 day window, not an exact-hour calculation).
FIXTURES: list[dict] = [
    # {"team": "Villa", "competition": "UEL", "kickoff": "2026-09-25 20:00", "round": "League phase MD2"},
    # {"team": "Palace", "competition": "UECL", "kickoff": "2026-09-25 18:45", "round": "League phase MD2"},
    # {"team": "Man Utd", "competition": "LEAGUE_CUP", "kickoff": "2026-09-24 19:45", "round": "Third round"},
]


def seed(season: int = CURRENT_SEASON) -> int:
    teams = read_sql(select(T.teams).where(T.teams.c.season == season))
    if teams.empty:
        log.warning("no teams ingested for season %s; run bootstrap ingest first", season)
        return 0

    by_short = {str(r["short_name"]).lower(): int(r["team_id"]) for _, r in teams.iterrows()}
    by_name = {str(r["name"]).lower(): int(r["team_id"]) for _, r in teams.iterrows()}

    now = datetime.utcnow()
    rows, skipped = [], []
    for f in FIXTURES:
        key = f["team"].strip().lower()
        team_id = by_short.get(key) or by_name.get(key)
        if team_id is None:
            skipped.append(f["team"])
            continue
        rows.append({
            "team_id": team_id, "season": season, "competition": f["competition"],
            "kickoff": datetime.strptime(f["kickoff"], "%Y-%m-%d %H:%M"),
            "round": f.get("round", ""), "source": "manual_seed", "known_at": now,
        })

    if rows:
        upsert(T.other_competition_fixtures, rows,
              ["team_id", "season", "competition", "kickoff"])
    if skipped:
        log.warning("could not resolve team names: %s -- check spelling against "
                    "the teams table", skipped)
    log.info("seeded %d cup/Europa fixture rows (%d skipped)", len(rows), len(skipped))
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    args = ap.parse_args()
    init_db()
    seed(args.season)
