"""Champions League fixtures, for rotation-risk congestion counting.

WHY THIS EXISTS
----------------
`features/build.py`'s congestion count (matches in the last 14 days, the
leading indicator of rotation) only ever looked at FPL's own `fixtures`
table, i.e. Premier League matches. A team playing a Champions League tie on
the Tuesday between two league games is exactly as fatigued and exactly as
likely to rotate, and the model could not see it.

WHY ONLY THE CHAMPIONS LEAGUE
-------------------------------
football-data.org's free tier covers the Champions League and the big
domestic leagues, but not the Europa League, the Conference League, the FA
Cup, or the League Cup. Checked directly against their own competition list
before building this: there is no clean free API for the other four. Since
those are a real but smaller contributor to rotation than the Champions
League itself, and building four fragile scrapers for information that
changes on a known, slow schedule is not a good trade, they are handled by a
hand-maintained seed instead (`scripts/seed_cup_fixtures.py`), which writes
into the SAME `other_competition_fixtures` table this module writes into.
The congestion count in `features/build.py` reads that table as a whole and
does not care which source populated a given row.

USAGE
-----
    from fplagent.ingestion.european_fixtures import ingest_champions_league
    ingest_champions_league(season=2026)

Requires `FOOTBALL_DATA_API_KEY` (free, from football-data.org). Soft-fails
like the rest of the ingestion layer: no key, no network, no name match all
degrade the congestion signal rather than stopping the pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime

import requests
from sqlalchemy import delete, select

from fplagent.config import CURRENT_SEASON, FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE
from fplagent.db import tables as T
from fplagent.db.repository import read_sql, upsert

log = logging.getLogger(__name__)

COMPETITION_CODE = "CL"     # football-data.org's code for the Champions League


def fetch_champions_league_matches(api_key: str | None = None) -> list[dict]:
    """All scheduled/finished Champions League matches football-data.org
    currently has. One call covers the whole competition, well inside the
    free tier's 10 requests/minute."""
    api_key = api_key or FOOTBALL_DATA_API_KEY
    if not api_key:
        log.info("no FOOTBALL_DATA_API_KEY set; Champions League fixtures "
                 "will not be counted toward congestion. Get a free key at "
                 "https://www.football-data.org/client/register")
        return []
    url = f"{FOOTBALL_DATA_BASE}/competitions/{COMPETITION_CODE}/matches"
    try:
        r = requests.get(url, headers={"X-Auth-Token": api_key}, timeout=20)
        r.raise_for_status()
        return r.json().get("matches", [])
    except Exception as e:      # noqa: BLE001
        log.warning("Champions League fixture fetch failed, congestion count "
                    "will be missing these matches: %s", e)
        return []


def _match_team(name: str, teams: "pd.DataFrame") -> int | None:      # noqa: F821
    # football-data.org spells clubs "Manchester City FC", "Newcastle United
    # FC", "Tottenham Hotspur FC"; a raw fuzzy score against FPL's "Man City"
    # / "Newcastle" / "Spurs" falls well short of any safe threshold, so the
    # regular Champions League clubs were exactly the ones never matched.
    # The shared resolver (exact, alias, then a strict fuzzy step) fixes that.
    from fplagent.ingestion.name_matching import resolve_team_id
    return resolve_team_id(name, teams)


def ingest_champions_league(season: int = CURRENT_SEASON,
                            api_key: str | None = None) -> int:
    """Pull, match team names to our team_ids, persist into
    `other_competition_fixtures`. Only Premier League clubs matter for our
    congestion count, so a non-match (a Bundesliga side that our fuzzy match
    can't place, say) is simply skipped rather than logged as an error --
    most Champions League fixtures are between two clubs we don't track."""
    matches = fetch_champions_league_matches(api_key)
    if not matches:
        return 0

    teams = read_sql(select(T.teams).where(T.teams.c.season == season))
    if teams.empty:
        log.warning("no teams ingested for season %s; run bootstrap ingest first", season)
        return 0

    now = datetime.utcnow()
    rows = []
    for m in matches:
        kickoff_raw = m.get("utcDate")
        if not kickoff_raw:
            continue
        try:
            kickoff = datetime.fromisoformat(kickoff_raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        home = (m.get("homeTeam") or {}).get("name", "")
        away = (m.get("awayTeam") or {}).get("name", "")
        stage = m.get("stage", "")
        for name in (home, away):
            team_id = _match_team(name, teams)
            if team_id is None:
                continue
            match_id = m.get("id")
            fixture_key = (f"football-data:{match_id}:{team_id}" if match_id
                           else f"football-data:{season}:UCL:{team_id}:{kickoff:%Y%m%d%H%M}")
            rows.append({
                "team_id": team_id, "season": season, "competition": "UCL",
                "kickoff": kickoff.replace(tzinfo=None), "fixture_key": fixture_key,
                "round": stage, "source": "football_data_api", "known_at": now,
            })

    if rows:
        keys = [r.get("fixture_key") for r in rows if r.get("fixture_key")]
        if keys:
            # Kickoff is mutable; remove stale rows with the same external
            # identity before inserting the current scheduled time.
            from fplagent.db.session import session_scope
            with session_scope() as sess:
                sess.execute(delete(T.other_competition_fixtures).where(
                    T.other_competition_fixtures.c.fixture_key.in_(keys)))
        upsert(T.other_competition_fixtures, rows,
              ["team_id", "season", "competition", "kickoff"])
    log.info("Champions League: %d team-fixture rows matched to our teams", len(rows))
    return len(rows)
