"""Fetch prior-season fixtures from the public vaastav dataset.

WHY THIS EXISTS
---------------
By GW5 the Dixon-Coles fit has ~50 matches to work with. Forty-two
parameters fit to a hundred goal observations, giving it rho at the edge
of the legal region and ratings that are all pinned near 1.0 because the
prior dominates. The model literally cannot tell the difference between
a good team and a bad one on five gameweeks of results.

Prior seasons fix this. The same 20 clubs have played ~760 more matches
over the preceding two seasons, and while the *teams* have changed, the
*clubs* have not: Arsenal is still a better side than Burnley, and that
was true in 2024-25 as it is now. The vaastav dataset has those results
for free, published as CSVs on GitHub.

WHAT IT WRITES
--------------
Rows into `historical_fixtures`, keyed on the persistent team `code`
rather than on `team_id`, because FPL assigns new ids to promoted clubs
every season. The consumer (`features.build.load_fixtures_with_history`)
maps those codes back onto the current season's `team_id`s at read time
and drops any club that is no longer in the Premier League.

WHAT IT DOES NOT DO
-------------------
It does not touch `fixtures` or `player_fixture_stats`. Prior-season
results are used only to stabilise the team-strength fit; nothing needs
player-level history from prior seasons, because the share estimator
already falls back to a positional prior when evidence is thin, and
reconciling player identities across seasons is not worth the marginal
improvement.

USAGE
-----
    python scripts/backfill_history.py            # 2 seasons before CURRENT_SEASON
    python scripts/backfill_history.py --seasons 3
    python scripts/backfill_history.py --season 2023   # one specific season

Safe to re-run: rows are upserted on (season, source_fixture_id).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Allow running as a script (`python scripts/backfill_history.py`) from the
# project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplagent.config import CURRENT_SEASON, VAASTAV_BASE  # noqa: E402
from fplagent.db import tables as T  # noqa: E402
from fplagent.db.repository import upsert  # noqa: E402
from fplagent.db.session import init_db  # noqa: E402

log = logging.getLogger(__name__)


def _season_folder(season: int) -> str:
    """The vaastav repository names folders like `2024-25`, not `2024`."""
    return f"{season}-{str(season + 1)[-2:]}"


def _fetch_team_codes(season: int) -> dict[int, int]:
    """FPL team_id (for that season) -> persistent code.

    Every historical fixture has team_h/team_a as the FPL team_id OF THAT
    SEASON. We need the persistent `code` to map onto the current season's
    team list, and the only place that mapping lives is teams.csv.
    """
    url = f"{VAASTAV_BASE}/{_season_folder(season)}/teams.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:      # noqa: BLE001
        log.warning("could not fetch teams.csv for %s from %s: %s", season, url, e)
        return {}
    if "id" not in df.columns or "code" not in df.columns:
        log.warning("teams.csv for %s is missing 'id' or 'code' columns", season)
        return {}
    return {int(r["id"]): int(r["code"])
            for _, r in df.iterrows()
            if pd.notna(r["id"]) and pd.notna(r["code"])}


def _fetch_fixtures(season: int, code_by_team: dict[int, int]) -> list[dict]:
    """Read fixtures.csv for one season, map team ids to codes, emit rows."""
    url = f"{VAASTAV_BASE}/{_season_folder(season)}/fixtures.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:      # noqa: BLE001
        log.warning("could not fetch fixtures.csv for %s from %s: %s", season, url, e)
        return []

    # Defensive column checks. The vaastav schema has been stable for years
    # but this script should fail loud rather than write partial data.
    required = {"id", "event", "team_h", "team_a", "kickoff_time",
                "finished", "team_h_score", "team_a_score"}
    missing = required - set(df.columns)
    if missing:
        log.warning("fixtures.csv for %s is missing columns %s; skipping",
                    season, sorted(missing))
        return []

    rows: list[dict] = []
    for _, r in df.iterrows():
        if pd.isna(r["event"]):
            # Postponed fixture with no rescheduled gameweek. Skip.
            continue
        home_code = code_by_team.get(int(r["team_h"])) if pd.notna(r["team_h"]) else None
        away_code = code_by_team.get(int(r["team_a"])) if pd.notna(r["team_a"]) else None
        if home_code is None or away_code is None:
            # A fixture whose team is not in teams.csv is a data integrity
            # problem for that season's pull. Skip rather than crash.
            continue
        kickoff = pd.to_datetime(r["kickoff_time"], errors="coerce")
        kickoff_dt = (kickoff.to_pydatetime() if pd.notna(kickoff) else None)
        finished = bool(r["finished"]) if pd.notna(r["finished"]) else False
        # A finished match is knowable ~3 hours after kickoff.
        known_at = (kickoff_dt + timedelta(hours=3)) if kickoff_dt else datetime.utcnow()
        rows.append({
            "season": season,
            "source_fixture_id": int(r["id"]),
            "gw": int(r["event"]),
            "home_team_code": int(home_code),
            "away_team_code": int(away_code),
            "kickoff": kickoff_dt,
            "finished": finished,
            "home_goals": (int(r["team_h_score"]) if finished and pd.notna(r["team_h_score"])
                           else None),
            "away_goals": (int(r["team_a_score"]) if finished and pd.notna(r["team_a_score"])
                           else None),
            "known_at": known_at,
            "source": "vaastav",
        })
    return rows


def backfill_season(season: int) -> int:
    """Fetch, parse and upsert one prior season. Returns row count."""
    log.info("backfilling season %s (%s)", season, _season_folder(season))
    codes = _fetch_team_codes(season)
    if not codes:
        log.warning("no team-code mapping for %s; skipping", season)
        return 0
    rows = _fetch_fixtures(season, codes)
    if not rows:
        log.warning("no fixtures fetched for %s", season)
        return 0
    n = upsert(T.historical_fixtures, rows, ["season", "source_fixture_id"])
    finished = sum(1 for r in rows if r["finished"])
    log.info("season %s: %d fixtures written (%d finished)", season, n, finished)
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None,
                        help="backfill a single specific season "
                             "(e.g. 2024 for 2024/25). Overrides --seasons.")
    parser.add_argument("--seasons", type=int, default=2,
                        help="how many prior seasons to backfill before "
                             "CURRENT_SEASON (default 2)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    init_db()

    if args.season is not None:
        seasons = [args.season]
    else:
        seasons = [CURRENT_SEASON - k for k in range(1, args.seasons + 1)]

    total = 0
    for s in seasons:
        try:
            total += backfill_season(s)
        except Exception as e:      # noqa: BLE001
            log.error("backfill for season %s failed: %s", s, e, exc_info=True)

    log.info("backfill complete: %d fixtures written across %d season(s)",
             total, len(seasons))
    return 0


if __name__ == "__main__":
    sys.exit(main())