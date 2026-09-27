"""Fit real league-translation coefficients from FBref season exports.

    python -m scripts.fit_league_translation \
        --pl data/fbref/premier_league.csv \
        --league Championship=data/fbref/championship.csv \
        --league Bundesliga=data/fbref/bundesliga.csv \
        ...

Writes into `league_translation`, `external_player_map` and
`external_player_season` (see `ingestion/league_snapshot.py` for what each
holds and why). Also writes a plain-JSON snapshot to
`data/league_translation_fitted.json` so the fitted numbers are inspectable
without a database, and so they ship with the repo as a real, reproducible
starting point rather than nothing.

Re-run this every few gameweeks as more Premier League data accumulates --
the ratios that fail the reliability gate early in a season routinely clear
it by November. See ingestion/league_snapshot.py's module docstring for why
the gate exists and what it protects against.
"""
from __future__ import annotations

import argparse
import json
import logging

from fplagent.config import CURRENT_SEASON
from fplagent.db.session import init_db

log = logging.getLogger(__name__)


def _player_id_lookup(season: int) -> dict[str, int]:
    """web_name -> player_id, so matched transfer-cohort rows can be linked
    to a real ingested player for the per-player cold-start path, not just
    the aggregate ratio."""
    try:
        from sqlalchemy import select
        from fplagent.db import tables as T
        from fplagent.db.repository import read_sql
        df = read_sql(select(T.players).where(T.players.c.season == season))
    except Exception:      # noqa: BLE001
        return {}
    if df.empty:
        return {}
    return dict(zip(df["web_name"], df["player_id"].astype(int)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pl", required=True, help="Premier League FBref Standard Stats export")
    ap.add_argument("--league", action="append", default=[], metavar="NAME=PATH",
                    help="a foreign league export, repeatable")
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    ap.add_argument("--json-out", default="data/league_translation_fitted.json")
    args = ap.parse_args()

    leagues = {}
    for item in args.league:
        if "=" not in item:
            ap.error(f"--league expects NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        leagues[name] = path
    if not leagues:
        ap.error("at least one --league NAME=PATH is required")

    init_db()
    from fplagent.ingestion.league_snapshot import fit_and_persist
    ids = _player_id_lookup(args.season)
    summary = fit_and_persist(args.pl, leagues, player_id_by_name=ids, season=args.season)

    print(f"Fitted {summary['total_fitted']} ratios across {len(leagues)} leagues, "
         f"{summary['total_gate_passed']} cleared the reliability gate.\n")
    for league, d in summary["leagues"].items():
        if "error" in d:
            print(f"  {league}: {d['error']}")
            continue
        print(f"  {league}: {d['n_matched_players']} matched players, "
             f"{d['external_records_linked']} linked to a Premier League player_id")
        for r in d["ratios"]:
            flag = "OK  " if r["meets_gate"] else "skip"
            print(f"      [{flag}] {r['position_group']:<4} {r['metric']:<7} "
                 f"ratio={r['ratio']:<7} n={r['n_players']:<3} "
                 f"target_min={r['target_minutes']:.0f}")

    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
