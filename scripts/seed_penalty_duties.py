"""Manually maintain primary/backup penalty takers.

WHY THIS EXISTS
---------------
ingestion/fbref.py scrapes FBref for this same data, but FBref sits behind
Cloudflare bot-detection and returns 403 to non-browser requests -- reliably
enough that you should not count on the scraper working. This script is the
practical alternative: the list of primary penalty takers is short (one or
two players per club, ~20-30 rows total) and changes only a handful of times
a season (an injury, a new signing, a manager switching takers after a
miss). Five minutes of manual entry, checked every few gameweeks, is more
reliable than fighting a WAF.

USAGE
-----
Edit the PENALTY_TAKERS list below -- one entry per club, `web_name` must
match the FPL web_name exactly (check with `fplagent doctor` or the
`players` table if unsure) -- then run:

    python scripts/seed_penalty_duties.py

Re-run it any time the list changes; it's idempotent (upserts on the real
primary key, so re-running with the same data is a no-op).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from fplagent.config import CURRENT_SEASON
from fplagent.db import tables as T
from fplagent.db.repository import read_sql, upsert
from fplagent.db.session import init_db

# Fill this in and keep it current. `order`: 1 = first choice, 2 = backup.
# `confidence`: how sure you are this is still accurate -- 0.9 for a taker
# who's scored multiple penalties this season, lower for an untested guess.
#
# SEEDED ENTRIES BELOW came from a round of web search across several FPL
# set-piece-tracking sites, cross-checked for agreement across sources.
# Read this as a STARTING POINT, not gospel:
#   * Several of the sites returned read like low-effort/uncertain FPL
#     content -- vague, templated, occasionally self-contradictory between
#     articles on the same nominal topic. Where two+ independently-styled
#     sources agreed, confidence is set higher (0.85-0.95). Where only one
#     source mentioned a name, confidence is deliberately lower (0.5-0.6).
#   * Genuinely DISPUTED roles are left OUT on purpose rather than guessed
#     at -- forcing a pick the sources themselves disagree on would be
#     worse than leaving it to the empirical in-season estimate. As of
#     writing that's Arsenal (Saka vs Gyokeres -- both have scored with the
#     other on the pitch, unresolved) and Liverpool (Isak vs Szoboszlai --
#     no same-pitch test yet).
#   * `web_name` must match the FPL web_name EXACTLY. The names below are
#     best-effort spellings; the script reports anything that doesn't match
#     your actual players table so you can fix it rather than silently
#     dropping it.
#   * Corner/free-kick duty is NOT modelled anywhere in this codebase right
#     now (no code path reads it), so it isn't seeded here even where a
#     source mentioned it -- e.g. Rayan Cherki is reported as Man City's
#     corner taker, not a penalty threat, so he isn't in this list and
#     shouldn't be expected to show up via this table at all.
#
# Re-verify anything you're relying on for a real decision -- this is meant
# to save you the trouble of re-searching from scratch, not to replace a
# final gut-check.
PENALTY_TAKERS: list[dict] = [
    # -- high confidence: consistent across multiple independent sources,
    #    and consistent with each club's established pattern --
    {"web_name": "Haaland",  "order": 1, "confidence": 0.95},   # Man City
    {"web_name": "Marmoush", "order": 2, "confidence": 0.35},   # Man City backup
    {"web_name": "Fernandes", "order": 1, "confidence": 0.92},  # Man Utd (Bruno) -- verify web_name; some seasons FPL uses "B.Fernandes"
    {"web_name": "Palmer",   "order": 1, "confidence": 0.85},   # Chelsea
    {"web_name": "Thiago",   "order": 1, "confidence": 0.80},   # Brentford (Igor Thiago)
    {"web_name": "Mateta",   "order": 1, "confidence": 0.80},   # Crystal Palace
    {"web_name": "Buendia",  "order": 1, "confidence": 0.70},   # Aston Villa
    {"web_name": "Watkins",  "order": 2, "confidence": 0.30},   # Aston Villa backup

    # -- medium confidence: single clear source or a conflicting mention
    #    elsewhere. Worth a personal sanity check before you lean on these --
    {"web_name": "Wood",     "order": 1, "confidence": 0.55},   # Nott'm Forest (Chris Wood)
    {"web_name": "Solanke",  "order": 1, "confidence": 0.55},   # Tottenham
    {"web_name": "Woltemade","order": 1, "confidence": 0.45},   # Newcastle -- one other source floated Wissa instead, unresolved
    {"web_name": "Diarra",   "order": 1, "confidence": 0.40},   # Sunderland -- single source only

    # -- Arsenal and Liverpool deliberately NOT seeded: genuinely disputed
    #    across sources as of writing. Add them once a real penalty this
    #    season settles it, or if you're confident enough yourself.
]


def main(season: int = CURRENT_SEASON) -> None:
    init_db()
    players = read_sql(select(T.players).where(T.players.c.season == season))
    if players.empty:
        print("No players in the database for this season. Run `fplagent "
              "ingest` first.")
        return

    now = datetime.utcnow()
    rows, unmatched = [], []
    for entry in PENALTY_TAKERS:
        match = players[players["web_name"] == entry["web_name"]]
        if match.empty:
            unmatched.append(entry["web_name"])
            continue
        rows.append({
            "player_id": int(match.iloc[0]["player_id"]), "season": season,
            "duty": "penalties", "order": entry["order"],
            "confidence": entry.get("confidence", 0.8),
            "known_at": now, "source": "manual",
        })

    if unmatched:
        print(f"Could not match {len(unmatched)} name(s), check spelling "
              f"against web_name in the players table: {unmatched}")

    if not rows:
        print("Nothing to write -- PENALTY_TAKERS is empty or nothing matched. "
              "Edit the list at the top of this script.")
        return

    n = upsert(T.set_piece_duties, rows, ["player_id", "season", "duty"])
    print(f"Upserted {n} penalty duty rows for season {season}.")


if __name__ == "__main__":
    main()
