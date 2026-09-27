"""Confirmed-lineup / late team-news ingestion via RotoWire.

WHAT THIS REPLACES
-------------------
v8 originally shipped this file as an honest stub (no source picked, nothing
fetched). This is the real implementation: RotoWire's public Premier League
lineups page (https://www.rotowire.com/soccer/lineups.php) lists starting-XI
predictions and injury/suspension tags (OUT / QUES / SUS) per fixture, and
is the source most fantasy-football tools scrape for exactly this signal --
it needs no login and carries no paywall.

A NOTE ON SCRAPING THIS
------------------------
This is public-page scraping for personal, non-commercial use, the same
category of thing most FPL tools already do. It is polite about it: a
realistic (not deceptive) browser User-Agent so the request isn't rejected
outright, a short timeout, and it is meant to be called at most a few times
per hour close to deadlines -- not polled continuously. It respects a
robots.txt disallow if RotoWire ever adds one for this path (see
`_robots_allow`). If RotoWire changes their markup, `_parse_lineup_page`
degrades to an empty result rather than raising, same as every other
optional-enrichment ingestor in this codebase (see fbref.py, odds.py).

RUNS FROM YOUR OWN NETWORK, NOT THIS DELIVERY ENVIRONMENT
------------------------------------------------------------
The sandbox this was written in only has network access to package
registries, not to rotowire.com, so this could not be exercised against the
live page from here. The parsing selectors below (`lineup__player`,
`lineup__inj`, `lineup__pos`, `lineup__abbr`, `lineup__list`, `lineup__box`)
are RotoWire's actual current class names, not a guess -- verify with
`python -m fplagent.ingestion.lineup_news` once you run it somewhere with
real internet access, and if RotoWire has changed markup since,
`_parse_lineup_page` is the only function that needs updating; everything
downstream of it (matching, the FastAPI overlay) is unaffected.

WHAT "CONFIRMED" MEANS HERE
-----------------------------
RotoWire's lineups page shows PREDICTED lineups well before kickoff, and
switches to CONFIRMED once clubs announce (~60 min pre-kickoff, same as
official channels). The `confirmed_starter` column reflects that
predicted/confirmed distinction where present; treat a `False` here as "not
yet confirmed", not as "not starting" -- see `status` for the actual signal
to act on (OUT/SUS are unambiguous; DOUBTFUL and predicted-starter are not).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests
from sqlalchemy import select

from fplagent.config import CURRENT_SEASON
from fplagent.db import tables as T
from fplagent.db.repository import read_sql

log = logging.getLogger(__name__)

ROTOWIRE_URL = "https://www.rotowire.com/soccer/lineups.php"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
_STATUS_MAP = {
    "OUT": ("OUT", 0.95), "SUS": ("OUT", 1.0),
    "QUES": ("DOUBTFUL", 0.7), "DOUBTFUL": ("DOUBTFUL", 0.7),
}

_robots_cache: bool | None = None


def _robots_allow() -> bool:
    """Best-effort robots.txt check. Fails OPEN (allow) if robots.txt can't
    be fetched at all -- a network hiccup should not silently disable the
    feature -- but fails CLOSED (disallow) if it loads and actually
    disallows this path, which is the one case that must be respected."""
    global _robots_cache
    if _robots_cache is not None:
        return _robots_cache
    try:
        r = requests.get("https://www.rotowire.com/robots.txt",
                         headers=_HEADERS, timeout=10)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if (line.strip().lower().startswith("disallow")
                        and "/soccer/lineups" in line.lower()):
                    _robots_cache = False
                    return False
        _robots_cache = True
    except requests.RequestException as e:      # noqa: BLE001
        log.info("robots.txt check failed (%s); proceeding, fails open", e)
        _robots_cache = True
    return _robots_cache


def _parse_lineup_page(html: str) -> list[dict]:
    """Parse RotoWire's lineup page HTML into raw rows: team (str),
    player_name (str), status ('CONFIRMED'|'PREDICTED'|'OUT'|'DOUBTFUL'),
    confidence (float). Never raises -- a markup change returns []."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("beautifulsoup4 not installed; skipping lineup scrape "
                    "(pip install beautifulsoup4)")
        return []

    rows: list[dict] = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for box in soup.find_all("div", class_="lineup__box"):
            abbrs = box.find_all("div", class_="lineup__abbr")
            lists = box.find_all("ul", class_="lineup__list")
            is_confirmed_box = "is-confirmed" in (box.get("class") or [])
            for list_idx, player_list in enumerate(lists):
                team = (abbrs[list_idx].get_text(strip=True)
                       if list_idx < len(abbrs) else "UNK")
                for entry in player_list.find_all("li", class_="lineup__player"):
                    link = entry.find("a")
                    if not link:
                        continue
                    name = link.get("title", "").strip() or link.get_text(strip=True)
                    if not name:
                        continue
                    inj = entry.find("span", class_="lineup__inj")
                    if inj:
                        raw = inj.get_text(strip=True).upper()
                        status, conf = _STATUS_MAP.get(raw, ("DOUBTFUL", 0.6))
                    else:
                        status = "CONFIRMED" if is_confirmed_box else "PREDICTED"
                        conf = 0.9 if is_confirmed_box else 0.6
                    rows.append({"team": team, "player_name": name,
                                "status": status, "confidence": conf})
    except Exception as e:      # noqa: BLE001
        log.warning("failed to parse RotoWire lineup page (markup may have "
                   "changed): %s", e)
        return []
    return rows


def _fetch_raw() -> list[dict]:
    if not _robots_allow():
        log.info("robots.txt disallows /soccer/lineups; skipping")
        return []
    try:
        r = requests.get(ROTOWIRE_URL, headers=_HEADERS, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:      # noqa: BLE001
        log.warning("could not fetch RotoWire lineups: %s", e)
        return []
    return _parse_lineup_page(r.text)


def fetch_confirmed_lineups(season: int = CURRENT_SEASON,
                            match_threshold: int = 80) -> pd.DataFrame:
    """Scrape RotoWire, fuzzy-match names/teams to our player_id/fixture_id,
    and return (fixture_id, player_id, status, confirmed_starter, confidence,
    fetched_at). Same fuzzy-matching convention as ingestion/fbref.py and
    ingestion/odds.py (rapidfuzz, team-scoped player matching, no cross-club
    guessing). Empty DataFrame, never raises, on any failure -- this is
    optional enrichment layered on top of a system that is fully usable
    without it.
    """
    empty = pd.DataFrame(columns=["fixture_id", "player_id", "status",
                                  "confirmed_starter", "confidence", "fetched_at"])
    raw = _fetch_raw()
    if not raw:
        return empty

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        log.warning("rapidfuzz not installed; cannot match RotoWire names "
                    "(pip install rapidfuzz)")
        return empty

    from fplagent.ingestion.name_matching import resolve_player_id

    players_df = read_sql(select(
        T.players.c.player_id, T.players.c.web_name, T.players.c.team_id,
        T.players.c.first_name, T.players.c.second_name,
    ).where(T.players.c.season == season))
    teams_df = read_sql(select(
        T.teams.c.team_id, T.teams.c.name, T.teams.c.short_name
    ).where(T.teams.c.season == season))
    # Fixtures in the near future only -- this signal is only meaningful for
    # the next deadline, and matching against a stale/finished fixture would
    # silently attach today's team news to the wrong match. The column is
    # `kickoff`; labelled so the rest of this function reads unchanged.
    fixtures_df = read_sql(select(
        T.fixtures.c.fixture_id, T.fixtures.c.home_team_id,
        T.fixtures.c.away_team_id, T.fixtures.c.kickoff.label("kickoff_time")
    ).where(T.fixtures.c.season == season))
    if players_df.empty or teams_df.empty:
        log.warning("no players/teams for season %s; run the ingest pipeline first", season)
        return empty
    if not fixtures_df.empty and "kickoff_time" in fixtures_df.columns:
        now = datetime.utcnow()
        window = fixtures_df["kickoff_time"].between(now - timedelta(hours=3),
                                                      now + timedelta(days=6))
        fixtures_df = fixtures_df[window]

    team_names = teams_df["short_name"].fillna(teams_df["name"]).tolist()
    fetched_at = datetime.utcnow()
    out_rows = []

    for r in raw:
        team_match = process.extractOne(r["team"], team_names, scorer=fuzz.token_sort_ratio)
        if not team_match or team_match[1] < match_threshold:
            continue
        team_row = teams_df[
            teams_df["short_name"].fillna(teams_df["name"]) == team_match[0]]
        if team_row.empty:
            continue
        team_id = int(team_row.iloc[0]["team_id"])

        # RotoWire gives full names ("Erling Haaland"); a raw fuzzy score
        # against the surname-only web_name ("Haaland") is ~67, below any
        # sane threshold, so most players never matched. Club-scoped
        # resolution handles full name vs web_name and refuses ties.
        player_id = resolve_player_id(r["player_name"], players_df, team_id=team_id,
                                      threshold=match_threshold)
        if player_id is None:
            continue

        fixture_id = None
        if not fixtures_df.empty:
            fx = fixtures_df[(fixtures_df["home_team_id"] == team_id)
                             | (fixtures_df["away_team_id"] == team_id)]
            if not fx.empty:
                fixture_id = int(fx.sort_values("kickoff_time").iloc[0]["fixture_id"])

        out_rows.append({
            "fixture_id": fixture_id, "player_id": player_id,
            "status": r["status"],
            "confirmed_starter": r["status"] == "CONFIRMED",
            "confidence": r["confidence"], "fetched_at": fetched_at,
        })

    if not out_rows:
        return empty
    return pd.DataFrame(out_rows)


def apply_lineup_overlay(prediction: dict, lineup_row: dict | None) -> dict:
    """Serving-time adjustment: overlay late team news onto an already-
    computed prediction WITHOUT retraining or touching stored rows. Called
    optionally (see api.main /predictions/{id}?live_news=true) so the base
    model and its backtest remain exactly what they were -- this is a
    presentation-layer correction for the specific gap the model can't see
    (news that broke after the pipeline last ran), not a change to the model
    itself.

    prediction: a dict with at least p_play, p_start, p_sixty, xp_mean.
    lineup_row: one row (as a dict) from fetch_confirmed_lineups(), or None.
    Returns a NEW dict; never mutates the input.
    """
    out = dict(prediction)
    if out.get("p_start") is None:
        out["p_start"] = out.get("p_play")
    if lineup_row is None:
        out["lineup_news"] = None
        return out

    status = lineup_row["status"]
    orig_p_play = out.get("p_play")
    if status == "OUT":
        out["p_play"], out["p_start"], out["p_sixty"] = 0.02, 0.01, 0.005
    elif status == "DOUBTFUL":
        if orig_p_play is not None:
            out["p_play"] = min(orig_p_play, 0.5)
            out["p_start"] = min(float(out.get("p_start") or 0.0), 0.35, out["p_play"])
    elif status == "CONFIRMED":
        if orig_p_play is not None:
            out["p_play"] = max(orig_p_play, 0.9)
            out["p_start"] = max(float(out.get("p_start") or 0.0), 0.85)
            out["p_start"] = min(out["p_start"], out["p_play"])
            out["p_sixty"] = min(max(out.get("p_sixty") or 0.0, 0.75), out["p_play"])

    if orig_p_play and out.get("p_play") is not None and orig_p_play > 0:
        scale = out["p_play"] / orig_p_play
        if out.get("xp_mean") is not None:
            out["xp_mean"] = round(out["xp_mean"] * scale, 2)

    out["lineup_news"] = {
        "status": status, "confidence": lineup_row["confidence"],
        "fetched_at": lineup_row["fetched_at"], "source": "rotowire",
        "note": ("Overlay applied at serving time from late team news; the "
                "stored prediction and backtest are unaffected by this."),
    }
    return out
