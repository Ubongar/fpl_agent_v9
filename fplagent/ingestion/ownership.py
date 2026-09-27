"""Effective ownership, sampled from FPL's own public API.

WHAT EO IS AND WHY OWNERSHIP ISN'T IT
---------------------------------------
`optimizers/objective.py`'s rank-aware value function needs effective
ownership: the share of the field's TOTAL POINTS that a player accounts for,
which is ownership PLUS captaincy (worth double) plus triple-captaincy (worth
triple, on chip weeks). A 30%-owned player who is rarely captained and a
30%-owned player who half the league captains are very different threats to
your rank, and raw ownership cannot tell them apart. v7 had no real EO and
fell back to `selected_by_percent`, silently treating every owner as an
equally-likely captain.

THERE IS NO OFFICIAL EO ENDPOINT
-----------------------------------
FPL's `bootstrap-static` publishes ownership, never captaincy. Third-party
sites that show EO (LiveFPL, Fantasy Football Scout and others) all compute
it the same way this module does: sample a large number of managers from
FPL's own PUBLIC, undocumented-but-open endpoints and count captaincy choices
directly. This is calling FPL's own API repeatedly, not scraping a
third-party site -- no different in kind from `ingestion/fpl_client.py`
already hitting `bootstrap-static` and `fixtures`, just aimed at more
endpoints.

    GET /api/leagues-classic/{league_id}/standings/?page_standings={n}
    GET /api/entry/{entry_id}/event/{gw}/picks/

The "Overall" classic league has the well-known, publicly stated id 314; its
standings are simply the top of the entire game, 50 managers per page. A few
hundred to a couple of thousand of the most competitive managers is plenty to
estimate captaincy shares to within a percentage point or two, and DOES
introduce a "the top managers, not a uniform sample of everyone" bias -- worth
being upfront about, since the top of the game trends toward the consensus
captain even more than the average manager does. That is a real limitation
of this method, not just of this implementation.

BEING A GOOD CITIZEN OF A PUBLIC-BUT-UNOFFICIAL API
-----------------------------------------------------
This is public data with no documented rate limit, which is a reason for
MORE caution, not less. A real delay between requests, a real user agent, and
a hard cap on how many entries get sampled per run are non-negotiable here.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import requests
from sqlalchemy import select

from fplagent.config import CURRENT_SEASON
from fplagent.db import tables as T
from fplagent.db.repository import upsert

log = logging.getLogger(__name__)

OVERALL_LEAGUE_ID = 314          # FPL's own "Overall" classic league
BASE = "https://fantasy.premierleague.com/api"
USER_AGENT = "fpl-agent/1 (+https://github.com/Ubongar/fpl_agent_v8; polite research use)"
REQUEST_DELAY_SECONDS = 0.6      # a real gap between requests, not a burst


def _get(path: str) -> dict | None:
    try:
        r = requests.get(f"{BASE}{path}", headers={"User-Agent": USER_AGENT}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:      # noqa: BLE001
        log.debug("ownership sampler: request to %s failed: %s", path, e)
        return None


def sample_manager_ids(n_managers: int = 1000) -> list[int]:
    """Entry ids of the top N managers in the Overall league, by rank."""
    ids: list[int] = []
    page = 1
    while len(ids) < n_managers:
        data = _get(f"/leagues-classic/{OVERALL_LEAGUE_ID}/standings/?page_standings={page}")
        if not data:
            break
        results = (data.get("standings") or {}).get("results", [])
        if not results:
            break
        ids.extend(int(r["entry"]) for r in results)
        if not (data.get("standings") or {}).get("has_next", False):
            break
        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    return ids[:n_managers]


def sample_effective_ownership(gw: int, season: int = CURRENT_SEASON,
                               n_managers: int = 1000,
                               n_managers_cap: int = 2000) -> int:
    """Sample top-manager picks for one gameweek and persist per-player EO.

    Soft-fails throughout: FPL's site being unreachable (as it is from this
    sandbox's restricted network) degrades captaincy decisions to raw
    ownership rather than stopping anything -- see
    `optimizers/objective.effective_value`'s fallback.
    """
    n_managers = min(n_managers, n_managers_cap)
    manager_ids = sample_manager_ids(n_managers)
    if not manager_ids:
        log.info("ownership sampler: could not reach the FPL API (this is "
                 "expected in a network-restricted environment). Effective "
                 "ownership will fall back to raw selected_by_percent.")
        return 0

    owned: dict[int, int] = {}
    captained: dict[int, int] = {}
    triple: dict[int, int] = {}
    # Sum of pick multipliers: 0 benched, 1 starting, 2 captain, 3 triple
    # captain (bench boost makes the bench 1). This IS effective ownership.
    multiplier_sum: dict[int, int] = {}
    n_sampled = 0

    for entry_id in manager_ids:
        picks = _get(f"/entry/{entry_id}/event/{gw}/picks/")
        time.sleep(REQUEST_DELAY_SECONDS)
        if not picks or "picks" not in picks:
            continue
        n_sampled += 1
        for p in picks["picks"]:
            pid = int(p["element"])
            owned[pid] = owned.get(pid, 0) + 1
            mult = int(p.get("multiplier", 1))
            multiplier_sum[pid] = multiplier_sum.get(pid, 0) + mult
            if p.get("is_captain"):
                captained[pid] = captained.get(pid, 0) + 1
                if mult >= 3:
                    triple[pid] = triple.get(pid, 0) + 1

    if n_sampled == 0:
        return 0

    now = datetime.utcnow()
    rows = []
    for pid, n_owned in owned.items():
        own_pct = n_owned / n_sampled
        cap_pct = captained.get(pid, 0) / n_sampled
        tc_pct = triple.get(pid, 0) / n_sampled
        # EO is the share of the field's points the player accounts for: the
        # mean pick multiplier. That is ownership + captaincy + triple
        # captaincy for STARTERS, but a benched pick (multiplier 0) scores
        # nothing for its owner, so it must not count. The old
        # own + cap + tc form gave every benched fourth defender and backup
        # keeper full EO.
        eo = multiplier_sum.get(pid, 0) / n_sampled
        rows.append({
            "player_id": pid, "season": season, "gw": gw,
            "sample_size": n_sampled, "ownership_pct": round(own_pct, 4),
            "captain_pct": round(cap_pct, 4), "triple_captain_pct": round(tc_pct, 4),
            "effective_ownership": round(eo, 4), "pulled_at": now,
        })

    upsert(T.effective_ownership, rows, ["player_id", "season", "gw"])
    log.info("effective ownership: sampled %d managers, %d players", n_sampled, len(rows))
    return len(rows)
