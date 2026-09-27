"""Ingestion: bootstrap, fixtures, gameweeks, per-fixture player stats.

Design rules enforced here, each one a v5 bug made structurally impossible:

* Cumulative values go ONLY to player_season_snapshots, with season_ prefixes.
* Per-fixture values go ONLY to player_fixture_stats, keyed by fixture_id.
  Nothing diffs cumulative totals to fake per-gameweek figures any more --
  element-summary already returns true per-fixture rows, so the diffing
  approach v5 used was both fragile and unnecessary.
* The current gameweek is read from the API's own events[], never configured.
* Every row records known_at, so the backtest can reconstruct the past. For
  per-fixture rows that is when the match finished, not when this system
  happened to pull it (see `_history_row`).

NON-FPL FIXTURES (v9)
---------------------
`other_competition_fixtures` was defined in the schema and read by
`features.build._load_other_competition_fixtures` from day one, but nothing
ever wrote to it. `ingestion/european_fixtures.py` was written to do so via
football-data.org's free Champions League endpoint, and `scripts/
seed_cup_fixtures.py` was written to cover the rest by hand. Neither was
called from `run_full_ingest`, so the table stayed empty and the minutes
model's `team_matches_last_14d` feature came out as a constant 2 for every
player -- which is why it kept appearing in the "dropped constant feature"
list. Both are now called from `run_full_ingest` and both soft-fail on a
missing key or a network outage, so a bad day at football-data.org degrades
the congestion signal but does not stop the pipeline.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd

from fplagent.config import CURRENT_SEASON
from fplagent.db import tables as T
from fplagent.ingestion.fpl_client import FPLClient
from fplagent.ingestion.quality import check_bootstrap, check_fixture_stats
from sqlalchemy import insert, select, update
from fplagent.db.repository import read_sql, upsert
from fplagent.db.session import session_scope

# fbref.py is imported lazily inside run_full_ingest, not here. It pulls in
# rapidfuzz, an optional dependency -- a top-level import would mean a
# machine without rapidfuzz installed fails to import this module at all,
# taking down the entire ingestion pipeline over one missing optional
# package used by an enrichment step.

log = logging.getLogger(__name__)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# A per-fixture stat line is knowable once the match is over and bonus is
# confirmed. Three hours after kickoff covers both; the demo uses the same.
KNOWN_AFTER_KICKOFF = timedelta(hours=3)


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _f_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _i_or_none(value) -> int | None:
    """Set-piece order is meaningfully absent for most players; 0 would be a
    lie and would make everyone a taker."""
    if value in (None, "", 0):
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _i(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def ingest_bootstrap(client: FPLClient | None = None,
                     season: int = CURRENT_SEASON) -> dict:
    client = client or FPLClient()
    data = client.bootstrap()

    report = check_bootstrap(data)
    report.persist()
    if report.failed:
        failed = [c.name for c in report.checks if c.failed]
        raise RuntimeError(f"bootstrap quality gate failed: {failed}")

    # `code` is FPL's persistent club identifier. It survives promotion and
    # relegation, unlike `team_id` which changes every season. Storing it
    # is what lets `load_fixtures_with_history` join prior-season fixtures
    # onto the current season's team list.
    team_rows = [{
        "team_id": t["id"], "season": season, "name": t["name"],
        "short_name": t["short_name"],
        "code": t.get("code"),
    } for t in data["teams"]]
    upsert(T.teams, team_rows, ["team_id", "season"])

    player_rows = [{
        "player_id": p["id"], "season": season, "web_name": p["web_name"],
        "first_name": p.get("first_name"), "second_name": p.get("second_name"),
        "team_id": p["team"], "position": POSITION_MAP.get(p["element_type"]),
        "birth_date": p.get("birth_date"), "now_cost": p["now_cost"] / 10.0,
        "penalties_order": _i_or_none(p.get("penalties_order")),
        "corners_order": _i_or_none(p.get("corners_and_indirect_freekicks_order")),
        "direct_fk_order": _i_or_none(p.get("direct_freekicks_order")),
    } for p in data["elements"]]
    upsert(T.players, player_rows, ["player_id", "season"])

    gw_rows = [{
        "season": season, "gw": e["id"],
        "deadline_time": _parse_dt(e.get("deadline_time")),
        "is_current": bool(e.get("is_current")),
        "is_next": bool(e.get("is_next")),
        "finished": bool(e.get("finished")),
        "average_entry_score": _f(e.get("average_entry_score")),
    } for e in data["events"]]
    upsert(T.gameweeks, gw_rows, ["season", "gw"])

    current = next((e["id"] for e in data["events"] if e.get("is_current")), None)
    nxt = next((e["id"] for e in data["events"] if e.get("is_next")), None)

    snap_rows = [{
        "player_id": p["id"], "season": season,
        "pulled_at": datetime.utcnow(), "as_of_gw": current or 0,
        "season_minutes": _i(p.get("minutes")),
        "season_total_points": _i(p.get("total_points")),
        "season_expected_goals": _f(p.get("expected_goals")),
        "season_expected_assists": _f(p.get("expected_assists")),
        "season_expected_goals_conceded": _f(p.get("expected_goals_conceded")),
        "form": _f(p.get("form")),
        "selected_by_percent": _f(p.get("selected_by_percent")),
        "transfers_in_event": _i(p.get("transfers_in_event")),
        "transfers_out_event": _i(p.get("transfers_out_event")),
        "now_cost": p["now_cost"] / 10.0,
        "team_id": p.get("team"),
        "position": POSITION_MAP.get(p.get("element_type")),
        "status": p.get("status"),
        "news": p.get("news", ""),
        "chance_of_playing_next_round": (
            None if p.get("chance_of_playing_next_round") is None
            else _f(p.get("chance_of_playing_next_round"))),
        "ep_next": _f_or_none(p.get("ep_next")),
        "season_yellow_cards": _i(p.get("yellow_cards")),
    } for p in data["elements"]]
    from fplagent.db.repository import insert_rows
    insert_rows(T.player_season_snapshots, snap_rows)

    n_duties = ingest_set_piece_duties(data, season)

    log.info("bootstrap: %d players, %d teams, current_gw=%s next_gw=%s (%s)",
             len(player_rows), len(team_rows), current, nxt, report.summary())
    return {"players": len(player_rows), "teams": len(team_rows),
            "declared_set_piece_duties": n_duties,
            "current_gw": current, "next_gw": nxt, "quality": report.summary()}


def ingest_fixtures(client: FPLClient | None = None,
                    season: int = CURRENT_SEASON) -> int:
    """v5 defined fetch_fixtures() and never called it, so the fixtures table
    was empty -- which meant there were no opponents, no home/away flags, and
    no match results to fit team strength against. Everything downstream was
    therefore unreachable. This is the single most important missing call in
    the entire v5 codebase."""
    client = client or FPLClient()
    data = client.fixtures()
    rows = [{
        "fixture_id": f["id"], "season": season, "gw": f.get("event"),
        "home_team_id": f["team_h"], "away_team_id": f["team_a"],
        "kickoff": _parse_dt(f.get("kickoff_time")),
        "finished": bool(f.get("finished")),
        "home_goals": f.get("team_h_score"), "away_goals": f.get("team_a_score"),
    } for f in data]
    upsert(T.fixtures, rows, ["fixture_id", "season"])
    log.info("fixtures: %d rows (%d finished)", len(rows),
             sum(1 for r in rows if r["finished"]))
    return len(rows)


def ingest_other_competition_fixtures(season: int = CURRENT_SEASON) -> dict:
    """Populate `other_competition_fixtures` from both available sources.

    Two calls, both soft-failing:

      * `ingest_champions_league` hits football-data.org's free tier for
        Champions League fixtures. Skipped silently if no
        FOOTBALL_DATA_API_KEY is configured, and returns 0 on any network
        failure.
      * `seed_cup_fixtures.seed` reads the manually-maintained list in
        `scripts/seed_cup_fixtures.py` for Europa League, Conference League,
        FA Cup and League Cup. Returns 0 if that file's FIXTURES list is
        empty (the current state), which is expected and harmless.

    Both write into the same table; the congestion feature reads the table
    as a whole and does not care which source populated a given row.
    """
    out: dict[str, int] = {"champions_league": 0, "manual_cup_seed": 0}

    try:
        from fplagent.ingestion.european_fixtures import ingest_champions_league
        out["champions_league"] = ingest_champions_league(season)
    except Exception as e:      # noqa: BLE001
        log.warning("Champions League fixture ingestion failed "
                    "(non-fatal, congestion signal will be missing these "
                    "matches): %s", e)

    try:
        from scripts.seed_cup_fixtures import seed as seed_cups
        out["manual_cup_seed"] = seed_cups(season)
    except Exception as e:      # noqa: BLE001
        log.warning("Manual cup fixture seed failed (non-fatal): %s", e)

    return out


# ---------------------------------------------------------------------------
# Per-fixture performance
# ---------------------------------------------------------------------------

_HISTORY_FIELDS = {
    "minutes": "minutes", "starts": "starts", "total_points": "total_points",
    "goals_scored": "goals_scored", "assists": "assists",
    "clean_sheets": "clean_sheets", "goals_conceded": "goals_conceded",
    "own_goals": "own_goals", "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed", "yellow_cards": "yellow_cards",
    "red_cards": "red_cards", "saves": "saves", "bonus": "bonus", "bps": "bps",
}
_HISTORY_FLOATS = {
    "influence": "influence", "creativity": "creativity", "threat": "threat",
    "ict_index": "ict_index", "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "expected_goal_involvements": "expected_goal_involvements",
    "expected_goals_conceded": "expected_goals_conceded",
}


def _history_row(pid: int, season: int, h: dict) -> dict | None:
    fixture_id = h.get("fixture")
    if fixture_id is None:
        return None
    # known_at is when the fact became knowable, which for a finished match
    # is shortly after kickoff -- not the moment this system pulled it.
    now = datetime.utcnow()
    kickoff = _parse_dt(h.get("kickoff_time"))
    known_at = min(now, kickoff + KNOWN_AFTER_KICKOFF) if kickoff else now
    row = {
        "player_id": pid, "season": season, "fixture_id": int(fixture_id),
        "gw": _i(h.get("round")),
        "opponent_team_id": _i(h.get("opponent_team")),
        "was_home": bool(h.get("was_home")),
        "value": _f(h.get("value")) / 10.0,
        "source": "element_summary",
        "known_at": known_at,
    }
    for src, dest in _HISTORY_FIELDS.items():
        row[dest] = _i(h.get(src))
    for src, dest in _HISTORY_FLOATS.items():
        row[dest] = _f(h.get(src))

    dc = h.get("defensive_contribution")
    cbi = _i(h.get("clearances_blocks_interceptions"))
    rec = _i(h.get("recoveries"))
    tkl = _i(h.get("tackles"))
    row["clearances_blocks_interceptions"] = cbi
    row["recoveries"] = rec
    row["tackles"] = tkl
    row["cbit"] = cbi + tkl
    row["cbirt"] = cbi + tkl + rec
    row["defensive_contribution"] = _i(dc) if dc is not None else (cbi + tkl + rec)
    return row


def ingest_player_history(player_ids: list[int], client: FPLClient | None = None,
                          season: int = CURRENT_SEASON, workers: int = 6) -> int:
    """Pull true per-fixture rows for the given players.

    Parallelised because this is 600+ sequential HTTP calls otherwise; capped
    at a modest worker count so we stay a well-behaved API consumer.
    """
    client = client or FPLClient()
    rows: list[dict] = []

    def fetch(pid: int) -> list[dict]:
        try:
            data = client.element_summary(pid)
        except Exception as e:      # noqa: BLE001
            log.warning("element-summary failed for %s: %s", pid, e)
            return []
        return [r for r in (_history_row(pid, season, h)
                            for h in data.get("history", [])) if r]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, pid): pid for pid in player_ids}
        for fut in as_completed(futures):
            rows.extend(fut.result())

    if rows:
        report = check_fixture_stats(pd.DataFrame(rows))
        report.persist()
        if report.failed:
            failed = [c.name for c in report.checks if c.failed]
            raise RuntimeError(f"fixture-stats quality gate failed: {failed}")
        upsert(T.player_fixture_stats, rows,
               ["player_id", "season", "fixture_id"])
    log.info("player history: %d fixture rows for %d players", len(rows), len(player_ids))
    return len(rows)


def ingest_set_piece_duties(data: dict, season: int = CURRENT_SEASON) -> int:
    """Append a new point-in-time duty version only when FPL changes it."""
    from fplagent.models.setpieces import order_to_share

    now = datetime.utcnow()
    field_map = {
        "penalties": "penalties_order",
        "corners": "corners_and_indirect_freekicks_order",
        "direct_fk": "direct_freekicks_order",
    }
    declared: dict[tuple[int, str], dict] = {}
    for p in data.get("elements", []):
        for duty, field in field_map.items():
            order = _i_or_none(p.get(field))
            if order is None:
                continue
            declared[(int(p["id"]), duty)] = {
                "player_id": int(p["id"]), "season": season, "duty": duty,
                "order": order, "share": order_to_share(duty, order),
                "confidence": round(max(0.9 - 0.2 * (order - 1), 0.3), 2),
                "known_at": now, "source": "fpl_bootstrap", "retired_at": None,
            }
    if not declared:
        log.warning("set-piece duties: bootstrap declared no duties; leaving history unchanged")
        return 0

    hist = T.set_piece_duty_history
    active_df = read_sql(select(hist).where(
        (hist.c.season == season) & (hist.c.source == "fpl_bootstrap")
        & hist.c.retired_at.is_(None)))
    active = {(int(r.player_id), str(r.duty)): r._asdict() for r in active_df.itertuples()}

    writes = []
    retire_ids = []
    declared_keys = set(declared)
    for key, old in active.items():
        if key not in declared_keys:
            retire_ids.append(int(old["duty_id"]))
    for key, row in declared.items():
        old = active.get(key)
        changed = (old is None or int(old["order"] or 99) != int(row["order"] or 99)
                   or abs(float(old["share"] or 0.0) - float(row["share"] or 0.0)) > 1e-9
                   or abs(float(old["confidence"] or 0.0) - float(row["confidence"] or 0.0)) > 1e-9)
        if changed:
            if old is not None:
                retire_ids.append(int(old["duty_id"]))
            writes.append(row)

    if retire_ids:
        with session_scope() as sess:
            sess.execute(update(hist).where(hist.c.duty_id.in_(retire_ids)).values(retired_at=now))
    if writes:
        from fplagent.db.repository import insert_rows
        insert_rows(hist, writes)
    log.info("set-piece duties: %d new history versions, %d retired", len(writes), len(set(retire_ids)))
    return len(writes)


def ingest_penalty_duties(season: int = CURRENT_SEASON,
                          csv_path: str | None = None) -> int:
    """Optional enrichment: primary penalty takers from a manually-exported
    FBref CSV."""
    if not csv_path:
        log.info("no --fbref-csv given; skipping penalty-duty enrichment")
        return 0

    try:
        from fplagent.ingestion.fbref import get_matched_penalty_duties
    except ImportError as e:
        log.info("penalty-duty enrichment skipped (missing dependency: %s)", e)
        return 0

    try:
        records = get_matched_penalty_duties(season, csv_path)
    except Exception as e:      # noqa: BLE001
        log.warning("penalty-duty CSV ingestion failed, leaving "
                    "set_piece_duties unchanged: %s", e)
        return 0

    if not records:
        return 0
    now = datetime.utcnow()
    rows = []
    for r in records:
        row = dict(r)
        row.setdefault("known_at", now)
        row.setdefault("source", "fbref")
        row.setdefault("retired_at", None)
        rows.append(row)
    n = _append_manual_duty_versions(rows)
    log.info("penalty duties: appended %d FBref duty versions from %s", n, csv_path)
    return n


def _append_manual_duty_versions(rows: list[dict]) -> int:
    """Append an operator/source assertion into the versioned duty table."""
    if not rows:
        return 0
    hist = T.set_piece_duty_history
    now = rows[0].get("known_at") or datetime.utcnow()
    with session_scope() as sess:
        for row in rows:
            q = select(hist).where(
                (hist.c.player_id == row["player_id"])
                & (hist.c.season == row["season"])
                & (hist.c.duty == row["duty"])
                & hist.c.retired_at.is_(None)
            ).order_by(hist.c.known_at.desc())
            old = sess.execute(q).mappings().first()
            if (old and int(old.get("order") or 99) == int(row.get("order") or 99)
                    and abs(float(old.get("share") or 0) - float(row.get("share") or 0)) < 1e-9):
                continue
            if old:
                sess.execute(update(hist).where(hist.c.duty_id == old["duty_id"]).values(retired_at=now))
            sess.execute(insert(T.set_piece_duty_history).values(row))
    return len(rows)


def run_full_ingest(season: int = CURRENT_SEASON, limit_players: int | None = None,
                    use_cache: bool = False, fbref_csv: str | None = None) -> dict:
    """One call, everything. This is what the daily job runs."""
    client = FPLClient(use_cache=use_cache)
    boot = ingest_bootstrap(client, season)
    n_fixtures = ingest_fixtures(client, season)
    n_duties = ingest_penalty_duties(season, fbref_csv)

    # Non-FPL fixtures feed the minutes model's congestion feature. Both
    # sources inside this call soft-fail; a bad day at football-data.org
    # degrades the congestion signal, not the pipeline.
    other_comp = ingest_other_competition_fixtures(season)

    ids = read_sql(select(T.players.c.player_id).where(T.players.c.season == season))
    player_ids = ids["player_id"].tolist()
    if limit_players:
        player_ids = player_ids[:limit_players]

    n_stats = ingest_player_history(player_ids, client, season)
    return {**boot, "fixtures": n_fixtures, "fixture_stats": n_stats,
            "penalty_duties": n_duties, "other_competition_fixtures": other_comp,
            "players_pulled": len(player_ids)}