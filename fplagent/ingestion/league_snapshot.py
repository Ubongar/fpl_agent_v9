"""Fitting real league-translation ratios from FBref season exports.

WHAT THIS REPLACES
-------------------
`models/cold_start.py` shipped a table (`SEED_TRANSLATION`) of coefficients
like "a Championship forward keeps 68% of his non-penalty goal rate moving to
the Premier League", documented as drawn from published research. Nobody
could find the source. Since the module was also never wired into a live
prediction, this had zero practical effect -- but it should not exist
un-sourced regardless.

This module computes the same kind of ratio from data we can point to: a
real FBref export of a source league, matched against real current-season
Premier League output for the same named players. The result is reproducible
-- rerun the script, get the same answer from the same inputs -- which
"published research, exact source unknown" never was.

WHY POOLED RATES, NOT A RATIO PER PLAYER
-----------------------------------------
The naive approach -- compute each matched player's individual ratio, then
average the ratios -- blows up on the many players who scored zero goals in
one leg or the other, and a single such player can swing a small-sample
average by a large factor. Instead this pools RAW TOTALS across every
matched player in a (league, position group) before dividing:

    source_rate = sum(goals across matched players) / sum(minutes) * 90
    target_rate = sum(goals across matched players in the PL) / sum(minutes) * 90
    ratio = target_rate / source_rate

This is the same shrink-the-denominator-not-the-numerator philosophy used in
`models/shares.py`, and it is far more stable against the zero-goal players
who make up most of any squad.

WHY THIS IS GATED, NOT TRUSTED OUTRIGHT
-----------------------------------------
Run against the actual data supplied for this fix -- seven foreign leagues'
2025/26 exports matched against the Premier League's 2026/27 season a few
gameweeks in -- most combinations do NOT clear a sane reliability bar. With
only a handful of Premier League matches played so far, a couple of
lucky-or-unlucky finishes swing a position group's ratio wildly: one
computed ratio came out at 5.7x, which is not a real effect, it is four
gameweeks of small numbers. `gate()` below reports every ratio computed, but
`meets_gate` only turns true once there is enough evidence to trust it over
the seed table, and `models/cold_start.py` respects that flag rather than
using every fitted number unconditionally.

This is also why the fit is meant to be RE-RUN as the season progresses,
not run once and frozen: the same matched players accumulate real Premier
League minutes every gameweek, and ratios that fail the gate in September
routinely clear it by November.

TWO SEASONS IS NOT PROVABLE FROM THE FILE ALONE
-------------------------------------------------
An FBref "Standard Stats" export carries no season column, and testing
against the real files here found at least one current Premier League
regular (established two seasons ago) still present in a "current season"
Championship export -- meaning these exports cannot be assumed to isolate
one clean, immediately-prior season. Rather than claim a precision the data
does not support, this module makes no season-alignment claim at all: it
treats a match as "this player has recorded output in both a foreign top
division and the Premier League", and leans on the reliability gate, not on
season bookkeeping, to keep the result honest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fplagent.config import CURRENT_SEASON, PARAMS
from fplagent.db import tables as T
from fplagent.db.repository import upsert
from fplagent.ingestion.fbref_common import read_fbref_csv

log = logging.getLogger(__name__)

POSITION_PREFIX = (("GK", "GK"), ("DF", "DEF"), ("FW", "ATT"), ("MF", "MID"))


def _position_group(fbref_pos: str) -> str:
    """FBref combines multi-position players like 'MFFW'; take the first
    recognised token. Defaults to MID, the most common swing position."""
    pos = str(fbref_pos or "").strip().upper()
    for prefix, group in POSITION_PREFIX:
        if pos.startswith(prefix):
            return group
    return "MID"


def parse_standard_stats(path: str | Path) -> pd.DataFrame | None:
    """One row per player: minutes, non-penalty goals, assists, cards,
    aggregated across any mid-export duplicate rows (a player who changed
    clubs mid-season appears once per club; summed here since the fit only
    needs total output, not club attribution)."""
    df = read_fbref_csv(path)
    if df is None:
        return None
    need = {"Player", "Pos", "Min", "Ast", "G-PK", "CrdY"}
    if not need.issubset(df.columns):
        log.warning("%s: missing columns for translation fitting (need %s, have %s)",
                    path, need, set(df.columns))
        return None
    for c in ("Min", "Ast", "G-PK", "CrdY", "CrdR"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Player", "Min"])
    agg = df.groupby(["Player", "Pos"], as_index=False).agg(
        Min=("Min", "sum"), Ast=("Ast", "sum"), GPK=("G-PK", "sum"),
        CrdY=("CrdY", "sum"),
        CrdR=("CrdR", "sum") if "CrdR" in df.columns else ("CrdY", "size"))
    agg = agg[agg["Min"] > 0].copy()
    agg["grp"] = agg["Pos"].apply(_position_group)
    return agg


@dataclass
class FittedRatio:
    source_league: str
    position_group: str
    metric: str                # np_g90 | a90
    ratio: float
    n_players: int
    source_minutes: float
    target_minutes: float
    meets_gate: bool
    confidence: float


def _gate(n_players: int, target_minutes: float, ratio: float) -> bool:
    p = PARAMS
    if n_players < p.translation_min_players:
        return False
    if target_minutes < p.translation_min_target_minutes:
        return False
    if not (p.translation_min_plausible_ratio <= ratio <= p.translation_max_plausible_ratio):
        return False
    return True


def match_transfer_cohort(source_df: pd.DataFrame, pl_df: pd.DataFrame,
                          min_source_minutes: float = 450.0,
                          min_target_minutes: float = 90.0,
                          fuzzy_threshold: float = 92.0) -> pd.DataFrame:
    """Name-match a foreign-league export against current PL output.

    This is a many-to-one name match for an AGGREGATE statistical purpose,
    not an attribution of a specific fact to a specific player (contrast
    `models/setpieces.py`, where a wrong match would misassign a real duty).
    A stray false match here just adds one noisy data point to a pooled
    total dominated by dozens of others, so a fairly high fuzzy threshold is
    enough; it does not need the surrounding-context checks a duty-critical
    match would.
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        log.warning("rapidfuzz not installed; cannot match transfer cohorts")
        return pd.DataFrame()

    src = source_df[source_df["Min"] >= min_source_minutes]
    pl_names = pl_df["Player"].tolist()
    rows = []
    for _, r in src.iterrows():
        hit = process.extractOne(r["Player"], pl_names, scorer=fuzz.token_sort_ratio)
        if not hit or hit[1] < fuzzy_threshold:
            continue
        pl_row = pl_df[pl_df["Player"] == hit[0]].iloc[0]
        if pl_row["Min"] < min_target_minutes:
            continue
        rows.append({
            "player": r["Player"], "pl_player": hit[0], "grp": r["grp"],
            "src_min": r["Min"], "src_ast": r["Ast"], "src_gpk": r["GPK"],
            "pl_min": pl_row["Min"], "pl_ast": pl_row["Ast"], "pl_gpk": pl_row["GPK"],
        })
    return pd.DataFrame(rows)


def fit_ratios(source_league: str, matched: pd.DataFrame) -> list[FittedRatio]:
    """Pooled ratio per (position group, metric) from a matched cohort."""
    out: list[FittedRatio] = []
    if matched.empty:
        return out
    for grp, g in matched.groupby("grp"):
        n = len(g)
        src_min_total = float(g["src_min"].sum())
        tgt_min_total = float(g["pl_min"].sum())
        for metric, src_col, tgt_col in (("np_g90", "src_gpk", "pl_gpk"),
                                         ("a90", "src_ast", "pl_ast")):
            src_rate = g[src_col].sum() / max(src_min_total / 90.0, 1e-6)
            tgt_rate = g[tgt_col].sum() / max(tgt_min_total / 90.0, 1e-6)
            if src_rate < 0.02:
                continue
            ratio = float(tgt_rate / src_rate)
            meets = _gate(n, tgt_min_total, ratio)
            confidence = (min(0.75, 0.15 + 0.1 * np.log1p(tgt_min_total / 90.0))
                         if meets else 0.1)
            out.append(FittedRatio(source_league, grp, metric, round(ratio, 4),
                                   n, src_min_total, tgt_min_total, meets,
                                   round(confidence, 3)))
    return out


def persist_fitted_ratios(ratios: list[FittedRatio]) -> int:
    """Every computed ratio is stored, gate-passing or not, so the number and
    its sample size stay inspectable. `cold_start.lookup_translation` is the
    place that decides whether to trust it."""
    now = datetime.utcnow()
    rows = [{
        "source_league": r.source_league, "position_group": r.position_group,
        "metric": r.metric, "ratio": r.ratio,
        "confidence": r.confidence if r.meets_gate else 0.0,
        "sample_size": r.n_players, "source": "fitted_transfer_cohort",
        "updated_at": now,
    } for r in ratios]
    if rows:
        upsert(T.league_translation, rows,
              ["source_league", "position_group", "metric"])
    return len(rows)


def persist_external_records(source_league: str, matched: pd.DataFrame,
                             player_id_by_name: dict[str, int],
                             season: int = CURRENT_SEASON) -> int:
    """Link matched players to their foreign-league output, so a future
    per-player cold-start lookup (not just the aggregate ratio) has
    something to read. Only linked when the PL name resolves to a real
    ingested player_id -- an unresolved name is dropped rather than guessed.
    """
    now = datetime.utcnow()
    ext_rows, map_rows = [], []
    for _, r in matched.iterrows():
        pid = player_id_by_name.get(r["pl_player"])
        if pid is None:
            continue
        ext_id = f"fbref:{source_league}:{r['player']}"
        ext_rows.append({
            "external_id": ext_id, "source": "fbref", "season": season,
            "league": source_league, "player_name": r["player"],
            "team": "", "position": r["grp"], "minutes": int(r["src_min"]),
            "np_xg": float(r["src_gpk"]), "xa": float(r["src_ast"]),
            "known_at": now,
        })
        map_rows.append({
            "player_id": pid, "source": "fbref", "external_id": ext_id,
            "match_score": 1.0, "method": "fuzzy", "verified": False,
        })
    if ext_rows:
        upsert(T.external_player_season, ext_rows,
              ["external_id", "source", "season", "league"])
        upsert(T.external_player_map, map_rows, ["player_id", "source"])
    return len(ext_rows)


def fit_and_persist(pl_csv: str | Path, league_csvs: dict[str, str | Path],
                    player_id_by_name: dict[str, int] | None = None,
                    season: int = CURRENT_SEASON) -> dict:
    """Run the whole pipeline: parse, match, fit, gate, persist.

    Returns a summary dict, not just a side effect, so a CLI or script can
    print exactly what was learned and what was rejected by the gate.
    """
    pl_df = parse_standard_stats(pl_csv)
    if pl_df is None:
        return {"error": f"could not parse Premier League export at {pl_csv}"}

    summary: dict = {"leagues": {}, "total_fitted": 0, "total_gate_passed": 0}
    for league, path in league_csvs.items():
        src_df = parse_standard_stats(path)
        if src_df is None:
            summary["leagues"][league] = {"error": f"could not parse {path}"}
            continue
        matched = match_transfer_cohort(src_df, pl_df)
        ratios = fit_ratios(league, matched)
        persist_fitted_ratios(ratios)
        n_ext = 0
        if player_id_by_name:
            n_ext = persist_external_records(league, matched, player_id_by_name, season)
        summary["leagues"][league] = {
            "n_matched_players": int(matched["player"].nunique()) if not matched.empty else 0,
            "ratios": [r.__dict__ for r in ratios],
            "external_records_linked": n_ext,
        }
        summary["total_fitted"] += len(ratios)
        summary["total_gate_passed"] += sum(1 for r in ratios if r.meets_gate)
    return summary


def load_bundled_fit(json_path: str | Path = "data/league_translation_fitted.json") -> int:
    """Load the fitted ratios shipped with the repo (see CHANGELOG_v7.md) into
    the database. Safe to call every time `init-db` runs: it is an upsert, so
    a later real fit (via scripts/fit_league_translation.py, rerun as the
    season lengthens) simply overwrites these with fresher numbers.
    """
    import json
    path = Path(json_path)
    if not path.exists():
        return 0
    with open(path) as f:
        summary = json.load(f)
    ratios = [FittedRatio(**{k: r[k] for k in
                            ("source_league", "position_group", "metric", "ratio",
                             "n_players", "source_minutes", "target_minutes",
                             "meets_gate", "confidence")})
             for d in summary.get("leagues", {}).values()
             for r in d.get("ratios", [])]
    persist_fitted_ratios(ratios)
    return len(ratios)
