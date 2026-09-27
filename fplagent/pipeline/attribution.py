"""'What changed, and why' -- week-over-week attribution.

THE MISSING LAYER (v5 audit, section 3.5)
-----------------------------------------
v5's LLM agent explained a recommendation in isolation. The far more useful
artefact is a DIFF: not "Semenyo is worth 6.4 points" but "Semenyo is worth
1.4 more than last week: 60% of that is the fixture, 30% is a rotation-risk
drop, 10% is share re-estimation."

Attribution of CHANGE is what builds trust in a model, because it is
falsifiable in a way a level is not. And it is cheap: the xP decomposition is
already additive by construction in the simulation's component output, so the
diff is a subtraction, not an inference. Run it before handing anything to the
LLM, so the model explains real arithmetic instead of inventing a story.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from fplagent.db import tables as T
from fplagent.db.repository import read_sql

COMPONENT_LABELS = {
    "appearance": "appearance points",
    "goals": "goal threat",
    "assists": "chance creation",
    "clean_sheet": "clean sheet odds",
    "goals_conceded": "goals-conceded risk",
    "saves": "save points",
    "defensive_contribution": "defensive contribution",
    "bonus": "bonus potential",
    "cards": "card risk",
    "own_goals": "own-goal risk",
    "penalty_misses": "penalty-miss risk",
}


def _load(season: int, gw: int, version: str | None) -> pd.DataFrame:
    conds = [T.predictions.c.season == season, T.predictions.c.gw == gw]
    if version:
        conds.append(T.predictions.c.model_version == version)
    df = read_sql(select(T.predictions).where(*conds))
    if df.empty:
        return df
    return df.sort_values("created_at").groupby("player_id", as_index=False).tail(1)


def attribute_change(season: int, gw_now: int, gw_prev: int,
                     version_now: str | None = None,
                     version_prev: str | None = None,
                     min_delta: float = 0.4) -> list[dict]:
    """Biggest movers, with the change decomposed into its drivers."""
    now = _load(season, gw_now, version_now)
    prev = _load(season, gw_prev, version_prev)
    if now.empty or prev.empty:
        return []

    players = read_sql(select(T.players).where(T.players.c.season == season))
    name = players.set_index("player_id")["web_name"].to_dict()

    merged = now.merge(prev, on="player_id", suffixes=("_now", "_prev"))
    merged["delta"] = merged["xp_mean_now"] - merged["xp_mean_prev"]
    movers = merged[merged["delta"].abs() >= min_delta].copy()
    movers = movers.reindex(movers["delta"].abs().sort_values(ascending=False).index)

    out = []
    for _, r in movers.iterrows():
        c_now = r["components_now"] if isinstance(r["components_now"], dict) else {}
        c_prev = r["components_prev"] if isinstance(r["components_prev"], dict) else {}
        drivers = []
        for key in set(c_now) | set(c_prev):
            if key.startswith("_") or key == "total":
                continue
            d = float(c_now.get(key, 0.0)) - float(c_prev.get(key, 0.0))
            if abs(d) >= 0.05:
                drivers.append({"component": COMPONENT_LABELS.get(key, key),
                                "delta": round(d, 3)})
        drivers.sort(key=lambda x: abs(x["delta"]), reverse=True)
        total_abs = sum(abs(d["delta"]) for d in drivers) or 1.0
        for d in drivers:
            d["share_of_change"] = round(abs(d["delta"]) / total_abs, 3)

        f_now = r["features_now"] if isinstance(r["features_now"], dict) else {}
        f_prev = r["features_prev"] if isinstance(r["features_prev"], dict) else {}
        context = []
        if abs(float(f_now.get("team_xg", 0)) - float(f_prev.get("team_xg", 0))) > 0.15:
            context.append(
                f"team expected goals moved from {f_prev.get('team_xg')} to "
                f"{f_now.get('team_xg')} (fixture change)")
        if abs(float(r["p_play_now"]) - float(r["p_play_prev"])) > 0.1:
            context.append(
                f"playing probability moved from {float(r['p_play_prev']):.0%} to "
                f"{float(r['p_play_now']):.0%}")

        out.append({
            "player_id": int(r["player_id"]),
            "name": name.get(int(r["player_id"]), str(r["player_id"])),
            "xp_now": round(float(r["xp_mean_now"]), 2),
            "xp_prev": round(float(r["xp_mean_prev"]), 2),
            "delta": round(float(r["delta"]), 2),
            "drivers": drivers[:5],
            "context": context,
        })
    return out


def narrate(change: dict) -> str:
    """One-line plain-English summary, used as grounded input to the LLM layer
    so it paraphrases arithmetic rather than inventing causes."""
    direction = "up" if change["delta"] > 0 else "down"
    parts = [f"{change['name']} is {direction} {abs(change['delta'])} points "
             f"({change['xp_prev']} to {change['xp_now']})"]
    if change["drivers"]:
        top = change["drivers"][0]
        parts.append(f"mostly {top['component']} "
                     f"({top['share_of_change']:.0%} of the move)")
    if change["context"]:
        parts.append(change["context"][0])
    return "; ".join(parts) + "."
