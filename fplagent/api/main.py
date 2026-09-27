"""FastAPI service.

A thin transport layer over the modules in fplagent/. No business logic lives
here -- every endpoint is a call into code that is unit-tested independently
and can be run from the CLI without a server.

    uvicorn fplagent.api.main:app --reload --port 8000
    open http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from fplagent import __version__
from fplagent.config import CURRENT_SEASON, TOTAL_MANAGERS
from fplagent.db import tables as T
from fplagent.db.repository import read_sql, resolve_target_gw
from fplagent.db.serialize import to_json_safe


class SafeJSONResponse(JSONResponse):
    """JSON that never contains NaN or Infinity."""

    def render(self, content) -> bytes:
        return json.dumps(to_json_safe(content), ensure_ascii=False,
                          allow_nan=False, separators=(",", ":"),
                          default=str).encode("utf-8")


API_KEY = os.getenv("FPL_AGENT_API_KEY", "")


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Optional API-key gate. Unset means local/dev mode remains open."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


app = FastAPI(
    title="FPL Agent",
    version=__version__,
    description="Prediction, simulation and optimisation for Fantasy Premier League.",
    default_response_class=SafeJSONResponse,
    dependencies=[Depends(require_api_key)] if API_KEY else [],
)

_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"],
)


def _num(x, default: float = 0.0) -> float:
    """A float, with None and NaN mapped to `default`."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(f) else f


def _player_names_map(season: int) -> dict[int, dict]:
    """player_id -> {web_name, position, team_id, short_name}."""
    players_df = read_sql(select(T.players).where(T.players.c.season == season))
    if players_df.empty:
        return {}
    teams_df = read_sql(select(T.teams).where(T.teams.c.season == season))
    short_by_team = ({int(r.team_id): str(r.short_name) for r in teams_df.itertuples()}
                     if not teams_df.empty else {})
    out: dict[int, dict] = {}
    for r in players_df.itertuples():
        team_id = int(r.team_id) if r.team_id is not None else None
        out[int(r.player_id)] = {
            "web_name": str(r.web_name) if r.web_name is not None else None,
            "position": str(r.position) if r.position is not None else None,
            "team_id": team_id,
            "short_name": short_by_team.get(team_id) if team_id is not None else None,
        }
    return out


def _decision_pack(df: pd.DataFrame, season: int | None = None, gw: int | None = None) -> pd.DataFrame:
    """Attach transparent decision metrics used by the v9 dashboard."""
    from fplagent.models.decision import decision_metrics
    if df.empty:
        return df
    out = df.copy()
    if season is not None:
        try:
            eo = _load_effective_ownership(season, gw or int(out["gw"].min()))
        except Exception:
            eo = {}
        if eo:
            out["eo"] = out["player_id"].map(lambda x: eo.get(int(x)))
    rows = []
    for r in out.to_dict("records"):
        rows.append(decision_metrics(r))
    metrics = pd.DataFrame(rows, index=out.index)
    for col in metrics.columns:
        out[col] = metrics[col]
    return out


def _captain_candidates(df, real_eo: dict[int, float], n: int = 50) -> list[dict]:
    """Captaincy inputs from a latest_predictions frame."""
    return [{
        "player_id": int(r.player_id), "name": r.web_name, "xp": float(r.xp_mean),
        "p10": _num(getattr(r, "xp_p10", None)), "p90": _num(getattr(r, "xp_p90", None)),
        "p_haul_10plus": _num(getattr(r, "p_haul_10plus", None)),
        "p_blank_le2": _num(getattr(r, "p_blank_le2", None)),
        "p_play": _num(r.p_play),
        "selected_by_percent": _num(getattr(r, "selected_by_percent", None)),
        **({"eo": real_eo[int(r.player_id)]} if int(r.player_id) in real_eo else {}),
    } for r in df.head(n).itertuples()]


@app.get("/health", tags=["system"])
def health():
    from fplagent.db.session import session_scope
    from sqlalchemy import text
    try:
        with session_scope() as s:
            s.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok,
            "season": CURRENT_SEASON, "time": datetime.now(timezone.utc)}


@app.get("/gameweeks/current", tags=["system"])
def current_gameweek(season: int = CURRENT_SEASON):
    gw = resolve_target_gw(season)
    if gw is None:
        raise HTTPException(404, "No gameweeks ingested. Run `fplagent ingest`.")
    row = read_sql(select(T.gameweeks).where(
        (T.gameweeks.c.season == season) & (T.gameweeks.c.gw == gw)))
    return {"season": season, "target_gw": gw,
            "deadline": None if row.empty else row.iloc[0]["deadline_time"],
            "source": "fpl_api_events"}


@app.get("/manager/{entry_id}/team", tags=["manager"])
def manager_team(entry_id: int, gw: int | None = None, season: int = CURRENT_SEASON):
    """A real manager's current squad, fetched live from FPL's own public
    endpoints.

    Predictions are looked up for the UPCOMING gameweek (the one
    `resolve_target_gw` returns), not for `team.gw`. `team.gw` is what FPL
    reports for the picks themselves -- usually the most recently COMPLETED
    gameweek -- and that gameweek frequently has no stored predictions at
    all: a fresh deployment has none before today, and `evaluate --replay`
    deliberately does not persist. Looking up predictions by `team.gw`
    therefore returned an empty frame and the frontend rendered all fifteen
    picks as "player <id>" with no club, price or xP.

    The prediction the manager actually cares about is for the gameweek
    they are about to play, so that is what this now fetches.
    """
    from fplagent.ingestion.manager_team import get_my_team
    from fplagent.optimizers.transfers import sell_price
    from fplagent.pipeline.run import latest_predictions

    try:
        team = get_my_team(entry_id, gw=gw, season=season)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:      # noqa: BLE001
        raise HTTPException(502, f"could not reach the FPL API: {e}")

    # `latest_predictions(season)` with no gw resolves the target gameweek
    # internally, and returns that gameweek's newest stored version. If the
    # target has no predictions yet it returns an empty frame, which is
    # better than silently falling back to a stale gameweek.
    preds = latest_predictions(season)
    pred_by_id = ({int(r.player_id): r for r in preds.itertuples()}
                 if not preds.empty else {})

    squad = []
    for pick in team.picks:
        r = pred_by_id.get(pick.player_id)
        price = float(r.now_cost) if r is not None else None
        row = {
            "player_id": pick.player_id,
            "name": r.web_name if r is not None else f"player {pick.player_id}",
            "position": r.position if r is not None else None,
            "club": r.short_name if r is not None else None,
            "price": price,
            "xp": float(r.xp_mean) if r is not None else None,
            "p_play": float(r.p_play) if r is not None else None,
            "is_captain": pick.is_captain, "is_vice_captain": pick.is_vice_captain,
            "multiplier": pick.multiplier,
            "purchase_price": pick.purchase_price,
            "purchase_price_known": pick.purchase_price_known,
        }
        if price is not None:
            row["sell_price"] = sell_price({"price": price, "purchase_price": pick.purchase_price})
        squad.append(row)

    unmatched = [p.player_id for p in team.picks if p.player_id not in pred_by_id]
    return {
        "entry_id": team.entry_id, "manager_name": team.manager_name,
        "team_name": team.team_name, "season": team.season, "gw": team.gw,
        "overall_rank": team.overall_rank, "overall_points": team.overall_points,
        "bank": team.bank, "team_value": team.team_value,
        "free_transfers": team.free_transfers,
        "free_transfers_is_estimated": team.free_transfers_is_estimated,
        "active_chip": team.active_chip, "chips_used": team.chips_used,
        "squad": squad,
        "unmatched_player_ids": unmatched,
        "note": (f"{len(unmatched)} player(s) have no stored prediction for "
                f"the upcoming gameweek -- run `fplagent predict` first"
                if unmatched else None),
    }


@app.get("/predictions", tags=["predictions"])
def predictions(season: int = CURRENT_SEASON, gw: int | None = None,
                position: str | None = None, team: str | None = None,
                max_price: float | None = None, min_price: float | None = None,
                search: str | None = None,
                sort: str = "xp_mean", desc: bool = True,
                offset: int = Query(0, ge=0),
                limit: int = Query(50, ge=1, le=1000)):
    """Paginated, filterable, sortable."""
    from fplagent.pipeline.run import latest_predictions
    df = latest_predictions(season, gw)
    if df.empty:
        raise HTTPException(404, "No predictions stored. Run `fplagent predict`.")
    if position:
        df = df[df["position"] == position.upper()]
    if team:
        df = df[df["short_name"].astype(str).str.upper() == team.upper()]
    if max_price is not None:
        df = df[df["now_cost"] <= max_price]
    if min_price is not None:
        df = df[df["now_cost"] >= min_price]
    if search:
        df = df[df["web_name"].astype(str).str.contains(search, case=False, na=False, regex=False)]
    if sort in df.columns:
        df = df.sort_values(sort, ascending=not desc)
    total = int(len(df))
    page = _decision_pack(df.iloc[offset:offset + limit], season, int(df["gw"].min()) if not df.empty else gw)
    return {
        "total": total, "offset": offset, "limit": limit,
        "model_version": df["model_version"].iloc[0] if total else None,
        "players": page.replace({float("nan"): None}).to_dict("records"),
    }


@app.get("/teams", tags=["reference"])
def teams(season: int = CURRENT_SEASON):
    df = read_sql(select(T.teams).where(T.teams.c.season == season))
    return {"teams": df.to_dict("records")}


@app.get("/evaluation/history", tags=["evaluation"])
def evaluation_history(season: int = CURRENT_SEASON, limit: int = Query(50, ge=1, le=200)):
    """Every backtest run, newest first."""
    df = read_sql(select(T.evaluation_runs).where(
        T.evaluation_runs.c.season == season))
    if df.empty:
        return {"runs": []}
    df = df.sort_values("created_at", ascending=False).head(limit)
    return {"runs": df.replace({float("nan"): None}).to_dict("records")}


@app.get("/evaluation/latest", tags=["evaluation"])
def evaluation_latest(season: int = CURRENT_SEASON, gw_from: int = 2,
                      gw_to: int | None = None, replay: bool = False,
                      auto_replay: bool = True, explain: bool = False):
    """Score the model."""
    from fplagent.evaluation.backtest import backtest
    result = backtest(season=season, gw_from=gw_from, gw_to=gw_to,
                      replay=replay, persist=False)
    if result.get("n", 0) == 0 and auto_replay and not replay:
        result = backtest(season=season, gw_from=gw_from, gw_to=gw_to,
                          replay=True, persist=False)
        result["mode"] = "replay"
        result["mode_note"] = (
            "No predictions were stored before these deadlines, so history was "
            "replayed through the point-in-time feature path. That is weaker "
            "evidence than scoring predictions written in advance.")
    else:
        result.setdefault("mode", "replay" if replay else "stored")
    if explain:
        from fplagent.agent.llm import explain_model_health
        result["explanation"] = explain_model_health(result)
    return result


@app.get("/evaluation/track-record", tags=["evaluation"])
def evaluation_track_record(season: int = CURRENT_SEASON, limit: int = Query(10, ge=1, le=52)):
    """Plain-language trust surface over evaluation_runs."""
    df = read_sql(select(T.evaluation_runs).where(T.evaluation_runs.c.season == season))
    if df.empty:
        return {"season": season, "runs_considered": 0,
                "summary": "No backtest runs stored yet for this season."}
    df = df.sort_values("created_at", ascending=False).head(limit)
    n = len(df)
    top10 = df["top10_captain_hit_rate"].dropna()
    top5 = df["top5_captain_hit_rate"].dropna()
    beat_baseline = df["beat_best_baseline"].dropna()

    lines = []
    if len(top10):
        lines.append(
            f"Across the last {len(top10)} scored gameweek(s), the actual "
            f"top scorer landed in the model's top-10 captain shortlist "
            f"{top10.mean():.0%} of the time"
            + (f", and in the top-5 {top5.mean():.0%} of the time" if len(top5) else "")
            + ".")
    if len(beat_baseline):
        lines.append(
            f"The model beat the best simple baseline (recent form / price / "
            f"next-fixture-only) in {beat_baseline.mean():.0%} of scored "
            f"gameweeks.")
    if not lines:
        lines.append("Stored runs exist but don't yet include captain hit-rate "
                     "or baseline-comparison fields -- re-run the backtest.")
    return {
        "season": season, "runs_considered": n,
        "top10_captain_hit_rate_avg": None if not len(top10) else round(float(top10.mean()), 3),
        "top5_captain_hit_rate_avg": None if not len(top5) else round(float(top5.mean()), 3),
        "beat_baseline_rate": None if not len(beat_baseline) else round(float(beat_baseline.mean()), 3),
        "summary": " ".join(lines),
    }


@app.get("/team-strength", tags=["predictions"])
def team_strength(season: int = CURRENT_SEASON, gw: int | None = None):
    df = read_sql(select(T.team_strength).where(T.team_strength.c.season == season))
    if gw is not None and not df.empty:
        df = df[df["gw"] == gw]
    if df.empty:
        return {"teams": []}
    latest = df.sort_values("created_at")["model_version"].iloc[-1]
    df = df[df["model_version"] == latest]
    df = df.sort_values("gw").groupby("team_id", as_index=False).tail(1)
    teams_df = read_sql(select(T.teams).where(T.teams.c.season == season))
    if not teams_df.empty:
        df = df.merge(teams_df[["team_id", "name", "short_name"]], on="team_id", how="left")
    return {"model_version": latest,
            "teams": df.replace({float("nan"): None}).to_dict("records")}


@app.get("/set-pieces", tags=["predictions"])
def set_pieces(season: int = CURRENT_SEASON, duty: str = "penalties"):
    """Declared set-piece duty, ordered."""
    from fplagent.features.build import load_set_piece_duties
    df = load_set_piece_duties(season)
    if not df.empty:
        df = df[df["duty"] == duty].copy()
    if df.empty:
        return {"duty": duty, "players": []}
    players_df = read_sql(select(T.players).where(T.players.c.season == season))
    teams_df = read_sql(select(T.teams).where(T.teams.c.season == season))
    if not players_df.empty:
        df = df.merge(players_df[["player_id", "web_name", "team_id", "position"]],
                      on="player_id", how="left")
    if not teams_df.empty:
        df = df.merge(teams_df[["team_id", "short_name"]], on="team_id", how="left")
    df = df.sort_values([c for c in ("short_name", "order") if c in df.columns])
    return {"duty": duty, "players": df.replace({float("nan"): None}).to_dict("records")}


@app.get("/changes", tags=["predictions"])
def changes(gw_now: int, gw_prev: int, season: int = CURRENT_SEASON,
            limit: int = Query(25, ge=1, le=100), explain: bool = False):
    from fplagent.pipeline.attribution import attribute_change, narrate
    out = attribute_change(season, gw_now, gw_prev)[:limit]
    result = {"changes": out, "narrative": [narrate(c) for c in out]}
    if explain and out:
        from fplagent.agent.llm import explain_changes
        result["briefing"] = explain_changes(out)
    return result


def _prediction_freshness(row) -> dict:
    """How old is the data this prediction is based on, in plain terms."""
    created = row.get("created_at") if hasattr(row, "get") else row["created_at"]
    if created is None or pd.isna(created):
        return {"age_hours": None, "note": "creation time not recorded"}
    age = (datetime.utcnow() - pd.Timestamp(created).to_pydatetime()).total_seconds() / 3600
    if age < 6:
        note = "fresh"
    elif age < 24:
        note = "same-day -- check for team news before a late deadline"
    else:
        note = ("more than a day old -- re-run the pipeline close to your "
               "deadline if there's been team news since")
    return {"age_hours": round(age, 1), "note": note}


@app.get("/predictions/{player_id}", tags=["predictions"])
def player_detail(player_id: int, season: int = CURRENT_SEASON,
                   explain: bool = False, live_news: bool = False):
    """Full drilldown: every component of the projection, the inputs that
    produced it, and the simulated distribution."""
    from fplagent.models.consistency import check_prediction
    from fplagent.models.distributions import prob_at_least_one
    from fplagent.models.decision import decision_metrics
    from fplagent.pipeline.predict import aggregate_to_gameweek

    df = read_sql(select(T.predictions).where(
        (T.predictions.c.season == season)
        & (T.predictions.c.player_id == player_id)))
    if df.empty:
        raise HTTPException(404, f"No predictions for player {player_id}")
    df = df.sort_values("created_at")
    row = df.iloc[-1]

    cur = df[(df["model_version"] == row["model_version"]) & (df["gw"] == row["gw"])]
    g = aggregate_to_gameweek(cur).iloc[0]
    if len(cur) > 1:
        components: dict = {}
        for comp in cur["components"]:
            for k, v in (comp or {}).items():
                components[k] = round(components.get(k, 0.0) + float(v or 0.0), 3)
        inputs = {"fixtures": list(cur["features"])}
    else:
        components, inputs = row["components"], row["features"]

    earlier = df[df["gw"] < row["gw"]]
    prev = None
    if not earlier.empty:
        pv = earlier[earlier["gw"] == earlier["gw"].max()]
        pv = pv[pv["model_version"] == pv.iloc[-1]["model_version"]]
        prev = {"xp_mean": float(pv["xp_mean"].sum()),
                "p_play": prob_at_least_one(pv["p_play"])}

    distribution = {"p10": g.get("xp_p10"), "p25": g.get("xp_p25"),
                    "p50": g.get("xp_p50"), "p75": g.get("xp_p75"),
                    "p90": g.get("xp_p90")}
    probabilities = {"play": g["p_play"], "sixty_plus": g["p_sixty"],
                     "haul_10plus": g["p_haul_10plus"],
                     "blank_2_or_less": g.get("p_blank_le2")}

    flags = check_prediction(
        {"xp_mean": g["xp_mean"], "p_play": g["p_play"],
         "xp_p10": g.get("xp_p10"), "xp_p90": g.get("xp_p90")},
        prev,
    )

    decision = decision_metrics({**g.to_dict(), "selected_by_percent": row.get("selected_by_percent"), "eo": row.get("eo")})

    names = _player_names_map(season)
    meta = names.get(player_id, {})

    result = {
        "player_id": player_id,
        "web_name": meta.get("web_name"),
        "position": meta.get("position"),
        "team_id": meta.get("team_id"),
        "short_name": meta.get("short_name"),
        "gw": int(row["gw"]),
        "model_version": row["model_version"],
        "n_fixtures": int(len(cur)),
        "expected_points": round(float(g["xp_mean"]), 3),
        "distribution": distribution, "probabilities": probabilities,
        "components": components, "inputs": inputs,
        "decision": decision,
        "quantile_grid": g.get("sim_quantiles", row["sim_quantiles"]),
        "flags": [f.__dict__ for f in flags],
        "freshness": _prediction_freshness(row),
    }

    if live_news:
        from fplagent.ingestion.lineup_news import (apply_lineup_overlay,
                                                     fetch_confirmed_lineups)
        news_df = fetch_confirmed_lineups(season)
        news_row = None
        if not news_df.empty:
            match = news_df[news_df["player_id"] == player_id]
            if not match.empty:
                news_row = match.iloc[0].to_dict()
        overlaid = apply_lineup_overlay(
            {"p_play": g["p_play"], "p_start": g.get("p_start"),
             "p_sixty": g["p_sixty"], "xp_mean": g["xp_mean"]}, news_row)
        if overlaid.get("lineup_news"):
            result["expected_points"] = overlaid["xp_mean"]
            result["probabilities"]["play"] = overlaid["p_play"]
            result["probabilities"]["start"] = overlaid["p_start"]
            result["probabilities"]["sixty_plus"] = overlaid["p_sixty"]
        result["lineup_news"] = overlaid["lineup_news"]
    if explain:
        from fplagent.agent.llm import explain_prediction
        result["explanation"] = explain_prediction(
            {"player_id": player_id, "web_name": meta.get("web_name")},
            components, inputs, {**distribution, **probabilities})
    return result


class ChipRequest(BaseModel):
    available_from_gw: int
    chip_values: dict = Field(default_factory=dict)
    already_used: dict = Field(default_factory=dict)


@app.post("/optimise/chips", tags=["optimisers"])
def optimise_chips(req: ChipRequest, explain: bool = False):
    """Chip scheduling under the two-window structure."""
    from fplagent.optimizers.chips import schedule_chips
    chip_values = {chip: {int(g): float(v) for g, v in (by_gw or {}).items()}
                   for chip, by_gw in req.chip_values.items()}
    already_used = {chip: [int(g) for g in gws]
                    for chip, gws in (req.already_used or {}).items()}
    sched = schedule_chips(chip_values, req.available_from_gw, already_used or None)
    result = {"assignments": sched.assignments, "total_value": sched.total_value,
              "status": sched.status, "notes": sched.notes}
    if explain:
        from fplagent.agent.llm import explain_chip_schedule
        result["explanation"] = explain_chip_schedule(result)
    return result


@app.get("/optimise/chips/auto", tags=["optimisers"])
def optimise_chips_auto(season: int = CURRENT_SEASON, gw: int | None = None,
                        horizon: int = 14, entry_id: int | None = None):
    """Auto-computed chip schedule from stored predictions."""
    from fplagent.optimizers.chip_values import compute_chip_values
    from fplagent.optimizers.chips import schedule_chips
    from fplagent.pipeline.run import latest_predictions

    start_gw = gw or resolve_target_gw(season)
    if start_gw is None:
        raise HTTPException(404, "No gameweek data stored. Run `fplagent ingest`.")

    horizon = max(1, min(int(horizon), 38))
    gws = [g for g in range(start_gw, min(start_gw + horizon, 39))]

    predictions_by_gw: dict[int, object] = {}
    for g in gws:
        preds = latest_predictions(season, g)
        if not preds.empty:
            predictions_by_gw[g] = preds

    if not predictions_by_gw:
        raise HTTPException(
            404,
            f"No predictions stored for any gameweek in {gws[0]}-{gws[-1]}. "
            f"Run `fplagent predict --gw N` for each, or use the CLI "
            f"`fplagent chips` command which predicts on demand.")

    current_squad = None
    if entry_id:
        try:
            from fplagent.ingestion.manager_team import get_my_team
            team = get_my_team(entry_id, gw=start_gw)
            ids = [int(p.player_id) for p in team.picks]
            if ids:
                pdf = read_sql(select(T.players).where(
                    (T.players.c.season == season)
                    & (T.players.c.player_id.in_(ids))))
                if not pdf.empty:
                    current_squad = [{
                        "player_id": int(r.player_id),
                        "name": str(r.web_name),
                        "position": str(r.position),
                        "club": str(r.team_id),
                        "price": float(r.now_cost or 4.0),
                    } for r in pdf.itertuples()]
        except Exception:      # noqa: BLE001
            current_squad = None

    cv = compute_chip_values(
        predictions_by_gw, sorted(predictions_by_gw),
        current_squad=current_squad)
    sched = schedule_chips(cv.values, available_from_gw=start_gw)

    return {
        "assignments": sched.assignments,
        "total_value": round(float(sched.total_value), 2),
        "status": sched.status,
        "notes": sched.notes + cv.notes,
        "per_gw_values": {
            chip: {int(g): round(float(v), 2) for g, v in vals.items()}
            for chip, vals in cv.values.items()
        },
        "horizon": sorted(predictions_by_gw),
        "entry_id_used": entry_id if current_squad is not None else None,
    }


class SquadRequest(BaseModel):
    budget: float = Field(100.0, gt=0.0, le=1_000.0)
    risk_lambda: float = Field(0.0, ge=0.0, le=1.0)
    locked_in: list[int] = Field(default_factory=list)
    excluded: list[int] = Field(default_factory=list)
    gw: int | None = None


@app.post("/optimise/squad", tags=["optimisers"])
def optimise(req: SquadRequest, season: int = CURRENT_SEASON, explain: bool = False):
    from fplagent.optimizers.squad import optimise_squad
    from fplagent.pipeline.run import latest_predictions
    df = latest_predictions(season, req.gw)
    if df.empty:
        raise HTTPException(404, "No predictions stored.")
    pool = [{
        "player_id": int(r.player_id), "name": r.web_name, "position": r.position,
        "club": str(getattr(r, "short_name", r.team_id)),
        "price": float(r.now_cost or 4.0), "xp": float(r.xp_mean),
        "p_play": float(r.p_play), "xp_p90": _num(getattr(r, "xp_p90", None), float(r.xp_mean)),
        "selected_by_percent": _num(getattr(r, "selected_by_percent", None)),
    } for r in df.itertuples() if _num(r.now_cost)]
    sol = optimise_squad(pool, budget=req.budget, risk_lambda=req.risk_lambda,
                         locked_in=req.locked_in, excluded=req.excluded)
    result = {"summary": sol.summary(), "starting_xi": sol.starting_xi,
              "bench": sol.bench, "captain": sol.captain,
              "vice_captain": sol.vice_captain}
    if explain:
        from fplagent.agent.llm import explain_squad
        result["explanation"] = explain_squad(result)
    return result


class CaptaincyRequest(BaseModel):
    risk_lambda: float = Field(0.0, ge=0.0, le=1.0)
    top_n: int = 10
    gw: int | None = None


def _load_effective_ownership(season: int, gw: int | None) -> dict[int, float]:
    """Real sampled EO if it exists for this gameweek."""
    try:
        q = select(T.effective_ownership).where(T.effective_ownership.c.season == season)
        if gw is not None:
            q = q.where(T.effective_ownership.c.gw == gw)
        df = read_sql(q)
    except Exception:      # noqa: BLE001
        return {}
    if df.empty:
        return {}
    if gw is None:
        df = df.sort_values("gw").groupby("player_id").tail(1)
    return dict(zip(df["player_id"].astype(int), df["effective_ownership"].astype(float)))


@app.post("/optimise/captaincy", tags=["optimisers"])
def captaincy(req: CaptaincyRequest, season: int = CURRENT_SEASON,
              explain: bool = False):
    from fplagent.optimizers.captaincy import rank_captains
    from fplagent.pipeline.run import latest_predictions
    df = latest_predictions(season, req.gw)
    if df.empty:
        raise HTTPException(404, "No predictions stored.")
    real_eo = _load_effective_ownership(season, req.gw)
    cands = _captain_candidates(df, real_eo)
    ranked = rank_captains(cands, req.risk_lambda, req.top_n)
    result = {"candidates": ranked}
    if explain and ranked:
        from fplagent.agent.llm import explain_captaincy
        result["explanation"] = explain_captaincy(ranked)
    return result


class TransferRequest(BaseModel):
    squad_player_ids: list[int]
    bank: float = 0.0
    free_transfers: int = 1
    horizon: int = 4
    risk_lambda: float = 0.0
    purchase_prices: dict[int, float] | None = None


@app.post("/optimise/transfers", tags=["optimisers"])
def transfers(req: TransferRequest, season: int = CURRENT_SEASON,
              explain: bool = False):
    from fplagent.optimizers.transfers import plan_transfers
    from fplagent.pipeline.run import latest_predictions
    df = latest_predictions(season)
    if df.empty:
        raise HTTPException(404, "No predictions stored.")
    base_gw = int(df["gw"].min())
    horizon = list(range(base_gw, base_gw + req.horizon))

    purchase_prices = req.purchase_prices or {}

    def entry(r):
        pid = int(r.player_id)
        return {"player_id": pid, "name": r.web_name,
                "position": r.position, "club": str(getattr(r, "short_name", r.team_id)),
                "price": float(r.now_cost or 4.0), "xp": float(r.xp_mean),
                "purchase_price": purchase_prices.get(pid)}

    rows = [entry(r) for r in df.itertuples() if _num(r.now_cost)]
    squad = [p for p in rows if p["player_id"] in set(req.squad_player_ids)]
    if len(squad) != 15:
        raise HTTPException(400, f"Expected 15 squad players, matched {len(squad)}")
    pool = [p for p in rows if p["player_id"] not in set(req.squad_player_ids)][:260]
    xp_by_gw = {g: {p["player_id"]: p["xp"] for p in rows} for g in horizon}
    plan = plan_transfers(squad, pool, xp_by_gw, horizon, req.bank,
                          req.free_transfers, risk_lambda=req.risk_lambda)
    result = {"moves": plan.moves, "hits": plan.hits_taken,
              "net_gain": plan.net_gain, "status": plan.status,
              "reasoning": plan.reasoning}
    if explain and plan.moves:
        from fplagent.agent.llm import explain_transfer_plan
        result["explanation"] = explain_transfer_plan(result, horizon)
    return result


@app.get("/quality", tags=["system"])
def quality(limit: int = 50, explain: bool = False):
    """Recent data-quality events."""
    df = read_sql(select(T.data_quality_events))
    if df.empty:
        return {"events": []}
    result = {"events": df.sort_values("created_at", ascending=False)
              .head(limit).to_dict("records")}
    if explain and result["events"]:
        from fplagent.agent.llm import explain_data_quality
        result["explanation"] = explain_data_quality(result)
    return result


class HeadToHeadRequest(BaseModel):
    player_a_id: int
    player_b_id: int
    gw: int | None = None
    same_team: bool = False


@app.post("/compare/captaincy", tags=["optimisers"])
def compare_captaincy(req: HeadToHeadRequest, season: int = CURRENT_SEASON):
    """Head-to-head: P(A outscores B) this gameweek."""
    from fplagent.models.distributions import pairwise_win_probability
    from fplagent.pipeline.run import latest_predictions
    df = latest_predictions(season, req.gw)
    if df.empty:
        raise HTTPException(404, "No predictions stored.")
    row_a = df[df["player_id"] == req.player_a_id]
    row_b = df[df["player_id"] == req.player_b_id]
    if row_a.empty or row_b.empty:
        missing = [pid for pid, r in
                   [(req.player_a_id, row_a), (req.player_b_id, row_b)] if r.empty]
        raise HTTPException(404, f"No stored predictions for player id(s) {missing}")
    a, b = row_a.iloc[0], row_b.iloc[0]
    outcome = pairwise_win_probability(a["sim_quantiles"], b["sim_quantiles"],
                                       same_team=req.same_team)
    return {
        "player_a": {"player_id": int(a["player_id"]), "name": a["web_name"],
                     "xp": float(a["xp_mean"])},
        "player_b": {"player_id": int(b["player_id"]), "name": b["web_name"],
                     "xp": float(b["xp_mean"])},
        **outcome,
    }


class RankImpactRequest(BaseModel):
    current_overall_rank: int
    xp_p10: float
    xp_p50: float
    xp_p90: float
    effective_ownership: float = Field(0.0, ge=0.0, le=1.0)
    total_managers: int = TOTAL_MANAGERS


def _rank_tier(swing: float) -> str:
    """Bucket a raw swing estimate into a tier rather than presenting a
    number with false precision."""
    a = abs(swing)
    if a < 2_000:
        return "small"
    if a < 20_000:
        return "moderate"
    if a < 100_000:
        return "large"
    return "very large"


@app.post("/rank-impact", tags=["optimisers"])
def rank_impact(req: RankImpactRequest):
    """Translate a pick's point distribution into a rough rank-swing
    estimate."""
    import numpy as np
    rank = max(1, req.current_overall_rank)
    total = max(rank, req.total_managers)
    percentile = rank / total
    from scipy.stats import norm
    z = norm.ppf(np.clip(percentile, 0.001, 0.999))
    density = norm.pdf(z)
    swing_scale = total * density * (1.0 - req.effective_ownership) * 0.004
    upside_pts = max(0.0, req.xp_p90 - req.xp_p50)
    downside_pts = max(0.0, req.xp_p50 - req.xp_p10)
    gain = swing_scale * upside_pts
    loss = swing_scale * downside_pts
    return {
        "upside_tier": _rank_tier(gain),
        "downside_tier": _rank_tier(loss),
        "note": ("Tiers (small/moderate/large/very large), not a number -- the "
                 "underlying estimate isn't precise enough to justify a digit "
                 "count, see 'basis' below."),
        "basis": "approximate heuristic -- see docstring; not fitted from real rank data",
    }


@app.get("/brief", tags=["optimisers"])
def weekly_brief(season: int = CURRENT_SEASON, gw: int | None = None,
                 explain: bool = False):
    """One call that gathers captaincy shortlist, chip status and any
    consistency flags together."""
    from fplagent.models.consistency import check_prediction
    from fplagent.optimizers.captaincy import rank_captains
    from fplagent.pipeline.run import latest_predictions

    df = latest_predictions(season, gw)
    if df.empty:
        raise HTTPException(404, "No predictions stored.")
    real_eo = _load_effective_ownership(season, gw)
    cands = _captain_candidates(df, real_eo)
    captaincy_ranked = rank_captains(cands, 0.0, 5)

    flags = []
    for r in df.head(30).itertuples():
        f = check_prediction({"xp_mean": r.xp_mean, "p_play": r.p_play,
                              "xp_p10": r.xp_p10, "xp_p90": r.xp_p90})
        if f:
            flags.append({"player_id": int(r.player_id), "name": r.web_name,
                          "flags": [x.__dict__ for x in f]})

    chip_status = {"note": "pass chip_values/available_from_gw to POST /optimise/chips "
                           "for a full schedule; this briefing only flags whether any "
                           "stored predictions look unstable this week"}

    payload = {"season": season, "gw": gw, "top_captaincy": captaincy_ranked,
               "flagged_players": flags, "chip_status": chip_status}
    if explain:
        from fplagent.agent.llm import explain_weekly_briefing
        payload["briefing"] = explain_weekly_briefing(payload)
    return payload


@app.get("/dashboard/overview", tags=["dashboard"])
def dashboard_overview(season: int = CURRENT_SEASON, gw: int | None = None):
    """Single-call command centre payload for the v9 dashboard."""
    from fplagent.pipeline.run import latest_predictions
    from fplagent.db.repository import latest_snapshot_per_player

    df = latest_predictions(season, gw)
    if df.empty:
        raise HTTPException(404, "No predictions stored.")
    gw_value = int(df["gw"].min())
    df = _decision_pack(df, season, gw_value)
    current = resolve_target_gw(season)
    strength = read_sql(select(T.team_strength).where((T.team_strength.c.season == season) & (T.team_strength.c.gw == gw_value)))
    teams_df = read_sql(select(T.teams).where(T.teams.c.season == season))
    if not strength.empty and not teams_df.empty:
        strength = strength.merge(teams_df[["team_id","name","short_name"]], on="team_id", how="left")
    quality = read_sql(select(T.data_quality_events).order_by(T.data_quality_events.c.created_at.desc()).limit(200))
    snaps = latest_snapshot_per_player(season)
    top = df.sort_values("xp_mean", ascending=False).head(8)
    values = df.dropna(subset=["value"]).sort_values("value", ascending=False).head(8)
    differentials = df.sort_values("ownership_adjusted_upside", ascending=False).head(8)
    upside = df.sort_values("upside_points", ascending=False).head(8)
    quality_by_severity = quality["severity"].value_counts().to_dict() if not quality.empty else {}
    return {
        "season": season, "gw": gw_value, "target_gw": current, "model_version": str(df["model_version"].iloc[0]),
        "summary": {
            "players": int(len(df)), "mean_xp": round(float(df["xp_mean"].mean()), 3),
            "median_xp": round(float(df["xp_mean"].median()), 3),
            "high_upside_players": int((df["upside_points"] >= 4).sum()),
            "low_play_players": int((df["p_play"] < 0.75).sum()),
            "low_certainty_players": int((df["certainty"] < 0.45).sum()),
            "quality_events": int(len(quality)),
        },
        "top_players": top[["player_id","web_name","position","short_name","now_cost","xp_mean","xp_p10","xp_p50","xp_p90","p_play","p_start","p_sixty","p_haul_10plus","value","certainty","ownership_adjusted_upside"]].to_dict("records"),
        "best_value": values[["player_id","web_name","position","short_name","now_cost","xp_mean","value","p_play","certainty"]].to_dict("records"),
        "differentials": differentials[["player_id","web_name","position","short_name","now_cost","xp_mean","p_haul_10plus","ownership_adjusted_upside","ownership"]].to_dict("records"),
        "upside": upside[["player_id","web_name","position","short_name","now_cost","xp_mean","xp_p90","upside_points","p_haul_10plus","certainty"]].to_dict("records"),
        "team_strength": strength.replace({float("nan"): None}).to_dict("records"),
        "quality": quality_by_severity,
        "market": (snaps[["player_id","selected_by_percent","transfers_in_event","transfers_out_event","now_cost"]].replace({float("nan"): None}).to_dict("records") if not snaps.empty else []),
    }


class CompareRequest(BaseModel):
    player_ids: list[int] = Field(..., min_length=2, max_length=8)
    season: int = CURRENT_SEASON
    gw: int | None = None


@app.post("/players/compare", tags=["dashboard"])
def compare_players(req: CompareRequest):
    from fplagent.pipeline.run import latest_predictions
    df = latest_predictions(req.season, req.gw)
    if df.empty:
        raise HTTPException(404, "No predictions stored.")
    wanted = [int(x) for x in req.player_ids]
    rows = _decision_pack(df[df["player_id"].isin(wanted)].copy(), req.season, int(df["gw"].min()))
    missing = [x for x in wanted if x not in set(rows["player_id"].astype(int))]
    if rows.empty:
        raise HTTPException(404, "None of those players has a current prediction.")
    cols = ["player_id","web_name","position","short_name","team_id","now_cost","xp_mean","xp_p10","xp_p25","xp_p50","xp_p75","xp_p90","p_play","p_start","p_sixty","p_haul_10plus","p_blank_le2","xp_se","value","certainty","upside_points","downside_points","range_points","start_security","sixty_security","minutes_stability","floor_adjusted_xp","ceiling_adjusted_xp","ownership_adjusted_upside","ownership_adjusted_downside","quartile_range_points","start_gap","bench_risk","floor_value_per_million","ceiling_value_per_million","ownership"]
    return {"season": req.season, "gw": int(rows["gw"].iloc[0]), "players": rows[cols].to_dict("records"), "missing": missing}


@app.get("/fixtures/outlook", tags=["dashboard"])
def fixture_outlook(season: int = CURRENT_SEASON, start_gw: int | None = None, horizon: int = Query(6, ge=1, le=10)):
    """Team fixture difficulty/outlook using the latest fitted strength surface."""
    gw = start_gw or resolve_target_gw(season)
    if gw is None:
        raise HTTPException(404, "No gameweek data stored.")
    fixtures = read_sql(select(T.fixtures).where((T.fixtures.c.season == season) & (T.fixtures.c.gw >= gw) & (T.fixtures.c.gw < gw + horizon)))
    if fixtures.empty:
        raise HTTPException(404, "No fixtures available for this horizon.")
    strengths = read_sql(select(T.team_strength).where((T.team_strength.c.season == season) & (T.team_strength.c.gw == gw)))
    s = {int(r.team_id): r for r in strengths.itertuples()}
    teams_df = read_sql(select(T.teams).where(T.teams.c.season == season))
    names = {int(r.team_id): (r.name, r.short_name) for r in teams_df.itertuples()}
    rows = []
    for r in fixtures.itertuples():
        for team_id, opp_id, home in ((int(r.home_team_id), int(r.away_team_id), True), (int(r.away_team_id), int(r.home_team_id), False)):
            opp = s.get(opp_id); own = s.get(team_id)
            difficulty = 3.0
            if opp is not None:
                difficulty = float(np.clip(3.0 + 0.9 * float(opp.attack) - 0.9 * float(own.defence if own is not None else 0.0), 1.0, 5.0))
            rows.append({"gw": int(r.gw), "team_id": team_id, "team": names.get(team_id, (str(team_id), str(team_id)))[1], "opponent_id": opp_id, "opponent": names.get(opp_id, (str(opp_id), str(opp_id)))[1], "home": home, "difficulty": round(difficulty, 2)})
    out = pd.DataFrame(rows)
    team_summary = out.groupby(["team_id","team"], as_index=False).agg(avg_difficulty=("difficulty","mean"), best_fixture=("difficulty","min"), worst_fixture=("difficulty","max"), fixtures=("gw","count"))
    return {"season": season, "start_gw": gw, "horizon": horizon, "fixtures": out.to_dict("records"), "teams": team_summary.sort_values("avg_difficulty").to_dict("records")}


@app.get("/market/watch", tags=["dashboard"])
def market_watch(season: int = CURRENT_SEASON, limit: int = Query(40, ge=5, le=100)):
    """Market activity, resolved to player NAMES.

    `transfer_ratio` is net transfers divided by the player's current owner
    count, not divided by ownership percent. The previous version divided a
    count by a percent, producing a number whose only possible use was
    relative ordering and whose magnitude (25,000+) was meaningless. It is
    now a genuine "fraction of the current owner base that moved this
    period", on a scale where 0.05 means 5% of owners transferred the player
    in (net) and 0.5 means half the owner base did.
    """
    from fplagent.db.repository import latest_snapshot_per_player
    snaps = latest_snapshot_per_player(season)
    if snaps.empty:
        return {"players": []}
    snaps = snaps.copy()
    snaps["net_transfers"] = (snaps["transfers_in_event"].fillna(0)
                              - snaps["transfers_out_event"].fillna(0))

    # Ownership is a percent (28.4 == 28.4%), so multiply by total managers
    # / 100 to get an approximate count of owners.
    owners_count = (snaps["selected_by_percent"].fillna(0) / 100.0) * TOTAL_MANAGERS
    snaps["transfer_ratio"] = snaps["net_transfers"] / owners_count.replace(0, np.nan)

    snaps = snaps.sort_values(["net_transfers", "transfer_ratio"], ascending=False).head(limit)

    names = _player_names_map(season)
    snaps["web_name"] = snaps["player_id"].map(
        lambda pid: (names.get(int(pid), {}) or {}).get("web_name"))
    snaps["position"] = snaps["player_id"].map(
        lambda pid: (names.get(int(pid), {}) or {}).get("position"))
    snaps["short_name"] = snaps["player_id"].map(
        lambda pid: (names.get(int(pid), {}) or {}).get("short_name"))

    cols = ["player_id", "web_name", "position", "short_name",
            "selected_by_percent", "transfers_in_event", "transfers_out_event",
            "net_transfers", "transfer_ratio", "now_cost", "form",
            "chance_of_playing_next_round", "status", "news"]
    cols = [c for c in cols if c in snaps.columns]
    return {
        "players": snaps[cols].replace({float("nan"): None}).to_dict("records"),
        "note": "Transfer pressure is a market signal, not a guaranteed price-change prediction. transfer_ratio is net transfers as a fraction of the current owner base.",
    }


@app.get("/model/dashboard", tags=["dashboard"])
def model_dashboard(season: int = CURRENT_SEASON):
    history = read_sql(select(T.evaluation_runs).where(T.evaluation_runs.c.season == season).order_by(T.evaluation_runs.c.created_at.desc()).limit(20))
    quality = read_sql(select(T.data_quality_events).order_by(T.data_quality_events.c.created_at.desc()).limit(100))
    return {"evaluation_runs": history.replace({float("nan"): None}).to_dict("records"), "quality_events": quality.replace({float("nan"): None}).to_dict("records"), "version": __version__}


class DecisionLogRequest(BaseModel):
    season: int = CURRENT_SEASON
    gw: int = Field(..., ge=1, le=38)
    rec_type: str
    recommended: dict = Field(default_factory=dict)
    actual: dict = Field(default_factory=dict)


@app.post("/decisions", tags=["decisions"])
def log_decision(req: DecisionLogRequest):
    """Log what the system recommended against what the manager actually did."""
    from fplagent.db.session import session_scope
    from sqlalchemy import insert as sa_insert
    payload = {"recommended": req.recommended, "actual": req.actual}
    with session_scope() as s:
        result = s.execute(
            sa_insert(T.recommendations).values(
                season=req.season, gw=req.gw, rec_type=req.rec_type,
                payload=payload))
        rec_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
    return {"status": "logged", "rec_id": rec_id}


@app.post("/decisions/{rec_id}/outcome", tags=["decisions"])
def log_decision_outcome(rec_id: int, actual_outcome: dict):
    """Attach the realised result once the gameweek has been played."""
    from sqlalchemy import update as sa_update
    from fplagent.db.session import session_scope
    with session_scope() as s:
        res = s.execute(
            sa_update(T.recommendations)
            .where(T.recommendations.c.rec_id == rec_id)
            .values(actual_outcome=actual_outcome, evaluated_at=datetime.utcnow()))
        matched = res.rowcount
    if matched == 0:
        raise HTTPException(404, f"No logged decision with id {rec_id}")
    return {"status": "updated"}


@app.get("/decisions/track-record", tags=["decisions"])
def decisions_track_record(season: int = CURRENT_SEASON, rec_type: str | None = None):
    """Personal decision history, resolved to player names where applicable."""
    q = select(T.recommendations).where(T.recommendations.c.season == season)
    if rec_type:
        q = q.where(T.recommendations.c.rec_type == rec_type)
    df = read_sql(q)
    if df.empty:
        return {"season": season, "n_logged": 0,
                "summary": "No decisions logged yet for this season."}
    df = df.sort_values("gw")
    evaluated = df[df["evaluated_at"].notna()]

    names = _player_names_map(season)

    def _resolve(payload):
        if not isinstance(payload, dict):
            return payload
        out = dict(payload)
        for key in ("recommended", "actual"):
            sub = out.get(key)
            if not isinstance(sub, dict):
                continue
            resolved = dict(sub)
            for field, value in list(resolved.items()):
                if field.endswith("_id") and isinstance(value, int):
                    info = names.get(int(value)) or {}
                    if info.get("web_name"):
                        resolved[field.replace("_id", "_name")] = info["web_name"]
                elif field.endswith("_ids") and isinstance(value, list):
                    resolved[field.replace("_ids", "_names")] = [
                        (names.get(int(v)) or {}).get("web_name", v) for v in value
                    ]
            out[key] = resolved
        return out

    records = []
    for r in df.itertuples():
        row = {
            "rec_id": int(r.rec_id), "season": int(r.season), "gw": int(r.gw),
            "rec_type": str(r.rec_type),
            "payload": _resolve(r.payload if isinstance(r.payload, dict) else {}),
            "created_at": getattr(r, "created_at", None),
            "actual_outcome": getattr(r, "actual_outcome", None),
            "evaluated_at": getattr(r, "evaluated_at", None),
        }
        records.append(row)

    return {
        "season": season, "n_logged": int(len(df)),
        "n_evaluated": int(len(evaluated)),
        "decisions": records,
        "summary": (
            f"{len(df)} decision(s) logged, {len(evaluated)} with an outcome "
            "recorded. Log outcomes via POST /decisions/{rec_id}/outcome once "
            "each gameweek finishes to build up a personal track record."
        ),
    }