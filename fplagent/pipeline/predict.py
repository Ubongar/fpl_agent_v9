"""The prediction pipeline: fixtures -> team strength -> shares -> simulation
-> the predictions table.

THIS IS THE FILE v5 DID NOT HAVE.

v5's ops/retrain_job.py ingested a snapshot, evaluated a hardcoded calibration
curve, and exited. It never called fit_team_strength, never called
estimate_player_shares, never called player_expected_points, and nothing
anywhere in the codebase ever wrote a row to the predictions table. The models
were a library of well-written, unconnected functions with ingestion bolted to
one end and a mock dashboard bolted to the other.

Everything here runs in one pass and persists its output, because a prediction
that is not written down before the deadline is not evidence of anything.

MULTI-SEASON TEAM STRENGTH (v9)
-------------------------------
The Dixon-Coles fit is now given prior-season fixtures alongside the current
season's, with prior-season rows time-decayed to ~30% of an equivalent
current-season match. This fixes the "50 matches, rho at the boundary"
problem: the fit has enough history to identify the parameters even in
August. Prior-season fixture ids are per-season, so `team_match_results`
does not use them as identity; only goals and dates flow through. See
`features.build.load_fixtures_with_history`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from fplagent.config import CURRENT_SEASON, PARAMS
from fplagent.db import tables as T
from fplagent.db.repository import read_sql, upsert
from fplagent.features.build import (
    build_minutes_training_set, build_player_features, load_fixtures,
    load_fixtures_with_history, load_history, team_match_results, team_match_xg,
)
from fplagent.models.distributions import (
    combine_standard_errors, convolve_quantile_grids, prob_at_least_one,
    quantile_from_grid,
)
from fplagent.ingestion.external_apis import fetch_external_metrics
from fplagent.ingestion.quality import check_predictions
from fplagent.models.cards import CardRate
from fplagent.models.defcon import DefconRate
from fplagent.models.dixon_coles import expected_goals, score_matrix
from fplagent.models.minutes import MinutesModel, MinutesPrediction
from fplagent.models.simulation import SimPlayer, simulate_fixture
from fplagent.models.team_strength import (
    TeamStrengthModel, blend_with_market, fit_team_strength,
)

log = logging.getLogger(__name__)


def model_version(tag: str | None = None) -> str:
    return f"{tag or 'v9'}-{datetime.utcnow():%Y%m%d-%H%M}"


POOL_QUOTA = {"GK": 3, "DEF": 8, "MID": 9, "FWD": 6}


def _squad_for_team(features: pd.DataFrame, team_id: int,
                    quota: dict | None = None) -> list[SimPlayer]:
    quota = quota or POOL_QUOTA
    sub = features[features["team_id"] == team_id].copy()
    parts = [g.sort_values("p_play", ascending=False).head(quota.get(pos, 6))
             for pos, g in sub.groupby("position")]
    sub = pd.concat(parts) if parts else sub.head(0)
    out = []
    for _, r in sub.iterrows():
        mp = MinutesPrediction(
            p_play=float(r["p_play"]), p_sixty=float(r["p_sixty"]),
            p_start_est=float(r.get("p_start", 0.0) or 0.0),
            expected_minutes=float(r["expected_minutes"]),
            source=str(r["minutes_source"]),
            drivers=r["minutes_drivers"] if isinstance(r["minutes_drivers"], dict) else {},
        )
        out.append(SimPlayer(
            player_id=int(r["player_id"]), position=str(r["position"]),
            minutes=mp,
            np_goal_share=float(r["np_goal_share"]),
            assist_share=float(r["assist_share"]),
            penalty_share=float(r["penalty_share"]),
            penalty_order=int(r.get("penalty_order", 99) or 99),
            fk_share=float(r.get("fk_share", 0.0) or 0.0),
            fk_order=int(r.get("fk_order", 99) or 99),
            corner_share=float(r.get("corner_share", 0.0) or 0.0),
            corner_order=int(r.get("corner_order", 99) or 99),
            defcon_rate=DefconRate(float(r["defcon_per90"]),
                                   float(r["defcon_dispersion"]), 0.0,
                                   str(r["defcon_confidence"])),
            card_rate=CardRate(float(r.get("yellow_per90", 0.13) or 0.13),
                               float(r.get("red_per90", 0.007) or 0.007),
                               0.0, str(r.get("card_confidence", "prior_only"))),
            baseline_bps_per90=float(r["baseline_bps_per90"]),
            unavailable=bool(r.get("unavailable", False)),
            name=str(r["web_name"]),
            team_id=int(team_id),
            now_cost=float(r["now_cost"]) if pd.notna(r.get("now_cost")) else None,
        ))
    return out


def _market_lambdas(fixture_id: int, season: int, before: datetime | None = None) -> tuple[float | None, float | None]:
    conds = [T.match_odds.c.season == season, T.match_odds.c.fixture_id == fixture_id]
    if before is not None:
        conds.append(T.match_odds.c.pulled_at < before)
    df = read_sql(select(T.match_odds).where(*conds))
    if df.empty:
        return None, None
    latest = df.sort_values("pulled_at").iloc[-1]
    if before is not None and pd.notna(latest["pulled_at"]):
        age_hours = (before - pd.Timestamp(latest["pulled_at"]).to_pydatetime()).total_seconds() / 3600.0
        if age_hours > 72:
            return None, None
    return (float(latest["implied_home_xg"]) if pd.notna(latest["implied_home_xg"]) else None,
            float(latest["implied_away_xg"]) if pd.notna(latest["implied_away_xg"]) else None)


def _fit_minutes_model(cutoff: datetime, season: int) -> MinutesModel | None:
    if not PARAMS.use_learned_minutes:
        log.info("learned minutes model disabled by PARAMS; using heuristic")
        return None
    try:
        X, y_play, y_start, y_sixty = build_minutes_training_set(cutoff, season)
        log.info(
            "minutes training set: X=%s y_play=%s(mean=%.3f) "
            "y_start=%s(mean=%.3f) y_sixty=%s(mean=%.3f)",
            X.shape, y_play.shape, float(y_play.mean()) if len(y_play) else 0.0,
            y_start.shape, float(y_start.mean()) if len(y_start) else 0.0,
            y_sixty.shape, float(y_sixty.mean()) if len(y_sixty) else 0.0,
        )
        model = MinutesModel()
        ok = model.fit(X, y_play, y_start, y_sixty)
        if not ok:
            log.info("MinutesModel not used: %s", model.refusal_reason)
            return None
        log.info(
            "minutes model fitted: n_train=%d calibrated=%s holdout=%s",
            model.n_train,
            any(c is not None for c in (
                model.play_calibrator, model.start_calibrator, model.sixty_calibrator)),
            model.holdout,
        )
        return model
    except Exception as e:      # noqa: BLE001
        log.warning(
            "learned minutes model unavailable, using heuristic: %s: %s",
            type(e).__name__, e, exc_info=True,
        )
        return None


def run_predictions(target_gw: int, season: int = CURRENT_SEASON,
                    cutoff: datetime | None = None,
                    version: str | None = None,
                    n_sims: int | None = None,
                    persist: bool = True,
                    use_market: bool = True,
                    strict_point_in_time: bool | None = None) -> pd.DataFrame:
    version = version or model_version()
    n_sims = n_sims or PARAMS.n_sims

    gw_row = read_sql(select(T.gameweeks).where(
        (T.gameweeks.c.season == season) & (T.gameweeks.c.gw == target_gw)))
    deadline = None
    if not gw_row.empty and pd.notna(gw_row.iloc[0]["deadline_time"]):
        deadline = pd.Timestamp(gw_row.iloc[0]["deadline_time"]).to_pydatetime()
    cutoff = cutoff or deadline or datetime.utcnow()
    if strict_point_in_time is None:
        cutoff_ts = pd.Timestamp(cutoff)
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.tz_localize("UTC")
        else:
            cutoff_ts = cutoff_ts.tz_convert("UTC")
        strict_point_in_time = cutoff_ts < (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1))

    # Current-season fixtures: used for the target-gameweek filter, the xG
    # lookup, and any code that expects a per-fixture_id join that must not
    # collide with prior-season ids.
    fixtures = load_fixtures(season)
    gw_fixtures = fixtures[fixtures["gw"] == target_gw]
    if gw_fixtures.empty:
        log.warning("no fixtures for season %s gw %s", season, target_gw)
        return pd.DataFrame()

    # The team-strength fit sees prior seasons too. Prior-season rows are
    # keyed on the persistent team `code` and mapped to the current
    # season's `team_id`; the `current_season` argument ensures the
    # "gw < cutoff_gw" filter is applied only to the current season.
    fixtures_with_history = load_fixtures_with_history(season)

    matches = team_match_results(
        fixtures_with_history, target_gw, cutoff,
        current_season=season,
        prior_season_age_multiplier=PARAMS.prior_season_age_multiplier,
    )
    history_for_xg = load_history(cutoff, season)
    xg_matches = team_match_xg(history_for_xg, fixtures, target_gw)
    ts = fit_team_strength(matches, target=PARAMS.team_strength_target,
                           xg_matches=xg_matches,
                           xg_weight=PARAMS.xg_blend_weight)
    log.info("team strength: %d matches, prior_only=%s, converged=%s, "
             "league_avg_goals=%.2f, home_adv=%.3f, rho=%.3f",
             ts.n_matches_fit, ts.prior_only, ts.converged,
             ts.league_avg_goals, ts.home_adv_mult, ts.rho)
    (_persist_team_strength(ts, season, target_gw, version) if persist else None)
    log.info("Fetching external API metrics (Elo, Per-90s)...")
    fetch_external_metrics()

    features = build_player_features(
        cutoff, target_gw, season,
        minutes_model=_fit_minutes_model(cutoff, season),
        strict_point_in_time=bool(strict_point_in_time),
    )
    if features.empty:
        log.warning("no player features built")
        return pd.DataFrame()

    model_conf = float(np.clip(ts.effective_n / 200.0, 0.0, 1.0))
    rows: list[dict] = []

    for _, fx in gw_fixtures.iterrows():
        home_id, away_id = int(fx["home_team_id"]), int(fx["away_team_id"])
        lam, mu = ts.expected_lambdas(home_id, away_id)

        market_used = False
        if use_market:
            m_lam, m_mu = _market_lambdas(int(fx["fixture_id"]), season, cutoff)
            market_used = m_lam is not None and m_mu is not None
            lam, mu = blend_with_market(lam, mu, m_lam, m_mu, model_conf)

        matrix = score_matrix(lam, mu, ts.rho, max_goals=PARAMS.max_goals)
        hxg, axg = expected_goals(matrix)

        home_squad = _squad_for_team(features, home_id)
        away_squad = _squad_for_team(features, away_id)
        if not home_squad and not away_squad:
            continue

        sim = simulate_fixture(home_squad, away_squad, matrix, hxg, axg,
                               n_sims=n_sims, seed=int(fx["fixture_id"]),
                               league_avg_xg=ts.league_avg_goals)

        for squad, is_home, team_xg, opp_xg in (
            (home_squad, True, hxg, axg), (away_squad, False, axg, hxg),
        ):
            for p in squad:
                s = sim.summarise(p.player_id)
                comp = {k: round(v, 3) for k, v in sim.components[p.player_id].items()}
                rows.append({
                    "model_version": version, "season": season, "gw": target_gw,
                    "fixture_id": int(fx["fixture_id"]), "player_id": p.player_id,
                    "team_id": int(p.team_id if p.team_id is not None else (home_id if is_home else away_id)),
                    "position": str(p.position),
                    "now_cost": p.now_cost,
                    "deadline_time": deadline, "created_at": datetime.utcnow(),
                    "p_play": round(p.minutes.p_play, 4),
                    "p_start": round(p.minutes.p_start, 4),
                    "p_sixty": round(p.minutes.p_sixty, 4),
                    "xp_mean": round(s["xp_mean"], 3),
                    "xp_p10": round(s["xp_p10"], 2), "xp_p25": round(s["xp_p25"], 2),
                    "xp_p50": round(s["xp_p50"], 2), "xp_p75": round(s["xp_p75"], 2),
                    "xp_p90": round(s["xp_p90"], 2),
                    "p_haul_10plus": round(s["p_haul_10plus"], 4),
                    "p_blank_le2": round(s["p_blank_le2"], 4),
                    "xp_se": round(s["xp_se"], 4),
                    "n_sims": int(s["n_sims"]),
                    "components": comp,
                    "features": {
                        "is_home": is_home,
                        "team_xg": round(team_xg, 3), "opp_xg": round(opp_xg, 3),
                        "market_used": market_used,
                        "np_goal_share": round(p.np_goal_share, 4),
                        "assist_share": round(p.assist_share, 4),
                        "penalty_share": round(p.penalty_share, 4),
                        "defcon_per90": round(p.defcon_rate.per90, 2) if p.defcon_rate else 0.0,
                        "minutes_source": p.minutes.source,
                        "minutes_drivers": p.minutes.drivers,
                    },
                    "sim_quantiles": sim.quantile_grid(p.player_id),
                })

    df = pd.DataFrame(rows)
    report = check_predictions(df)
    report.persist()
    if report.failed:
        failed = [c.name for c in report.checks if c.failed]
        raise RuntimeError(f"prediction quality gate failed: {failed}")

    if persist and not df.empty:
        upsert(T.predictions, df.to_dict("records"),
               ["model_version", "season", "gw", "player_id", "fixture_id"])
        log.info("persisted %d predictions for GW%d as %s", len(df), target_gw, version)
    return df


def _persist_team_strength(ts: TeamStrengthModel, season: int, gw: int,
                           version: str) -> None:
    rows = [{
        "season": season, "gw": gw, "team_id": t, "model_version": version,
        "attack": round(ts.attack_mult[t], 4),
        "defence": round(ts.defence_mult.get(t, 1.0), 4),
        "home_adv": round(ts.home_adv_mult, 4), "rho": round(ts.rho, 4),
        "league_avg_goals": round(ts.league_avg_goals, 4),
        "n_matches_fit": ts.n_matches_fit,
        "effective_n": round(ts.effective_n, 2),
        "target": ts.target,
    } for t in ts.attack]
    if rows:
        upsert(T.team_strength, rows, ["season", "gw", "team_id", "model_version"])


def aggregate_to_gameweek(preds: pd.DataFrame) -> pd.DataFrame:
    """Collapse fixture-level predictions to gameweek totals."""
    if preds.empty:
        return preds
    agg = preds.groupby(["player_id", "gw", "season"], as_index=False).agg(
        xp_mean=("xp_mean", "sum"),
        p_play=("p_play", prob_at_least_one),
        p_start=("p_start", prob_at_least_one),
        p_sixty=("p_sixty", prob_at_least_one),
        n_fixtures=("fixture_id", "count"),
        p_haul_10plus=("p_haul_10plus", "first"),
        xp_se=("xp_se", combine_standard_errors),
        model_version=("model_version", "first"),
        team_id=("team_id", "first") if "team_id" in preds.columns else ("player_id", "first"),
        position=("position", "first") if "position" in preds.columns else ("player_id", "first"),
        now_cost=("now_cost", "first") if "now_cost" in preds.columns else ("player_id", "first"),
    )
    if "sim_quantiles" in preds.columns:
        grids = (preds.groupby(["player_id", "gw", "season"])["sim_quantiles"]
                      .apply(lambda s: convolve_quantile_grids(list(s))))
        agg["sim_quantiles"] = [
            grids.get((r.player_id, r.gw, r.season), []) for r in agg.itertuples()]
        haul = []
        for r in agg.itertuples():
            g = np.asarray(getattr(r, "sim_quantiles", []) or [], dtype=float)
            if g.size > 1:
                qgrid = np.linspace(0.0, 1.0, g.size)
                cdf_at = float(np.interp(10.0, g, qgrid, left=0.0, right=1.0))
                haul.append(round(float(np.clip(1.0 - cdf_at, 0.0, 1.0)), 4))
            else:
                haul.append(np.nan)
        agg["p_haul_10plus"] = haul
        for q, col in ((0.10, "xp_p10"), (0.25, "xp_p25"), (0.50, "xp_p50"),
                       (0.75, "xp_p75"), (0.90, "xp_p90")):
            agg[col] = [round(quantile_from_grid(g, q), 2) for g in agg["sim_quantiles"]]
    if "p_blank_le2" in preds.columns:
        single = preds.groupby(["player_id", "gw", "season"])["p_blank_le2"].first()
        blank = []
        for r in agg.itertuples():
            g = np.asarray(getattr(r, "sim_quantiles", []) or [], dtype=float)
            if r.n_fixtures > 1 and g.size > 1:
                cdf_at = float(np.interp(2.0, g, np.linspace(0.0, 1.0, g.size), left=0.0, right=1.0))
                blank.append(round(float(np.clip(cdf_at, 0.0, 1.0)), 4))
            else:
                blank.append(single.get((r.player_id, r.gw, r.season)))
        agg["p_blank_le2"] = np.clip(np.asarray(blank, dtype=float), 0.0, 1.0)
    return agg