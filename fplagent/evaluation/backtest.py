"""Walk-forward backtest against stored predictions.

Two ways to run it, and the difference matters:

* replay=False (default, and the only one that is real evidence): score the
  predictions the system ACTUALLY WROTE before each deadline. This is the only
  honest measure of live performance.
* replay=True: re-run run_predictions() at each historical deadline using the
  point-in-time feature path. Necessary for evaluating a model change against
  history, and honest as long as every feature respects `known_at` -- which is
  enforced in the repository layer rather than left to discipline.

v5's backtest passed predict_fn and actuals_fn that both returned {}, so n was
always 0, so the gate always self-skipped and returned True. It measured
nothing, and reported success while doing so.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import select

from fplagent.config import CURRENT_SEASON
from fplagent.db import tables as T
from fplagent.db.repository import insert_rows, read_sql
from fplagent.evaluation import baselines as B
from fplagent.evaluation import metrics as M
from fplagent.models.distributions import convolve_quantile_grids, prob_at_least_one

log = logging.getLogger(__name__)


def _actuals(season: int, gw: int) -> pd.DataFrame:
    df = read_sql(select(T.player_fixture_stats).where(
        (T.player_fixture_stats.c.season == season)
        & (T.player_fixture_stats.c.gw == gw)))
    if df.empty:
        return df
    # Sum across fixtures: a double gameweek's actual score is both legs.
    return (df.groupby("player_id", as_index=False)
              .agg(actual=("total_points", "sum"),
                   minutes=("minutes", "sum"),
                   started=("starts", "max")))


def _deadline(season: int, gw: int) -> datetime | None:
    gw_row = read_sql(select(T.gameweeks).where(
        (T.gameweeks.c.season == season) & (T.gameweeks.c.gw == gw)))
    if gw_row.empty or pd.isna(gw_row.iloc[0]["deadline_time"]):
        return None
    return pd.Timestamp(gw_row.iloc[0]["deadline_time"]).to_pydatetime()


def _ep_next(season: int, gw: int) -> pd.Series:
    """FPL's own `ep_next` as it stood before this gameweek's deadline.

    The snapshot table stores it precisely so this baseline can be scored;
    v7 checked for an `ep_next` column that nothing ever merged in, so the
    strongest of the three baselines was silently never evaluated.
    """
    deadline = _deadline(season, gw)
    if deadline is None:
        return pd.Series(dtype=float)
    snaps = read_sql(select(T.player_season_snapshots).where(
        (T.player_season_snapshots.c.season == season)
        & (T.player_season_snapshots.c.pulled_at < deadline)))
    if snaps.empty or "ep_next" not in snaps:
        return pd.Series(dtype=float)
    latest = snaps.sort_values("pulled_at").groupby("player_id").tail(1)
    return latest.set_index("player_id")["ep_next"].astype(float)


def _stored_predictions(season: int, gw: int, version: str | None) -> pd.DataFrame:
    conds = [T.predictions.c.season == season, T.predictions.c.gw == gw]
    if version:
        conds.append(T.predictions.c.model_version == version)
    df = read_sql(select(T.predictions).where(*conds))
    if df.empty:
        return df
    # Live evidence is strictly pre-deadline. Never fall back to a post-deadline
    # row just because an historical gameweek has no valid stored prediction.
    # That kind of fallback makes the track record look complete while scoring
    # information the manager could not have had at the time.
    deadline = _deadline(season, gw)
    if deadline is None:
        return pd.DataFrame()
    created = pd.to_datetime(df["created_at"])
    df = df[created < pd.Timestamp(deadline)]
    if df.empty:
        return df
    latest = df.sort_values("created_at")["model_version"].iloc[-1]
    df = df[df["model_version"] == latest]
    # Fixture-level -> gameweek level. Means add; quantile grids convolve;
    # probabilities combine via prob_at_least_one (models.distributions),
    # not max() -- a DGW player has two chances to start, not one.
    agg = df.groupby("player_id", as_index=False).agg(
        xp_mean=("xp_mean", "sum"),
        p_play=("p_play", prob_at_least_one),
        p_start=("p_start", prob_at_least_one),
        p_sixty=("p_sixty", prob_at_least_one),
        model_version=("model_version", "first"),
    )
    grids = (df.groupby("player_id")["sim_quantiles"]
               .apply(lambda s: _convolve_grids(list(s))))
    agg["sim_quantiles"] = agg["player_id"].map(grids)
    return agg


# The convolution lives in models.distributions so the serving path and the
# evaluation path cannot disagree about what a double gameweek looks like.
_convolve_grids = convolve_quantile_grids


def backtest(season: int = CURRENT_SEASON, gw_from: int = 2, gw_to: int | None = None,
             version: str | None = None, replay: bool = False,
             persist: bool = True) -> dict:
    gw_to = gw_to or gw_from
    frames = []

    for gw in range(gw_from, gw_to + 1):
        actual = _actuals(season, gw)
        if actual.empty:
            log.info("GW%d has no actuals yet; skipping", gw)
            continue

        if replay:
            from fplagent.pipeline.predict import run_predictions
            cutoff = _deadline(season, gw)
            preds = run_predictions(gw, season, cutoff=cutoff,
                                    version=version or "replay", persist=False)
            if preds.empty:
                continue
            # v6 aggregated replay probabilities with max() while the stored
            # path used prob_at_least_one, and threw the quantile grids away
            # entirely, so CRPS and PIT were unavailable for any historical
            # evaluation. Both now go through the same aggregation as live.
            from fplagent.pipeline.predict import aggregate_to_gameweek
            preds = aggregate_to_gameweek(preds)
        else:
            preds = _stored_predictions(season, gw, version)

        if preds.empty:
            log.info("GW%d has no stored predictions; skipping", gw)
            continue

        # LEFT join, not inner. An inner join silently drops every player who
        # was predicted but did not appear, which is survivorship bias in the
        # model's favour: the confident predictions for players who then did
        # not feature are exactly the ones that should be punished. A player
        # with no actuals row scored zero, and that is the honest label.
        merged = preds.merge(actual, on="player_id", how="left")
        merged["actual"] = merged["actual"].fillna(0.0)
        merged["minutes"] = merged["minutes"].fillna(0)
        merged["started"] = merged["started"].fillna(0)
        merged["gw"] = gw
        ep = _ep_next(season, gw)
        if not ep.empty:
            merged["ep_next"] = merged["player_id"].map(ep)
        frames.append(merged)

    if not frames:
        return {"n": 0, "note": "no overlapping predictions and actuals"}

    df = pd.concat(frames, ignore_index=True)
    players = read_sql(select(T.players).where(T.players.c.season == season))
    if not players.empty:
        current = players[["player_id", "position", "now_cost"]].rename(
            columns={"position": "_current_position", "now_cost": "_current_now_cost"})
        df = df.merge(current, on="player_id", how="left")
        if "position" not in df:
            df["position"] = df["_current_position"]
        else:
            df["position"] = df["position"].fillna(df["_current_position"])
        if "now_cost" not in df:
            df["now_cost"] = df["_current_now_cost"]
        else:
            df["now_cost"] = df["now_cost"].fillna(df["_current_now_cost"])
        df = df.drop(columns=[c for c in ("_current_position", "_current_now_cost") if c in df.columns])

    result: dict = {
        "season": season, "gw_from": gw_from, "gw_to": gw_to,
        "model_version": version or (df["model_version"].iloc[0]
                                     if "model_version" in df else "replay"),
        "n": int(len(df)),
        "mae": round(M.mae(df["xp_mean"], df["actual"]), 4),
        "rmse": round(M.rmse(df["xp_mean"], df["actual"]), 4),
        "spearman_overall": round(M.spearman(df["xp_mean"], df["actual"]), 4),
        "spearman_by_position": M.spearman_by_position(df),
          # Captaincy metrics rank by raw xP. The risk-tilted version
        # (effective_value with a ceiling bonus) was tested at risk 0.4 on
        # GW2-5 and produced a WORSE capture rate (0.097 vs 0.139). The
        # cause is compressed ceilings: on early-season data every attacker
        # sits at p90 10-12, so tilting toward ceiling picks from a pool of
        # near-identical candidates and just adds noise. Reverting to the
        # risk-0 ranking, which is what the model's own top-of-list would
        # produce without modification. The CLI default remains 0.25 because
        # a human reading a ranked list can see context the metric cannot.
        "top5_captain_hit_rate": round(M.top_k_hit_rate(df, 5), 4),
        "top10_captain_hit_rate": round(M.top_k_hit_rate(df, 10), 4),
        "captaincy_points_captured": round(M.captaincy_points_captured(df), 4),
    }

    # --- distributional scoring ------------------------------------------
    grids = [g for g in df.get("sim_quantiles", pd.Series(dtype=object))
             if isinstance(g, (list, np.ndarray)) and len(g) > 1]
    if len(grids) >= 20:
        mask = df["sim_quantiles"].apply(
            lambda g: isinstance(g, (list, np.ndarray)) and len(g) > 1)
        sub = df[mask]
        crps = [M.crps_from_quantiles(g, a)
                for g, a in zip(sub["sim_quantiles"], sub["actual"])]
        pit = M.pit_values(list(sub["sim_quantiles"]), sub["actual"].values)
        ks, hist = M.pit_uniformity(pit)
        result["crps"] = round(float(np.mean(crps)), 4)
        result["pit_ks_stat"] = round(ks, 4)
        result["pit_histogram"] = [round(h, 4) for h in hist]
        result["pit_interpretation"] = _interpret_pit(hist)

    # --- probability calibration ------------------------------------------
    # Calibrate each probability against the event it actually predicts.
    # v6 scored p_sixty against the `starts` flag. A starter withdrawn on 55
    # minutes started and did not reach sixty, so that comparison measured
    # neither quantity.
    if "p_sixty" in df and "minutes" in df:
        reached60 = (df["minutes"].fillna(0).astype(float) >= 60).astype(float)
        curve = M.calibration_curve(df["p_sixty"].values, reached60.values)
        result["sixty_calibration"] = curve
        result["sixty_calibration_ece"] = round(M.expected_calibration_error(curve), 4)
    if "p_start" in df and "started" in df:
        started = df["started"].fillna(0).astype(float).clip(0, 1)
        curve = M.calibration_curve(df["p_start"].values, started.values)
        result["start_calibration"] = curve
        result["start_calibration_ece"] = round(M.expected_calibration_error(curve), 4)
    if "p_play" in df and "minutes" in df:
        played = (df["minutes"].fillna(0).astype(float) > 0).astype(float)
        curve = M.calibration_curve(df["p_play"].values, played.values)
        result["play_calibration"] = curve
        result["play_calibration_ece"] = round(M.expected_calibration_error(curve), 4)

    # --- baselines, scored on identical rows ------------------------------
    # Every baseline is evaluated on EXACTLY the same (player, gameweek) rows
    # with EXACTLY the same metrics. Anything less is not a comparison.
    bl: dict[str, dict] = {}

    def _score_frame(pred_col: str) -> dict:
        sub = df[df[pred_col].notna()]
        if len(sub) < 10:
            return {"n": int(len(sub))}
        return {
            "n": int(len(sub)),
            "mae": round(M.mae(sub[pred_col], sub["actual"]), 4),
            "rmse": round(M.rmse(sub[pred_col], sub["actual"]), 4),
            "spearman": round(M.spearman(sub[pred_col], sub["actual"]), 4),
            "top5_hit": round(M.top_k_hit_rate(sub, 5, pred_col, "actual"), 4),
        }

    roll_values = []
    price_values = []
    for target_gw in sorted(df["gw"].unique()):
        mask = df["gw"] == target_gw
        deadline = _deadline(season, int(target_gw))
        if deadline is None:
            continue
        history = read_sql(select(T.player_fixture_stats).where(
            (T.player_fixture_stats.c.season == season)
            & (T.player_fixture_stats.c.known_at < deadline)))
        roll = B.rolling_mean_baseline(history, int(target_gw)) if not history.empty else pd.Series(dtype=float)
        roll_values.append(df.loc[mask, "player_id"].map(roll))

        # Price baseline must use the last price known before this deadline,
        # not today's price after months of price changes.
        snaps = read_sql(select(T.player_season_snapshots).where(
            (T.player_season_snapshots.c.season == season)
            & (T.player_season_snapshots.c.pulled_at < deadline)))
        if not snaps.empty:
            latest = snaps.sort_values("pulled_at").groupby("player_id").tail(1)
            price_frame = latest[["player_id", "now_cost"]].dropna()
            price_pred = B.price_baseline(price_frame.rename(columns={"now_cost": "now_cost"}))
        else:
            price_pred = pd.Series(dtype=float)
        price_values.append(df.loc[mask, "player_id"].map(price_pred))
    if roll_values:
        df["baseline_rolling"] = pd.concat(roll_values).reindex(df.index)
        bl["rolling_6gw"] = _score_frame("baseline_rolling")
    if price_values:
        df["baseline_price"] = pd.concat(price_values).reindex(df.index)
        bl["price"] = _score_frame("baseline_price")

    if "ep_next" in df.columns:
        bl["fpl_ep_next"] = _score_frame("ep_next")

    model_scores = {"mae": result["mae"], "rmse": result["rmse"],
                    "spearman": result["spearman_overall"],
                    "top5_hit": result["top5_captain_hit_rate"]}
    result["baselines"] = bl
    result["baseline_comparison"] = B.compare(model_scores, bl)
    # A bare "is our Spearman higher" boolean gated version promotion in v6.
    # With one season of highly correlated rows a 0.02 gap is noise. The
    # comparison is now block-bootstrapped BY GAMEWEEK, which respects the
    # correlation structure, and promotion requires the interval to clear
    # zero rather than the point estimate to.
    result["baseline_significance"] = B.bootstrap_comparison(df, bl)
    result["beat_best_baseline"] = bool(
        result["baseline_significance"].get("beats_all_significantly", False))

    if persist:
        insert_rows(T.evaluation_runs, [{
            "model_version": result["model_version"], "season": season,
            "gw_from": gw_from, "gw_to": gw_to, "n": result["n"],
            "mae": result["mae"], "rmse": result["rmse"],
            "spearman_overall": result["spearman_overall"],
            "spearman_by_position": result["spearman_by_position"],
            "top5_captain_hit_rate": result["top5_captain_hit_rate"],
            "top10_captain_hit_rate": result["top10_captain_hit_rate"],
            "crps": result.get("crps"), "pit_ks_stat": result.get("pit_ks_stat"),
            "pit_histogram": result.get("pit_histogram"),
            "baselines": bl, "beat_best_baseline": result["beat_best_baseline"],
        }])
    return result


def _interpret_pit(hist: list[float]) -> str:
    """Plain-English read of the PIT histogram, because a KS statistic tells
    you something is wrong without telling you what."""
    if not hist:
        return "insufficient data"
    n = len(hist)
    edges = hist[0] + hist[-1]
    middle = sum(hist[n // 3: 2 * n // 3])
    uniform_edges = 2.0 / n
    uniform_middle = (2 * n // 3 - n // 3) / n
    if edges > uniform_edges * 1.5:
        return ("U-shaped: predicted distributions are too NARROW. The model is "
                "overconfident -- real outcomes land in the tails more often than "
                "it expects. Widen the variance, especially for minutes.")
    if middle > uniform_middle * 1.3:
        return ("Centre-heavy: predicted distributions are too WIDE. The model is "
                "underconfident -- outcomes cluster nearer the median than it "
                "expects. Tighten the variance.")
    return "Approximately uniform: the predicted distributions are well calibrated."
