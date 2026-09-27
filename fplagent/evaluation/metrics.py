"""Evaluation metrics that actually mean something for FPL.

WHY MAE IS NOT ENOUGH (v5 audit, section 3.1)
---------------------------------------------
FPL scores are zero-inflated and heavy-tailed: most players blank, a few haul.
A model that predicts 2.0 for every player in the league gets a respectable
MAE and is worth precisely nothing, because it cannot rank anybody. v5 gated
its entire retraining process on MAE alone.

What actually drives decisions:

  * RANK, within position. You never choose between a midfielder and a
    goalkeeper; you choose between midfielders. Spearman within position is
    the metric that maps onto the decision you are really making.
  * TOP-K HIT RATE. For captaincy, only the top of the ranking matters. Being
    well calibrated on the 400th-best player is irrelevant.
  * DISTRIBUTIONAL CALIBRATION. The entire point of running 20,000
    simulations is the distribution. If you claim p90 = 14, scores should
    exceed 14 about 10% of the time. CRPS scores the whole predicted
    distribution against the realised point; the PIT histogram shows whether
    it is over- or under-dispersed. v5 computed neither and so had no way to
    know whether its Monte Carlo was telling the truth.

CAPTAINCY METRICS USE EFFECTIVE VALUE, NOT RAW xP (v9)
------------------------------------------------------
`top_k_hit_rate` and `captaincy_points_captured` rank candidates. If they
rank by `xp_mean`, they will pick defenders and goalkeepers as captains in
nearly every gameweek, because defensive clean-sheet equity gives them the
highest floor in expected points. That is not what a good captain choice
looks like: the armband's doubling effect rewards CEILING, and ceiling
belongs to attackers.

The metrics now accept a `risk_lambda` parameter and, when nonzero, rank
by `optimizers.objective.effective_value` -- the same function the CLI's
`captains` command uses. At risk_lambda=0.0 the behaviour is identical to
before (pure xP), which preserves the ability to compare like for like
against older runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual))))


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(actual)) ** 2)))


def spearman(pred: np.ndarray, actual: np.ndarray) -> float:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    if len(pred) < 3 or np.all(pred == pred[0]) or np.all(actual == actual[0]):
        return float("nan")
    return float(stats.spearmanr(pred, actual).statistic)


def spearman_by_position(df: pd.DataFrame, pred_col: str = "xp_mean",
                         actual_col: str = "actual",
                         pos_col: str = "position") -> dict[str, float]:
    out = {}
    for pos, g in df.groupby(pos_col):
        out[str(pos)] = round(spearman(g[pred_col].values, g[actual_col].values), 4)
    return out


def _rank_column(df: pd.DataFrame, risk_lambda: float,
                 fallback_col: str = "xp_mean") -> np.ndarray:
    """The column the captaincy metrics sort by.

    At risk_lambda == 0 this is `fallback_col` unchanged (raw xP), which
    preserves backward compatibility with older evaluation runs. At any
    nonzero risk_lambda, `effective_value` is computed from whatever
    columns are present in `df` (xp_mean, xp_p90, selected_by_percent /
    eo). Missing optional columns are treated as zero, which reduces
    effective_value to xp_mean for that row.
    """
    if risk_lambda == 0.0:
        return df[fallback_col].to_numpy(dtype=float)
    from fplagent.optimizers.objective import effective_value
    rows = df.to_dict("records")
    return np.asarray([
        effective_value({
            "xp": float(r.get(fallback_col, 0.0)),
            "xp_p90": float(r.get("xp_p90", r.get(fallback_col, 0.0)) or
                            r.get(fallback_col, 0.0)),
            "eo": float(r.get("eo", r.get("selected_by_percent", 0.0)) or 0.0),
        }, risk_lambda)
        for r in rows
    ], dtype=float)


def top_k_hit_rate(df: pd.DataFrame, k: int = 5, pred_col: str = "xp_mean",
                   actual_col: str = "actual", gw_col: str = "gw",
                   risk_lambda: float = 0.0) -> float:
    """Fraction of gameweeks where the actual highest scorer was inside the
    model's top k. This is the captaincy question, asked directly.

    At risk_lambda=0.0, this ranks by raw xP and will pick defenders. At
    nonzero risk_lambda, it ranks by effective value and will pick players
    with genuine ceiling -- which is the question captaincy actually asks.
    """
    if df.empty or gw_col not in df.columns:
        return float("nan")
    ranked = df.copy()
    ranked["_rank"] = _rank_column(ranked, risk_lambda, pred_col)
    hits, total = 0, 0
    for _, g in ranked.groupby(gw_col):
        if len(g) < k:
            continue
        top_pred = set(g.nlargest(k, "_rank")["player_id"])
        best_actual = g.loc[g[actual_col].idxmax(), "player_id"]
        hits += int(best_actual in top_pred)
        total += 1
    return float(hits / total) if total else float("nan")


def captaincy_points_captured(df: pd.DataFrame, pred_col: str = "xp_mean",
                              actual_col: str = "actual",
                              gw_col: str = "gw",
                              risk_lambda: float = 0.0) -> float:
    """Points your captain would have scored, as a fraction of the best
    possible. More informative than a hit rate: getting the second-best
    captain is nearly as good as the best, and this says so.

    At risk_lambda=0.0, ranks by raw xP and picks defenders. At nonzero
    risk_lambda, ranks by effective value and asks the question a manager
    actually faces: who is most likely to be the top scorer, not who has
    the highest floor.
    """
    if df.empty or gw_col not in df.columns:
        return float("nan")
    ranked = df.copy()
    ranked["_rank"] = _rank_column(ranked, risk_lambda, pred_col)
    got, best = 0.0, 0.0
    for _, g in ranked.groupby(gw_col):
        if g.empty:
            continue
        pick = g.loc[g["_rank"].idxmax()]
        got += float(pick[actual_col])
        best += float(g[actual_col].max())
    return float(got / best) if best else float("nan")


def crps_from_quantiles(quantile_grid: list[float], actual: float) -> float:
    """Continuous Ranked Probability Score from a stored quantile grid."""
    q = np.asarray(quantile_grid, dtype=float)
    n = len(q)
    if n < 2:
        return abs(float(q[0]) - actual) if n else float("nan")
    levels = np.linspace(0, 1, n)
    diff = actual - q
    pinball = np.where(diff >= 0, levels * diff, (levels - 1) * diff)
    return float(2.0 * pinball.mean())


def pit_values(quantile_grids: list[list[float]], actuals: np.ndarray,
               seed: int = 0) -> np.ndarray:
    """Probability Integral Transform: F(actual) under each prediction."""
    rng = np.random.default_rng(seed)
    out = []
    for grid, a in zip(quantile_grids, actuals):
        q = np.sort(np.asarray(grid, dtype=float))
        n = len(q)
        levels = np.linspace(0, 1, n)
        left = int(np.searchsorted(q, a, side="left"))
        right = int(np.searchsorted(q, a, side="right"))
        if right - left <= 1:
            out.append(float(np.interp(a, q, levels)))
            continue
        lo = levels[left - 1] if left > 0 else 0.0
        hi = levels[right - 1] if right < n else 1.0
        out.append(float(lo + rng.random() * (hi - lo)))
    return np.array(out)


def pit_uniformity(pit: np.ndarray, bins: int = 10) -> tuple[float, list[float]]:
    """KS statistic against uniform, plus the histogram for display."""
    if len(pit) < 5:
        return float("nan"), []
    ks = float(stats.kstest(pit, "uniform").statistic)
    hist, _ = np.histogram(pit, bins=bins, range=(0, 1))
    return ks, [float(h / len(pit)) for h in hist]


def calibration_curve(probs: np.ndarray, outcomes: np.ndarray,
                      n_bins: int = 10) -> list[dict]:
    """Real calibration, from real predictions and real outcomes."""
    probs, outcomes = np.asarray(probs, float), np.asarray(outcomes, float)
    if probs.size == 0:
        return []
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "predicted_rate": round(float(probs[mask].mean()), 4),
            "actual_rate": round(float(outcomes[mask].mean()), 4),
            "n": int(mask.sum()),
        })
    return rows


def expected_calibration_error(curve: list[dict]) -> float:
    """Sample-weighted mean gap between predicted and observed rates."""
    if not curve:
        return float("nan")
    total = sum(b["n"] for b in curve)
    if total == 0:
        return float("nan")
    return float(sum(b["n"] * abs(b["predicted_rate"] - b["actual_rate"])
                     for b in curve) / total)