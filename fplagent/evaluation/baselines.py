"""Baselines the model must beat, or it is decoration.

This is the single most important discipline in the whole repository and the
one most homegrown FPL models skip. Three baselines, all free:

1. FPL's OWN `ep_next`. The game publishes an expected-points estimate for
   every player in bootstrap-static. It costs nothing to fetch. Most
   hand-built FPL models lose to it, and the ones that do not only know it
   because they checked.
2. A 6-GAMEWEEK ROLLING MEAN. The dumbest possible persistence model.
3. PRICE. The market's own estimate, set by people with more information than
   you, updated continuously, requiring zero modelling.

Every evaluation run scores all three on exactly the same rows with exactly
the same metrics. If the model does not clear all three, the honest conclusion
is that the modelling has not paid for itself yet -- and knowing that early is
worth far more than a flattering MAE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fplagent.evaluation import metrics as M


def rolling_mean_baseline(history: pd.DataFrame, target_gw: int,
                          window: int = 6) -> pd.Series:
    past = history[history["gw"] < target_gw]
    if past.empty:
        return pd.Series(dtype=float)
    recent = (past.sort_values("gw").groupby("player_id")
                  .tail(window).groupby("player_id")["total_points"].mean())
    return recent


def price_baseline(players: pd.DataFrame) -> pd.Series:
    """Price as a direct predictor, rescaled to a points-like magnitude so the
    error metrics are comparable rather than absurd."""
    if players.empty:
        return pd.Series(dtype=float)
    p = players.set_index("player_id")["now_cost"].astype(float)
    return (p - p.min()) / max(p.max() - p.min(), 1e-6) * 6.0


def score_baseline(pred: pd.Series, actual: pd.Series,
                   frame: pd.DataFrame | None = None) -> dict:
    idx = pred.index.intersection(actual.index)
    if len(idx) < 3:
        return {"n": int(len(idx))}
    p, a = pred.loc[idx].values, actual.loc[idx].values
    out = {
        "n": int(len(idx)),
        "mae": round(M.mae(p, a), 4),
        "rmse": round(M.rmse(p, a), 4),
        "spearman": round(M.spearman(p, a), 4),
    }
    if frame is not None:
        f = frame.loc[frame["player_id"].isin(idx)].copy()
        f["pred"] = f["player_id"].map(pred)
        f["actual"] = f["player_id"].map(actual)
        out["top5_hit"] = round(M.top_k_hit_rate(f, 5, "pred", "actual"), 4)
    return out


def compare(model_scores: dict, baseline_scores: dict[str, dict],
            primary: str = "spearman") -> dict:
    """Did the model beat every baseline on the metric that matters?

    Defaults to Spearman rather than MAE, because rank is what transfer and
    captaincy decisions consume, and a low-variance constant predictor can win
    on MAE while being useless.
    """
    model_val = model_scores.get(primary)
    beat, detail = True, {}
    for name, s in baseline_scores.items():
        bval = s.get(primary)
        if model_val is None or bval is None:
            detail[name] = None
            continue
        won = model_val > bval if primary in ("spearman", "top5_hit") else model_val < bval
        detail[name] = {"baseline": bval, "model": model_val, "model_wins": bool(won)}
        beat = beat and won
    return {"primary_metric": primary, "model_beats_all_baselines": beat,
            "per_baseline": detail}


def bootstrap_comparison(df, baseline_scores: dict[str, dict],
                         pred_col: str = "xp_mean",
                         actual_col: str = "actual",
                         n_boot: int = 400, seed: int = 3) -> dict:
    """Block-bootstrap the model-minus-baseline Spearman gap, by gameweek.

    Rows inside one gameweek are not independent: they share fixtures, team
    strength estimates and a single realisation of the weekend. Resampling
    individual rows would therefore produce an interval far too narrow and
    declare victory on noise. Resampling whole GAMEWEEKS preserves the
    correlation structure, which is the whole point of a block bootstrap.
    """
    import numpy as np
    import pandas as pd

    if df is None or df.empty or "gw" not in df:
        return {"n_gameweeks": 0, "beats_all_significantly": False}

    baseline_cols = {"rolling_6gw": "baseline_rolling", "price": "baseline_price",
                     "fpl_ep_next": "ep_next"}
    usable = {name: col for name, col in baseline_cols.items()
              if col in df.columns and name in baseline_scores}
    gws = sorted(df["gw"].unique())
    if len(gws) < 2 or not usable:
        return {"n_gameweeks": len(gws), "beats_all_significantly": False,
                "note": "need at least two gameweeks and one scored baseline"}

    rng = np.random.default_rng(seed)
    by_gw = {g: d for g, d in df.groupby("gw")}
    out: dict = {"n_gameweeks": len(gws), "n_boot": n_boot, "per_baseline": {}}
    all_clear = True

    for name, col in usable.items():
        deltas = []
        for _ in range(n_boot):
            pick = rng.choice(gws, size=len(gws), replace=True)
            sample = pd.concat([by_gw[g] for g in pick], ignore_index=True)
            sub = sample[sample[col].notna()]
            if len(sub) < 10:
                continue
            m = M.spearman(sub[pred_col].values, sub[actual_col].values)
            b = M.spearman(sub[col].values, sub[actual_col].values)
            if not (np.isnan(m) or np.isnan(b)):
                deltas.append(m - b)
        if len(deltas) < 20:
            out["per_baseline"][name] = {"note": "insufficient resamples"}
            all_clear = False
            continue
        d = np.asarray(deltas)
        lo, hi = np.percentile(d, [2.5, 97.5])
        clear = bool(lo > 0)
        all_clear = all_clear and clear
        out["per_baseline"][name] = {
            "mean_spearman_gain": round(float(d.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "significant": clear,
        }

    out["beats_all_significantly"] = all_clear
    return out
