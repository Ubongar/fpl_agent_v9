"""Defensive contribution modelling.

THE MODELLING POINT (v5 audit, section 3)
-----------------------------------------
DefCon pays a flat 2 points for crossing a threshold, capped at 2 per match.
That cap changes the problem completely. The quantity you need is

    P(actions >= threshold)

NOT E[actions] scaled by some rate. A centre-back averaging 18 CBIT per match
and one averaging 10 have very different point expectations from a rate model
(3.6 vs 2.0) and nearly identical ones in reality (both ~2.0, because both
clear the bar almost every week). A rate model systematically overvalues
high-volume defenders -- exactly the players DefCon was introduced to reward,
so the error lands precisely where it hurts most.

Actions per 90 are modelled as Negative Binomial rather than Poisson because
real defensive-action counts are overdispersed: they depend on how much of the
game a team spends without the ball, which varies hugely by fixture. Fitting
a dispersion parameter from the player's own history, shrunk toward a
positional prior, captures that.

The fixture adjustment matters too: a defender facing a top side will face
more of the ball and record more actions. We scale the player's base rate by
the opponent's expected goals, which is already fitted in team_strength.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom

from fplagent.config import PARAMS
from fplagent.models.scoring import DEFCON_THRESHOLD

# Positional priors for defensive actions per 90, from published 2025/26
# distributions. Used when a player has little or no history.
# NOTE: these are per-90 rates of the stat that COUNTS for that position:
# CBIT for defenders (no recoveries), CBIRT for midfielders and forwards
# (recoveries included). Feeding a defender a CBIRT count against a CBIT
# threshold, as v6 did, inflates every defender's DefCon probability, which
# is 2 points a week on the most heavily owned position group.
DEFCON_PRIOR_PER90 = {"GK": 0.5, "DEF": 9.6, "MID": 7.0, "FWD": 2.5}

# Which stored column each position's threshold is measured against.
DEFCON_STAT = {"DEF": "cbit", "MID": "cbirt", "FWD": "cbirt"}
# Variance-to-mean ratio prior. >1 == overdispersed relative to Poisson.
DEFCON_DISPERSION_PRIOR = 1.8


@dataclass
class DefconRate:
    per90: float
    dispersion: float
    games_used: float
    confidence: str


def estimate_defcon_rate(position: str,
                         actions_history: list[float],
                         minutes_history: list[int]) -> DefconRate:
    """Shrunk per-90 defensive-action rate for one player."""
    prior = DEFCON_PRIOR_PER90.get(position, 5.0)
    if not actions_history or not minutes_history:
        return DefconRate(prior, DEFCON_DISPERSION_PRIOR, 0.0, "prior_only")

    mins = np.array(minutes_history, dtype=float)
    acts = np.array(actions_history, dtype=float)
    played = mins > 0
    if not played.any():
        return DefconRate(prior, DEFCON_DISPERSION_PRIOR, 0.0, "prior_only")

    games_equiv = float(mins[played].sum() / 90.0)
    raw_per90 = float(acts[played].sum() / max(mins[played].sum() / 90.0, 1e-6))

    k = PARAMS.defcon_shrinkage_games
    w = games_equiv / (games_equiv + k)
    per90 = w * raw_per90 + (1 - w) * prior

    # Empirical dispersion from COUNTS, not from per-90 rates.
    # v6 computed the variance-to-mean ratio of rates. Dividing a small count
    # by a small minutes fraction inflates its variance enormously, so a
    # player with a few cameo appearances looked wildly overdispersed when he
    # was only sparsely observed. Counts are what the negative binomial
    # actually models, so the dispersion has to be measured on them.
    if played.sum() >= 3:
        exposure = np.maximum(mins[played] / 90.0, 1e-6)
        expected = per90 * exposure
        resid = acts[played] - expected
        mean_count = max(float(expected.mean()), 1e-6)
        vmr = float(max((resid ** 2).mean() / mean_count, 1.0))
        dispersion = w * vmr + (1 - w) * DEFCON_DISPERSION_PRIOR
    else:
        dispersion = DEFCON_DISPERSION_PRIOR

    conf = "data_driven" if games_equiv >= 2 * k else ("shrunk" if games_equiv > 0 else "prior_only")
    return DefconRate(float(per90), float(max(dispersion, 1.001)), games_equiv, conf)


def _nbinom_params(mean: float, dispersion: float) -> tuple[float, float]:
    """Convert (mean, variance-to-mean ratio) to scipy's (n, p)."""
    mean = max(mean, 1e-6)
    var = mean * max(dispersion, 1.001)
    p = mean / var
    n = mean * p / max(1 - p, 1e-9)
    return max(n, 1e-6), float(np.clip(p, 1e-6, 1 - 1e-9))


def defcon_probability(position: str, rate: DefconRate, expected_minutes: float,
                       opponent_xg: float = 1.4, league_avg_xg: float = 1.4) -> float:
    """P(player reaches his positional DefCon threshold this match).

    Scaled two ways: linearly by minutes on the pitch, and by how much of the
    ball the opponent is expected to have (proxied by opponent xG relative to
    league average, damped to 0.35 because possession share moves defensive
    volume less than proportionally).
    """
    threshold = DEFCON_THRESHOLD.get(position)
    if threshold is None or expected_minutes <= 0:
        return 0.0

    fixture_mult = float(np.clip((opponent_xg / max(league_avg_xg, 1e-6)) ** 0.35, 0.7, 1.45))
    expected_actions = rate.per90 * (expected_minutes / 90.0) * fixture_mult
    n, p = _nbinom_params(expected_actions, rate.dispersion)
    return float(nbinom.sf(threshold - 1, n, p))


def sample_defcon_actions(position: str, rate: DefconRate, minutes: np.ndarray,
                          opponent_xg: float, league_avg_xg: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Draw per-simulation action counts, so DefCon points correlate correctly
    with minutes played inside the joint match simulation."""
    if position not in DEFCON_THRESHOLD:
        return np.zeros_like(minutes, dtype=int)
    fixture_mult = float(np.clip((opponent_xg / max(league_avg_xg, 1e-6)) ** 0.35, 0.7, 1.45))
    mean = np.maximum(rate.per90 * (minutes / 90.0) * fixture_mult, 1e-6)
    var = mean * rate.dispersion
    p = mean / var
    n = mean * p / np.maximum(1 - p, 1e-9)
    return rng.negative_binomial(np.maximum(n, 1e-6), np.clip(p, 1e-6, 1 - 1e-9))
