"""Shared distribution arithmetic.

WHY THIS IS A MODULE AND NOT A HELPER
-------------------------------------
v6 had two different answers to "what is a double gameweek player's
distribution". `evaluation/backtest._convolve_grids` resampled from each
fixture's quantile grid and added the draws, which is correct. Meanwhile
`pipeline/run.latest_predictions`, the function that actually feeds the squad
optimiser, the captaincy ranking and the UI, aggregated `xp_p10` with `min`
and `xp_p90` with `max`.

Percentiles do not combine that way. Two fixtures each with p90 = 12 do not
give a gameweek p90 of 24, because both legs hauling at once is far rarer
than one leg hauling. The serving path was therefore reporting ceilings it
had never computed, on exactly the gameweeks where Triple Captain and Bench
Boost decisions get made.

One implementation, imported by both.
"""
from __future__ import annotations

import numpy as np

DEFAULT_GRID_POINTS = 21


def convolve_quantile_grids(grids: list, n_draws: int = 4000,
                            seed: int = 11,
                            points: int | None = None) -> list[float]:
    """Combine per-fixture quantile grids into one gameweek grid.

    Treats the fixtures as independent, which slightly overstates the spread
    (a manager who rests a player in the first leg is a little more likely to
    start him in the second), but is enormously closer than min/max.
    """
    grids = [np.asarray(g, dtype=float) for g in grids
             if g is not None and len(np.asarray(g)) > 1]
    if not grids:
        return []
    if len(grids) == 1:
        return [float(x) for x in grids[0]]

    rng = np.random.default_rng(seed)
    total = np.zeros(n_draws)
    for g in grids:
        levels = np.linspace(0, 1, len(g))
        total += np.interp(rng.random(n_draws), levels, g)
    n_points = points or len(grids[0])
    return [float(v) for v in np.percentile(total, np.linspace(0, 100, n_points))]


def quantile_from_grid(grid: list[float], q: float) -> float:
    """Read one quantile off a stored grid."""
    g = np.asarray(grid, dtype=float)
    if g.size == 0:
        return 0.0
    if g.size == 1:
        return float(g[0])
    levels = np.linspace(0, 1, len(g))
    return float(np.interp(np.clip(q, 0.0, 1.0), levels, g))


def pairwise_win_probability(grid_a: list[float], grid_b: list[float],
                             n_draws: int = 10000, seed: int = 7,
                             same_team: bool = False) -> dict:
    """P(player A outscores player B) this gameweek, from their two stored
    marginal quantile grids.

    HONESTY ABOUT WHAT THIS IS. Each grid is the marginal distribution from
    the joint per-match simulation in models/simulation.py -- correct on its
    own. Comparing two players by resampling their marginals independently
    is correct when they are in different fixtures (nothing links the two
    matches). When same_team=True the two players share a match outcome and
    are genuinely correlated (both do better when their team wins big), which
    this independent resampling does not capture -- the real joint draws are
    not persisted per player, only the aggregated grid is. The result is
    still informative but is flagged as approximate in that case, rather than
    silently presented as exact.
    """
    ga, gb = np.asarray(grid_a, dtype=float), np.asarray(grid_b, dtype=float)
    if ga.size < 2 or gb.size < 2:
        return {"p_a_wins": None, "p_b_wins": None, "p_tie": None,
                "note": "insufficient distribution data for one or both players"}
    rng = np.random.default_rng(seed)
    levels_a, levels_b = np.linspace(0, 1, len(ga)), np.linspace(0, 1, len(gb))
    draws_a = np.interp(rng.random(n_draws), levels_a, ga)
    draws_b = np.interp(rng.random(n_draws), levels_b, gb)
    a_wins = float(np.mean(draws_a > draws_b))
    b_wins = float(np.mean(draws_b > draws_a))
    tie = float(1.0 - a_wins - b_wins)
    result = {"p_a_wins": round(a_wins, 3), "p_b_wins": round(b_wins, 3),
              "p_tie": round(tie, 3),
              "mean_margin": round(float(np.mean(draws_a - draws_b)), 2)}
    if same_team:
        result["approximate"] = True
        result["note"] = ("Both players share a fixture; true correlation "
                          "between their scores is not captured by this "
                          "independent resampling, so treat this probability "
                          "as directional rather than exact.")
    return result


def prob_at_least_one(x) -> float:
    """P(at least one occurrence) across independent events: 1 - prod(1-p).

    A double-gameweek player has TWO chances to play, start or haul, not one.
    `max()` across the fixtures throws that away.
    """
    return float(1.0 - np.prod(1.0 - np.clip(np.asarray(x, dtype=float), 0.0, 1.0)))


def combine_standard_errors(ses) -> float:
    """Monte Carlo errors on independent fixtures add in quadrature."""
    s = np.asarray(ses, dtype=float)
    return float(np.sqrt(np.nansum(s ** 2)))
