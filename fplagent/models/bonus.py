"""Bonus Points System model.

THE BUG THIS FIXES (v5 audit, item 1.1)
---------------------------------------
v5 mapped expected BPS to expected bonus with:

    THRESHOLDS = [(50, 0.0), (30, 0.3), (20, 0.7), (10, 1.4), (3, 2.5)]
    for rank, expected in THRESHOLDS:
        if bps >= rank: return expected

First match wins, so BPS=100 returned 0.0 bonus and BPS=5 returned 2.5. The
mapping was monotonically backwards: every elite attacker got zero expected
bonus and every fringe player got the maximum. The variable was named
`percentile_rank` but compared against a raw BPS score, which is how it went
unnoticed.

The deeper problem is that ANY absolute BPS -> bonus table is wrong in
principle. Bonus is awarded by RANK within a fixture: 3 points to the highest
BPS on the pitch, 2 to the second, 1 to the third. Whether 45 BPS wins bonus
depends entirely on what the other 21 players did. So v6 never uses a table
in the main path -- it ranks simulated BPS within the simulated fixture, which
the joint Monte Carlo makes natural.

2026/27 BPS CHANGES applied here:
  * Players are no longer penalised for being tackled (the old -1).
  * Clearances, blocks and interceptions now yield 1 BPS per 3 actions,
    previously per 2. This is a BPS action total, not the broader DefCon
    CBIT/CBIRT total.
  * Goalkeeper calculations changed, with extra credit for saves from big
    chances.
"""
from __future__ import annotations

import numpy as np

# BPS values for the actions this pipeline can observe or simulate.
BPS_APPEARANCE_60 = 6
BPS_APPEARANCE_SUB = 3
BPS_GOAL = {"GK": 12, "DEF": 12, "MID": 18, "FWD": 24}
BPS_ASSIST = 9
BPS_PENALTY_GOAL = 12        # position-neutral in 2026/27
BPS_CLEAN_SHEET = {"GK": 12, "DEF": 12, "MID": 0, "FWD": 0}
BPS_SAVE = 2                      # 2 BPS per save
BPS_BIG_CHANCE_SAVE_BONUS = 1     # additional, 2026/27
BPS_PENALTY_SAVE = 7
BPS_PENALTY_MISS = -6
BPS_OWN_GOAL = -6
BPS_YELLOW = -3
BPS_RED = -9
# -4 BPS for EVERY goal conceded (GK/DEF). Not to be confused with the
# fantasy-points rule of -1 per 2 conceded, which is what this constant used
# to encode: at -1 per 2, a defender shipping four still outranked most
# attackers on BPS and collected bonus he could never have won.
BPS_GOAL_CONCEDED = -4
# 2026/27: 1 BPS per 3 clearances/blocks/interceptions (was per 2).
BPS_CBI_PER = 3
BPS_RECOVERY_PER = 3
BPS_TACKLED_PENALTY = 0           # 2026/27: removed


def simulate_bps(position: str, minutes: np.ndarray, goals: np.ndarray,
                 assists: np.ndarray, goals_conceded: np.ndarray,
                 saves: np.ndarray, defensive_actions: np.ndarray | None = None,
                 baseline_bps_per90: float = 0.0,
                 penalties_saved: np.ndarray | None = None,
                 penalties_missed: np.ndarray | None = None,
                 own_goals: np.ndarray | None = None,
                 yellow_cards: np.ndarray | None = None,
                 red_cards: np.ndarray | None = None,
                 inside_box_saves: np.ndarray | None = None,
                 big_chance_saves: np.ndarray | None = None,
                 penalty_goals: np.ndarray | None = None,
                 cbi_actions: np.ndarray | None = None) -> np.ndarray:
    """Vectorised BPS for one player across all simulation draws.

    v6 defined BPS_PENALTY_SAVE, BPS_PENALTY_MISS, BPS_OWN_GOAL, BPS_YELLOW
    and BPS_RED and then applied none of them, so a simulated penalty miss
    cost the player two points but left his bonus rank untouched. Since bonus
    is a rank across twenty-two players, an event that should sink someone
    down the order was invisible to it. All five now feed through.

    `inside_box_saves` and `big_chance_saves` are optional because the current
    pipeline may not have event-level save-location metadata. We do not invent
    those labels; callers can pass them when the source supplies them.
    """
    played = minutes > 0
    bps = np.where(minutes >= 60, BPS_APPEARANCE_60,
                   np.where(played, BPS_APPEARANCE_SUB, 0)).astype(float)
    # Penalty goals are worth 12 BPS regardless of playing position in the
    # current rules; open-play goals keep their positional value.
    goals_arr = np.asarray(goals)
    if penalty_goals is None:
        pen_goals = np.zeros_like(goals_arr, dtype=np.int32)
    else:
        pen_goals = np.clip(np.asarray(penalty_goals), 0, goals_arr)
    open_play_goals = np.maximum(goals_arr - pen_goals, 0)
    bps += open_play_goals * BPS_GOAL.get(position, 12)
    bps += pen_goals * BPS_PENALTY_GOAL
    bps += assists * BPS_ASSIST
    cs = (goals_conceded == 0) & (minutes >= 60)
    bps += cs * BPS_CLEAN_SHEET.get(position, 0)
    if position == "GK":
        bps += saves * BPS_SAVE
        if inside_box_saves is not None:
            bps += np.maximum(np.asarray(inside_box_saves), 0) * 1.0
        if big_chance_saves is not None:
            bps += np.maximum(np.asarray(big_chance_saves), 0) * BPS_BIG_CHANCE_SAVE_BONUS
        if penalties_saved is not None:
            bps += np.asarray(penalties_saved) * BPS_PENALTY_SAVE
    if position in ("GK", "DEF"):
        bps += goals_conceded * BPS_GOAL_CONCEDED
    # 2026/27 BPS awards 1 point per 3 clearances/blocks/interceptions.
    # Do NOT pass the DefCon CBIT/CBIRT total here: that would incorrectly give
    # BPS credit for tackles and/or recoveries. The simulation therefore leaves
    # this optional when no event-level CBI model exists, and the player's
    # historical baseline absorbs the unmodelled contribution.
    if cbi_actions is not None:
        bps += np.maximum(np.asarray(cbi_actions), 0) // BPS_CBI_PER
    if penalties_missed is not None:
        bps += np.asarray(penalties_missed) * BPS_PENALTY_MISS
    if own_goals is not None:
        bps += np.asarray(own_goals) * BPS_OWN_GOAL
    if yellow_cards is not None:
        bps += np.asarray(yellow_cards) * BPS_YELLOW
    if red_cards is not None:
        bps += np.asarray(red_cards) * BPS_RED
    # Everything this pipeline cannot observe per-simulation (passing volume,
    # key passes, dribbles, big chances created) enters as a per-90 baseline
    # estimated from the player's own BPS history net of the accounted terms.
    bps += baseline_bps_per90 * (minutes / 90.0)
    return np.where(played, bps, 0.0)


def bonus_from_ranks(bps_matrix: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Award 3/2/1 bonus by rank within each simulated fixture.

    bps_matrix: (n_players, n_sims) simulated BPS for every player in ONE
    fixture, both teams together. Returns the same shape filled with 0/1/2/3.

    Rank is computed as 1 + (number of players with strictly greater BPS),
    which gives FPL's actual tie behaviour for free: tied players all receive
    the higher award, so a two-way tie at the top yields 3 points each and the
    next player down receives 1. A naive argsort breaks ties arbitrarily and
    biases every defender's expected bonus downward, because defenders tie on
    BPS far more often than attackers do.

    Fully vectorised, chunked over simulations to bound peak memory.
    """
    n_players, n_sims = bps_matrix.shape
    bonus = np.zeros((n_players, n_sims), dtype=np.float32)
    if n_players == 0:
        return bonus

    award = np.array([0.0, 3.0, 2.0, 1.0], dtype=np.float32)
    for lo in range(0, n_sims, chunk):
        hi = min(lo + chunk, n_sims)
        block = bps_matrix[:, lo:hi].astype(np.float32)
        # strictly-greater count, broadcast over the player axis
        greater = (block[None, :, :] > block[:, None, :]).sum(axis=1)
        rank = greater + 1
        got = np.where((rank <= 3) & (block > 0), award[np.clip(rank, 0, 3)], 0.0)
        bonus[:, lo:hi] = got
    return bonus


def estimate_baseline_bps_per90(bps_history: list[float], minutes_history: list[int],
                                accounted_history: list[float] | None = None,
                                prior: float = 8.0, shrinkage_games: float = 5.0) -> float:
    """The part of a player's BPS that this pipeline cannot simulate directly:
    passing volume, key passes, dribbles, big chances created.

    Estimated as (observed BPS - simulable BPS) per 90, shrunk to a positional
    prior. This is what stops a deep-lying playmaker's bonus potential from
    being invisible just because he rarely scores.
    """
    if not bps_history or not minutes_history:
        return prior
    mins = np.array(minutes_history, dtype=float)
    obs = np.array(bps_history, dtype=float)
    acc = np.array(accounted_history, dtype=float) if accounted_history else np.zeros_like(obs)
    played = mins > 0
    if not played.any():
        return prior
    games = float(mins[played].sum() / 90.0)
    residual = float((obs[played] - acc[played]).sum() / max(games, 1e-6))
    w = games / (games + shrinkage_games)
    return float(max(w * residual + (1 - w) * prior, 0.0))
