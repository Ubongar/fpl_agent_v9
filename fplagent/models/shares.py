"""Player attacking shares: what fraction of his team's output belongs to him.

WHAT CHANGED IN v7
------------------
v6's headline claim was that penalties were separated from open play. They
were not. `estimate_shares` has an argument named `player_np_goals`, and
`features/build.py` passed it `h["goals_scored"]`, which is TOTAL goals with
penalties included. FPL does not publish a penalty/open-play split, so the
"np" prefix had never once been true.

The damage compounded downstream. `load_set_piece_duties` gave the declared
taker a penalty share of 1.0, and the simulator then split the scoreline into
open-play and penalty goals and allocated the penalty goals a SECOND time
using that share. Measured on a forward with 10 of his team's 20 goals, four
of them penalties: true open-play share 0.30, pipeline share 0.41, plus full
penalty allocation on top. The error landed hardest on exactly the players
who dominate captaincy decisions.

v7 fixes it at the source. `estimate_penalties_scored` in `setpieces.py`
reconstructs each player's penalty goals from declared duty and the team's
penalty events, non-penalty goals are computed at ingest, and this module
receives genuine open-play numbers.

WHAT CHANGED IN v9
------------------
1. A SINGLE HIGH-SCORING FIXTURE NO LONGER CONTRIBUTES ITS FULL WEIGHT.
   The robust fix is to cap each fixture's contribution to the numerator at
   MAX_GOALS_PER_FIXTURE_FOR_SHARE (2.0). This is a standard winsorising step
   for heavy-tailed count data.
2. THE FUNCTION IS METRIC-AGNOSTIC: GOALS OR xG.

THREE OTHER CORRECTIONS
-----------------------
1. ASSISTS ARE NO LONGER DIVIDED BY GOALS.
2. SHARES ARE RESOLVED AGAINST THE PITCH, NOT JUST CLIPPED.
3. THE PENALTY POSTERIOR IS NO LONGER THE PRIMARY SOURCE.

WHAT CHANGED IN v10
-------------------
External API Blending: The function now accepts per-90 baselines from
external APIs (like StrikerAPI) and blends them with the player's historical
data. This prevents the model from overreacting to short-term FPL luck when
underlying metrics (xG/xA) suggest a different true talent level.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fplagent.config import PARAMS

# Share of a team's non-penalty goals / assists typically taken by a player in
# each position, given that he is a regular starter.
GOAL_SHARE_PRIOR = {"GK": 0.002, "DEF": 0.055, "MID": 0.115, "FWD": 0.230}
ASSIST_SHARE_PRIOR = {"GK": 0.004, "DEF": 0.085, "MID": 0.150, "FWD": 0.105}

# Fraction of goals that carry an assist. Defined here and imported by the
# simulator so the share denominator and the sampler cannot drift apart.
ASSISTED_RATE = 0.75

# Kept for callers that want a league-average penalty rate. The simulator no
# longer uses it: penalties are drawn as events and attributed by duty.
PENALTY_GOAL_RATE = 0.105
PENALTY_CONVERSION = 0.79

# Robust cap on the per-fixture numerator.
MAX_GOALS_PER_FIXTURE_FOR_SHARE = 2.0


@dataclass
class ShareEstimate:
    np_goal_share: float
    assist_share: float
    penalty_share: float          # probability of taking a penalty, given he is on
    minutes_equiv: float
    confidence: str               # prior_only | translated | shrunk | data_driven
    provenance: dict = field(default_factory=dict)


def estimate_shares(position: str,
                    player_np_goals: list[float],
                    player_assists: list[float],
                    team_np_goals: list[float],
                    team_assists: list[float] | None,
                    minutes_played: list[int],
                    declared_penalty_share: float | None = None,
                    player_pens_taken: list[float] | None = None,
                    team_pens: list[float] | None = None,
                    prior_override: dict | None = None,
                    metric: str = "goals",
                    player_expected_assists: list[float] | None = None,
                    team_expected_assists: list[float] | None = None,
                    xa_blend_weight: float = 0.65,
                    baseline_npxg90: float = 0.0,
                    baseline_xa90: float = 0.0,
                    blend_weight: float = 0.5) -> ShareEstimate:
    """All lists aligned, one entry per fixture the player's team played."""

    prior_g = (prior_override or {}).get("np_goal_share", GOAL_SHARE_PRIOR.get(position, 0.08))
    prior_a = (prior_override or {}).get("assist_share", ASSIST_SHARE_PRIOR.get(position, 0.10))
    conf_seed = (prior_override or {}).get("confidence", "prior_only")

    if not minutes_played:
        return ShareEstimate(prior_g, prior_a, declared_penalty_share or 0.0,
                             0.0, conf_seed, {"reason": "no_history", "metric": metric})

    mins = np.array(minutes_played, dtype=float)
    on_pitch = np.clip(mins / 90.0, 0.0, 1.0)
    minutes_equiv = float(on_pitch.sum())

    team_np = np.array(team_np_goals, dtype=float)
    denom_goals = float((team_np * on_pitch).sum())

    if team_assists:
        team_as = np.array(team_assists, dtype=float)
        denom_assists = float((team_as * on_pitch).sum())
    else:
        denom_assists = denom_goals * ASSISTED_RATE

    player_np_raw = np.array(player_np_goals, dtype=float)
    player_np_capped = np.minimum(player_np_raw, MAX_GOALS_PER_FIXTURE_FOR_SHARE)
    player_np = float(player_np_capped.sum())

    player_as_raw = np.array(player_assists, dtype=float)
    player_as = float(np.minimum(player_as_raw, MAX_GOALS_PER_FIXTURE_FOR_SHARE).sum())

    # --- EXTERNAL API BLENDING (v10) ---
    api_blend_used = 0.0
    if baseline_npxg90 > 0 and minutes_equiv > 0:
        api_expected_goals = baseline_npxg90 * minutes_equiv
        player_np = (player_np * (1.0 - blend_weight)) + (api_expected_goals * blend_weight)
        api_blend_used = blend_weight

    if baseline_xa90 > 0 and minutes_equiv > 0:
        api_expected_assists = baseline_xa90 * minutes_equiv
        player_as = (player_as * (1.0 - blend_weight)) + (api_expected_assists * blend_weight)
    # -----------------------------------

    k = PARAMS.share_shrinkage_games
    w = minutes_equiv / (minutes_equiv + k)

    if denom_goals <= 0:
        np_goal_share = prior_g
        conf = "prior_only" if minutes_equiv < 1 else "shrunk"
    else:
        np_goal_share = w * (player_np / denom_goals) + (1 - w) * prior_g
        conf = "data_driven" if minutes_equiv >= 2 * k else "shrunk"

    if denom_assists <= 0:
        observed_assist_share = prior_a
    else:
        observed_assist_share = w * (player_as / denom_assists) + (1 - w) * prior_a

    assist_share = observed_assist_share
    xa_share = None
    xa_weight_used = 0.0
    if player_expected_assists is not None and team_expected_assists is not None:
        player_xa = np.asarray(player_expected_assists, dtype=float)
        team_xa = np.asarray(team_expected_assists, dtype=float)
        if (len(player_xa) == len(mins) and len(team_xa) == len(mins)
                and np.isfinite(player_xa).any() and np.isfinite(team_xa).any()):
            player_xa = np.nan_to_num(player_xa, nan=0.0, posinf=0.0, neginf=0.0)
            team_xa = np.nan_to_num(team_xa, nan=0.0, posinf=0.0, neginf=0.0)
            denom_xa = float((team_xa * on_pitch).sum())
            if denom_xa > 1e-8:
                xa_share = w * (float(player_xa.sum()) / denom_xa) + (1 - w) * prior_a
                xa_weight_used = float(np.clip(xa_blend_weight, 0.0, 1.0))
                assist_share = ((1.0 - xa_weight_used) * observed_assist_share
                                + xa_weight_used * xa_share)

    observed_pen_share = None
    if player_pens_taken is not None and team_pens is not None:
        pens_team = float(np.array(team_pens, dtype=float).sum())
        pens_player = float(np.array(player_pens_taken, dtype=float).sum())
        if pens_team > 0:
            observed_pen_share = (pens_player + 1.0) / (pens_team + 4.0)

    penalty_share = (declared_penalty_share if declared_penalty_share is not None
                     else (observed_pen_share or 0.0))

    prov = {
        "metric": metric,
        "shrinkage_weight": round(w, 3),
        "denominator_goals": round(denom_goals, 2),
        "denominator_assists": round(denom_assists, 2),
        "raw_player_goals": round(float(player_np_raw.sum()), 2),
        "capped_player_goals": round(player_np, 2),
        "per_fixture_cap": MAX_GOALS_PER_FIXTURE_FOR_SHARE,
        "observed_assist_share": round(float(observed_assist_share), 4),
        "xa_share": None if xa_share is None else round(float(xa_share), 4),
        "xa_blend_weight_used": round(float(xa_weight_used), 3),
        "penalty_source": ("declared" if declared_penalty_share is not None
                           else ("observed" if observed_pen_share else "none")),
        "api_npxg90": baseline_npxg90,
        "api_xa90": baseline_xa90,
        "api_blend_weight_used": round(api_blend_used, 3),
    }
    if declared_penalty_share is not None and observed_pen_share is not None:
        prov["observed_penalty_share"] = round(observed_pen_share, 3)
        if abs(observed_pen_share - declared_penalty_share) > 0.35:
            prov["duty_disagreement"] = True

    return ShareEstimate(
        np_goal_share=float(np.clip(np_goal_share, 0.0, 0.85)),
        assist_share=float(np.clip(assist_share, 0.0, 0.85)),
        penalty_share=float(np.clip(penalty_share, 0.0, 1.0)),
        minutes_equiv=minutes_equiv,
        confidence=conf,
        provenance=prov,
    )


def normalise_shares(shares: np.ndarray, cap: float = 0.85) -> np.ndarray:
    """Scale a team's shares so they sum to at most 1."""
    shares = np.clip(np.asarray(shares, dtype=float), 0.0, cap)
    total = shares.sum()
    if total <= 1.0:
        return shares
    return shares / total


def resolve_on_pitch_shares(shares: np.ndarray, on_pitch: np.ndarray,
                            absorb_remainder: bool = False) -> np.ndarray:
    """Per-draw shares over the players actually on the pitch."""
    shares = np.asarray(shares, dtype=float)[:, None]
    avail = np.asarray(on_pitch, dtype=float) * shares
    col = avail.sum(axis=0)
    if absorb_remainder:
        out = np.divide(avail, np.maximum(col, 1e-12),
                        out=np.zeros_like(avail), where=col > 1e-12)
        zero = col <= 1e-12
        if zero.any():
            on = np.asarray(on_pitch, dtype=bool)
            counts = on[:, zero].sum(axis=0)
            uniform = np.divide(on[:, zero].astype(float),
                                np.maximum(counts[None, :], 1),
                                out=np.zeros((on.shape[0], int(zero.sum()))),
                                where=counts[None, :] > 0)
            out[:, zero] = uniform
        return out
    over = col > 1.0
    if over.any():
        avail[:, over] = avail[:, over] / col[over]
    return avail