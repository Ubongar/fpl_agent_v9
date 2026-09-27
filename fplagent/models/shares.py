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

   The shrinkage weight w = minutes / (minutes + k) treats the player's TOTAL
   non-penalty goals as if each goal were equally informative. That is wrong
   when a player's season is concentrated in one game: a forward who scored
   3-4 in one match and nothing in the surrounding four has a small sample
   whose entire signal is a single observation.

   The robust fix is to cap each fixture's contribution to the numerator at
   MAX_GOALS_PER_FIXTURE_FOR_SHARE (2.0). This is a standard winsorising step
   for heavy-tailed count data and is defensible on its own terms -- one
   game cannot outweigh five -- but the effect is material here: measured on
   an early-season forward whose 4-goal haul against a strong side otherwise
   dominated his share estimate, capping at 2 brings the blended share from
   ~0.40 down to ~0.30, closer to what a mid-table striker's base rate
   supports. The cap is applied to the NUMERATOR only. The denominator (team
   non-penalty goals over the same fixtures) is left un-capped because a team
   scoring 4 in one match is not an outlier the way an individual 4-goal haul
   is; team goals are far less concentrated.

2. THE FUNCTION IS METRIC-AGNOSTIC: GOALS OR xG.

   `metric` names what the caller is passing in the "goals" arguments.
   When set to "xg", the same ratio arithmetic is applied to expected goals
   instead of actual goals. xG is a less noisy estimator of a player's
   underlying attacking involvement -- a striker whose two goals in five
   games came from 0.15 xG per match is a worse finisher than a striker
   whose one goal came from 0.8 xG -- and using it removes the finishing
   luck component from the share estimate. It is a strict improvement for
   everyone whose goal counts misrepresent their underlying quality, but
   it does NOT help a player whose one big game was legitimately high xG
   (a genuine hat-trick of good chances): for that case, only the
   per-fixture cap above is effective. The two fixes complement each other.

   Priors are unchanged between metrics: over a large sample, the
   distribution of goal share and xG share at a given position are close
   enough that the same GOAL_SHARE_PRIOR values are a defensible prior for
   both. The alternative -- calibrating separate xG priors -- would require
   a season of data we do not yet have.

THREE OTHER CORRECTIONS
-----------------------
1. ASSISTS ARE NO LONGER DIVIDED BY GOALS.
   v6 computed `raw_a = player_assists / denom_goals`. Only about 75% of
   goals are assisted, so the resulting share sat on a different scale from
   the one the simulator consumes and was biased low by roughly a quarter.

2. SHARES ARE RESOLVED AGAINST THE PITCH, NOT JUST CLIPPED.
   `normalise_shares` only scaled down when the total exceeded one. The
   simulator now renormalises per simulation draw against the eleven actually
   playing, which is the only place the question can be answered.

3. THE PENALTY POSTERIOR IS NO LONGER THE PRIMARY SOURCE.
   v6 inferred duty from observed penalties. A team awarded no penalties had
   `pens_team == 0`, the posterior returned 0.0, and the model concluded
   nobody at that club takes them. Declared duty from `bootstrap-static` is
   now primary (see `setpieces.py`); the observational posterior is kept as a
   cross-check that can flag a duty changing hands.
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

# Robust cap on the per-fixture numerator. A player who scores 3 in one game
# is genuinely better than one who scores 2, but the difference should not be
# able to outweigh an entire season of counter-evidence. Two goals is a
# natural breakpoint: it is roughly the 99th percentile of individual
# per-fixture goal counts in the Premier League, so anything above it is by
# definition an outlier observation.
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
                    metric: str = "goals") -> ShareEstimate:
    """All lists aligned, one entry per fixture the player's team played.

    `player_np_goals` must be NON-PENALTY events. The caller is responsible
    for subtracting estimated penalty contributions (goals or xG depending
    on `metric`); passing raw `goals_scored` here is the v6 bug.

    `metric` names what the numbers are. The arithmetic is identical either
    way; this only affects logging and the emitted provenance, so a reader
    can tell whether a share was computed from goals or xG. See the module
    docstring for why xG is the better numerator.

    The denominator is the team's output WHILE THE PLAYER WAS ON THE PITCH,
    prorated by minutes, so a share reflects a player's rate when playing
    rather than one diluted by matches he sat out.

    Individual fixtures are capped at MAX_GOALS_PER_FIXTURE_FOR_SHARE in the
    numerator. See the module docstring for why.
    """
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

    # Assists get their own denominator. If the caller has real team assist
    # counts, use them; otherwise fall back to goals scaled by the assisted
    # rate, which is the correct conversion between the two scales.
    if team_assists:
        team_as = np.array(team_assists, dtype=float)
        denom_assists = float((team_as * on_pitch).sum())
    else:
        denom_assists = denom_goals * ASSISTED_RATE

    # Robust numerator: cap each individual fixture's contribution. The sum
    # is what feeds the share ratio, so capping one very high game prevents
    # it from dominating every other observation the player has.
    player_np_raw = np.array(player_np_goals, dtype=float)
    player_np_capped = np.minimum(player_np_raw, MAX_GOALS_PER_FIXTURE_FOR_SHARE)
    player_np = float(player_np_capped.sum())

    # Assists are less heavy-tailed than goals (a "hat-trick of assists" is
    # much rarer than a goal hat-trick) but the same principle applies; cap
    # at 2 to be consistent.
    player_as_raw = np.array(player_assists, dtype=float)
    player_as = float(np.minimum(player_as_raw, MAX_GOALS_PER_FIXTURE_FOR_SHARE).sum())

    k = PARAMS.share_shrinkage_games
    w = minutes_equiv / (minutes_equiv + k)

    if denom_goals <= 0:
        np_goal_share = prior_g
        conf = "prior_only" if minutes_equiv < 1 else "shrunk"
    else:
        np_goal_share = w * (player_np / denom_goals) + (1 - w) * prior_g
        conf = "data_driven" if minutes_equiv >= 2 * k else "shrunk"

    if denom_assists <= 0:
        assist_share = prior_a
    else:
        assist_share = w * (player_as / denom_assists) + (1 - w) * prior_a

    # Declared duty wins. The observational posterior below exists only to
    # disagree with it out loud, never to override it silently.
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
        "penalty_source": ("declared" if declared_penalty_share is not None
                           else ("observed" if observed_pen_share else "none")),
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
    """Scale a team's shares so they sum to at most 1.

    Kept for the static case. The simulator prefers `resolve_on_pitch_shares`
    below, which does the same job per simulation draw against the players
    actually playing.
    """
    shares = np.clip(np.asarray(shares, dtype=float), 0.0, cap)
    total = shares.sum()
    if total <= 1.0:
        return shares
    return shares / total


def resolve_on_pitch_shares(shares: np.ndarray, on_pitch: np.ndarray,
                            absorb_remainder: bool = False) -> np.ndarray:
    """Per-draw shares over the players actually on the pitch.

    shares:   (n_players,) static share estimates.
    on_pitch: (n_players, n_sims) boolean.

    With `absorb_remainder=False` the columns sum to at most 1 and the
    shortfall means "a player outside our pool did it", which is correct for
    open-play goals: a sixteen-man view is not the whole squad, and own goals
    exist.

    With `absorb_remainder=True` the columns are renormalised to sum to 1
    among whoever is on. That is correct for a set piece whose taker is known
    to be on the pitch, and it is the mechanism that makes a backup penalty
    taker worth more when the primary is rested.
    """
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