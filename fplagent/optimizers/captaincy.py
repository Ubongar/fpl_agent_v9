"""Captaincy selection.

THE BUG THIS FIXES (v5 audit, item 1.6)
---------------------------------------
v5 scored captains as

    (0.4*xp_mean + 0.4*p90 + 0.2*p10) * start_prob

but xp_mean ALREADY had start_prob folded into every one of its terms. Start
probability was therefore applied twice, systematically punishing exactly the
rotation-risk players where the discount is meant to be a single, careful
adjustment.

THE DEEPER POINT: for pure expected points, the captain choice is trivial,
because E[2X] = 2E[X]. Ceiling only matters when your objective is RANK
rather than total, and then what matters is not the ceiling in isolation but
the ceiling relative to what your rivals own. So this module ranks on
effective value (xp discounted by effective ownership), and reports ceiling,
floor and differential upside as decision context rather than smuggling them
into a single opaque weighted score.
"""
from __future__ import annotations

import numpy as np

from fplagent.optimizers.objective import effective_value


def rank_captains(candidates: list[dict], risk_lambda: float = 0.0,
                  top_n: int = 10) -> list[dict]:
    """candidates: {player_id, name, xp, p10, p90, p_haul_10plus, eo, p_play}

    No hidden weighting. The ranking metric is stated, and every number a
    human would want in order to disagree with it is returned alongside.
    """
    scored = []
    for c in candidates:
        ev = effective_value(c, risk_lambda)
        eo = float(c.get("eo", c.get("selected_by_percent", 0.0)) or 0.0)
        eo = eo / 100.0 if eo > 1.0 else eo
        xp = float(c.get("xp", 0.0))
        scored.append({
            **c,
            "captain_score": round(ev, 3),
            "expected_captain_points": round(2 * xp, 2),
            "ceiling_p90": c.get("p90"),
            "floor_p10": c.get("p10"),
            "p_haul": c.get("p_haul_10plus"),
            # Points that move you relative to the field: the fraction of
            # rivals who do NOT already own him, times his expected return.
            "differential_points": round(2 * xp * (1 - eo), 2),
            "effective_ownership": round(eo, 3),
            "risk_note": _risk_note(c),
        })
    scored.sort(key=lambda r: r["captain_score"], reverse=True)
    return scored[:top_n]


def _risk_note(c: dict) -> str:
    # `or 1.0` would turn a genuine 0.0 (ruled out) into 1.0 and suppress
    # the warning for exactly the player who most needs it.
    raw = c.get("p_play")
    p_play = 1.0 if raw is None else float(raw)
    if p_play < 0.7:
        return (f"Start probability only {p_play:.0%}. A benched captain is the "
                f"single most damaging outcome in FPL -- the armband does not "
                f"roll over to the vice-captain unless he plays zero minutes.")
    if float(c.get("p_blank_le2", 0.0) or 0.0) > 0.5:
        return "More likely than not to return 2 points or fewer. High variance pick."
    return ""


def compare_captains(a_samples: np.ndarray, b_samples: np.ndarray) -> dict:
    """Head-to-head on the joint simulation draws, not on summary statistics.

    Two players with identical means and identical p90s can still have very
    different head-to-head probabilities -- especially teammates, whose
    outcomes are correlated because they share a scoreline. Only the joint
    samples can answer this.
    """
    a, b = np.asarray(a_samples, float), np.asarray(b_samples, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = 2 * (a - b)      # captaincy doubles the difference
    return {
        "p_a_outscores_b": float((a > b).mean()),
        "p_tie": float((a == b).mean()),
        "mean_swing": float(diff.mean()),
        "p_swing_10plus": float((diff >= 10).mean()),
        "p_swing_minus10": float((diff <= -10).mean()),
        "regret_p10": float(np.percentile(diff, 10)),
    }
