"""Decision theory: what you are actually optimising.

THE DEEPEST GAP IN v5 (audit section 3.4)
-----------------------------------------
Expected points is the right objective only if your goal is to maximise your
season total in isolation. If your goal is OVERALL RANK, or beating a
mini-league, the correct objective is a function of the distribution of

    your score  -  the field's score

which makes effective ownership a first-class term, not an optional "mode"
bolted on beside pure xP maximisation.

The intuition, concretely: captaining a 60%-owned Haaland has near-zero rank
variance. Whatever he does, most of your rivals do it too. Captaining a
4%-owned player with the SAME expected points has enormous rank variance. The
expected points are identical and the decisions are opposite, depending on
whether you are chasing the field or protecting a lead.

    effective_value = xp * (1 - risk_lambda * EO)

At risk_lambda = 0 this is pure expected points, which is correct when you are
simply trying to score as much as possible. At risk_lambda = 1 it is pure
differential value: a player owned by everyone contributes nothing to your
rank, because his points are already in everyone else's total.

The right setting is not a matter of taste. It depends on where you are in the
distribution and how much time is left, which is what recommend_risk_lambda
below encodes.
"""
from __future__ import annotations

import numpy as np


# How much of a player's "differential upside" -- the part of his ceiling
# that most rivals do not share -- gets added back into his value as
# risk_lambda rises. Zero recovers the old formula exactly.
CEILING_WEIGHT = 0.5


def effective_value(player: dict, risk_lambda: float = 0.0) -> float:
    """Rank-aware value of a player.

    v7's version was `xp * (1 - risk_lambda * EO)`: monotone in xP and blind
    to the SHAPE of a player's distribution. Two players with identical xP
    and EO but very different spreads -- a nailed, low-ceiling defender and a
    boom-or-bust forward -- scored identically. That directly contradicts the
    point of a differential: a differential's value is specifically that it
    might return far more than its mean while most of the field has nothing
    riding on it.

    This adds that back in explicitly, as a separate, inspectable term rather
    than folding it into a single opaque score:

        value = xp * (1 - risk_lambda * EO)
                + risk_lambda * CEILING_WEIGHT * (ceiling - xp) * (1 - EO)

    The second term is the player's upside ABOVE his own mean, discounted by
    how much of the field already shares it. A high-EO player's ceiling
    barely counts, because if he hauls, so does most of the field, and it
    does nothing for your RANK. A low-EO player's ceiling counts close to in
    full. At risk_lambda = 0 this is exactly xP, unchanged from before.

    `player["xp_p90"]` is read as the ceiling when present (the simulator
    stores it with every prediction); falls back to xp itself -- a zero
    ceiling bonus -- when it is not, which recovers the old behaviour exactly
    for a caller that has not supplied a distribution.

    EO is read from `player["eo"]` when present (a real sampled effective
    ownership -- see `ingestion/ownership.py`) and falls back to
    `selected_by_percent` (raw ownership) otherwise, flagged in the returned
    detail so a caller can tell which one was actually used.
    """
    xp = float(player.get("xp", 0.0))
    if risk_lambda <= 0:
        return xp
    raw_eo = player.get("eo")
    used_real_eo = raw_eo is not None
    eo = float(raw_eo if used_real_eo else player.get("selected_by_percent", 0.0) or 0.0)
    if eo > 1.0:
        eo = eo / 100.0
    eo = float(np.clip(eo, 0.0, 1.0))
    lam = float(np.clip(risk_lambda, 0.0, 1.0))

    # Candidates built by optimizers/captaincy.py and the API use "p90";
    # accept "xp_p90" too for callers that pass a raw predictions row.
    ceiling = float(player.get("p90", player.get("xp_p90", xp)) or xp)
    upside = max(ceiling - xp, 0.0)
    ceiling_bonus = lam * CEILING_WEIGHT * upside * (1.0 - eo)

    return xp * (1.0 - lam * eo) + ceiling_bonus


def effective_value_detail(player: dict, risk_lambda: float = 0.0) -> dict:
    """Same computation as `effective_value`, with every term broken out.
    Used by the captaincy view so a person can see why a differential
    outranked a safer, higher-xP pick rather than trusting an opaque score.
    """
    xp = float(player.get("xp", 0.0))
    raw_eo = player.get("eo")
    used_real_eo = raw_eo is not None
    eo = float(raw_eo if used_real_eo else player.get("selected_by_percent", 0.0) or 0.0)
    if eo > 1.0:
        eo = eo / 100.0
    eo = float(np.clip(eo, 0.0, 1.0))
    lam = float(np.clip(risk_lambda, 0.0, 1.0))
    ceiling = float(player.get("p90", player.get("xp_p90", xp)) or xp)
    upside = max(ceiling - xp, 0.0)
    ceiling_bonus = lam * CEILING_WEIGHT * upside * (1.0 - eo)
    base = xp * (1.0 - lam * eo)
    return {
        "value": base + ceiling_bonus, "xp": xp, "eo": eo,
        "eo_is_real": used_real_eo, "base_term": base,
        "ceiling_bonus": ceiling_bonus, "ceiling": ceiling,
    }


def recommend_risk_lambda(percentile_rank: float | None, gws_remaining: int) -> dict:
    """Suggest a risk setting from your position and the time left.

    Leading with little time left -> protect: mirror the template, take
    risk_lambda near 0 so your score tracks the field. Behind with time to
    spare -> differentiate: you cannot climb by owning what everyone owns, so
    push risk_lambda up and accept the variance.

    This is a recommendation with its reasoning attached, not an automatic
    setting -- the choice is genuinely yours to make.
    """
    if percentile_rank is None:
        return {"risk_lambda": 0.0,
                "reason": "No rank supplied; defaulting to pure expected points."}
    p = float(np.clip(percentile_rank, 0.0, 1.0))
    time_factor = float(np.clip(gws_remaining / 38.0, 0.0, 1.0))

    if p <= 0.10:
        base = 0.15 * time_factor
        reason = ("You are near the top. Your main risk is variance, not lack of "
                  "upside -- staying close to the template protects the position.")
    elif p >= 0.60:
        base = 0.55 + 0.25 * time_factor
        reason = ("You are well behind. Matching the template locks in the gap; "
                  "differentials are the only mechanism that closes it.")
    else:
        base = 0.25 + 0.30 * (p - 0.10) / 0.50 * time_factor
        reason = ("Mid-table. A moderate differential tilt improves the upside "
                  "without exposing you to a collapse.")
    if gws_remaining <= 5 and p <= 0.30:
        base *= 0.5
        reason += " Few gameweeks remain, so the downside of variance is larger."
    return {"risk_lambda": round(float(np.clip(base, 0.0, 1.0)), 3), "reason": reason}


def rank_delta_distribution(my_samples: np.ndarray, template_samples: np.ndarray) -> dict:
    """The distribution that actually determines your rank movement.

    Both inputs are Monte Carlo samples of a total gameweek score -- yours and
    a template squad's. Their DIFFERENCE is what moves you up or down, and it
    is not derivable from the two means, because the correlation between the
    squads (shared players, shared fixtures) suppresses the variance.
    """
    delta = np.asarray(my_samples, float) - np.asarray(template_samples, float)
    return {
        "mean_gain": float(delta.mean()),
        "p_beat_template": float((delta > 0).mean()),
        "p_gain_10plus": float((delta >= 10).mean()),
        "p_lose_10plus": float((delta <= -10).mean()),
        "p10": float(np.percentile(delta, 10)),
        "p90": float(np.percentile(delta, 90)),
        "std": float(delta.std()),
    }
