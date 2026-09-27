"""Deterministic consistency flags on a player's projection.

WHY THIS EXISTS
----------------
Every component here already comes from an independently-modelled, unit-
tested source (minutes.py, defcon.py, simulation.py...). Combining them is
where silent contradictions live: nothing forces xp_mean and p_play to move
together, and a genuine data or pipeline bug often shows up first as one
number disagreeing with another, not as either number being obviously wrong
in isolation.

These checks are cheap, always-on, and independent of the LLM layer -- they
run whether or not LLM_API_KEY is set. The LLM's job (see agent/llm.py) is to
narrate a projection in prose; this module's job is to decide whether that
narration should carry a warning. It is deliberately conservative: thresholds
flag "worth a human's attention", not "definitely wrong".
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConsistencyFlag:
    code: str
    message: str
    severity: str = "warning"   # "warning" | "info"


def check_prediction(current: dict, previous: dict | None = None) -> list[ConsistencyFlag]:
    """current / previous: rows shaped like the /predictions/{player_id}
    response (xp_mean, p_play, p_sixty, xp_p90, xp_p10, ...). previous is the
    prior gameweek's row for the same player, if one exists.
    """
    flags: list[ConsistencyFlag] = []

    xp = current.get("xp_mean")
    p_play = current.get("p_play")
    p10 = current.get("xp_p10")
    p90 = current.get("xp_p90")

    # A meaningfully high expected-points figure alongside a coin-flip or
    # worse chance of playing at all is internally inconsistent unless the
    # ceiling explains it -- flag it rather than let the headline number
    # imply more certainty than the model actually has.
    if xp is not None and p_play is not None and xp >= 4.0 and p_play < 0.5:
        flags.append(ConsistencyFlag(
            "high_xp_low_pplay",
            f"xP is {xp:.1f} but start probability is only {p_play:.0%} -- "
            "the headline number is being carried by a small chance of a big "
            "haul, not a likely outcome.",
        ))

    if p10 is not None and p90 is not None and xp is not None:
        spread = p90 - p10
        if xp > 0 and spread > 3 * max(xp, 0.5):
            flags.append(ConsistencyFlag(
                "wide_distribution",
                f"The p10-p90 range ({p10:.1f}-{p90:.1f}) is very wide "
                f"relative to xP ({xp:.1f}) -- treat the mean as unstable "
                "this week, not a confident point estimate.",
            ))

    if previous:
        prev_xp = previous.get("xp_mean")
        prev_pplay = previous.get("p_play")
        if prev_xp and xp is not None and prev_pplay is not None and p_play is not None:
            xp_up = prev_xp > 0 and (xp - prev_xp) / prev_xp >= 0.25
            pplay_down = (p_play - prev_pplay) <= -0.15
            if xp_up and pplay_down:
                flags.append(ConsistencyFlag(
                    "diverging_week_over_week",
                    f"xP rose {((xp - prev_xp) / prev_xp):.0%} week-over-week "
                    f"while start probability fell {prev_pplay:.0%} -> "
                    f"{p_play:.0%} -- these should usually move together; "
                    "worth checking what each component driver changed.",
                ))

    return flags
