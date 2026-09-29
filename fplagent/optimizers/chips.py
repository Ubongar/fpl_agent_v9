"""Chip scheduling under the 2026/27 two-window structure.

WHAT CHANGED THAT v5's PLANNER CANNOT SEE
-----------------------------------------
There are now TWO sets of chips -- Wildcard, Free Hit, Bench Boost, Triple
Captain -- one per half of the season. The first set must be played before the
GW19 deadline and CANNOT be carried over; anything unused at that point is
simply lost. The second set is available from GW20.

That makes chip timing a CONSTRAINED SCHEDULING problem across two windows
with a use-it-or-lose-it deadline, not the "sort the weeks by score and take
the best one" ranking v5 implemented. The difference is material in December:
a ranking says "your best Bench Boost week is GW26"; a scheduler says "your
best FIRST-HALF Bench Boost week is GW17, and if you wait for GW26 you forfeit
the first-half chip entirely."

Additional rules encoded here: one chip per gameweek, Free Hit unusable in
GW1, and a Free Hit played in GW19 blocks GW20 (the squad reverts at the next
deadline, so consecutive Free Hits are impossible).
"""
from __future__ import annotations

from dataclasses import dataclass, field


from fplagent.config import FIRST_HALF_LAST_GW, TOTAL_GWS
from fplagent.optimizers import _lp

CHIPS = ("wildcard", "free_hit", "bench_boost", "triple_captain")


def _require_pulp():
    """PuLP is an optional import so the module can be loaded without it
    (for the type signatures and dataclasses), but every function that
    actually builds an ILP calls this first."""
    try:
        import pulp
    except ImportError as exc:  # pragma: no cover - dependency availability
        raise RuntimeError(
            "PuLP is required for chip scheduling. Install project "
            "requirements with `pip install -r requirements.txt`."
        ) from exc
    return pulp


@dataclass
class ChipSchedule:
    assignments: dict[str, dict[str, int | None]] = field(default_factory=dict)
    total_value: float = 0.0
    status: str = ""
    notes: list[str] = field(default_factory=list)


def schedule_chips(chip_value_by_gw: dict[str, dict[int, float]],
                   available_from_gw: int,
                   chips_already_used: dict[str, list[int]] | None = None,
                   first_half_last_gw: int = FIRST_HALF_LAST_GW,
                   total_gws: int = TOTAL_GWS) -> ChipSchedule:
    """chip_value_by_gw: {chip_name: {gw: estimated point value}}

    Returns the best assignment of each half-season chip to a gameweek.
    """
    pulp = _require_pulp()
    used = chips_already_used or {}
    gws = [g for g in range(available_from_gw, total_gws + 1)]
    if not gws:
        return ChipSchedule(status="season over")

    first_half = [g for g in gws if g <= first_half_last_gw]
    second_half = [g for g in gws if g > first_half_last_gw]

    prob = pulp.LpProblem("chips", pulp.LpMaximize)
    x: dict[tuple[str, str, int], object] = {}
    for chip in CHIPS:
        for half, window in (("first", first_half), ("second", second_half)):
            if used.get(chip) and any(
                (g <= first_half_last_gw) == (half == "first") for g in used[chip]
            ):
                continue                       # that half's copy is already spent
            for g in window:
                if chip == "free_hit" and g == 1:
                    continue                   # Free Hit is not allowed in GW1
                x[(chip, half, g)] = _lp.binary(prob, f"x_{chip}_{half}_{g}")

    if not x:
        return ChipSchedule(status="no chips remaining")

    prob += pulp.lpSum(chip_value_by_gw.get(chip, {}).get(g, 0.0) * var
                       for (chip, half, g), var in x.items())

    # Each remaining chip copy is played at most once.
    for chip in CHIPS:
        for half in ("first", "second"):
            keys = [k for k in x if k[0] == chip and k[1] == half]
            if keys:
                prob += pulp.lpSum(x[k] for k in keys) <= 1

    # One chip per gameweek, across all chips and both halves.
    for g in gws:
        keys = [k for k in x if k[2] == g]
        if keys:
            prob += pulp.lpSum(x[k] for k in keys) <= 1

    # A Free Hit in GW19 makes GW20 impossible: the squad reverts at the next
    # deadline, so the second-half copy cannot be played the following week.
    fh19 = [k for k in x if k[0] == "free_hit" and k[2] == first_half_last_gw]
    fh20 = [k for k in x if k[0] == "free_hit" and k[2] == first_half_last_gw + 1]
    if fh19 and fh20:
        prob += pulp.lpSum(x[k] for k in fh19) + pulp.lpSum(x[k] for k in fh20) <= 1

    status = _lp.solve(prob)

    assignments: dict[str, dict[str, int | None]] = {
        c: {"first": None, "second": None} for c in CHIPS}
    for (chip, half, g), var in x.items():
        if var.value() and var.value() > 0.5:
            assignments[chip][half] = g

    notes = []
    for chip in CHIPS:
        if assignments[chip]["first"] is None and first_half and not used.get(chip):
            notes.append(
                f"No first-half {chip.replace('_', ' ')} scheduled. It expires at the "
                f"GW{first_half_last_gw} deadline and cannot be carried over -- an "
                f"unplayed chip is worth exactly zero.")
        else:
            g = assignments[chip]["first"]
            if g is not None:
                notes.append(f"First-half {chip.replace('_', ' ')}: GW{g} "
                             f"(value {chip_value_by_gw.get(chip, {}).get(g, 0):.1f}).")

    return ChipSchedule(
        assignments=assignments,
        total_value=float(pulp.value(prob.objective) or 0.0),
        status=status, notes=notes,
    )


def bench_boost_value(bench_xp_by_gw: dict[int, float]) -> dict[int, float]:
    """Bench Boost is worth exactly the bench's expected points."""
    return dict(bench_xp_by_gw)


def triple_captain_value(best_captain_xp_by_gw: dict[int, float]) -> dict[int, float]:
    """Triple Captain adds ONE extra multiple, not two.

    You already double your captain for free, so the chip's marginal value is
    a single additional copy of his expected points. Treating it as 3x the
    baseline (as most naive planners do) overstates the chip by 100%.
    """
    return {g: v for g, v in best_captain_xp_by_gw.items()}


def free_hit_value(current_squad_xp_by_gw: dict[int, float],
                   optimal_squad_xp_by_gw: dict[int, float]) -> dict[int, float]:
    """Free Hit is worth the DIFFERENCE between your squad and the best
    possible one-week squad -- which is why it peaks in blank gameweeks, when
    your own squad's value collapses and the optimal one does not."""
    return {g: optimal_squad_xp_by_gw.get(g, 0.0) - current_squad_xp_by_gw.get(g, 0.0)
            for g in optimal_squad_xp_by_gw}