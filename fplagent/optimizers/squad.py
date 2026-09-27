"""Squad optimisation by integer linear programming.

THE BUG THIS FIXES (v5 audit, item 1.8)
---------------------------------------
v5 maximised total expected points across all FIFTEEN squad members:

    prob += lpSum(p["xp"] * x[p["player_id"]] for p in pool)

But you only score ELEVEN. Optimising the 15-man total makes the solver treat
a bench slot as worth exactly as much as a starting slot, so it will happily
spend 30m on players who never play. The classic symptom is a "perfect" squad
with four premium benchwarmers and a threadbare XI.

The correct formulation has TWO sets of binaries:

    squad[p]  -- 15 selected, budget, club cap, positional quotas
    start[p]  -- 11 selected, a valid formation, and start[p] <= squad[p]
    captain[p] -- exactly 1, and captain[p] <= start[p]

maximising  sum(xp * start)
           + lambda * sum(xp * (squad - start))
           + sum(xp * captain)

The bench weight lambda (~0.12) gives bench players SOME value -- they matter
for autosubs and for Bench Boost -- without letting them drive selection.
Including the captain binary means the armband is chosen jointly with the
squad rather than bolted on afterwards, which genuinely changes the answer:
it is worth paying up for a premium captain in a way a separate captaincy
step cannot see.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fplagent.config import PARAMS

log = logging.getLogger(__name__)


def _require_pulp():
    try:
        import pulp
    except ImportError as exc:  # pragma: no cover - dependency availability
        raise RuntimeError(
            "PuLP is required for optimiser execution. "
            "Install project requirements with "
            "`pip install -r requirements.txt`."
        ) from exc

    return pulp


SQUAD_QUOTA = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

# Valid FPL formations:
# exactly 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 total.
XI_MIN = {
    "GK": 1,
    "DEF": 3,
    "MID": 2,
    "FWD": 1,
}

XI_MAX = {
    "GK": 1,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

MAX_PER_CLUB = 3


@dataclass
class SquadSolution:
    squad: list[dict] = field(default_factory=list)
    starting_xi: list[dict] = field(default_factory=list)
    bench: list[dict] = field(default_factory=list)
    captain: dict | None = None
    vice_captain: dict | None = None
    total_cost: float = 0.0
    xi_expected_points: float = 0.0
    captain_extra_expected_points: float = 0.0
    objective: float = 0.0
    formation: str = ""
    status: str = ""

    def summary(self) -> dict:
        return {
            "formation": self.formation,
            "total_cost": round(self.total_cost, 1),
            "xi_expected_points": round(
                self.xi_expected_points,
                2,
            ),
            "captain_extra_expected_points": round(
                self.captain_extra_expected_points,
                2,
            ),
            "captain": (
                self.captain["name"]
                if self.captain
                else None
            ),
            "vice_captain": (
                self.vice_captain["name"]
                if self.vice_captain
                else None
            ),
            "status": self.status,
        }


def optimise_squad(
    pool: list[dict],
    budget: float = 100.0,
    bench_weight: float | None = None,
    locked_in: list[int] | None = None,
    excluded: list[int] | None = None,
    max_per_club: int = MAX_PER_CLUB,
    risk_lambda: float = 0.0,
    time_limit: int = 30,
) -> SquadSolution:
    """pool entries:
    {player_id, name, position, club, price, xp, [eo], [xp_p90]}

    risk_lambda in [0,1] shifts the objective from raw expected points toward
    differential value -- see fplagent.optimizers.objective for why that is a
    different question, and often the one you actually care about.
    """

    pulp = _require_pulp()

    bench_weight = (
        PARAMS.bench_weight
        if bench_weight is None
        else bench_weight
    )

    locked_in = set(locked_in or [])
    excluded = set(excluded or [])

    pool = [
        p
        for p in pool
        if p["player_id"] not in excluded
    ]

    if len(pool) < 15:
        return SquadSolution(
            status="infeasible: pool smaller than 15"
        )

    from fplagent.optimizers.objective import effective_value

    value = {
        p["player_id"]: effective_value(
            p,
            risk_lambda,
        )
        for p in pool
    }

    prob = pulp.LpProblem(
        "fpl_squad",
        pulp.LpMaximize,
    )

    ids = [
        p["player_id"]
        for p in pool
    ]

    squad = pulp.LpVariable.dicts(
        "squad",
        ids,
        cat="Binary",
    )

    start = pulp.LpVariable.dicts(
        "start",
        ids,
        cat="Binary",
    )

    cap = pulp.LpVariable.dicts(
        "cap",
        ids,
        cat="Binary",
    )

    prob += pulp.lpSum(
        value[i] * start[i]
        + bench_weight * value[i] * (
            squad[i] - start[i]
        )
        + value[i] * cap[i]
        for i in ids
    )

    prob += pulp.lpSum(
        squad[i]
        for i in ids
    ) == 15

    prob += pulp.lpSum(
        start[i]
        for i in ids
    ) == 11

    prob += pulp.lpSum(
        cap[i]
        for i in ids
    ) == 1

    prob += pulp.lpSum(
        p["price"] * squad[p["player_id"]]
        for p in pool
    ) <= budget

    for i in ids:
        prob += start[i] <= squad[i]
        prob += cap[i] <= start[i]

    by_pos: dict[str, list[int]] = {}

    for p in pool:
        by_pos.setdefault(
            p["position"],
            [],
        ).append(
            p["player_id"]
        )

    for pos, n in SQUAD_QUOTA.items():
        members = by_pos.get(
            pos,
            [],
        )

        prob += pulp.lpSum(
            squad[i]
            for i in members
        ) == n

        prob += pulp.lpSum(
            start[i]
            for i in members
        ) >= XI_MIN[pos]

        prob += pulp.lpSum(
            start[i]
            for i in members
        ) <= XI_MAX[pos]

    by_club: dict[str, list[int]] = {}

    for p in pool:
        by_club.setdefault(
            str(p.get("club", "?")),
            [],
        ).append(
            p["player_id"]
        )

    for members in by_club.values():
        prob += pulp.lpSum(
            squad[i]
            for i in members
        ) <= max_per_club

    for i in locked_in:
        if i in squad:
            prob += squad[i] == 1

    prob.solve(
        pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=time_limit,
        )
    )

    status = pulp.LpStatus[prob.status]

    # "Not Solved" means CBC hit the time limit without proving optimality.
    # v6 accepted it, then read .value() off unsolved variables, got None for
    # every one, and returned an EMPTY squad with a success-shaped status.
    # A timeout is a result you can sometimes use and must always be told
    # about, so it is accepted only when the variables actually hold values.
    if status not in (
        "Optimal",
        "Not Solved",
    ):
        return SquadSolution(
            status=f"solver: {status}"
        )

    if any(
        squad[i].value() is None
        for i in ids
    ):
        return SquadSolution(
            status=(
                f"solver: {status} "
                f"(no feasible assignment within "
                f"{time_limit}s -- widen the pool "
                f"or raise time_limit)"
            )
        )

    lookup = {
        p["player_id"]: p
        for p in pool
    }

    chosen = [
        lookup[i]
        for i in ids
        if squad[i].value()
        and squad[i].value() > 0.5
    ]

    xi = [
        lookup[i]
        for i in ids
        if start[i].value()
        and start[i].value() > 0.5
    ]

    bench = [
        p
        for p in chosen
        if p not in xi
    ]

    captain_ids = [
        i
        for i in ids
        if cap[i].value()
        and cap[i].value() > 0.5
    ]

    captain = (
        lookup[captain_ids[0]]
        if captain_ids
        else None
    )

    vice = max(
        (
            p
            for p in xi
            if p is not captain
        ),
        key=lambda p: value[p["player_id"]],
        default=None,
    )

    counts = {
        pos: sum(
            1
            for p in xi
            if p["position"] == pos
        )
        for pos in SQUAD_QUOTA
    }

    formation = (
        f"{counts['DEF']}-"
        f"{counts['MID']}-"
        f"{counts['FWD']}"
    )

    # Bench order matters for autosubs: the first outfield sub on should be
    # the one most likely to actually play, not merely the highest xP.
    bench_outfield = sorted(
        [
            p
            for p in bench
            if p["position"] != "GK"
        ],
        key=lambda p: (
            p.get("p_play", 0.5),
            p.get("xp", 0.0),
        ),
        reverse=True,
    )

    bench_gk = [
        p
        for p in bench
        if p["position"] == "GK"
    ]

    return SquadSolution(
        squad=chosen,
        starting_xi=xi,
        bench=bench_outfield + bench_gk,
        captain=captain,
        vice_captain=vice,
        total_cost=sum(
            p["price"]
            for p in chosen
        ),
        xi_expected_points=sum(
            p.get("xp", 0.0)
            for p in xi
        ),
        captain_extra_expected_points=(
            captain.get("xp", 0.0)
            if captain
            else 0.0
        ),
        objective=float(
            pulp.value(prob.objective)
            or 0.0
        ),
        formation=formation,
        status=status,
    )


def best_xi_from_squad(
    squad: list[dict],
    risk_lambda: float = 0.0,
) -> SquadSolution:
    """Pick the best legal XI and captain from a squad you already own.

    Used every gameweek, and by the bench-boost and autosub logic.
    """

    pulp = _require_pulp()

    from fplagent.optimizers.objective import effective_value

    ids = [
        p["player_id"]
        for p in squad
    ]

    lookup = {
        p["player_id"]: p
        for p in squad
    }

    value = {
        i: effective_value(
            lookup[i],
            risk_lambda,
        )
        for i in ids
    }

    prob = pulp.LpProblem(
        "fpl_xi",
        pulp.LpMaximize,
    )

    start = pulp.LpVariable.dicts(
        "start",
        ids,
        cat="Binary",
    )

    cap = pulp.LpVariable.dicts(
        "cap",
        ids,
        cat="Binary",
    )

    prob += pulp.lpSum(
        value[i] * start[i]
        + value[i] * cap[i]
        for i in ids
    )

    prob += pulp.lpSum(
        start[i]
        for i in ids
    ) == 11

    prob += pulp.lpSum(
        cap[i]
        for i in ids
    ) == 1

    for i in ids:
        prob += cap[i] <= start[i]

    by_pos: dict[str, list[int]] = {}

    for p in squad:
        by_pos.setdefault(
            p["position"],
            [],
        ).append(
            p["player_id"]
        )

    for pos in SQUAD_QUOTA:
        members = by_pos.get(
            pos,
            [],
        )

        prob += pulp.lpSum(
            start[i]
            for i in members
        ) >= XI_MIN[pos]

        prob += pulp.lpSum(
            start[i]
            for i in members
        ) <= XI_MAX[pos]

    prob.solve(
        pulp.PULP_CBC_CMD(
            msg=False
        )
    )

    status = pulp.LpStatus[prob.status]

    # An infeasible squad (e.g. too few available at a position) leaves the
    # variables at arbitrary or None values; reading them would present an
    # illegal XI as if it were the answer.
    if status != "Optimal":
        return SquadSolution(
            squad=squad,
            status=f"solver: {status}",
        )

    xi = [
        lookup[i]
        for i in ids
        if start[i].value()
        and start[i].value() > 0.5
    ]

    bench = [
        p
        for p in squad
        if p not in xi
    ]

    captain_ids = [
        i
        for i in ids
        if cap[i].value()
        and cap[i].value() > 0.5
    ]

    captain = (
        lookup[captain_ids[0]]
        if captain_ids
        else None
    )

    counts = {
        pos: sum(
            1
            for p in xi
            if p["position"] == pos
        )
        for pos in SQUAD_QUOTA
    }

    return SquadSolution(
        squad=squad,
        starting_xi=xi,
        bench=bench,
        captain=captain,
        vice_captain=max(
            (
                p
                for p in xi
                if p is not captain
            ),
            key=lambda p: value[p["player_id"]],
            default=None,
        ),
        total_cost=sum(
            p["price"]
            for p in squad
        ),
        xi_expected_points=sum(
            p.get("xp", 0.0)
            for p in xi
        ),
        captain_extra_expected_points=(
            captain.get("xp", 0.0)
            if captain
            else 0.0
        ),
        formation=(
            f"{counts['DEF']}-"
            f"{counts['MID']}-"
            f"{counts['FWD']}"
        ),
        status=status,
    )