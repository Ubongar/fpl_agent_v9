"""One place that knows which PuLP generation is installed.

WHY THIS FILE EXISTS
--------------------
PuLP 4.0 is a from-scratch rewrite on a Rust core. Four things the optimisers
relied on changed at once, so `requirements.txt` had to pin `PuLP<4.0` until
this layer existed:

  1. Variables. `pulp.LpVariable(name, cat=...)` no longer constructs
     anything; a variable must be created from the problem it belongs to,
     with `prob.add_variable(name, ...)`. (3.3+ also has `add_variable`, so
     it is used wherever it exists; older 3.x falls back to the constructor.)
  2. Solver. `pulp.PULP_CBC_CMD` is gone and CBC is no longer bundled. The
     default solver is HiGHS, which needs the separate `highspy` package.
  3. Status. `prob.status` (an int) and `pulp.LpStatus` are gone. `solve()`
     now returns a stats object whose `.status` is an `LpSolveStatus` enum.
  4. Time limits. CBC reports "Not Solved" when it stops on the clock;
     HiGHS reports `TimeLimit` (and similar) instead.

The optimisers call only the four functions below, so they run unchanged on
either generation. `solve()` returns the same status STRINGS the 3.x
`pulp.LpStatus` table used ("Optimal", "Infeasible", "Unbounded",
"Not Solved", "Undefined"), so every existing `status in (...)` branch keeps
its meaning, including the "Not Solved means a timeout, then check the
variables really hold values" handling in squad.py and transfers.py.

PuLP itself is imported lazily, inside the functions, because the optimiser
modules deliberately stay importable without PuLP installed (each one calls
its own `_require_pulp()` for a clear error at the point of use). Importing
this module therefore never imports PuLP.

Everything else the optimisers use (`LpProblem`, `LpMaximize`, `lpSum`,
`value`, `LpVariable.value()`, `prob += constraint`) is unchanged in 4.0.
"""
from __future__ import annotations

# 4.x statuses that mean "the solver stopped before proving optimality".
# CBC surfaced all of these as "Not Solved"; callers accept that string and
# then verify the variables actually hold values before using them.
_STOPPED = {"NotSolved", "TimeLimit", "GapLimit", "NodeLimit", "IterationLimit",
            "SolutionLimit", "Interrupted", "Stopped", "MemoryLimit"}


def _pulp():
    import pulp
    return pulp


def is_v4() -> bool:
    """True on PuLP 4.x, decided by the API that actually differs.

    Deliberately NOT keyed on `LpProblem.add_variable`: PuLP 3.3 already ships
    it (as a forward-compatible alias), so testing for it would misclassify
    3.3.x as 4.x and send it to a HiGHS solver it may not have. What 4.x
    uniquely has is the `LpSolveStatus` enum, and what it uniquely lacks is
    the bundled CBC command and the int `LpStatus` table.
    """
    pulp = _pulp()
    return hasattr(pulp, "LpSolveStatus") and not hasattr(pulp, "PULP_CBC_CMD")


def version() -> str:
    return getattr(_pulp(), "__version__", "unknown")


def new_var(prob, name: str, *, low: float | None = None,
            up: float | None = None, cat: str = "Continuous"):
    """Create a variable on `prob`. cat: "Continuous" | "Integer" | "Binary".

    `prob.add_variable` exists on PuLP 3.3+ and 4.x and is the ONLY way to
    create a variable on 4.x, so it is used wherever it exists. Older PuLP
    falls back to the standalone constructor.
    """
    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, lowBound=low, upBound=up, cat=cat)
    return _pulp().LpVariable(name, lowBound=low, upBound=up, cat=cat)


def binary(prob, name: str):
    return new_var(prob, name, cat="Binary")


def _solver(time_limit: float | None):
    pulp = _pulp()
    if is_v4():
        if "HiGHS" not in pulp.listSolvers(onlyAvailable=True):
            raise RuntimeError(
                "PuLP 4.x needs the HiGHS solver, which is not bundled. "
                "Install it with: pip install highspy")
        return pulp.HiGHS(msg=False, timeLimit=time_limit)
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)


def solve(prob, time_limit: float | None = None) -> str:
    """Solve `prob` and return a 3.x-style status string."""
    pulp = _pulp()
    if is_v4():
        stats = prob.solve(_solver(time_limit))
        name = getattr(stats.status, "name", str(stats.status))
        if name == "Optimal":
            return "Optimal"
        if name == "Infeasible":
            return "Infeasible"
        if name == "Unbounded":
            return "Unbounded"
        if name in _STOPPED:
            return "Not Solved"
        return "Undefined"
    prob.solve(_solver(time_limit))
    return pulp.LpStatus[prob.status]
