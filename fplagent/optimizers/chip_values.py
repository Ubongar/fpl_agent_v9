"""Build chip-value estimates from stored predictions.

THE GAP THIS FILLS
------------------
`optimizers/chips.py` contains the ILP that schedules each chip to its
best gameweek, but it expects a `chip_value_by_gw` dict on input and
nothing in the codebase ever produced one. That is why `schedule_chips`
was never called by anything: the consumer existed, the producer did not.

This module is the producer. It takes predictions across a horizon of
gameweeks and computes, for each chip, its point value in each gameweek:

  bench_boost      sum of the 4 bench players' expected points
  triple_captain   expected points of the best captain candidate
                   (the chip adds ONE extra copy of his score, not two --
                   see chips.triple_captain_value)
  free_hit         difference between the optimal 15-man squad's XI points
                   and a reference squad's XI points
  wildcard         difference between the optimal 15-man squad's XI points
                   and the current squad's XI points, aggregated over the
                   horizon (this is the "play wildcard now" signal)

TWO MODES
---------
`current_squad=None` (default): values are computed against the OPTIMAL
squad for each gameweek. This is the right mode for a fresh deployment
with no user team: the schedule then reflects "the best possible use of
each chip", which is what a user deciding when to play a chip wants to
know.

`current_squad=[...]`: values are computed against the user's actual
squad. Bench Boost becomes the user's specific bench xP; Free Hit and
Wildcard become the marginal improvement available from a rebuild. This
is the right mode once `fplagent my-team` has been run.

WHY THE USER SQUAD IS RE-EVALUATED PER GAMEWEEK
-----------------------------------------------
The user's squad is FIXED -- they own the same 15 players in GW6 and
GW13 -- but the expected points of those 15 players are DIFFERENT in
every gameweek. If we simply evaluated the reference XI once and reused
it, Free Hit and Wildcard would be identical in every gameweek and would
have no way to detect a blank gameweek, a double gameweek, or a
favourable fixture swing. The rebuild below applies each gameweek's
predictions to the SAME 15 players before picking their best legal XI.

WHY THE REFERENCE XI IS NOT BUILT BY THE FULL OPTIMISER
-------------------------------------------------------
`optimise_squad` chooses WHICH 15 players to buy, subject to budget and
3-per-club. For a fixed user squad we do not want any of that -- we know
the 15 and just need to know their best legal XI under FPL's formation
rules. Calling the full optimiser would work in most cases but fails
whenever the user's squad happens to breach a constraint the optimiser
enforces (e.g. a squad with four defenders after a chip, or a squad
missing a forward). `_best_xi_from_fixed_squad` is a direct search over
the eight legal formations, correct and infeasible-proof.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# The eight legal FPL outfield splits. Summing to 10 in each case.
_LEGAL_FORMATIONS: list[tuple[int, int, int]] = [
    (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2),
    (4, 5, 1), (5, 2, 3), (5, 3, 2), (5, 4, 1),
]


@dataclass
class ChipValues:
    """Point values per chip per gameweek, plus the reference squads used."""
    values: dict[str, dict[int, float]] = field(default_factory=dict)
    reference_xi_points: dict[int, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _pool_from_predictions(preds, club_col: str = "short_name") -> list[dict]:
    """Convert a `latest_predictions` frame into the pool format that
    `optimise_squad` expects."""
    out = []
    for r in preds.itertuples():
        cost = float(getattr(r, "now_cost", 0.0) or 0.0)
        if cost <= 0:
            continue
        out.append({
            "player_id": int(r.player_id),
            "name": str(getattr(r, "web_name", r.player_id)),
            "position": str(r.position),
            "club": str(getattr(r, club_col, r.team_id)),
            "price": cost,
            "xp": float(r.xp_mean),
            "p_play": float(getattr(r, "p_play", 0.0) or 0.0),
            "xp_p90": float(getattr(r, "xp_p90", 0.0) or 0.0),
        })
    return out


def _xi_points(squad_solution) -> float:
    """Sum of the starting XI's xP, ignoring captaincy (the chip marginal
    values in this module assume a separate captain decision)."""
    return float(sum(float(p.get("xp", 0.0)) for p in squad_solution.starting_xi))


def _bench_points(squad_solution) -> float:
    """Sum of the bench players' xP, excluding the backup keeper.

    The reserve goalkeeper scores via a different mechanism (autosubs for a
    starting keeper sent off) and its inclusion would overstate Bench Boost
    by roughly a keeper's clean-sheet equity in every gameweek.
    """
    return float(sum(float(p.get("xp", 0.0))
                     for p in squad_solution.bench
                     if p.get("position") != "GK"))


def _best_xi_from_fixed_squad(squad_pool: list[dict]) -> float:
    """Sum of the highest-xP legal XI's expected points from a fixed squad.

    Searches the eight legal outfield formations, sorting each position
    group by descending xP and taking the top-k of each. Because the
    position groups are disjoint, greedy within a formation is optimal,
    and the max across formations is the global optimum.

    Unlike `optimise_squad`, this does NOT enforce positional quotas on
    the SQUAD -- a user whose wildcard-then-wildcard left them with an
    unusual mix still gets an answer for whatever legal XI their squad
    can field.
    """
    by_pos: dict[str, list[dict]] = {}
    for p in squad_pool:
        by_pos.setdefault(str(p.get("position", "")), []).append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: -float(x.get("xp", 0.0)))

    gks = by_pos.get("GK", [])
    if not gks:
        return 0.0
    best_gk = float(gks[0].get("xp", 0.0))

    best = 0.0
    for d, m, f in _LEGAL_FORMATIONS:
        if (len(by_pos.get("DEF", [])) < d
                or len(by_pos.get("MID", [])) < m
                or len(by_pos.get("FWD", [])) < f):
            continue
        total = (best_gk
                 + sum(float(p.get("xp", 0.0)) for p in by_pos["DEF"][:d])
                 + sum(float(p.get("xp", 0.0)) for p in by_pos["MID"][:m])
                 + sum(float(p.get("xp", 0.0)) for p in by_pos["FWD"][:f]))
        best = max(best, total)
    return best


def compute_chip_values(
    predictions_by_gw: dict[int, object],
    horizon_gws: list[int] | None = None,
    current_squad: list[dict] | None = None,
    budget: float = 100.0,
    risk_lambda: float = 0.0,
) -> ChipValues:
    """Compute per-chip values across a horizon.

    predictions_by_gw: {gw: latest_predictions-style frame}. Each frame
      must have columns player_id, web_name, position, short_name/team_id,
      now_cost, xp_mean, xp_p90, p_play. The output of
      `pipeline.run.latest_predictions(gw=N)` satisfies this.

    horizon_gws: which gameweeks to include. Defaults to the sorted keys
      of predictions_by_gw.

    current_squad: optional. A list of player dicts with at least the keys
      player_id, position, club, price. The `xp` field, if present, is
      IGNORED -- the squad is re-evaluated against each gameweek's own
      predictions. See the module docstring.

    Returns a ChipValues object with a `values` dict shaped exactly like
    what `chips.schedule_chips` expects:
      {"bench_boost": {gw: xp}, "triple_captain": {gw: xp},
       "free_hit": {gw: xp}, "wildcard": {gw: xp}}
    """
    from fplagent.optimizers.squad import optimise_squad

    if not predictions_by_gw:
        return ChipValues(notes=["no predictions supplied"])

    gws = sorted(horizon_gws or predictions_by_gw.keys())

    bench_boost: dict[int, float] = {}
    triple_captain: dict[int, float] = {}
    free_hit: dict[int, float] = {}
    wildcard: dict[int, float] = {}
    reference_xi: dict[int, float] = {}
    notes: list[str] = []

    user_squad_pool = current_squad

    for gw in gws:
        preds = predictions_by_gw.get(gw)
        if preds is None or getattr(preds, "empty", True):
            notes.append(f"GW{gw}: no predictions; skipped")
            continue

        pool = _pool_from_predictions(preds)
        if not pool:
            notes.append(f"GW{gw}: no priced players; skipped")
            continue

        try:
            optimal = optimise_squad(pool, budget=budget, risk_lambda=risk_lambda)
        except Exception as e:      # noqa: BLE001
            notes.append(f"GW{gw}: optimiser failed ({e}); skipped")
            continue

        if optimal.status not in ("Optimal", "Not Solved"):
            notes.append(f"GW{gw}: squad status {optimal.status}; skipped")
            continue

        opt_xi = _xi_points(optimal)
        reference_xi[gw] = opt_xi
        bench_boost[gw] = _bench_points(optimal)

        # Triple Captain's marginal value is the ADDITIONAL copy of the
        # captain's score, not the whole 3x. The base 2x is free.
        if optimal.captain is not None:
            triple_captain[gw] = float(optimal.captain.get("xp", 0.0))
        else:
            triple_captain[gw] = 0.0

        # Free Hit / Wildcard against a real user squad: re-evaluate the
        # FIXED 15 players against THIS gameweek's predictions, then take
        # their best legal XI. The player identities do not change week to
        # week but their xP does -- a blank gameweek, a double, or a
        # favourable fixture swing all affect the reference XI's score, and
        # the Free Hit value should reflect that.
        if user_squad_pool is not None:
            preds_by_id = ({int(r.player_id): r for r in preds.itertuples()}
                           if not preds.empty else {})
            user_pool_this_gw: list[dict] = []
            for p in user_squad_pool:
                r = preds_by_id.get(int(p["player_id"]))
                if r is not None:
                    user_pool_this_gw.append({
                        **p,
                        "xp": float(r.xp_mean),
                        "p_play": float(getattr(r, "p_play", 0.0) or 0.0),
                        "xp_p90": float(getattr(r, "xp_p90", 0.0) or 0.0),
                    })
                else:
                    # Player has no fixture this GW (blank) -> xP = 0.
                    user_pool_this_gw.append({**p, "xp": 0.0,
                                              "p_play": 0.0, "xp_p90": 0.0})
            user_xi = _best_xi_from_fixed_squad(user_pool_this_gw)
            free_hit[gw] = max(0.0, opt_xi - user_xi)
            wildcard[gw] = max(0.0, opt_xi - user_xi)
        else:
            free_hit[gw] = 0.0
            wildcard[gw] = 0.0

    if user_squad_pool is None:
        notes.append(
            "No current_squad supplied. Bench Boost and Triple Captain "
            "values are computed against the optimal squad, which is the "
            "right proxy for a fresh deployment. Free Hit and Wildcard "
            "values are 0 because without a reference squad they have no "
            "meaningful magnitude. Rerun with --entry-id YOUR_FPL_ID to "
            "supply your own squad.")
    else:
        notes.append(
            f"Reference squad: {len(user_squad_pool)} players. Free Hit "
            f"and Wildcard are computed against YOUR squad re-evaluated "
            f"at each gameweek, not against the optimal squad.")

    return ChipValues(
        values={
            "bench_boost": bench_boost,
            "triple_captain": triple_captain,
            "free_hit": free_hit,
            "wildcard": wildcard,
        },
        reference_xi_points=reference_xi,
        notes=notes,
    )