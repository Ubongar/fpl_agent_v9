"""Multi-gameweek transfer planning.

WHAT v5 DID: a greedy single swap maximising this week's xP gain, with a
`free_transfers` count and no cap.

WHAT THE 2026/27 RULES ACTUALLY REQUIRE:
  * You may BANK UP TO FIVE free transfers. Not two -- five. That changes
    planning completely, because saving is now a genuinely powerful move
    rather than a marginal one.
  * Each transfer beyond your free allowance costs 4 points.
  * A transfer's value is not this week's gain, it is the gain over the whole
    horizon you will hold the player for. A greedy weekly optimiser
    systematically buys players with one good fixture and then pays another
    hit to get rid of them.

So this solves the real problem: over a 3-5 gameweek horizon, choose WHICH
transfers to make and WHEN, subject to budget, squad legality, the free
transfer accumulation rule, and hit costs. Modelled as an ILP over
(player, gameweek) decisions rather than a greedy scan.

CAPTAINCY IS IN THE OBJECTIVE
-----------------------------
The starting XI objective used to sum `val * start` with no captain term,
which meant a premium striker was worth the same to the transfer planner as
a budget midfielder if their xP were equal. They are not equal: the armband
DOUBLES one player's score, and the marginal value of buying a captain-
worthy premium is that doubling, not the raw xP. The captain binary is now
part of the objective (`val * cap`, one cap per gameweek, cap <= start),
mirroring what `optimizers/squad.py` already does.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field


from fplagent.optimizers import _lp
from fplagent.optimizers.squad import MAX_PER_CLUB, SQUAD_QUOTA, XI_MAX, XI_MIN

log = logging.getLogger(__name__)


def _require_pulp():
    try:
        import pulp
    except ImportError as exc:  # pragma: no cover - dependency availability
        raise RuntimeError(
            "PuLP is required for optimiser execution. Install project requirements with `pip install -r requirements.txt`."
        ) from exc
    return pulp

MAX_BANKED_FREE_TRANSFERS = 5
HIT_COST = 4
DEFAULT_MAX_TRANSFERS_PER_GW = 20


@dataclass
class TransferPlan:
    moves: list[dict] = field(default_factory=list)
    total_gain: float = 0.0
    hits_taken: int = 0
    net_gain: float = 0.0
    status: str = ""
    reasoning: list[str] = field(default_factory=list)


def sell_price(p: dict) -> float:
    """FPL sells a risen asset back at purchase price plus half the rise,
    rounded down to the nearest 0.1m -- never at its current market price.
    A player who has fallen sells at his current price with no floor at the
    purchase price (FPL does not protect the downside).

    A module-level function, not a closure inside `plan_transfers`, because
    `ingestion/manager_team.py`'s real-squad endpoint needs the exact same
    rule to report a fetched team's true sell values -- one implementation
    for both callers, not two that could drift apart.
    """
    bought = p.get("purchase_price")
    now = float(p["price"])
    if bought is None:
        return now
    bought = float(bought)
    if now <= bought:
        return now
    # In whole tenths: prices are 0.1m steps, and doing the halving in float
    # millions loses a tenth to representation error (7.1 - 6.9 is
    # 0.19999..., which floored to a 0.0m profit instead of 0.1m).
    rise_tenths = int(round((now - bought) * 10.0))
    return round(bought + (rise_tenths // 2) / 10.0, 1)


def free_transfers_after(current_ft: int, transfers_made: int) -> int:
    """FPL's accumulation rule, capped at five."""
    used = min(transfers_made, current_ft)
    remaining = current_ft - used
    return min(remaining + 1, MAX_BANKED_FREE_TRANSFERS)


def plan_transfers(current_squad: list[dict], candidate_pool: list[dict],
                   xp_by_gw: dict[int, dict[int, float]],
                   horizon_gws: list[int], bank: float = 0.0,
                   free_transfers: int = 1, max_transfers_per_gw: int | None = None,
                   risk_lambda: float = 0.0, time_limit: int = 45) -> TransferPlan:
    """ILP over the horizon.

    xp_by_gw: {gw: {player_id: expected_points}} for every player considered.
    Missing entries are treated as 0, which is exactly right for a blank
    gameweek -- a player with no fixture genuinely cannot score.
    """
    owned = {p["player_id"]: p for p in current_squad}
    pool = {p["player_id"]: p for p in candidate_pool}
    universe = {**pool, **owned}
    ids = list(universe)
    gws = sorted(horizon_gws)
    # Current FPL rules cap ordinary transfers at 20 in a single gameweek.
    # (Wildcard/Free Hit are separate chip mechanics and are not represented
    # by this ordinary-transfer planner.)
    pulp = _require_pulp()
    max_transfers_per_gw = (DEFAULT_MAX_TRANSFERS_PER_GW if max_transfers_per_gw is None
                            else max(0, int(max_transfers_per_gw)))
    if not gws:
        return TransferPlan(status="no horizon supplied")

    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)

    # own[i][g] : do we hold player i during gameweek g
    own = {(i, g): _lp.binary(prob, f"own_{i}_{g}") for i in ids for g in gws}
    start = {(i, g): _lp.binary(prob, f"st_{i}_{g}") for i in ids for g in gws}
    cap = {(i, g): _lp.binary(prob, f"cap_{i}_{g}") for i in ids for g in gws}
    buy = {(i, g): _lp.binary(prob, f"buy_{i}_{g}") for i in ids for g in gws}
    sell = {(i, g): _lp.binary(prob, f"sell_{i}_{g}") for i in ids for g in gws}
    hits = {g: _lp.new_var(prob, f"hits_{g}", low=0, cat="Integer") for g in gws}

    from fplagent.optimizers.objective import effective_value

    def val(i: int, g: int) -> float:
        xp = xp_by_gw.get(g, {}).get(i, 0.0)
        return effective_value({**universe[i], "xp": xp}, risk_lambda)

    # The captain term doubles one player per gameweek, exactly as it does
    # in the real game. Without it, the transfer planner values a premium
    # striker the same as an equally-priced budget midfielder at equal xP,
    # which under-buys captain candidates. See module docstring.
    prob += (pulp.lpSum(val(i, g) * start[(i, g)] + val(i, g) * cap[(i, g)]
                        for i in ids for g in gws)
             - HIT_COST * pulp.lpSum(hits[g] for g in gws))

    # --- squad legality every week ---------------------------------------
    for g in gws:
        prob += pulp.lpSum(own[(i, g)] for i in ids) == 15
        prob += pulp.lpSum(start[(i, g)] for i in ids) == 11
        prob += pulp.lpSum(cap[(i, g)] for i in ids) == 1
        for pos, n in SQUAD_QUOTA.items():
            members = [i for i in ids if universe[i]["position"] == pos]
            prob += pulp.lpSum(own[(i, g)] for i in members) == n
            prob += pulp.lpSum(start[(i, g)] for i in members) >= XI_MIN[pos]
            prob += pulp.lpSum(start[(i, g)] for i in members) <= XI_MAX[pos]
        clubs: dict[str, list[int]] = {}
        for i in ids:
            clubs.setdefault(str(universe[i].get("club", "?")), []).append(i)
        for members in clubs.values():
            prob += pulp.lpSum(own[(i, g)] for i in members) <= MAX_PER_CLUB
        for i in ids:
            prob += start[(i, g)] <= own[(i, g)]
            prob += cap[(i, g)] <= start[(i, g)]

    # --- budget -----------------------------------------------------------
    # A player you already own is worth his SELL price to the budget, not
    # his market price: that is what keeping him forgoes. Costing owned
    # players at market price made any squad with a price rise unable to
    # afford itself, so the plan came back infeasible for exactly the
    # managers whose picks had gone up.
    total_budget = sum(sell_price(p) for p in current_squad) + bank
    cost = {i: (sell_price(owned[i]) if i in owned else float(universe[i]["price"]))
            for i in ids}
    for g in gws:
        prob += pulp.lpSum(cost[i] * own[(i, g)] for i in ids) <= total_budget

    # --- transfer continuity ---------------------------------------------
    first = gws[0]
    for i in ids:
        prev = 1 if i in owned else 0
        prob += own[(i, first)] - prev == buy[(i, first)] - sell[(i, first)]
        prob += buy[(i, first)] + sell[(i, first)] <= 1
        for a, b in zip(gws, gws[1:]):
            prob += own[(i, b)] - own[(i, a)] == buy[(i, b)] - sell[(i, b)]
            prob += buy[(i, b)] + sell[(i, b)] <= 1

    # --- free transfer accumulation, capped at five ----------------------
    ft = {g: _lp.new_var(prob, f"ft_{g}", low=0, up=MAX_BANKED_FREE_TRANSFERS,
                         cat="Integer") for g in gws}
    prob += ft[first] == min(free_transfers, MAX_BANKED_FREE_TRANSFERS)
    for g in gws:
        n_in = pulp.lpSum(buy[(i, g)] for i in ids)
        prob += n_in <= max_transfers_per_gw
        prob += hits[g] >= n_in - ft[g]
        prob += hits[g] >= 0
    # Free transfers roll forward, capped at five. The rule is
    #     ft[next] = min(ft[now] - min(transfers, ft[now]) + 1, 5)
    # and the inner min() is the part v6 dropped. It wrote
    #     ft[b] <= ft[a] - (total transfers) + 1
    # which subtracts EVERY transfer instead of only the free ones actually
    # consumed. With one free transfer and three moves that gives
    # ft[b] <= -1 against a lowBound of 0, so the plan is infeasible and the
    # solver routes around it: measured, it would only ever take three
    # transfers in the final gameweek of the horizon, where no successor
    # constraint exists. With two moves it gives ft[b] <= 0, silently
    # destroying a free transfer FPL would have granted.
    #
    # `used` linearises min(n_in, ft[g]). The two upper bounds alone are not
    # enough: the solver would set used=0 to carry a larger ft into next week
    # and get transfers for free. The lower bound n_in - hits pins it,
    # because n_in - max(0, n_in - ft) is exactly min(n_in, ft).
    used = {g: _lp.new_var(prob, f"used_{g}", low=0,
                           up=MAX_BANKED_FREE_TRANSFERS,
                           cat="Integer") for g in gws}
    for g in gws:
        prob += used[g] <= pulp.lpSum(buy[(i, g)] for i in ids)
        prob += used[g] <= ft[g]
        prob += used[g] >= pulp.lpSum(buy[(i, g)] for i in ids) - hits[g]
    cap_hit = {b: _lp.binary(prob, f"ft_cap_{b}")
               for _, b in zip(gws, gws[1:])}
    for a, b in zip(gws, gws[1:]):
        raw_next = ft[a] - used[a] + 1
        # Exact min(raw_next, 5). raw_next is at most 6, and it equals 6 only
        # when five transfers were banked and none were used. The binary gate
        # prevents the solver from understating the real carry-forward balance.
        prob += ft[b] <= raw_next
        prob += ft[b] <= MAX_BANKED_FREE_TRANSFERS
        prob += ft[b] >= raw_next - MAX_BANKED_FREE_TRANSFERS * cap_hit[b]
        prob += ft[b] >= MAX_BANKED_FREE_TRANSFERS * cap_hit[b]

    status = _lp.solve(prob, time_limit)
    if status not in ("Optimal", "Not Solved"):
        return TransferPlan(status=f"solver: {status}")
    # A timeout with no incumbent leaves every variable at None, which the
    # extraction below would read as "no moves, roll the transfer" -- a
    # success-shaped answer the solver never produced.
    if any(own[(i, g)].value() is None for i in ids for g in gws):
        return TransferPlan(
            status=f"solver: {status} (no feasible plan within {time_limit}s)")

    moves, reasoning = [], []
    total_hits = 0
    for g in gws:
        ins = [universe[i] for i in ids if buy[(i, g)].value() and buy[(i, g)].value() > 0.5]
        outs = [universe[i] for i in ids if sell[(i, g)].value() and sell[(i, g)].value() > 0.5]
        h = int(round(hits[g].value() or 0))
        total_hits += h
        if ins or outs:
            moves.append({
                "gw": g,
                "in": [{"player_id": p["player_id"], "name": p.get("name"),
                        "price": p["price"],
                        "xp": round(xp_by_gw.get(g, {}).get(p["player_id"], 0.0), 2)}
                       for p in ins],
                "out": [{"player_id": p["player_id"], "name": p.get("name"),
                         "price": p["price"],
                         "xp": round(xp_by_gw.get(g, {}).get(p["player_id"], 0.0), 2)}
                        for p in outs],
                "hits": h,
            })
        else:
            reasoning.append(f"GW{g}: roll the transfer. Banking is worth more than "
                             f"any available move this week.")

    xi_total = sum(sum(xp_by_gw.get(g, {}).get(i, 0.0)
                       for i in ids if start[(i, g)].value() and start[(i, g)].value() > 0.5)
                   for g in gws)
    return TransferPlan(
        moves=moves, total_gain=round(xi_total, 2), hits_taken=total_hits,
        net_gain=round(xi_total - HIT_COST * total_hits, 2),
        status=status, reasoning=reasoning,
    )


def evaluate_hit(gain_over_horizon: float, n_transfers: int,
                 free_transfers: int) -> dict:
    """Is a hit worth it? Answered over the horizon, not over one week.

    The classic mistake is comparing a 4-point hit against a single
    gameweek's gain. If you will hold the player for six weeks, the relevant
    comparison is the six-week gain -- and conversely, a 5-point gain this
    week that reverses next week is not worth 4 points.
    """
    paid = max(0, n_transfers - free_transfers)
    cost = paid * HIT_COST
    return {
        "transfers": n_transfers, "free_used": min(n_transfers, free_transfers),
        "hits": paid, "point_cost": cost,
        "gross_gain": round(gain_over_horizon, 2),
        "net_gain": round(gain_over_horizon - cost, 2),
        "worth_it": gain_over_horizon > cost,
        "breakeven_gain_required": cost,
    }