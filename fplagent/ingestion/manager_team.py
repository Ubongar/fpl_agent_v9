"""Fetch a real manager's current FPL squad, for the transfer planner.

WHY THIS EXISTS
----------------
Every optimiser in this project (`optimizers/transfers.py`, `squad.py`)
takes a squad as a plain list of dicts. Until now the only way to build that
list was to click together fifteen players by hand in the dashboard, which
works but means re-entering your team every time you want a fresh transfer
plan. FPL's own site does not expose this as a documented public API, but
the same handful of endpoints the site's own frontend calls are open with no
authentication required, for any team ID -- this is the same class of access
as `ingestion/ownership.py`'s effective-ownership sampler, aimed at FPL's own
endpoints rather than a third party.

    GET /api/entry/{id}/                    -- team name, manager, overall rank
    GET /api/entry/{id}/event/{gw}/picks/    -- the 15 players for one gameweek
    GET /api/entry/{id}/history/             -- gw-by-gw transfer counts and chip usage
    GET /api/entry/{id}/transfers/           -- every transfer ever made, with prices

WHY FREE TRANSFERS ARE RECOMPUTED, NOT READ DIRECTLY
-------------------------------------------------------
FPL's API never exposes "free transfers remaining" as a field. It exposes
how many transfers were MADE each gameweek, which lets the same accumulation
rule already implemented and tested in
`optimizers.transfers.free_transfers_after` be replayed gameweek by
gameweek from the start of the season -- one source of truth for the rule,
not a second copy of it that could drift from the first. Wildcard and Free
Hit gameweeks are skipped in that replay, because using either chip does not
consume or reset the free-transfer count.

WHY PURCHASE PRICE COMES FROM THE TRANSFERS ENDPOINT, NOT GUESSED
---------------------------------------------------------------------
`optimizers/transfers.py`'s `sell_price()` needs a real purchase price to
compute the true sell-back value (buy price plus half any rise). The
`/transfers/` endpoint records exactly that for every player who has EVER
been transferred in. A player still in the squad from the season's opening
gameweek has no such record -- FPL's initial squad selection is not itself a
"transfer" -- and this module says so explicitly (`purchase_price: None`,
`purchase_price_known: False`) rather than quietly assuming his purchase
price equals his current one, which would understate a risen player's
banked profit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fplagent.config import CURRENT_SEASON
from fplagent.ingestion.fpl_client import FPLClient
from fplagent.optimizers.transfers import free_transfers_after

log = logging.getLogger(__name__)

CHIPS_THAT_DONT_CONSUME_TRANSFERS = {"wildcard", "freehit"}


@dataclass
class SquadPick:
    player_id: int
    squad_position: int          # 1-15, FPL's own ordering (1-11 start, 12-15 bench)
    is_captain: bool
    is_vice_captain: bool
    multiplier: int              # 0 benched/unused, 1 normal, 2 captain, 3 triple captain
    purchase_price: float | None
    purchase_price_known: bool


@dataclass
class ManagerTeam:
    entry_id: int
    manager_name: str
    team_name: str
    season: int
    gw: int
    overall_rank: int | None
    overall_points: int | None
    bank: float
    team_value: float
    free_transfers: int
    free_transfers_is_estimated: bool
    active_chip: str | None
    chips_used: list[str]
    picks: list[SquadPick] = field(default_factory=list)

    def squad_player_ids(self) -> list[int]:
        return [p.player_id for p in self.picks]

    def captain_id(self) -> int | None:
        return next((p.player_id for p in self.picks if p.is_captain), None)

    def vice_captain_id(self) -> int | None:
        return next((p.player_id for p in self.picks if p.is_vice_captain), None)


def _purchase_price_map(client: FPLClient, entry_id: int,
                        current_player_ids: set[int]) -> dict[int, float]:
    """player_id -> most recent buy price (millions) from real transfer
    history. Only covers players who have actually been transferred in at
    some point this season; an original squad member from gameweek one is
    not present here, by design -- see the module docstring."""
    try:
        transfers = client.get(f"entry/{entry_id}/transfers/")
    except Exception as e:      # noqa: BLE001
        log.warning("could not fetch transfer history for entry %s: %s", entry_id, e)
        return {}
    if not isinstance(transfers, list):
        return {}
    # Keep the MOST RECENT buy of each currently-held player. Sorted
    # explicitly rather than trusting the endpoint's order: a player bought,
    # sold and bought back must be priced at the second purchase.
    transfers = sorted(transfers, key=lambda t: (t.get("event") or 0, str(t.get("time") or "")),
                       reverse=True)
    prices: dict[int, float] = {}
    for t in transfers:
        pid = int(t.get("element_in", -1))
        if pid in current_player_ids and pid not in prices:
            prices[pid] = float(t.get("element_in_cost", 0)) / 10.0
    return prices


def _replay_free_transfers(client: FPLClient, entry_id: int,
                           through_gw: int) -> tuple[int, bool, list[str]]:
    """Walk the season gameweek by gameweek to reconstruct the free-transfer
    balance, using the exact same rule the transfer planner itself uses.

    Returns (free_transfers, is_estimated, chips_played). `is_estimated` is
    True whenever the history is incomplete (e.g. the season has just
    started and there is only partial data, or a gameweek's chip status
    could not be read) -- the caller should treat the number as a reasonable
    default rather than a certainty in that case, same spirit as everywhere
    else in this project that flags confidence rather than hiding it.
    `chips_played` comes from the same response: /entry/{id}/ itself carries
    no chip list, only /history/ does.
    """
    try:
        history = client.get(f"entry/{entry_id}/history/")
    except Exception as e:      # noqa: BLE001
        log.warning("could not fetch gw history for entry %s: %s", entry_id, e)
        return 1, True, []

    current = history.get("current", []) if isinstance(history, dict) else []
    raw_chips = (history.get("chips") or []) if isinstance(history, dict) else []
    chips = {c.get("event"): c.get("name") for c in raw_chips}
    chips_played = [c.get("name") for c in raw_chips
                    if c.get("name") and (c.get("event") or 0) <= through_gw]

    ft = 1
    estimated = not current
    rows = sorted(current, key=lambda r: r.get("event", 0))
    # The first gameweek on record is the opening squad selection, not a
    # transfer gameweek. That is gw 1 only for managers who joined at the
    # start; someone who joined at gw 5 has no rows before it, and treating
    # gw 5 as a zero-transfer week wrongly banked an extra free transfer.
    first_gw = rows[0].get("event") if rows else None
    for row in rows:
        gw = row.get("event")
        if gw is None or gw > through_gw:
            break
        if gw == first_gw:
            continue      # opening squad selection, not a transfer gameweek
        chip = chips.get(gw)
        if chip in CHIPS_THAT_DONT_CONSUME_TRANSFERS:
            continue
        made = int(row.get("event_transfers", 0) or 0)
        ft = free_transfers_after(ft, made)
    return ft, estimated, chips_played


def get_my_team(entry_id: int, gw: int | None = None,
                season: int = CURRENT_SEASON,
                client: FPLClient | None = None) -> ManagerTeam:
    """Everything the transfer planner and squad view need for one real
    manager, assembled from FPL's own public endpoints. Raises on a
    network/lookup failure rather than returning a half-built team silently
    -- the caller (the API layer) is the right place to turn that into a
    4xx with a clear message, not this module papering over it."""
    client = client or FPLClient()

    entry = client.get(f"entry/{entry_id}/")
    if not isinstance(entry, dict) or "id" not in entry:
        raise ValueError(f"no such FPL entry: {entry_id}")

    target_gw = gw or entry.get("current_event") or 1
    picks_resp = client.get(f"entry/{entry_id}/event/{target_gw}/picks/")
    raw_picks = picks_resp.get("picks", []) if isinstance(picks_resp, dict) else []
    if not raw_picks:
        raise ValueError(f"no picks found for entry {entry_id}, gameweek {target_gw} "
                         f"-- the gameweek may not have started yet")

    entry_history = picks_resp.get("entry_history", {}) or {}
    current_ids = {int(p["element"]) for p in raw_picks}
    prices = _purchase_price_map(client, entry_id, current_ids)
    ft, ft_estimated, chips_played = _replay_free_transfers(client, entry_id, target_gw)

    picks = [SquadPick(
        player_id=int(p["element"]), squad_position=int(p["position"]),
        is_captain=bool(p.get("is_captain")), is_vice_captain=bool(p.get("is_vice_captain")),
        multiplier=int(p.get("multiplier", 1)),
        purchase_price=prices.get(int(p["element"])),
        purchase_price_known=int(p["element"]) in prices,
    ) for p in raw_picks]

    return ManagerTeam(
        entry_id=entry_id,
        manager_name=f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
        team_name=entry.get("name", ""), season=season, gw=target_gw,
        overall_rank=entry.get("summary_overall_rank"),
        overall_points=entry.get("summary_overall_points"),
        bank=float(entry_history.get("bank", 0)) / 10.0,
        team_value=float(entry_history.get("value", 0)) / 10.0,
        free_transfers=ft, free_transfers_is_estimated=ft_estimated,
        active_chip=picks_resp.get("active_chip"),
        chips_used=chips_played or (
            [c.get("name") for c in entry.get("chips_played", [])] if entry.get("chips_played") else []),
        picks=picks,
    )
