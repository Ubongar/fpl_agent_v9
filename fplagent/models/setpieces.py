"""Set-piece duty as a SHARE, conditional on who is on the pitch.

WHY THIS IS ITS OWN MODEL
-------------------------
v6 treated penalty duty as a binary fact: `load_set_piece_duties` handed the
declared first-choice taker a share of 1.0 and everybody else 0.0. Two things
are wrong with that, and both of them cost real points.

1. A TEAM HAS MORE THAN ONE TAKER.
   Almost every side has a second and often a third name on the list. The
   first-choice taker is not on the pitch for something like 20-25% of his
   team's penalties once you account for rotation, substitution, injury and
   the odd occasion he hands it over. Those penalties do get taken, by
   somebody, and that somebody is predictable.

2. WHO TAKES IT DEPENDS ON WHO IS PLAYING AT THE TIME.
   This is the part a static share cannot express. If the first choice is off
   the pitch, the duty does not evaporate; it falls to the next man on the
   list who IS on. So the right object is not a fixed per-player probability,
   it is a per-player PRIORITY that gets resolved against the eleven actually
   playing, separately in every simulation draw.

   The consequence is exactly the one that makes this worth modelling: if you
   own both the first and second choice taker, their expected points are
   different, and the difference is computable. The backup is worth more on a
   team whose primary is a rotation risk than on one whose primary plays 90
   minutes every week, and no single number attached to either player can say
   that.

HOW DUTY IS RESOLVED
--------------------
Each (team, duty) has an ordered list of players. Order 1 is first choice.
`ORDER_SHARE` gives the probability that each rank takes the set piece GIVEN
that everyone ahead of him is available -- it is not 1.0 for the first choice
even then, because takers do hand over, and because a declared list is a
statement of intent rather than a guarantee.

At simulation time `resolve_duty_shares` takes the raw shares and the
per-draw on-pitch mask and renormalises over the players actually playing.
The residual (nobody on our list is on) is left with the "other" slot rather
than forced onto someone, so a team whose entire declared list is off the
pitch does not have a penalty attributed to a centre-back.

WHERE THE DUTY LIST COMES FROM
------------------------------
Primary source is FPL's own `bootstrap-static`, which publishes
`penalties_order`, `corners_and_indirect_freekicks_order` and
`direct_freekicks_order` for every player. It is first-party, it updates on
every ingest, and crucially it is a statement of DUTY rather than a record of
past events -- so a team that has not yet been awarded a penalty this season
still has a known taker. That is the case v6 could not handle at all: with no
penalty awarded, `pens_team` was 0, the Beta posterior returned 0.0, and the
model concluded nobody at that club takes penalties.

FBref's PKatt is kept as an optional cross-check, because observed attempts
are evidence about what actually happens rather than what was declared, and
the two disagreeing is a useful signal that a duty has changed hands.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DUTIES = ("penalties", "direct_fk", "corners")

# P(this rank takes it | everyone ahead of him is on the pitch).
# The first choice does not get 1.0: takers hand over after a miss, when
# chasing a hat-trick, or when a specialist substitute is on. The residual
# mass that these do not account for goes to the "other" slot, which is the
# honest representation of "somebody not on our list took it".
ORDER_SHARE = {
    "penalties": {1: 0.86, 2: 0.70, 3: 0.55},
    "direct_fk": {1: 0.62, 2: 0.45, 3: 0.30},
    "corners":   {1: 0.55, 2: 0.40, 3: 0.25},
}
DEFAULT_ORDER_SHARE = 0.20      # ranks beyond 3rd

# How much of a team's goals arrive from each set-piece route. Used to turn a
# duty share into an expected-points contribution.
PENALTY_CONVERSION = 0.79       # PL penalty conversion rate
DIRECT_FK_GOAL_RATE = 0.065     # P(a direct free kick attempt is scored)

# A corner taker does not score from corners; he ASSISTS from them. This is
# the fraction of a team's assists that come from dead-ball deliveries, which
# is the mechanism by which a nailed corner taker on a tall side is worth
# more than his open-play creativity suggests.
SET_PIECE_ASSIST_FRACTION = 0.28


@dataclass
class DutyHolder:
    player_id: int
    order: int
    share: float
    confidence: float
    source: str = "fpl"


@dataclass
class TeamDuty:
    """One team's ordered list for one duty."""
    team_id: int
    duty: str
    holders: list[DutyHolder] = field(default_factory=list)

    def share_vector(self, player_ids: list[int]) -> np.ndarray:
        """Raw (pre-pitch-resolution) share for each player in `player_ids`."""
        by_id = {h.player_id: h.share for h in self.holders}
        return np.array([by_id.get(pid, 0.0) for pid in player_ids], dtype=float)


def order_to_share(duty: str, order: int | None) -> float:
    """Declared rank -> conditional probability of taking the set piece."""
    if order is None or order <= 0:
        return 0.0
    table = ORDER_SHARE.get(duty, ORDER_SHARE["penalties"])
    return float(table.get(int(order), DEFAULT_ORDER_SHARE))


def build_team_duty(team_id: int, duty: str,
                    orders: dict[int, int],
                    confidence: dict[int, float] | None = None,
                    source: str = "fpl") -> TeamDuty:
    """orders: {player_id: declared order, 1 == first choice}."""
    confidence = confidence or {}
    holders = [
        DutyHolder(player_id=int(pid), order=int(o),
                   share=order_to_share(duty, o),
                   confidence=float(confidence.get(pid, 0.8 if o == 1 else 0.6)),
                   source=source)
        for pid, o in sorted(orders.items(), key=lambda kv: kv[1])
        if o is not None and int(o) > 0
    ]
    return TeamDuty(team_id=team_id, duty=duty, holders=holders)


def resolve_duty_shares(raw_shares: np.ndarray, on_pitch: np.ndarray,
                        order: np.ndarray) -> np.ndarray:
    """Resolve a declared duty list against who is actually playing.

    raw_shares: (n_players,) conditional share from ORDER_SHARE, 0 for
        players with no declared duty.
    on_pitch:   (n_players, n_sims) boolean.
    order:      (n_players,) declared rank; a large number for non-takers.

    Returns (n_players, n_sims) probabilities that sum to AT MOST 1 down each
    column. The shortfall is deliberate and means "taken by someone not on
    our list", which is a real outcome and must not be forced onto a player.

    The resolution is sequential down the order, which is what makes this
    different from simply renormalising. Walking the list in rank order and
    giving each available taker his conditional share of the REMAINING
    probability reproduces the actual decision procedure at the ground: the
    first-choice taker takes it if he is on, otherwise it falls to the second
    choice, and so on. A flat renormalisation would instead hand the second
    choice a fixed slice even when the first choice is on the pitch, which
    understates the primary and overstates the backup in the common case.
    """
    raw_shares = np.asarray(raw_shares, dtype=float)
    on_pitch = np.asarray(on_pitch, dtype=bool)
    n_players, n_sims = on_pitch.shape
    out = np.zeros((n_players, n_sims), dtype=float)
    if n_players == 0:
        return out

    remaining = np.ones(n_sims, dtype=float)
    for idx in np.argsort(np.asarray(order, dtype=float)):
        s = raw_shares[idx]
        if s <= 0:
            continue
        take = on_pitch[idx] * s * remaining
        out[idx] = take
        remaining = remaining - take
    return out


def expected_penalty_goals(team_penalties_per_match: float,
                           duty_share: float,
                           conversion: float = PENALTY_CONVERSION) -> float:
    """Convenience for display and for the cold-start path."""
    return float(team_penalties_per_match * duty_share * conversion)


def estimate_penalties_scored(player_goals: np.ndarray,
                              player_pen_share: float,
                              team_pens_scored: np.ndarray) -> np.ndarray:
    """Best estimate of how many of a player's goals were penalties.

    FPL publishes `goals_scored` and `penalties_missed` but never penalties
    SCORED, which is why v6 ended up feeding total goals into an argument
    named `player_np_goals` and double-counting every taker. Attributing the
    team's penalty goals in proportion to declared duty is the honest
    reconstruction, and it is bounded by the player's own goals so it can
    never claim he scored more penalties than goals.
    """
    team_pens_scored = np.asarray(team_pens_scored, dtype=float)
    player_goals = np.asarray(player_goals, dtype=float)
    est = team_pens_scored * float(np.clip(player_pen_share, 0.0, 1.0))
    return np.minimum(est, player_goals)
