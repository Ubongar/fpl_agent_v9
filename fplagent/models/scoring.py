"""The complete FPL 2026/27 scoring function.

v5 modelled appearance, goals, assists, clean sheets and an (inverted) bonus
approximation. It modelled NONE of: goals conceded, saves, cards, own goals,
penalty misses, penalty saves, or defensive contribution points. Those are
not edge cases -- for a goalkeeper they are most of the scoring surface.

This module is deliberately a pure function of a realised match outcome, with
no probability anywhere. Everything probabilistic happens in simulation.py by
calling score_player() once per Monte Carlo draw. Keeping the rulebook
separate from the maths means the rules can be unit-tested against real FPL
scorelines, and a mid-season rule change is a one-file edit.
"""
from __future__ import annotations

from dataclasses import dataclass

POSITIONS = ("GK", "DEF", "MID", "FWD")

APPEARANCE_UNDER_60 = 1
APPEARANCE_60_PLUS = 2

GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}

# -1 for every 2 goals conceded, GK and DEF only. NOTE: unlike the clean
# sheet, this carries NO 60-minute requirement. It applies to goals conceded
# while the player was on the pitch, at any minute count. v6 gated it on 60
# minutes, which handed a free point to every substitute defender and every
# keeper hooked early, and overstated the floor of exactly the rotation-risk
# players whose floor you are trying to measure.
CONCEDED_PENALTY_POSITIONS = {"GK", "DEF"}
SAVE_POINTS_PER = 3          # +1 per 3 saves
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_POINTS = -2
OWN_GOAL_POINTS = -2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3

# Defensive contribution ("DefCon"), introduced 2025/26, unchanged for 2026/27.
# Defenders: >= 10 combined clearances, blocks, interceptions, tackles (CBIT).
# Midfielders and forwards: >= 12 of those PLUS ball recoveries (CBIRT).
# Worth 2 points, CAPPED at 2 per match -- 20 actions still pays 2, not 4.
# The cap is why this must be modelled as a threshold-crossing probability
# and never as a rate: E[actions] * points_per_action overvalues every
# high-volume centre-back in the league.
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
DEFCON_POINTS = 2
DEFCON_MINUTES_FLOOR = 0     # actions count from the first minute


@dataclass(frozen=True)
class MatchLine:
    """One player's realised contribution in one fixture."""
    position: str
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    team_goals_conceded: int = 0
    saves: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    own_goals: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    defensive_actions: int = 0
    bonus: int = 0


def appearance_points(minutes: int) -> int:
    """1 point for 1-59 minutes, 2 for 60+, 0 for an unused substitute.

    v5 used `p_start * 2`, which treats a guaranteed 70-minute starter and a
    guaranteed 20-minute cameo as the same linear fraction of 2 points. They
    are not: the cameo is worth exactly 1, deterministically.
    """
    if minutes <= 0:
        return 0
    return APPEARANCE_60_PLUS if minutes >= 60 else APPEARANCE_UNDER_60


def clean_sheet_points(position: str, minutes: int, goals_conceded: int) -> int:
    """60+ minutes AND nothing conceded while on the pitch. The 60-minute
    gate is real here, which is precisely why it must not be copied onto the
    conceded deduction below."""
    if minutes < 60 or goals_conceded > 0:
        return 0
    return CLEAN_SHEET_POINTS.get(position, 0)


def conceded_points(position: str, minutes: int, goals_conceded: int) -> int:
    """`goals_conceded` must be goals conceded WHILE THIS PLAYER WAS ON, not
    for the whole match. The caller owns that distinction; the rulebook just
    applies the arithmetic."""
    if position not in CONCEDED_PENALTY_POSITIONS or minutes <= 0:
        return 0
    return -(goals_conceded // 2)


def save_points(position: str, saves: int) -> int:
    if position != "GK":
        return 0
    return saves // SAVE_POINTS_PER


def defcon_points(position: str, defensive_actions: int) -> int:
    threshold = DEFCON_THRESHOLD.get(position)
    if threshold is None:
        return 0
    return DEFCON_POINTS if defensive_actions >= threshold else 0


def score_player(line: MatchLine) -> int:
    """Total FPL points for one player in one fixture. Exhaustive."""
    if line.minutes <= 0:
        return 0
    pts = appearance_points(line.minutes)
    pts += line.goals * GOAL_POINTS.get(line.position, 4)
    pts += line.assists * ASSIST_POINTS
    pts += clean_sheet_points(line.position, line.minutes, line.team_goals_conceded)
    pts += conceded_points(line.position, line.minutes, line.team_goals_conceded)
    pts += save_points(line.position, line.saves)
    pts += line.penalties_saved * PENALTY_SAVE_POINTS
    pts += line.penalties_missed * PENALTY_MISS_POINTS
    pts += line.own_goals * OWN_GOAL_POINTS
    pts += line.yellow_cards * YELLOW_CARD_POINTS
    pts += line.red_cards * RED_CARD_POINTS
    pts += defcon_points(line.position, line.defensive_actions)
    pts += line.bonus
    return int(pts)


def score_components(line: MatchLine) -> dict[str, int]:
    """The same total, itemised. Drives the UI drilldown and the LLM
    explanation -- an unexplained number is an untrusted number."""
    if line.minutes <= 0:
        return {"appearance": 0, "total": 0}
    comp = {
        "appearance": appearance_points(line.minutes),
        "goals": line.goals * GOAL_POINTS.get(line.position, 4),
        "assists": line.assists * ASSIST_POINTS,
        "clean_sheet": clean_sheet_points(line.position, line.minutes, line.team_goals_conceded),
        "goals_conceded": conceded_points(line.position, line.minutes, line.team_goals_conceded),
        "saves": save_points(line.position, line.saves),
        "penalties_saved": line.penalties_saved * PENALTY_SAVE_POINTS,
        "penalties_missed": line.penalties_missed * PENALTY_MISS_POINTS,
        "own_goals": line.own_goals * OWN_GOAL_POINTS,
        "cards": line.yellow_cards * YELLOW_CARD_POINTS + line.red_cards * RED_CARD_POINTS,
        "defensive_contribution": defcon_points(line.position, line.defensive_actions),
        "bonus": line.bonus,
    }
    comp["total"] = sum(comp.values())
    return comp
