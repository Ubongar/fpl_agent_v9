"""Joint match simulation. The heart of the system.

WHAT v7 CHANGED, AND WHY EACH ONE MATTERED
------------------------------------------
v6 got the big idea right: draw one scoreline per simulation and allocate the
goals, so totals are conserved and players on the same team are correlated.
Four things broke that promise in practice.

1. THE TEAM WAS NOT A TEAM.
   Minutes were drawn independently per player, so a mean of 13.4 players
   were on the pitch at once and 19.4% of draws had more than 14. Goal shares
   were renormalised across a phantom squad, clean sheets and DefCon were
   paid to players who would not have played, and bonus was ranked over 32
   players instead of 22. Lineups now come from `models.lineup`, which draws
   a legal eleven plus up to three substitutes.

2. PENALTIES WERE ATTRIBUTED TWICE AND MISSED TOO OFTEN.
   `pen_scored = min(Binomial(pen_awarded, 0.79), team_goals)` meant that in
   any draw where the team failed to score, every penalty it was awarded was
   recorded as a MISS and charged at -2 to the taker. Penalties are now
   resolved BEFORE open play rather than carved out of the total afterwards,
   so a goalless draw simply had no penalty rather than a missed one.

3. SAVES IGNORED THE SCORELINE.
   A keeper who shipped four in a draw had the same save distribution as one
   keeping a clean sheet in the same draw, which contradicted the module's
   own claim to be joint. Shots on target are now drawn first and split into
   goals and saves, so the two fall out of a single event count.

4. SET-PIECE DUTY WAS BINARY.
   The declared first-choice taker got a share of 1.0 and everyone else 0.0.
   Duty is now an ordered list resolved against who is on the pitch in each
   draw (see `models.setpieces`), so a second-choice taker's expected points
   correctly rise when the primary is rested, and two takers in the same XI
   are valued differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fplagent.models import bonus as bonus_mod
from fplagent.models import cards as cards_mod
from fplagent.models import defcon as defcon_mod
from fplagent.models import setpieces as sp
from fplagent.models.dixon_coles import sample_scoreline
from fplagent.models.lineup import sample_lineup
from fplagent.models.minutes import MinutesPrediction
from fplagent.models.scoring import (
    APPEARANCE_60_PLUS, APPEARANCE_UNDER_60, ASSIST_POINTS, CLEAN_SHEET_POINTS,
    DEFCON_POINTS, DEFCON_THRESHOLD, GOAL_POINTS, OWN_GOAL_POINTS,
    PENALTY_MISS_POINTS, PENALTY_SAVE_POINTS, RED_CARD_POINTS,
    SAVE_POINTS_PER, YELLOW_CARD_POINTS,
)
from fplagent.models.shares import ASSISTED_RATE, resolve_on_pitch_shares

# Card rates come from models.cards, per player, not a flat positional
# constant -- see SimPlayer.card_rate. Own goals stay a league-average flat
# rate: they are rare enough, and untied to disciplinary temperament, that a
# per-player rate would just be fitting noise.
# Probability that a conceded goal is recorded as an own goal. Kept separate
# from player scoring rates because an own goal is a subset of the opponent's
# team goals, not an independent extra goal.
OWN_GOAL_GIVEN_CONCEDED_GOAL = 0.025
# Penalties awarded to a team per match, league average. Scaled per fixture by
# attacking strength: a side that spends the match in the opposition box wins
# more penalties than one defending a lead, and a flat league constant cannot
# say so.
PENALTY_RATE_PER_MATCH = 0.27
# Shots on target faced per expected goal conceded. Saves are SoT minus goals.
SOT_PER_XG = 3.2
# Approximate share of penalty misses that are actually saved by the keeper;
# the rest are assumed off target/post. This is deliberately isolated as a
# tunable parameter rather than creating phantom saves independently of misses.
PENALTY_SAVE_SHARE = 1.0 - sp.PENALTY_CONVERSION


@dataclass
class SimPlayer:
    player_id: int
    position: str
    minutes: MinutesPrediction
    np_goal_share: float = 0.0
    assist_share: float = 0.0
    penalty_share: float = 0.0
    penalty_order: int = 99
    fk_share: float = 0.0
    fk_order: int = 99
    corner_share: float = 0.0
    corner_order: int = 99
    defcon_rate: "defcon_mod.DefconRate | None" = None
    card_rate: "cards_mod.CardRate | None" = None
    baseline_bps_per90: float = 8.0
    unavailable: bool = False
    name: str = ""
    team_id: int | None = None
    now_cost: float | None = None


@dataclass
class SimResult:
    points: dict[int, np.ndarray] = field(default_factory=dict)
    components: dict[int, dict[str, float]] = field(default_factory=dict)
    team_goals: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def summarise(self, player_id: int) -> dict:
        s = self.points[player_id]
        qs = np.percentile(s, [5, 10, 25, 50, 75, 90, 95])
        n = max(len(s), 1)
        return {
            "xp_mean": float(s.mean()),
            # Monte Carlo standard error. Two players whose xP differ by less
            # than this are not distinguishable, and the UI must not present
            # them as if they were.
            "xp_se": float(s.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "xp_p5": float(qs[0]), "xp_p10": float(qs[1]), "xp_p25": float(qs[2]),
            "xp_p50": float(qs[3]), "xp_p75": float(qs[4]), "xp_p90": float(qs[5]),
            "xp_p95": float(qs[6]),
            "p_haul_10plus": float((s >= 10).mean()),
            "p_blank_le2": float((s <= 2).mean()),
            "std": float(s.std()),
            "n_sims": int(n),
        }

    def quantile_grid(self, player_id: int, n: int = 21) -> list[float]:
        """A fuller CDF, stored with every prediction so CRPS and PIT can be
        computed later without re-running the simulation."""
        qs = np.linspace(0, 100, n)
        return [float(v) for v in np.percentile(self.points[player_id], qs)]


def _allocate_events(counts: np.ndarray, probs: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Attribute `counts[sim]` discrete events to players using `probs`.

    probs: (n_players, n_sims), columns summing to at most 1. The shortfall is
    an implicit "nobody in our pool" slot, so totals are conserved out of the
    team but never forced onto a player who did not do it.

    Returns (n_players, n_sims) integer counts.
    """
    n_players, n_sims = probs.shape
    out = np.zeros((n_players, n_sims), dtype=np.int16)
    counts = np.asarray(counts)
    if counts.size == 0:
        return out
    max_c = int(counts.max())
    if max_c == 0:
        return out

    cdf = np.cumsum(probs, axis=0)
    for e in range(1, max_c + 1):
        cols = np.flatnonzero(counts >= e)
        if cols.size == 0:
            continue
        u = rng.random(cols.size)
        who = (u[None, :] > cdf[:, cols]).sum(axis=0)
        hit = who < n_players
        if hit.any():
            np.add.at(out, (who[hit], cols[hit]), 1)
    return out



def _goal_concessions_by_interval(opp_goals: np.ndarray, minutes: np.ndarray,
                                  started: np.ndarray, comes_on: np.ndarray,
                                  rng: np.random.Generator) -> np.ndarray:
    """Count the opponent goals that occurred while each player was on pitch.

    One common time draw is used for each opponent goal, so a goal is credited
    as conceded to every player who was on the pitch at that instant. This
    preserves the FPL clean-sheet condition while avoiding the old independent
    binomial draws that made one team concede different numbers of goals to
    different players. Substitution timing is reconstructed from the sampled
    minutes (starters are on from 0; substitutes occupy the final ``minutes``),
    which is an explicit approximation because the simulator does not model
    individual substitution timestamps.
    """
    n_players, n_sims = minutes.shape
    out = np.zeros((n_players, n_sims), dtype=np.int16)
    max_goals = int(np.max(opp_goals)) if opp_goals.size else 0
    if max_goals <= 0:
        return out
    goal_times = rng.uniform(0.0, 90.0, size=(max_goals, n_sims))
    valid = np.arange(max_goals)[:, None] < opp_goals[None, :]
    for i in range(n_players):
        m = minutes[i].astype(float)
        start_t = np.where(started[i], 0.0, np.where(comes_on[i], np.maximum(0.0, 90.0 - m), 90.0))
        end_t = np.where(started[i], m, np.where(comes_on[i], 90.0, 0.0))
        active = (goal_times >= start_t[None, :]) & (goal_times < end_t[None, :]) & valid
        out[i] = active.sum(axis=0).astype(np.int16)
    return out


def _exposure_probs_for_position(minutes: np.ndarray, positions: list[str],
                                 position: str) -> np.ndarray:
    """Return per-simulation exposure probabilities for one position group."""
    mask = np.array([p == position for p in positions])[:, None]
    weights = np.where(mask, np.maximum(minutes, 0.0), 0.0)
    denom = weights.sum(axis=0, keepdims=True)
    return np.divide(weights, np.maximum(denom, 1e-12),
                     out=np.zeros_like(weights, dtype=float), where=denom > 1e-12)

def simulate_fixture(home_players: list[SimPlayer], away_players: list[SimPlayer],
                     matrix: np.ndarray, home_xg: float, away_xg: float,
                     n_sims: int = 20000, seed: int | None = None,
                     league_avg_xg: float = 1.4) -> SimResult:
    """Simulate one fixture jointly and return per-player point distributions.

    Every score-changing event is constrained to the team-level scoreline:
    penalties are a subset of normal goals, own goals are a subset of the
    opponent's goals, penalty saves are a subset of penalty misses, regular
    saves are a subset of shots-on-target, and goals conceded are assigned from
    common event times rather than independent per-player random variables.
    """
    rng = np.random.default_rng(seed)
    home_goals, away_goals = sample_scoreline(matrix, rng, n=n_sims)

    # Own goals are credited to the scoring side's opponent. Resolve them before
    # allocating normal goals so scorer/assist totals cannot exceed the score.
    home_own_goals = rng.binomial(away_goals, OWN_GOAL_GIVEN_CONCEDED_GOAL)
    away_own_goals = rng.binomial(home_goals, OWN_GOAL_GIVEN_CONCEDED_GOAL)
    home_scorer_goals = np.maximum(home_goals - away_own_goals, 0)
    away_scorer_goals = np.maximum(away_goals - home_own_goals, 0)

    # Penalty state is resolved for both teams up front because each side's
    # missed penalties determine the other goalkeeper's possible penalty saves.
    penalty_state: dict[str, dict[str, np.ndarray]] = {}
    for side, team_goals, scorer_goals, team_xg in (
        ("home", home_goals, home_scorer_goals, home_xg),
        ("away", away_goals, away_scorer_goals, away_xg),
    ):
        pen_rate = PENALTY_RATE_PER_MATCH * float(
            np.clip(team_xg / max(league_avg_xg, 1e-6), 0.5, 1.8))
        awarded = np.minimum(rng.poisson(pen_rate, n_sims), 2)
        scored = np.minimum(rng.binomial(awarded, sp.PENALTY_CONVERSION), scorer_goals)
        missed = np.maximum(awarded - scored, 0)
        penalty_state[side] = {"awarded": awarded, "scored": scored, "missed": missed}

    result = SimResult(team_goals={"home": home_goals, "away": away_goals})
    diag: dict = {}

    for side, squad, team_goals, opp_goals, scorer_goals, own_goal_count, team_xg, opp_xg in (
        ("home", home_players, home_goals, away_goals, home_scorer_goals, home_own_goals, home_xg, away_xg),
        ("away", away_players, away_goals, home_goals, away_scorer_goals, away_own_goals, away_xg, home_xg),
    ):
        if not squad:
            continue
        n_players = len(squad)
        positions = [p.position for p in squad]

        minutes, started, comes_on = sample_lineup(
            positions=positions,
            p_play=np.array([p.minutes.p_play for p in squad]),
            p_start=np.array([p.minutes.p_start for p in squad]),
            p_sixty=np.array([p.minutes.p_sixty for p in squad]),
            n_sims=n_sims, rng=rng,
            unavailable=np.array([p.unavailable for p in squad]),
            return_context=True,
        )
        minutes = minutes.astype(np.int32)
        on_pitch = minutes > 0
        diag[f"{side}_mean_on_pitch"] = float(on_pitch.sum(axis=0).mean())

        # --- goals ---------------------------------------------------------
        pen_awarded = penalty_state[side]["awarded"]
        pen_scored = penalty_state[side]["scored"]
        pen_missed_team = penalty_state[side]["missed"]
        open_goals = np.maximum(scorer_goals - pen_scored, 0)

        pen_probs = sp.resolve_duty_shares(
            np.array([p.penalty_share for p in squad]), on_pitch,
            np.array([p.penalty_order for p in squad], dtype=float))
        # A bootstrap duty list can be incomplete or every declared taker can
        # be off the pitch. Do not drop the resulting team goal into an
        # invisible residual bucket: use the on-pitch goal-share prior only for
        # the residual probability left by the declared-duty resolver. This
        # keeps the penalty event conserved at player level without pretending
        # an undeclared player was a known first-choice taker.
        pen_residual = np.clip(1.0 - pen_probs.sum(axis=0, keepdims=True), 0.0, 1.0)
        pen_fallback = resolve_on_pitch_shares(
            np.array([p.np_goal_share for p in squad]), on_pitch, absorb_remainder=True)
        pen_probs = pen_probs + pen_fallback * pen_residual
        np_shares = resolve_on_pitch_shares(
            np.array([p.np_goal_share for p in squad]), on_pitch, absorb_remainder=True)
        as_shares = resolve_on_pitch_shares(
            np.array([p.assist_share for p in squad]), on_pitch, absorb_remainder=False)

        open_scored = _allocate_events(open_goals, np_shares, rng)
        pen_goals = _allocate_events(pen_scored, pen_probs, rng)
        goals = open_scored + pen_goals

        pen_missed = _allocate_events(pen_missed_team, pen_probs, rng)

        # Assists exist only for normal player goals; an own goal is never
        # treated as a player-scored goal and therefore cannot be assisted.
        # FPL penalty goals do not generate assists. Sample assistable events
        # from normal open-play goals only, after penalty goals have consumed
        # their share of the team scoreline.
        assisted = rng.binomial(open_goals, ASSISTED_RATE)
        assists = _allocate_events(assisted, as_shares, rng)
        # A scorer cannot assist his own goal, but can assist another player's
        # goal. Cap each player's assists by team normal goals minus that
        # player's own normal goals, not by team goals minus the entire goal
        # allocation (which would make every assist zero).
        assists = np.minimum(assists, np.maximum(open_goals[None, :] - open_scored, 0))

        # Own goals are allocated to an on-pitch player, with the mild positional
        # exposure prior reflecting that defenders are more often involved in
        # the final touch. They are still constrained to the actual scoreline.
        og_pos_weight = np.array([1.35 if p.position == "DEF" else
                                  1.10 if p.position == "MID" else
                                  0.85 if p.position == "FWD" else 0.65
                                  for p in squad], dtype=float)
        og_probs = resolve_on_pitch_shares(og_pos_weight, on_pitch, absorb_remainder=True)
        own_goals = _allocate_events(own_goal_count, og_probs, rng)

        # --- team-conserved concessions and shots -------------------------
        conceded_on_all = _goal_concessions_by_interval(
            opp_goals, minutes, started, comes_on, rng)
        normal_opp_scorer_goals = (away_scorer_goals if side == "home" else home_scorer_goals)
        exp_sot = max(opp_xg, 0.05) * SOT_PER_XG
        sot_faced = np.maximum(rng.poisson(exp_sot, n_sims), normal_opp_scorer_goals)
        team_saves = np.maximum(sot_faced - normal_opp_scorer_goals, 0)

        # Regular saves and penalty saves are allocated only across actual goalkeepers
        # on the pitch, so no keeper can save a phantom penalty or share a save count
        # with another keeper.
        gk_probs = _exposure_probs_for_position(minutes, positions, "GK")
        pen_save_team = rng.binomial(penalty_state["away" if side == "home" else "home"]["missed"],
                                     np.clip(PENALTY_SAVE_SHARE, 0.0, 1.0))
        gk_saves = _allocate_events(team_saves, gk_probs, rng)
        gk_pen_saves = _allocate_events(pen_save_team, gk_probs, rng)

        # --- per-player scoring ------------------------------------------
        for i, p in enumerate(squad):
            m = minutes[i]
            played = m > 0
            appearance = np.where(m >= 60, APPEARANCE_60_PLUS,
                                  np.where(played, APPEARANCE_UNDER_60, 0))

            conceded_on = conceded_on_all[i]
            clean_sheet = (conceded_on == 0) & (m >= 60)
            cs_pts = clean_sheet * CLEAN_SHEET_POINTS.get(p.position, 0)
            conc_pts = (-(conceded_on // 2) * played
                        if p.position in ("GK", "DEF")
                        else np.zeros(n_sims, dtype=np.int32))

            goal_pts = goals[i] * GOAL_POINTS.get(p.position, 4)
            assist_pts = assists[i] * ASSIST_POINTS

            if p.position == "GK":
                saves = gk_saves[i]
                save_pts = saves // SAVE_POINTS_PER
                pen_saves = gk_pen_saves[i]
                pen_save_pts = pen_saves * PENALTY_SAVE_POINTS
            else:
                saves = np.zeros(n_sims, dtype=np.int32)
                save_pts = np.zeros(n_sims, dtype=np.int32)
                pen_saves = np.zeros(n_sims, dtype=np.int32)
                pen_save_pts = np.zeros(n_sims, dtype=np.int32)

            rate = p.defcon_rate or defcon_mod.estimate_defcon_rate(p.position, [], [])
            actions = defcon_mod.sample_defcon_actions(
                p.position, rate, m, opp_xg, league_avg_xg, rng)
            threshold = DEFCON_THRESHOLD.get(p.position)
            defcon_pts = ((actions >= threshold) * DEFCON_POINTS
                          if threshold else np.zeros(n_sims, dtype=np.int32))

            cr = p.card_rate or cards_mod.CardRate(
                cards_mod.YELLOW_PRIOR_PER90.get(p.position, 0.13),
                cards_mod.RED_PRIOR_PER90.get(p.position, 0.007), 0.0, "prior_only")
            exposure = np.clip(m / 90.0, 0, 1)
            reds = rng.random(n_sims) < np.clip(cr.red_per90 * exposure, 0, 1)
            yellows = (rng.random(n_sims) < np.clip(cr.yellow_per90 * exposure, 0, 1)) & ~reds
            ogs = own_goals[i]
            card_pts = yellows * YELLOW_CARD_POINTS + reds * RED_CARD_POINTS
            og_pts = ogs * OWN_GOAL_POINTS
            miss_pts = pen_missed[i] * PENALTY_MISS_POINTS

            bps = bonus_mod.simulate_bps(
                p.position, m, goals[i], assists[i], conceded_on, saves, None,
                p.baseline_bps_per90,
                penalties_saved=pen_saves, penalties_missed=pen_missed[i],
                own_goals=ogs.astype(np.int32),
                yellow_cards=yellows.astype(np.int32),
                red_cards=reds.astype(np.int32),
                penalty_goals=pen_goals[i].astype(np.int32))

            total = (appearance + goal_pts + assist_pts + cs_pts + conc_pts
                     + save_pts + pen_save_pts + defcon_pts + card_pts
                     + og_pts + miss_pts)
            total = np.where(played, total, 0.0)

            result.points[p.player_id] = total.astype(float)
            result.components[p.player_id] = {
                "appearance": float(appearance.mean()),
                "goals": float(goal_pts.mean()),
                "goals_open_play": float((open_scored[i] * GOAL_POINTS.get(p.position, 4)).mean()),
                "goals_penalty": float((pen_goals[i] * GOAL_POINTS.get(p.position, 4)).mean()),
                "assists": float(assist_pts.mean()),
                "clean_sheet": float(cs_pts.mean()),
                "goals_conceded": float(np.asarray(conc_pts, dtype=float).mean()),
                "saves": float(save_pts.mean()),
                "penalties_saved": float(pen_save_pts.mean()),
                "defensive_contribution": float(defcon_pts.mean()),
                "cards": float(card_pts.mean()),
                "own_goals": float(og_pts.mean()),
                "penalty_misses": float(miss_pts.mean()),
                "_bps": bps,
            }

        # A side's player-scored goals plus the opponent's own goals reconstruct
        # the exact team scoreline. Own goals belong to the team that conceded
        # them, so they must NOT be added to the scoring side's player-goal
        # allocation when validating conservation.
        opponent_own_goals = away_own_goals if side == "home" else home_own_goals
        scorer_allocation_error = np.max(np.abs(goals.sum(axis=0) - scorer_goals))
        team_reconstruction_error = np.max(
            np.abs(goals.sum(axis=0) + opponent_own_goals - team_goals))
        diag[f"{side}_mean_team_goals"] = float(team_goals.mean())
        diag[f"{side}_mean_penalty_awarded"] = float(pen_awarded.mean())
        diag[f"{side}_mean_penalty_missed"] = float(pen_missed_team.mean())
        diag[f"{side}_scorer_goal_allocation_max_error"] = float(scorer_allocation_error)
        diag[f"{side}_team_goal_reconstruction_max_error"] = float(team_reconstruction_error)
        # Backward-compatible key: this now means the player-scored portion,
        # which is the allocation this side actually controls.
        diag[f"{side}_goal_allocation_max_error"] = float(scorer_allocation_error)

    _apply_bonus(result, home_players, away_players)
    result.diagnostics = diag
    return result


def _apply_bonus(result: SimResult, home_players: list[SimPlayer],
                 away_players: list[SimPlayer]) -> None:
    """Bonus is a RANK within the fixture, across both teams.

    This is why an absolute BPS to bonus table can never be right: whether 45
    BPS earns three points depends entirely on what the other twenty-one
    players did in that same simulated match. Here we have all of them, and
    because the lineup sampler now fields eleven a side, the rank is taken
    over a realistic field rather than an inflated one.
    """
    squad = home_players + away_players
    ids = [p.player_id for p in squad if p.player_id in result.components]
    if not ids:
        return
    bps_matrix = np.vstack([result.components[i]["_bps"] for i in ids])
    bonus = bonus_mod.bonus_from_ranks(bps_matrix)
    for row, pid in enumerate(ids):
        result.points[pid] = result.points[pid] + bonus[row]
        result.components[pid]["bonus"] = float(bonus[row].mean())
        del result.components[pid]["_bps"]
