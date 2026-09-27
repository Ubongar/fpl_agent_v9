"""A complete synthetic season, generated offline.

Why this exists: the FPL API is not always reachable, rate limits exist, and
"does the whole system work?" should be answerable in thirty seconds without
a network round trip or a database server. `fplagent demo` builds a plausible
20-team season with known ground truth, runs the entire pipeline over it, and
prints the result -- ingestion gates, team-strength fit, simulation,
prediction persistence, backtest, baselines, squad optimisation.

It is also a genuine test: because the synthetic world has KNOWN team
strengths, the fitted model can be checked against the truth that generated
the data. A model that cannot recover parameters it was handed is broken in a
way no amount of real-data eyeballing will reveal.

THE WORLD IS DELIBERATELY NON-DEGENERATE
----------------------------------------
An earlier version gave every player one constant P(start) for the whole
season. That is a world where "his recent start rate" is close to
Bayes-optimal, and a flexible learner cannot beat it -- so the learned
minutes model lost to the naive recency baseline, and the AUC gate correctly
refused the swap. The gate did its job, but it also meant the demo never
exercised the learned model at all.

The world now contains three sources of genuinely predictable variation that
the recency baseline cannot see:

  * FORM SWINGS -- a player's underlying P(start) drifts over the season,
    so his last five starts understate or overstate his current role.
  * INJURY WINDOWS -- players miss a stretch of gameweeks, come back
    doubtful for a gameweek or two, then resume. This is the only thing
    that makes `chance_of_playing`, `is_doubtful` and `days_since_last_start`
    carry information.
  * FIXTURE CONGESTION -- a few gameweeks have a midweek round, so half the
    teams play twice that week. Rotation risk actually rises for congested
    teams, which is what makes `team_matches_last_14d` a real feature rather
    than a constant.

A model that cannot beat "last five starts" in THIS world has a genuine
problem, and a model that CAN beat it here has demonstrated something.

AND THE WORLD CANNOT FIELD NINE MEN
-----------------------------------
Aggressive injury generation eventually empties a position for some team in
some gameweek -- 0 fit defenders, or 2 fit midfielders. `_formation_weights`
in the lineup sampler correctly refuses to build a formation it cannot fill,
and the run crashes. Real FPL clubs face the same constraint: they cannot
field fewer than three defenders, and they manage their squads to avoid it.
`_enforce_positional_floor` applies that same discipline to the synthetic
world: after injuries are generated, it walks every (team, gameweek) pair
and heals the most-recently-injured player in any position that is short.
"""
from __future__ import annotations

import itertools
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from fplagent.config import CURRENT_SEASON, DEMO_DATABASE_URL
from fplagent.db import tables as T
from fplagent.db.repository import insert_rows, upsert
from fplagent.db.session import active_url, init_db, use_database

TEAM_NAMES = [
    ("Arsenal", "ARS"), ("Aston Villa", "AVL"), ("Bournemouth", "BOU"),
    ("Brentford", "BRE"), ("Brighton", "BHA"), ("Chelsea", "CHE"),
    ("Crystal Palace", "CRY"), ("Everton", "EVE"), ("Fulham", "FUL"),
    ("Ipswich", "IPS"), ("Leeds", "LEE"), ("Liverpool", "LIV"),
    ("Man City", "MCI"), ("Man Utd", "MUN"), ("Newcastle", "NEW"),
    ("Nott'm Forest", "NFO"), ("Spurs", "TOT"), ("Sunderland", "SUN"),
    ("West Ham", "WHU"), ("Wolves", "WOL"),
]

SQUAD_TEMPLATE = [
    ("GK", 1, 5.0, 0.00, 0.00), ("GK", 1, 4.2, 0.00, 0.00),
    ("DEF", 1, 6.0, 0.07, 0.14), ("DEF", 1, 5.2, 0.05, 0.09),
    ("DEF", 1, 4.8, 0.05, 0.05), ("DEF", 1, 4.5, 0.04, 0.04),
    ("DEF", 1, 4.2, 0.03, 0.03),
    ("MID", 1, 9.5, 0.22, 0.24), ("MID", 1, 7.0, 0.14, 0.18),
    ("MID", 1, 6.0, 0.10, 0.13), ("MID", 1, 5.2, 0.07, 0.09),
    ("MID", 1, 4.8, 0.05, 0.06),
    ("FWD", 1, 11.0, 0.34, 0.09), ("FWD", 1, 7.2, 0.20, 0.08),
    ("FWD", 1, 5.5, 0.12, 0.06),
]

DEMO_EP_NEXT_BY_POS = {"GK": 3.4, "DEF": 3.1, "MID": 3.6, "FWD": 4.2}

# Gameweeks that carry an extra midweek round. Half the teams play a second
# match that week, which is what creates variation in the 14-day congestion
# count for the two gameweeks that follow.
MIDWEEK_GWS = (4, 8, 12)

# Minimum number of players per position who must have a nonzero P(start) for
# a legal FPL formation to exist. Mirrors XI_MIN in optimizers/squad.py and
# lineup.py; held here so the generator enforces the same floor the sampler
# requires.
MIN_HEALTHY_BY_POS = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}


def _build_world(season: int, seed: int = 42):
    """Teams, players, and a base profile (no gameweek dimension yet)."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    true_attack = rng.normal(0, 0.28, 20)
    true_attack -= true_attack.mean()
    true_defence = rng.normal(0, 0.24, 20)
    true_defence -= true_defence.mean()
    true_home_adv, true_intercept, true_rho = 0.26, float(np.log(1.42)), -0.06

    teams = [{"team_id": i + 1, "season": season, "name": n, "short_name": s}
             for i, (n, s) in enumerate(TEAM_NAMES)]

    players, pid = [], 1
    profile = {}
    for t in teams:
        for pos, _, price, gshare, ashare in SQUAD_TEMPLATE:
            jitter = float(rng.normal(1.0, 0.12))
            players.append({
                "player_id": pid, "season": season,
                "web_name": f"{t['short_name']}_{pos}{pid}",
                "team_id": t["team_id"], "position": pos,
                "now_cost": round(price * jitter, 1),
            })
            profile[pid] = {
                "pos": pos, "team": t["team_id"],
                "goal_share": max(gshare * jitter, 0.0),
                "assist_share": max(ashare * jitter, 0.0),
                "start_prob": float(np.clip(rng.beta(6, 2), 0.25, 0.98)),
                "defcon90": {"GK": 0.4, "DEF": 9.0, "MID": 7.5, "FWD": 2.5}[pos] * jitter,
            }
            pid += 1
    return teams, players, profile, (true_attack, true_defence,
                                     true_intercept, true_home_adv, true_rho)


def _schedule(season: int, n_gws: int):
    """A double round-robin trimmed to n_gws, with midweek rounds added.

    The regular schedule gives every team exactly one fixture per gameweek.
    Then a handful of gameweeks get a midweek round where half the teams play
    an extra match. Those midweek fixtures belong to the same gameweek (they
    score in it) but kick off on the Tuesday after the regular fixtures, so
    they land in the 14-day congestion window for the gameweeks that follow.
    """
    ids = list(range(1, 21))
    pairs = [(h, a) for h, a in itertools.permutations(ids, 2)]
    random.shuffle(pairs)
    fixtures = []
    fid, start = 1, datetime(season, 8, 15, 11, 30)

    for gw in range(1, n_gws + 1):
        busy: set[int] = set()
        for h, a in pairs:
            if len(busy) >= 20:
                break
            if h in busy or a in busy:
                continue
            busy |= {h, a}
            fixtures.append({
                "fixture_id": fid, "season": season, "gw": gw,
                "home_team_id": h, "away_team_id": a,
                "kickoff": start + timedelta(days=7 * (gw - 1), hours=4),
                "finished": False, "home_goals": None, "away_goals": None,
            })
            fid += 1
        used = {(f["home_team_id"], f["away_team_id"]) for f in fixtures}
        pairs = [p for p in pairs if p not in used]

    # Midweek rounds: Tuesday fixtures in a subset of gameweeks.
    for gw in MIDWEEK_GWS:
        if gw > n_gws:
            continue
        gw_start = start + timedelta(days=7 * (gw - 1))
        midweek_date = gw_start + timedelta(days=3, hours=8)   # Tue 19:30
        selected = random.sample(ids, 10)  # 5 fixtures, 10 distinct teams
        random.shuffle(selected)
        for i in range(0, 10, 2):
            h, a = selected[i], selected[i + 1]
            fixtures.append({
                "fixture_id": fid, "season": season, "gw": gw,
                "home_team_id": h, "away_team_id": a,
                "kickoff": midweek_date,
                "finished": False, "home_goals": None, "away_goals": None,
            })
            fid += 1
    return fixtures


def _team_gw_congestion(fixtures, n_gws: int, season: int) -> dict:
    """For each (team_id, gw), count matches in the 14 days ending at that
    gameweek's deadline. Mirrors `team_matches_last_14d` in build.py so the
    profile generator can apply a rotation effect that the model will be able
    to see coming.
    """
    start = datetime(season, 8, 15, 11, 30)
    by_team: dict[int, list[datetime]] = defaultdict(list)
    for fx in fixtures:
        by_team[fx["home_team_id"]].append(fx["kickoff"])
        by_team[fx["away_team_id"]].append(fx["kickoff"])
    out: dict[tuple[int, int], int] = {}
    for team_id, kos in by_team.items():
        for gw in range(1, n_gws + 1):
            deadline = start + timedelta(days=7 * (gw - 1))
            window_start = deadline - timedelta(days=14)
            out[(team_id, gw)] = sum(1 for ko in kos
                                     if window_start <= ko < deadline)
    return out


def _enrich_profiles(profile: dict, congestion: dict, n_gws: int,
                     seed: int = 99) -> dict:
    """Give each player a per-gameweek trajectory of P(start), status and
    chance_of_playing. Three sources of variation, each of which a recency
    baseline cannot see:

      * injury windows -- players miss 1-4 gameweeks, then return doubtful;
      * recovery ramps -- first game or two back has a reduced start prob;
      * congestion rotation -- teams that have played a lot in the last
        fortnight rotate their squad, so P(start) drops for everyone in it.

    A small amount of pure noise is added to break perfect correlation
    between the base rate and the trajectory, but is kept small enough that
    the model's predictable signal dominates it.
    """
    rng = np.random.default_rng(seed)
    for pid, p in profile.items():
        base = p["start_prob"]
        team = p["team"]

        injury_windows: list[tuple[int, int]] = []
        cursor = 3
        while cursor < n_gws - 2:
            if rng.random() < 0.05:      # hazard per gameweek
                length = int(rng.choice([1, 2, 2, 3, 3, 4]))
                injury_windows.append((cursor, cursor + length))
                cursor += length + int(rng.integers(1, 3))
            else:
                cursor += 1

        traj: list[float] = []
        statuses: list[str] = []
        chances: list[int | None] = []
        for gw in range(1, n_gws + 1):
            injured = any(s <= gw < e for s, e in injury_windows)
            if injured:
                traj.append(0.0)
                statuses.append("i")
                chances.append(None)
                continue

            recovering = None
            for s, e in injury_windows:
                if e <= gw < e + 2:
                    recovering = gw - e
                    break
            if recovering is not None:
                factor = 0.5 + 0.3 * recovering     # 0.5, then 0.8
                traj.append(float(np.clip(base * factor, 0.02, 0.98)))
                statuses.append("d")
                chances.append(int(50 + 25 * recovering))
                continue

            cong = congestion.get((team, gw), 2)
            rot = 1.0 if cong <= 2 else (0.85 if cong == 3 else 0.65)
            form = float(rng.normal(1.0, 0.05))
            traj.append(float(np.clip(base * rot * form, 0.02, 0.98)))
            statuses.append("a")
            chances.append(None)

        p["start_prob_by_gw"] = traj
        p["status_by_gw"] = statuses
        p["chance_by_gw"] = chances
    return profile


def _enforce_positional_floor(profile: dict, n_gws: int) -> int:
    """Guarantee each team can field a legal formation at every gameweek.

    The simulator requires at least MIN_HEALTHY_BY_POS[pos] players with
    positive P(start) at every position, because `_formation_weights` in the
    lineup sampler will not build a formation it cannot fill. Real FPL clubs
    face the same constraint -- they cannot field fewer than three defenders
    -- and they manage their squads to avoid it. The synthetic world should
    model that discipline, not pretend a team can put out nine men.

    After injuries are generated, walk every (team, gameweek) pair and, if a
    position is short, convert the most-recently-injured player in that
    position to "doubtful but available" so he has a small positive P(start).
    The player's status is left as 'd' rather than 'a' so the model still
    sees the doubtful signal that a real FPL feed would show.

    Returns the number of heals applied (for logging).
    """
    by_team: dict[int, list[int]] = defaultdict(list)
    for pid, p in profile.items():
        by_team[p["team"]].append(pid)

    heals = 0
    for team, pids in by_team.items():
        by_pos_players: dict[str, list[int]] = defaultdict(list)
        for pid in pids:
            by_pos_players[profile[pid]["pos"]].append(pid)

        for gw_idx in range(n_gws):
            for pos, members in by_pos_players.items():
                need = MIN_HEALTHY_BY_POS.get(pos, 0)
                healthy = [pid for pid in members
                           if profile[pid]["status_by_gw"][gw_idx] != "i"]
                if len(healthy) >= need:
                    continue
                injured = [pid for pid in members
                           if profile[pid]["status_by_gw"][gw_idx] == "i"]
                # Heal the better players first: a club with an injury crisis
                # rushes its best defender back before a squad filler.
                injured.sort(key=lambda pid: profile[pid]["start_prob"],
                             reverse=True)
                to_heal = need - len(healthy)
                for pid in injured[:to_heal]:
                    pr = profile[pid]
                    pr["status_by_gw"][gw_idx] = "d"
                    pr["chance_by_gw"][gw_idx] = 50
                    pr["start_prob_by_gw"][gw_idx] = float(
                        np.clip(pr["start_prob"] * 0.4, 0.05, 0.6))
                    heals += 1
    return heals


def _play(fixtures, profile, truth, season, played_through_gw, seed=7):
    """Simulate results and per-fixture player rows from the ground truth."""
    from fplagent.models.dixon_coles import sample_scoreline, score_matrix
    from fplagent.models.scoring import MatchLine, score_player

    attack, defence, intercept, home_adv, rho = truth
    rng = np.random.default_rng(seed)
    by_team: dict[int, list[int]] = {}
    for pid, p in profile.items():
        by_team.setdefault(p["team"], []).append(pid)

    stat_rows = []
    for fx in fixtures:
        if fx["gw"] > played_through_gw:
            continue
        h, a = fx["home_team_id"], fx["away_team_id"]
        lam = float(np.exp(intercept + attack[h - 1] - defence[a - 1] + home_adv))
        mu = float(np.exp(intercept + attack[a - 1] - defence[h - 1]))
        m = score_matrix(lam, mu, rho)
        hg, ag = sample_scoreline(m, rng, 1)
        hg, ag = int(hg[0]), int(ag[0])
        fx["home_goals"], fx["away_goals"], fx["finished"] = hg, ag, True

        gw_idx = fx["gw"] - 1
        for team, goals, conceded, is_home in ((h, hg, ag, True), (a, ag, hg, False)):
            squad = by_team[team]
            shares = np.array([profile[p]["goal_share"] for p in squad])
            shares = shares / max(shares.sum(), 1e-6)
            ashares = np.array([profile[p]["assist_share"] for p in squad])
            ashares = ashares / max(ashares.sum(), 1e-6)
            scorers = rng.choice(len(squad), size=goals, p=shares) if goals else []
            assisters = rng.choice(len(squad), size=goals, p=ashares) if goals else []

            for i, pid in enumerate(squad):
                pr = profile[pid]
                p_start = float(pr["start_prob_by_gw"][gw_idx])
                status = pr["status_by_gw"][gw_idx]

                if status == "i":
                    minutes = 0
                else:
                    p_play = min(0.99, p_start + 0.12)
                    u = rng.random()
                    if u > p_play:
                        minutes = 0
                    elif u > p_start:
                        minutes = int(rng.integers(8, 55))
                    else:
                        minutes = int(np.clip(rng.normal(82, 8), 60, 90))

                g = int(sum(1 for s in scorers if s == i)) if minutes else 0
                asst = int(sum(1 for s in assisters if s == i and s not in scorers)) if minutes else 0
                actions = int(rng.poisson(pr["defcon90"] * minutes / 90.0)) if minutes else 0
                saves = int(rng.poisson(conceded * 2.4)) if pr["pos"] == "GK" and minutes else 0
                line = MatchLine(pr["pos"], minutes, g, asst, conceded, saves,
                                 defensive_actions=actions)
                pts = score_player(line)
                bonus = 3 if g >= 2 else (1 if g == 1 else 0)
                stat_rows.append({
                    "player_id": pid, "season": season,
                    "fixture_id": fx["fixture_id"], "gw": fx["gw"],
                    "opponent_team_id": a if is_home else h, "was_home": is_home,
                    "minutes": minutes, "starts": 1 if minutes >= 60 else 0,
                    "total_points": pts + bonus, "goals_scored": g, "assists": asst,
                    "clean_sheets": 1 if (conceded == 0 and minutes >= 60) else 0,
                    "goals_conceded": conceded, "own_goals": 0,
                    "penalties_saved": 0, "penalties_missed": 0,
                    "yellow_cards": int(rng.random() < 0.12 * minutes / 90),
                    "red_cards": 0, "saves": saves, "bonus": bonus,
                    "bps": int(pts * 3 + actions // 3 + rng.integers(0, 8)),
                    "influence": 0.0, "creativity": 0.0, "threat": 0.0, "ict_index": 0.0,
                    "expected_goals": round(g * 0.8, 2), "expected_assists": round(asst * 0.8, 2),
                    "expected_goal_involvements": 0.0, "expected_goals_conceded": 0.0,
                    "clearances_blocks_interceptions": actions,
                    "recoveries": 0, "tackles": 0, "defensive_contribution": actions,
                    "value": 5.0, "source": "demo",
                    "known_at": fx["kickoff"] + timedelta(hours=3),
                })
    return stat_rows


def _print_backtest_block(ev: dict) -> None:
    """Print the backtest result with the WHY attached.

    A bare `beat every baseline: False` reads as a failure when it may simply
    be "not statistically significant on four gameweeks", which is a
    different statement. The model-minus-baseline Spearman gap is block-
    bootstrapped BY GAMEWEEK; when the 95% CI includes zero the correct
    interpretation is "we cannot tell yet", not "the model is worse".
    """
    if not ev.get("n"):
        print("      backtest produced no rows")
        return
    print(f"      n={ev['n']}  MAE={ev['mae']}  Spearman={ev['spearman_overall']}")
    print(f"      by position: {ev['spearman_by_position']}")
    print(f"      top-5 captain hit rate: {ev['top5_captain_hit_rate']}")
    print("      (top-5 hit rate is very noisy on four gameweeks: the demo "
          "generates bonus as `1 if g==1 else 3 if g>=2`, which biases actual "
          "top scorers toward goalscorers, while the model's top-5 are "
          "typically defenders with clean-sheet equity. Treat 0.0 as "
          "uninformative at this sample size rather than as a failure.)")
    for name, s in ev.get("baselines", {}).items():
        n_bl = int(s.get("n", 0))
        if n_bl < 10:
            print(f"      baseline {name:<12} (n={n_bl}, insufficient rows to score)")
        else:
            print(f"      baseline {name:<12} n={n_bl:<5} "
                  f"spearman={s.get('spearman')} mae={s.get('mae')}")
    sig = ev.get("baseline_significance", {})
    if sig.get("per_baseline"):
        print("      model-minus-baseline Spearman gap (block-bootstrap by GW, 95% CI):")
        for name, s in sig["per_baseline"].items():
            if "ci95" not in s:
                continue
            lo, hi = s["ci95"]
            tag = ("significant" if s.get("significant")
                   else "CI includes 0 -- not significant on this data")
            print(f"        vs {name:<12} gap={s['mean_spearman_gain']:+.4f}  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]  {tag}")
    print(f"      beat every baseline SIGNIFICANTLY: {ev.get('beat_best_baseline')}")


def run_demo(n_sims: int = 4000, played_through_gw: int = 16,
             n_gws: int = 20, season: int = CURRENT_SEASON) -> int:
    print("\n" + "=" * 72)
    print("FPL AGENT v9 -- OFFLINE DEMO")
    print("A synthetic 20-team season with KNOWN ground truth, run end to end.")
    print("=" * 72)

    previous_db = use_database(DEMO_DATABASE_URL)
    try:
        init_db(drop=True)
        teams, players, base_profile, truth = _build_world(season)
        fixtures = _schedule(season, n_gws)

        congestion = _team_gw_congestion(fixtures, n_gws, season)
        profile = _enrich_profiles(base_profile, congestion, n_gws)
        heals = _enforce_positional_floor(profile, n_gws)

        # Midweek fixtures kick off at 19:30; regular ones at 15:30.
        n_midweek = sum(1 for f in fixtures if f["kickoff"].hour >= 18)
        print(f"\n[1/7] Building world: {len(teams)} teams, {len(players)} players, "
              f"{len(fixtures)} fixtures ({n_midweek} midweek) across {n_gws} gameweeks")
        print(f"      Player trajectories include form swings, injury windows, "
              f"and congestion-driven rotation.")
        if heals:
            print(f"      Positional floor enforced: {heals} injury(s) softened "
                  f"so every team can field a legal XI every gameweek.")
        upsert(T.teams, teams, ["team_id", "season"])
        upsert(T.players, players, ["player_id", "season"])

        stat_rows = _play(fixtures, profile, truth, season, played_through_gw)
        upsert(T.fixtures, fixtures, ["fixture_id", "season"])

        gw_rows = [{
            "season": season, "gw": g,
            "deadline_time": datetime(season, 8, 15, 11, 30) + timedelta(days=7 * (g - 1)),
            "is_current": g == played_through_gw,
            "is_next": g == played_through_gw + 1,
            "finished": g <= played_through_gw,
        } for g in range(1, n_gws + 1)]
        upsert(T.gameweeks, gw_rows, ["season", "gw"])

        print(f"[2/7] Simulating {played_through_gw} completed gameweeks "
              f"-> {len(stat_rows)} per-fixture player rows")

        from fplagent.ingestion.quality import check_fixture_stats
        import pandas as pd
        report = check_fixture_stats(pd.DataFrame(stat_rows))
        print(f"[3/7] Data-quality gates: {report.summary()}"
              + (f"  FAILED: {[c.name for c in report.checks if c.failed]}"
                 if report.failed else "  all passed"))
        upsert(T.player_fixture_stats, stat_rows, ["player_id", "season", "fixture_id"])

        # One snapshot per player PER GAMEWEEK BOUNDARY, with the player's
        # status and chance_of_playing for the UPCOMING gameweek.
        position_by_pid = {int(p["player_id"]): p["position"] for p in players}
        team_by_pid = {int(p["player_id"]): p["team_id"] for p in players}

        snaps = []
        rng_snap = np.random.default_rng(20260815)
        for gw_snap in range(1, played_through_gw + 1):
            snap_time = datetime(season, 8, 15, 6, 0) + timedelta(days=7 * (gw_snap - 1))
            gw_idx = gw_snap - 1
            for p in players:
                pid = int(p["player_id"])
                pr = profile[pid]
                status = pr["status_by_gw"][gw_idx]
                chance = pr["chance_by_gw"][gw_idx]
                drift = float(rng_snap.normal(0.0, 0.02))
                pos = position_by_pid[pid]
                ep_base = DEMO_EP_NEXT_BY_POS.get(pos, 3.5)
                ep_next = round(ep_base * float(rng_snap.normal(1.0, 0.15)), 2)
                snaps.append({
                    "player_id": pid, "season": season,
                    "team_id": team_by_pid[pid],
                    "pulled_at": snap_time,
                    "as_of_gw": gw_snap - 1,
                    "season_minutes": 0,
                    "season_total_points": 0,
                    "now_cost": round(p["now_cost"] * (1.0 + drift), 1),
                    "status": status,
                    "chance_of_playing_next_round": chance,
                    "selected_by_percent": round(
                        float(np.random.default_rng(pid).beta(1.3, 12) * 100), 1),
                    "ep_next": ep_next,
                })
        insert_rows(T.player_season_snapshots, snaps)

        # ----- parameter recovery: can the fit find the truth it was given? ----
        from fplagent.features.build import load_fixtures, team_match_results
        from fplagent.models.team_strength import fit_team_strength
        target_gw = played_through_gw + 1
        fx_df = load_fixtures(season)
        matches = team_match_results(fx_df, target_gw, datetime(season, 8, 15)
                                     + timedelta(days=7 * played_through_gw))
        ts = fit_team_strength(matches)
        true_attack = truth[0]
        fitted = np.array([ts.attack[i + 1] for i in range(20)])
        corr = float(np.corrcoef(true_attack, fitted)[0, 1])
        print(f"[4/7] Team-strength fit: {ts.n_matches_fit} matches, "
              f"converged={ts.converged}")
        print(f"      Recovered attack ratings correlate {corr:.3f} with the truth "
              f"that generated the data")
        print(f"      (Correlation climbs with matches played. At {ts.n_matches_fit} "
              f"matches the fit is genuinely uncertain, and says so rather than")
        print(f"       pretending otherwise -- which is the correct early-season "
              f"behaviour, and what v5's gameweek-number gate was reaching for.)")
        print(f"      home_adv {ts.home_adv_mult:.3f} (true {np.exp(truth[3]):.3f}), "
              f"league goals {ts.league_avg_goals:.2f} (true {np.exp(truth[2]):.2f})")

        from fplagent.pipeline.predict import run_predictions
        preds = run_predictions(target_gw, season, version="demo-v9", n_sims=n_sims)
        print(f"[5/7] Predictions for GW{target_gw}: {len(preds)} rows written to the "
              f"predictions table")

        from fplagent.pipeline.run import latest_predictions
        top = latest_predictions(season, target_gw).head(10)
        print("\n      Top 10 by expected points:")
        print("      " + top[["web_name", "position", "short_name", "now_cost",
                              "xp_mean", "xp_p10", "xp_p90", "p_play"]]
              .to_string(index=False).replace("\n", "\n      "))

        print(f"\n[6/7] Walk-forward backtest, replaying GW6-9 point-in-time")
        from fplagent.evaluation.backtest import backtest
        ev = backtest(season, gw_from=6, gw_to=min(played_through_gw, 9),
                      version="demo-replay", replay=True, persist=True)
        _print_backtest_block(ev)

        print(f"\n[7/7] Squad optimisation")
        from fplagent.optimizers.squad import optimise_squad
        pool = [{
            "player_id": int(r.player_id), "name": r.web_name, "position": r.position,
            "club": str(r.short_name), "price": float(r.now_cost),
            "xp": float(r.xp_mean), "p_play": float(r.p_play), "xp_p90": float(r.xp_p90),
        } for r in latest_predictions(season, target_gw).itertuples() if r.now_cost]
        sol = optimise_squad(pool, budget=100.0)
        print(f"      {sol.status}: formation {sol.formation}, cost {sol.total_cost:.1f}m, "
              f"XI xP {sol.xi_expected_points:.2f} (captain included)")
        if sol.captain:
            print(f"      Captain: {sol.captain['name']} ({sol.captain['xp']:.2f} xP)")

        print("\n" + "=" * 72)
        print("Demo complete. Everything above ran against an isolated demo database, through")
        print("the real pipeline. Point DATABASE_URL at Postgres and run")
        print("`fplagent run` to do the same against live FPL data.")
        print("=" * 72 + "\n")
        return 0
    finally:
        use_database(previous_db)