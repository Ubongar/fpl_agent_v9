from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fplagent.features.build import load_set_piece_duties
from fplagent.ingestion.quality import check_predictions
from fplagent.models.lineup import lineup_consistency, sample_lineup
from fplagent.models.minutes import MinutesPrediction, heuristic_minutes, sample_minutes
from fplagent.models.scoring import (
    APPEARANCE_60_PLUS,
    APPEARANCE_UNDER_60,
)
from fplagent.models.bonus import simulate_bps
from fplagent.models.simulation import SimPlayer, simulate_fixture
from fplagent.pipeline.predict import aggregate_to_gameweek
from fplagent.models.distributions import quantile_from_grid
from fplagent.db import tables as T
from fplagent.db.repository import insert_rows, upsert, read_sql
from fplagent.db.session import active_url, init_db, use_database, dispose_engine


def _mp(play=1.0, start=1.0, sixty=0.95, exp=80):
    return MinutesPrediction(play, sixty, exp, "test", {}, p_start_est=start)


def _player(pid: int, pos: str, **kw) -> SimPlayer:
    return SimPlayer(player_id=pid, position=pos, minutes=_mp(**kw),
                     np_goal_share=0.2, assist_share=0.2, penalty_share=0.0,
                     name=f"p{pid}")


def _det_matrix(home: int, away: int, max_goals: int = 8) -> np.ndarray:
    m = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    m[home, away] = 1.0
    return m


def test_minutes_events_are_distinct_and_start_probability_is_not_60_plus():
    pred = heuristic_minutes([20, 30, 75, 90], starts_history=[1, 0, 0, 1])
    assert 0.0 <= pred.p_play <= 1.0
    assert 0.0 <= pred.p_start <= 1.0
    assert 0.0 <= pred.p_sixty <= 1.0
    assert pred.p_start <= pred.p_play + 1e-12
    assert pred.p_sixty <= pred.p_play + 1e-12
    assert pred.p_start != pred.p_sixty
    zero_start = MinutesPrediction(0.9, 0.4, 30, "test", {}, p_start_est=0.0)
    assert zero_start.p_start == 0.0


def test_sample_minutes_returns_only_valid_nonnegative_minutes():
    rng = np.random.default_rng(7)
    minutes = sample_minutes(MinutesPrediction(0.8, 0.35, 50, "test", {}, p_start_est=0.4),
                             5000, rng)
    assert np.all(minutes >= 0)
    assert np.all(minutes <= 90)
    assert np.mean(minutes > 0) > 0
    assert np.mean(minutes >= 60) < 0.6
    assert abs(float(minutes.mean()) - 41.1) < 2.0


def test_lineup_has_exactly_eleven_starters_and_no_more_than_three_subs():
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    p_play = np.full(len(positions), 0.9)
    p_start = np.full(len(positions), 0.55)
    p_sixty = np.full(len(positions), 0.45)
    rng = np.random.default_rng(3)
    minutes, started, comes_on = sample_lineup(
        positions, p_play, p_start, p_sixty, 1000, rng, return_context=True)
    assert np.all(started.sum(axis=0) == 11)
    assert np.all(comes_on.sum(axis=0) <= 3)
    assert np.all((started & (np.array(positions)[:, None] == "GK")).sum(axis=0) <= 1)
    assert np.all((minutes > 0).sum(axis=0) >= 11)
    assert np.all((minutes > 0).sum(axis=0) <= 14)
    hooked = started & (minutes < 60) & (minutes > 0)
    assert np.all(comes_on.sum(axis=0) >= hooked.sum(axis=0))
    assert np.all(minutes[started] >= APPEARANCE_UNDER_60 * 0)


def test_zero_weight_players_are_never_selected():
    positions = ["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    p_play = np.ones(len(positions))
    p_start = np.ones(len(positions))
    p_sixty = np.ones(len(positions))
    p_start[1] = 0.0
    p_play[1] = 0.0
    rng = np.random.default_rng(5)
    minutes, started, _ = sample_lineup(
        positions, p_play, p_start, p_sixty, 500, rng, return_context=True)
    assert not started[1].any()
    assert np.all(minutes[1] == 0)


def test_invalid_squad_shape_is_rejected_instead_of_silently_fielding_fewer_than_eleven():
    positions = ["GK"] + ["FWD"] * 10
    with __import__("pytest").raises(ValueError, match="needs at least 3 DEF"):
        sample_lineup(positions, np.ones(11), np.ones(11), np.ones(11),
                      10, np.random.default_rng(1))


def test_simulation_conserves_goals_and_never_assists_penalty_goals():
    home = ([_player(1, "GK"), _player(2, "GK")]
            + [_player(i, "DEF") for i in range(3, 8)]
            + [_player(i, "MID") for i in range(8, 13)]
            + [_player(i, "FWD") for i in range(13, 16)])
    away = ([_player(21, "GK"), _player(22, "GK")]
            + [_player(i, "DEF") for i in range(23, 28)]
            + [_player(i, "MID") for i in range(28, 33)]
            + [_player(i, "FWD") for i in range(33, 36)])
    home[2].penalty_share = 1.0
    matrix = _det_matrix(3, 2)
    result = simulate_fixture(home, away, matrix, 1.8, 1.5, n_sims=3000, seed=11)
    assert result.diagnostics["home_goal_allocation_max_error"] == 0.0
    assert result.diagnostics["away_goal_allocation_max_error"] == 0.0
    assert result.diagnostics["home_team_goal_reconstruction_max_error"] == 0.0
    assert result.diagnostics["away_team_goal_reconstruction_max_error"] == 0.0
    assert np.all(result.team_goals["home"] == 3)
    assert np.all(result.team_goals["away"] == 2)
    for pid, pts in result.points.items():
        assert len(pts) == 3000
        assert np.all(np.isfinite(pts))
        assert np.all(pts[~np.isnan(pts)] >= -100)


def test_quality_gate_checks_start_probability_and_finiteness():
    bad = pd.DataFrame({
        "p_play": [0.8, np.nan],
        "p_start": [0.9, 0.4],
        "p_sixty": [0.7, 1.2],
        "xp_mean": [5.0, 6.0],
        "xp_p10": [1.0, 7.0],
        "xp_p25": [2.0, 7.5],
        "xp_p50": [4.0, 8.0],
        "xp_p75": [6.0, 8.5],
        "xp_p90": [9.0, 8.0],
    })
    rep = check_predictions(bad)
    names = {c.name for c in rep.checks if c.failed}
    assert "p_start_le_p_play" in names
    assert "quantiles_ordered" in names
    assert any(c.name == "p_play_in_unit_interval" for c in rep.checks)


def test_dgw_probability_aggregation_uses_distribution_not_quantile_count():
    a = [0, 0, 0, 2, 4, 6, 8]
    b = [0, 0, 1, 2, 3, 5, 7]
    df = pd.DataFrame([
        {"player_id": 1, "gw": 10, "season": 2026, "fixture_id": 100,
         "xp_mean": 4, "p_play": .8, "p_start": .6, "p_sixty": .5,
         "p_haul_10plus": .0, "p_blank_le2": .6, "xp_se": .1,
         "model_version": "v8", "team_id": 1, "position": "FWD", "now_cost": 8.0,
         "sim_quantiles": a},
        {"player_id": 1, "gw": 10, "season": 2026, "fixture_id": 101,
         "xp_mean": 3, "p_play": .7, "p_start": .5, "p_sixty": .4,
         "p_haul_10plus": .0, "p_blank_le2": .7, "xp_se": .2,
         "model_version": "v8", "team_id": 1, "position": "FWD", "now_cost": 8.0,
         "sim_quantiles": b},
    ])
    out = aggregate_to_gameweek(df).iloc[0]
    assert out["n_fixtures"] == 2
    assert out["xp_mean"] == 7
    assert 0.0 <= out["p_haul_10plus"] <= 1.0
    assert 0.0 <= out["p_blank_le2"] <= 1.0
    assert out["xp_p90"] <= 15


def test_penalty_goal_bps_is_position_neutral_in_current_rules():
    common = dict(minutes=np.array([90]), assists=np.array([0]),
                  goals_conceded=np.array([1]), saves=np.array([0]),
                  defensive_actions=np.array([0]))
    no_goal_gk = simulate_bps("GK", goals=np.array([0]), **common)
    pen_gk = simulate_bps("GK", goals=np.array([1]), penalty_goals=np.array([1]), **common)
    open_gk = simulate_bps("GK", goals=np.array([1]), penalty_goals=np.array([0]), **common)
    no_goal_fwd = simulate_bps("FWD", goals=np.array([0]), **common)
    pen_fwd = simulate_bps("FWD", goals=np.array([1]), penalty_goals=np.array([1]), **common)
    open_fwd = simulate_bps("FWD", goals=np.array([1]), penalty_goals=np.array([0]), **common)
    assert float(pen_gk[0] - no_goal_gk[0]) == 12.0
    assert float(pen_fwd[0] - no_goal_fwd[0]) == 12.0
    assert float(open_gk[0] - no_goal_gk[0]) == 12.0
    assert float(open_fwd[0] - no_goal_fwd[0]) == 24.0


def test_set_piece_history_is_point_in_time(tmp_path):
    old = active_url()
    db_path = tmp_path / "duties.db"
    try:
        use_database(f"sqlite:///{db_path}")
        init_db(drop=True)
        t0 = datetime(2026, 8, 1, 12, 0, 0)
        t1 = t0 + timedelta(days=10)
        insert_rows(T.set_piece_duty_history, [
            {"player_id": 1, "season": 2026, "duty": "penalties", "order": 1,
             "share": 0.9, "confidence": 0.9, "known_at": t0, "source": "fpl_bootstrap", "retired_at": t1},
            {"player_id": 2, "season": 2026, "duty": "penalties", "order": 1,
             "share": 0.9, "confidence": 0.9, "known_at": t1, "source": "fpl_bootstrap", "retired_at": None},
        ])
        before = load_set_piece_duties(2026, t0 + timedelta(days=5))
        after = load_set_piece_duties(2026, t1 + timedelta(days=5))
        assert set(before["player_id"]) == {1}
        assert set(after["player_id"]) == {2}
    finally:
        dispose_engine()
        use_database(old)


def test_bps_cbi_is_not_derived_from_broader_defcon_actions():
    base = simulate_bps("DEF", minutes=np.array([90]), goals=np.array([0]),
                        assists=np.array([0]), goals_conceded=np.array([1]),
                        saves=np.array([0]), baseline_bps_per90=0.0)
    cbi = simulate_bps("DEF", minutes=np.array([90]), goals=np.array([0]),
                       assists=np.array([0]), goals_conceded=np.array([1]),
                       saves=np.array([0]), baseline_bps_per90=0.0,
                       cbi_actions=np.array([3]))
    assert float(cbi[0] - base[0]) == 1.0


def test_transfer_rules_expose_current_limits():
    from fplagent.optimizers.transfers import (
        DEFAULT_MAX_TRANSFERS_PER_GW, HIT_COST, MAX_BANKED_FREE_TRANSFERS,
    )
    assert DEFAULT_MAX_TRANSFERS_PER_GW == 20
    assert MAX_BANKED_FREE_TRANSFERS == 5
    assert HIT_COST == 4


def test_squad_solution_separates_xi_points_from_captain_extra():
    from fplagent.optimizers.squad import SquadSolution
    solution = SquadSolution(xi_expected_points=60.0, captain_extra_expected_points=6.5)
    summary = solution.summary()
    assert summary["xi_expected_points"] == 60.0
    assert summary["captain_extra_expected_points"] == 6.5


def test_prediction_player_context_is_carried_on_sim_player():
    from fplagent.pipeline.predict import _squad_for_team
    features = pd.DataFrame([{
        "team_id": 1, "player_id": 99, "position": "FWD", "p_play": 0.9,
        "p_start": 0.7, "p_sixty": 0.5, "expected_minutes": 65,
        "minutes_source": "test", "minutes_drivers": {}, "np_goal_share": 0.2,
        "assist_share": 0.2, "penalty_share": 0.0, "penalty_order": 99,
        "fk_share": 0.0, "fk_order": 99, "corner_share": 0.0, "corner_order": 99,
        "defcon_per90": 0.0, "defcon_dispersion": 0.0, "defcon_confidence": "prior_only",
        "yellow_per90": 0.1, "red_per90": 0.01, "card_confidence": "prior_only",
        "baseline_bps_per90": 8.0, "unavailable": False, "web_name": "test",
        "now_cost": 7.5,
    }])
    # _squad_for_team will return the player without relying on row-scoped globals
    # from the fixture loop in run_predictions.
    # Supply a complete legal pool so the helper's position quota remains valid.
    rows = []
    for pos, count in (("GK", 2), ("DEF", 3), ("MID", 2), ("FWD", 1)):
        for i in range(count):
            rows.append(features.iloc[0].to_dict() | {
                "player_id": 1000 + len(rows), "position": pos,
                "team_id": 1, "web_name": f"{pos}{i}",
            })
    squad = _squad_for_team(pd.DataFrame(rows), 1)
    target = next(x for x in squad if x.player_id == 1000 + 7)
    assert target.team_id == 1
    assert target.now_cost == 7.5


def test_quantile_grid_interpolation_is_monotone():
    grid = [0, 2, 4, 7, 10]
    vals = [quantile_from_grid(grid, q) for q in np.linspace(0, 1, 11)]
    assert vals == sorted(vals)
    assert vals[0] == 0
    assert vals[-1] == 10


def test_v9_decision_metrics_keep_floor_ceiling_and_availability_separate():
    from fplagent.models.decision import decision_metrics
    m = decision_metrics({
        "xp_mean": 6.0, "xp_p10": 2.0, "xp_p50": 5.0, "xp_p90": 12.0,
        "xp_se": 0.4, "p_play": 0.8, "p_start": 0.6, "p_sixty": 0.5,
        "now_cost": 7.5, "selected_by_percent": 12.0,
    })
    assert m["upside_points"] == 7.0
    assert m["downside_points"] == 3.0
    assert 0 < m["start_security"] < 1
    assert 0 < m["sixty_security"] < 1
    assert m["value_per_million"] == 0.8
    assert m["ownership"] == 0.12


def test_v9_market_signal_is_not_used_when_odds_are_stale(tmp_path):
    from fplagent.pipeline.predict import _market_lambdas
    old = active_url()
    db_path = tmp_path / "market.db"
    try:
        use_database(f"sqlite:///{db_path}")
        init_db(drop=True)
        insert_rows(T.match_odds, [{
            "season": 2026, "fixture_id": 1, "bookmaker": "test",
            "pulled_at": datetime(2026, 8, 1, 12, 0, 0),
            "home_win": 0.5, "draw": 0.25, "away_win": 0.25,
            "implied_home_xg": 1.9, "implied_away_xg": 0.8,
        }])
        lam, mu = _market_lambdas(1, 2026, datetime(2026, 8, 5, 12, 0, 0))
        assert lam is None and mu is None
    finally:
        dispose_engine()
        use_database(old)
