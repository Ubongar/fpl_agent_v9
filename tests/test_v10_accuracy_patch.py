from __future__ import annotations

import pandas as pd
import pytest

from fplagent.features.build import _price_rank_in_team
from fplagent.models.shares import estimate_shares
from fplagent.models.minutes import _chronological_train_cal_test_slices


def _assist_share_inputs():
    return {
        "position": "MID",
        "player_np_goals": [0.0, 0.0, 0.0, 0.0],
        "player_assists": [0.0, 0.0, 0.0, 0.0],
        "team_np_goals": [1.0, 1.0, 1.0, 1.0],
        "team_assists": [1.0, 1.0, 1.0, 1.0],
        "minutes_played": [90, 90, 90, 90],
    }


def test_xa_share_blend_moves_estimate_toward_underlying_chance_creation():
    common = _assist_share_inputs()
    observed_only = estimate_shares(**common)
    with_xa = estimate_shares(
        **common,
        player_expected_assists=[0.4, 0.4, 0.4, 0.4],
        team_expected_assists=[1.0, 1.0, 1.0, 1.0],
        xa_blend_weight=0.65,
    )
    # With 4 x 90-minute observations and the current 12-game shrinkage prior:
    # observed estimate = .1125; xA estimate = .2125; blended estimate = .1775.
    assert observed_only.assist_share == pytest.approx(0.1125)
    assert with_xa.assist_share == pytest.approx(0.1775)
    assert with_xa.assist_share > observed_only.assist_share
    assert with_xa.provenance["xa_blend_weight_used"] == pytest.approx(0.65)


def test_xa_blend_weight_zero_preserves_observed_assists_estimate():
    common = _assist_share_inputs()
    observed_only = estimate_shares(**common)
    weight_zero = estimate_shares(
        **common,
        player_expected_assists=[0.4, 0.4, 0.4, 0.4],
        team_expected_assists=[1.0, 1.0, 1.0, 1.0],
        xa_blend_weight=0.0,
    )
    assert weight_zero.assist_share == pytest.approx(observed_only.assist_share)


def test_price_rank_uses_snapshot_prices_not_current_player_table_values():
    # Current-table prices intentionally contradict historical snapshot prices.
    players = pd.DataFrame({
        "player_id": [1, 2],
        "team_id": [1, 1],
        "now_cost": [100, 50],
    })
    status_idx = pd.DataFrame({
        "team_id": [1, 1],
        "now_cost": [50, 100],
    }, index=pd.Index([1, 2], name="player_id"))

    ranks = _price_rank_in_team(players, status_idx)
    assert ranks == {1: 2.0, 2: 1.0}


def test_price_rank_returns_empty_when_no_point_in_time_snapshot_exists():
    players = pd.DataFrame({
        "player_id": [1, 2],
        "team_id": [1, 1],
        "now_cost": [100, 50],
    })
    assert _price_rank_in_team(players, None) == {}


def test_minutes_model_uses_disjoint_chronological_train_cal_test_blocks():
    train, calibration, test = _chronological_train_cal_test_slices(600, 0.2)
    train_idx = list(range(*train.indices(600)))
    calibration_idx = list(range(*calibration.indices(600)))
    test_idx = list(range(*test.indices(600)))

    assert len(train_idx) == 360
    assert len(calibration_idx) == 120
    assert len(test_idx) == 120
    assert max(train_idx) < min(calibration_idx)
    assert max(calibration_idx) < min(test_idx)
    assert set(train_idx).isdisjoint(calibration_idx)
    assert set(calibration_idx).isdisjoint(test_idx)
    assert set(train_idx).isdisjoint(test_idx)
