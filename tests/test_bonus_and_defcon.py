from fplagent.models import bonus


def test_penalty_goal_bps_is_position_neutral():
    assert bonus.BPS_PENALTY_GOAL == 12
