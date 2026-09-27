from fplagent.models.decision import decision_metrics


def test_decision_metrics_do_not_make_a_player_a_zero_ownership_differential():
    m = decision_metrics({"xp_mean": 6, "xp_p10": 2, "xp_p50": 5, "xp_p90": 10, "p_play": .9, "p_start": .8, "p_sixty": .7, "selected_by_percent": 20, "now_cost": 8})
    assert 0 < m["ownership"] <= 1
