from fplagent.models.decision import decision_metrics


def test_start_probability_cannot_exceed_play_probability():
    m = decision_metrics({"xp_mean": 5, "xp_p10": 1, "xp_p50": 5, "xp_p90": 9, "p_play": .7, "p_start": .6, "p_sixty": .5, "now_cost": 5})
    assert m["start_security"] <= 1
    assert m["start_gap"] >= 0
