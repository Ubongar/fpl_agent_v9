

def test_ownership_falls_back_to_selected_by_when_real_eo_missing():
    from fplagent.models.decision import decision_metrics

    row = {"xp_mean": 5, "xp_p10": 2, "xp_p50": 5, "xp_p90": 9,
           "p_play": 0.9, "p_start": 0.7, "p_sixty": 0.6,
           "now_cost": 7.0, "selected_by_percent": 20.0, "eo": None}
    metrics = decision_metrics(row)
    assert metrics["ownership"] == 0.2
    assert metrics["ownership_adjusted_upside"] == 3.2


def test_decision_metrics_exposes_transparent_comparison_dimensions():
    from fplagent.models.decision import decision_metrics

    out = decision_metrics({
        "xp_mean": 7.0,
        "xp_se": 0.25,
        "xp_p10": 3.0,
        "xp_p25": 4.5,
        "xp_p50": 6.5,
        "xp_p75": 8.5,
        "xp_p90": 12.0,
        "p_play": 0.9,
        "p_start": 0.75,
        "p_sixty": 0.65,
        "selected_by_percent": 0.2,
        "now_cost": 8.0,
    })
    assert out["quartile_range_points"] == 4.0
    assert out["start_gap"] == 0.15
    assert 0 <= out["bench_risk"] <= 1
    assert out["floor_value_per_million"] > 0
    assert out["ceiling_value_per_million"] > out["floor_value_per_million"]


def test_decision_metric_invariants_for_comparison():
    from fplagent.models.decision import decision_metrics

    out = decision_metrics({
        "xp_mean": 5.0, "xp_se": 0.5,
        "xp_p10": 1.0, "xp_p25": 2.0, "xp_p50": 4.0,
        "xp_p75": 6.0, "xp_p90": 10.0,
        "p_play": 0.8, "p_start": 0.6, "p_sixty": 0.55,
        "selected_by_percent": 12.0, "now_cost": 6.5,
    })
    assert 0 <= out["start_gap"] <= 1
    assert 0 <= out["bench_risk"] <= 1
    assert out["quartile_range_points"] == 4.0
    assert out["floor_value_per_million"] < out["ceiling_value_per_million"]
