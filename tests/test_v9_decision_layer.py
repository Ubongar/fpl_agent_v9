from fplagent.models.decision import decision_metrics


def test_v91_exposes_comparison_dimensions():
    row = {"xp_mean": 5, "xp_p10": 1, "xp_p25": 3, "xp_p50": 5, "xp_p75": 7, "xp_p90": 11,
           "xp_se": 0.4, "p_play": .9, "p_start": .8, "p_sixty": .7, "selected_by_percent": 10, "now_cost": 5}
    out = decision_metrics(row)
    for key in ("quartile_range_points", "start_gap", "bench_risk", "floor_value_per_million", "ceiling_value_per_million"):
        assert key in out
