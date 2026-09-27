"""Decision-layer metrics built from the probabilistic projection, not an opaque score.

The predictive model should estimate a distribution; the decision layer exposes
separate interpretable dimensions so a manager can choose how to trade them off.
"""
from __future__ import annotations

import math


def _num(value, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(x) else x


def decision_metrics(row: dict) -> dict:
    """Return a transparent metric pack for one player prediction."""
    xp = _num(row.get("xp_mean"))
    p10 = _num(row.get("xp_p10"), xp)
    p50 = _num(row.get("xp_p50"), xp)
    p90 = _num(row.get("xp_p90"), xp)
    p_play = max(0.0, min(1.0, _num(row.get("p_play"))))
    p_start = max(0.0, min(1.0, _num(row.get("p_start"))))
    p_sixty = max(0.0, min(1.0, _num(row.get("p_sixty"))))
    eo_raw = row.get("eo")
    if eo_raw is None:
        eo_raw = row.get("selected_by_percent", 0.0)
    eo = _num(eo_raw)
    if eo > 1:
        eo /= 100.0
    eo = max(0.0, min(1.0, eo))
    price = _num(row.get("now_cost", row.get("price", 0.0)))

    uncertainty = _num(row.get("xp_se"), 0.0)
    certainty = max(0.0, min(1.0, 1.0 - uncertainty / max(abs(xp) + 1.0, 1.0)))
    upside = max(0.0, p90 - p50)
    downside = max(0.0, p50 - p10)
    start_security = p_start / p_play if p_play > 1e-9 else 0.0
    sixty_security = p_sixty / p_play if p_play > 1e-9 else 0.0
    minutes_stability = 0.55 * start_security + 0.45 * sixty_security
    value = xp / price if price > 0 else None
    floor_adjusted = 0.65 * xp + 0.35 * p10
    ceiling_adjusted = 0.65 * xp + 0.35 * p90
    ownership_adjusted_upside = upside * (1.0 - eo)
    ownership_adjusted_downside = downside * (1.0 - eo)
    quartile_range = max(0.0, _num(row.get("xp_p75"), p50) - _num(row.get("xp_p25"), p50))
    start_gap = max(0.0, p_play - p_start)
    bench_risk = max(0.0, 1.0 - start_security)
    floor_value = floor_adjusted / price if price > 0 else None
    ceiling_value = ceiling_adjusted / price if price > 0 else None
    return {
        "certainty": round(certainty, 4),
        "uncertainty_points": round(uncertainty, 3),
        "upside_points": round(upside, 3),
        "downside_points": round(downside, 3),
        "range_points": round(max(0.0, p90 - p10), 3),
        "start_security": round(start_security, 4),
        "sixty_security": round(sixty_security, 4),
        "minutes_stability": round(minutes_stability, 4),
        "value_per_million": round(value, 4) if value is not None else None,
        "floor_adjusted_xp": round(floor_adjusted, 3),
        "ceiling_adjusted_xp": round(ceiling_adjusted, 3),
        "ownership_adjusted_upside": round(ownership_adjusted_upside, 3),
        "ownership_adjusted_downside": round(ownership_adjusted_downside, 3),
        "quartile_range_points": round(quartile_range, 3),
        "start_gap": round(start_gap, 4),
        "bench_risk": round(bench_risk, 4),
        "floor_value_per_million": round(floor_value, 4) if floor_value is not None else None,
        "ceiling_value_per_million": round(ceiling_value, 4) if ceiling_value is not None else None,
        "ownership": round(eo, 4),
    }
