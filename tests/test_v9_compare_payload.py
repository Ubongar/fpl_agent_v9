"""Comparison payload: position-aware fixture difficulty and model flags.

The unit tests need nothing but the module. The integration test drives the
real /players/compare endpoint against the demo database produced by
`fplagent demo`; it skips (rather than fails) when that database has not been
built yet, so a bare `pytest` on a fresh checkout stays green while CI, which
runs the demo first, exercises it for real.
"""
import pytest

from fplagent.api.main import _position_difficulty

ROW = {"def_difficulty": 2.0, "att_difficulty": 4.0}


def test_defenders_and_keepers_use_defensive_difficulty():
    assert _position_difficulty("GK", ROW) == 2.0
    assert _position_difficulty("DEF", ROW) == 2.0


def test_forwards_use_attacking_difficulty():
    assert _position_difficulty("FWD", ROW) == 4.0


def test_midfielders_blend_both():
    assert _position_difficulty("MID", ROW) == 3.0


def _demo_client():
    from pathlib import Path
    from fplagent.config import DEMO_DATABASE_URL
    path = DEMO_DATABASE_URL.replace("sqlite:///", "")
    if not Path(path).exists():
        pytest.skip("demo database not built (run `fplagent demo`)")
    from fastapi.testclient import TestClient
    from fplagent.db.session import use_database
    previous = use_database(DEMO_DATABASE_URL)
    import fplagent.api.main as m
    return TestClient(m.app), previous


def test_compare_endpoint_returns_fixtures_and_flags():
    client, previous = _demo_client()
    try:
        page = client.get("/predictions?limit=50&sort=xp_mean&desc=true")
        if page.status_code != 200 or not page.json().get("players"):
            pytest.skip("demo database has no predictions")
        ids = [p["player_id"] for p in page.json()["players"][:3]]
        res = client.post("/players/compare", json={"player_ids": ids})
        assert res.status_code == 200
        body = res.json()
        assert body["fixture_horizon"] == 5
        for p in body["players"]:
            assert isinstance(p["flags"], list)
            assert isinstance(p["next_fixtures"], list)
            for f in p["next_fixtures"]:
                assert 1.0 <= f["difficulty"] <= 5.0
            if p["next_fixtures"]:
                assert 1.0 <= p["avg_difficulty"] <= 5.0
            else:
                assert p["avg_difficulty"] is None
    finally:
        from fplagent.db.session import use_database
        use_database(previous)
