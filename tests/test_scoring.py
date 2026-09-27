from fplagent.models.scoring import MatchLine, score_player


def test_basic_scorer_is_numeric():
    out = score_player(MatchLine(position="MID", minutes=90, goals=1, assists=0, team_goals_conceded=0))
    assert isinstance(out, int)
    assert out > 0
