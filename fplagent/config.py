"""Central configuration.

Design note (v6): nothing in this file is a hand-edited piece of *state*.
v5 had CURRENT_GW as an env var that a human had to remember to bump, which
was the direct cause of a whole class of "why is it ingesting GW4" confusion.
The current gameweek is DERIVED from the FPL API's own events[] feed (stored
by ingestion.pipeline.ingest_bootstrap, read back by
db.repository.resolve_target_gw). Config holds only genuine configuration:
where things live, what credentials to use, and tunable model
hyperparameters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from fplagent import __version__

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'fpl_agent.db'}")
IS_SQLITE = DATABASE_URL.startswith("sqlite")

DEMO_DATABASE_URL = os.getenv("DEMO_DATABASE_URL",
                              f"sqlite:///{(ROOT / 'fpl_agent_demo.db').as_posix()}")

# --------------------------------------------------------------------------
# External sources
# --------------------------------------------------------------------------
FPL_BASE = os.getenv("FPL_BASE", "https://fantasy.premierleague.com/api")
FPL_TIMEOUT = float(os.getenv("FPL_TIMEOUT", "20"))
FPL_USER_AGENT = os.getenv(
    "FPL_USER_AGENT",
    f"fpl-agent/{__version__} (personal analytics project; contact via github)",
)
VAASTAV_BASE = os.getenv(
    "VAASTAV_BASE",
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data",
)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = os.getenv("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE = os.getenv("FOOTBALL_DATA_BASE", "https://api.football-data.org/v4")

# --------------------------------------------------------------------------
# LLM (provider-agnostic)
# --------------------------------------------------------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "") or None
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# --------------------------------------------------------------------------
# Season identity
# --------------------------------------------------------------------------
CURRENT_SEASON = int(os.getenv("FPL_SEASON", "2026"))
FIRST_HALF_LAST_GW = 19   # chips reset at the GW19 deadline
TOTAL_GWS = 38

# Approximate FPL overall entrant count, used where a rough "field size"
# is needed (rank-swing estimates, ownership-vs-transfers ratios). FPL
# does not publish this directly; it is a widely-cited approximate figure
# that has been stable within +/- 15% for several seasons. Anything that
# depends on it should be labelled as a heuristic, not a fitted model.
TOTAL_MANAGERS = 11_000_000


@dataclass(frozen=True)
class ModelParams:
    """Tunable model hyperparameters, in one place so a sweep can vary them."""

    # Dixon-Coles time decay, per day. 0.0065/day halves a match's weight in
    # ~107 days, i.e. roughly a third of a season.
    decay_per_day: float = 0.0065
    # Below this many matches the joint fit is unstable; fall back to priors.
    min_matches_for_fit: int = 30
    # Multi-season team strength. Prior-season fixtures contribute to the
    # fit but are weighted as if they were OLDER than they are, because the
    # same club is not the same team two years later -- personnel, manager,
    # and league context all shift.
    #
    # `prior_season_age_multiplier` scales a prior-season match's age
    # BEFORE the standard exponential decay is applied. At 0.25, a match
    # 500 days old is treated as if it were 125 days old, giving it roughly
    # 30-40% of a same-age current-season match.
    #
    # Raised from 2 to 3 after the two-season fit stabilised rho from
    # -0.359 to -0.128 and lifted Spearman from 0.6131 to 0.6187. A third
    # season (~1140 prior matches vs 760) gives the fit noticeably more
    # evidence to identify attack/defence parameters against, which
    # primarily matters for the first six gameweeks of a season -- after
    # that the current season dominates the decayed weighting anyway.
    #
    # Setting `prior_seasons_to_include` to 0 disables the mechanism and
    # reverts to single-season fitting.
    prior_seasons_to_include: int = 3
    prior_season_age_multiplier: float = 0.25
    # Goal ceiling for the scoreline matrix. 8 covers >99.99% of PL matches.
    max_goals: int = 8
    # Shrinkage strength for player attacking shares, in 90-minute equivalents.
    #
    # Raised from 6.0 to 12.0 after auditing an early-season case: a forward
    # who scored 3-4 in one game and nothing in the surrounding matches had
    # a 5-fixture, ~450-minute sample, giving w = minutes / (minutes + k) =
    # 5 / 11 = 0.45 on his own data at k=6. That is more than enough weight
    # for a single outlier game to drag his share estimate far above what a
    # mid-table striker's base rate supports. At k=12 the same player gets
    # w = 5/17 = 0.29, so the positional prior dominates and the model
    # correctly treats the sample as thin. A full-season starter reaches
    # w = 0.5 around ~1080 minutes (~GW12), which is the right point at
    # which to trust his own rate over the prior.
    share_shrinkage_games: float = 12.0
    defcon_shrinkage_games: float = 5.0
    card_shrinkage_games: float = 10.0
    translation_min_players: int = 5
    translation_min_target_minutes: float = 900.0
    translation_max_plausible_ratio: float = 3.0
    translation_min_plausible_ratio: float = 0.15
    n_sims: int = 20000
    prediction_horizon: int = 4
    horizon_n_sims: int = 5000
    use_learned_minutes: bool = True
    bench_weight: float = 0.12
    default_risk_lambda: float = 0.0
    # What the Dixon-Coles fit is trained against:
    #   goals -- realised goals only, as v6 did
    #   xg    -- expected goals only, if xg_matches is supplied
    #   blend -- a weighted average of the two, `xg_blend_weight` on xG
    #
    # Lowered from 0.6 to 0.4. At 0.6 the fit was getting rho = -0.128,
    # whereas a goals-only fit gives rho = -0.098. The xG response is
    # smoother than goals and produces fewer exact low-score cells for the
    # tau correction to fire on, which pushes the fitted rho further from
    # zero to compensate. Both fits are defensible, but the goals-weighted
    # version tracks the actual match-level process more closely and rho
    # sits nearer the empirically observed PL range (-0.05 to +0.05).
    #
    # The reason not to go all the way to goals-only: xG is a genuinely
    # less noisy measurement of the underlying scoring process than a
    # 1-0 result is, and gives the fit useful information that goals alone
    # throw away. 0.4 is a compromise -- enough xG to reduce noise, enough
    # goals to keep rho honest.
    team_strength_target: str = "blend"     # goals | xg | blend
    xg_blend_weight: float = 0.4


PARAMS = ModelParams()


@dataclass(frozen=True)
class Thresholds:
    """Data-quality gate thresholds. These are the gates that ACTUALLY fire."""
    min_players_expected: int = 500
    max_players_expected: int = 900
    max_null_rate: float = 0.05
    max_minutes_per_fixture: int = 130
    max_share: float = 0.95


GATES = Thresholds()