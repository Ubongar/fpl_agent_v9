"""Central configuration."""
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

# External API Keys
OPENFOOT_API_KEY = os.getenv("OPENFOOT_API_KEY", "")
BZZOIRO_API_KEY = os.getenv("BZZOIRO_API_KEY", "")
FOOTBALLDATA_API_KEY = os.getenv("FOOTBALLDATA_API_KEY", "")
STRIKER_API_KEY = os.getenv("STRIKER_API_KEY", "")
SPORTMONKS_API_KEY = os.getenv("SPORTMONKS_API_KEY", "")

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

TOTAL_MANAGERS = 11_000_000


@dataclass(frozen=True)
class ModelParams:
    """Tunable model hyperparameters, in one place so a sweep can vary them."""

    decay_per_day: float = 0.0065
    min_matches_for_fit: int = 30
    prior_seasons_to_include: int = 3
    prior_season_age_multiplier: float = 0.25
    max_goals: int = 8
    share_shrinkage_games: float = 12.0
    xa_assist_blend_weight: float = 0.65
    external_api_blend_weight: float = 0.5  # How much to trust API per-90 stats vs FPL historicals
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
    team_strength_target: str = "blend"
    xg_blend_weight: float = 0.4


PARAMS = ModelParams()


@dataclass(frozen=True)
class Thresholds:
    """Data-quality gate thresholds."""
    min_players_expected: int = 500
    max_players_expected: int = 900
    max_null_rate: float = 0.05
    max_minutes_per_fixture: int = 130
    max_share: float = 0.95


GATES = Thresholds()