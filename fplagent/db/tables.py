"""Schema, defined once in SQLAlchemy Core so it is portable between SQLite
(zero-setup local runs) and Postgres (real deployment).

Three schema decisions carried over from the v5 audit, each fixing a real
data-loss or data-corruption bug:

1. FIXTURE GRAIN, NOT GAMEWEEK GRAIN.
   v5 keyed player performance on (player_id, gw) with a UNIQUE constraint.
   In a double gameweek a player has two fixtures with the same `round`, so
   the second row silently overwrote the first and half the player's points
   vanished. Here the grain is (player_id, season, fixture_id). Gameweek
   totals are a VIEW over that, never the storage grain.

2. CUMULATIVE AND PER-FIXTURE VALUES NEVER SHARE A COLUMN.
   v5 diffed `total_points` correctly but wrote `minutes`, `expected_goals`
   etc. straight from bootstrap-static, which are ALSO season-cumulative.
   The same column then held 78 (per-fixture, from backfill) and 340
   (cumulative, from snapshots). Here cumulative values live only in
   `player_season_snapshots`, with `season_` prefixes, and per-fixture values
   live only in `player_fixture_stats`. The prefix is the type system.

3. EVERY FACT CARRIES `known_at`.
   Walk-forward backtesting is fiction unless you can reconstruct what was
   knowable before a deadline. Every feature-bearing row records when the
   information became available, and every training query filters
   `known_at < deadline_time`. Without this the backtest leaks the future.

MULTI-SEASON SUPPORT (v9)
-------------------------
`teams` gains a `code` column. FPL assigns each club a persistent `code`
that survives promotion and relegation, while the `team_id` (`id` in the
bootstrap payload) changes every season. The `code` is what allows joining
prior-season history onto the current-season team list.

`historical_fixtures` stores prior-season fixtures fetched from the public
vaastav dataset. It deliberately does NOT store player-level data -- the
team-strength fit is the only consumer, and it needs goals and dates only.
Player-level history for prior seasons is not needed because the attacking
share estimator already uses a positional prior when evidence is thin, and
the effort required to reconcile player identities across seasons is not
worth the marginal improvement to shares.
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)

metadata = MetaData()

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

teams = Table(
    "teams", metadata,
    Column("team_id", Integer, primary_key=True),
    Column("season", Integer, primary_key=True),
    # FPL's persistent club identifier. Survives promotion/relegation;
    # `team_id` does not. This is the join key for prior-season history.
    # Nullable so a pre-migration row does not crash the schema; the next
    # `ingest` populates it.
    Column("code", Integer, index=True, nullable=True),
    Column("name", String(64)),
    Column("short_name", String(8)),
    Column("promoted", Boolean, default=False),
)

players = Table(
    "players", metadata,
    Column("player_id", Integer, primary_key=True),
    Column("season", Integer, primary_key=True),
    Column("web_name", String(64)),
    Column("first_name", String(64)),
    Column("second_name", String(64)),
    Column("team_id", Integer),
    Column("position", String(3)),            # GK / DEF / MID / FWD
    Column("birth_date", String(10)),         # ISO date, for identity matching
    Column("now_cost", Float),
    Column("first_pl_season", Integer),       # cold-start: null == never played PL
    Column("penalties_order", Integer),       # from bootstrap-static, null == not a taker
    Column("corners_order", Integer),
    Column("direct_fk_order", Integer),
)

gameweeks = Table(
    "gameweeks", metadata,
    Column("season", Integer, primary_key=True),
    Column("gw", Integer, primary_key=True),
    Column("deadline_time", DateTime),
    Column("is_current", Boolean, default=False),
    Column("is_next", Boolean, default=False),
    Column("finished", Boolean, default=False),
    Column("average_entry_score", Float),
)

fixtures = Table(
    "fixtures", metadata,
    Column("fixture_id", Integer, primary_key=True),
    Column("season", Integer, primary_key=True),
    Column("gw", Integer, index=True),
    Column("home_team_id", Integer),
    Column("away_team_id", Integer),
    Column("kickoff", DateTime),
    Column("finished", Boolean, default=False),
    Column("home_goals", Integer),
    Column("away_goals", Integer),
)
Index("ix_fixtures_season_gw", fixtures.c.season, fixtures.c.gw)

# Prior-season fixtures, sourced from the public vaastav dataset. Keyed on
# the persistent team `code`, not on `team_id`, because `team_id` changes
# every season. The consumer (`features.build.load_fixtures_with_history`)
# maps codes onto the current season's `team_id`s and drops any club that
# is not in the current Premier League.
historical_fixtures = Table(
    "historical_fixtures", metadata,
    # season == the starting calendar year of the season, matching the
    # convention in `fixtures`. 2024 == the 2024/25 season.
    Column("season", Integer, primary_key=True),
    Column("source_fixture_id", Integer, primary_key=True),
    Column("gw", Integer, index=True),
    Column("home_team_code", Integer, index=True),
    Column("away_team_code", Integer, index=True),
    Column("kickoff", DateTime),
    Column("finished", Boolean, default=False),
    Column("home_goals", Integer),
    Column("away_goals", Integer),
    Column("known_at", DateTime, server_default=func.now()),
    Column("source", String(24), default="vaastav"),
)
Index("ix_hist_fx_season_gw", historical_fixtures.c.season,
      historical_fixtures.c.gw)

# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

# Immutable log of SEASON-CUMULATIVE values as returned by bootstrap-static.
# Every column here is cumulative and named to say so. Nothing downstream may
# treat these as per-gameweek figures.
player_season_snapshots = Table(
    "player_season_snapshots", metadata,
    Column("snapshot_id", Integer, primary_key=True, autoincrement=True),
    Column("player_id", Integer, index=True),
    Column("season", Integer),
    Column("pulled_at", DateTime, server_default=func.now()),
    Column("as_of_gw", Integer),
    Column("season_minutes", Integer),
    Column("season_total_points", Integer),
    Column("season_expected_goals", Float),
    Column("season_expected_assists", Float),
    Column("season_expected_goals_conceded", Float),
    Column("form", Float),
    Column("selected_by_percent", Float),
    Column("transfers_in_event", Integer),
    Column("transfers_out_event", Integer),
    Column("now_cost", Float),
    Column("team_id", Integer),
    Column("position", String(3)),
    Column("status", String(2)),
    Column("news", Text),
    Column("chance_of_playing_next_round", Float),
    Column("ep_next", Float),
    Column("season_yellow_cards", Integer),
)

# The single source of truth for per-fixture performance. One row per player
# per fixture -- double gameweeks produce two rows, as they must.
player_fixture_stats = Table(
    "player_fixture_stats", metadata,
    Column("player_id", Integer, nullable=False),
    Column("season", Integer, nullable=False),
    Column("fixture_id", Integer, nullable=False),
    Column("gw", Integer, nullable=False, index=True),
    Column("opponent_team_id", Integer),
    Column("was_home", Boolean),
    Column("minutes", Integer),
    Column("starts", Integer),
    Column("total_points", Integer),
    Column("goals_scored", Integer),
    Column("assists", Integer),
    Column("clean_sheets", Integer),
    Column("goals_conceded", Integer),
    Column("own_goals", Integer),
    Column("penalties_saved", Integer),
    Column("penalties_missed", Integer),
    Column("yellow_cards", Integer),
    Column("red_cards", Integer),
    Column("saves", Integer),
    Column("bonus", Integer),
    Column("bps", Integer),
    Column("influence", Float),
    Column("creativity", Float),
    Column("threat", Float),
    Column("ict_index", Float),
    Column("expected_goals", Float),
    Column("expected_assists", Float),
    Column("expected_goal_involvements", Float),
    Column("expected_goals_conceded", Float),
    Column("clearances_blocks_interceptions", Integer),
    Column("recoveries", Integer),
    Column("tackles", Integer),
    Column("defensive_contribution", Integer),
    Column("cbit", Integer),
    Column("cbirt", Integer),
    Column("penalties_scored_est", Float),
    Column("value", Float),
    Column("source", String(24), nullable=False),
    Column("known_at", DateTime, server_default=func.now()),
    UniqueConstraint("player_id", "season", "fixture_id", name="uq_pfs_grain"),
)
Index("ix_pfs_player_season", player_fixture_stats.c.player_id, player_fixture_stats.c.season)

set_piece_duties = Table(
    "set_piece_duties", metadata,
    Column("player_id", Integer, primary_key=True),
    Column("season", Integer, primary_key=True),
    Column("duty", String(16), primary_key=True),
    Column("order", Integer),
    Column("share", Float),
    Column("confidence", Float),
    Column("known_at", DateTime, server_default=func.now()),
    Column("source", String(24)),
    Column("retired_at", DateTime),
)

set_piece_duty_history = Table(
    "set_piece_duty_history", metadata,
    Column("duty_id", Integer, primary_key=True, autoincrement=True),
    Column("player_id", Integer, nullable=False, index=True),
    Column("season", Integer, nullable=False, index=True),
    Column("duty", String(16), nullable=False),
    Column("order", Integer),
    Column("share", Float),
    Column("confidence", Float),
    Column("known_at", DateTime, server_default=func.now()),
    Column("source", String(24)),
    Column("retired_at", DateTime),
)
Index("ix_spdh_lookup", set_piece_duty_history.c.player_id,
      set_piece_duty_history.c.season, set_piece_duty_history.c.duty,
      set_piece_duty_history.c.known_at)

# ---------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------

team_strength = Table(
    "team_strength", metadata,
    Column("season", Integer, primary_key=True),
    Column("gw", Integer, primary_key=True),
    Column("team_id", Integer, primary_key=True),
    Column("model_version", String(48), primary_key=True),
    Column("attack", Float),
    Column("defence", Float),
    Column("home_adv", Float),
    Column("rho", Float),
    Column("league_avg_goals", Float),
    Column("n_matches_fit", Integer),
    Column("effective_n", Float),
    Column("target", String(8)),
    Column("created_at", DateTime, server_default=func.now()),
)

predictions = Table(
    "predictions", metadata,
    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
    Column("model_version", String(48), nullable=False, index=True),
    Column("season", Integer, nullable=False),
    Column("gw", Integer, nullable=False, index=True),
    Column("fixture_id", Integer),
    Column("player_id", Integer, nullable=False),
    Column("team_id", Integer),
    Column("position", String(8)),
    Column("now_cost", Float),
    Column("deadline_time", DateTime),
    Column("created_at", DateTime, server_default=func.now()),
    Column("p_play", Float),
    Column("p_start", Float),
    Column("p_sixty", Float),
    Column("xp_mean", Float),
    Column("xp_p10", Float),
    Column("xp_p25", Float),
    Column("xp_p50", Float),
    Column("xp_p75", Float),
    Column("xp_p90", Float),
    Column("p_haul_10plus", Float),
    Column("p_blank_le2", Float),
    Column("xp_se", Float),
    Column("n_sims", Integer),
    Column("components", JSON),
    Column("features", JSON),
    Column("sim_quantiles", JSON),
    UniqueConstraint("model_version", "season", "gw", "player_id", "fixture_id",
                     name="uq_prediction_grain"),
)

model_versions = Table(
    "model_versions", metadata,
    Column("model_version", String(48), primary_key=True),
    Column("season", Integer),
    Column("target_gw", Integer),
    Column("gws", JSON),
    Column("created_at", DateTime, server_default=func.now()),
    Column("promoted", Boolean),
    Column("promotion_reason", Text),
    Column("evaluation", JSON),
)

recommendations = Table(
    "recommendations", metadata,
    Column("rec_id", Integer, primary_key=True, autoincrement=True),
    Column("season", Integer),
    Column("gw", Integer),
    Column("rec_type", String(32)),
    Column("model_version", String(48)),
    Column("payload", JSON),
    Column("created_at", DateTime, server_default=func.now()),
    Column("actual_outcome", JSON),
    Column("evaluated_at", DateTime),
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

evaluation_runs = Table(
    "evaluation_runs", metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("model_version", String(48), index=True),
    Column("season", Integer),
    Column("gw_from", Integer),
    Column("gw_to", Integer),
    Column("n", Integer),
    Column("mae", Float),
    Column("rmse", Float),
    Column("spearman_overall", Float),
    Column("spearman_by_position", JSON),
    Column("top5_captain_hit_rate", Float),
    Column("top10_captain_hit_rate", Float),
    Column("crps", Float),
    Column("pit_ks_stat", Float),
    Column("pit_histogram", JSON),
    Column("baselines", JSON),
    Column("beat_best_baseline", Boolean),
    Column("created_at", DateTime, server_default=func.now()),
)

calibration_bins = Table(
    "calibration_bins", metadata,
    Column("model_version", String(48), primary_key=True),
    Column("season", Integer, primary_key=True),
    Column("gw", Integer, primary_key=True),
    Column("target", String(24), primary_key=True),
    Column("bucket", String(16), primary_key=True),
    Column("predicted_rate", Float),
    Column("actual_rate", Float),
    Column("n", Integer),
)

data_quality_events = Table(
    "data_quality_events", metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, server_default=func.now()),
    Column("stage", String(48)),
    Column("check_name", String(64)),
    Column("severity", String(8)),
    Column("detail", JSON),
)

pipeline_runs = Table(
    "pipeline_runs", metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("model_version", String(48)),
    Column("season", Integer),
    Column("gw", Integer),
    Column("started_at", DateTime, server_default=func.now()),
    Column("finished_at", DateTime),
    Column("status", String(16)),
    Column("stages", JSON),
    Column("error", Text),
)

# ---------------------------------------------------------------------------
# Cold start / league translation
# ---------------------------------------------------------------------------

league_translation = Table(
    "league_translation", metadata,
    Column("source_league", String(48), primary_key=True),
    Column("position_group", String(16), primary_key=True),
    Column("metric", String(32), primary_key=True),
    Column("ratio", Float),
    Column("confidence", Float),
    Column("sample_size", Integer),
    Column("source", String(32)),
    Column("updated_at", DateTime, server_default=func.now()),
)

external_player_map = Table(
    "external_player_map", metadata,
    Column("player_id", Integer, primary_key=True),
    Column("source", String(24), primary_key=True),
    Column("external_id", String(64)),
    Column("match_score", Float),
    Column("method", String(24)),
    Column("verified", Boolean, default=False),
)

external_player_season = Table(
    "external_player_season", metadata,
    Column("external_id", String(64), primary_key=True),
    Column("source", String(24), primary_key=True),
    Column("season", Integer, primary_key=True),
    Column("league", String(48), primary_key=True),
    Column("player_name", String(96)),
    Column("team", String(64)),
    Column("position", String(16)),
    Column("minutes", Integer),
    Column("np_xg", Float),
    Column("xa", Float),
    Column("defensive_actions", Integer),
    Column("progressive_passes", Integer),
    Column("progressive_carries", Integer),
    Column("shot_creating_actions", Float),
    Column("known_at", DateTime, server_default=func.now()),
)

# ---------------------------------------------------------------------------
# Congestion beyond FPL's own fixture list
# ---------------------------------------------------------------------------

other_competition_fixtures = Table(
    "other_competition_fixtures", metadata,
    Column("team_id", Integer, primary_key=True),
    Column("season", Integer, primary_key=True),
    Column("competition", String(32), primary_key=True),
    Column("kickoff", DateTime, primary_key=True),
    Column("fixture_key", String(96), index=True),
    Column("round", String(48)),
    Column("source", String(24)),
    Column("known_at", DateTime, server_default=func.now()),
)

# ---------------------------------------------------------------------------
# Effective ownership
# ---------------------------------------------------------------------------

effective_ownership = Table(
    "effective_ownership", metadata,
    Column("player_id", Integer, primary_key=True),
    Column("season", Integer, primary_key=True),
    Column("gw", Integer, primary_key=True),
    Column("sample_size", Integer),
    Column("ownership_pct", Float),
    Column("captain_pct", Float),
    Column("triple_captain_pct", Float),
    Column("effective_ownership", Float),
    Column("pulled_at", DateTime, server_default=func.now()),
)

# ---------------------------------------------------------------------------
# Market signal
# ---------------------------------------------------------------------------

match_odds = Table(
    "match_odds", metadata,
    Column("season", Integer, primary_key=True),
    Column("fixture_id", Integer, primary_key=True),
    Column("bookmaker", String(32), primary_key=True),
    Column("pulled_at", DateTime, primary_key=True),
    Column("home_win", Float),
    Column("draw", Float),
    Column("away_win", Float),
    Column("over_2_5", Float),
    Column("under_2_5", Float),
    Column("devigged_home", Float),
    Column("devigged_draw", Float),
    Column("devigged_away", Float),
    Column("implied_home_xg", Float),
    Column("implied_away_xg", Float),
)

ALL_TABLES = list(metadata.tables.values())