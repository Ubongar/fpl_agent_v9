"""Point-in-time feature construction."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from fplagent.config import CURRENT_SEASON, PARAMS
from fplagent.db import tables as T
from fplagent.db.repository import read_sql
from fplagent.models import bonus as bonus_mod
from fplagent.models import cards as cards_mod
from fplagent.models import cold_start as cold_start_mod
from fplagent.models import defcon as defcon_mod
from fplagent.models import setpieces as sp
from fplagent.models.minutes import (
    FEATURE_ORDER, UNAVAILABLE, MinutesModel, heuristic_minutes,
)
from fplagent.models.shares import estimate_shares
from fplagent.ingestion.external_apis import load_external_metrics

log = logging.getLogger(__name__)

YELLOW_BAN_THRESHOLDS = ((5, 19), (10, 32), (15, 38))
PENALTY_XG_PER_ATTEMPT = 0.79


def load_history(cutoff: datetime, season: int = CURRENT_SEASON) -> pd.DataFrame:
    stmt = select(T.player_fixture_stats).where(
        (T.player_fixture_stats.c.known_at < cutoff)
        & (T.player_fixture_stats.c.season <= season))
    df = read_sql(stmt)
    if df.empty:
        return df
    return df.sort_values(["player_id", "season", "gw", "fixture_id"])


def load_fixtures(season: int = CURRENT_SEASON) -> pd.DataFrame:
    return read_sql(select(T.fixtures).where(T.fixtures.c.season == season))


def load_fixtures_with_history(season: int = CURRENT_SEASON,
                               n_prior: int | None = None) -> pd.DataFrame:
    current = load_fixtures(season)
    n_prior = PARAMS.prior_seasons_to_include if n_prior is None else n_prior
    if n_prior <= 0:
        return current

    teams_df = read_sql(select(T.teams).where(T.teams.c.season == season))
    if teams_df.empty or "code" not in teams_df.columns:
        return current
    code_to_id: dict[int, int] = {}
    for r in teams_df.itertuples():
        try:
            code = int(r.code) if r.code is not None else None
        except (TypeError, ValueError):
            code = None
        if code is not None:
            code_to_id[code] = int(r.team_id)
    if not code_to_id:
        return current

    prior_seasons = [season - k for k in range(1, n_prior + 1)]
    hist = read_sql(select(T.historical_fixtures).where(
        T.historical_fixtures.c.season.in_(prior_seasons)))
    if hist.empty:
        return current

    def _map_code(c) -> int | None:
        try:
            return code_to_id.get(int(c))
        except (TypeError, ValueError):
            return None

    hist = hist.copy()
    hist["home_team_id"] = hist["home_team_code"].map(_map_code)
    hist["away_team_id"] = hist["away_team_code"].map(_map_code)
    hist = hist[hist["home_team_id"].notna() & hist["away_team_id"].notna()].copy()
    if hist.empty:
        return current

    hist = hist.rename(columns={"source_fixture_id": "fixture_id"})

    cols = ["season", "fixture_id", "gw", "home_team_id", "away_team_id",
            "kickoff", "finished", "home_goals", "away_goals"]
    cols = [c for c in cols if c in current.columns and c in hist.columns]
    return pd.concat([current[cols], hist[cols]], ignore_index=True)


def load_players(season: int = CURRENT_SEASON) -> pd.DataFrame:
    return read_sql(select(T.players).where(T.players.c.season == season))


def load_latest_status(cutoff: datetime, season: int = CURRENT_SEASON) -> pd.DataFrame:
    stmt = select(T.player_season_snapshots).where(
        (T.player_season_snapshots.c.season == season)
        & (T.player_season_snapshots.c.pulled_at < cutoff))
    df = read_sql(stmt)
    if df.empty:
        return df
    return (df.sort_values("pulled_at").groupby("player_id", as_index=False)
              .tail(1).reset_index(drop=True))


def _load_other_competition_fixtures(season: int, cutoff: datetime | None = None) -> pd.DataFrame:
    try:
        conds = [T.other_competition_fixtures.c.season == season]
        if cutoff is not None:
            conds.extend([T.other_competition_fixtures.c.known_at < cutoff,
                          T.other_competition_fixtures.c.kickoff < cutoff])
        df = read_sql(select(T.other_competition_fixtures).where(*conds))
    except Exception:
        return pd.DataFrame(columns=["team_id", "kickoff"])
    return df


def load_set_piece_duties(season: int, cutoff: datetime | None = None) -> pd.DataFrame:
    try:
        sp_t = T.set_piece_duty_history
        conds = [sp_t.c.season == season]
        if cutoff is not None:
            conds.extend([sp_t.c.known_at < cutoff,
                          sp_t.c.retired_at.is_(None) | (sp_t.c.retired_at >= cutoff)])
        else:
            conds.append(sp_t.c.retired_at.is_(None))
        df = read_sql(select(sp_t).where(*conds))
        if not df.empty:
            return (df.sort_values("known_at")
                      .groupby(["player_id", "duty"], as_index=False).tail(1)
                      .reset_index(drop=True))
    except Exception:
        pass
    sp_t = T.set_piece_duties
    conds = [sp_t.c.season == season]
    if cutoff is not None:
        conds.extend([sp_t.c.known_at < cutoff,
                      sp_t.c.retired_at.is_(None) | (sp_t.c.retired_at >= cutoff)])
    else:
        conds.append(sp_t.c.retired_at.is_(None))
    df = read_sql(select(sp_t).where(*conds))
    return df if not df.empty else pd.DataFrame(columns=["player_id", "duty", "order", "share"])


def duty_lookup(duties: pd.DataFrame, duty: str) -> tuple[dict, dict]:
    if duties.empty:
        return {}, {}
    d = duties[duties["duty"] == duty]
    if d.empty:
        return {}, {}
    share = {int(r["player_id"]): float(r["share"] if pd.notna(r.get("share")) else
                                        sp.order_to_share(duty, r["order"]))
             for _, r in d.iterrows()}
    order = {int(r["player_id"]): int(r["order"]) for _, r in d.iterrows()
             if pd.notna(r["order"])}
    return share, order


def team_match_results(fixtures: pd.DataFrame, cutoff_gw: int,
                       as_of: datetime,
                       current_season: int | None = None,
                       prior_season_age_multiplier: float | None = None
                       ) -> list[tuple[int, int, int, int, float]]:
    if fixtures.empty:
        return []
    done = fixtures[(fixtures["finished"] == True) &
                    fixtures["home_goals"].notna()]
    if done.empty:
        return []
    if current_season is not None and "season" in done.columns:
        done = done[
            ((done["season"] == current_season) & (done["gw"] < cutoff_gw))
            | (done["season"] < current_season)
        ]
    else:
        done = done[done["gw"] < cutoff_gw]
    if done.empty:
        return []

    multiplier = (PARAMS.prior_season_age_multiplier
                  if prior_season_age_multiplier is None
                  else prior_season_age_multiplier)

    out = []
    for _, r in done.iterrows():
        ko = r["kickoff"] if pd.notna(r["kickoff"]) else as_of
        ko_dt = pd.Timestamp(ko).to_pydatetime()
        if pd.notna(r["kickoff"]) and ko_dt + timedelta(hours=3) > as_of:
            continue
        days = max((as_of - ko_dt).days, 0)
        if (current_season is not None and "season" in r.index
                and int(r["season"]) < current_season):
            days = days * multiplier
        out.append((int(r["home_team_id"]), int(r["away_team_id"]),
                    int(r["home_goals"]), int(r["away_goals"]), float(days)))
    return out


def team_match_xg(history: pd.DataFrame, fixtures: pd.DataFrame,
                  cutoff_gw: int) -> list[tuple[int, int, float, float, float]]:
    if history.empty or fixtures.empty:
        return []
    h = history[history["expected_goals"].notna()
                & history["season"].isin(fixtures["season"].unique())].copy()
    if h.empty:
        return []
    agg = (h.groupby(["fixture_id", "was_home"], as_index=False)["expected_goals"]
             .sum())
    fx = fixtures[fixtures["gw"] < cutoff_gw][
        ["fixture_id", "home_team_id", "away_team_id"]]
    home = agg[agg["was_home"] == True].rename(
        columns={"expected_goals": "home_xg"})[["fixture_id", "home_xg"]]
    away = agg[agg["was_home"] == False].rename(
        columns={"expected_goals": "away_xg"})[["fixture_id", "away_xg"]]
    m = fx.merge(home, on="fixture_id", how="inner").merge(
        away, on="fixture_id", how="inner")
    return [(int(r.home_team_id), int(r.away_team_id),
             float(r.home_xg), float(r.away_xg), 0.0)
            for r in m.itertuples()]


def _team_output_by_fixture(history: pd.DataFrame,
                            fixtures: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "fixture_id", "was_home"]
    cols = keys + ["team_goals", "team_assists"]
    if history.empty:
        return pd.DataFrame(columns=cols)

    assists = (history.groupby(keys, as_index=False)["assists"]
                      .sum().rename(columns={"assists": "team_assists"}))

    goals = (history.groupby(keys, as_index=False)["goals_scored"]
                    .sum().rename(columns={"goals_scored": "team_goals"}))
    if not fixtures.empty and "home_goals" in fixtures:
        fx = fixtures[fixtures["home_goals"].notna()]
        rows = []
        for r in fx.itertuples():
            rows.append({"season": int(r.season), "fixture_id": int(r.fixture_id),
                         "was_home": True, "team_goals": float(r.home_goals)})
            rows.append({"season": int(r.season), "fixture_id": int(r.fixture_id),
                         "was_home": False, "team_goals": float(r.away_goals)})
        if rows:
            goals = (pd.concat([pd.DataFrame(rows), goals], ignore_index=True)
                       .drop_duplicates(keys, keep="first"))

    out = goals.merge(assists, on=keys, how="outer")
    out["team_goals"] = out["team_goals"].fillna(0.0)
    out["team_assists"] = out["team_assists"].fillna(0.0)
    return out[cols]


def _team_output_xg_by_fixture(history: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "fixture_id", "was_home"]
    cols = keys + ["team_xg"]
    if history.empty or "expected_goals" not in history:
        return pd.DataFrame(columns=cols)
    h = history[history["expected_goals"].notna()]
    if h.empty:
        return pd.DataFrame(columns=cols)
    g = (h.groupby(keys, as_index=False)["expected_goals"].sum()
          .rename(columns={"expected_goals": "team_xg"}))
    return g[cols]


def _team_output_xa_by_fixture(history: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "fixture_id", "was_home"]
    cols = keys + ["team_xa"]
    if history.empty or "expected_assists" not in history:
        return pd.DataFrame(columns=cols)
    h = history[history["expected_assists"].notna()]
    if h.empty:
        return pd.DataFrame(columns=cols)
    g = (h.groupby(keys, as_index=False)["expected_assists"].sum()
          .rename(columns={"expected_assists": "team_xa"}))
    return g[cols]


def _team_penalties_by_fixture(history: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "fixture_id", "was_home"]
    cols = keys + ["team_pens_missed", "team_pens_est"]
    if history.empty:
        return pd.DataFrame(columns=cols)
    g = (history.groupby(keys, as_index=False)["penalties_missed"]
                .sum().rename(columns={"penalties_missed": "team_pens_missed"}))
    g["team_pens_est"] = (g["team_pens_missed"] / max(1.0 - sp.PENALTY_CONVERSION, 1e-6)
                          * sp.PENALTY_CONVERSION)
    return g[cols]


def _apply_doubtful_floor(mp, status_code: str, chance: float | None,
                          suspended: bool) -> None:
    mp.p_play = float(np.nan_to_num(mp.p_play, nan=0.0))
    mp.p_start_est = float(np.nan_to_num(mp.p_start_est, nan=0.0))
    mp.p_sixty = float(np.nan_to_num(mp.p_sixty, nan=0.0))
    mp.expected_minutes = float(np.nan_to_num(mp.expected_minutes, nan=0.0))
    if suspended or status_code in UNAVAILABLE:
        return
    if chance is None:
        return
    try:
        chance_val = float(chance)
    except (TypeError, ValueError):
        return
    if not np.isfinite(chance_val) or chance_val <= 0:
        return
    floor_play = float(np.clip(chance_val / 100.0 * 0.5, 0.0, 0.95))
    if mp.p_play < floor_play:
        mp.p_play = floor_play
    if mp.p_start_est < floor_play * 0.5:
        mp.p_start_est = floor_play * 0.5
    if mp.p_sixty < floor_play * 0.4:
        mp.p_sixty = floor_play * 0.4
    mp.expected_minutes = max(mp.expected_minutes, floor_play * 45.0)


def build_player_features(cutoff: datetime, target_gw: int,
                          season: int = CURRENT_SEASON,
                          lookback: int = 12,
                          minutes_model: MinutesModel | None = None,
                          strict_point_in_time: bool = False) -> pd.DataFrame:
    # 1. Load external API cache
    external_metrics = load_external_metrics()
    elo_data = external_metrics.get("elo", {})
    per90_data = external_metrics.get("player_per90", {})
    team_set_pieces = external_metrics.get("team_set_pieces", {})
    match_predictions = external_metrics.get("match_predictions", {})

    players = load_players(season)
    if players.empty:
        return pd.DataFrame()

    history = load_history(cutoff, season)
    status = load_latest_status(cutoff, season)
    fixtures = load_fixtures(season)
    team_out = _team_output_by_fixture(history, fixtures)
    team_xg_out = _team_output_xg_by_fixture(history)
    team_xa_out = _team_output_xa_by_fixture(history)
    team_pens = _team_penalties_by_fixture(history)
    duties = load_set_piece_duties(season, cutoff)
    pen_share, pen_order = duty_lookup(duties, "penalties")
    fk_share, fk_order = duty_lookup(duties, "direct_fk")
    cnr_share, cnr_order = duty_lookup(duties, "corners")

    status_idx = status.set_index("player_id") if not status.empty else None
    fixture_kick = ({(int(r.season), int(r.fixture_id)): r.kickoff
                     for r in fixtures.itertuples()}
                    if not fixtures.empty else {})

    # Map fixtures to teams for the target GW to look up Bzzoiro predictions
    gw_fixtures = fixtures[fixtures["gw"] == target_gw]
    opponent_map = {}
    fixture_map = {}
    for _, fx in gw_fixtures.iterrows():
        h_id, a_id = int(fx["home_team_id"]), int(fx["away_team_id"])
        opponent_map[h_id] = a_id
        opponent_map[a_id] = h_id
        fixture_map[h_id] = int(fx["fixture_id"])
        fixture_map[a_id] = int(fx["fixture_id"])

    teams_df = read_sql(select(T.teams).where(T.teams.c.season == season))
    team_id_to_name = {int(r.team_id): str(r.name).lower() for r in teams_df.itertuples()} if not teams_df.empty else {}

    congestion: dict[int, int] = {}
    window_start = cutoff - timedelta(days=14)
    if not fixtures.empty:
        for _, f in fixtures[fixtures["kickoff"].notna()].iterrows():
            ko = pd.Timestamp(f["kickoff"]).to_pydatetime()
            if window_start <= ko < cutoff:
                for t in (int(f["home_team_id"]), int(f["away_team_id"])):
                    congestion[t] = congestion.get(t, 0) + 1
    other_fixtures = _load_other_competition_fixtures(season, cutoff)
    if not other_fixtures.empty:
        for _, f in other_fixtures.iterrows():
            ko = pd.Timestamp(f["kickoff"]).to_pydatetime()
            if window_start <= ko < cutoff:
                congestion[int(f["team_id"])] = congestion.get(int(f["team_id"]), 0) + 1

    grouped = dict(tuple(history.groupby("player_id"))) if not history.empty else {}
    price_rank = _price_rank_in_team(players, status_idx)

    cohort_prices = ({int(pid): float(v) for pid, v in status_idx["now_cost"].items()}
                     if status_idx is not None and "now_cost" in status_idx else {})
    cohort_stats = _build_cohort_stats(players, grouped, team_out, team_pens, pen_share,
                                       price_by_player=cohort_prices)
    external = _load_external_records(season, cutoff)

    rows = []

    for _, p in players.iterrows():
        pid = int(p["player_id"])
        h = grouped.get(pid, pd.DataFrame()).tail(lookback)

        minutes_hist = h["minutes"].tolist() if not h.empty else []
        starts_hist = h["starts"].fillna(0).astype(int).tolist() if not h.empty and "starts" in h else []
        st = (status_idx.loc[pid].to_dict()
              if status_idx is not None and pid in status_idx.index else {})
        
        if strict_point_in_time:
            required_snapshot_fields = ("team_id", "position", "now_cost")
            if not st or any(
                key not in st or st.get(key) is None or pd.isna(st.get(key))
                for key in required_snapshot_fields
            ):
                continue
        
        pos = str(st.get("position") or p["position"])
        team_id = int(st.get("team_id") or p["team_id"])

        chance = st.get("chance_of_playing_next_round")
        chance = (None if chance is None or (isinstance(chance, float) and np.isnan(chance))
                  else float(chance))
        suspended, season_yellows = _suspension_due(h, st, target_gw)

        # --- EXTERNAL METRICS INJECTION ---
        p_name_lower = str(p["web_name"]).lower()
        player_ext = per90_data.get(p_name_lower, {})
        baseline_npxg90 = player_ext.get("npxg90", 0.0)
        baseline_xa90 = player_ext.get("xa90", 0.0)
        
        team_name = team_id_to_name.get(team_id, "")
        team_elo = elo_data.get(team_name, 1500.0)
        
        opp_team_id = opponent_map.get(team_id)
        opp_name = team_id_to_name.get(opp_team_id, "") if opp_team_id else ""
        opp_elo = elo_data.get(opp_name, 1500.0)
        elo_diff = team_elo - opp_elo
        
        set_piece_metrics = team_set_pieces.get(team_name, {})
        team_xgsp = set_piece_metrics.get("xgsp", 0.0)
        
        # Look up Bzzoiro prediction for the fixture
        fixture_id = fixture_map.get(team_id)
        bzz_pred = match_predictions.get(str(fixture_id)) if fixture_id else None
        # ----------------------------------

        feat = {
            "starts_last_5": float(sum(1 for x in starts_hist[-5:] if x > 0)),
            "minutes_mean_last_5": float(np.mean(minutes_hist[-5:])) if minutes_hist else 0.0,
            "minutes_mean_last_3": float(np.mean(minutes_hist[-3:])) if minutes_hist else 0.0,
            "played_last_match": 1.0 if minutes_hist and minutes_hist[-1] > 0 else 0.0,
            "started_last_match": 1.0 if starts_hist and starts_hist[-1] > 0 else 0.0,
            "days_since_last_start": _days_since_last_start(minutes_hist, starts_hist),
            "chance_of_playing": float(chance) if chance is not None else np.nan,
            "is_doubtful": 1.0 if str(st.get("status", "a")) == "d" else 0.0,
            "price_rank_in_team": float(price_rank.get(pid, np.nan)),
            "team_matches_last_14d": float(congestion.get(team_id, 0)),
            "season_minutes_share": (float(np.mean(minutes_hist) / 90.0)
                                     if minutes_hist else 0.0),
            "selected_by_percent": float(st.get("selected_by_percent") or 0.0),
            "elo_difference": float(elo_diff),
        }

        status_code = str(st.get("status", "a"))

        mp = None
        if minutes_model is not None and minutes_model.fitted and not suspended:
            mp = minutes_model.predict(feat)
        if mp is None:
            mp = heuristic_minutes(
                minutes_hist, status=status_code,
                chance_of_playing=chance,
                starts_history=starts_hist,
                days_since_return=_days_since_return(h, fixture_kick, cutoff),
                rotation_risk=_rotation_risk(minutes_hist, starts_hist),
                congestion_matches_14d=congestion.get(team_id, 0),
                is_suspended_next=suspended,
            )
        elif suspended or status_code in UNAVAILABLE:
            mp.p_play = mp.p_start_est = mp.p_sixty = mp.expected_minutes = 0.0
        else:
            _apply_doubtful_floor(mp, status_code, chance, suspended)

        declared_pen = pen_share.get(pid)
        if not h.empty:
            keys = ["season", "fixture_id", "was_home"]
            hh = h.merge(team_out, on=keys, how="left")
            hh = hh.merge(team_pens, on=keys, how="left")
            hh = hh.merge(team_xg_out, on=keys, how="left")
            hh = hh.merge(team_xa_out, on=keys, how="left")

            tg = hh["team_goals"].fillna(0.0).to_numpy()
            ta = hh["team_assists"].fillna(0.0).to_numpy()
            tp = hh["team_pens_est"].fillna(0.0).to_numpy()

            player_goals = hh["goals_scored"].fillna(0).to_numpy(dtype=float)
            pens_scored_est = sp.estimate_penalties_scored(
                player_goals, declared_pen if declared_pen is not None else 0.0, tp)

            metric = "goals"
            np_goals = np.maximum(player_goals - pens_scored_est, 0.0)
            np_team_goals = np.maximum(tg - tp, 0.0)
            if "expected_goals" in hh.columns and hh["expected_goals"].notna().any():
                player_xg = hh["expected_goals"].fillna(0).to_numpy(dtype=float)
                pens_xg_est = pens_scored_est * PENALTY_XG_PER_ATTEMPT
                np_goals = np.maximum(player_xg - pens_xg_est, 0.0)
                team_xg = hh["team_xg"].fillna(np.nan).to_numpy(dtype=float)
                if not np.isnan(team_xg).all():
                    team_xg = np.where(np.isnan(team_xg), tg, team_xg)
                    np_team_goals = np.maximum(team_xg - pens_xg_est, 0.0)
                    metric = "xg"

            share = estimate_shares(
                pos,
                player_np_goals=np_goals.tolist(),
                player_assists=hh["assists"].fillna(0).tolist(),
                team_np_goals=np_team_goals.tolist(),
                team_assists=ta.tolist(),
                minutes_played=hh["minutes"].fillna(0).tolist(),
                declared_penalty_share=declared_pen,
                player_pens_taken=pens_scored_est.tolist(),
                team_pens=tp.tolist(),
                metric=metric,
                player_expected_assists=(
                    hh["expected_assists"].fillna(0).tolist()
                    if "expected_assists" in hh and hh["expected_assists"].notna().any()
                    else None
                ),
                team_expected_assists=(
                    hh["team_xa"].fillna(0).tolist()
                    if "team_xa" in hh and hh["team_xa"].notna().any()
                    else None
                ),
                xa_blend_weight=PARAMS.xa_assist_blend_weight,
                baseline_npxg90=baseline_npxg90,
                baseline_xa90=baseline_xa90,
                blend_weight=PARAMS.external_api_blend_weight,
            )
            defcon_rate = defcon_mod.estimate_defcon_rate(
                pos, _defcon_series(hh, pos), hh["minutes"].fillna(0).tolist())
            card_rate = cards_mod.estimate_card_rate(
                pos, hh["yellow_cards"].fillna(0).tolist(),
                hh["red_cards"].fillna(0).tolist() if "red_cards" in hh else [],
                hh["minutes"].fillna(0).tolist())

            def _col(name: str) -> np.ndarray:
                return (hh[name].fillna(0).to_numpy(dtype=float)
                        if name in hh else np.zeros(len(hh)))
            mins_arr = _col("minutes")
            accounted = bonus_mod.simulate_bps(
                pos, mins_arr, _col("goals_scored"), _col("assists"),
                _col("goals_conceded"), _col("saves"),
                None, 0.0,
                penalties_saved=_col("penalties_saved"),
                penalties_missed=_col("penalties_missed"),
                own_goals=_col("own_goals"),
                yellow_cards=_col("yellow_cards"),
                red_cards=_col("red_cards"),
                cbi_actions=_col("clearances_blocks_interceptions")).tolist()
            baseline_bps = bonus_mod.estimate_baseline_bps_per90(
                hh["bps"].fillna(0).tolist(), mins_arr.tolist(), accounted)
        else:
            ext = external.get(pid)
            cohort = cohort_stats.get(pos, cohort_stats.get("MID", {}))
            if ext is not None:
                cs_prior = cold_start_mod.translated_prior(
                    position=pos, source_league=ext["league"],
                    foreign_np_xg90=ext["np_xg_per90"], foreign_xa90=ext["xa_per90"],
                    foreign_defcon90=ext.get("defcon_per90", 0.0),
                    foreign_minutes=ext["minutes"], age=None,
                    team_np_xg_per_match=1.4)
            else:
                cs_prior = cold_start_mod.cohort_prior(pos, float(p["now_cost"] or 0.0), cohort)
            share = estimate_shares(
                pos, [], [], [], None, [], declared_penalty_share=declared_pen,
                prior_override={"np_goal_share": cs_prior.np_goal_share,
                               "assist_share": cs_prior.assist_share,
                               "confidence": cs_prior.tier},
                baseline_npxg90=baseline_npxg90,
                baseline_xa90=baseline_xa90,
                blend_weight=PARAMS.external_api_blend_weight,
            )
            defcon_rate = defcon_mod.estimate_defcon_rate(pos, [], [])
            defcon_rate.per90 = max(cs_prior.defcon_per90, defcon_rate.per90)
            card_rate = cards_mod.estimate_card_rate(pos, [], [], [])
            baseline_bps = 8.0

        rows.append({
            "player_id": pid, "season": season, "web_name": p["web_name"],
            "team_id": team_id, "position": pos,
            "price": float(st.get("now_cost") or p["now_cost"] or 0.0),
            "p_play": mp.p_play, "p_start": mp.p_start, "p_sixty": mp.p_sixty,
            "expected_minutes": mp.expected_minutes,
            "minutes_source": mp.source, "minutes_drivers": mp.drivers,
            "np_goal_share": share.np_goal_share,
            "assist_share": share.assist_share,
            "penalty_share": share.penalty_share,
            "penalty_order": int(pen_order.get(pid, 99)),
            "fk_share": float(fk_share.get(pid, 0.0)),
            "fk_order": int(fk_order.get(pid, 99)),
            "corner_share": float(cnr_share.get(pid, 0.0)),
            "corner_order": int(cnr_order.get(pid, 99)),
            "share_confidence": share.confidence,
            "share_provenance": share.provenance,
            "defcon_per90": defcon_rate.per90,
            "defcon_dispersion": defcon_rate.dispersion,
            "defcon_confidence": defcon_rate.confidence,
            "yellow_per90": card_rate.yellow_per90,
            "red_per90": card_rate.red_per90,
            "card_confidence": card_rate.confidence,
            "baseline_bps_per90": baseline_bps,
            "selected_by_percent": float(st.get("selected_by_percent") or 0.0),
            "status": status_code,
            "news": st.get("news", ""),
            "unavailable": bool(suspended or status_code in ("i", "s", "u", "n")),
            "season_yellow_cards": season_yellows,
            "suspended": bool(suspended),
            "games_of_history": int(len(h)),
            "team_xgsp": team_xgsp,
            "bzzoiro_expected_goals_home": bzz_pred.get("expected_goals_home") if bzz_pred else None,
            "bzzoiro_expected_goals_away": bzz_pred.get("expected_goals_away") if bzz_pred else None,
            "bzzoiro_confidence": bzz_pred.get("confidence") if bzz_pred else None,
        })
    return pd.DataFrame(rows)


def _defcon_series(hh: pd.DataFrame, position: str) -> list[float]:
    col = defcon_mod.DEFCON_STAT.get(position)
    if col and col in hh.columns and hh[col].notna().any():
        return hh[col].fillna(0).tolist()
    return hh["defensive_contribution"].fillna(0).tolist()


def _build_cohort_stats(players: pd.DataFrame, grouped: dict,
                        team_out: pd.DataFrame, team_pens: pd.DataFrame,
                        pen_share: dict, price_by_player: dict[int, float] | None = None) -> dict:
    if not grouped or team_out.empty:
        return {}
    recs = []
    for _, p in players.iterrows():
        pid = int(p["player_id"])
        h = grouped.get(pid)
        if h is None or h.empty or len(h) < 2:
            continue
        keys = ["season", "fixture_id", "was_home"]
        hh = h.merge(team_out, on=keys, how="left")
        hh = hh.merge(team_pens, on=keys, how="left")
        tg = hh["team_goals"].fillna(0.0).to_numpy()
        tp = hh["team_pens_est"].fillna(0.0).to_numpy() if "team_pens_est" in hh else np.zeros(len(hh))
        goals = hh["goals_scored"].fillna(0).to_numpy(dtype=float)
        np_goals = np.maximum(goals - tp * pen_share.get(pid, 0.0), 0.0)
        np_team = np.maximum(tg - tp, 0.0)
        sh = estimate_shares(p["position"], np_goals.tolist(),
                             hh["assists"].fillna(0).tolist(), np_team.tolist(),
                             None, hh["minutes"].fillna(0).tolist())
        mins = hh["minutes"].fillna(0)
        price = float((price_by_player or {}).get(pid, p["now_cost"] or 0.0))
        recs.append({"position": p["position"], "price": price,
                    "np_goal_share": sh.np_goal_share, "assist_share": sh.assist_share,
                    "p_play": float((mins > 0).mean())})
    if not recs:
        return {}
    df = pd.DataFrame(recs)
    out: dict = {}
    for pos, g in df.groupby("position"):
        out[pos] = {"np_goal_share": float(g["np_goal_share"].mean()),
                    "assist_share": float(g["assist_share"].mean()),
                    "p_play": float(g["p_play"].mean()), "n": int(len(g)),
                    "defcon_per90": defcon_mod.DEFCON_PRIOR_PER90.get(pos, 5.0)}
    return out


def _load_external_records(season: int, cutoff: datetime | None = None) -> dict[int, dict]:
    try:
        from sqlalchemy import select
        from fplagent.db import tables as T
        from fplagent.db.repository import read_sql
        m_conds = [T.external_player_map.c.source == "fbref"]
        if cutoff is not None:
            m_conds.append(T.external_player_map.c.known_at < cutoff)
        m = read_sql(select(T.external_player_map).where(*m_conds))
        if m.empty:
            return {}
        e_conds = [(T.external_player_season.c.source == "fbref"),
                   (T.external_player_season.c.season == season)]
        if cutoff is not None:
            e_conds.append(T.external_player_season.c.known_at < cutoff)
        e = read_sql(select(T.external_player_season).where(*e_conds))
        if e.empty:
            return {}
    except Exception:
        return {}
    merged = m.merge(e, on="external_id", how="inner")
    out = {}
    for _, r in merged.iterrows():
        mins = float(r["minutes"] or 0)
        out[int(r["player_id"])] = {
            "league": r["league"], "minutes": mins,
            "np_xg_per90": float(r["np_xg"] or 0.0) / max(mins / 90.0, 1e-6),
            "xa_per90": float(r["xa"] or 0.0) / max(mins / 90.0, 1e-6),
        }
    return out


def _price_rank_in_team(players: pd.DataFrame, status_idx) -> dict[int, float]:
    if players.empty or status_idx is None or status_idx.empty:
        return {}
    if not {"now_cost", "team_id"}.issubset(status_idx.columns):
        return {}
    df = players[["player_id"]].drop_duplicates().copy()
    df["now_cost"] = df["player_id"].map(status_idx["now_cost"])
    df["team_id"] = df["player_id"].map(status_idx["team_id"])
    df = df.dropna(subset=["now_cost", "team_id"])
    if df.empty:
        return {}
    df["rank"] = df.groupby("team_id")["now_cost"].rank(ascending=False, method="min")
    return {int(r.player_id): float(r.rank) for r in df.itertuples()}


def _days_since_last_start(minutes_hist: list[int], starts_hist: list[int] | None = None) -> float:
    flags = starts_hist if starts_hist is not None else [int(m >= 60) for m in minutes_hist]
    for k, started in enumerate(reversed(flags), start=1):
        if started:
            return float(k)
    return 99.0


def _rotation_risk(minutes_hist: list[int], starts_hist: list[int] | None = None) -> float:
    if len(minutes_hist) < 4:
        return 0.0
    arr = np.array(minutes_hist[-6:], dtype=float)
    if starts_hist is None:
        started = (arr >= 60).astype(float)
    else:
        started = np.asarray(starts_hist[-len(arr):], dtype=float)
    if started.mean() in (0.0, 1.0):
        return 0.0
    return float(min(1.0, 4.0 * started.mean() * (1 - started.mean())))


def _days_since_return(h: pd.DataFrame, kickoffs: dict, cutoff: datetime) -> int | None:
    if h.empty or len(h) < 2:
        return None
    mins = h["minutes"].tolist()
    for i in range(len(mins) - 1, 0, -1):
        if mins[i] > 0 and mins[i - 1] == 0:
            row = h.iloc[i]
            ko = kickoffs.get((int(row["season"]), int(row["fixture_id"])))
            if ko is None or pd.isna(ko):
                return None
            return max((cutoff - pd.Timestamp(ko).to_pydatetime()).days, 0)
        if mins[i] == 0:
            continue
        break
    return None


def _suspension_due(h: pd.DataFrame, status: dict,
                    target_gw: int) -> tuple[bool, int]:
    season_yellows = 0
    if not h.empty:
        season_yellows = int(h["yellow_cards"].fillna(0).sum())
    snap = status.get("season_yellow_cards")
    if snap is not None and not (isinstance(snap, float) and np.isnan(snap)):
        season_yellows = int(snap)

    if h.empty:
        return False, season_yellows

    last = h.iloc[-1]
    if int(last.get("red_cards", 0) or 0) > 0:
        return True, season_yellows

    if int(last.get("yellow_cards", 0) or 0) > 0:
        for threshold, valid_until_gw in YELLOW_BAN_THRESHOLDS:
            if season_yellows >= threshold and target_gw <= valid_until_gw:
                before = season_yellows - int(last.get("yellow_cards", 0) or 0)
                if before < threshold:
                    return True, season_yellows
    return False, season_yellows


def build_minutes_training_set(cutoff: datetime,
                               season: int = CURRENT_SEASON
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build strictly point-in-time labels for P(play), P(start), P(60+)."""
    history = read_sql(select(T.player_fixture_stats).where(
        (T.player_fixture_stats.c.known_at < cutoff)
        & (T.player_fixture_stats.c.season == season)))
    if history.empty:
        empty = np.empty(0)
        return np.empty((0, len(FEATURE_ORDER))), empty, empty, empty

    fixtures = load_fixtures(season)
    kickoff = ({(int(r.season), int(r.fixture_id)): r.kickoff
                for r in fixtures.itertuples() if pd.notna(r.kickoff)}
               if not fixtures.empty else {})
    history = history.copy()
    history["_kickoff"] = [kickoff.get((int(se), int(fid)))
                            for se, fid in zip(history["season"], history["fixture_id"])]
    history = history[history["_kickoff"].notna()].copy()
    if history.empty:
        empty = np.empty(0)
        return np.empty((0, len(FEATURE_ORDER))), empty, empty, empty
    history["_kickoff"] = pd.to_datetime(history["_kickoff"])

    fixtures_by_key: dict[tuple[int, int], dict] = {}
    if not fixtures.empty:
        for r in fixtures.itertuples():
            fixtures_by_key[(int(r.season), int(r.fixture_id))] = {
                "home_team_id": int(r.home_team_id),
                "away_team_id": int(r.away_team_id),
            }

    snaps = read_sql(select(T.player_season_snapshots).where(
        (T.player_season_snapshots.c.season == season)
        & (T.player_season_snapshots.c.pulled_at < cutoff)))
    snaps_by_player = ({int(pid): g.sort_values("pulled_at").reset_index(drop=True)
                        for pid, g in snaps.groupby("player_id")}
                       if not snaps.empty else {})

    def _snapshot_before(pid: int, target_kickoff: pd.Timestamp) -> dict:
        g = snaps_by_player.get(pid)
        if g is None or g.empty:
            return {}
        idx = int(g["pulled_at"].searchsorted(target_kickoff, side="left"))
        return g.iloc[idx - 1].to_dict() if idx > 0 else {}

    rank_cache: dict[tuple[int, int], dict[int, float]] = {}
    if not snaps.empty and "team_id" in snaps.columns and "now_cost" in snaps.columns:
        for (fseason, fid), ko in kickoff.items():
            ko_ts = pd.Timestamp(ko)
            before = snaps[snaps["pulled_at"] < ko_ts].copy()
            if before.empty:
                continue
            before = (before.sort_values("pulled_at")
                            .groupby("player_id", as_index=False).tail(1))
            before = before[before["now_cost"].notna() & before["team_id"].notna()].copy()
            if before.empty:
                continue
            before["price_rank"] = before.groupby("team_id")["now_cost"].rank(
                ascending=False, method="min")
            rank_cache[(fseason, fid)] = {
                int(r.player_id): float(r.price_rank) for r in before.itertuples()
            }

    other = _load_other_competition_fixtures(season, cutoff)
    league_fixture_rows = list(fixtures[fixtures["kickoff"].notna()].itertuples()) \
        if not fixtures.empty else []

    def _team_congestion(team_id: int, ko: pd.Timestamp) -> int:
        target = ko.to_pydatetime()
        window = target - timedelta(days=14)
        count = 0
        for f in league_fixture_rows:
            if f.home_team_id != team_id and f.away_team_id != team_id:
                continue
            fko = pd.Timestamp(f.kickoff).to_pydatetime()
            if window <= fko < target and bool(f.finished):
                count += 1
        if not other.empty:
            for r in other.itertuples():
                if int(r.team_id) != team_id or pd.isna(r.kickoff):
                    continue
                fko = pd.Timestamp(r.kickoff).to_pydatetime()
                if window <= fko < target:
                    count += 1
        return count

    history = history.sort_values(["player_id", "_kickoff", "fixture_id"])
    rows: list[tuple] = []
    for pid, g in history.groupby("player_id", sort=False):
        g = g.sort_values(["_kickoff", "fixture_id"]).reset_index(drop=True)
        for i, target in g.iterrows():
            target_ko = pd.Timestamp(target["_kickoff"])
            prior = g[(g["_kickoff"] < target_ko)
                       & (pd.to_datetime(g["known_at"]) < target_ko)].copy()
            if len(prior) < 3:
                continue
            mins = prior["minutes"].fillna(0).astype(int).tolist()
            starts = prior["starts"].fillna(0).astype(int).tolist()
            st = _snapshot_before(int(pid), target_ko)
            chance = st.get("chance_of_playing_next_round")
            chance = (np.nan if chance is None or
                      (isinstance(chance, float) and np.isnan(chance))
                      else float(chance))
            past5, past3 = mins[-5:], mins[-3:]
            past_starts5 = starts[-5:]

            target_team_id: int | None = None
            fx = fixtures_by_key.get((int(target["season"]), int(target["fixture_id"])))
            if fx is not None:
                was_home = bool(target.get("was_home", False))
                target_team_id = (fx["home_team_id"] if was_home
                                  else fx["away_team_id"])
            if target_team_id is None:
                target_team_id = st.get("team_id")
            congestion = (_team_congestion(int(target_team_id), target_ko)
                          if target_team_id is not None else np.nan)
            
            x = [
                float(sum(1 for x0 in past_starts5 if x0 > 0)),
                float(np.mean(past5)) if past5 else 0.0,
                float(np.mean(past3)) if past3 else 0.0,
                1.0 if mins[-1] > 0 else 0.0,
                1.0 if starts[-1] > 0 else 0.0,
                _days_since_last_start(mins, starts),
                chance,
                1.0 if str(st.get("status", "a")) == "d" else 0.0,
                float(rank_cache.get((int(target["season"]), int(target["fixture_id"])), {})
                      .get(int(pid), np.nan)),
                float(congestion),
                float(np.mean(mins) / 90.0) if mins else 0.0,
                float(st.get("selected_by_percent") or 0.0),
                0.0, # Elo difference placeholder for historical training
            ]
            target_minutes = int(target["minutes"] or 0)
            target_started = int(target["starts"] or 0) > 0
            rows.append((target["season"], target["gw"], target["fixture_id"], x,
                         1 if target_minutes > 0 else 0,
                         1 if target_started else 0,
                         1 if target_minutes >= 60 else 0))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    if not rows:
        empty = np.empty(0)
        return np.empty((0, len(FEATURE_ORDER))), empty, empty, empty
    X = np.asarray([r[3] for r in rows], dtype=float)
    y_play = np.asarray([r[4] for r in rows], dtype=int)
    y_start = np.asarray([r[5] for r in rows], dtype=int)
    y_sixty = np.asarray([r[6] for r in rows], dtype=int)
    return X, y_play, y_start, y_sixty