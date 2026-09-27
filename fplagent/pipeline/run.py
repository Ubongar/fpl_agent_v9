"""The daily orchestrator. One command, whole system.

Replaces v5's ops/retrain_job.py, which ingested a snapshot, evaluated a
hardcoded three-literal calibration curve, and exited 1 -- unconditionally,
on every run, forever.

Stage order and failure policy:

    ingest        FAIL HARD on input quality gates. Bad data in is the one
                  thing you cannot recover from downstream.
    ownership     SOFT-FAIL. Calls FPL's public-but-unofficial picks
                  endpoints. Rate limits and transient outages are expected;
                  the objective falls back to raw ownership when it fails.
    fit/predict   FAIL HARD on output sanity (unordered quantiles,
                  probabilities outside [0,1], zero spread).
    evaluate      NEVER fails the run. Evaluation reports; it does not gate.
                  A model that scores badly at GW5 should still produce
                  predictions -- you cannot improve what you cannot see. What
                  a bad evaluation does is refuse to PROMOTE the version.

    promote       Gated on beating the baselines, which is a real bar
                  rather than a noise-dominated accuracy threshold.

The distinction between "produce output" and "promote a version" is the one
v5 collapsed, and collapsing it is what made bypassing the gate feel
necessary.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime

from sqlalchemy import select

from fplagent.config import CURRENT_SEASON
from fplagent.db import tables as T
from fplagent.db.repository import (
    insert_rows,
    latest_snapshot_per_player,
    read_sql,
    resolve_target_gw,
    upsert,
)
from fplagent.db.session import init_db

log = logging.getLogger(__name__)


def run_daily(
    season: int = CURRENT_SEASON,
    target_gw: int | None = None,
    skip_ingest: bool = False,
    limit_players: int | None = None,
    n_sims: int | None = None,
    evaluate: bool = True,
    use_cache: bool = False,
    fbref_csv: str | None = None,
    sample_ownership: bool = True,
    ownership_n_managers: int = 200,
) -> dict:
    init_db()
    started = datetime.utcnow()
    stages: dict[str, dict] = {}
    version = None
    status = "running"
    error = None

    try:
        # ---------------- ingest ----------------------------------------
        if not skip_ingest:
            from fplagent.ingestion.pipeline import run_full_ingest

            stages["ingest"] = run_full_ingest(
                season,
                limit_players,
                use_cache,
                fbref_csv=fbref_csv,
            )

            # The market is the strongest free prior available and v6 never
            # wrote a single row of it. Soft-fails on a missing key.
            from fplagent.ingestion.odds import ingest_odds

            stages["odds"] = {"fixtures_priced": ingest_odds(season)}

            log.info("ingest complete: %s", stages["ingest"])
        else:
            stages["ingest"] = {"skipped": True}

        # ---------------- resolve the gameweek from the API, not config --
        gw = target_gw or resolve_target_gw(season)

        if gw is None:
            raise RuntimeError(
                "Could not resolve a target gameweek. Run `fplagent ingest` "
                "first so the gameweeks table is populated from the FPL API."
            )

        stages["target_gw"] = {
            "gw": gw,
            "source": "fpl_api_events",
        }

        # ---------------- effective ownership (SOFT-FAIL) ----------------
        # The rank-aware objective's differential term reads EO and falls
        # back to raw ownership when it is missing. That fallback is a real
        # degradation: ownership and EO diverge hardest on the captaincy-
        # heavy premiums, which are exactly the players the risk tilt is
        # supposed to weight correctly. Sampling is optional and can be
        # turned off for CI or offline runs.
        if sample_ownership:
            try:
                from fplagent.ingestion.ownership import sample_effective_ownership
                sample_gw = gw - 1 if gw > 1 else gw
                stages["ownership"] = {
                    "sample_gw": sample_gw,
                    "players_sampled": sample_effective_ownership(
                        sample_gw, season, n_managers=ownership_n_managers
                    )
                }
            except Exception as e:      # noqa: BLE001
                log.warning(
                    "effective ownership sampling failed (non-fatal): %s", e
                )
                stages["ownership"] = {"error": str(e)}
        else:
            stages["ownership"] = {"skipped": True}

        # ---------------- predict ---------------------------------------
        from fplagent.pipeline.predict import model_version, run_predictions

        version = model_version()

        preds = run_predictions(
            gw,
            season,
            version=version,
            n_sims=n_sims,
        )

        stages["predict"] = {
            "model_version": version,
            "rows": int(len(preds)),
            "gw": gw,
            "top": (
                preds.nlargest(5, "xp_mean")[
                    ["player_id", "xp_mean"]
                ].to_dict("records")
                if not preds.empty
                else []
            ),
        }

        # ---------------- evaluate (reports, never blocks) ---------------
        if evaluate and gw > 2:
            try:
                from fplagent.evaluation.backtest import backtest

                stages["evaluate"] = backtest(
                    season,
                    gw_from=2,
                    gw_to=gw - 1,
                )

            except Exception as e:  # noqa: BLE001
                log.warning(
                    "evaluation failed (non-fatal): %s",
                    e,
                )

                stages["evaluate"] = {
                    "error": str(e),
                }

        # ---------------- promotion decision ----------------------------
        ev = stages.get("evaluate", {})

        promote: bool | None = None

        if ev.get("n", 0) >= 50:
            promote = bool(ev.get("beat_best_baseline"))

            stages["promote"] = {
                "promoted": promote,
                "reason": (
                    "beat every baseline on rank correlation"
                    if promote
                    else
                    "did NOT beat all baselines -- predictions were still "
                    "written, but this version is not marked current"
                ),
                "comparison": ev.get("baseline_comparison"),
            }

        else:
            # The message below is deliberately explicit about WHY there
            # are no scored rows. On a fresh deployment the backtest reads
            # stored predictions created before each historical deadline;
            # if the pipeline has only been running for a few days, none
            # exist yet, and "n=0" is expected rather than a failure. Users
            # who saw the old "not enough evidence" message reasonably
            # assumed something was broken.
            stages["promote"] = {
                "promoted": None,
                "reason": (
                    f"only {ev.get('n', 0)} scored rows so far. This is "
                    "expected on a fresh deployment: the backtest scores "
                    "predictions written BEFORE each historical deadline, "
                    "and the pipeline has not been running long enough for "
                    "any to exist yet. Predictions are written and the "
                    "system is fully usable. Promotion becomes possible "
                    "once 2-3 live gameweeks are behind you."
                ),
            }

        # The decision is recorded where it can be read. It used to live
        # only inside pipeline_runs.stages, and `model_versions` -- the table
        # built to hold it -- was never written.
        upsert(
            T.model_versions,
            [
                {
                    "model_version": version,
                    "season": season,
                    "target_gw": gw,
                    "gws": [gw],
                    "promoted": promote,
                    "promotion_reason": stages["promote"]["reason"],
                    "evaluation": {
                        k: ev.get(k)
                        for k in (
                            "n",
                            "mae",
                            "spearman_overall",
                            "beat_best_baseline",
                            "baseline_comparison",
                        )
                        if k in ev
                    },
                }
            ],
            ["model_version"],
        )

        status = "ok"

    except Exception as e:  # noqa: BLE001
        status = "failed"
        error = f"{e}\n{traceback.format_exc()}"

        log.error(
            "pipeline failed: %s",
            e,
        )

    insert_rows(
        T.pipeline_runs,
        [
            {
                "model_version": version,
                "season": season,
                "gw": stages.get("target_gw", {}).get("gw"),
                "started_at": started,
                "finished_at": datetime.utcnow(),
                "status": status,
                "stages": stages,
                "error": error,
            }
        ],
    )

    return {
        "status": status,
        "model_version": version,
        "stages": stages,
        "error": error,
    }


def latest_predictions(
    season: int = CURRENT_SEASON,
    gw: int | None = None,
):
    """Most recent model version's predictions, aggregated to gameweek level.

    With no `gw`, this is ONE gameweek: the target gameweek from the stored
    FPL events when it has predictions, else the newest version's first
    gameweek. v7 returned every gameweek the newest version touched, so any
    version spanning two gameweeks listed each player twice -- enough to
    make /optimise/transfers reject a valid 15-man squad.

    If the target gameweek has no predictions yet, this returns an EMPTY
    frame rather than falling back to an older gameweek. The previous
    behaviour silently returned the last predicted gameweek as if it were
    the target, which was fine in the demo (target == predicted) and
    actively misleading on live data where a gap of one or more gameweeks
    is easy to hit. Silent stale data is worse than an honest empty frame.
    """

    import pandas as pd

    conds = [
        T.predictions.c.season == season,
    ]

    if gw:
        conds.append(
            T.predictions.c.gw == gw
        )

    df = read_sql(
        select(T.predictions).where(*conds)
    )

    if df.empty:
        return df

    if not gw:
        target = resolve_target_gw(season)

        if target is not None:
            if (df["gw"] == target).any():
                df = df[df["gw"] == target]
            else:
                log.warning(
                    "latest_predictions: no predictions stored for target "
                    "GW%s (newest stored is GW%s). Returning an empty frame "
                    "rather than stale data from an earlier gameweek.",
                    target, df["gw"].max(),
                )
                return df.iloc[0:0]

    latest = (
        df.sort_values("created_at")["model_version"].iloc[-1]
    )

    df = df[df["model_version"] == latest]

    if not gw:
        df = df[df["gw"] == df["gw"].min()]

    players = read_sql(
        select(T.players).where(
            T.players.c.season == season
        )
    )

    teams = read_sql(
        select(T.teams).where(
            T.teams.c.season == season
        )
    )

    # Double gameweeks are convolved, not min/max-ed. v6 took xp_p10 as the
    # MINIMUM across fixtures and xp_p90 as the MAXIMUM, which are not the
    # gameweek's percentiles and are not even close for a two-fixture week.
    # Every consumer of this function -- the squad ILP, captaincy, the UI --
    # was reading ceilings that had never been computed.
    from fplagent.pipeline.predict import aggregate_to_gameweek

    agg = aggregate_to_gameweek(df)

    # Enrich predictions with player metadata.
    #
    # aggregate_to_gameweek() may already contain team_id, position and
    # now_cost. Merging those same columns from players would create
    # team_id_x/team_id_y (and similar columns), leaving no plain team_id
    # for the team-name merge below.
    player_meta = players[
        [
            "player_id",
            "web_name",
            "position",
            "team_id",
            "now_cost",
        ]
    ].drop_duplicates("player_id")

    # web_name is not normally present in the aggregate, so merge it
    # directly.
    agg = agg.merge(
        player_meta[
            [
                "player_id",
                "web_name",
            ]
        ],
        on="player_id",
        how="left",
    )

    # Use player metadata as a fallback when the prediction aggregate
    # does not contain these fields.
    for column in (
        "position",
        "team_id",
        "now_cost",
    ):
        if column in agg.columns:
            metadata = player_meta.set_index("player_id")[column]

            agg[column] = agg[column].fillna(
                agg["player_id"].map(metadata)
            )

        else:
            metadata = player_meta[
                [
                    "player_id",
                    column,
                ]
            ]

            agg = agg.merge(
                metadata,
                on="player_id",
                how="left",
            )

    # Add the team's short name using the clean team_id column.
    if not teams.empty:
        agg = agg.merge(
            teams[
                [
                    "team_id",
                    "short_name",
                ]
            ],
            on="team_id",
            how="left",
        )

    # Ownership lives only in the snapshots. Without it every consumer that
    # reads `selected_by_percent` -- the captaincy objective's template
    # discount above all -- silently saw 0% for every player.
    snaps = latest_snapshot_per_player(season)

    if (
        not snaps.empty
        and "selected_by_percent" in snaps
    ):
        agg = agg.merge(
            snaps[
                [
                    "player_id",
                    "selected_by_percent",
                ]
            ],
            on="player_id",
            how="left",
        )

    # Prevent division by zero when calculating value.
    agg["value"] = (
        agg["xp_mean"]
        / agg["now_cost"].replace(
            0,
            float("nan"),
        )
    ).round(3)

    agg["model_version"] = latest

    return (
        agg.sort_values(
            "xp_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )