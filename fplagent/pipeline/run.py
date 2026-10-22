"""The daily orchestrator. One command, whole system."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

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

STATUS_FILE = Path("data/pipeline_status.json")

def _update_status(status: str, message: str):
    STATUS_FILE.parent.mkdir(exist_ok=True, parents=True)
    STATUS_FILE.write_text(json.dumps({"status": status, "message": message}))


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
        _update_status("running", "Ingesting FPL API and Odds...")
        # ---------------- ingest ----------------------------------------
        if not skip_ingest:
            from fplagent.ingestion.pipeline import run_full_ingest

            stages["ingest"] = run_full_ingest(
                season,
                limit_players,
                use_cache,
                fbref_csv=fbref_csv,
            )

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

        _update_status("running", "Sampling Effective Ownership...")
        # ---------------- effective ownership (SOFT-FAIL) ----------------
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

        _update_status("running", "Training Minutes Model & Running Predictions...")
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

        _update_status("running", "Evaluating Model Performance...")
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
        _update_status("idle", "Pipeline finished successfully.")

    except Exception as e:  # noqa: BLE001
        status = "failed"
        error = f"{e}\n{traceback.format_exc()}"
        _update_status("error", f"Pipeline failed: {str(e)}")

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
    """Most recent model version's predictions, aggregated to gameweek level."""
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

    from fplagent.pipeline.predict import aggregate_to_gameweek

    agg = aggregate_to_gameweek(df)

    player_meta = players[
        [
            "player_id",
            "web_name",
            "position",
            "team_id",
            "now_cost",
        ]
    ].drop_duplicates("player_id")

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