"""Command line interface.

    fplagent init-db
    fplagent ingest [--limit 50]
    fplagent predict [--gw 6]
    fplagent run                      # the whole daily pipeline
    fplagent evaluate --from 2 --to 5
    fplagent squad --budget 100
    fplagent captains
    fplagent chips [--entry-id]       # chip schedule across the horizon
    fplagent transfers [--entry-id]   # multi-GW transfer plan
    fplagent changes --from 5 --to 6
    fplagent demo                     # seeded synthetic season, no network
    fplagent doctor                   # diagnose the environment
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_init_db(args) -> int:
    from fplagent.db.session import init_db
    tables = init_db(drop=args.drop)
    try:
        from fplagent.ingestion.league_snapshot import load_bundled_fit
        n = load_bundled_fit()
        if n:
            print(f"Loaded {n} real (data-fitted) league-translation ratios "
                 f"from data/league_translation_fitted.json")
    except Exception as e:      # noqa: BLE001
        print(f"(could not load bundled league translation fit: {e})", file=sys.stderr)
    print(f"Created {len(tables)} tables:")
    for t in tables:
        print(f"  - {t}")
    return 0


def cmd_ingest(args) -> int:
    from fplagent.db.session import init_db
    from fplagent.ingestion.pipeline import run_full_ingest
    init_db()
    _print(run_full_ingest(limit_players=args.limit, use_cache=args.cache,
                           fbref_csv=args.fbref_csv))
    return 0


def cmd_predict(args) -> int:
    from fplagent.db.repository import resolve_target_gw
    from fplagent.config import CURRENT_SEASON
    from fplagent.pipeline.predict import run_predictions
    gw = args.gw or resolve_target_gw(CURRENT_SEASON)
    if gw is None:
        print("No target gameweek found. Run `fplagent ingest` first.", file=sys.stderr)
        return 1
    df = run_predictions(gw, n_sims=args.sims)
    if df.empty:
        print("No predictions produced.", file=sys.stderr)
        return 1
    from fplagent.pipeline.run import latest_predictions
    top = latest_predictions(gw=gw).head(args.top)
    cols = ["web_name", "position", "short_name", "now_cost", "xp_mean",
            "xp_p10", "xp_p90", "p_play", "p_haul_10plus"]
    cols = [c for c in cols if c in top.columns]
    print(f"\nGW{gw} -- top {args.top} by expected points\n")
    print(top[cols].to_string(index=False))
    return 0


def cmd_run(args) -> int:
    from fplagent.pipeline.run import run_daily
    result = run_daily(target_gw=args.gw, skip_ingest=args.skip_ingest,
                       limit_players=args.limit, n_sims=args.sims,
                       use_cache=args.cache, fbref_csv=args.fbref_csv)
    _print(result)
    return 0 if result["status"] == "ok" else 1


def cmd_evaluate(args) -> int:
    from fplagent.evaluation.backtest import backtest
    _print(backtest(gw_from=args.gw_from, gw_to=args.gw_to, replay=args.replay))
    return 0


def cmd_squad(args) -> int:
    from fplagent.optimizers.squad import optimise_squad
    from fplagent.pipeline.run import latest_predictions
    preds = latest_predictions()
    if preds.empty:
        print("No predictions stored. Run `fplagent predict` first.", file=sys.stderr)
        return 1
    pool = [{
        "player_id": int(r.player_id), "name": r.web_name,
        "position": r.position, "club": str(getattr(r, "short_name", r.team_id)),
        "price": float(r.now_cost or 4.0), "xp": float(r.xp_mean),
        "p_play": float(r.p_play), "xp_p90": float(r.xp_p90),
        "eo": float(getattr(r, "selected_by_percent", 0.0) or 0.0),
    } for r in preds.itertuples() if r.now_cost]
    sol = optimise_squad(pool, budget=args.budget, risk_lambda=args.risk)
    print(json.dumps(sol.summary(), indent=2))
    print("\nStarting XI:")
    for p in sorted(sol.starting_xi, key=lambda x: ["GK", "DEF", "MID", "FWD"].index(x["position"])):
        mark = " (C)" if sol.captain and p["player_id"] == sol.captain["player_id"] else ""
        print(f"  {p['position']:<4} {p['name']:<18} {p['price']:>5.1f}  {p['xp']:>5.2f}{mark}")
    print("Bench:")
    for p in sol.bench:
        print(f"  {p['position']:<4} {p['name']:<18} {p['price']:>5.1f}  {p['xp']:>5.2f}")
    return 0


def cmd_my_team(args) -> int:
    from fplagent.ingestion.manager_team import get_my_team
    try:
        team = get_my_team(args.entry_id, gw=args.gw)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"{team.team_name} ({team.manager_name}), GW{team.gw}")
    print(f"  bank: {team.bank:.1f}m  value: {team.team_value:.1f}m  "
         f"free transfers: {team.free_transfers}"
         f"{' (estimated)' if team.free_transfers_is_estimated else ''}")
    if team.active_chip:
        print(f"  active chip: {team.active_chip}")
    print(f"  squad: {', '.join(str(p.player_id) for p in team.picks)}")
    known = sum(1 for p in team.picks if p.purchase_price_known)
    print(f"  purchase price known for {known}/{len(team.picks)} players "
         f"(the rest have held since before their first recorded transfer)")
    return 0


def cmd_captains(args) -> int:
    from sqlalchemy import select
    from fplagent.db import tables as T
    from fplagent.db.repository import read_sql
    from fplagent.optimizers.captaincy import rank_captains
    from fplagent.pipeline.run import latest_predictions
    preds = latest_predictions()
    if preds.empty:
        print("No predictions stored.", file=sys.stderr)
        return 1
    try:
        eo_df = read_sql(select(T.effective_ownership))
        real_eo = (dict(zip(eo_df.sort_values("gw").groupby("player_id").tail(1)
                            ["player_id"].astype(int),
                            eo_df.sort_values("gw").groupby("player_id").tail(1)
                            ["effective_ownership"].astype(float)))
                   if not eo_df.empty else {})
    except Exception:      # noqa: BLE001
        real_eo = {}
    pool = preds.nlargest(min(60, len(preds)), "xp_mean")
    cands = [{
        "player_id": int(r.player_id), "name": r.web_name, "xp": float(r.xp_mean),
        "p10": float(r.xp_p10), "p90": float(r.xp_p90),
        "p_haul_10plus": float(r.p_haul_10plus), "p_play": float(r.p_play),
        "selected_by_percent": float(getattr(r, "selected_by_percent", 0.0) or 0.0),
        **({"eo": real_eo[int(r.player_id)]} if int(r.player_id) in real_eo else {}),
    } for r in pool.itertuples()]
    for c in rank_captains(cands, risk_lambda=args.risk):
        note = f"  <- {c['risk_note']}" if c["risk_note"] else ""
        print(f"{c['name']:<18} xP {c['xp']:>5.2f}  x2 {c['expected_captain_points']:>5.2f}  "
              f"ceiling {c['ceiling_p90']:>5.1f}  haul {c['p_haul']:.0%}{note}")
    return 0


def _ensure_horizon_predictions(season: int, start_gw: int, horizon: int,
                                n_sims: int | None = None) -> list[int]:
    """Make sure stored predictions exist for every gameweek in
    [start_gw, start_gw + horizon), returning the list of gameweeks that
    were actually predicted (skip blanks, postponed fixtures, etc)."""
    from fplagent.db import tables as T
    from fplagent.db.repository import read_sql
    from fplagent.pipeline.predict import run_predictions
    from sqlalchemy import select

    gws = list(range(start_gw, start_gw + horizon))
    existing = set(read_sql(select(T.predictions.c.gw).where(
        (T.predictions.c.season == season)
        & (T.predictions.c.gw.in_(gws))
    ))["gw"].tolist())

    made: list[int] = list(existing)
    for gw in gws:
        if gw in existing:
            continue
        try:
            df = run_predictions(gw, season, n_sims=n_sims)
            if not df.empty:
                made.append(gw)
        except Exception as e:      # noqa: BLE001
            print(f"(prediction for GW{gw} failed: {e})", file=sys.stderr)
    return sorted(made)


def _load_user_squad_for_chips(entry_id: int, season: int, gw: int) -> list[dict] | None:
    """Fetch a real manager's picks and hydrate with player metadata for
    the chip evaluator. Returns None on any failure so the caller falls
    back to the optimal-squad reference rather than crashing.

    The returned dicts carry only the fields the chip evaluator needs at
    construction time -- player identity, position, club, price. The `xp`
    for each player is looked up per gameweek inside `compute_chip_values`,
    so nothing xP-related is set here.
    """
    from sqlalchemy import select
    from fplagent.db import tables as T
    from fplagent.db.repository import read_sql
    from fplagent.ingestion.manager_team import get_my_team

    try:
        team = get_my_team(entry_id, gw=gw)
    except Exception as e:      # noqa: BLE001
        print(f"(my-team failed: {e})", file=sys.stderr)
        return None
    ids = [int(p.player_id) for p in team.picks]
    if not ids:
        return None

    pdf = read_sql(select(T.players).where(
        (T.players.c.season == season)
        & (T.players.c.player_id.in_(ids))))
    if pdf.empty:
        return None

    teams = read_sql(select(T.teams).where(T.teams.c.season == season))
    short_by_team = ({int(r.team_id): str(r.short_name) for r in teams.itertuples()}
                     if not teams.empty else {})

    pool: list[dict] = []
    for r in pdf.itertuples():
        pool.append({
            "player_id": int(r.player_id),
            "name": str(r.web_name),
            "position": str(r.position),
            "club": short_by_team.get(int(r.team_id), str(r.team_id)),
            "price": float(r.now_cost or 4.0),
        })
    return pool


def cmd_chips(args) -> int:
    """Compute per-chip values across a horizon and schedule them."""
    from fplagent.config import CURRENT_SEASON
    from fplagent.db.repository import resolve_target_gw
    from fplagent.pipeline.run import latest_predictions
    from fplagent.optimizers.chip_values import compute_chip_values
    from fplagent.optimizers.chips import schedule_chips

    season = args.season or CURRENT_SEASON
    start_gw = args.gw or resolve_target_gw(season)
    if start_gw is None:
        print("No target gameweek found. Run `fplagent ingest` first.", file=sys.stderr)
        return 1

    horizon = args.horizon
    gws_predicted = _ensure_horizon_predictions(season, start_gw, horizon,
                                                n_sims=args.sims)
    if not gws_predicted:
        print("No predictions available for any gameweek in the horizon.",
              file=sys.stderr)
        return 1

    predictions_by_gw = {gw: latest_predictions(season, gw) for gw in gws_predicted}
    predictions_by_gw = {k: v for k, v in predictions_by_gw.items() if not v.empty}
    if not predictions_by_gw:
        print("No predictions stored for any gameweek in the horizon.",
              file=sys.stderr)
        return 1

    # If a real FPL team id is supplied, build the reference squad once.
    # The chip_values module re-evaluates those players against each
    # gameweek's predictions, so only the identity/position/club/price are
    # needed here.
    current_squad = None
    if getattr(args, "entry_id", None):
        current_squad = _load_user_squad_for_chips(args.entry_id, season, start_gw)
        if current_squad is None:
            print(f"(could not load team {args.entry_id}; falling back to "
                  f"optimal-squad reference)", file=sys.stderr)

    cv = compute_chip_values(predictions_by_gw, gws_predicted,
                             current_squad=current_squad)
    schedule = schedule_chips(cv.values, available_from_gw=start_gw)

    print(f"Chip schedule from GW{start_gw}, horizon {gws_predicted[0]}-{gws_predicted[-1]}")
    print(json.dumps({
        "assignments": schedule.assignments,
        "total_value": round(schedule.total_value, 2),
        "status": schedule.status,
        "notes": schedule.notes + cv.notes,
        "per_gw_values": {chip: {int(g): round(v, 2) for g, v in vals.items()}
                          for chip, vals in cv.values.items()},
    }, indent=2))
    return 0


def cmd_transfers(args) -> int:
    """Multi-gameweek transfer plan."""
    from fplagent.config import CURRENT_SEASON
    from fplagent.db.repository import resolve_target_gw
    from fplagent.pipeline.run import latest_predictions
    from fplagent.optimizers.transfers import plan_transfers

    season = args.season or CURRENT_SEASON
    start_gw = args.gw or resolve_target_gw(season)
    if start_gw is None:
        print("No target gameweek found.", file=sys.stderr)
        return 1

    # The current squad either comes from the user's FPL team, or from the
    # optimal squad if no entry id was supplied. Building the optimal squad
    # on the fly for the first gameweek is fine -- it is what the manager
    # would have if they followed the model's advice to the letter.
    if args.entry_id:
        from fplagent.ingestion.manager_team import get_my_team
        try:
            team = get_my_team(args.entry_id, gw=start_gw)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        # manager_team returns objects with player_id and current price; we
        # need to hydrate them from `players` for position/club.
        from sqlalchemy import select
        from fplagent.db import tables as T
        from fplagent.db.repository import read_sql
        ids = [p.player_id for p in team.picks]
        pdf = read_sql(select(T.players).where(
            (T.players.c.season == season)
            & (T.players.c.player_id.in_(ids))))
        price_by_id = {int(r.player_id): float(r.now_cost or 4.0) for r in pdf.itertuples()}
        current_squad = [{
            "player_id": int(r.player_id), "name": str(r.web_name),
            "position": str(r.position), "club": str(r.team_id),
            "price": price_by_id.get(int(r.player_id), 4.0),
        } for r in pdf.itertuples()]
        bank = float(getattr(team, "bank", 0.0) or 0.0)
        free_transfers = int(getattr(team, "free_transfers", 1) or 1)
    else:
        from fplagent.optimizers.squad import optimise_squad
        preds_first = latest_predictions(season, start_gw)
        if preds_first.empty:
            print(f"No predictions for GW{start_gw}.", file=sys.stderr)
            return 1
        pool = [{
            "player_id": int(r.player_id), "name": r.web_name,
            "position": r.position, "club": str(getattr(r, "short_name", r.team_id)),
            "price": float(r.now_cost or 4.0), "xp": float(r.xp_mean),
            "p_play": float(r.p_play), "xp_p90": float(r.xp_p90),
        } for r in preds_first.itertuples() if r.now_cost]
        sol = optimise_squad(pool, budget=100.0)
        if sol.status != "Optimal":
            print(f"Could not build a reference squad: {sol.status}", file=sys.stderr)
            return 1
        current_squad = [{
            "player_id": int(p["player_id"]), "name": p["name"],
            "position": p["position"], "club": p["club"], "price": p["price"],
        } for p in sol.squad]
        bank = 0.0
        free_transfers = 1

    horizon = args.horizon
    gws_predicted = _ensure_horizon_predictions(season, start_gw, horizon,
                                                n_sims=args.sims)
    if not gws_predicted:
        print("No predictions available for any gameweek in the horizon.",
              file=sys.stderr)
        return 1

    xp_by_gw: dict[int, dict[int, float]] = {}
    for gw in gws_predicted:
        p = latest_predictions(season, gw)
        if p.empty:
            continue
        xp_by_gw[gw] = {int(r.player_id): float(r.xp_mean) for r in p.itertuples()}

    # Candidate pool is the union of every GW's predictions, deduplicated by
    # player_id with the CURRENT price (the plan is priced at today's market,
    # not at the horizon's starting price, which is a simplification but a
    # defensible one when the horizon is short).
    universe: dict[int, dict] = {}
    for p in gws_predicted:
        preds = latest_predictions(season, p)
        if preds.empty:
            continue
        for r in preds.itertuples():
            pid = int(r.player_id)
            if pid not in universe:
                universe[pid] = {
                    "player_id": pid, "name": str(r.web_name),
                    "position": str(r.position),
                    "club": str(getattr(r, "short_name", r.team_id)),
                    "price": float(r.now_cost or 4.0),
                }
    candidate_pool = list(universe.values())

    plan = plan_transfers(
        current_squad=current_squad,
        candidate_pool=candidate_pool,
        xp_by_gw=xp_by_gw,
        horizon_gws=gws_predicted,
        bank=bank,
        free_transfers=free_transfers,
        risk_lambda=args.risk,
    )

    print(json.dumps({
        "status": plan.status,
        "horizon": gws_predicted,
        "total_gain": plan.total_gain,
        "hits_taken": plan.hits_taken,
        "net_gain": plan.net_gain,
        "moves": plan.moves,
        "reasoning": plan.reasoning,
    }, indent=2))
    return 0


def cmd_changes(args) -> int:
    from fplagent.pipeline.attribution import attribute_change, narrate
    changes = attribute_change(gw_now=args.gw_to, gw_prev=args.gw_from,
                               season=args.season)
    if not changes:
        print("No stored predictions for both gameweeks yet.")
        return 0
    for c in changes[:args.top]:
        print(narrate(c))
    return 0


def cmd_demo(args) -> int:
    from fplagent.demo import run_demo
    return run_demo(n_sims=args.sims)


def cmd_doctor(args) -> int:
    from fplagent.doctor import diagnose
    return diagnose()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("fplagent", description="FPL Agent v9")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init-db", help="create all tables")
    s.add_argument("--drop", action="store_true", help="drop existing tables first")
    s.set_defaults(func=cmd_init_db)

    s = sub.add_parser("ingest", help="pull bootstrap, fixtures and per-fixture stats")
    s.add_argument("--limit", type=int, default=None, help="only N players (fast test)")
    s.add_argument("--cache", action="store_true", help="cache API responses to disk")
    s.add_argument("--fbref-csv", type=str, default=None,
                   help="OPTIONAL. Path to a manually-exported FBref "
                        "'Standard Stats' CSV. Penalty takers now come from "
                        "FPL's own declared penalties_order on every ingest, "
                        "so this is only needed for set-piece detail FPL does "
                        "not publish. See ingestion/fbref.py for export steps.")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("predict", help="predict a gameweek and persist it")
    s.add_argument("--gw", type=int, default=None)
    s.add_argument("--sims", type=int, default=None)
    s.add_argument("--top", type=int, default=25)
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("run", help="full daily pipeline")
    s.add_argument("--gw", type=int, default=None)
    s.add_argument("--skip-ingest", action="store_true")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--sims", type=int, default=None)
    s.add_argument("--cache", action="store_true")
    s.add_argument("--fbref-csv", type=str, default=None,
                   help="OPTIONAL. Manually-exported FBref 'Standard Stats' "
                        "CSV. Penalty duty comes from FPL automatically now; "
                        "this only adds detail FPL does not publish.")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("evaluate", help="walk-forward backtest vs baselines")
    s.add_argument("--from", dest="gw_from", type=int, default=2)
    s.add_argument("--to", dest="gw_to", type=int, default=None)
    s.add_argument("--replay", action="store_true",
                   help="re-predict history instead of scoring stored predictions")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("squad", help="optimal 15 + XI + captain")
    s.add_argument("--budget", type=float, default=100.0)
    s.add_argument("--risk", type=float, default=0.0,
                   help="0 = pure xP, 1 = pure differential")
    s.set_defaults(func=cmd_squad)

    s = sub.add_parser("my-team", help="fetch a real manager's current squad from FPL")
    s.add_argument("entry_id", type=int, help="your FPL team ID (from the URL on the FPL site)")
    s.add_argument("--gw", type=int, default=None)
    s.set_defaults(func=cmd_my_team)

    s = sub.add_parser("captains", help="captaincy ranking")
    s.add_argument("--risk", type=float, default=0.25,
                   help="0 = pure xP, 1 = pure differential (default 0.25)")
    s.set_defaults(func=cmd_captains)

    s = sub.add_parser("chips", help="chip schedule across the horizon")
    s.add_argument("--gw", type=int, default=None,
                   help="starting gameweek (default: resolved target)")
    s.add_argument("--horizon", type=int, default=14,
                   help="how many gameweeks to look ahead (default 14)")
    s.add_argument("--entry-id", type=int, default=None,
                   help="your FPL team ID; if supplied, Free Hit and "
                        "Wildcard values are computed against your squad, "
                        "otherwise they are 0")
    s.add_argument("--season", type=int, default=None)
    s.add_argument("--sims", type=int, default=None)
    s.set_defaults(func=cmd_chips)

    s = sub.add_parser("transfers", help="multi-gameweek transfer plan")
    s.add_argument("--entry-id", type=int, default=None,
                   help="your FPL team ID; if omitted, plans from the "
                        "optimal squad")
    s.add_argument("--gw", type=int, default=None)
    s.add_argument("--horizon", type=int, default=4,
                   help="how many gameweeks to plan over (default 4)")
    s.add_argument("--season", type=int, default=None)
    s.add_argument("--risk", type=float, default=0.0)
    s.add_argument("--sims", type=int, default=None)
    s.set_defaults(func=cmd_transfers)

    s = sub.add_parser("changes", help="what moved since last week, and why")
    s.add_argument("--from", dest="gw_from", type=int, required=True)
    s.add_argument("--to", dest="gw_to", type=int, required=True)
    s.add_argument("--season", type=int, default=None)
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(func=cmd_changes)

    s = sub.add_parser("demo", help="synthetic season, end to end, no network")
    s.add_argument("--sims", type=int, default=4000)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("doctor", help="diagnose the environment")
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    if getattr(args, "season", None) is None and hasattr(args, "season"):
        from fplagent.config import CURRENT_SEASON
        args.season = CURRENT_SEASON
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())