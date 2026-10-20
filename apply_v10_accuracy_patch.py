#!/usr/bin/env python
"""Apply the FPL Agent v10 accuracy patch to an existing repository checkout.

Run from the repository root:
    python apply_v10_accuracy_patch.py

The script validates every expected source anchor before writing anything.
It refuses to patch files that have drifted from the expected v9 layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap

ROOT = Path.cwd()


def fail(message: str) -> "NoReturn":
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def read_required(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        fail(f"Missing {relative}. Run this script from the fpl_agent_v9 repository root.")
    return path.read_text(encoding="utf-8")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        fail(f"Expected exactly one match for {label}, found {count}. No files have been written.")
    return source.replace(old, new, 1)


def replace_between(source: str, start_marker: str, end_marker: str,
                    replacement: str, label: str) -> str:
    start = source.find(start_marker)
    if start < 0 or source.find(start_marker, start + 1) >= 0:
        fail(f"Could not uniquely find start marker for {label}. No files have been written.")
    end = source.find(end_marker, start + len(start_marker))
    if end < 0:
        fail(f"Could not find end marker for {label}. No files have been written.")
    end += len(end_marker)
    return source[:start] + replacement + source[end:]


def patch_config(source: str) -> str:
    return replace_once(
        source,
        '    share_shrinkage_games: float = 12.0\n',
        '    share_shrinkage_games: float = 12.0\n'
        '    # Experimental blend weight for expected-assist (xA) share estimates.\n'
        '    # Set to 0.0 to retain the observed-assists-only estimate.\n'
        '    xa_assist_blend_weight: float = 0.65\n',
        "config xA blend setting",
    )


def patch_shares(source: str) -> str:
    source = replace_once(
        source,
        '                    prior_override: dict | None = None,\n'
        '                    metric: str = "goals") -> ShareEstimate:\n',
        '                    prior_override: dict | None = None,\n'
        '                    metric: str = "goals",\n'
        '                    player_expected_assists: list[float] | None = None,\n'
        '                    team_expected_assists: list[float] | None = None,\n'
        '                    xa_blend_weight: float = 0.65) -> ShareEstimate:\n',
        "estimate_shares signature",
    )
    source = replace_once(
        source,
        '    if denom_assists <= 0:\n'
        '        assist_share = prior_a\n'
        '    else:\n'
        '        assist_share = w * (player_as / denom_assists) + (1 - w) * prior_a\n',
        '    # Actual assists are noisy in short samples. Keep their shrunk estimate,\n'
        '    # then blend toward a second estimate derived from expected assists (xA)\n'
        '    # only when both player-level and team-level xA are available.\n'
        '    if denom_assists <= 0:\n'
        '        observed_assist_share = prior_a\n'
        '    else:\n'
        '        observed_assist_share = w * (player_as / denom_assists) + (1 - w) * prior_a\n'
        '\n'
        '    assist_share = observed_assist_share\n'
        '    xa_share = None\n'
        '    xa_weight_used = 0.0\n'
        '    if player_expected_assists is not None and team_expected_assists is not None:\n'
        '        player_xa = np.asarray(player_expected_assists, dtype=float)\n'
        '        team_xa = np.asarray(team_expected_assists, dtype=float)\n'
        '        if (len(player_xa) == len(mins) and len(team_xa) == len(mins)\n'
        '                and np.isfinite(player_xa).any() and np.isfinite(team_xa).any()):\n'
        '            player_xa = np.nan_to_num(player_xa, nan=0.0, posinf=0.0, neginf=0.0)\n'
        '            team_xa = np.nan_to_num(team_xa, nan=0.0, posinf=0.0, neginf=0.0)\n'
        '            denom_xa = float((team_xa * on_pitch).sum())\n'
        '            if denom_xa > 1e-8:\n'
        '                xa_share = w * (float(player_xa.sum()) / denom_xa) + (1 - w) * prior_a\n'
        '                xa_weight_used = float(np.clip(xa_blend_weight, 0.0, 1.0))\n'
        '                assist_share = ((1.0 - xa_weight_used) * observed_assist_share\n'
        '                                + xa_weight_used * xa_share)\n',
        "assist-share xA blend",
    )
    source = replace_once(
        source,
        '        "raw_player_goals": round(float(player_np_raw.sum()), 2),\n'
        '        "capped_player_goals": round(player_np, 2),\n'
        '        "per_fixture_cap": MAX_GOALS_PER_FIXTURE_FOR_SHARE,\n'
        '        "penalty_source": ("declared" if declared_penalty_share is not None\n'
        '                           else ("observed" if observed_pen_share else "none")),\n',
        '        "raw_player_goals": round(float(player_np_raw.sum()), 2),\n'
        '        "capped_player_goals": round(player_np, 2),\n'
        '        "per_fixture_cap": MAX_GOALS_PER_FIXTURE_FOR_SHARE,\n'
        '        "observed_assist_share": round(float(observed_assist_share), 4),\n'
        '        "xa_share": None if xa_share is None else round(float(xa_share), 4),\n'
        '        "xa_blend_weight_used": round(float(xa_weight_used), 3),\n'
        '        "penalty_source": ("declared" if declared_penalty_share is not None\n'
        '                           else ("observed" if observed_pen_share else "none")),\n',
        "xA provenance metadata",
    )
    return source


def patch_features(source: str) -> str:
    source = replace_once(
        source,
        'def _team_penalties_by_fixture(history: pd.DataFrame) -> pd.DataFrame:\n',
        'def _team_output_xa_by_fixture(history: pd.DataFrame) -> pd.DataFrame:\n'
        '    """Team xA per fixture, used as the denominator for player xA share."""\n'
        '    keys = ["season", "fixture_id", "was_home"]\n'
        '    cols = keys + ["team_xa"]\n'
        '    if history.empty or "expected_assists" not in history:\n'
        '        return pd.DataFrame(columns=cols)\n'
        '    h = history[history["expected_assists"].notna()]\n'
        '    if h.empty:\n'
        '        return pd.DataFrame(columns=cols)\n'
        '    g = (h.groupby(keys, as_index=False)["expected_assists"].sum()\n'
        '          .rename(columns={"expected_assists": "team_xa"}))\n'
        '    return g[cols]\n'
        '\n'
        '\n'
        'def _team_penalties_by_fixture(history: pd.DataFrame) -> pd.DataFrame:\n',
        "team xA aggregator insertion",
    )
    source = replace_once(
        source,
        '    team_xg_out = _team_output_xg_by_fixture(history)\n',
        '    team_xg_out = _team_output_xg_by_fixture(history)\n'
        '    team_xa_out = _team_output_xa_by_fixture(history)\n',
        "team xA aggregate setup",
    )
    source = replace_once(
        source,
        '            hh = hh.merge(team_xg_out, on=keys, how="left")\n',
        '            hh = hh.merge(team_xg_out, on=keys, how="left")\n'
        '            hh = hh.merge(team_xa_out, on=keys, how="left")\n',
        "team xA feature merge",
    )
    source = replace_once(
        source,
        '                team_pens=tp.tolist(),\n'
        '                metric=metric,\n'
        '            )\n',
        '                team_pens=tp.tolist(),\n'
        '                metric=metric,\n'
        '                player_expected_assists=(\n'
        '                    hh["expected_assists"].fillna(0).tolist()\n'
        '                    if "expected_assists" in hh and hh["expected_assists"].notna().any()\n'
        '                    else None\n'
        '                ),\n'
        '                team_expected_assists=(\n'
        '                    hh["team_xa"].fillna(0).tolist()\n'
        '                    if "team_xa" in hh and hh["team_xa"].notna().any()\n'
        '                    else None\n'
        '                ),\n'
        '                xa_blend_weight=PARAMS.xa_assist_blend_weight,\n'
        '            )\n',
        "xA arguments to share estimator",
    )
    source = replace_once(
        source,
        'def build_player_features(cutoff: datetime, target_gw: int,\n'
        '                          season: int = CURRENT_SEASON,\n'
        '                          lookback: int = 12,\n'
        '                          minutes_model: MinutesModel | None = None) -> pd.DataFrame:\n',
        'def build_player_features(cutoff: datetime, target_gw: int,\n'
        '                          season: int = CURRENT_SEASON,\n'
        '                          lookback: int = 12,\n'
        '                          minutes_model: MinutesModel | None = None,\n'
        '                          strict_point_in_time: bool = False) -> pd.DataFrame:\n',
        "point-in-time feature-builder signature",
    )
    source = replace_once(
        source,
        '        st = (status_idx.loc[pid].to_dict()\n'
        '              if status_idx is not None and pid in status_idx.index else {})\n'
        '        pos = str(st.get("position") or p["position"])\n',
        '        st = (status_idx.loc[pid].to_dict()\n'
        '              if status_idx is not None and pid in status_idx.index else {})\n'
        '        # Historical replay must not silently use the current players table\n'
        '        # for dynamic metadata when no snapshot existed before the deadline.\n'
        '        if strict_point_in_time:\n'
        '            required_snapshot_fields = ("team_id", "position", "now_cost")\n'
        '            if not st or any(\n'
        '                key not in st or st.get(key) is None or pd.isna(st.get(key))\n'
        '                for key in required_snapshot_fields\n'
        '            ):\n'
        '                continue\n'
        '        pos = str(st.get("position") or p["position"])\n',
        "strict historical metadata guard",
    )
    old_rank = (
        'def _price_rank_in_team(players: pd.DataFrame, status_idx) -> dict[int, float]:\n'
        '    if players.empty:\n'
        '        return {}\n'
        '    df = players[["player_id", "team_id", "now_cost"]].copy()\n'
        '    if status_idx is not None and "now_cost" in status_idx:\n'
        '        snap = status_idx["now_cost"]\n'
        '        df["now_cost"] = df["player_id"].map(snap).fillna(df["now_cost"])\n'
        '    if status_idx is not None and "team_id" in status_idx:\n'
        '        team = status_idx["team_id"]\n'
        '        df["team_id"] = df["player_id"].map(team).fillna(df["team_id"])\n'
        '    df["rank"] = df.groupby("team_id")["now_cost"].rank(ascending=False, method="min")\n'
        '    return {int(r.player_id): float(r.rank) for r in df.itertuples()}\n'
    )
    new_rank = (
        'def _price_rank_in_team(players: pd.DataFrame, status_idx) -> dict[int, float]:\n'
        '    """Rank prices using only the latest pre-cutoff snapshot, never current fallback data."""\n'
        '    if players.empty or status_idx is None or status_idx.empty:\n'
        '        return {}\n'
        '    if not {"now_cost", "team_id"}.issubset(status_idx.columns):\n'
        '        return {}\n'
        '    df = players[["player_id"]].drop_duplicates().copy()\n'
        '    df["now_cost"] = df["player_id"].map(status_idx["now_cost"])\n'
        '    df["team_id"] = df["player_id"].map(status_idx["team_id"])\n'
        '    df = df.dropna(subset=["now_cost", "team_id"])\n'
        '    if df.empty:\n'
        '        return {}\n'
        '    df["rank"] = df.groupby("team_id")["now_cost"].rank(ascending=False, method="min")\n'
        '    return {int(r.player_id): float(r.rank) for r in df.itertuples()}\n'
    )
    source = replace_once(source, old_rank, new_rank, "point-in-time price rank")
    return source


def patch_predict(source: str) -> str:
    source = replace_once(
        source,
        '                    persist: bool = True,\n'
        '                    use_market: bool = True) -> pd.DataFrame:\n',
        '                    persist: bool = True,\n'
        '                    use_market: bool = True,\n'
        '                    strict_point_in_time: bool | None = None) -> pd.DataFrame:\n',
        "prediction strict mode argument",
    )
    source = replace_once(
        source,
        '    cutoff = cutoff or deadline or datetime.utcnow()\n',
        '    cutoff = cutoff or deadline or datetime.utcnow()\n'
        '    if strict_point_in_time is None:\n'
        '        cutoff_ts = pd.Timestamp(cutoff)\n'
        '        if cutoff_ts.tzinfo is None:\n'
        '            cutoff_ts = cutoff_ts.tz_localize("UTC")\n'
        '        else:\n'
        '            cutoff_ts = cutoff_ts.tz_convert("UTC")\n'
        '        strict_point_in_time = cutoff_ts < (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1))\n',
        "automatic strict mode for historical replay",
    )
    source = replace_once(
        source,
        '    features = build_player_features(cutoff, target_gw, season,\n'
        '                                     minutes_model=_fit_minutes_model(cutoff, season))\n',
        '    features = build_player_features(\n'
        '        cutoff, target_gw, season,\n'
        '        minutes_model=_fit_minutes_model(cutoff, season),\n'
        '        strict_point_in_time=bool(strict_point_in_time),\n'
        '    )\n',
        "pass strict point-in-time setting",
    )
    return source


def patch_minutes(source: str) -> str:
    helper_anchor = 'class MinutesModel:\n'
    helper = (
        'def _chronological_train_cal_test_slices(n_rows: int, holdout_frac: float = 0.2\n'
        '                                         ) -> tuple[slice, slice, slice]:\n'
        '    """Return disjoint chronological 60/20/20 train/calibration/test slices by default."""\n'
        '    if n_rows < 1:\n'
        '        return slice(0, 0), slice(0, 0), slice(0, 0)\n'
        '    if not 0.0 < holdout_frac < 0.5:\n'
        '        raise ValueError("holdout_frac must be between 0 and 0.5")\n'
        '    test_start = int(n_rows * (1.0 - holdout_frac))\n'
        '    calibration_start = int(test_start * 0.75)\n'
        '    return (slice(0, calibration_start),\n'
        '            slice(calibration_start, test_start),\n'
        '            slice(test_start, n_rows))\n'
        '\n'
        '\n'
    )
    source = replace_once(source, helper_anchor, helper + helper_anchor,
                          "chronological split helper")
    start_marker = '        cut = int(len(X) * (1.0 - holdout_frac))\n'
    end_marker = '        self.n_train = len(X_tr)\n'
    new_block = textwrap.indent(textwrap.dedent("""\
train_slice, calibration_slice, test_slice = _chronological_train_cal_test_slices(
    len(X), holdout_frac
)
X_tr = X[train_slice]
X_cal = X[calibration_slice]
X_ho = X[test_slice]
y_play_tr = y_play[train_slice]
y_start_tr = y_start[train_slice]
y_sixty_tr = y_sixty[train_slice]

if (len(X_tr) < 100 or len(X_cal) < 50 or len(X_ho) < 50
        or len(np.unique(y_play_tr)) < 2
        or len(np.unique(y_start_tr)) < 2
        or len(np.unique(y_sixty_tr)) < 2):
    self.refusal_reason = (
        "insufficient chronological train/calibration/test rows "
        "or class diversity; staying on heuristic"
    )
    log.info("MinutesModel: %s", self.refusal_reason)
    return False

common = dict(max_iter=300, learning_rate=0.06, max_depth=5,
              l2_regularization=1.0, early_stopping=False,
              random_state=7)
self.play_model = HistGradientBoostingClassifier(**common).fit(X_tr, y_play_tr)
self.start_model = HistGradientBoostingClassifier(**common).fit(X_tr, y_start_tr)
self.sixty_model = HistGradientBoostingClassifier(**common).fit(X_tr, y_sixty_tr)

# Calibrators see the middle chronological block only. The final block remains
# untouched until scoring the fitted models.
for name, model, y in (("play", self.play_model, y_play),
                       ("start", self.start_model, y_start),
                       ("sixty", self.sixty_model, y_sixty)):
    raw_cal = model.predict_proba(X_cal)[:, 1]
    y_cal = y[calibration_slice]
    calibrator = None
    if len(np.unique(y_cal)) >= 2:
        calibrator = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0
        )
        calibrator.fit(raw_cal, y_cal)
        setattr(self, f"{name}_calibrator", calibrator)

    raw_test = model.predict_proba(X_ho)[:, 1]
    y_test = y[test_slice]
    calibrated = (
        calibrator.predict(raw_test) if calibrator is not None else raw_test
    )
    self._holdout_pred[name] = np.asarray(calibrated, dtype=float)
    self._holdout_y[name] = np.asarray(y_test, dtype=float)
    self.holdout[f"{name}_brier_raw"] = float(np.mean((raw_test - y_test) ** 2))
    self.holdout[f"{name}_brier_cal"] = float(
        np.mean((calibrated - y_test) ** 2)
    )

self.n_train = len(X_tr)
""").rstrip(), "        ") + "\n"
    source = replace_between(source, start_marker, end_marker, new_block,
                             "minutes model train/calibration/test split")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and prepare changes without writing files")
    args = parser.parse_args()

    required = [
        "pyproject.toml",
        "fplagent/config.py",
        "fplagent/models/shares.py",
        "fplagent/features/build.py",
        "fplagent/pipeline/predict.py",
        "fplagent/models/minutes.py",
    ]
    originals = {name: read_required(name) for name in required}

    # Prepare every change in memory first. If any anchor mismatches, nothing is written.
    changed = {
        "fplagent/config.py": patch_config(originals["fplagent/config.py"]),
        "fplagent/models/shares.py": patch_shares(originals["fplagent/models/shares.py"]),
        "fplagent/features/build.py": patch_features(originals["fplagent/features/build.py"]),
        "fplagent/pipeline/predict.py": patch_predict(originals["fplagent/pipeline/predict.py"]),
        "fplagent/models/minutes.py": patch_minutes(originals["fplagent/models/minutes.py"]),
    }

    new_test = ROOT / "tests" / "test_v10_accuracy_patch.py"
    if args.dry_run:
        print("Dry run successful. These existing files would be updated:")
        for name in changed:
            print(f"  {name}")
        if not new_test.exists():
            print("  tests/test_v10_accuracy_patch.py (new)")
        return

    for relative, content in changed.items():
        (ROOT / relative).write_text(content, encoding="utf-8")
        print(f"Updated {relative}")

    bundled_test = Path(__file__).resolve().parent / "tests" / "test_v10_accuracy_patch.py"
    if not new_test.exists() and bundled_test.exists():
        new_test.parent.mkdir(parents=True, exist_ok=True)
        new_test.write_text(bundled_test.read_text(encoding="utf-8"), encoding="utf-8")
        print("Created tests/test_v10_accuracy_patch.py")

    print("\nPatch applied. Run: pytest -q")
    print("Historical replay skips known players without a usable pre-deadline snapshot.")
    print("Historical players absent from the current player table still require roster backfill.")


if __name__ == "__main__":
    main()
