"""Data-quality gates.

THE POINT (v5 audit, section 3.6)
---------------------------------
v5's "safety gate" checked model accuracy against a hardcoded three-point
calibration curve, which made it a constant that failed on every run
regardless of data. Worse, accuracy gating at gameweek 5 is meaningless
anyway: with four matches played, ANY honest accuracy check is dominated by
noise, so the only two possible behaviours are "always fail" (what v5 did) or
"always pass" (what bypassing it did).

The gate you actually want is on INPUTS, not on accuracy. Row counts against
expectation, null rates, unit sanity, distributional drift on ingested
features. Those work perfectly from day one, at any sample size, and they
catch the failure modes that actually occur: an API shape change, a partial
ingest, a units regression, a season rollover.

Severity is three-level and only `fail` stops the pipeline. A gate that
cries wolf gets bypassed, and a bypassed gate is worse than no gate, because
it looks like protection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fplagent.config import GATES
from fplagent.db.repository import log_quality_event

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    severity: str      # info | warn | fail
    detail: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.severity == "fail"


@dataclass
class QualityReport:
    stage: str
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, severity: str, **detail) -> None:
        self.checks.append(CheckResult(name, severity, detail))

    @property
    def failed(self) -> bool:
        return any(c.failed for c in self.checks)

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "warn"]

    def persist(self) -> None:
        for c in self.checks:
            if c.severity != "info":
                log_quality_event(self.stage, c.name, c.severity, c.detail)

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for c in self.checks:
            counts[c.severity] = counts.get(c.severity, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no checks"


def check_bootstrap(data: dict) -> QualityReport:
    rep = QualityReport("ingest.bootstrap")
    elements = data.get("elements", [])
    teams = data.get("teams", [])
    events = data.get("events", [])

    n = len(elements)
    if n < GATES.min_players_expected or n > GATES.max_players_expected:
        rep.add("player_count_in_range", "fail", n=n,
                expected=[GATES.min_players_expected, GATES.max_players_expected])
    else:
        rep.add("player_count_in_range", "info", n=n)

    if len(teams) != 20:
        rep.add("team_count_is_20", "fail", n=len(teams))
    else:
        rep.add("team_count_is_20", "info", n=20)

    if not events:
        rep.add("events_present", "fail", n=0)
    else:
        rep.add("events_present", "info", n=len(events))

    # Schema drift: a field silently disappearing is the most common way an
    # upstream API change corrupts a pipeline quietly rather than loudly.
    required = {"id", "web_name", "team", "element_type", "now_cost",
                "total_points", "minutes", "status"}
    if elements:
        missing = required - set(elements[0].keys())
        rep.add("required_fields_present", "fail" if missing else "info",
                missing=sorted(missing))

    # Null rate on the fields models actually depend on.
    if elements:
        df = pd.DataFrame(elements)
        for col in ("now_cost", "status", "element_type", "team"):
            if col in df:
                rate = float(df[col].isna().mean())
                rep.add(f"null_rate.{col}",
                        "fail" if rate > GATES.max_null_rate else "info",
                        rate=round(rate, 4))
    return rep


def check_fixture_stats(df: pd.DataFrame) -> QualityReport:
    """Unit sanity on per-fixture rows.

    The `minutes <= 130` check is not paranoia. It is the exact assertion that
    would have caught v5's most damaging bug, where season-cumulative minutes
    (340, 780, 2100) were written into a column that models read as per-fixture
    values and divided by 90. One cheap range check, and that entire class of
    corruption becomes a loud pipeline failure instead of a silently wrong
    model nobody can explain.
    """
    rep = QualityReport("ingest.fixture_stats")
    if df.empty:
        rep.add("rows_present", "fail", n=0)
        return rep
    rep.add("rows_present", "info", n=len(df))

    if "minutes" in df:
        bad = df[df["minutes"] > GATES.max_minutes_per_fixture]
        rep.add("minutes_per_fixture_plausible",
                "fail" if len(bad) else "info",
                n_bad=int(len(bad)),
                max_seen=int(df["minutes"].max()),
                sample=bad.head(3)[["player_id", "gw", "minutes"]].to_dict("records") if len(bad) else [])

    if "total_points" in df:
        bad = df[(df["total_points"] < -10) | (df["total_points"] > 40)]
        rep.add("points_per_fixture_plausible",
                "warn" if len(bad) else "info", n_bad=int(len(bad)))

    dup_cols = [c for c in ("player_id", "season", "fixture_id") if c in df]
    if len(dup_cols) == 3:
        dupes = int(df.duplicated(subset=dup_cols).sum())
        rep.add("no_duplicate_grain", "fail" if dupes else "info", n_dupes=dupes)

    # A player appearing twice in one gameweek is CORRECT (double gameweek)
    # and must not be flagged -- but it must be visible, because v5's schema
    # silently discarded exactly these rows.
    if {"player_id", "gw"}.issubset(df.columns):
        dgw = df.groupby(["player_id", "gw"]).size()
        n_dgw = int((dgw > 1).sum())
        rep.add("double_gameweek_rows_preserved", "info", n_player_gw_pairs=n_dgw)
    return rep


def check_predictions(df: pd.DataFrame) -> QualityReport:
    rep = QualityReport("predict.output")
    if df.empty:
        rep.add("rows_present", "fail", n=0)
        return rep
    rep.add("rows_present", "info", n=len(df))

    for col, lo, hi in (("p_play", 0.0, 1.0), ("p_start", 0.0, 1.0), ("p_sixty", 0.0, 1.0)):
        if col in df:
            vals = pd.to_numeric(df[col], errors="coerce")
            bad = int((~np.isfinite(vals.to_numpy(dtype=float)) |
                       (vals < lo).to_numpy() | (vals > hi).to_numpy()).sum())
            rep.add(f"{col}_in_unit_interval", "fail" if bad else "info", n_bad=bad)

    if {"p_play", "p_start", "p_sixty"}.issubset(df.columns):
        bad_start = int((df["p_start"] > df["p_play"] + 1e-9).sum())
        bad_sixty = int((df["p_sixty"] > df["p_play"] + 1e-9).sum())
        rep.add("p_start_le_p_play", "fail" if bad_start else "info", n_bad=bad_start)
        rep.add("p_sixty_le_p_play", "fail" if bad_sixty else "info", n_bad=bad_sixty)
    elif {"p_play", "p_sixty"}.issubset(df.columns):
        bad = int((df["p_sixty"] > df["p_play"] + 1e-9).sum())
        rep.add("p_sixty_le_p_play", "fail" if bad else "info", n_bad=bad)

    if "xp_mean" in df:
        bad = int(((df["xp_mean"] < -3) | (df["xp_mean"] > 20)).sum())
        rep.add("xp_plausible", "warn" if bad else "info", n_bad=bad,
                max_xp=float(df["xp_mean"].max()))
        # A model that predicts the same number for everyone passes MAE checks
        # and is worthless. Spread is a first-class health metric.
        spread = float(df["xp_mean"].std()) if len(df) > 1 else 0.0
        rep.add("xp_has_spread", "fail" if spread < 0.25 else "info",
                std=round(spread, 3))

    if {"xp_p10", "xp_p25", "xp_p50", "xp_p75", "xp_p90"}.issubset(df.columns):
        ordered = df["xp_p10"].le(df["xp_p25"]) & df["xp_p25"].le(df["xp_p50"]) & df["xp_p50"].le(df["xp_p75"]) & df["xp_p75"].le(df["xp_p90"])
        bad = int((~ordered).sum())
        rep.add("quantiles_ordered", "fail" if bad else "info", n_bad=bad)
    elif {"xp_p10", "xp_p90"}.issubset(df.columns):
        bad = int((df["xp_p10"] > df["xp_p90"]).sum())
        rep.add("quantiles_ordered", "fail" if bad else "info", n_bad=bad)
    return rep
