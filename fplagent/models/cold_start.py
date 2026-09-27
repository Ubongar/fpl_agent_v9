"""Cold start: players with no Premier League history.

Three tiers, in descending order of information. v5's roadmap treated all
new players as one "no data" case; in practice the tiers are very different
and conflating them throws away most of the available signal.

TIER 1 -- TRANSLATED FOREIGN DATA.
    A player arriving from a league we have data for gets his underlying
    metrics translated through a fitted exchange rate. Wirtz arriving from the
    Bundesliga is not an unknown quantity; he is a known quantity measured on
    a different scale.

TIER 2 -- PROMOTED-TEAM PLAYERS.
    Treated as a Championship -> Premier League translation, NOT as no-data.
    The Championship->PL coefficient is among the best-sampled in football:
    three teams every single year, decades deep. v5's roadmap put promoted
    players in the positional/price-cohort bucket, which discards a genuinely
    strong signal.

TIER 3 -- GENUINE NO-DATA.
    Teenagers with no senior minutes anywhere. Positional + price cohort
    baseline. This tier is much narrower than it first appears.

WHY THE TRUST THRESHOLD IS DYNAMIC
----------------------------------
The Bayesian weight on current-season data is minutes/(minutes + M). v5 used
a flat M. But M encodes "how much do I trust the prior", and that differs
enormously: a translated Bundesliga prior with 3000 minutes behind it and a
high-confidence coefficient deserves a large M (resist the new data); a
price-cohort guess deserves a small one (abandon it the moment real data
arrives). M is therefore derived from the prior's own confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Seed coefficients: ratio of per-90 output retained on moving TO the Premier
# League. THESE ARE UNVERIFIED. No source could be confirmed for these
# specific numbers, and no confidence value here should be read as "backed
# by research" -- it is a placeholder used only until a real coefficient can
# be fitted from data (see `ingestion/league_snapshot.py`), and every
# confidence below is discounted a further 0.5x in `lookup_translation`
# precisely because of that. Once `db.tables.league_translation` holds a
# fitted ratio that clears the reliability gate for a given
# (league, position, metric), that value is used instead and this table is
# never consulted for it.
SEED_TRANSLATION = {
    # (source_league, position_group, metric): (ratio, confidence, n)
    ("Bundesliga", "ATT", "np_xg90"): (0.78, 0.70, 140),
    ("Bundesliga", "ATT", "xa90"): (0.80, 0.68, 140),
    ("Bundesliga", "MID", "np_xg90"): (0.76, 0.65, 120),
    ("Bundesliga", "DEF", "defcon90"): (0.95, 0.60, 90),
    ("La Liga", "ATT", "np_xg90"): (0.84, 0.72, 160),
    ("La Liga", "ATT", "xa90"): (0.83, 0.70, 160),
    ("La Liga", "MID", "np_xg90"): (0.82, 0.66, 130),
    ("Serie A", "ATT", "np_xg90"): (0.86, 0.70, 150),
    ("Serie A", "DEF", "defcon90"): (0.92, 0.62, 100),
    ("Ligue 1", "ATT", "np_xg90"): (0.74, 0.66, 130),
    ("Ligue 1", "MID", "xa90"): (0.76, 0.62, 110),
    ("Eredivisie", "ATT", "np_xg90"): (0.58, 0.60, 90),
    ("Eredivisie", "MID", "xa90"): (0.62, 0.55, 80),
    ("Primeira Liga", "ATT", "np_xg90"): (0.62, 0.58, 85),
    ("Championship", "ATT", "np_xg90"): (0.66, 0.80, 400),
    ("Championship", "MID", "xa90"): (0.70, 0.78, 380),
    ("Championship", "DEF", "defcon90"): (1.02, 0.75, 350),
    ("Championship", "GK", "saves90"): (0.88, 0.70, 120),
    ("MLS", "ATT", "np_xg90"): (0.55, 0.50, 60),
    ("Saudi Pro League", "ATT", "np_xg90"): (0.52, 0.40, 35),
}

POSITION_GROUP = {"GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "ATT"}

# Age curve: how much of a player's foreign output we expect to survive the
# move, purely as a function of age. Younger players are both more likely to
# improve and more variable, so they regress harder toward the cohort mean.
AGE_ADJUSTMENT = {
    (0, 20): 0.88, (20, 23): 0.95, (23, 27): 1.00,
    (27, 30): 0.97, (30, 33): 0.90, (33, 99): 0.80,
}


def age_factor(age: float | None) -> float:
    if age is None:
        return 1.0
    for (lo, hi), f in AGE_ADJUSTMENT.items():
        if lo <= age < hi:
            return f
    return 1.0


@dataclass
class ColdStartPrior:
    np_goal_share: float
    assist_share: float
    defcon_per90: float
    p_play: float
    confidence: float          # 0-1, drives the dynamic trust threshold M
    tier: str                  # translated | promoted | cohort
    notes: dict


def trust_threshold(confidence: float, floor: float = 180.0,
                    ceiling: float = 1200.0) -> float:
    """Minutes of current-season evidence needed to half-override the prior.

    A high-confidence translated prior resists new data (M near the ceiling);
    a low-confidence cohort guess is abandoned quickly (M near the floor).
    This is item 15 of the original roadmap, made concrete.
    """
    c = float(np.clip(confidence, 0.0, 1.0))
    return float(floor + (ceiling - floor) * c)


def blend_prior_with_current(prior_value: float, current_value: float,
                             current_minutes: float, confidence: float) -> tuple[float, float]:
    """Returns (blended_value, weight_on_current)."""
    m = trust_threshold(confidence)
    w = current_minutes / (current_minutes + m)
    return float(w * current_value + (1 - w) * prior_value), float(w)


# The rest of this module asks for "np_xg90" / "xa90" (an xG-flavoured
# name from when the intent was to fit against expected goals). The only
# data actually available to fit against -- FBref "Standard Stats" exports --
# carries realised goals and assists, not xG, so the fitted table is keyed on
# "np_g90" / "a90" instead. This maps between the two rather than silently
# treating a noisier realised-output ratio as if it were an xG-based one.
_METRIC_ALIAS = {"np_xg90": "np_g90", "xa90": "a90"}


def _fitted_lookup(source_league: str, group: str, metric: str) -> tuple[float, float, int] | None:
    """Read a real, gate-passed ratio from the database, if one exists.

    Imported lazily to avoid a hard dependency from this pure-function module
    on the database layer -- `translated_prior` and `cohort_prior` are used
    in tests and in synthetic contexts (the demo) with no database at all,
    and must keep working there.
    """
    try:
        from sqlalchemy import select
        from fplagent.db import tables as T
        from fplagent.db.repository import read_sql
    except Exception:      # noqa: BLE001
        return None
    fitted_metric = _METRIC_ALIAS.get(metric, metric)
    try:
        df = read_sql(select(T.league_translation).where(
            (T.league_translation.c.source_league == source_league)
            & (T.league_translation.c.position_group == group)
            & (T.league_translation.c.metric == fitted_metric)))
    except Exception:      # noqa: BLE001
        return None
    if df.empty:
        return None
    row = df.iloc[0]
    # confidence of 0.0 marks a ratio that was computed but did NOT clear
    # ingestion.league_snapshot's reliability gate -- persisted for
    # transparency, never trusted as a live prior.
    if float(row["confidence"]) <= 0.0:
        return None
    return float(row["ratio"]), float(row["confidence"]), int(row["sample_size"])


def lookup_translation(source_league: str, position: str, metric: str,
                       table: dict | None = None) -> tuple[float, float, int]:
    group = POSITION_GROUP.get(position, "MID")

    if table is None:
        fitted = _fitted_lookup(source_league, group, metric)
        if fitted is not None:
            return fitted

    table = SEED_TRANSLATION if table is None else table
    key = (source_league, group, metric)
    if key in table:
        ratio, conf, n = table[key]
        # Halved because this table's provenance is unverified -- see the
        # module-level note above SEED_TRANSLATION.
        return ratio, conf * 0.5, n
    # Fall back to the same league's attacking coefficient, discounted again
    # for being both unverified AND for the wrong position.
    for (lg, grp, met), val in table.items():
        if lg == source_league and met == metric:
            return val[0], val[1] * 0.25, val[2]
    return 0.70, 0.10, 0


def translated_prior(position: str, source_league: str,
                     foreign_np_xg90: float, foreign_xa90: float,
                     foreign_defcon90: float, foreign_minutes: int,
                     age: float | None,
                     team_np_xg_per_match: float,
                     volume_context_ratio: float = 1.0,
                     table: dict | None = None) -> ColdStartPrior:
    """Tier 1 / Tier 2: translate foreign per-90 output into an FPL share.

    `volume_context_ratio` separates VOLUME from QUALITY. A player moving from
    a side that dominated its league to a mid-table Premier League team will
    see his shot and chance-creation volume fall even if his per-touch quality
    is unchanged. Pass old_team_share_of_league / new_team_expected_share.
    Quality translates through the exchange rate; volume translates through
    this ratio. Collapsing the two is the most common way league-translation
    models go wrong.
    """
    g_ratio, g_conf, g_n = lookup_translation(source_league, position, "np_xg90", table)
    a_ratio, a_conf, _ = lookup_translation(source_league, position, "xa90", table)
    d_ratio, d_conf, _ = lookup_translation(source_league, position, "defcon90", table)

    af = age_factor(age)
    vol = float(np.clip(volume_context_ratio, 0.4, 1.6))

    pl_np_xg90 = foreign_np_xg90 * g_ratio * af * vol
    pl_xa90 = foreign_xa90 * a_ratio * af * vol
    pl_defcon90 = foreign_defcon90 * d_ratio

    # Convert per-90 output into a share of the new team's expected goals.
    denom = max(team_np_xg_per_match, 0.35)
    np_goal_share = float(np.clip(pl_np_xg90 / denom, 0.0, 0.7))
    assist_share = float(np.clip(pl_xa90 / denom, 0.0, 0.7))

    # Sample size of the foreign record itself feeds the confidence, alongside
    # the coefficient's own confidence. 2700 minutes == 30 full matches.
    minutes_conf = float(np.clip(foreign_minutes / 2700.0, 0.0, 1.0))
    confidence = float(np.clip(0.6 * g_conf + 0.4 * minutes_conf, 0.05, 0.95))

    return ColdStartPrior(
        np_goal_share=np_goal_share,
        assist_share=assist_share,
        defcon_per90=float(max(pl_defcon90, 0.0)),
        p_play=0.6,
        confidence=confidence,
        tier="translated",
        notes={
            "source_league": source_league,
            "goal_ratio": g_ratio, "assist_ratio": a_ratio,
            "age_factor": af, "volume_ratio": vol,
            "coefficient_sample": g_n,
        },
    )


def cohort_prior(position: str, price: float, cohort_stats: dict) -> ColdStartPrior:
    """Tier 3: no usable record anywhere. Fall back to what players of this
    position and price band have historically produced.

    `cohort_stats` comes from the database: the observed mean share and DefCon
    rate for players in the same position within +/- 0.5m of this price. Price
    is a genuinely informative prior because it encodes the game's own
    expectation of the player, set by people with more information than us.
    """
    band = cohort_stats.get("np_goal_share")
    return ColdStartPrior(
        np_goal_share=float(band if band is not None else 0.06),
        assist_share=float(cohort_stats.get("assist_share", 0.07)),
        defcon_per90=float(cohort_stats.get("defcon_per90", 5.0)),
        p_play=float(cohort_stats.get("p_play", 0.4)),
        confidence=0.20,
        tier="cohort",
        notes={"position": position, "price": price,
               "cohort_n": cohort_stats.get("n", 0)},
    )
