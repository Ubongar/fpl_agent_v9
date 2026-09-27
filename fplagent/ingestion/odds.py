"""Betting odds ingestion: the market as a prior on every fixture.

WHY THIS FILE EXISTS
--------------------
v6 shipped `blend_with_market()`, a `match_odds` table, an `ODDS_API_KEY`
config entry and a read in `predict.py` that looked for rows. Nothing
anywhere ever wrote one. The single strongest free prior available to the
system was fully implemented and permanently inert, and because the blend
silently no-ops on an empty table, nothing ever complained.

v8: bookmaker team names are resolved through ingestion.name_matching. The
Odds API spells clubs in full ("Manchester City", "West Ham United") and the
old raw fuzzy match against FPL's short forms ("Man City", "West Ham")
scored ~70 against a threshold of 75, so fixtures involving eight clubs never
received a market prior.

WHAT THE MARKET KNOWS THAT A GOALS MODEL DOES NOT
-------------------------------------------------
A Dixon-Coles fit sees results. The market sees team news, suspensions,
managerial changes, motivation, whether a side is resting players for a
European tie, and the weight of money from people who are paid to be right.
In the first six gameweeks of a season, when the fit has almost no evidence,
the market is close to the only real information available.

FROM PRICES TO EXPECTED GOALS
-----------------------------
Bookmaker prices are not probabilities: they carry an overround (the vig).
The steps are:

  1. Convert decimal odds to implied probabilities (1/odds).
  2. Remove the overround. Proportional scaling is the usual shortcut and it
     is biased, because the margin is not applied evenly across outcomes.
     Shin's method models the margin as arising from insider trading and
     recovers noticeably better probabilities on longshots, which is exactly
     where the naive method is worst and where FPL differentials live.
  3. Solve for the (lambda, mu) whose Dixon-Coles scoreline matrix best
     reproduces the de-vigged 1X2 probabilities, and the over/under 2.5 line
     when it is available.

Step 3 is a two-parameter fit against three or five targets, so it is
over-determined and stable. Inverting through the same scoreline model the
rest of the system uses matters: it means the market lambda and the fitted
lambda are the same kind of object and can be blended in log space.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import requests
from scipy.optimize import minimize

from fplagent.config import CURRENT_SEASON, ODDS_API_BASE, ODDS_API_KEY
from fplagent.db import tables as T
from fplagent.db.repository import read_sql, upsert
from fplagent.ingestion.name_matching import resolve_team_id
from fplagent.models.dixon_coles import score_matrix

log = logging.getLogger(__name__)

SPORT_KEY = "soccer_epl"
DEFAULT_BOOKMAKER = "pinnacle"


# ---------------------------------------------------------------------------
# De-vigging
# ---------------------------------------------------------------------------

def implied_probabilities(odds: list[float]) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    return 1.0 / np.maximum(o, 1.0001)


def devig_proportional(raw: np.ndarray) -> np.ndarray:
    """Scale to sum to one. Simple, and biased against longshots."""
    raw = np.asarray(raw, dtype=float)
    return raw / max(raw.sum(), 1e-9)


def devig_shin(raw: np.ndarray, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Shin's method for removing the bookmaker margin.

    Models the observed prices as arising from a bookmaker protecting itself
    against a proportion `z` of insider money. Recovers fairer probabilities
    than proportional scaling, most visibly on longshots: proportional
    de-vigging systematically overstates the chance of the unlikely outcome,
    and an FPL differential is by definition a bet on an unlikely outcome.
    """
    raw = np.asarray(raw, dtype=float)
    total = raw.sum()
    if total <= 1.0 + 1e-9:
        return devig_proportional(raw)

    z = 0.0
    for _ in range(max_iter):
        root = np.sqrt(z ** 2 + 4.0 * (1.0 - z) * raw ** 2 / total)
        p = (root - z) / (2.0 * (1.0 - z))
        s = p.sum()
        if abs(s - 1.0) < tol:
            break
        z += (s - 1.0) * 0.5
        z = float(np.clip(z, 0.0, 0.35))
    p = np.maximum(p, 1e-9)
    return p / p.sum()


# ---------------------------------------------------------------------------
# Probabilities to expected goals
# ---------------------------------------------------------------------------

def _outcome_probs(lam: float, mu: float, rho: float = -0.05) -> tuple[float, float, float, float]:
    m = score_matrix(lam, mu, rho)
    n = m.shape[0]
    idx = np.arange(n)
    home = float(np.tril(m, -1).sum())      # home goals > away goals
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())
    totals = idx[:, None] + idx[None, :]
    over25 = float(m[totals >= 3].sum())
    return home, draw, away, over25


def implied_lambdas(p_home: float, p_draw: float, p_away: float,
                    p_over25: float | None = None,
                    rho: float = -0.05) -> tuple[float, float]:
    """Solve for the (lambda, mu) reproducing these de-vigged probabilities.

    Inverting through the SAME scoreline model the rest of the pipeline uses
    is the point. A market number derived some other way would not be
    comparable with the fitted lambda, and blending the two in log space
    would be mixing different quantities.
    """
    target = np.array([p_home, p_draw, p_away], dtype=float)

    def loss(theta: np.ndarray) -> float:
        lam, mu = np.exp(theta)
        h, d, a, o = _outcome_probs(float(lam), float(mu), rho)
        err = float(((np.array([h, d, a]) - target) ** 2).sum())
        if p_over25 is not None:
            err += 0.5 * (o - p_over25) ** 2
        return err

    res = minimize(loss, x0=np.log([1.4, 1.2]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 600})
    lam, mu = np.exp(res.x)
    return float(np.clip(lam, 0.05, 6.0)), float(np.clip(mu, 0.05, 6.0))


# ---------------------------------------------------------------------------
# The Odds API
# ---------------------------------------------------------------------------

def fetch_odds(api_key: str | None = None, regions: str = "eu",
               markets: str = "h2h,totals") -> list[dict]:
    """One request covers every upcoming fixture, so a weekly pull sits
    comfortably inside the free tier's 500 requests a month.

    Region is "eu" because that is the region The Odds API lists Pinnacle
    (DEFAULT_BOOKMAKER) under; with "uk" it is never returned and every
    fixture silently fell back to whichever book happened to be listed first.
    """
    api_key = api_key or ODDS_API_KEY
    if not api_key:
        log.info("no ODDS_API_KEY set; skipping odds ingestion. The market "
                 "blend will no-op, which is safe but leaves the single "
                 "strongest free prior unused.")
        return []
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
    params = {"apiKey": api_key, "regions": regions, "markets": markets,
              "oddsFormat": "decimal"}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        remaining = r.headers.get("x-requests-remaining")
        if remaining is not None:
            log.info("odds api: %s requests remaining this month", remaining)
        return r.json()
    except Exception as e:      # noqa: BLE001
        log.warning("odds fetch failed, leaving match_odds unchanged: %s", e)
        return []


def _match_fixture(event: dict, fixtures, teams) -> int | None:
    """Map a bookmaker's team names onto our fixture ids."""
    if fixtures.empty or teams.empty:
        return None
    h = resolve_team_id(event.get("home_team", ""), teams)
    a = resolve_team_id(event.get("away_team", ""), teams)
    if h is None or a is None:
        return None
    m = fixtures[(fixtures["home_team_id"] == h) & (fixtures["away_team_id"] == a)]
    if m.empty:
        return None
    return int(m.iloc[0]["fixture_id"])


def ingest_odds(season: int = CURRENT_SEASON, api_key: str | None = None,
                bookmaker: str = DEFAULT_BOOKMAKER) -> int:
    """Pull, de-vig, invert to expected goals, and persist.

    Soft-fails at every step: no key, no network, no match, unparseable
    payload. The market blend already handles a missing row correctly, so an
    odds outage degrades the model rather than stopping the pipeline.
    """
    from sqlalchemy import select

    events = fetch_odds(api_key)
    if not events:
        return 0

    fixtures = read_sql(select(T.fixtures).where(
        (T.fixtures.c.season == season) & (T.fixtures.c.finished.is_(False))))
    teams = read_sql(select(T.teams).where(T.teams.c.season == season))
    if fixtures.empty or teams.empty:
        log.warning("no fixtures/teams for season %s; run ingest first", season)
        return 0

    now = datetime.utcnow()
    rows: list[dict] = []
    unmatched: list[str] = []

    for ev in events:
        fixture_id = _match_fixture(ev, fixtures, teams)
        if fixture_id is None:
            unmatched.append(f"{ev.get('home_team')} v {ev.get('away_team')}")
            continue
        books = ev.get("bookmakers", [])
        book = next((b for b in books if b.get("key") == bookmaker), None) or (
            books[0] if books else None)
        if not book:
            continue

        h2h = next((m for m in book.get("markets", []) if m.get("key") == "h2h"), None)
        if not h2h:
            continue
        prices = {o.get("name"): float(o.get("price", 0)) for o in h2h.get("outcomes", [])}
        home_o = prices.get(ev.get("home_team"))
        away_o = prices.get(ev.get("away_team"))
        draw_o = prices.get("Draw")
        if not (home_o and away_o and draw_o):
            continue

        raw = implied_probabilities([home_o, draw_o, away_o])
        fair = devig_shin(raw)

        over25 = None
        over_price = under_price = None
        totals = next((m for m in book.get("markets", []) if m.get("key") == "totals"), None)
        if totals:
            line = [o for o in totals.get("outcomes", [])
                    if abs(float(o.get("point", 0)) - 2.5) < 1e-6]
            if len(line) == 2:
                o_raw = implied_probabilities([float(x["price"]) for x in line])
                o_fair = devig_shin(o_raw)
                first_is_over = line[0]["name"].lower().startswith("over")
                over25 = float(o_fair[0] if first_is_over else o_fair[1])
                over_price = float(line[0]["price"] if first_is_over else line[1]["price"])
                under_price = float(line[1]["price"] if first_is_over else line[0]["price"])

        lam, mu = implied_lambdas(float(fair[0]), float(fair[1]), float(fair[2]), over25)

        rows.append({
            "season": season, "fixture_id": fixture_id, "bookmaker": book.get("key", "?"),
            "pulled_at": now,
            "home_win": home_o, "draw": draw_o, "away_win": away_o,
            "over_2_5": over_price, "under_2_5": under_price,
            "devigged_home": round(float(fair[0]), 5),
            "devigged_draw": round(float(fair[1]), 5),
            "devigged_away": round(float(fair[2]), 5),
            "implied_home_xg": round(lam, 4),
            "implied_away_xg": round(mu, 4),
        })

    if rows:
        upsert(T.match_odds, rows, ["season", "fixture_id", "bookmaker", "pulled_at"])
    if unmatched:
        log.warning("odds: %d events could not be matched to a fixture: %s",
                    len(unmatched), unmatched[:5])
    log.info("odds: %d fixtures priced from %s", len(rows), bookmaker)
    return len(rows)
