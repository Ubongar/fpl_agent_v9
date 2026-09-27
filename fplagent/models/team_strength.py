"""Joint maximum-likelihood attack/defence ratings for every team at once.

THE BUG THIS FIXES (v5 audit, item 1.4)
---------------------------------------
v5 fitted multiplicative attack/defence ratings and then normalised them
post-hoc:

    attack  = {k: v / mean(attack)  ...}
    defence = {k: v / mean(defence) ...}

But lambda = attack * defence * home_adv * league_avg_goals. Dividing attack
by `a` and defence by `d` rescales every lambda by 1/(a*d) while
league_avg_goals is left untouched. Unless a*d happened to equal 1, every
expected-goals number downstream -- and therefore every clean-sheet
probability and every player's xG -- was systematically wrong by a constant
factor nobody could see.

The fix is not to patch the normalisation, it is to remove the need for it.
Parameterising in LOG space with a sum-to-zero constraint makes the model
identifiable by construction:

    log lambda_home = mu + alpha_h - beta_a + gamma
    log lambda_away = mu + alpha_a - beta_h

with sum(alpha) = sum(beta) = 0. `mu` absorbs the league scoring level, so
there is nothing left over to renormalise. This also conditions the optimiser
far better than bounding raw multipliers to (0.3, 3.0), and lets us hand
scipy an analytic-quality objective computed over vectorised arrays rather
than a Python loop over scipy.stats calls.

STRONGER PRIORS ON rho AND RATINGS (v9)
---------------------------------------
Field observation: with only ~50 matches fitted (GW5-6), rho was clustering
at -0.47, which is the edge of the legal region and implies a 74% boost to
the 0-0 probability over what a Poisson with the same lambdas would produce.
Real Premier League rho sits around -0.05 to +0.05.

The cause is the flatness of the likelihood surface on 50 matches: even a
weak prior can be overwhelmed by noise. The prior weight on rho is now 20.0
instead of 8.0 (2.5x stronger), and the alpha/beta ridge is 1.5x stronger.
Both are still time-decayed by effective_n, so they dominate absolutely at
GW3 and fade entirely by GW20. The intent is unchanged -- "absent evidence,
every team is average and every match is Poisson" -- but the pull toward
that intent is now strong enough to resist a small number of odd results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from fplagent.config import PARAMS
from fplagent.models.dixon_coles import rho_bounds

log = logging.getLogger(__name__)


@dataclass
class TeamStrengthModel:
    """Fitted ratings. attack/defence are in LOG space and sum to zero.

    `attack_mult` / `defence_mult` expose the exponentiated view for display,
    where 1.0 is exactly league average -- but nothing internal uses them,
    precisely so the v5 rescaling bug cannot reappear.
    """
    attack: dict[int, float] = field(default_factory=dict)
    defence: dict[int, float] = field(default_factory=dict)
    intercept: float = 0.0          # log baseline goals per team per match
    home_adv: float = 0.25          # additive in log space
    rho: float = -0.05
    n_matches_fit: int = 0
    effective_n: float = 0.0
    target: str = "goals"        # goals | xg | blend
    converged: bool = False
    fitted_at: datetime = field(default_factory=datetime.utcnow)
    prior_only: bool = False

    @property
    def attack_mult(self) -> dict[int, float]:
        return {k: float(np.exp(v)) for k, v in self.attack.items()}

    @property
    def defence_mult(self) -> dict[int, float]:
        return {k: float(np.exp(v)) for k, v in self.defence.items()}

    @property
    def league_avg_goals(self) -> float:
        return float(np.exp(self.intercept))

    @property
    def home_adv_mult(self) -> float:
        return float(np.exp(self.home_adv))

    def expected_lambdas(self, home_team: int, away_team: int) -> tuple[float, float]:
        """The single place in the codebase where expected goals are formed."""
        ah = self.attack.get(home_team, 0.0)
        aa = self.attack.get(away_team, 0.0)
        dh = self.defence.get(home_team, 0.0)
        da = self.defence.get(away_team, 0.0)
        # The sum-to-zero parameterisation bounds only n-1 ratings; the
        # implied last one is -sum(others) and can reach +/-28 in log space.
        # Inside the likelihood that is clipped, but nothing clipped it here,
        # so a pathological fit could emit an astronomically large lambda
        # straight into the scoreline matrix.
        log_lam = np.clip(self.intercept + ah - da + self.home_adv, -4.0, 2.0)
        log_mu = np.clip(self.intercept + aa - dh, -4.0, 2.0)
        return float(np.exp(log_lam)), float(np.exp(log_mu))

    def rank_table(self) -> list[dict]:
        # A higher defence rating means FEWER goals conceded (it is subtracted
        # in log lambda), so net strength is attack TIMES defence. Dividing,
        # as this used to, ranked the best defensive sides lowest.
        rows = []
        for t in self.attack:
            rows.append({
                "team_id": t,
                "attack": round(self.attack_mult[t], 3),
                "defence": round(self.defence_mult.get(t, 1.0), 3),
                "net": round(self.attack_mult[t] * self.defence_mult.get(t, 1.0), 3),
            })
        return sorted(rows, key=lambda r: r["net"], reverse=True)


def _pack(n: int, alpha, beta, intercept, home_adv, rho) -> np.ndarray:
    # Only n-1 free parameters per rating vector; the last is -sum(others).
    return np.concatenate([alpha[:-1], beta[:-1], [intercept, home_adv, rho]])


def _unpack(p: np.ndarray, n: int):
    alpha_free = p[:n - 1]
    beta_free = p[n - 1:2 * (n - 1)]
    alpha = np.append(alpha_free, -alpha_free.sum())
    beta = np.append(beta_free, -beta_free.sum())
    intercept, home_adv, rho = p[-3], p[-2], p[-1]
    return alpha, beta, intercept, home_adv, rho


def _neg_log_likelihood(p, n, hi, ai, hg, ag, w, ridge=0.0, home_prior=0.25):
    """Weighted Dixon-Coles negative log-likelihood, fully vectorised.

    Uses the Poisson log-pmf written out directly (k*log(l) - l - log(k!))
    rather than scipy.stats.poisson.pmf in a loop: ~100x faster and far
    better behaved numerically, which starts to matter the moment multiple
    seasons of fixtures are in the fit.
    """
    alpha, beta, intercept, home_adv, rho = _unpack(p, n)

    log_lam = intercept + alpha[hi] - beta[ai] + home_adv
    log_mu = intercept + alpha[ai] - beta[hi]
    log_lam = np.clip(log_lam, -8, 3)
    log_mu = np.clip(log_mu, -8, 3)
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    # gammaln generalises the factorial to real arguments, so a blended
    # (non-integer) response is a valid quasi-Poisson likelihood rather than
    # an error. This is what lets the xG blend go through one fit.
    ll = (hg * log_lam - lam - gammaln(hg + 1.0)) + \
         (ag * log_mu - mu - gammaln(ag + 1.0))

    # tau adjustment, only defined on the four low-score cells.
    # With a blended response those cells are rarely hit exactly, so
    # the comparison is done on a rounded copy: the correction still applies
    # to the matches that were genuinely 0-0, 1-0, 0-1 or 1-1.
    hg_i, ag_i = np.rint(hg), np.rint(ag)
    tau = np.ones_like(lam)
    m00 = (hg_i == 0) & (ag_i == 0)
    m01 = (hg_i == 0) & (ag_i == 1)
    m10 = (hg_i == 1) & (ag_i == 0)
    m11 = (hg_i == 1) & (ag_i == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    # A parameter set that makes a realised scoreline impossible is invalid.
    # v6 returned a flat 1e12, which hands L-BFGS-B a cliff with no gradient
    # and stalls the optimiser at the boundary. A smooth quadratic barrier
    # keeps the region unreachable while still pointing the optimiser back
    # toward the feasible interior.
    bad = tau <= 1e-6
    if np.any(bad):
        penalty = 1e6 * float(np.sum((1e-6 - tau[bad]) ** 2)) + 1e4
        tau = np.where(bad, 1e-6, tau)
        return float(-(w * (ll + np.log(tau))).sum() + penalty)
    ll = ll + np.log(tau)

    nll = -(w * ll).sum()

    # Ridge penalty: a Bayesian prior that, absent evidence, every team is
    # average. This is not cosmetic -- with only a handful of matches played
    # the likelihood surface is nearly flat and an unregularised fit will
    # happily declare a team who beat two poor sides the best in the league.
    # The penalty decays in relative terms as real evidence accumulates,
    # which is exactly the early-season behaviour v5 tried and failed to get
    # by gating the pipeline on gameweek number.
    #
    # v9 raises two of the coefficients. Field observation: with only ~50
    # matches, rho clustered at -0.47, the edge of the legal region, which
    # implies a 74% boost to 0-0 probability. Real PL rho is -0.05 to +0.05.
    # The likelihood on 50 matches is too flat to resist noise; the priors
    # need to be strong enough to keep the fit near its Bayesian prior until
    # real evidence accumulates. 20.0 on rho (was 8.0) and 1.5x on alpha/beta
    # (was 1.0x) both decay with effective_n, so they dominate at GW3 and
    # vanish by GW20.
    if ridge > 0:
        nll += ridge * 1.5 * (np.sum(alpha ** 2) + np.sum(beta ** 2))
        nll += ridge * 4.0 * (home_adv - home_prior) ** 2
        nll += ridge * 20.0 * (rho + 0.05) ** 2

    return float(nll)


def fit_team_strength(matches: list[tuple[int, int, int, int, float]],
                      decay: float | None = None,
                      min_matches: int | None = None,
                      ridge_strength: float = 6.0,
                      target: str = "blend",
                      xg_matches: list[tuple[int, int, float, float, float]] | None = None,
                      xg_weight: float = 0.6) -> TeamStrengthModel:
    """matches: (home_team_id, away_team_id, home_goals, away_goals, days_ago).

    `target` selects what the ratings are fitted against:

      goals  -- realised goals, as v6 did.
      xg     -- expected goals, if `xg_matches` is supplied.
      blend  -- a weighted average of the two, `xg_weight` on xG.

    Goals are a low-count, high-variance outcome, so a goals-only fit needs a
    large number of matches before the ratings mean anything, which is exactly
    the period of the season when you most need them. Expected goals measures
    the same underlying process with far less noise. It is not a free lunch:
    xG misses finishing quality and game state, so the default blends rather
    than replaces. The blend is done on the RESPONSE (a pseudo-count) rather
    than on two separate fits, which keeps it one likelihood and one set of
    identifiable parameters.

    Returns a prior-only model (all ratings 0.0 in log space == league
    average) when there is not enough data, so this is always safe to call,
    including on gameweek 1 of a new season.
    """
    decay = PARAMS.decay_per_day if decay is None else decay
    min_matches = PARAMS.min_matches_for_fit if min_matches is None else min_matches

    team_ids = sorted({m[0] for m in matches} | {m[1] for m in matches})
    n = len(team_ids)
    if n < 2 or len(matches) < min_matches:
        log.info("team_strength: %d matches < %d required; using priors",
                 len(matches), min_matches)
        return TeamStrengthModel(
            attack={t: 0.0 for t in team_ids},
            defence={t: 0.0 for t in team_ids},
            intercept=float(np.log(1.4)),
            n_matches_fit=len(matches),
            effective_n=0.0,
            target="prior",
            prior_only=True,
        )

    idx = {t: i for i, t in enumerate(team_ids)}
    hi = np.array([idx[m[0]] for m in matches])
    ai = np.array([idx[m[1]] for m in matches])
    hg = np.array([m[2] for m in matches], dtype=float)
    ag = np.array([m[3] for m in matches], dtype=float)

    if target in ("xg", "blend") and xg_matches:
        xg_lookup = {(int(r[0]), int(r[1])): (float(r[2]), float(r[3]))
                     for r in xg_matches}
        hx = np.array([xg_lookup.get((m[0], m[1]), (np.nan, np.nan))[0]
                       for m in matches], dtype=float)
        ax = np.array([xg_lookup.get((m[0], m[1]), (np.nan, np.nan))[1]
                       for m in matches], dtype=float)
        have = ~(np.isnan(hx) | np.isnan(ax))
        if have.any():
            wx = xg_weight if target == "blend" else 1.0
            hg = np.where(have, (1 - wx) * hg + wx * hx, hg)
            ag = np.where(have, (1 - wx) * ag + wx * ax, ag)
        else:
            target = "goals"
    else:
        target = "goals"
    days = np.array([m[4] for m in matches], dtype=float)
    raw_w = np.exp(-decay * days)
    # Kish effective sample size, computed BEFORE normalisation. v6 wrote
    # `w = w / w.mean()` and then took `w.sum()` as the effective count,
    # which is identically len(matches) by construction. The comment claimed
    # time decay was respected; it never reached the regulariser, so ridge
    # strength tracked calendar matches instead of evidence and promoted
    # teams got confident ratings they had not earned.
    effective_n = float((raw_w.sum() ** 2) / max((raw_w ** 2).sum(), 1e-9))
    w = raw_w / raw_w.mean()

    mean_goals = float(np.mean(np.concatenate([hg, ag])))
    x0 = _pack(n, np.zeros(n), np.zeros(n),
               np.log(max(mean_goals, 0.2)), 0.25, -0.05)

    # rho is bounded by the tightest constraint any fixture imposes; using the
    # mean scoring level is a safe, data-derived stand-in for that.
    lo_rho, hi_rho = rho_bounds(mean_goals, mean_goals)
    bounds = ([(-1.5, 1.5)] * (2 * (n - 1))
              + [(np.log(0.3), np.log(5.0)), (-0.3, 0.8),
                 (lo_rho * 0.9, hi_rho * 0.9)])

    # Regularisation scaled to the evidence: strong when a handful of matches
    # have been played, negligible by mid-season. effective_n is the weighted
    # match count, so time decay is respected here too.
    ridge = ridge_strength / max(effective_n, 1.0)

    res = minimize(_neg_log_likelihood, x0,
                   args=(n, hi, ai, hg, ag, w, ridge),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 800, "ftol": 1e-9})

    alpha, beta, intercept, home_adv, rho = _unpack(res.x, n)
    if not res.success:
        log.warning("team_strength: optimiser did not converge (%s)", res.message)

    return TeamStrengthModel(
        attack={t: float(alpha[idx[t]]) for t in team_ids},
        defence={t: float(beta[idx[t]]) for t in team_ids},
        intercept=float(intercept),
        home_adv=float(home_adv),
        rho=float(rho),
        n_matches_fit=len(matches),
        effective_n=effective_n,
        target=target,
        converged=bool(res.success),
    )


def fit_confidence(n_matches: int) -> float:
    """How much the fit should be trusted, on [0, 1].

    Used to weight the market blend and to decide how hard to shrink. Full
    confidence at ~200 matches (half a season across 20 teams); near zero in
    August. This replaces the idea of gating the pipeline on gameweek number
    with something continuous that degrades gracefully instead of flipping.
    """
    return float(np.clip(n_matches / 200.0, 0.0, 1.0))


def blend_with_market(lam: float, mu: float,
                      market_lam: float | None, market_mu: float | None,
                      model_confidence: float) -> tuple[float, float]:
    """Blend the fitted lambdas with odds-implied ones in LOG space.

    Betting markets are the single strongest publicly available prior on match
    outcome; they aggregate team news, motivation and money in a way no
    goals-based fit can. But they are a prior, not a replacement -- so the
    weight given to the market falls as the model's own fit gains confidence.

    model_confidence is in [0, 1]; at 0 the market wins outright, at 1 the
    market contributes ~25%. Blending in log space keeps the result a valid
    rate and prevents a small market number from dragging lambda to zero.
    """
    if market_lam is None or market_mu is None:
        return lam, mu
    w_market = float(np.clip(0.75 * (1.0 - model_confidence) + 0.25, 0.0, 1.0))
    w_model = 1.0 - w_market
    lam_b = float(np.exp(w_model * np.log(max(lam, 1e-6)) +
                         w_market * np.log(max(market_lam, 1e-6))))
    mu_b = float(np.exp(w_model * np.log(max(mu, 1e-6)) +
                        w_market * np.log(max(market_mu, 1e-6))))
    return lam_b, mu_b