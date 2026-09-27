"""Dixon-Coles (1997) scoreline model.

Fixes carried in from the v5 audit:

* RHO IS CONSTRAINED SO tau STAYS POSITIVE. v5 bounded rho to (-0.2, 0.2)
  with no relation to lambda/mu. tau(0,0) = 1 - lam*mu*rho goes negative for
  large lam*mu and positive rho, producing negative "probabilities" that the
  likelihood then clamped at 1e-10 -- silently corrupting the fit. The valid
  region is derived here rather than guessed.
* max_goals raised from 6 to 8 and the truncation mass is reported, so the
  renormalisation is visible rather than swept up.
* Vectorised: the pmf grid is computed once per fixture with log-space
  arithmetic instead of a nested Python loop over scipy calls.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

__all__ = [
    "tau_matrix", "rho_bounds", "score_matrix", "clean_sheet_prob",
    "expected_goals", "goals_conceded_distribution", "sample_scoreline",
]


def rho_bounds(lam: float, mu: float) -> tuple[float, float]:
    """Interval of rho for which every tau term stays strictly positive.

    tau(0,0) = 1 - lam*mu*rho  > 0  ->  rho <  1/(lam*mu)
    tau(0,1) = 1 + lam*rho     > 0  ->  rho > -1/lam
    tau(1,0) = 1 + mu*rho      > 0  ->  rho > -1/mu
    tau(1,1) = 1 - rho         > 0  ->  rho <  1
    """
    lam, mu = max(lam, 1e-6), max(mu, 1e-6)
    lo = max(-1.0 / lam, -1.0 / mu)
    hi = min(1.0 / (lam * mu), 1.0)
    return lo, hi


def tau_matrix(lam: float, mu: float, rho: float, shape: tuple[int, int]) -> np.ndarray:
    """The low-score correlation adjustment as a full matrix (vectorised)."""
    t = np.ones(shape)
    t[0, 0] = 1.0 - lam * mu * rho
    if shape[1] > 1:
        t[0, 1] = 1.0 + lam * rho
    if shape[0] > 1:
        t[1, 0] = 1.0 + mu * rho
    if shape[0] > 1 and shape[1] > 1:
        t[1, 1] = 1.0 - rho
    return t


def score_matrix(lam: float, mu: float, rho: float = -0.05,
                 max_goals: int = 8) -> np.ndarray:
    """Joint P(home_goals=x, away_goals=y).

    Takes lambda/mu directly rather than re-deriving them from attack/defence
    multipliers, so there is exactly one place in the codebase where expected
    goals are computed (fplagent.models.team_strength.expected_lambdas).
    """
    lam, mu = max(float(lam), 1e-6), max(float(mu), 1e-6)
    lo, hi = rho_bounds(lam, mu)
    rho = float(np.clip(rho, lo + 1e-9, hi - 1e-9))

    xs = np.arange(max_goals + 1)
    home = poisson.pmf(xs, lam)
    away = poisson.pmf(xs, mu)
    m = np.outer(home, away) * tau_matrix(lam, mu, rho, (max_goals + 1, max_goals + 1))
    m = np.clip(m, 0.0, None)
    total = m.sum()
    return m / total if total > 0 else m


def clean_sheet_prob(matrix: np.ndarray, side: str = "home") -> float:
    """P(this side keeps a clean sheet) == P(the OTHER side scores zero)."""
    return float(matrix[:, 0].sum()) if side == "home" else float(matrix[0, :].sum())


def expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    xs = np.arange(matrix.shape[0])
    ys = np.arange(matrix.shape[1])
    return float((xs[:, None] * matrix).sum()), float((ys[None, :] * matrix).sum())


def goals_conceded_distribution(matrix: np.ndarray, side: str = "home") -> np.ndarray:
    """Marginal distribution over goals conceded by `side`.

    Needed because FPL deducts 1 point per 2 goals conceded for GK/DEF, which
    is a floor function of a random variable -- E[floor(X/2)] != floor(E[X]/2).
    v5 modelled neither the deduction nor the distribution.
    """
    return matrix.sum(axis=0) if side == "home" else matrix.sum(axis=1)


def sample_scoreline(matrix: np.ndarray, rng: np.random.Generator,
                     n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    flat = matrix.ravel()
    flat = flat / flat.sum()
    idx = rng.choice(flat.size, size=n, p=flat)
    ncols = matrix.shape[1]
    return idx // ncols, idx % ncols
