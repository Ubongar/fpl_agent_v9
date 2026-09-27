"""Card risk: per-player, shrunk, not a positional flat rate.

WHY THIS EXISTS
---------------
v7 modelled yellow and red cards as a positional league average applied
uniformly: every midfielder got the same 0.16 yellows per 90, every defender
0.17, regardless of who they actually were. That is a real simplification --
some players are genuinely more card-prone than others (a combative
ball-winning midfielder draws far more yellows than a deep-lying passer at
the same position), and it costs nothing to fix: `yellow_cards` and
`red_cards` are already ingested per fixture per player, unused for this.

THE METHOD, DELIBERATELY THE SAME SHAPE AS DEFCON
--------------------------------------------------
This mirrors `models/defcon.py` on purpose rather than inventing a new
pattern: shrunk per-90 rate toward a positional prior, weighted by 90-minute
equivalents so a handful of cameo appearances cannot masquerade as a stable
rate. Cards are rarer than defensive actions, so the prior carries more
weight for longer (the shrinkage constant here is larger), and the Monte
Carlo already draws each player's cards as a Bernoulli per simulation off a
per-90 rate -- this module only replaces the FLAT rate that feeds that draw
with a PLAYER-SPECIFIC one.

Red cards are pooled with yellows for the purpose of shrinkage (both draw on
the same disciplinary-temperament signal, and reds are too rare on their own
to fit a stable per-player rate independently), but reported and sampled
separately so the -3 event stays distinct from the -1 event downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fplagent.config import PARAMS

# League-average rates per 90, by position. Used as the prior and as the
# fallback when a player has no card history at all.
YELLOW_PRIOR_PER90 = {"GK": 0.04, "DEF": 0.17, "MID": 0.16, "FWD": 0.12}
RED_PRIOR_PER90 = {"GK": 0.004, "DEF": 0.010, "MID": 0.007, "FWD": 0.006}


@dataclass
class CardRate:
    yellow_per90: float
    red_per90: float
    games_used: float
    confidence: str          # prior_only | shrunk | data_driven


def estimate_card_rate(position: str, yellow_history: list[float],
                       red_history: list[float],
                       minutes_history: list[int]) -> CardRate:
    """Shrunk per-90 card rates for one player.

    `minutes_history`, `yellow_history` and `red_history` are aligned, one
    entry per fixture. Shrinkage is in 90-minute equivalents, same as
    `defcon.estimate_defcon_rate` and `shares.estimate_shares`, so a player
    with ten 20-minute cameos is correctly treated as having a fifth of a
    full season's evidence, not ten games of it.
    """
    y_prior = YELLOW_PRIOR_PER90.get(position, 0.13)
    r_prior = RED_PRIOR_PER90.get(position, 0.007)

    if not minutes_history or not yellow_history:
        return CardRate(y_prior, r_prior, 0.0, "prior_only")

    mins = np.array(minutes_history, dtype=float)
    yel = np.array(yellow_history, dtype=float)
    red = np.array(red_history, dtype=float) if red_history else np.zeros_like(yel)

    played = mins > 0
    if not played.any():
        return CardRate(y_prior, r_prior, 0.0, "prior_only")

    games_equiv = float(mins[played].sum() / 90.0)
    raw_y90 = float(yel[played].sum() / max(games_equiv, 1e-6))
    raw_r90 = float(red[played].sum() / max(games_equiv, 1e-6))

    k = PARAMS.card_shrinkage_games
    w = games_equiv / (games_equiv + k)
    y90 = w * raw_y90 + (1 - w) * y_prior
    r90 = w * raw_r90 + (1 - w) * r_prior

    conf = ("data_driven" if games_equiv >= 2 * k
            else ("shrunk" if games_equiv > 0 else "prior_only"))
    return CardRate(float(max(y90, 0.0)), float(max(r90, 0.0)), games_equiv, conf)
