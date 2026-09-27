# FPL Agent v9 — model + product overhaul

## Why v9 exists

v8 hardened the statistical engine. v9 adds a decision-intelligence layer and a much broader dashboard contract so the product can answer not only “who projects well?” but also “how certain is that projection, what is the downside, what is the upside, how fragile is the minutes assumption, and how does ownership change the payoff?”

## Model / engine changes

### 1. Transparent decision metrics
Every current prediction can now be translated into separate metrics:

- Monte Carlo uncertainty and certainty;
- P10 / P50 / P90 range, upside and downside;
- start security and 60+ security conditional on P(play);
- minutes stability;
- xP per £m;
- floor-adjusted xP and ceiling-adjusted xP;
- ownership-adjusted upside and downside.

No new metric is presented as a universal score.

### 2. Stale-market protection
The team-strength model can blend betting-market information, but only when the snapshot was pulled within 72 hours before the prediction cutoff. Older odds are ignored rather than silently reused.

### 3. Version bump
Prediction versions move from `v8-*` to `v9-*`.

## Dashboard changes

The dashboard has been rebuilt around a decision workflow:

`Command Center -> Player Lab -> Compare -> Fixture Lab -> Market Watch -> Squad/Transfers/Captaincy/Chips -> Model Lab`

New API-backed surfaces:

- `/dashboard/overview`
- `/players/compare`
- `/fixtures/outlook`
- `/market/watch`
- `/model/dashboard`

The old planner pages remain available and inherit the new visual system rather than being removed.

## Important limitations that remain explicit

- Fixture Lab difficulty is a model-derived proxy, not the Premier League's official FDR.
- Market Watch transfer pressure is not a guaranteed price-change probability.
- A learned minutes model still falls back to the heuristic when training evidence/calibration is insufficient.
- The ILP optimizer still requires PuLP at runtime.

## Validation

The Python regression suite contains 18 passing tests after the v9 additions, including decision-metric invariants and stale-market protection.

### 4. Ownership fallback hardening

When an effective-ownership sample is missing, the decision layer now falls back to the stored selected-by percentage instead of treating missing effective ownership as zero ownership. A regression test protects this path.
