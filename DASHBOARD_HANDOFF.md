# Dashboard handoff for FPL Agent v8

The backend is the source of truth. The frontend should consume FastAPI/OpenAPI and should not reproduce optimisation, scoring or data-versioning rules.

## Recommended application areas

### Overview
Show current gameweek, model status, prediction freshness, quality status, team strength snapshot and a compact set of actionable distributions.

Primary API:

```text
/gameweeks/current
/health
/quality
/team-strength
/predictions
```

### Player Explorer
Filter by position/team/fixture and surface xP mean, standard error, p10/p50/p90, P(play), P(start), P(60+), haul probability and explanatory components.

Primary API:

```text
/predictions
/predictions/{player_id}
/set-pieces
/changes
```

### Squad Builder
Consume the optimiser result rather than rebuilding constraints in the browser. Display squad, starting XI, bench order, formation, budget, captain, vice-captain and solver status.

Primary API:

```text
POST /optimise/squad
```

### Transfers
Show current squad state, free-transfer bank, proposed moves, hits, budget and projected multi-gameweek gain. Make the 5-bank/20-transfer/4-point rules visible as metadata, not duplicated constants.

Primary API:

```text
GET  /manager/{entry_id}/team
POST /optimise/transfers
```

### Captaincy
Show expected captain points, P10/P90, haul probability, effective ownership context and head-to-head distributions. Do not turn the result into an unexplained single score.

Primary API:

```text
POST /optimise/captaincy
POST /compare/captaincy
POST /rank-impact
```

### Chips
Show the first-half/second-half availability windows, scheduled gameweek, value estimate and rule notes. Do not model chip legality independently in TypeScript.

Primary API:

```text
POST /optimise/chips
```

### Evaluation / Model health
Show walk-forward metrics, baseline comparisons, calibration/quantile checks, promotion state and historical model versions.

Primary API:

```text
/evaluation/history
/evaluation/latest
/evaluation/track-record
```

### Manager / Decisions
Allow a user to load a real FPL team and record decision outcomes separately from model predictions.

Primary API:

```text
GET  /manager/{entry_id}/team
POST /decisions
POST /decisions/{rec_id}/outcome
GET  /decisions/track-record
```

## Frontend state boundaries

The frontend owns:

- filters and URL state;
- presentation preferences;
- selected comparison players;
- chart windows;
- loading/error states.

The backend owns:

- FPL rules;
- point-in-time data selection;
- all scoring arithmetic;
- Monte Carlo simulation;
- optimiser constraints;
- transfer/captain/chip legality;
- model confidence and diagnostics.

## Error contract

Every API call should treat HTTP errors and solver/model status fields as first-class UI states. A solver timeout, insufficient historical data or unavailable enrichment source must never render as an empty successful recommendation.

## OpenAPI-first integration

Run the API and use its generated schema to create frontend types. The frontend should not infer response shapes from SQLAlchemy tables or Python internals.

Suggested development loop:

```text
backend change
    -> OpenAPI change
    -> regenerate TS types
    -> frontend compile catches renamed fields
```

## Dashboard implementation order

1. Shell/navigation + API client.
2. Overview + Player Explorer.
3. Squad Builder + Captaincy.
4. Transfers + Manager team.
5. Chips + Evaluation.
6. Decisions + explanation drilldowns.

The dashboard should initially focus on rendering the existing distributions and diagnostics faithfully. New model features should land in the backend/API first, then be exposed to the UI.
