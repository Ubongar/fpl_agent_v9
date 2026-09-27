# FPL Agent v8 — architecture and defect audit

## Scope

This audit covered the full extracted backend repository, including ingestion, feature construction, probabilistic models, simulation, persistence, prediction, evaluation, optimisation, API and release hygiene.

## Release blockers found and fixed

### 1. Runtime prediction path was broken by a stale row variable
`pipeline/predict.py` persisted `team_id`, `position` and price through a row variable that did not exist in the player loop. Prediction creation could therefore fail with a `NameError` once it reached persistence.

**Fix:** `SimPlayer` now carries `team_id` and `now_cost`, and prediction rows read the context from the simulated player.

### 2. Minutes model conflated starting with reaching 60 minutes
`p_start` had been aliased to the 60+ probability, which made calibration target the wrong event.

**Fix:** independent P(play), P(start) and P(60+) events, with separate labels and probabilities.

### 3. Minutes sampling was internally inconsistent
The expected-minutes calculation and the sampled distribution used different mixtures.

**Fix:** expected minutes now uses the same starter/withdrawal/long-sub/short-sub structure as the simulator.

### 4. Lineup sampling could create impossible teams
The earlier sampler could field too many players, too few players after multiple withdrawals, or zero-weight players.

**Fix:** hard formation validation, exact XI construction, zero-weight exclusion, substitution cap, and replacement coverage for every simulated starter withdrawal.

### 5. Goal and assist events could break conservation
The earlier event layer could over-allocate goals, treat penalty goals as assistable, or mix own goals into scorer totals.

**Fix:** scoreline first; own goals, penalty goals and open-play goals are explicit subsets; assists are limited to open-play goals; diagnostics distinguish scorer allocation from team-score reconstruction.

### 6. Penalty BPS was incorrect
Penalty goals had been valued using positional goal BPS rather than the current position-neutral penalty-goal value.

**Fix:** `BPS_PENALTY_GOAL = 12` and simulation passes the realised penalty-goal array into the BPS function.

### 7. BPS defensive actions were confused with DefCon actions
The simulator was using CBIT/CBIRT-style DefCon totals as though they were the BPS clearances/blocks/interceptions bucket, which can include tackles and recoveries.

**Fix:** the BPS function accepts an explicit `cbi_actions` input. Historical baseline estimation uses the actual clearances/blocks/interceptions source column; the live simulator does not fabricate an event-level CBI rate.

### 8. Free-transfer rule was under-modelled
The code had the five-transfer banking rule but the ordinary weekly transfer cap default was inconsistent with current 2026/27 rules.

**Fix:** default ordinary transfer cap is now 20; free-transfer bank remains capped at five and the 4-point hit cost is explicit.

### 9. Transfer carry-forward could be gamed by the ILP
Upper bounds alone let the solver understate `used` and overstate future banked transfers.

**Fix:** exact linearisation and binary cap gate enforce the true capped carry-forward recurrence.

### 10. Solver dependency broke unrelated imports
PuLP was imported at module load in optimiser modules. That meant a missing solver could prevent API/model imports even when optimisation was not being used.

**Fix:** lazy solver import with an explicit runtime error telling the operator to install project requirements.

### 11. Squad solution mixed XI points with captain extra
`xi_expected_points` included the captain's extra copy while being named as XI expected points.

**Fix:** `xi_expected_points` is now the un-doubled XI sum, with `captain_extra_expected_points` stored separately.

### 12. Documentation claimed features that were not in the archive
README/RUN referenced a Next.js dashboard, Streamlit UI and ~120 tests that were not present.

**Fix:** documentation is now backend-accurate. The dashboard is explicitly the next phase, and the checked-in regression suite is the source of truth.

### 13. Secrets were present in the uploaded archive
The source archive contained a populated `.env` with database, odds, external-source and LLM credentials.

**Fix:** `.env` is excluded from delivery; `.env.example` contains configuration keys only.

## Statistical and architecture checks retained

- Chronological tail holdout for the learned minutes model.
- Point-in-time historical feature construction.
- Versioned set-piece duty reads.
- Deadline-aware odds reads.
- Immutable/pre-deadline prediction checks.
- Distribution-based DGW probability aggregation.
- Joint scoreline simulation and common goal times for on-pitch concessions.
- Fixture-wide BPS ranking with tie handling.
- Defensive-contribution threshold sampling instead of a linear rate-to-points shortcut.

## Current known limitations

The project should not claim more observability than the input sources provide. Current simulation does not synthesize inside-box-save or big-chance-save labels, and does not claim to reproduce every hidden BPS event. The historical per-player BPS baseline absorbs unmodelled components; adding event-level telemetry later can replace that approximation incrementally.

The learned minutes model is deliberately gated by sufficient labelled rows and remains fallback-safe.

PuLP was not present in the delivery sandbox. Therefore the regression suite validates non-solver logic and optimizer module importability, but the CBC ILP execution itself was not run here. A normal installation from `requirements.txt` is required for final solver-level integration testing.

## Verification performed

```text
pytest -q                         -> 15 passed
python -m compileall -q ...       -> passed
python -m fplagent.cli --help    -> passed
```

Additional targeted checks verified that high withdrawal pressure never reduced the simulated team below 11 active players within the 15-player simulation pool.
