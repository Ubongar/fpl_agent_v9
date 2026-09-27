# FPL Agent v9.1

A probabilistic Fantasy Premier League research and decision system with a Python prediction/simulation/optimisation backend and a Next.js dashboard.

## Architecture

FPL / FBref / odds / lineup data -> ingestion -> quality gates -> point-in-time features -> team strength + player models -> joint Monte Carlo simulation -> predictions/evaluation -> optimisers -> FastAPI -> Next.js dashboard.

The dashboard is intentionally decision-oriented. It exposes distributions and uncertainty separately rather than reducing every player to one opaque score.

## Key capabilities

- Joint match Monte Carlo with a shared scoreline and legal lineups.
- Minutes model with separate P(play), P(start), and P(60+).
- Expected points, quantiles, haul/blank probabilities and simulation uncertainty.
- Floor/ceiling, minutes-security, value and ownership-adjusted decision metrics.
- Historical walk-forward evaluation and calibration diagnostics.
- Squad, transfer, captaincy and chip optimisation.
- Player Compare workspace for 2–8 players.
- Fixture Lab, Market Watch, Model Lab and manager-specific My Team views.
- LLM explanations through an OpenAI-compatible endpoint.

## Important data integrity commitments

Predictions are point-in-time. Historical feature rows preserve their original known-at timestamp. Re-ingestion is not allowed to rewrite historical availability into the present.

Simulation events are coupled so the same realised match drives player outcomes. Illegal squad configurations raise hard errors rather than silently producing corrupted simulations.

## Run

### Backend

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m fplagent.cli doctor
uvicorn fplagent.api.main:app --host 127.0.0.1 --port 8000
```

### Dashboard

```bash
cd dashboard
npm ci
npm run dev
```

Open http://localhost:3000.

## Environment

The supplied `.env` is included in this delivery because it was explicitly requested. Treat it as local-only and rotate the database/odds/football-data/LLM credentials before sharing the repository or committing it.

The dashboard reads `FPL_API_URL` server-side through the Next.js same-origin proxy. `FPL_AGENT_API_KEY` is never exposed to browser code.

## Testing

```bash
pytest -q
cd dashboard
npm ci
npm run typecheck
npm run lint
npm run build
```
