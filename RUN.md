# FPL Agent v9.1 — Runbook

## 1. Python environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Environment

Copy `.env.example` to `.env` when creating a new environment. The delivered `.env` already contains the requested local configuration; rotate those secrets before publication.

## 3. Database

Postgres is the primary deployment target. SQLite can be used for local model tests where supported by the configured SQLAlchemy URL.

## 4. Seed static support data

```bash
python scripts/fit_league_translation.py
python scripts/seed_translation.py
python scripts/seed_penalty_duties.py
python scripts/seed_cup_fixtures.py
```

Only run seed commands when their upstream inputs or hand-maintained data are intentionally being refreshed.

## 5. Run ingestion/model pipeline

```bash
python -m fplagent.cli doctor
python -m fplagent.cli demo
python -m fplagent.cli run
```

`demo` is offline/synthetic. `run` consumes configured live services.

## 6. Start API

```bash
uvicorn fplagent.api.main:app --host 127.0.0.1 --port 8000
```

## 7. Start dashboard

In a second terminal:

```bash
cd dashboard
npm ci
npm run dev
```

The Next.js app talks to FastAPI through `/api/*`, keeping the browser on one origin. Configure `FPL_API_URL` and optional `FPL_AGENT_API_KEY` in the dashboard/server environment.

## 8. Verify Player Compare

Open:

```text
http://localhost:3000/compare
```

Select 2–8 players, choose a comparison lens, select a reference player and run the pairwise simulation.

## 9. Test

```bash
pytest -q
```

For the dashboard:

```bash
cd dashboard
npm run typecheck
npm run lint
npm run build
```

## 10. Architecture notes

The predictive model estimates distributions. The decision layer derives interpretable views such as floor/ceiling, minutes security, value efficiency and ownership leverage. The dashboard should not be treated as an oracle: inspect the distribution, the data quality state and the model evaluation history before acting.
