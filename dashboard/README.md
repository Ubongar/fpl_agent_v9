# FPL Agent Dashboard — v3

A decision-intelligence dashboard for the FPL Agent backend.

## What changed in v3

### Player Compare is a first-class workspace
- Select 2–8 players from the Player Lab or directly from `/compare`.
- Persist comparison selections in the browser while moving between views.
- Choose a reference player and inspect every other player's delta against it.
- Switch between six comparison lenses: output, floor, ceiling, minutes, value and differential.
- Compare P10/P25/P50/P75/P90, haul/blank probabilities, Monte Carlo uncertainty, minutes security, value and ownership leverage.
- Run a pairwise simulated outcome comparison using the backend's stored distribution.

### Workspace redesign
- Desktop side navigation groups Core, Planning and Evidence surfaces.
- Mobile navigation keeps the highest-value workflows reachable.
- Player Compare is permanently visible in the main navigation and header.
- Player X-ray links directly into the comparison desk.

### Model decision layer surfaced in the UI
The UI can consume the v9 decision metrics:
- certainty and Monte Carlo uncertainty
- upside / downside / total range
- start security / 60+ security
- minutes stability
- xP per £m
- floor / ceiling adjusted xP
- floor / ceiling value per £m
- ownership-adjusted upside / downside
- quartile range
- start gap and bench risk

## Run

```bash
npm ci
npm run dev
```

Default URL: `http://localhost:3000`

The dashboard talks to the FastAPI backend through the same-origin `/api/*` route.

Create `.env.local` from `.env.example` and set:

```text
FPL_API_URL=http://127.0.0.1:8000
FPL_AGENT_API_KEY=
```

Never commit a real `.env` or `.env.local` file.

## Validation

The clean source release intentionally excludes `node_modules` and `.next`.
Install dependencies locally before running typecheck/build:

```bash
npm ci
npm run typecheck
npm run lint
npm run build
```

The source was syntax-checked in the release environment across all TS/TSX files.
