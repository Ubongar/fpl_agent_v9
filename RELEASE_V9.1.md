# FPL Agent v9.1 — Full Build

This is a merged release built around the requested repository structure.

## Included

- Python backend/model v9.1.
- Original data snapshots and league translation data.
- Seed/utility scripts.
- Full dashboard source under `dashboard/`.
- First-class `/compare` Player Comparison page.
- Command Center, Player Lab, My Team, Fixture Lab, Market Watch and Model Lab.
- Regression tests and lightweight Streamlit companion.
- Root `.env`, because it was explicitly supplied for this delivery, plus sanitized `.env.example`.

## Release hygiene

Excluded: `node_modules`, `.next`, Python caches/bytecode, pytest caches and runtime databases.

## Validation

The bundled v8 regression suite passed in the build environment. The merged repository suite contains 30 tests covering the regression suite, decision layer, optimizer imports, simulation legality, scoring and static data readability.
