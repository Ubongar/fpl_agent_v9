# Dashboard v3 change log

## New
- `src/app/compare/page.tsx` — full Player Comparison workspace.
- `src/components/CompareUI.tsx` — comparison headers, metric lenses, delta matrix, profile bars and summary cards.
- `src/components/AppNav.tsx` — desktop workspace navigation + mobile navigation.
- `REPLACE_THIS_DASHBOARD.md` — direct drop-in replacement instructions.

## Upgraded
- `src/app/layout.tsx` — full workspace shell and persistent Compare access.
- `src/app/players/page.tsx` — persistent comparison set and sticky comparison tray.
- `src/app/players/[id]/page.tsx` — direct add-to-comparison action.
- `src/app/page.tsx` — quick top-pair comparison entry.
- `src/lib/types.ts` — transparent decision metrics including bench risk, start gap and floor/ceiling value.
- Backend `fplagent/models/decision.py` — additional comparison dimensions.
- Backend `fplagent/api/main.py` — compare endpoint exposes the additional dimensions.
- Backend regression coverage extended for the new metrics.
