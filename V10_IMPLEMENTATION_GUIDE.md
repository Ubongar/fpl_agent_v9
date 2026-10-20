# FPL Agent v10 accuracy patch

## Files updated

1. `fplagent/config.py` — adds the tunable xA assist-share blend weight (`0.65`, or `0.0` to disable).
2. `fplagent/models/shares.py` — blends observed assists with an xA-based estimate only when both player and team xA data are available; records provenance.
3. `fplagent/features/build.py` — aggregates team xA, passes xA to the share model, uses snapshot-only prices for price ranks, and adds a strict historical metadata mode.
4. `fplagent/pipeline/predict.py` — automatically uses strict point-in-time metadata for cutoffs more than 24 hours in the past.
5. `fplagent/models/minutes.py` — separates model training, calibration, and final testing chronologically.
6. `tests/test_v10_accuracy_patch.py` — new tests for xA blending, snapshot-only pricing, and chronological split boundaries.

## Apply on Windows

From the repository root (the folder containing `pyproject.toml`):

```powershell
git switch -c v10-accuracy-xa
python apply_v10_accuracy_patch.py --dry-run
python apply_v10_accuracy_patch.py
pytest -q
```

The ZIP preserves the paths for the test file and implementation guide. The patch script makes the source edits after validating all expected anchors; if it reports a mismatch, stop and inspect the current file instead of forcing the patch.

## Important

- xA is an experimental additional signal, not a guaranteed accuracy improvement. Keep it only if walk-forward backtests beat the observed-assists-only version and the existing baselines over multiple gameweeks.
- Historical replay cannot reconstruct players missing from the current player table without historical roster backfill. This patch avoids current-metadata fallback for players it can identify, but it does not invent historical rosters.
- Strict replay may predict fewer players if a valid pre-deadline snapshot is absent. That is preferable to evaluating predictions built with future information.
- Minutes-model calibration scores may look worse after this change because the final test observations are no longer used to fit the calibrator. That is a more honest estimate.
- The starting xA blend of `0.65` is a tunable value; tune on training/validation weeks and evaluate final settings on later untouched weeks.
