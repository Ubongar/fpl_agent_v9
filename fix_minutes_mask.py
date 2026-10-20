from pathlib import Path
import sys

path = Path("fplagent/models/minutes.py")
src = path.read_text(encoding="utf-8")

old = '''        X = np.asarray(X, dtype=float)
        self.feature_mask = _usable_feature_mask(X)
        self.dropped_features = [name for name, keep in
                                 zip(FEATURE_ORDER, self.feature_mask) if not keep]
        if not self.feature_mask.any():
            self.refusal_reason = "every feature is constant"
            log.warning("MinutesModel: %s; staying on heuristic", self.refusal_reason)
            return False
        if self.dropped_features:
            log.info("MinutesModel: dropped %d constant feature(s): %s",
                     len(self.dropped_features), ", ".join(self.dropped_features))
        X = X[:, self.feature_mask]

        if (len(np.unique(y_play)) < 2 or len(np.unique(y_start)) < 2
                or len(np.unique(y_sixty)) < 2):
            self.refusal_reason = "one or more targets have a single class"
            return False

        train_slice, calibration_slice, test_slice = _chronological_train_cal_test_slices(
            len(X), holdout_frac
        )
        X_tr = X[train_slice]
        X_cal = X[calibration_slice]
        X_ho = X[test_slice]
        y_play_tr = y_play[train_slice]
        y_start_tr = y_start[train_slice]
        y_sixty_tr = y_sixty[train_slice]
'''

new = '''        X = np.asarray(X, dtype=float)

        if (len(np.unique(y_play)) < 2 or len(np.unique(y_start)) < 2
                or len(np.unique(y_sixty)) < 2):
            self.refusal_reason = "one or more targets have a single class"
            return False

        train_slice, calibration_slice, test_slice = _chronological_train_cal_test_slices(
            len(X), holdout_frac
        )

        # Compute the usable-feature mask on the training slice only.
        # A column that is constant in X_tr makes sklearn's HistGradientBoosting
        # binning fail with "window shape cannot be larger than input array shape",
        # even if that column varies elsewhere in the full chronological data.
        X_tr_raw = X[train_slice]
        self.feature_mask = _usable_feature_mask(X_tr_raw)
        self.dropped_features = [name for name, keep in
                                 zip(FEATURE_ORDER, self.feature_mask) if not keep]
        if not self.feature_mask.any():
            self.refusal_reason = "every feature is constant in the training slice"
            log.warning("MinutesModel: %s; staying on heuristic", self.refusal_reason)
            return False
        if self.dropped_features:
            log.info("MinutesModel: dropped %d constant feature(s) in training slice: %s",
                     len(self.dropped_features), ", ".join(self.dropped_features))
        X = X[:, self.feature_mask]

        X_tr = X[train_slice]
        X_cal = X[calibration_slice]
        X_ho = X[test_slice]
        y_play_tr = y_play[train_slice]
        y_start_tr = y_start[train_slice]
        y_sixty_tr = y_sixty[train_slice]
'''

if old not in src:
    print("ERROR: expected block not found. No changes made.")
    sys.exit(1)

path.write_text(src.replace(old, new, 1), encoding="utf-8")
print("Patched fplagent/models/minutes.py")