"""Playing-time model: the highest-leverage model in the whole system.

WHY THIS COMES FIRST
--------------------
A player who does not start scores zero, however good the xG model is.
Everything downstream multiplies through minutes, so an error here propagates
into every single number the system produces. v5 handled it with three
hand-tuned constants and a base rate of `mean(minutes)/90`.

THREE STRUCTURAL FIXES
----------------------
1. THREE SEPARATE EVENTS, NOT ONE.
   FPL pays 1 point for 1-59 minutes and 2 for 60+. v5 collapsed this to
   `p_start * 2`. v6 predicts P(play at all) and P(60+ | play) separately,
   because a nailed-on 75-minute starter and a guaranteed 20-minute substitute
   are completely different assets that v5 could not distinguish.

2. start_prob != minutes/90.
   v5 set base_prob = mean_minutes/90, so a player who plays exactly 60
   minutes every single week got start_prob 0.67 -- as if he were a rotation
   risk. He is not; he is certain to play. Rate of appearance and length of
   appearance are different quantities and are modelled separately here.

3. A LEARNED MODEL, NOT A HEURISTIC -- BUT ONLY WHEN IT EARNS THE SWAP.
   The heuristic remains as a cold-start fallback and is used whenever there
   are too few labelled rows to fit. Once there are, a gradient-boosted
   classifier on point-in-time features is a CANDIDATE for replacing it.
   The swap is gated on a holdout check: the model must beat a naive
   recency baseline on ranking (AUC), or the swap is refused and the
   heuristic is kept. A model that trains successfully is not automatically
   a model that ranks better. On distributions where a flexible learner
   cannot beat a recency average -- such as synthetic data with constant
   per-player hazards -- the swap is correctly refused rather than allowed
   to silently degrade the pipeline. `refusal_reason` on the instance
   records WHY the swap was refused, so the caller can log it accurately.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

UNAVAILABLE = {"i", "s", "u", "n"}   # injured / suspended / unavailable / not in squad
DOUBTFUL = "d"


@dataclass
class MinutesPrediction:
    p_play: float
    p_sixty: float
    expected_minutes: float
    source: str                                  # heuristic | gbm
    drivers: dict = field(default_factory=dict)  # for the explanation layer

    p_start_est: float = 0.0

    @property
    def p_start(self) -> float:
        """P(named in the starting XI).

        NOT the same as P(60+): a starter withdrawn on 55 minutes started and
        did not reach sixty. v6 aliased the two and persisted p_sixty into
        the `p_start` column, so the backtest then calibrated P(60+) against
        the `starts` flag and measured the wrong thing. They are separate
        numbers now.
        """
        return float(np.clip(self.p_start_est, 0.0, 1.0))


def _recency_weights(n: int, half_life: float = 4.0) -> np.ndarray:
    if n == 0:
        return np.array([])
    ages = np.arange(n - 1, -1, -1, dtype=float)
    w = 0.5 ** (ages / half_life)
    return w / w.sum()


def heuristic_minutes(minutes_history: list[int], status: str = "a",
                      chance_of_playing: float | None = None,
                      starts_history: list[int] | None = None,
                      days_since_return: int | None = None,
                      rotation_risk: float = 0.0,
                      congestion_matches_14d: int = 0,
                      is_suspended_next: bool = False) -> MinutesPrediction:
    """Transparent fallback. Every adjustment is reported in `drivers` so the
    number is explainable even before a model is fitted."""
    drivers: dict[str, float] = {}

    if is_suspended_next or status in UNAVAILABLE:
        return MinutesPrediction(0.0, 0.0, 0.0, "heuristic",
                                 {"unavailable": 1.0, "status": status})

    if not minutes_history:
        p_play, p_start, p_sixty = 0.45, 0.30, 0.24
        exp_min = (min(p_start, p_sixty) * 80.0
                   + max(p_start - p_sixty, 0.0) * 52.0
                   + max(p_sixty - p_start, 0.0) * 72.0
                   + max(p_play - p_start - max(p_sixty - p_start, 0.0), 0.0) * 25.0)
        drivers["no_history"] = 1.0
    else:
        hist = np.array(minutes_history, dtype=float)
        w = _recency_weights(len(hist))
        if starts_history is None:
            start_hist = (hist >= 60).astype(float)
        else:
            start_hist = np.asarray(starts_history, dtype=float)
            if len(start_hist) != len(hist):
                raise ValueError("starts_history must align with minutes_history")
            start_hist = (start_hist > 0).astype(float)
        p_play = float(np.dot((hist > 0).astype(float), w))
        p_start = float(np.dot(start_hist, w))
        p_sixty = float(np.dot((hist >= 60).astype(float), w))
        played = hist[hist > 0]
        started_mins = hist[start_hist.astype(bool)]
        avg_len = float(played.mean()) if played.size else 30.0
        avg_starter_len = float(started_mins.mean()) if started_mins.size else avg_len
        sub_mins = hist[(hist > 0) & ~start_hist.astype(bool)]
        avg_sub_len = float(sub_mins.mean()) if sub_mins.size else 22.0
        p60_start = min(p_start, p_sixty)
        early_starter_mass = max(p_start - p60_start, 0.0)
        long_sub_mass = max(p_sixty - p_start, 0.0)
        short_sub_mass = max(p_play - p_start - long_sub_mass, 0.0)
        exp_min = (p60_start * avg_starter_len
                   + early_starter_mass * min(avg_starter_len, 52.0)
                   + long_sub_mass * 72.0
                   + short_sub_mass * avg_sub_len)
        drivers["base_p_play"] = round(p_play, 3)
        drivers["base_p_start"] = round(p_start, 3)
        drivers["base_p_sixty"] = round(p_sixty, 3)

    if chance_of_playing is not None:
        c = float(np.clip(chance_of_playing / 100.0, 0.0, 1.0))
        p_play *= c
        p_start *= c
        p_sixty *= c
        drivers["chance_of_playing"] = round(c, 3)
    elif status == DOUBTFUL:
        p_play *= 0.55
        p_start *= 0.50
        p_sixty *= 0.45
        drivers["doubtful_flag"] = 0.55

    if days_since_return is not None and days_since_return < 21:
        ramp = float(np.clip(days_since_return / 21.0, 0.0, 1.0))
        factor = 0.5 + 0.5 * ramp
        p_sixty *= factor
        p_start *= (0.8 + 0.2 * ramp)
        p_play *= (0.8 + 0.2 * ramp)
        drivers["return_from_injury"] = round(factor, 3)

    if rotation_risk > 0:
        factor = 1.0 - 0.35 * float(np.clip(rotation_risk, 0.0, 1.0))
        p_play *= factor
        p_start *= factor
        p_sixty *= factor
        drivers["rotation_risk"] = round(factor, 3)

    if congestion_matches_14d >= 3:
        factor = 1.0 - 0.06 * (congestion_matches_14d - 2)
        factor = float(np.clip(factor, 0.75, 1.0))
        p_play *= factor
        p_start *= factor
        p_sixty *= factor
        drivers["congestion"] = round(factor, 3)

    p_play = float(np.clip(p_play, 0.0, 0.99))
    p_start = float(np.clip(min(p_start, p_play), 0.0, 0.99))
    p_sixty = float(np.clip(min(p_sixty, p_play), 0.0, 0.99))
    exp_min = float(np.clip(exp_min * (p_play / max(drivers.get("base_p_play", p_play), 1e-6))
                            if "base_p_play" in drivers else exp_min, 0.0, 90.0))
    return MinutesPrediction(p_play, p_sixty, round(exp_min, 1), "heuristic", drivers,
                             p_start_est=p_start)


def sample_minutes(pred: MinutesPrediction, n_sims: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Draw realised minutes per simulation.

    Three-state: doesn't feature (0), substitute appearance (uniform 5-59),
    full-ish appearance (normal around 80, truncated to [60, 90]). This is what
    makes appearance points, clean sheets and DefCon correlate properly --
    they all depend on the SAME realised minutes draw, not on independent
    coin flips.
    """
    u = rng.random(n_sims)
    plays = u < pred.p_play
    starts = u < pred.p_start
    p60_from_starters = min(pred.p_sixty, pred.p_start)
    long_start_prob = (p60_from_starters / pred.p_start
                       if pred.p_start > 1e-9 else 0.0)
    remaining_60_mass = max(pred.p_sixty - p60_from_starters, 0.0)
    sub_rate = max(pred.p_play - pred.p_start, 0.0)
    long_sub_prob = min(remaining_60_mass / sub_rate, 1.0) if sub_rate > 1e-9 else 0.0
    starter_long = rng.random(n_sims) < long_start_prob
    starter_short = rng.integers(46, 60, size=n_sims)
    starter_full = np.clip(rng.normal(80, 8, size=n_sims), 60, 90).astype(int)
    starter_minutes = np.where(starter_long, starter_full, starter_short)
    sub_long = rng.random(n_sims) < long_sub_prob
    sub_short = rng.integers(5, 46, size=n_sims)
    sub_full = np.clip(rng.normal(72, 9, size=n_sims), 60, 90).astype(int)
    sub_minutes = np.where(sub_long, sub_full, sub_short)
    return np.where(starts, starter_minutes, np.where(plays, sub_minutes, 0))


# ---------------------------------------------------------------------------
# Learned model
# ---------------------------------------------------------------------------

FEATURE_ORDER = [
    "starts_last_5", "minutes_mean_last_5", "minutes_mean_last_3",
    "played_last_match", "started_last_match", "days_since_last_start",
    "chance_of_playing", "is_doubtful", "price_rank_in_team",
    "team_matches_last_14d", "season_minutes_share", "selected_by_percent",
]


def _usable_feature_mask(X: np.ndarray) -> np.ndarray:
    """Columns with at least two distinct non-NaN values.

    sklearn's HistGradientBoostingClassifier fails on a constant feature
    with `ValueError: window shape cannot be larger than input array shape`
    from `sliding_window_view(distinct_values, 2)` inside its binning step.
    A constant feature carries no information anyway, so the correct fix is
    to drop it rather than jitter it into being non-constant. An all-NaN
    column is treated as constant because the value is unknown for every row.
    """
    mask = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[~np.isnan(col)]
        mask[j] = np.unique(col).size >= 2
    return mask


class MinutesModel:
    """Gradient-boosted P(play) and P(60+), with the heuristic as fallback.

    A successful `fit` is a NECESSARY but not SUFFICIENT condition for the
    swap. The last step of `fit` compares the model's holdout AUC against a
    naive recency baseline for every target it has fitted. If the model does
    not beat the naive baseline on a target, `refusal_reason` records which
    one and by how much, and `fitted` returns False so the caller falls
    back to the heuristic.
    """

    def __init__(self) -> None:
        self.play_model = None
        self.start_model = None
        self.sixty_model = None
        self.play_calibrator = None
        self.start_calibrator = None
        self.sixty_calibrator = None
        self.n_train = 0
        self.holdout: dict = {}
        self.feature_mask: np.ndarray | None = None
        self.dropped_features: list[str] = []
        # Human-readable reason the last `fit` call did not result in a
        # usable model. Empty when `fit` succeeded. Set at every early return
        # so a caller can log WHY the swap was refused rather than a generic
        # "fit returned False".
        self.refusal_reason: str = ""
        self._holdout_pred: dict[str, np.ndarray] = {}
        self._holdout_y: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y_play: np.ndarray, y_start: np.ndarray,
            y_sixty: np.ndarray, min_rows: int = 400, holdout_frac: float = 0.2) -> bool:
        """Fit classifiers with a chronological split, calibration, and gate."""
        self.refusal_reason = ""
        if X is None or len(X) < min_rows:
            self.refusal_reason = (
                f"only {0 if X is None else len(X)} rows, minimum is {min_rows}")
            log.info("MinutesModel: %s; staying on heuristic", self.refusal_reason)
            return False
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            self.refusal_reason = "scikit-learn not installed"
            log.warning("scikit-learn not installed; staying on heuristic")
            return False

        X = np.asarray(X, dtype=float)
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

        cut = int(len(X) * (1.0 - holdout_frac))
        X_tr, X_ho = X[:cut], X[cut:]
        if (len(X_ho) < 50 or len(np.unique(y_play[:cut])) < 2
                or len(np.unique(y_start[:cut])) < 2
                or len(np.unique(y_sixty[:cut])) < 2):
            X_tr, X_ho = X, X[:0]

        common = dict(max_iter=300, learning_rate=0.06, max_depth=5,
                      l2_regularization=1.0, early_stopping=False,
                      random_state=7)
        self.play_model = HistGradientBoostingClassifier(**common).fit(
            X_tr, y_play[:len(X_tr)])
        self.start_model = HistGradientBoostingClassifier(**common).fit(
            X_tr, y_start[:len(X_tr)])
        self.sixty_model = HistGradientBoostingClassifier(**common).fit(
            X_tr, y_sixty[:len(X_tr)])

        if len(X_ho) >= 50:
            for name, model, y in (("play", self.play_model, y_play),
                                   ("start", self.start_model, y_start),
                                   ("sixty", self.sixty_model, y_sixty)):
                raw = model.predict_proba(X_ho)[:, 1]
                y_ho = y[len(X_tr):]
                if len(np.unique(y_ho)) < 2:
                    continue
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(raw, y_ho)
                setattr(self, f"{name}_calibrator", iso)
                calibrated = iso.predict(raw)
                self._holdout_pred[name] = np.asarray(calibrated, dtype=float)
                self._holdout_y[name] = np.asarray(y_ho, dtype=float)
                self.holdout[f"{name}_brier_raw"] = float(np.mean((raw - y_ho) ** 2))
                self.holdout[f"{name}_brier_cal"] = float(
                    np.mean((calibrated - y_ho) ** 2))

        self.n_train = len(X_tr)

        # --- swap gate ------------------------------------------------------
        # Compare holdout AUC against `starts_last_5/5`, which is FEATURE_ORDER[0]
        # and is always kept by the mask. On data with constant per-player
        # hazards this naive baseline is close to Bayes-optimal and a flexible
        # learner will lose to it. That is exactly when we want the learner to
        # lose, because a flexible learner fitting noise is strictly worse
        # than the recency average it would replace.
        if len(X_ho) >= 100 and self._holdout_pred:
            try:
                from sklearn.metrics import roc_auc_score
            except ImportError:
                roc_auc_score = None      # type: ignore[assignment]
            if roc_auc_score is not None:
                naive = X_ho[:, 0] / 5.0
                fails: list[tuple[str, float, float]] = []
                for name in ("play", "start", "sixty"):
                    if name not in self._holdout_pred:
                        continue
                    y_ho = self._holdout_y[name]
                    try:
                        model_auc = float(roc_auc_score(y_ho, self._holdout_pred[name]))
                        naive_auc = float(roc_auc_score(y_ho, naive))
                    except ValueError:
                        continue
                    self.holdout[f"{name}_auc_model"] = round(model_auc, 4)
                    self.holdout[f"{name}_auc_naive"] = round(naive_auc, 4)
                    if model_auc < naive_auc:
                        fails.append((name, model_auc, naive_auc))
                if fails:
                    worst = min(fails, key=lambda t: t[1] - t[2])
                    self.refusal_reason = (
                        f"holdout AUC for {worst[0]} is {worst[1]:.3f}, "
                        f"below the naive last-5-starts baseline {worst[2]:.3f}"
                    )
                    log.info(
                        "MinutesModel: refusing the swap and keeping the "
                        "heuristic -- %s", self.refusal_reason)
                    self.play_model = None
                    self.start_model = None
                    self.sixty_model = None
                    self.play_calibrator = None
                    self.start_calibrator = None
                    self.sixty_calibrator = None
                    return False

        return True

    @property
    def fitted(self) -> bool:
        return self.play_model is not None

    def predict(self, features: dict) -> MinutesPrediction | None:
        if not self.fitted:
            return None
        x = np.array([[features.get(k, np.nan) for k in FEATURE_ORDER]], dtype=float)
        if self.feature_mask is not None:
            x = x[:, self.feature_mask]
        p_play = float(self.play_model.predict_proba(x)[0, 1])
        p_start = float(self.start_model.predict_proba(x)[0, 1])
        p_sixty = float(self.sixty_model.predict_proba(x)[0, 1])
        if self.play_calibrator is not None:
            p_play = float(self.play_calibrator.predict([p_play])[0])
        if self.start_calibrator is not None:
            p_start = float(self.start_calibrator.predict([p_start])[0])
        if self.sixty_calibrator is not None:
            p_sixty = float(self.sixty_calibrator.predict([p_sixty])[0])
        # Guard against NaN before it reaches the built-in min/max below.
        # Python's min/max are not NaN-safe: max(nan, 0.0) returns nan
        # because every comparison against NaN is False, so a NaN raw or
        # calibrated probability would otherwise survive silently and ride
        # through `_apply_doubtful_floor` and `_apply_degenerate_floor`
        # downstream, where the same non-finite-comparisons-are-False trap
        # lets it slip past their floors too.
        p_play = float(np.nan_to_num(p_play, nan=0.0))
        p_start = float(np.nan_to_num(p_start, nan=0.0))
        p_sixty = float(np.nan_to_num(p_sixty, nan=0.0))
        p_play = float(np.clip(p_play, 0.0, 1.0))
        p_start = min(max(p_start, 0.0), p_play)
        p_sixty = min(max(p_sixty, 0.0), p_play)
        p60_start = min(p_start, p_sixty)
        early_starter_mass = max(p_start - p60_start, 0.0)
        long_sub_mass = max(p_sixty - p_start, 0.0)
        short_sub_mass = max(p_play - p_start - long_sub_mass, 0.0)
        exp_min = (p60_start * 80.0 + early_starter_mass * 52.0
                   + long_sub_mass * 72.0 + short_sub_mass * 25.0)
        drivers = {"n_train": self.n_train,
                   "dropped_features": list(self.dropped_features),
                   "calibrated": any(c is not None for c in (
                       self.play_calibrator, self.start_calibrator, self.sixty_calibrator))}
        drivers.update({k: round(v, 4) for k, v in self.holdout.items()})
        return MinutesPrediction(p_play, p_sixty, round(exp_min, 1), "gbm", drivers,
                                 p_start_est=p_start)