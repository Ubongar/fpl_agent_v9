"""Joint lineup sampling: exactly eleven start, and substitutes are correlated.

THE BUG THIS FIXES
------------------
v6 drew every player's minutes independently:

    minutes = np.vstack([sample_minutes(p.minutes, n_sims, rng) for p in squad])

Nothing anywhere constrained the result to a legal team. Measured on a
realistic 16-man pool (11 nailed starters at p_play 0.97, 5 fringe at 0.55),
the simulator put a mean of 13.4 players on the pitch, a maximum of 16, and
had more than 14 on in 19.4% of draws.

THE MODEL
---------
Per simulation draw:

  1. Pick the goalkeeper: exactly one, weighted by P(start).
  2. Pick ten outfield players without replacement, weighted by P(start),
     subject to formation legality (3-5 DEF, 2-5 MID, 1-3 FWD).
  3. Assign starters a minutes draw: mostly 90, with a hazard of being
     withdrawn that rises when P(start) exceeds P(60+).
  4. Make up to three outfield substitutions from the remaining squad.

THE DEGENERATE-INPUT SAFETY NET
-------------------------------
If a whole position group arrives with p_start = 0 -- which happens on real
data during injury crises, and in the demo when the learned minutes model is
used on doubtful players -- the formation check has nothing positive to work
with and the sampler raises. Fielding a legal XI is a HARD constraint, so the
correct response is not to crash: it is to elevate enough players to cover
one legal formation and field a team.

THE INVARIANT THAT MUST HOLD
----------------------------
Not "each position has at least XI_MIN positive-weight players" -- that is
weaker than it sounds. A squad with 3 positive-weight DEF, 3 MID and 1 FWD
satisfies every per-position minimum, and yet NO legal outfield split is
covered by it: (3,4,3) needs 4 MID, (4,4,2) needs 4 DEF, (4,5,1) needs 4
DEF, (5,4,1) needs 5 DEF, and so on. Every legal formation requires at
least one position with four or more positive-weight players.

The real invariant is: AT LEAST ONE FORMATION in `FORMATIONS` must be fully
covered by positive-weight players. `_apply_degenerate_floor` checks that
directly, and if no formation is covered it elevates the shortfall for
whichever formation is closest to being feasible. This is idempotent (if
some formation is already covered, it returns unchanged) and safe to call
after every calibration iteration.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
N_OUTFIELD = 10
MAX_SUBS = 3

STARTER_FULL_MEAN = 89.0
STARTER_FULL_SD = 3.0
WITHDRAWN_MIN = 46
WITHDRAWN_MAX = 88
SUB_MIN = 4
SUB_MAX = 45

# Tiny positive weight used by the degenerate-input safety net. Small enough
# that the resulting distribution is still dominated by whatever real signal
# remains elsewhere in the squad, large enough to clear the `> 1e-12`
# positive-weight check in `_formation_weights` and `_gumbel_topk`, and
# robust to being multiplied down by the calibration loop's 0.3 clip floor
# eight times (1e-6 * 0.3^8 ~= 6.5e-11, still comfortably above 1e-12).
_DEGENERATE_FLOOR = 1e-6


def _gumbel_topk(weights: np.ndarray, k: int, n_sims: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Sample k distinct indices per draw, without replacement, proportional
    to `weights`. Returns (k, n_sims) integer indices."""
    w = np.asarray(weights, dtype=float)
    n = len(w)
    positive = w > 1e-12
    if positive.sum() < k:
        raise ValueError(f"not enough positive-weight players to select {k}")
    k = min(k, n)
    logw = np.full(n, -np.inf, dtype=float)
    logw[positive] = np.log(w[positive])
    g = rng.gumbel(size=(n, n_sims))
    keys = logw[:, None] + g
    idx = np.argpartition(-keys, kth=k - 1, axis=0)[:k]
    return idx


def _withdrawal_prob(p_start: np.ndarray, p_sixty: np.ndarray) -> np.ndarray:
    """Approximate P(starter is withdrawn before 60 | starter)."""
    p_start = np.asarray(p_start, dtype=float)
    p_sixty = np.asarray(p_sixty, dtype=float)
    return np.clip((p_start - p_sixty) / np.maximum(p_start, 1e-9), 0.0, 1.0)


def legal_formations() -> list[tuple[int, int, int]]:
    """Every legal outfield split (DEF, MID, FWD) summing to ten."""
    out = []
    for d in range(XI_MIN["DEF"], XI_MAX["DEF"] + 1):
        for m in range(XI_MIN["MID"], XI_MAX["MID"] + 1):
            f = N_OUTFIELD - d - m
            if XI_MIN["FWD"] <= f <= XI_MAX["FWD"]:
                out.append((d, m, f))
    return out


FORMATIONS = legal_formations()


def _formation_weights(pos: np.ndarray, start_w: np.ndarray) -> np.ndarray:
    """Score each legal formation by the strength of the XI it would field."""
    sorted_w = {}
    for p in ("DEF", "MID", "FWD"):
        w = np.sort(start_w[pos == p])[::-1]
        sorted_w[p] = np.concatenate([w, np.zeros(XI_MAX[p] + 1)])

    scores = []
    for (d, m, f) in FORMATIONS:
        counts = {"DEF": d, "MID": m, "FWD": f}
        feasible = all(np.sum((pos == p) & (start_w > 1e-12)) >= counts[p] for p in counts)
        if not feasible:
            scores.append(0.0)
            continue
        scores.append(float(sum(sorted_w[p][:counts[p]].sum() for p in counts)))

    scores = np.asarray(scores, dtype=float)
    if scores.sum() <= 0:
        raise ValueError(
            "squad has no feasible FPL starting formation with positive "
            "P(start) weights; check GK/DEF/MID/FWD positions and availability"
        )
    scores = scores ** 6
    return scores / scores.sum()


def _apply_degenerate_floor(pos: np.ndarray, start_w: np.ndarray) -> np.ndarray:
    """Guarantee that at least one legal formation is covered by positive
    weights, elevating the shortfall for the closest formation if not.

    The invariant this enforces is NOT "each position has at least XI_MIN
    positive-weight players". That is weaker than it looks: a squad with 3
    DEF, 3 MID and 1 FWD satisfies every per-position minimum, and yet no
    legal outfield split is covered by it (every formation in FORMATIONS
    needs at least one position with four or more). The invariant that
    actually matters is "some formation in FORMATIONS is feasible", and this
    function checks it directly.

    Idempotent: if any formation is already covered, every branch below is a
    no-op and the input is returned unchanged. That is what makes it safe to
    call once at entry and again after every calibration iteration.

    NaN guard: np.maximum(nan, _DEGENERATE_FLOOR) returns nan, not the
    floor, so an un-sanitised NaN weight would survive "elevation" here
    unchanged and go on to break the formation check downstream. Treat a NaN
    weight as zero -- it is exactly the "unknown, no signal" case this floor
    exists to handle.
    """
    w = np.nan_to_num(np.asarray(start_w, dtype=float), nan=0.0)

    # Positive-weight count per position. GK included: the goalkeeper slot is
    # chosen separately in `_draw_starters` but is just as mandatory.
    positive_count: dict[str, int] = {}
    for p in ("GK", "DEF", "MID", "FWD"):
        mask = pos == p
        positive_count[p] = int((w[mask] > 1e-12).sum()) if mask.any() else 0

    # Fast path: some formation is already fully covered, and at least one
    # goalkeeper is available. Nothing to do.
    gk_ok = positive_count["GK"] >= XI_MIN["GK"]
    if gk_ok and any(
        positive_count["DEF"] >= d and positive_count["MID"] >= m
        and positive_count["FWD"] >= f
        for (d, m, f) in FORMATIONS
    ):
        return w

    # No formation is feasible. Choose the formation with the smallest total
    # shortfall across positions and elevate exactly that shortfall. This is
    # deterministic and gives the sampler the best available legal shape
    # rather than an arbitrary one.
    best_formation = None
    best_shortfall = None
    for (d, m, f) in FORMATIONS:
        needs = {"DEF": d, "MID": m, "FWD": f}
        shortfall = sum(max(0, needs[p] - positive_count[p]) for p in needs)
        if best_shortfall is None or shortfall < best_shortfall:
            best_formation = (d, m, f)
            best_shortfall = shortfall

    # GK shortfall is handled separately because no formation in FORMATIONS
    # names a goalkeeper -- it is outfield-only by construction.
    needs: dict[str, int] = {
        "DEF": best_formation[0],
        "MID": best_formation[1],
        "FWD": best_formation[2],
    }
    if not gk_ok:
        needs["GK"] = XI_MIN["GK"]

    for pos_name, need in needs.items():
        mask = pos == pos_name
        if not mask.any():
            continue
        pos_w = w[mask]
        have = int((pos_w > 1e-12).sum())
        if have >= need:
            continue
        order = np.argsort(-pos_w)[:need]
        idxs = np.flatnonzero(mask)[order]
        w[idxs] = np.maximum(w[idxs], _DEGENERATE_FLOOR)
        log.warning(
            "sample_lineup: only %d %s player(s) had positive P(start), "
            "needed %d for the closest legal formation %d-%d-%d. Elevated "
            "%d low-weight player(s) so a legal XI could be built. Check the "
            "minutes model's output for doubtful and returning players -- a "
            "position group at p_start = 0 means the model was either "
            "undertrained in that region or is contradicting FPL's published "
            "chance.",
            have, pos_name, need, best_formation[0], best_formation[1],
            best_formation[2], need - have,
        )
    return w


def _calibrate_start_weights(pos: np.ndarray, start_w: np.ndarray,
                             target: np.ndarray, rng: np.random.Generator,
                             n_cal: int = 2000, iters: int = 8) -> np.ndarray:
    """Adjust selection weights so realised starting rates track P(start).

    Each iteration multiplies weights by a clipped ratio, which can shrink a
    small positive weight below the 1e-12 positivity threshold that
    `_formation_weights` and `_gumbel_topk` use to decide whether a player is
    selectable at all. A position group that starts the loop with just enough
    positive-weight players can therefore end an iteration with too few, and
    the next call to `_draw_starters` raises from `_formation_weights` even
    though its input was legal one iteration earlier.

    The fix is to re-apply `_apply_degenerate_floor` after every iteration.
    It is idempotent and cheap, and it preserves the hard constraint that at
    least one legal formation must always be buildable.
    """
    w = np.asarray(start_w, dtype=float).copy()
    active = target > 1e-9
    if not active.any():
        return w
    for _ in range(iters):
        started = _draw_starters(pos, w, n_cal, rng)
        realised = started.mean(axis=1)
        ratio = np.where(active, target / np.maximum(realised, 1e-3), 1.0)
        w = np.clip(w * np.clip(ratio, 0.3, 3.0), 0.0, 1e9)
        # Re-floor after every iteration. See the docstring for why this is
        # necessary and not merely defensive.
        w = _apply_degenerate_floor(pos, w)
    return w


def _draw_starters(pos: np.ndarray, start_w: np.ndarray, n_sims: int,
                   rng: np.random.Generator) -> np.ndarray:
    """One goalkeeper plus a legal outfield formation. Returns (n, n_sims)."""
    n = len(pos)
    started = np.zeros((n, n_sims), dtype=bool)
    is_gk = pos == "GK"

    gk_idx = np.flatnonzero(is_gk)
    if gk_idx.size:
        chosen = _gumbel_topk(start_w[gk_idx], 1, n_sims, rng)[0]
        started[gk_idx[chosen], np.arange(n_sims)] = True

    fw = _formation_weights(pos, start_w)
    form_choice = rng.choice(len(FORMATIONS), size=n_sims, p=fw)
    for p in ("DEF", "MID", "FWD"):
        idx = np.flatnonzero(pos == p)
        if idx.size == 0:
            continue
        slot = {"DEF": 0, "MID": 1, "FWD": 2}[p]
        needed = np.minimum(np.array([FORMATIONS[c][slot] for c in form_choice]),
                            idx.size)
        kmax = int(needed.max()) if needed.size else 0
        if kmax == 0:
            continue
        pick = _gumbel_topk(start_w[idx], kmax, n_sims, rng)
        take = np.arange(kmax)[:, None] < needed[None, :]
        rows = idx[pick]
        cols = np.broadcast_to(np.arange(n_sims), rows.shape)
        started[rows[take], cols[take]] = True
    return started


def sample_lineup(positions: list[str], p_play: np.ndarray, p_start: np.ndarray,
                  p_sixty: np.ndarray,
                  n_sims: int, rng: np.random.Generator,
                  unavailable: np.ndarray | None = None,
                  calibrate: bool = True,
                  return_context: bool = False):
    """Draw realised minutes for a whole squad jointly.

    Returns (n_players, n_sims) integer minutes in which exactly eleven
    players start in a legal formation and up to three of the simulated
    outfield bench players appear. Every simulated starter withdrawal is
    paired with replacement coverage inside that pool.
    """
    positions = list(positions)
    n = len(positions)
    if n < 11:
        raise ValueError(f"lineup sampling requires at least 11 players, got {n}")
    counts = {p: positions.count(p) for p in ("GK", "DEF", "MID", "FWD")}
    for p, minimum in XI_MIN.items():
        if counts[p] < minimum:
            raise ValueError(
                f"lineup sampling needs at least {minimum} {p} player(s), got {counts[p]}"
            )
    # NaN guard: np.clip does not remove NaN (np.clip(nan, 0, 1) is still
    # nan), so a NaN reaching here from upstream would flow straight into
    # `_apply_degenerate_floor`, `_withdrawal_prob`, and every other
    # `np.maximum`/`np.minimum` call below, all of which treat NaN as
    # neither elevated nor floored. Zero it out at the entry point, once,
    # before anything downstream has a chance to silently no-op on it.
    p_play = np.nan_to_num(np.asarray(p_play, dtype=float), nan=0.0)
    p_start = np.nan_to_num(np.asarray(p_start, dtype=float), nan=0.0)
    p_sixty = np.nan_to_num(np.asarray(p_sixty, dtype=float), nan=0.0)
    p_play = np.clip(p_play, 0.0, 1.0)
    p_start = np.clip(p_start, 0.0, 1.0)
    p_start = np.minimum(p_start, p_play)
    p_sixty = np.clip(p_sixty, 0.0, 1.0)
    p_sixty = np.minimum(p_sixty, p_play)
    if n == 0:
        return np.zeros((0, n_sims), dtype=np.int16)

    avail = np.ones(n, dtype=float)
    if unavailable is not None:
        avail = np.where(np.asarray(unavailable, dtype=bool), 0.0, 1.0)
    avail = avail * (p_play > 1e-9)

    pos = np.array(positions)
    is_gk = pos == "GK"
    start_w = p_start * avail

    # Defensive safety net. Guarantee that at least one legal formation is
    # covered by positive-weight players, before any calibration starts
    # mutating weights. See _apply_degenerate_floor for the exact invariant.
    start_w = _apply_degenerate_floor(pos, start_w)

    hooked_p = _withdrawal_prob(p_start, p_sixty)
    hooked_p = np.where(is_gk, 0.0, hooked_p)

    if calibrate:
        start_w = _calibrate_start_weights(pos, start_w, p_start, rng)

    started = _draw_starters(pos, start_w, n_sims, rng)

    hooked = (rng.random((n, n_sims)) < hooked_p[:, None]) & started
    if MAX_SUBS > 0:
        for col in range(n_sims):
            rows = np.flatnonzero(hooked[:, col])
            if rows.size > MAX_SUBS:
                key = (np.log(np.maximum(hooked_p[rows], 1e-12))
                       + rng.gumbel(size=rows.size))
                keep = rows[np.argpartition(-key, MAX_SUBS - 1)[:MAX_SUBS]]
                hooked[rows, col] = False
                hooked[keep, col] = True
    full = np.clip(rng.normal(STARTER_FULL_MEAN, STARTER_FULL_SD, size=(n, n_sims)),
                   60, 90)
    full = np.where(is_gk[:, None], 90, full)
    early = rng.integers(WITHDRAWN_MIN, WITHDRAWN_MAX + 1, size=(n, n_sims))
    minutes = np.where(started, np.where(hooked, early, full), 0).astype(np.int16)

    start_rate = started.mean(axis=1)
    bench_pref = np.maximum(p_play - p_start, 0.0) * avail
    bench_pref = np.where(is_gk, 0.0, bench_pref)

    def _draw_subs(q: np.ndarray, min_required: np.ndarray | None = None) -> np.ndarray:
        wants_on = (rng.random((n, n_sims)) < q[:, None]) & (~started)
        keys = (np.log(np.maximum(bench_pref, 1e-12))[:, None]
                + rng.gumbel(size=(n, n_sims)))
        keys = np.where(wants_on, keys, -np.inf)
        ranked = np.argsort(-keys, axis=0)
        rank_of = np.empty_like(ranked)
        np.put_along_axis(rank_of, ranked, np.arange(n)[:, None], axis=0)
        result = wants_on & (rank_of < MAX_SUBS)

        if min_required is not None:
            for col in range(n_sims):
                required = int(np.clip(min_required[col], 0, MAX_SUBS))
                have = int(result[:, col].sum())
                if have >= required:
                    continue
                candidates = np.flatnonzero((~started[:, col]) & (~result[:, col]) & (~is_gk))
                if candidates.size == 0:
                    continue
                need = min(required - have, MAX_SUBS - have, candidates.size)
                keys_col = np.log(np.maximum(bench_pref[candidates], 1e-12)) + rng.gumbel(size=candidates.size)
                chosen = candidates[np.argpartition(-keys_col, need - 1)[:need]]
                result[chosen, col] = True
        result[is_gk] = False
        return result

    required_subs = hooked.sum(axis=0)
    q = np.clip((p_play - start_rate) / np.maximum(1.0 - start_rate, 1e-6), 0.0, 1.0)
    comes_on = _draw_subs(q, min_required=required_subs)

    if calibrate:
        realised = (started | comes_on).mean(axis=1)
        deficit = p_play - realised
        bump = np.clip(deficit / np.maximum(1.0 - start_rate, 1e-6), -1.0, 1.0)
        comes_on = _draw_subs(np.clip(q + bump, 0.0, 1.0), min_required=required_subs)

    for col in range(n_sims):
        required = int(np.clip(required_subs[col], 0, MAX_SUBS))
        have = int(comes_on[:, col].sum())
        if have >= required:
            continue
        candidates = np.flatnonzero((~started[:, col]) & (~comes_on[:, col]) & (~is_gk))
        if candidates.size == 0:
            raise RuntimeError("lineup invariant violated: no bench player available to replace a withdrawn starter")
        need = min(required - have, MAX_SUBS - have, candidates.size)
        if need <= 0:
            raise RuntimeError("lineup invariant violated: substitution capacity exhausted")
        keys_col = np.log(np.maximum(bench_pref[candidates], 1e-12)) + rng.gumbel(size=candidates.size)
        chosen = candidates[np.argpartition(-keys_col, need - 1)[:need]]
        comes_on[chosen, col] = True

    on_now = comes_on & (~started)
    sub_rate_target = np.maximum(p_play - p_start, 0.0)
    start_long_mass = np.minimum(p_sixty, p_start)
    long_sub_prob = np.divide(
        np.maximum(p_sixty - start_long_mass, 0.0),
        np.maximum(sub_rate_target, 1e-9),
        out=np.zeros(n, dtype=float), where=sub_rate_target > 1e-9)
    long_sub_prob = np.clip(long_sub_prob, 0.0, 1.0)
    sub_long = rng.random((n, n_sims)) < long_sub_prob[:, None]
    sub_short_minutes = rng.integers(SUB_MIN, SUB_MAX + 1, size=(n, n_sims))
    sub_long_minutes = np.clip(rng.normal(72, 9, size=(n, n_sims)), 60, 90).astype(int)
    sub_minutes = np.where(sub_long, sub_long_minutes, sub_short_minutes)
    minutes = np.where(on_now, sub_minutes, minutes).astype(np.int16)

    gk_idx = np.flatnonzero(is_gk)
    if gk_idx.size > 1:
        gk_on = on_now[gk_idx].any(axis=0)
        if gk_on.any():
            replacement_mins = (on_now[gk_idx] * minutes[gk_idx]).max(axis=0)
            for g in gk_idx:
                was_starter = started[g] & gk_on
                minutes[g] = np.where(
                    was_starter,
                    np.clip(90 - replacement_mins, 1, 90),
                    minutes[g]).astype(np.int16)
    if return_context:
        return minutes, started, on_now
    return minutes


def lineup_consistency(positions: list[str], p_start: np.ndarray) -> dict:
    """How far the per-player minutes model is from fielding a legal team."""
    pos = np.array(list(positions))
    p_start = np.asarray(p_start, dtype=float)
    gk = float(p_start[pos == "GK"].sum())
    outfield = float(p_start[pos != "GK"].sum())
    return {
        "implied_starting_gk": round(gk, 2),
        "implied_starting_outfield": round(outfield, 2),
        "required_outfield": N_OUTFIELD,
        "excess": round(outfield - N_OUTFIELD, 2),
        "note": ("per-player P(start) cannot be honoured alongside the "
                 "eleven-player constraint; the sampler prioritises the "
                 "constraint and spreads the difference proportionally"
                 if abs(outfield - N_OUTFIELD) > 1.0 else "consistent"),
    }