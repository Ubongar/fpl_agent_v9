"""Provider-agnostic LLM layer.

Any endpoint speaking the OpenAI chat-completions shape works: OpenAI,
Groq, Together, Fireworks, a local vLLM, Gemini through a compatibility
proxy. Switching provider is three .env lines, not a code change.

THE IMPORTANT DESIGN RULE: the model never computes anything. It is handed
finished arithmetic -- the xP decomposition, the attribution diff, the
simulation percentiles -- and asked to put it in English. An LLM asked to
reason about numbers it cannot see will produce fluent, confident, wrong
explanations, and a wrong explanation of a right number is worse than no
explanation at all, because it teaches you to trust the wrong thing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from fplagent.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

log = logging.getLogger(__name__)

# In-process cache for explain=true responses. Keyed on a hash of the exact
# prompt sent, not on any notion of a "data version" -- the underlying rows
# (predictions, evaluation_runs, ...) are written once per pipeline run and
# never mutated in place, so identical prompt content IS identical
# underlying data. TTL bounds staleness if the process stays up across a
# pipeline re-run without a restart. Per-process only: this does not share
# across workers, which is fine for the single-process deployment this
# system ships with (see RUN.md); a multi-worker deployment would want this
# backed by the database or Redis instead.
_CACHE_TTL_SECONDS = 15 * 60
_cache: dict[str, tuple[float, str]] = {}


def _cache_key(system: str, prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}|{system}|{prompt}".encode()).hexdigest()


def cache_stats() -> dict:
    now = time.time()
    live = sum(1 for ts, _ in _cache.values() if now - ts < _CACHE_TTL_SECONDS)
    return {"entries": len(_cache), "live": live, "ttl_seconds": _CACHE_TTL_SECONDS}

SYSTEM_PROMPT = """You are the explanation layer of a Fantasy Premier League \
analytics system. You are given finished model output: expected points, their \
component decomposition, simulation percentiles, and probabilities.

Rules you must follow:
- Explain ONLY the numbers you are given. Never compute, estimate or invent one.
- If a number looks surprising, say so plainly rather than rationalising it.
- Always state the main uncertainty. A recommendation with hidden risk is worse \
than no recommendation.
- Be concise and concrete. No hype, no filler, no "as an AI".
- The user makes the decision. You inform it."""


def _client():
    try:
        from openai import OpenAI
    except ImportError:
        return None
    if not LLM_API_KEY:
        return None
    kwargs: dict[str, Any] = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs)


def available() -> bool:
    return _client() is not None


def complete(prompt: str, system: str = SYSTEM_PROMPT,
             max_tokens: int = 700, temperature: float = 0.2) -> str:
    key = _cache_key(system, prompt, LLM_MODEL)
    hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL_SECONDS:
        return hit[1]
    client = _client()
    if client is None:
        return _fallback(prompt)
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        text = r.choices[0].message.content or ""
        _cache[key] = (time.time(), text)
        return text
    except Exception as e:      # noqa: BLE001
        log.warning("LLM call failed: %s", e)
        return _fallback(prompt, error=str(e))


def _fallback(prompt: str, error: str | None = None) -> str:
    """Deterministic explanation when no LLM is configured.

    Deliberately NOT a stub that raises. The system must remain fully usable
    with no API key -- the LLM is a presentation layer, and no analytical
    capability may depend on it.
    """
    head = "[no LLM configured -- showing the raw model output]"
    if error:
        head = f"[LLM unavailable: {error} -- showing the raw model output]"
    return f"{head}\n\n{prompt}"


def explain_prediction(player: dict, components: dict, features: dict,
                       percentiles: dict) -> str:
    payload = {
        "player": player, "expected_points_breakdown": components,
        "inputs": features, "simulation_percentiles": percentiles,
    }
    return complete(
        "Explain this player's projection for the coming gameweek in 3-4 "
        "sentences. Identify which component contributes most, and name the "
        "single biggest risk.\n\n" + json.dumps(payload, indent=2, default=str))


def explain_changes(changes: list[dict]) -> str:
    from fplagent.pipeline.attribution import narrate
    lines = [narrate(c) for c in changes[:12]]
    return complete(
        "These are the biggest week-over-week projection changes, already "
        "decomposed into drivers. Summarise what moved and why in a short "
        "briefing. Do not add causes that are not in the data.\n\n"
        + "\n".join(lines))


def explain_transfer_plan(plan: dict, horizon: list[int]) -> str:
    return complete(
        f"Explain this transfer plan over gameweeks {horizon}. Say plainly "
        f"whether any points hits are justified and what would make the plan "
        f"wrong.\n\n" + json.dumps(plan, indent=2, default=str))


def explain_captaincy(candidates: list[dict]) -> str:
    """candidates: the ranked output of optimizers.captaincy.rank_captains."""
    return complete(
        "These are ranked captaincy candidates (effective value already "
        "accounts for ownership). Explain the top choice, name the strongest "
        "differential option and why someone might take the risk, and state "
        "the single biggest thing that would flip the recommendation.\n\n"
        + json.dumps(candidates[:10], indent=2, default=str))


def explain_chip_schedule(schedule: dict) -> str:
    """schedule: the dict returned by optimizers.chips.schedule_chips
    (assignments, total_value, status, notes)."""
    return complete(
        "This is a chip schedule across the two half-season windows "
        "(use-it-or-lose-it at the GW19 boundary). Explain when each chip "
        "is assigned and why, and flag anything expiring unused.\n\n"
        + json.dumps(schedule, indent=2, default=str))


def explain_squad(solution: dict) -> str:
    """solution: {summary, starting_xi, bench, captain, vice_captain} from
    optimizers.squad.optimise_squad."""
    return complete(
        "This is an optimised 15-man squad and starting XI under budget and "
        "quota constraints. Explain the captain/vice choice, name the "
        "weakest bench player and why he was still worth his price, and "
        "state one trade-off a human might reasonably disagree with.\n\n"
        + json.dumps(solution, indent=2, default=str))


def explain_model_health(evaluation: dict) -> str:
    """evaluation: the dict returned by /evaluation/latest (mae, rmse,
    spearman, crps, pit_ks_stat, top5/top10 captain hit rate, baselines...)."""
    return complete(
        "This is the model's own backtest scorecard: rank correlation, "
        "captain hit rate, CRPS/PIT calibration diagnostics, and how it "
        "compares to simple baselines. Explain in plain language, for "
        "someone who is not a statistician, whether these numbers say the "
        "model is trustworthy right now, and name the single weakest metric "
        "honestly rather than only the strong ones.\n\n"
        + json.dumps(evaluation, indent=2, default=str))


def explain_data_quality(quality: dict) -> str:
    """quality: the dict returned by GET /quality (recent quality-gate
    events: name, severity, detail)."""
    return complete(
        "These are recent data-quality gate events from the ingestion "
        "pipeline (info/warn/fail severity). Summarise what's currently "
        "healthy vs. what needs attention, in plain language, and say "
        "whether any 'fail' severity event means today's predictions should "
        "be trusted less.\n\n" + json.dumps(quality, indent=2, default=str))


def explain_weekly_briefing(payload: dict) -> str:
    """payload: captaincy candidates, transfer plan, chip schedule status,
    and consistency flags, gathered together so the model can reason across
    pages instead of narrating each in isolation."""
    return complete(
        "You are given this gameweek's captaincy shortlist, transfer plan, "
        "chip schedule status, and any consistency flags raised on the "
        "squad's predictions -- all together, not one at a time. Write a "
        "short briefing (5-8 sentences) that connects them: e.g. whether the "
        "transfer plan and captaincy pick reinforce or conflict with each "
        "other, whether a flagged inconsistency should change the "
        "recommendation, and whether this is a week to play a chip. State "
        "the single biggest risk in the plan as a whole.\n\n"
        + json.dumps(payload, indent=2, default=str))
