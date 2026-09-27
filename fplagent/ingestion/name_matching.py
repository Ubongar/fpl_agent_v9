"""Resolving external names to FPL's own ids: clubs and players.

WHY THIS IS ONE MODULE
----------------------
Four ingestors map names from outside sources onto FPL ids -- The Odds API,
football-data.org, FBref exports and RotoWire -- and each carried its own
copy of `process.extractOne(name, fpl_names, scorer=token_sort_ratio)` with a
threshold of 75-80. That scorer cannot bridge the way FPL abbreviates clubs:
"Manchester City" against FPL's "Man City" scores 70, "West Ham United"
against "West Ham" 70, "Tottenham Hotspur" against "Spurs" nowhere near. So
eight of twenty Premier League clubs silently failed to match: their
fixtures got no market prior, no Champions League congestion and no FBref
penalty duty, and nothing complained, because "no match" was treated as the
normal case for a club we do not track.

Players had the same problem in a different shape. FBref and RotoWire
publish full names ("Erling Haaland") while FPL's `web_name` is usually the
surname ("Haaland"), and those two score 67.

The rules here, in order: exact match on a normalised name, then a
maintained alias table, then fuzzy matching with a scorer suited to the
comparison. Never a guess: an ambiguous or low-scoring match returns None so
the caller skips the row rather than attaching data to the wrong club or the
wrong player.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Tokens that carry no identity ("Arsenal FC" is "Arsenal").
_NOISE_TOKENS = {"fc", "afc", "cf", "sc", "the"}

# Letters NFKD does not decompose into base + accent, so the ASCII step below
# would DELETE them ("Ødegaard" -> "degaard", "Højlund" -> "hjlund") while an
# external source spelling "Odegaard" keeps the O. Transliterate first.
_TRANSLIT = str.maketrans({
    "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ß": "ss", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "TH", "ı": "i",
})

# Normalised spelling used elsewhere -> normalised FPL `teams.name`. Extend
# this rather than lowering a threshold: an alias is exact, a lower
# threshold is a false match waiting to happen.
TEAM_ALIASES: dict[str, str] = {
    "manchester city": "man city",
    "manchester united": "man utd",
    "manchester utd": "man utd",
    "man united": "man utd",
    "tottenham hotspur": "spurs",
    "tottenham": "spurs",
    "newcastle united": "newcastle",
    "newcastle utd": "newcastle",
    "nottingham forest": "nott'm forest",
    "nott'ham forest": "nott'm forest",
    "nottm forest": "nott'm forest",
    "wolverhampton wanderers": "wolves",
    "wolverhampton": "wolves",
    "brighton and hove albion": "brighton",
    "brighton hove albion": "brighton",
    "brighton and hove": "brighton",
    "west ham united": "west ham",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "ipswich town": "ipswich",
    "sheffield united": "sheffield utd",
    "luton town": "luton",
    "west bromwich albion": "west brom",
    "norwich city": "norwich",
    "cardiff city": "cardiff",
    "swansea city": "swansea",
    "hull city": "hull",
    "stoke city": "stoke",
    "huddersfield town": "huddersfield",
    "queens park rangers": "qpr",
    "coventry city": "coventry",
    "birmingham city": "birmingham",
}


def normalise_name(value: object) -> str:
    """Lower-case, accent-free, punctuation-free, noise tokens dropped.

    Apostrophes survive because FPL's own "Nott'm Forest" contains one.
    """
    if value is None:
        return ""
    s = str(value).replace("’", "'").replace("&", " and ").translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9' ]+", " ", s.lower())
    tokens = [t for t in s.split() if t not in _NOISE_TOKENS]
    return " ".join(tokens)


def resolve_team_id(name: object, teams: pd.DataFrame,
                    threshold: float = 85.0) -> int | None:
    """FPL team_id for a club name or three-letter code, or None.

    `teams` needs `team_id`, `name` and `short_name` columns (the `teams`
    table). The fuzzy step only runs when exact and alias lookup fail, and
    its default threshold is deliberately high: a Champions League feed is
    mostly clubs we do not track, and a false match would book a European
    tie against the wrong Premier League side.
    """
    key = normalise_name(name)
    if not key or teams is None or teams.empty:
        return None

    by_key: dict[str, int] = {}
    for r in teams.itertuples():
        for col in ("name", "short_name"):
            v = getattr(r, col, None)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                by_key.setdefault(normalise_name(v), int(r.team_id))
    if key in by_key:
        return by_key[key]
    alias = TEAM_ALIASES.get(key)
    if alias is not None and alias in by_key:
        return by_key[alias]

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return None
    names = {normalise_name(r.name): int(r.team_id)
             for r in teams.itertuples() if getattr(r, "name", None)}
    if not names:
        return None
    hit = process.extractOne(key, list(names), scorer=fuzz.token_sort_ratio)
    if hit and hit[1] >= threshold:
        return names[hit[0]]
    return None


def _full_name(row) -> str:
    first = getattr(row, "first_name", None) or ""
    second = getattr(row, "second_name", None) or ""
    if isinstance(first, float):
        first = ""
    if isinstance(second, float):
        second = ""
    return normalise_name(f"{first} {second}")


def resolve_player_id(name: object, players: pd.DataFrame,
                      team_id: int | None = None,
                      threshold: float = 85.0) -> int | None:
    """FPL player_id for a person's name, or None.

    `players` needs `player_id`, `web_name`, `team_id` and ideally
    `first_name`/`second_name` (the `players` table).

    Scoped to one club whenever the caller knows it. Inside a club, a
    surname-only `web_name` ("Haaland") is accepted as a match for a full
    name ("Erling Haaland") because two squad-mates sharing a surname is rare
    and is caught by the tie check. Without a club, only the full name is
    compared, and the caller should pass a stricter threshold.
    """
    key = normalise_name(name)
    if not key or players is None or players.empty:
        return None
    pool = players if team_id is None else players[players["team_id"] == team_id]
    if pool.empty:
        return None

    exact: set[int] = set()
    for r in pool.itertuples():
        web = normalise_name(getattr(r, "web_name", ""))
        if key == _full_name(r) or (team_id is not None and key == web):
            exact.add(int(r.player_id))
    if len(exact) == 1:
        return exact.pop()
    if len(exact) > 1:
        return None           # two people with the same name: do not guess

    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None
    scored: list[tuple[float, int]] = []
    for r in pool.itertuples():
        full = _full_name(r)
        web = normalise_name(getattr(r, "web_name", ""))
        s_full = fuzz.token_sort_ratio(key, full) if full else 0.0
        s_web = (fuzz.token_set_ratio(key, web)
                 if (web and team_id is not None) else 0.0)
        scored.append((max(s_full, s_web), int(r.player_id)))
    scored.sort(reverse=True)
    best, pid = scored[0]
    if best < threshold:
        return None
    if len(scored) > 1 and scored[1][0] >= best and scored[1][1] != pid:
        return None           # tie at the top: ambiguous
    return pid
