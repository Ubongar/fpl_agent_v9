"""JSON that every database and every HTTP client accepts.

Python's json module writes NaN and Infinity as bare tokens. SQLite stores
them and Python reads them back, which is why this never showed up locally.
PostgreSQL's json type rejects them, and so does a browser's JSON.parse.

An evaluation window where one position group has too few rows for a rank
correlation produces a NaN. On Postgres that single NaN made the
`evaluation_runs` insert fail, and then the `pipeline_runs` insert -- whose
`stages` column carries the same dict -- failed too, so `fplagent run`
crashed at its very last step after doing all of the work.

`json_dumps` is installed as the engine's JSON serializer (see
db/session.py), so every JSON column goes through it, and the API uses
`to_json_safe` on computed responses for the same reason.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from decimal import Decimal
from typing import Any

try:                                    # both are hard dependencies, but keep
    import numpy as _np                 # this module importable without them
except ImportError:                     # pragma: no cover
    _np = None
try:
    import pandas as _pd
except ImportError:                     # pragma: no cover
    _pd = None


def _safe_key(k: Any) -> Any:
    if k is None or isinstance(k, (str, bool, int)):
        return k
    if isinstance(k, float):
        return k if math.isfinite(k) else str(k)
    if _np is not None and isinstance(k, _np.generic):
        return _safe_key(k.item())
    return str(k)


def to_json_safe(obj: Any) -> Any:
    """Recursively convert to plain JSON types; non-finite floats become None."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {_safe_key(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_json_safe(v) for v in obj]
    if _np is not None:
        if isinstance(obj, _np.generic):
            return to_json_safe(obj.item())
        if isinstance(obj, _np.ndarray):
            return [to_json_safe(v) for v in obj.tolist()]
    if _pd is not None:
        if obj is _pd.NaT or obj is _pd.NA:
            return None
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return to_json_safe(float(obj))
    return obj


def json_dumps(obj: Any) -> str:
    return json.dumps(to_json_safe(obj), default=str)
