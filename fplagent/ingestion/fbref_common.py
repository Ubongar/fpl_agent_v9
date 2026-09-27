"""Shared low-level reader for FBref 'Standard Stats' style CSV exports.

Both the penalty-duty enrichment (`ingestion/fbref.py`) and the league
translation fitting (`ingestion/league_snapshot.py`) read the same export
format, so the messy part -- finding the real header row -- lives in one
place.

WHY A PREAMBLE SCAN, NOT A FIXED SKIPROWS
------------------------------------------
Real exports are inconsistent about what comes before the header: some start
straight with the multi-row column header, some carry a citation line and
blank rows first ("-- When using SR data, please cite us..."). A fixed
`skiprows=N` broke on the very first batch of real user-uploaded files tested
against it -- five of eight had a preamble the earlier version did not
expect. Scanning for the first line containing both "Player" and "Squad" is
robust to either shape and costs nothing on the common case.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def read_fbref_csv(path: str | Path) -> pd.DataFrame | None:
    """Parse an FBref export into a flat, de-duplicated-header DataFrame.

    Returns None (never raises) on anything that doesn't look like a
    Player/Squad table -- this is optional enrichment data and a malformed
    export must never take down the pipeline around it.

    Handles:
      * a citation/blank-line preamble before the real header
      * comma OR tab delimiting (browser/OS dependent on clipboard paste)
      * FBref's own multi-row column header (Playing Time / Performance /
        Per 90 Minutes group labels above the real column names)
      * repeated header rows scattered through the body (FBref repeats the
        header every N rows in a long export, same as its HTML table)
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:      # noqa: BLE001
        log.warning("could not read %s: %s", path, e)
        return None

    header_idx = None
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.replace("\t", ",").split(",")]
        if "Player" in cells and "Squad" in cells:
            header_idx = i
            break
    if header_idx is None:
        log.warning("%s: no line containing both 'Player' and 'Squad' found; "
                    "not an FBref Standard Stats export", path)
        return None

    text = "".join(lines[header_idx:])
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as e:      # noqa: BLE001
        log.warning("could not parse %s as CSV: %s", path, e)
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    df.columns = [str(c).strip() for c in df.columns]

    if "Player" not in df.columns:
        log.warning("%s: header row found but no 'Player' column after parsing", path)
        return None

    # Repeated header rows mid-export show up as a data row whose "Player"
    # value is literally the string "Player".
    df = df[df["Player"] != "Player"].copy()
    return df
