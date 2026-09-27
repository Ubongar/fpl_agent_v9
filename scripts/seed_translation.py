"""Load the seed league-translation coefficients into the database.

These are published estimates, NOT values fitted from your own data. Each row
carries a confidence, and every downstream use of a coefficient weights by it.
Run this once after init-db so `league_translation` is populated; refit the
table properly once multi-season transfer history has been backfilled.

    python scripts/seed_translation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplagent.db import tables as T
from fplagent.db.repository import upsert
from fplagent.db.session import init_db
from fplagent.models.cold_start import SEED_TRANSLATION


def main() -> None:
    init_db()
    rows = [{
        "source_league": league, "position_group": group, "metric": metric,
        "ratio": ratio, "confidence": conf, "sample_size": n, "source": "seed",
    } for (league, group, metric), (ratio, conf, n) in SEED_TRANSLATION.items()]
    n = upsert(T.league_translation, rows,
               ["source_league", "position_group", "metric"])
    print(f"Seeded {n} translation coefficients.")
    print("These are estimates, not fits. Treat new-signing projections as the "
          "weakest numbers the system produces until you refit them.")


if __name__ == "__main__":
    main()
