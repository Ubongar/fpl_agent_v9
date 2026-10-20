import logging
from fplagent.ingestion.pipeline import ingest_bootstrap
from fplagent.config import CURRENT_SEASON

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Fetching latest player news and prices from FPL...")
    try:
        # This grabs the latest players table (prices, news, injuries)
        ingest_bootstrap(season=CURRENT_SEASON)
        print("Successfully updated player news and prices.")
    except Exception as e:
        print(f"Failed to update news: {e}")