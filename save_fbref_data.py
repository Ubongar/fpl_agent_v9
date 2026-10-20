import soccerdata as sd
import pandas as pd
from pathlib import Path

# Map the exact filenames you already have to the soccerdata league names
LEAGUES = {
    "premier_league": "ENG-Premier League",
    "la_liga": "ESP-La Liga",
    "serie_a": "ITA-Serie A",
    "bundesliga": "GER-Bundesliga",
    "ligue_1": "FRA-Ligue 1",
    "eredivisie": "NED-Eredivisie",
    "primeira_liga": "POR-Primeira Liga",
    "championship": "ENG-Championship"
}

# The model uses these for cold-start players (foreign leagues)
SEASONS = ["2024-2025"] 

data_dir = Path("data/fbref_snapshots")
data_dir.mkdir(parents=True, exist_ok=True)

for file_name, league_name in LEAGUES.items():
    for season in SEASONS:
        print(f"Fetching {league_name} {season}...")
        try:
            # Initialize the scraper
            fbref = sd.FBref(leagues=[league_name], seasons=season)
            
            # Get the player-season standard stats (includes penalties, xG, xA, etc.)
            player_stats = fbref.read_player_season_stats(stat_type="standard")
            
            # Save to an Excel file named exactly like your existing files
            output_path = data_dir / f"{file_name}.xlsx"
            player_stats.to_excel(output_path)
            print(f"  [OK] Saved {output_path}")
            
        except Exception as e:
            print(f"  [ERROR] Failed for {league_name} {season}: {e}")