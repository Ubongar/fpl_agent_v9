import json
import logging
import requests
from pathlib import Path
from fplagent.config import (
    OPENFOOT_API_KEY, BZZOIRO_API_KEY, FOOTBALLDATA_API_KEY,
    STRIKER_API_KEY, SPORTMONKS_API_KEY
)

log = logging.getLogger(__name__)
CACHE_FILE = Path("data/external_metrics.json")

def fetch_external_metrics():
    """Fetches Elo, advanced stats, and match predictions from free-tier APIs."""
    if not any([OPENFOOT_API_KEY, BZZOIRO_API_KEY, FOOTBALLDATA_API_KEY, STRIKER_API_KEY, SPORTMONKS_API_KEY]):
        log.warning("No external API keys found. Skipping external metrics.")
        return

    metrics = {
        "elo": {},
        "player_per90": {},
        "team_set_pieces": {},
        "league_xg": {},
        "match_predictions": {} # New: Store Bzzoiro predictions
    }

    # --- 1. OpenFootAPI (Fixtures & Search - Free Starter tier) ---
    if OPENFOOT_API_KEY and OPENFOOT_API_KEY != "your_openfoot_key_here":
        try:
            headers = {"Authorization": f"Bearer {OPENFOOT_API_KEY}", "Accept": "application/json"}
            resp = requests.get("https://openfootapi.com/v1/matches?date=2026-10-20", headers=headers)
            if resp.status_code == 200:
                metrics["league_xg"]["openfoot_matches"] = resp.json().get("data", [])
        except Exception as e:
            log.error(f"OpenFootAPI fetch failed: {e}")

    # --- 2. Bzzoiro Sports Data (Free tier includes xG, shotmaps, AND predictions) ---
    if BZZOIRO_API_KEY:
        try:
            headers = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
            # Fetch team stats with xG
            teams_resp = requests.get("https://sports.bzzoiro.com/api/v2/teams/?league=premier-league", headers=headers)
            if teams_resp.status_code == 200:
                for team in teams_resp.json().get("results", []):
                    team_name = team["name"].lower()
                    metrics["team_set_pieces"][team_name] = {
                        "xg": team.get("stats", {}).get("xg", 0.0),
                        "xga": team.get("stats", {}).get("xga", 0.0)
                    }
            
            # Fetch match predictions (Free tier)
            # The API accepts filters like league_id and upcoming.
            # For simplicity, we fetch recent/upcoming predictions.
            preds_resp = requests.get("https://sports.bzzoiro.com/api/v2/predictions/?league=premier-league&upcoming=true", headers=headers)
            if preds_resp.status_code == 200:
                for pred in preds_resp.json().get("results", []):
                    event = pred.get("event", {})
                    event_id = event.get("id")
                    if not event_id:
                        continue
                    
                    markets = pred.get("markets", {})
                    metrics["match_predictions"][str(event_id)] = {
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "prob_home": markets.get("match_result", {}).get("prob_home"),
                        "prob_draw": markets.get("match_result", {}).get("prob_draw"),
                        "prob_away": markets.get("match_result", {}).get("prob_away"),
                        "expected_goals_home": markets.get("expected_goals", {}).get("home"),
                        "expected_goals_away": markets.get("expected_goals", {}).get("away"),
                        "prob_over_25": markets.get("over_under", {}).get("prob_over_25"),
                        "prob_btts": markets.get("btts", {}).get("prob_yes"),
                        "most_likely_score": markets.get("score", {}).get("most_likely"),
                        "confidence": pred.get("model", {}).get("confidence")
                    }
        except Exception as e:
            log.error(f"Bzzoiro API fetch failed: {e}")

    # --- 3. Footballdata.io (Free tier includes xG, 2,000 req/month) ---
    if FOOTBALLDATA_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {FOOTBALLDATA_API_KEY}"}
            # Premier League is league_id=15. Fetch match stats with xG.
            stats_resp = requests.get("https://footballdata.io/api/v1/leagues/15/stats", headers=headers)
            if stats_resp.status_code == 200:
                metrics["league_xg"]["footballdata_premier_league"] = stats_resp.json().get("data", {})
        except Exception as e:
            log.error(f"Footballdata.io fetch failed: {e}")

    # --- 4. StrikerAPI (Elo Ratings & Player Per-90) [Free: 500 credits/month] ---
    if STRIKER_API_KEY:
        try:
            headers = {"x-api-key": STRIKER_API_KEY}
            elo_resp = requests.post("https://mcp.strikerapi.com/v1/get_ratings", headers=headers, json={"league": "Premier League"})
            if elo_resp.status_code == 200:
                for team in elo_resp.json().get("data", []):
                    metrics["elo"][team["name"].lower()] = float(team["elo"])
            
            players_resp = requests.post("https://mcp.strikerapi.com/v1/get_player", headers=headers, json={"league": "Premier League"})
            if players_resp.status_code == 200:
                for p in players_resp.json().get("data", []):
                    metrics["player_per90"][p["name"].lower()] = {
                        "npxg90": float(p.get("npxg90", 0.0)),
                        "xa90": float(p.get("xa90", 0.0))
                    }
        except Exception as e:
            log.error(f"StrikerAPI fetch failed: {e}")

    # --- 5. Sportmonks (Free tier: Basic data for Danish & Scottish leagues) ---
    if SPORTMONKS_API_KEY:
        try:
            season_id = "YOUR_SPORTMONKS_SEASON_ID" 
            url = f"https://api.sportmonks.com/v3/football/teams/seasons/{season_id}"
            params = {"api_token": SPORTMONKS_API_KEY}
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                for team in resp.json().get("data", []):
                    team_name = team["name"].lower()
                    metrics["team_set_pieces"].setdefault(team_name, {})["basic_data"] = True
        except Exception as e:
            log.error(f"Sportmonks fetch failed: {e}")

    # Save to cache
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved external metrics to {CACHE_FILE}")

def load_external_metrics() -> dict:
    """Loads the cached metrics for the feature builder."""
    if not CACHE_FILE.exists():
        return {}
    with open(CACHE_FILE, "r") as f:
        return json.load(f)