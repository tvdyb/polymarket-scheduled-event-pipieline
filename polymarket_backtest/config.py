"""Configuration and constants for the low-volatility event backtest system."""

import os
from pathlib import Path

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# ── Hard Filter Keywords ────────────────────────────────────────────────────

# Sports: leagues, orgs, game terms, player props
SPORTS_KEYWORDS = [
    # Leagues & organizations
    "nfl", "nba", "mlb", "nhl", "fifa", "epl", "premier league",
    "champions league", "europa league", "serie a", "la liga", "bundesliga",
    "ligue 1", "mls", "wnba", "ncaa", "ufc", "mma", "boxing", "wrestling",
    "tennis", "golf", "formula 1", "f1", "nascar", "indycar", "pga tour",
    "atp", "wta", "cricket", "ipl", "lpl", "lck", "lcs",
    "super bowl", "world cup", "march madness", "stanley cup", "world series",
    # Esports
    "counter-strike", "dota 2", "valorant", "league of legends", "lol:",
    "overwatch", "call of duty", "rocket league", "csgo", "cs2",
    # Game/match terms
    "cover the spread", "over/under", "o/u", "spread:",
    "moneyline", "point spread", "total kills",
    "map 1 winner", "map 2 winner", "map 3 winner",
    "1h spread", "set winner", "set 1 winner",
    # Match patterns
    " vs ", " vs. ", " v. ",
    # Player prop patterns
    "passing yards", "rushing yards", "receiving yards", "touchdowns",
    "rebounds", "assists", "strikeouts", "home runs",
    "points o/u", "rebounds o/u", "assists o/u",
]

# Awards: ceremony names and patterns
AWARDS_KEYWORDS = [
    "oscar", "academy award", "grammy", "emmy", "tony award",
    "golden globe", "sag award", "pga award", "dga award", "bafta",
    "cannes", "sundance", "billboard music", "mtv award", "bet award",
    "critic", "choice award", "spirit award", "annie award",
    "best picture", "best director", "best actor", "best actress",
    "best supporting", "best original", "best adapted", "best animated",
    "best documentary", "best international", "best visual",
    "best comedy", "best drama", "best series", "best limited",
    "best new artist", "album of the year", "song of the year",
    "record of the year", "best sports program",
    "win best", "nominated for",
]

# Election/political contest patterns
ELECTION_KEYWORDS = [
    "win the election", "win the primary", "win the nomination",
    "win the runoff", "win the seat", "win the race",
    "republican primary", "democratic primary", "primary winner",
    "senate primary", "governor primary", "congressional",
    "caucus winner", "electoral votes",
]

# Filter thresholds
MIN_VOLUME = 1000
MIN_TRADING_DAYS = 7
MAX_EVENT_GROUP_SIZE = 2  # standalone or YES/NO pair only
LLM_CONFIDENCE_THRESHOLD = 0.70

# Backtest params
ENTRY_DAYS_BEFORE = [14, 7, 5, 3]
EXIT_DAYS_BEFORE = [2, 1, 0]

# Data
MARKETS_TO_FETCH = 750000
CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./output")

# API
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
REQUEST_TIMEOUT = 15

# DeepSeek
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
