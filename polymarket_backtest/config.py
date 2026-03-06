"""Configuration and constants for the low-volatility event backtest system."""

import os
from pathlib import Path

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Hard filters
SPORTS_KEYWORDS = [
    "nfl", "nba", "mlb", "nhl", "fifa", "epl", "premier league",
    "super bowl", "world cup", "champions league", "march madness",
    "ncaa", "ufc", "boxing", "wrestling", "tennis", "golf",
    "formula 1", "f1", "nascar", "mls", "wnba",
    "win the", "beat the", "cover the spread", "over/under",
    "score", "points", "game", "match", "season",
]

MIN_VOLUME = 1000
MIN_TRADING_DAYS = 7
LLM_CONFIDENCE_THRESHOLD = 0.70

# Backtest params
ENTRY_DAYS_BEFORE = [14, 7, 5, 3]
EXIT_DAYS_BEFORE = [2, 1, 0]

# Data
MARKETS_TO_FETCH = 10000
CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./output")

# API
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
REQUEST_TIMEOUT = 15

# DeepSeek
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_BATCH_SIZE = 50
LLM_RATE_LIMIT_DELAY = 0.5  # seconds between calls


def ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
