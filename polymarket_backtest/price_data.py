"""Phase 3: Fetch CLOB price history for filtered markets only.

This runs AFTER all filtering is complete — the lookahead firewall.
"""

import json
import time
from datetime import datetime, timezone

import requests
from tqdm import tqdm

from .config import CLOB_BASE_URL, CACHE_DIR, REQUEST_TIMEOUT


def _parse_timestamp(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e12:
            v /= 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (ValueError, OSError):
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _fetch_price_history(token_id: str) -> list[dict]:
    """Fetch daily price history for a single token from CLOB API."""
    try:
        r = requests.get(
            f"{CLOB_BASE_URL}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": 60},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code >= 400:
            return []
        payload = r.json() if r.text else {}
        hist = payload.get("history", []) if isinstance(payload, dict) else []
        if not isinstance(hist, list):
            return []
    except requests.exceptions.RequestException:
        return []

    # Normalize to daily prices
    daily = {}
    for point in hist:
        ts = _parse_timestamp(point.get("t") or point.get("timestamp") or point.get("time"))
        if not ts:
            continue

        price = point.get("p") or point.get("price")
        if price is None:
            h = point.get("h") or point.get("high")
            l_val = point.get("l") or point.get("low")
            if h is not None and l_val is not None:
                try:
                    price = (float(h) + float(l_val)) / 2.0
                except (ValueError, TypeError):
                    continue
        if price is None:
            continue

        try:
            price = float(price)
        except (ValueError, TypeError):
            continue

        date_str = ts.strftime("%Y-%m-%d")
        # Keep last price of each day
        daily[date_str] = price

    return [{"date": d, "price": p} for d, p in sorted(daily.items())]


def fetch_prices_for_markets(markets: list[dict]) -> dict[str, list[dict]]:
    """Fetch price history for all filtered markets. Caches per-token.

    Returns dict mapping market_id -> list of {date, price} dicts.
    """
    cache_path = CACHE_DIR / "price_histories.json"

    # Load cache
    cached = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except (json.JSONDecodeError, IOError):
            cached = {}

    prices = {}
    tokens_to_fetch = []

    for m in markets:
        mid = m["id"]
        token_ids = m.get("clob_token_ids", [])
        # Use first token (YES token) for price history
        token_id = token_ids[0] if token_ids else None

        if not token_id:
            continue

        if token_id in cached:
            prices[mid] = cached[token_id]
        else:
            tokens_to_fetch.append((mid, token_id))

    if tokens_to_fetch:
        print(f"Fetching price history for {len(tokens_to_fetch)} markets ({len(cached)} cached)...")

        for mid, token_id in tqdm(tokens_to_fetch, desc="Fetching prices"):
            history = _fetch_price_history(token_id)
            cached[token_id] = history
            prices[mid] = history
            time.sleep(0.2)

        # Save updated cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached, f)

        print(f"Price data fetched for {len(prices)} markets")

    return prices
