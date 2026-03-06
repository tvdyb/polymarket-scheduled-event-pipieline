"""Phase 1: Fetch historic markets from Polymarket Gamma API."""

import json
import time

import requests
from tqdm import tqdm

from .config import GAMMA_BASE_URL, CACHE_DIR, MARKETS_TO_FETCH, REQUEST_TIMEOUT


def fetch_all_markets(max_markets: int = MARKETS_TO_FETCH, batch_size: int = 100) -> list[dict]:
    """Pull recently-closed markets from Gamma API with pagination.

    Fetches newest-first (order=closedTime desc) because the CLOB only retains
    price history for ~1 month of closed markets. Saves to JSONL cache.
    """
    cache_path = CACHE_DIR / "markets_raw.jsonl"

    if cache_path.exists():
        print(f"Loading cached markets from {cache_path}")
        markets = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    markets.append(json.loads(line))
        print(f"Loaded {len(markets)} cached markets")
        return markets

    print(f"Fetching up to {max_markets} recently-closed markets (newest first)...")
    markets = []
    offset = 0
    seen_ids = set()

    with tqdm(total=max_markets, desc="Fetching markets") as pbar:
        while len(markets) < max_markets:
            try:
                r = requests.get(
                    f"{GAMMA_BASE_URL}/markets",
                    params={
                        "closed": "true",
                        "limit": batch_size,
                        "offset": offset,
                        "order": "closedTime",
                        "ascending": "false",
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                r.raise_for_status()
                batch = r.json()

                if not isinstance(batch, list) or len(batch) == 0:
                    print(f"\nNo more markets at offset {offset}")
                    break

                # Deduplicate
                new = 0
                for m in batch:
                    mid = m.get("id") or m.get("conditionId")
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        markets.append(m)
                        new += 1

                pbar.update(new)
                offset += batch_size

                if new == 0:
                    print(f"\nAll duplicates at offset {offset}, stopping")
                    break

                time.sleep(0.3)

            except requests.exceptions.RequestException as e:
                print(f"\nAPI error at offset {offset}: {e}")
                time.sleep(2)
                continue

    markets = markets[:max_markets]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for m in markets:
            f.write(json.dumps(m) + "\n")

    print(f"Fetched and cached {len(markets)} markets to {cache_path}")
    return markets


def parse_market(raw: dict) -> dict:
    """Extract the fields we need from a raw Gamma API market object."""
    # Parse volume/liquidity safely
    volume = 0.0
    try:
        volume = float(raw.get("volume") or raw.get("volumeNum") or 0)
    except (ValueError, TypeError):
        pass

    liquidity = 0.0
    try:
        liquidity = float(raw.get("liquidity") or raw.get("liquidityNum") or 0)
    except (ValueError, TypeError):
        pass

    # Parse outcome prices
    outcomes_raw = raw.get("outcomePrices") or raw.get("outcomes") or "[]"
    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except (json.JSONDecodeError, TypeError):
            outcomes_raw = []

    final_price = None
    if isinstance(outcomes_raw, list) and len(outcomes_raw) > 0:
        try:
            final_price = float(outcomes_raw[0])
        except (ValueError, TypeError, IndexError):
            pass

    # Get token IDs for price history fetching
    clob_token_ids = raw.get("clobTokenIds")
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids = json.loads(clob_token_ids)
        except (json.JSONDecodeError, TypeError):
            clob_token_ids = []
    if not isinstance(clob_token_ids, list):
        clob_token_ids = []

    # Determine outcome
    outcome = raw.get("outcome") or raw.get("resolution")
    if outcome is None and final_price is not None:
        outcome = "YES" if final_price > 0.5 else "NO"

    return {
        "id": str(raw.get("id") or raw.get("conditionId") or ""),
        "condition_id": str(raw.get("conditionId") or ""),
        "question": str(raw.get("question") or raw.get("title") or ""),
        "description": str(raw.get("description") or ""),
        "category": str(raw.get("category") or ""),
        "start_date": raw.get("startDate") or raw.get("createdAt") or "",
        "end_date": raw.get("endDate") or raw.get("end_date_iso") or "",
        "resolution_date": raw.get("resolutionDate") or raw.get("resolvedAt") or "",
        "volume": volume,
        "liquidity": liquidity,
        "outcome": str(outcome or ""),
        "final_price": final_price,
        "clob_token_ids": clob_token_ids,
        "slug": raw.get("slug") or "",
        "_raw": raw,  # keep raw for LLM context
    }
