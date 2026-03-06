"""Phase 1: Fetch historic markets from Polymarket Gamma API.

Uses async concurrent requests to fetch 750K+ markets in ~10 minutes
instead of ~70 minutes sequentially.
"""

import asyncio
import json

import httpx
from tqdm import tqdm

from .config import GAMMA_BASE_URL, CACHE_DIR, MARKETS_TO_FETCH, REQUEST_TIMEOUT

FETCH_CONCURRENCY = 30  # parallel page requests


async def _fetch_page(client: httpx.AsyncClient, offset: int, batch_size: int, semaphore: asyncio.Semaphore) -> list[dict]:
    """Fetch a single page of markets."""
    async with semaphore:
        for attempt in range(3):
            try:
                r = await client.get(
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
                data = r.json()
                return data if isinstance(data, list) else []
            except Exception:
                await asyncio.sleep(1.0 * (attempt + 1))
    return []


async def _fetch_all_async(max_markets: int, batch_size: int = 100) -> list[dict]:
    """Fetch all pages concurrently."""
    semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)
    total_pages = (max_markets + batch_size - 1) // batch_size
    offsets = [i * batch_size for i in range(total_pages)]

    async with httpx.AsyncClient() as client:
        tasks = [_fetch_page(client, offset, batch_size, semaphore) for offset in offsets]

        markets = []
        seen_ids = set()
        empty_streak = 0

        with tqdm(total=len(tasks), desc="Fetching market pages") as pbar:
            # Process in order to detect end of data
            for i, task in enumerate(asyncio.as_completed(tasks)):
                batch = await task
                pbar.update(1)

                if not batch:
                    empty_streak += 1
                    if empty_streak > 20:
                        break
                    continue

                empty_streak = 0
                for m in batch:
                    mid = m.get("id") or m.get("conditionId")
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        markets.append(m)

    return markets[:max_markets]


def fetch_all_markets(max_markets: int = MARKETS_TO_FETCH, batch_size: int = 100) -> list[dict]:
    """Pull closed markets from Gamma API with concurrent pagination. Saves to JSONL cache."""
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

    print(f"Fetching up to {max_markets} closed markets ({FETCH_CONCURRENCY} concurrent)...")

    markets = asyncio.run(_fetch_all_async(max_markets, batch_size))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for m in markets:
            f.write(json.dumps(m) + "\n")

    print(f"Fetched and cached {len(markets)} markets to {cache_path}")
    return markets


def count_event_group_sizes(raw_markets: list[dict]) -> dict[str, int]:
    """Pre-pass: count how many markets share each event ID."""
    counts: dict[str, int] = {}
    for m in raw_markets:
        for e in m.get("events", []):
            eid = e.get("id")
            if eid:
                counts[eid] = counts.get(eid, 0) + 1
    return counts


def parse_market(raw: dict, event_group_sizes: dict[str, int] | None = None) -> dict:
    """Extract the fields we need from a raw Gamma API market object."""
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

    clob_token_ids = raw.get("clobTokenIds")
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids = json.loads(clob_token_ids)
        except (json.JSONDecodeError, TypeError):
            clob_token_ids = []
    if not isinstance(clob_token_ids, list):
        clob_token_ids = []

    outcome = raw.get("outcome") or raw.get("resolution")
    if outcome is None and final_price is not None:
        outcome = "YES" if final_price > 0.5 else "NO"

    group_size = 1
    if event_group_sizes:
        for e in raw.get("events", []):
            eid = e.get("id")
            if eid and eid in event_group_sizes:
                group_size = max(group_size, event_group_sizes[eid])

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
        "event_group_size": group_size,
        "_raw": raw,
    }
