"""Fetch top whale wallets and their trade history from Polymarket public API."""

import json
import time
from pathlib import Path

import httpx
import polars as pl
from tqdm import tqdm

POLYMARKET_DATA_API = "https://data-api.polymarket.com"
LEADERBOARD_URL = "https://polymarket.com/leaderboard"

DEFAULT_CACHE_DIR = Path("./data/whales")


async def fetch_leaderboard_wallets(top_n: int = 50) -> list[dict]:
    """Fetch top PnL wallets from Polymarket data API."""
    async with httpx.AsyncClient(timeout=30) as client:
        wallets = []
        offset = 0
        while len(wallets) < top_n:
            r = await client.get(
                f"{POLYMARKET_DATA_API}/leaderboard",
                params={"limit": min(50, top_n - len(wallets)), "offset": offset},
            )
            if r.status_code != 200:
                break
            data = r.json()
            if not data:
                break
            wallets.extend(data)
            offset += len(data)
        return wallets[:top_n]


async def fetch_wallet_activity(address: str, limit: int = 1000) -> list[dict]:
    """Fetch trade history for a single wallet."""
    async with httpx.AsyncClient(timeout=30) as client:
        all_trades = []
        cursor = None
        while len(all_trades) < limit:
            params = {"user": address, "limit": 100}
            if cursor:
                params["cursor"] = cursor
            r = await client.get(f"{POLYMARKET_DATA_API}/activity", params=params)
            if r.status_code != 200:
                break
            data = r.json()
            if not data:
                break
            all_trades.extend(data)
            # Simple pagination — break if we got less than requested
            if len(data) < 100:
                break
            cursor = data[-1].get("id")
        return all_trades[:limit]


async def fetch_wallet_positions(address: str) -> list[dict]:
    """Fetch current positions for a wallet."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{POLYMARKET_DATA_API}/positions", params={"user": address})
        if r.status_code != 200:
            return []
        return r.json() or []


def build_whale_trades_df(whale_wallets: list[dict], activities: dict[str, list[dict]]) -> pl.DataFrame:
    """Build a polars DataFrame of whale trades from raw API data."""
    rows = []
    for wallet in whale_wallets:
        addr = wallet.get("address") or wallet.get("userAddress", "")
        pnl = wallet.get("pnl") or wallet.get("totalPnl", 0)
        trades = activities.get(addr, [])
        for t in trades:
            rows.append({
                "whale_address": addr,
                "whale_pnl": float(pnl),
                "market_id": t.get("market", t.get("conditionId", "")),
                "timestamp": int(t.get("timestamp", 0)),
                "side": t.get("side", t.get("type", "")),
                "price": float(t.get("price", 0)),
                "size": float(t.get("size", t.get("amount", 0))),
                "outcome": t.get("outcome", ""),
            })

    if not rows:
        return pl.DataFrame(schema={
            "whale_address": pl.Utf8, "whale_pnl": pl.Float64,
            "market_id": pl.Utf8, "timestamp": pl.Int64,
            "side": pl.Utf8, "price": pl.Float64,
            "size": pl.Float64, "outcome": pl.Utf8,
        })

    return pl.DataFrame(rows)


def save_whale_data(df: pl.DataFrame, cache_dir: Path = DEFAULT_CACHE_DIR):
    """Save whale trades to parquet for fast reload."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "whale_trades.parquet"
    df.write_parquet(path)
    print(f"Saved {len(df)} whale trades to {path}")


def load_whale_data(cache_dir: Path = DEFAULT_CACHE_DIR) -> pl.DataFrame | None:
    """Load cached whale trades."""
    path = cache_dir / "whale_trades.parquet"
    if path.exists():
        return pl.read_parquet(path)
    return None
