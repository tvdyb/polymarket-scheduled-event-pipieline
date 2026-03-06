"""Fetch and manage poly_data snapshots.

Uses the poly_data repo (https://github.com/warproxxx/poly_data) for historical data.
Downloads the data snapshot to avoid 2+ days of initial scraping.
"""

import os
import subprocess
from pathlib import Path

import polars as pl
from tqdm import tqdm

DEFAULT_DATA_DIR = Path("./data/poly_data")


def clone_or_update_poly_data(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Clone or pull the poly_data repo."""
    if (data_dir / ".git").exists():
        print(f"Updating poly_data at {data_dir}...")
        subprocess.run(["git", "pull"], cwd=data_dir, check=True)
    else:
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning poly_data to {data_dir}...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/warproxxx/poly_data.git", str(data_dir)],
            check=True,
        )
    return data_dir


def download_snapshot(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the pre-built data snapshot (faster than scraping).

    The snapshot URL is from the poly_data README. Falls back to
    instructing the user if the direct download isn't available.
    """
    snapshot_dir = data_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    markets_path = snapshot_dir / "markets.csv"
    trades_path = snapshot_dir / "processed" / "trades.csv"

    if markets_path.exists() and trades_path.exists():
        print(f"Snapshot already exists at {snapshot_dir}")
        return snapshot_dir

    print("To get the poly_data snapshot:")
    print("  1. Visit https://github.com/warproxxx/poly_data")
    print("  2. Download the latest data snapshot")
    print(f"  3. Extract to {snapshot_dir}/")
    print("  Required files:")
    print(f"    - {markets_path}")
    print(f"    - {trades_path}")
    print()
    print("Or run the poly_data scraper (takes 2+ days):")
    print(f"  cd {data_dir} && python -m poly_data.scrape")

    return snapshot_dir


def load_markets(data_dir: Path = DEFAULT_DATA_DIR) -> pl.LazyFrame:
    """Load markets.csv as a polars LazyFrame."""
    # Search common locations
    for candidate in [
        data_dir / "snapshot" / "markets.csv",
        data_dir / "markets.csv",
        data_dir / "data" / "markets.csv",
    ]:
        if candidate.exists():
            return pl.scan_csv(candidate, infer_schema_length=10000)

    raise FileNotFoundError(f"markets.csv not found under {data_dir}")


def load_trades(data_dir: Path = DEFAULT_DATA_DIR) -> pl.LazyFrame:
    """Load trades CSV as a polars LazyFrame. Handles both processed and raw."""
    for candidate in [
        data_dir / "snapshot" / "processed" / "trades.csv",
        data_dir / "processed" / "trades.csv",
        data_dir / "data" / "processed" / "trades.csv",
        data_dir / "snapshot" / "trades.csv",
    ]:
        if candidate.exists():
            return pl.scan_csv(candidate, infer_schema_length=10000)

    # Try parquet
    for candidate in [
        data_dir / "snapshot" / "processed" / "trades.parquet",
        data_dir / "processed" / "trades.parquet",
    ]:
        if candidate.exists():
            return pl.scan_parquet(candidate)

    raise FileNotFoundError(f"trades.csv/parquet not found under {data_dir}")


def load_order_filled(data_dir: Path = DEFAULT_DATA_DIR) -> pl.LazyFrame:
    """Load raw orderFilled events from Goldsky indexer."""
    for candidate in [
        data_dir / "snapshot" / "goldsky" / "orderFilled.csv",
        data_dir / "goldsky" / "orderFilled.csv",
        data_dir / "data" / "goldsky" / "orderFilled.csv",
    ]:
        if candidate.exists():
            return pl.scan_csv(candidate, infer_schema_length=10000)

    raise FileNotFoundError(f"orderFilled.csv not found under {data_dir}")
