"""Run the overhauled liquidity reversion backtest.

Processes ALL trades (no sampling), with latency-aware fills,
notional sizing, resolution filters, and concentration limits.
"""

import json
import time
from pathlib import Path

import polars as pl

from polymarket_backtester.liquidity_reversion.config import BacktestConfig
from polymarket_backtester.liquidity_reversion.backtester import LiquidityReversionBacktester
from polymarket_backtester.liquidity_reversion.reporting import (
    print_metrics, write_trade_log, write_equity_curve,
)

DATA_DIR = Path("./data/poly_data")
OUTPUT_DIR = Path("./output/liquidity_reversion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(config: BacktestConfig):
    """Load markets and ALL trades (no sampling)."""
    print("Loading markets...")
    markets_df = pl.read_csv(
        DATA_DIR / "markets.csv",
        infer_schema_length=10000,
        schema_overrides={
            "token1": pl.Utf8, "token2": pl.Utf8,
            "condition_id": pl.Utf8, "id": pl.Utf8,
        },
    )
    print(f"  {len(markets_df)} markets loaded")

    print(f"\nLoading ALL trades ({config.start_date} to {config.end_date})...")
    print("  No sampling — processing every trade.")
    t0 = time.time()

    trades_df = (
        pl.scan_csv(
            DATA_DIR / "processed" / "trades.csv",
            infer_schema_length=10000,
        )
        .filter(pl.col("timestamp") >= config.start_date)
        .filter(pl.col("timestamp") < config.end_date)
        .with_columns([
            pl.col("timestamp").str.to_datetime().dt.epoch("s").alias("timestamp_epoch"),
            pl.col("taker_direction").alias("taker_side"),
            pl.col("usd_amount").alias("size"),
            pl.when(pl.col("nonusdc_side") == "token1")
              .then(pl.lit("Yes"))
              .otherwise(pl.lit("No"))
              .alias("outcome"),
        ])
        .select([
            pl.col("timestamp_epoch").alias("timestamp"),
            pl.col("market_id").cast(pl.Utf8).alias("market_id"),
            "price",
            "size",
            "taker_side",
            "outcome",
            "maker",
            "taker",
        ])
        .collect()
    )

    elapsed = time.time() - t0
    print(f"  {len(trades_df):,} trades loaded in {elapsed:.1f}s")
    print(f"  Timestamp range: {trades_df['timestamp'].min()} - {trades_df['timestamp'].max()}")

    return markets_df, trades_df


def main():
    config = BacktestConfig()
    markets_df, trades_df = load_data(config)

    print(f"\nRunning liquidity reversion backtest...")
    print(f"  Config: sizing={config.sizing_mode}, max_notional={config.max_notional}")
    print(f"  Latency: {config.latency_trades} trades / {config.latency_seconds}s, "
          f"fill depth: {config.fill_depth_trades}, timeout: {config.fill_timeout_seconds}s")
    print(f"  Entry band: [{config.entry_price_min}, {config.entry_price_max}]")
    print(f"  Risk: {config.max_total_positions} max positions, "
          f"{config.max_positions_per_market}/market, "
          f"${config.max_notional_per_market}/market notional")

    bt = LiquidityReversionBacktester(config)
    bt.load_markets(markets_df)
    metrics = bt.run(trades_df, show_progress=True)

    # Print report
    print_metrics(metrics)

    # Write trade log
    trade_log_path = OUTPUT_DIR / "trade_log.csv"
    write_trade_log(bt.position_manager.closed_trades, trade_log_path)
    print(f"\nTrade log: {trade_log_path}")

    # Write equity curve
    equity_path = OUTPUT_DIR / "equity_curve.csv"
    write_equity_curve(bt._equity_curve, equity_path)
    print(f"Equity curve: {equity_path}")

    # Write metrics JSON
    metrics_path = OUTPUT_DIR / "metrics.json"
    save_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (set, type))}
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
