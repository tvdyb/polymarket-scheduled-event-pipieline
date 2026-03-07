"""Run the overhauled liquidity reversion backtest.

Processes ALL trades (no sampling), with latency-aware fills,
notional sizing, resolution filters, and concentration limits.

Usage:
  python run_liquidity_reversion.py              # in-sample (Jul-Oct 2024)
  python run_liquidity_reversion.py --oos        # out-of-sample (Jan-Apr 2025)
  python run_liquidity_reversion.py --both       # run both and compare
"""

import argparse
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
            # Normalize price to YES-side: if trade is for NO token, YES price = 1 - price
            pl.when(pl.col("nonusdc_side") == "token1")
              .then(pl.col("price"))
              .otherwise(1.0 - pl.col("price"))
              .alias("yes_price"),
        ])
        .select([
            pl.col("timestamp_epoch").alias("timestamp"),
            pl.col("market_id").cast(pl.Utf8).alias("market_id"),
            pl.col("yes_price").alias("price"),
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


def run_backtest(config: BacktestConfig, label: str):
    """Run a single backtest with given config and label."""
    output_dir = Path(f"./output/liquidity_reversion/{label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    markets_df, trades_df = load_data(config)

    print(f"\n{'='*70}")
    print(f"  Running: {label} ({config.start_date} to {config.end_date})")
    print(f"{'='*70}")
    print(f"  Config: sizing={config.sizing_mode}, max_notional={config.max_notional}")
    print(f"  Impact threshold: {config.impact_threshold}, min VWAP trades: {config.min_vwap_trades}")
    print(f"  Latency: {config.latency_trades} trades / {config.latency_seconds}s, "
          f"fill depth: {config.fill_depth_trades}, timeout: {config.fill_timeout_seconds}s")
    print(f"  Entry band: [{config.entry_price_min}, {config.entry_price_max}]")
    print(f"  Risk: {config.max_total_positions} max positions, "
          f"{config.max_positions_per_market}/market, "
          f"${config.max_notional_per_market}/market notional")

    bt = LiquidityReversionBacktester(config)
    bt.load_markets(markets_df)
    metrics = bt.run(trades_df, show_progress=True)

    print_metrics(metrics)

    write_trade_log(bt.position_manager.closed_trades, output_dir / "trade_log.csv")
    write_equity_curve(bt._equity_curve, output_dir / "equity_curve.csv")

    save_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (set, type))}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)

    print(f"\nOutputs saved to {output_dir}/")
    return metrics


def print_comparison(is_metrics: dict, oos_metrics: dict):
    """Print side-by-side comparison of in-sample vs out-of-sample."""
    print(f"\n\n{'='*80}")
    print("  IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'In-Sample':>18} {'Out-of-Sample':>18} {'Delta':>12}")
    print("-" * 80)

    rows = [
        ("Gross PnL", "gross_pnl", "${:>+14,.2f}"),
        ("Net PnL", "net_pnl", "${:>+14,.2f}"),
        ("Transaction Costs", "transaction_costs", "${:>14,.2f}"),
        ("Win Rate", "win_rate", "{:>14.1%}"),
        ("Profit Factor", "profit_factor", "{:>14.2f}"),
        ("Sharpe (annualized)", "sharpe", "{:>14.2f}"),
        ("Max Drawdown", "max_drawdown", "${:>14,.2f}"),
        ("Total Trades", "total_trades", "{:>14,}"),
        ("Signals Generated", "total_signals_generated", "{:>14,}"),
        ("Fill Rate", "fill_rate", "{:>14.1%}"),
        ("Avg Hold (hours)", "avg_hold_hours", "{:>14.1f}"),
        ("Avg Winner", "avg_winner", "${:>+14,.2f}"),
        ("Avg Loser", "avg_loser", "${:>+14,.2f}"),
    ]

    for label, key, fmt in rows:
        is_val = is_metrics.get(key, 0)
        oos_val = oos_metrics.get(key, 0)
        try:
            is_str = fmt.format(is_val)
            oos_str = fmt.format(oos_val)
            if isinstance(is_val, (int, float)) and isinstance(oos_val, (int, float)) and is_val != 0:
                delta_pct = (oos_val - is_val) / abs(is_val) * 100
                delta_str = f"{delta_pct:>+10.1f}%"
            else:
                delta_str = ""
        except (ValueError, TypeError):
            is_str = str(is_val)[:18]
            oos_str = str(oos_val)[:18]
            delta_str = ""
        print(f"  {label:<28} {is_str:>18} {oos_str:>18} {delta_str:>12}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Liquidity Reversion Backtest")
    parser.add_argument("--oos", action="store_true", help="Run out-of-sample only (Jan-Apr 2025)")
    parser.add_argument("--both", action="store_true", help="Run both in-sample and out-of-sample")
    args = parser.parse_args()

    is_config = BacktestConfig(start_date="2024-07-01", end_date="2024-10-01")
    oos_config = BacktestConfig(start_date="2025-01-01", end_date="2025-04-01")

    if args.both:
        is_metrics = run_backtest(is_config, "in_sample")
        oos_metrics = run_backtest(oos_config, "out_of_sample")
        print_comparison(is_metrics, oos_metrics)
    elif args.oos:
        run_backtest(oos_config, "out_of_sample")
    else:
        run_backtest(is_config, "in_sample")


if __name__ == "__main__":
    main()
