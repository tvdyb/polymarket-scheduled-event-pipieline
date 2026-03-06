"""CLI entry point: pick strategy, date range, params, run backtest."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

from ..data.fetch_poly_data import load_markets, load_trades
from ..engine.backtester import Backtester
from ..engine.metrics import print_metrics
from ..strategies.scheduled_momentum import ScheduledMomentumStrategy
from ..strategies.liquidity_reversion import LiquidityReversionStrategy
from ..strategies.whale_follow import WhaleFollowStrategy
from ..strategies.cross_market_arb import CrossMarketArbStrategy
from ..strategies.resolution_catalyst import ResolutionCatalystStrategy
from .visualize import plot_equity_curve, plot_trade_distribution

STRATEGIES = {
    "scheduled_momentum": ScheduledMomentumStrategy,
    "liquidity_reversion": LiquidityReversionStrategy,
    "whale_follow": WhaleFollowStrategy,
    "cross_market_arb": CrossMarketArbStrategy,
    "resolution_catalyst": ResolutionCatalystStrategy,
}


def load_config(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Polymarket Strategy Backtester")
    parser.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--data-dir", default="./data/poly_data", help="poly_data directory")
    parser.add_argument("--cash", type=float, default=10000, help="Initial cash")
    parser.add_argument("--slippage", type=float, default=0.01, help="Slippage in cents")
    parser.add_argument("--output", default="./output/backtest", help="Output prefix")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir)

    # Parse date range to epoch seconds
    start_ts = int(datetime.strptime(args.start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(args.end, "%Y-%m-%d").timestamp())

    print(f"Loading data from {data_dir}...")

    # Load markets
    try:
        markets_lf = load_markets(data_dir)
        markets_df = markets_lf.collect()
        print(f"  Markets: {len(markets_df)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run data/fetch_poly_data.py first to download the poly_data snapshot.")
        sys.exit(1)

    # Load trades
    try:
        trades_lf = load_trades(data_dir)
        # Filter by date range
        trades_df = (
            trades_lf
            .filter(pl.col("timestamp") >= start_ts)
            .filter(pl.col("timestamp") <= end_ts)
            .collect()
        )
        print(f"  Trades: {len(trades_df)} (from {args.start} to {args.end})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if len(trades_df) == 0:
        print("No trades in date range. Check your data and date range.")
        sys.exit(1)

    # Initialize strategy
    strategy_cls = STRATEGIES[args.strategy]
    strategy_params = config.get("strategies", {}).get(args.strategy, {})
    strategy = strategy_cls(**strategy_params)

    print(f"\nRunning {args.strategy} strategy...")
    print(f"  Cash: ${args.cash:,.0f}  Slippage: {args.slippage}c")

    # Run backtest
    bt = Backtester(
        strategy=strategy,
        initial_cash=args.cash,
        slippage_cents=args.slippage,
    )
    bt.load_markets(markets_df)
    metrics = bt.run(trades_df)

    # Print results
    print_metrics(metrics)

    # Save results
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = f"{args.output}_{args.strategy}"

    # Save metrics JSON
    metrics_path = f"{output_prefix}_metrics.json"
    # Remove non-serializable items
    save_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (set, type))}
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    # Plot equity curve
    if bt.portfolio.equity_curve:
        equity_path = f"{output_prefix}_equity.png"
        plot_equity_curve(bt.portfolio.equity_curve, equity_path, title=f"{args.strategy} Equity Curve")
        print(f"Equity curve saved to {equity_path}")

    # Plot trade distribution
    if bt.portfolio.closed_trades:
        dist_path = f"{output_prefix}_trades.png"
        plot_trade_distribution(bt.portfolio.closed_trades, dist_path, title=f"{args.strategy} Trade Distribution")
        print(f"Trade distribution saved to {dist_path}")


if __name__ == "__main__":
    main()
