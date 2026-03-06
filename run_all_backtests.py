"""Run all 5 strategies against poly_data and print results."""

import json
import time
from pathlib import Path

import polars as pl

from polymarket_backtester.engine.backtester import Backtester
from polymarket_backtester.engine.metrics import compute_metrics, print_metrics
from polymarket_backtester.strategies.scheduled_momentum import ScheduledMomentumStrategy
from polymarket_backtester.strategies.liquidity_reversion import LiquidityReversionStrategy
from polymarket_backtester.strategies.whale_follow import WhaleFollowStrategy
from polymarket_backtester.strategies.cross_market_arb import CrossMarketArbStrategy
from polymarket_backtester.strategies.resolution_catalyst import ResolutionCatalystStrategy
from polymarket_backtester.analysis.visualize import plot_equity_curve, plot_trade_distribution

DATA_DIR = Path("./data/poly_data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Date range for backtesting
START_DATE = "2024-07-01"
END_DATE = "2024-10-01"
# Sample every Nth trade to speed up (1 = no sampling)
SAMPLE_RATE = 5

STRATEGIES = {
    "scheduled_momentum": ScheduledMomentumStrategy(
        hours_before_expiry=24, edge_threshold=0.03, max_position_usd=500,
        min_market_volume=1000 / SAMPLE_RATE, min_price=0.70, max_price=0.97,
    ),
    "liquidity_reversion": LiquidityReversionStrategy(
        impact_threshold=0.05, low_volume_threshold=50000 / SAMPLE_RATE,
        max_hold_seconds=14400, reversion_target_pct=0.60, max_position_usd=300,
    ),
    "whale_follow": WhaleFollowStrategy(
        min_whale_size=5000, follow_delay_seconds=900,
        take_profit=0.05, stop_loss=0.05,
        max_market_volume=200000 / SAMPLE_RATE, max_position_usd=300,
    ),
    "cross_market_arb": CrossMarketArbStrategy(
        min_inversion_cents=0.02, min_event_markets=3,
        max_hold_seconds=86400, max_position_usd=300,
    ),
    "resolution_catalyst": ResolutionCatalystStrategy(
        entry_hours_before=48, exit_hours_before=2,
        half_spread=0.03, max_inventory=1000, max_position_usd=300,
    ),
}


def load_and_prepare_data():
    """Load markets and trades, map columns to backtester format."""
    print("Loading markets...")
    markets_df = pl.read_csv(
        DATA_DIR / "markets.csv",
        infer_schema_length=10000,
        schema_overrides={"token1": pl.Utf8, "token2": pl.Utf8, "condition_id": pl.Utf8, "id": pl.Utf8},
    )
    print(f"  {len(markets_df)} markets loaded")

    # Build token->market mapping and token->side mapping for trade enrichment
    # In poly_data: token1 = answer1 (usually Yes), token2 = answer2 (usually No)
    print(f"\nLoading trades ({START_DATE} to {END_DATE})...")
    print("  This may take a few minutes on the 33GB file...")

    t0 = time.time()
    trades_lf = pl.scan_csv(
        DATA_DIR / "processed" / "trades.csv",
        infer_schema_length=10000,
    )

    # Filter date range, select and rename columns
    trades_df = (
        trades_lf
        .filter(pl.col("timestamp") >= START_DATE)
        .filter(pl.col("timestamp") < END_DATE)
        .with_columns([
            # Convert ISO timestamp to epoch seconds
            pl.col("timestamp").str.to_datetime().dt.epoch("s").alias("timestamp_epoch"),
            # Map taker_direction -> taker_side
            pl.col("taker_direction").alias("taker_side"),
            # Use usd_amount as size
            pl.col("usd_amount").alias("size"),
            # Map nonusdc_side to outcome
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

    # Sample to speed up backtesting
    if SAMPLE_RATE > 1:
        trades_df = trades_df.gather_every(SAMPLE_RATE)
        print(f"  Sampled every {SAMPLE_RATE}th trade: {len(trades_df):,} trades")

    elapsed = time.time() - t0
    print(f"  {len(trades_df):,} trades loaded in {elapsed:.1f}s")
    print(f"  Date range: {trades_df['timestamp'].min()} - {trades_df['timestamp'].max()}")

    return markets_df, trades_df


def run_strategy(name, strategy, markets_df, trades_df):
    """Run a single strategy and return metrics."""
    print(f"\n{'='*60}")
    print(f"  STRATEGY: {name}")
    print(f"{'='*60}")

    bt = Backtester(
        strategy=strategy,
        initial_cash=10_000.0,
        slippage_cents=0.01,
        max_position_usd=500.0,
    )
    bt.load_markets(markets_df)
    metrics = bt.run(trades_df, show_progress=True)
    print_metrics(metrics)

    # Save outputs
    prefix = OUTPUT_DIR / name

    # Metrics JSON
    save_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (set, type))}
    with open(f"{prefix}_metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)

    # Equity curve
    if bt.portfolio.equity_curve:
        plot_equity_curve(bt.portfolio.equity_curve, f"{prefix}_equity.png", title=f"{name}")

    # Trade distribution
    if bt.portfolio.closed_trades:
        plot_trade_distribution(bt.portfolio.closed_trades, f"{prefix}_trades.png", title=f"{name}")

    return metrics


def main():
    markets_df, trades_df = load_and_prepare_data()

    all_results = {}
    for name, strategy in STRATEGIES.items():
        try:
            metrics = run_strategy(name, strategy, markets_df, trades_df)
            all_results[name] = metrics
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Summary table
    print(f"\n\n{'='*80}")
    print("  SUMMARY — ALL STRATEGIES")
    print(f"{'='*80}")
    print(f"{'Strategy':<25} {'Total PnL':>12} {'Win Rate':>10} {'Trades':>8} {'Sharpe':>8} {'Max DD':>10}")
    print("-" * 80)
    for name, m in all_results.items():
        if "error" in m:
            print(f"{name:<25} {'ERROR':>12}")
            continue
        pnl = m.get("realized_pnl", 0)
        wr = m.get("win_rate", 0)
        trades = m.get("total_trades", 0)
        sharpe = m.get("sharpe", 0)
        mdd = m.get("max_drawdown_pct", 0)
        print(f"{name:<25} ${pnl:>+11,.2f} {wr:>9.1%} {trades:>8} {sharpe:>8.2f} {mdd:>9.1%}")

    # Save summary
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (set, type))}
                   for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
