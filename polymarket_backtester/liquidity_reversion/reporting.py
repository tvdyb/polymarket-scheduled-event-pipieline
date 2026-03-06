"""Structured reporting: metrics, trade log CSV, equity curve CSV."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from .config import BacktestConfig
from .position_manager import ClosedPosition


def compute_metrics(closed: list[ClosedPosition], equity_curve: list[dict],
                    config: BacktestConfig, filter_counts: dict[str, int],
                    total_signals: int, total_fills: int,
                    runtime_seconds: float) -> dict:
    """Compute all required metrics from completed backtest."""
    result: dict = {}

    # Basic counts
    result["total_trades"] = len(closed)
    result["total_signals_generated"] = total_signals
    result["total_fills"] = total_fills
    result["fill_rate"] = total_fills / total_signals if total_signals > 0 else 0.0

    if not closed:
        result["gross_pnl"] = 0.0
        result["net_pnl"] = 0.0
        return result

    # Gross / Net PnL
    gross_pnl = sum(t.pnl for t in closed)
    total_entry_notional = sum(t.entry_notional for t in closed)
    transaction_costs = total_entry_notional * config.transaction_cost_pct
    net_pnl = gross_pnl - transaction_costs

    result["gross_pnl"] = gross_pnl
    result["transaction_costs"] = transaction_costs
    result["net_pnl"] = net_pnl

    # Win/loss
    winners = [t for t in closed if t.pnl > 0]
    losers = [t for t in closed if t.pnl <= 0]
    result["win_rate"] = len(winners) / len(closed)
    result["avg_winner"] = sum(t.pnl for t in winners) / len(winners) if winners else 0.0
    result["avg_loser"] = sum(t.pnl for t in losers) / len(losers) if losers else 0.0

    # Profit factor
    gross_wins = sum(t.pnl for t in winners)
    gross_losses = abs(sum(t.pnl for t in losers))
    result["profit_factor"] = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Hold time
    hold_times = [t.hold_seconds for t in closed]
    result["avg_hold_seconds"] = sum(hold_times) / len(hold_times)
    result["avg_hold_hours"] = result["avg_hold_seconds"] / 3600

    # Max drawdown from equity curve
    if equity_curve:
        peak = equity_curve[0]["cumulative_pnl"]
        max_dd = 0.0
        for pt in equity_curve:
            val = pt["cumulative_pnl"]
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        result["max_drawdown"] = max_dd

    # Sharpe: annualized from DAILY PnL series (not per-trade)
    if equity_curve and len(equity_curve) > 1:
        daily_pnls = [equity_curve[i]["cumulative_pnl"] - equity_curve[i - 1]["cumulative_pnl"]
                      for i in range(1, len(equity_curve))]
        if daily_pnls:
            mean_daily = sum(daily_pnls) / len(daily_pnls)
            var_daily = sum((r - mean_daily) ** 2 for r in daily_pnls) / (len(daily_pnls) - 1) if len(daily_pnls) > 1 else 0
            std_daily = math.sqrt(var_daily) if var_daily > 0 else 1e-9
            result["sharpe"] = (mean_daily / std_daily) * math.sqrt(365)
        else:
            result["sharpe"] = 0.0
    else:
        result["sharpe"] = 0.0

    # Entry price distribution
    buckets = {
        "0.00-0.15": (0.0, 0.15),
        "0.15-0.30": (0.15, 0.30),
        "0.30-0.50": (0.30, 0.50),
        "0.50-0.70": (0.50, 0.70),
        "0.70-0.85": (0.70, 0.85),
        "0.85-1.00": (0.85, 1.0),
    }
    entry_dist: dict[str, dict] = {}
    for label, (lo, hi) in buckets.items():
        in_bucket = [t for t in closed if lo <= t.entry_price < hi]
        if in_bucket:
            entry_dist[label] = {
                "count": len(in_bucket),
                "pnl": sum(t.pnl for t in in_bucket),
                "pct_of_trades": len(in_bucket) / len(closed),
                "pct_of_pnl": sum(t.pnl for t in in_bucket) / gross_pnl if gross_pnl != 0 else 0,
            }
    result["entry_price_distribution"] = entry_dist

    # Top 5 markets by PnL contribution
    by_market: dict[str, list[ClosedPosition]] = defaultdict(list)
    for t in closed:
        by_market[t.market_id].append(t)
    market_pnl = [
        {
            "market_id": mid,
            "trade_count": len(trades),
            "cumulative_pnl": sum(t.pnl for t in trades),
            "pct_of_total": sum(t.pnl for t in trades) / gross_pnl if gross_pnl != 0 else 0,
        }
        for mid, trades in by_market.items()
    ]
    market_pnl.sort(key=lambda x: abs(x["cumulative_pnl"]), reverse=True)
    result["top_5_markets"] = market_pnl[:5]

    # Exit reason breakdown
    exit_reasons: dict[str, int] = defaultdict(int)
    for t in closed:
        exit_reasons[t.exit_reason] += 1
    result["exit_reasons"] = dict(exit_reasons)

    # Signals filtered breakdown
    result["signals_filtered"] = dict(filter_counts)

    result["runtime_seconds"] = runtime_seconds

    return result


def write_trade_log(closed: list[ClosedPosition], path: Path):
    """Write every trade to CSV."""
    fieldnames = [
        "signal_time", "fill_time", "market_id", "side", "signal_price",
        "fill_price", "slippage_bps", "shares", "entry_notional",
        "exit_time", "exit_price", "pnl", "hold_seconds", "exit_reason",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in closed:
            writer.writerow({
                "signal_time": t.signal_time,
                "fill_time": t.fill_time,
                "market_id": t.market_id,
                "side": t.side,
                "signal_price": f"{t.signal_price:.6f}",
                "fill_price": f"{t.entry_price:.6f}",
                "slippage_bps": f"{t.slippage_bps:.1f}",
                "shares": f"{t.shares:.2f}",
                "entry_notional": f"{t.entry_notional:.2f}",
                "exit_time": t.exit_time,
                "exit_price": f"{t.exit_price:.6f}",
                "pnl": f"{t.pnl:.2f}",
                "hold_seconds": t.hold_seconds,
                "exit_reason": t.exit_reason,
            })


def write_equity_curve(equity_curve: list[dict], path: Path):
    """Write daily equity curve to CSV."""
    fieldnames = ["timestamp", "cumulative_pnl", "open_positions", "total_notional_exposure"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pt in equity_curve:
            writer.writerow({
                "timestamp": pt["timestamp"],
                "cumulative_pnl": f"{pt['cumulative_pnl']:.2f}",
                "open_positions": pt["open_positions"],
                "total_notional_exposure": f"{pt['total_notional_exposure']:.2f}",
            })


def print_metrics(metrics: dict):
    """Pretty-print structured report to console."""
    print("\n" + "=" * 70)
    print("  LIQUIDITY REVERSION BACKTEST REPORT")
    print("=" * 70)

    print(f"\n--- Performance ---")
    print(f"  Gross PnL:           ${metrics.get('gross_pnl', 0):>+12,.2f}")
    print(f"  Transaction costs:   ${metrics.get('transaction_costs', 0):>12,.2f}")
    print(f"  Net PnL:             ${metrics.get('net_pnl', 0):>+12,.2f}")
    print(f"  Win rate:            {metrics.get('win_rate', 0):>11.1%}")
    print(f"  Avg winner:          ${metrics.get('avg_winner', 0):>+12,.2f}")
    print(f"  Avg loser:           ${metrics.get('avg_loser', 0):>+12,.2f}")
    print(f"  Profit factor:       {metrics.get('profit_factor', 0):>12.2f}")
    print(f"  Sharpe (annualized): {metrics.get('sharpe', 0):>12.2f}")
    print(f"  Max drawdown:        ${metrics.get('max_drawdown', 0):>12,.2f}")

    print(f"\n--- Execution ---")
    print(f"  Signals generated:   {metrics.get('total_signals_generated', 0):>12,}")
    print(f"  Fills:               {metrics.get('total_fills', 0):>12,}")
    print(f"  Fill rate:           {metrics.get('fill_rate', 0):>11.1%}")
    print(f"  Trades (closed):     {metrics.get('total_trades', 0):>12,}")
    print(f"  Avg hold time:       {metrics.get('avg_hold_hours', 0):>11.1f}h")

    # Signals filtered
    filtered = metrics.get("signals_filtered", {})
    if filtered:
        print(f"\n--- Signals Filtered ---")
        for reason, count in sorted(filtered.items(), key=lambda x: -x[1]):
            print(f"  {reason:<30} {count:>8,}")

    # Exit reasons
    exit_reasons = metrics.get("exit_reasons", {})
    if exit_reasons:
        print(f"\n--- Exit Reasons ---")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<30} {count:>8,}")

    # Entry price distribution
    dist = metrics.get("entry_price_distribution", {})
    if dist:
        print(f"\n--- Entry Price Distribution ---")
        print(f"  {'Bucket':<12} {'Count':>8} {'PnL':>12} {'% Trades':>10} {'% PnL':>10}")
        print("  " + "-" * 54)
        for label, stats in dist.items():
            print(f"  {label:<12} {stats['count']:>8,} ${stats['pnl']:>+11,.2f} {stats['pct_of_trades']:>9.1%} {stats['pct_of_pnl']:>9.1%}")

    # Top 5 markets
    top5 = metrics.get("top_5_markets", [])
    if top5:
        print(f"\n--- Top 5 Markets by |PnL| ---")
        print(f"  {'Market ID':<20} {'Trades':>8} {'PnL':>12} {'% Total':>10}")
        print("  " + "-" * 52)
        for m in top5:
            mid = m["market_id"][:18]
            print(f"  {mid:<20} {m['trade_count']:>8,} ${m['cumulative_pnl']:>+11,.2f} {m['pct_of_total']:>9.1%}")

    print(f"\n  Runtime: {metrics.get('runtime_seconds', 0):.1f}s")
    print("=" * 70)

    # Sanity checks
    print("\n--- Sanity Checks ---")
    sharpe = metrics.get("sharpe", 0)
    fill_rate = metrics.get("fill_rate", 0)
    avg_hold = metrics.get("avg_hold_seconds", 0)
    top_conc = abs(top5[0]["pct_of_total"]) if top5 else 0

    _check("Sharpe < 5", sharpe < 5, f"{sharpe:.2f}")
    _check("Fill rate < 80%", fill_rate < 0.80, f"{fill_rate:.1%}")
    _check("No market > 15% of PnL", top_conc < 0.15, f"{top_conc:.1%}")
    _check("Avg hold > 30s", avg_hold > 30, f"{avg_hold:.0f}s")
    print()


def _check(label: str, passed: bool, value: str):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label:<35} (actual: {value})")
