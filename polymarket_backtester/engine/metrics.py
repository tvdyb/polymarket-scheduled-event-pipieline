"""Performance metrics: Sharpe, drawdown, win rate, PnL breakdown."""

from __future__ import annotations

import math
from collections import defaultdict

from .portfolio import Portfolio, ClosedTrade, EquityPoint


def compute_metrics(portfolio: Portfolio) -> dict:
    """Compute all performance metrics from a completed backtest."""
    trades = portfolio.closed_trades
    equity = portfolio.equity_curve

    result = {
        "total_trades": len(trades),
        "initial_cash": portfolio.initial_cash,
        "final_cash": portfolio.cash,
        "realized_pnl": portfolio.realized_pnl,
        "open_positions": len(portfolio.positions),
    }

    if not trades:
        return result

    # Win/loss
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    result["win_rate"] = len(winners) / len(trades)
    result["avg_winner"] = sum(t.pnl for t in winners) / len(winners) if winners else 0
    result["avg_loser"] = sum(t.pnl for t in losers) / len(losers) if losers else 0
    result["best_trade"] = max(t.pnl for t in trades)
    result["worst_trade"] = min(t.pnl for t in trades)
    result["avg_hold_seconds"] = sum(t.hold_time_seconds for t in trades) / len(trades)
    result["avg_hold_hours"] = result["avg_hold_seconds"] / 3600

    # Returns
    returns = [t.pct_return for t in trades]
    result["mean_return"] = sum(returns) / len(returns)
    result["median_return"] = sorted(returns)[len(returns) // 2]

    # Sharpe (annualized, 365 days)
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1e-9
        # Approximate trades per year
        if len(trades) >= 2:
            span_seconds = trades[-1].exit_ts - trades[0].entry_ts
            if span_seconds > 0:
                trades_per_year = len(trades) / (span_seconds / 86400 / 365)
            else:
                trades_per_year = len(trades)
        else:
            trades_per_year = len(trades)
        result["sharpe"] = (mean_r / std_r) * math.sqrt(trades_per_year)
    else:
        result["sharpe"] = 0.0

    # Max drawdown from equity curve
    if equity:
        result.update(_compute_drawdown(equity))

    # Turnover
    total_traded = sum(abs(t.qty * t.entry_price) + abs(t.qty * t.exit_price) for t in trades)
    avg_value = sum(e.total_value for e in equity) / len(equity) if equity else portfolio.initial_cash
    result["turnover"] = total_traded / avg_value if avg_value > 0 else 0

    # Breakdown by category
    result["by_category"] = _breakdown_by_field(trades, "category")

    # Breakdown by hold time buckets
    result["by_hold_time"] = _breakdown_by_hold_time(trades)

    return result


def _compute_drawdown(equity: list[EquityPoint]) -> dict:
    peak = equity[0].total_value
    max_dd = 0.0
    max_dd_pct = 0.0

    for e in equity:
        if e.total_value > peak:
            peak = e.total_value
        dd = peak - e.total_value
        dd_pct = dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    return {"max_drawdown": max_dd, "max_drawdown_pct": max_dd_pct}


def _breakdown_by_field(trades: list[ClosedTrade], field: str) -> dict[str, dict]:
    groups: dict[str, list[ClosedTrade]] = defaultdict(list)
    for t in trades:
        key = getattr(t, field, "unknown") or "unknown"
        groups[key].append(t)

    result = {}
    for key, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        winners = [t for t in group if t.pnl > 0]
        result[key] = {
            "trades": len(group),
            "win_rate": len(winners) / len(group),
            "total_pnl": sum(t.pnl for t in group),
            "mean_return": sum(t.pct_return for t in group) / len(group),
        }
    return result


def _breakdown_by_hold_time(trades: list[ClosedTrade]) -> dict[str, dict]:
    buckets = {
        "<1h": (0, 3600),
        "1-6h": (3600, 21600),
        "6-24h": (21600, 86400),
        "1-7d": (86400, 604800),
        ">7d": (604800, float("inf")),
    }
    groups: dict[str, list[ClosedTrade]] = {k: [] for k in buckets}
    for t in trades:
        for label, (lo, hi) in buckets.items():
            if lo <= t.hold_time_seconds < hi:
                groups[label].append(t)
                break

    result = {}
    for label, group in groups.items():
        if group:
            winners = [t for t in group if t.pnl > 0]
            result[label] = {
                "trades": len(group),
                "win_rate": len(winners) / len(group),
                "total_pnl": sum(t.pnl for t in group),
            }
    return result


def print_metrics(metrics: dict):
    """Pretty-print backtest metrics to console."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"Total trades:      {metrics.get('total_trades', 0):>8}")
    print(f"Win rate:          {metrics.get('win_rate', 0):>7.1%}")
    print(f"Realized PnL:      ${metrics.get('realized_pnl', 0):>+10.2f}")
    print(f"Mean return:       {metrics.get('mean_return', 0):>+7.1%}")
    print(f"Sharpe:            {metrics.get('sharpe', 0):>8.2f}")
    print(f"Max drawdown:      {metrics.get('max_drawdown_pct', 0):>7.1%}")
    print(f"Avg hold time:     {metrics.get('avg_hold_hours', 0):>7.1f}h")
    print(f"Best trade:        ${metrics.get('best_trade', 0):>+10.2f}")
    print(f"Worst trade:       ${metrics.get('worst_trade', 0):>+10.2f}")
    print(f"Avg winner:        ${metrics.get('avg_winner', 0):>+10.2f}")
    print(f"Avg loser:         ${metrics.get('avg_loser', 0):>+10.2f}")
    print(f"Turnover:          {metrics.get('turnover', 0):>8.1f}x")

    by_cat = metrics.get("by_category", {})
    if by_cat:
        print("\n--- By Category ---")
        print(f"{'Category':<20} {'Trades':>7} {'Win%':>7} {'PnL':>10}")
        print("-" * 48)
        for cat, stats in by_cat.items():
            print(f"{cat:<20} {stats['trades']:>7} {stats['win_rate']:>6.1%} ${stats['total_pnl']:>+9.2f}")

    by_hold = metrics.get("by_hold_time", {})
    if by_hold:
        print("\n--- By Hold Time ---")
        print(f"{'Bucket':<12} {'Trades':>7} {'Win%':>7} {'PnL':>10}")
        print("-" * 40)
        for bucket, stats in by_hold.items():
            print(f"{bucket:<12} {stats['trades']:>7} {stats['win_rate']:>6.1%} ${stats['total_pnl']:>+9.2f}")
