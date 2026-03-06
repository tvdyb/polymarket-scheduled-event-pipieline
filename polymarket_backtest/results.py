"""Phase 5: Results output — console summary, CSV export, charts."""

import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR


def _compute_stats(trades: list[dict]) -> dict:
    """Compute summary statistics for a list of trades."""
    n = len(trades)
    if n == 0:
        return {"trade_count": 0}

    returns = [t["pct_return"] for t in trades]
    mean_ret = sum(returns) / n
    median_ret = sorted(returns)[n // 2]
    win_rate = sum(1 for r in returns if r > 0) / n
    std_ret = (sum((r - mean_ret) ** 2 for r in returns) / n) ** 0.5 if n > 1 else 0
    sharpe = (mean_ret / std_ret) * sqrt(252) if std_ret > 0 else 0  # annualized
    worst = min(returns)
    best = max(returns)

    return {
        "trade_count": n,
        "win_rate": win_rate,
        "mean_return": mean_ret,
        "median_return": median_ret,
        "std_return": std_ret,
        "sharpe_annualized": sharpe,
        "worst_trade": worst,
        "best_trade": best,
    }


def print_console_summary(
    all_results: dict[str, list[dict]],
    filter_stats: dict,
    price_stats: dict,
    highlight_combo: str = "(7,1)",
):
    """Print the formatted console summary."""
    print("\n" + "=" * 60)
    print("=== BACKTEST RESULTS ===")
    print("Strategy: Buy hard-dated niche event markets (text-classified only, no vol filter)")
    print("=" * 60)

    print(f"\nTotal markets fetched:      {filter_stats.get('total_markets', '?'):>8,}")
    print(f"After sports/volume filter: {filter_stats.get('after_hard_filter', '?'):>8,}")
    print(f"After LLM filter:           {filter_stats.get('after_llm_filter', '?'):>8,}")
    print(f"With sufficient price data: {price_stats.get('markets_with_prices', '?'):>8,}")

    # Show all combos
    print("\n--- Performance by (Entry Days, Exit Days) ---")
    print(f"{'Combo':<12} {'Trades':>7} {'Win%':>7} {'Mean':>8} {'Median':>8} {'Std':>8} {'Sharpe':>8} {'Worst':>9} {'Best':>9}")
    print("-" * 90)

    for combo_key, trades in sorted(all_results.items()):
        stats = _compute_stats(trades)
        if stats["trade_count"] == 0:
            continue
        marker = " <<<" if combo_key == highlight_combo else ""
        print(
            f"{combo_key:<12} {stats['trade_count']:>7} "
            f"{stats['win_rate']:>6.1%} "
            f"{stats['mean_return']:>+7.1%} "
            f"{stats['median_return']:>+7.1%} "
            f"{stats['std_return']:>7.1%} "
            f"{stats['sharpe_annualized']:>8.2f} "
            f"{stats['worst_trade']:>+8.1%} "
            f"{stats['best_trade']:>+8.1%}"
            f"{marker}"
        )

    # Detailed breakdown for highlighted combo
    if highlight_combo in all_results:
        trades = all_results[highlight_combo]
        if trades:
            _print_event_type_breakdown(trades, highlight_combo)
            _print_vol_diagnostic(trades)


def _print_event_type_breakdown(trades: list[dict], combo: str):
    """Print performance breakdown by event type."""
    by_type = defaultdict(list)
    for t in trades:
        by_type[t.get("event_type", "unknown")].append(t)

    print(f"\n--- By Event Type ({combo}) ---")
    print(f"{'Type':<20} {'Trades':>7} {'Win%':>7} {'Mean Return':>12}")
    print("-" * 50)

    for etype, type_trades in sorted(by_type.items(), key=lambda x: -len(x[1])):
        stats = _compute_stats(type_trades)
        print(
            f"{etype:<20} {stats['trade_count']:>7} "
            f"{stats['win_rate']:>6.1%} "
            f"{stats['mean_return']:>+11.1%}"
        )


def _print_vol_diagnostic(trades: list[dict]):
    """Print observed volatility diagnostic (post-hoc, NOT used for filtering)."""
    trades_with_vol = [t for t in trades if t.get("observed_pre_entry_std") is not None]
    if not trades_with_vol:
        return

    print("\n--- Observed Vol Diagnostic (post-hoc, not used for filtering) ---")

    low_vol = [t for t in trades_with_vol if t["observed_pre_entry_std"] < 0.05]
    high_vol = [t for t in trades_with_vol if t["observed_pre_entry_std"] > 0.15]

    pct_low = len(low_vol) / len(trades_with_vol) if trades_with_vol else 0
    pct_high = len(high_vol) / len(trades_with_vol) if trades_with_vol else 0

    print(f"% of selected trades where pre-entry std < 0.05:   {pct_low:.0%}")
    print(f"% of selected trades where pre-entry std > 0.15:   {pct_high:.0%}")

    low_stats = _compute_stats(low_vol) if low_vol else {"sharpe_annualized": 0}
    high_stats = _compute_stats(high_vol) if high_vol else {"sharpe_annualized": 0}

    print(f"Sharpe when pre-entry std < 0.05:   {low_stats['sharpe_annualized']:.2f}")
    print(f"Sharpe when pre-entry std > 0.15:   {high_stats['sharpe_annualized']:.2f}")

    if pct_low > 0:
        print(f"-> LLM correctly identified low-vol markets ~{pct_low:.0%} of the time.")


def export_csv(all_results: dict[str, list[dict]], combo: str = "(7,1)"):
    """Export trade-level CSV for a given (N,M) combo."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades = all_results.get(combo, [])
    if not trades:
        print(f"No trades for combo {combo}, skipping CSV export")
        return

    csv_path = OUTPUT_DIR / f"trades_{combo.replace('(','').replace(')','').replace(',','_')}.csv"
    fieldnames = [
        "market_id", "question", "event_type", "event_date",
        "entry_date", "exit_date", "exit_reason", "entry_price", "exit_price",
        "pnl", "pct_return", "resolution_outcome",
        "observed_pre_entry_std", "llm_confidence", "llm_reasoning", "category",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)

    print(f"CSV exported: {csv_path} ({len(trades)} trades)")

    # Also export all combos summary
    summary_path = OUTPUT_DIR / "backtest_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "combo", "trade_count", "win_rate", "mean_return", "median_return",
            "std_return", "sharpe_annualized", "worst_trade", "best_trade",
        ])
        writer.writeheader()
        for combo_key, trades in sorted(all_results.items()):
            stats = _compute_stats(trades)
            if stats["trade_count"] > 0:
                writer.writerow({"combo": combo_key, **stats})

    print(f"Summary CSV exported: {summary_path}")


def generate_charts(all_results: dict[str, list[dict]], combo: str = "(7,1)"):
    """Generate matplotlib charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades = all_results.get(combo, [])
    if not trades:
        print(f"No trades for combo {combo}, skipping charts")
        return

    # Sort by entry date for cumulative curve
    sorted_trades = sorted(trades, key=lambda t: t.get("entry_date") or "")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Low-Vol Event Backtest Results — Entry: {combo}", fontsize=14)

    # 1. Cumulative return curve
    ax1 = axes[0, 0]
    cumulative = []
    cum_val = 1.0
    for t in sorted_trades:
        cum_val *= (1 + t["pct_return"])
        cumulative.append(cum_val)
    ax1.plot(range(len(cumulative)), cumulative, linewidth=1)
    ax1.set_title("Cumulative Return (Compounded)")
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Portfolio Value ($1 start)")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # 2. Return distribution histogram
    ax2 = axes[0, 1]
    returns = [t["pct_return"] for t in trades]
    ax2.hist(returns, bins=50, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    mean_r = sum(returns) / len(returns)
    ax2.axvline(x=mean_r, color="green", linestyle="--", alpha=0.7, label=f"Mean: {mean_r:+.1%}")
    ax2.set_title("Return Distribution")
    ax2.set_xlabel("Return (%)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Scatter: observed pre-entry vol vs return
    ax3 = axes[1, 0]
    vol_trades = [(t["observed_pre_entry_std"], t["pct_return"]) for t in trades
                  if t.get("observed_pre_entry_std") is not None]
    if vol_trades:
        vols, rets = zip(*vol_trades)
        ax3.scatter(vols, rets, alpha=0.4, s=15)
        ax3.set_title("Pre-Entry Vol vs Return\n(validates LLM classification)")
        ax3.set_xlabel("Observed Pre-Entry Std Dev")
        ax3.set_ylabel("Return (%)")
        ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No vol data available", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Pre-Entry Vol vs Return")

    # 4. Heatmap: win rate across all (N, M) combos
    ax4 = axes[1, 1]
    entry_vals = sorted(set(int(k.split(",")[0].strip("(")) for k in all_results.keys()))
    exit_vals = sorted(set(int(k.split(",")[1].strip(")")) for k in all_results.keys()))

    if entry_vals and exit_vals:
        heatmap_data = np.full((len(exit_vals), len(entry_vals)), np.nan)
        for i, m_exit in enumerate(exit_vals):
            for j, n_entry in enumerate(entry_vals):
                key = f"({n_entry},{m_exit})"
                combo_trades = all_results.get(key, [])
                if combo_trades:
                    stats = _compute_stats(combo_trades)
                    heatmap_data[i, j] = stats["win_rate"] * 100

        im = ax4.imshow(heatmap_data, aspect="auto", cmap="RdYlGn", vmin=30, vmax=70)
        ax4.set_xticks(range(len(entry_vals)))
        ax4.set_xticklabels([str(v) for v in entry_vals])
        ax4.set_yticks(range(len(exit_vals)))
        ax4.set_yticklabels([str(v) for v in exit_vals])
        ax4.set_xlabel("Entry (days before)")
        ax4.set_ylabel("Exit (days before)")
        ax4.set_title("Win Rate Heatmap (%)")

        # Annotate cells
        for i in range(len(exit_vals)):
            for j in range(len(entry_vals)):
                val = heatmap_data[i, j]
                if not np.isnan(val):
                    ax4.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9)

        plt.colorbar(im, ax=ax4)
    else:
        ax4.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Win Rate Heatmap")

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "backtest_charts.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts saved: {chart_path}")
