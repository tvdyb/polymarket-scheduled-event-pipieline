"""Visualization: equity curves, trade distributions, heatmaps."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..engine.portfolio import ClosedTrade, EquityPoint


def plot_equity_curve(equity: list[EquityPoint], output_path: str, title: str = "Equity Curve"):
    """Plot portfolio value over time."""
    times = [datetime.utcfromtimestamp(e.timestamp) for e in equity]
    values = [e.total_value for e in equity]
    cash = [e.cash for e in equity]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, values, linewidth=1.2, label="Total Value")
    ax.plot(times, cash, linewidth=0.8, alpha=0.5, label="Cash")
    ax.axhline(y=equity[0].total_value, color="gray", linestyle="--", alpha=0.4, label="Initial")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trade_distribution(trades: list[ClosedTrade], output_path: str, title: str = "Trade Distribution"):
    """Plot PnL distribution and cumulative PnL."""
    pnls = [t.pnl for t in trades]
    cum_pnl = np.cumsum(pnls)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(pnls, bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    mean_pnl = np.mean(pnls)
    ax1.axvline(x=mean_pnl, color="green", linestyle="--", alpha=0.7, label=f"Mean: ${mean_pnl:+.2f}")
    ax1.set_title("PnL Distribution")
    ax1.set_xlabel("PnL ($)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative PnL
    ax2 = axes[1]
    ax2.plot(range(len(cum_pnl)), cum_pnl, linewidth=1)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("Cumulative PnL")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Cumulative PnL ($)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_heatmap(data: dict[str, dict], output_path: str, metric: str = "win_rate", title: str = ""):
    """Plot a heatmap from nested dict data (e.g., by_category breakdown)."""
    if not data:
        return

    labels = list(data.keys())
    values = [data[k].get(metric, 0) for k in labels]

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.4)))
    bars = ax.barh(labels, values, color=["green" if v > 0 else "red" for v in values])
    ax.set_xlabel(metric)
    ax.set_title(title or f"By Category: {metric}")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
