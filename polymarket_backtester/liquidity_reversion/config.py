"""All backtest parameters in one place. No magic numbers elsewhere."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    # --- Execution simulation ---
    latency_trades: int = 3          # min trades to skip before fill eligible
    latency_seconds: int = 5         # min wall-clock seconds before fill eligible
    fill_depth_trades: int = 3       # VWAP over this many trades for fill price
    fill_timeout_seconds: int = 300  # cancel unfilled order after 5 min

    # --- Position sizing ---
    sizing_mode: str = "notional"    # "notional" (shares = max_notional) or "cost" (shares = budget / price)
    max_notional: float = 300.0      # max shares (notional mode) or max dollar cost (cost mode)
    volume_cap_pct: float = 0.02     # cap position at % of trailing 1h volume

    # --- Signal detection (impact detector) ---
    impact_threshold: float = 0.08   # min price deviation from VWAP to trigger signal
    low_volume_threshold: float = 50_000.0  # only trade in markets below this 24h volume
    min_vwap_trades: int = 5         # min trades in 1h window for VWAP to be meaningful
    min_trade_size_usd: float = 100.0  # ignore trades below this size for signal detection

    # --- Trade management ---
    max_hold_seconds: int = 14400    # 4h max hold
    reversion_target_pct: float = 0.60  # expect 60% reversion of impact

    # --- Filters ---
    entry_price_min: float = 0.15    # min YES price for entry
    entry_price_max: float = 0.85    # max YES price for entry
    resolution_proximity_hours: float = 4.0  # block entries within N hours of close
    forced_exit_hours: float = 1.0   # force-exit open positions within N hours of close

    # --- Risk / concentration limits ---
    max_positions_per_market: int = 3
    max_notional_per_market: float = 2_000.0
    max_total_positions: int = 20

    # --- Costs ---
    transaction_cost_pct: float = 0.01  # 1% round-trip estimate
    exit_spread_cents: float = 0.02    # spread penalty applied on exit fills

    # --- Portfolio ---
    initial_cash: float = 10_000.0

    # --- Data ---
    start_date: str = "2024-07-01"
    end_date: str = "2024-10-01"
