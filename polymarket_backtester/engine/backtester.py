"""Core backtest engine: event-driven iteration through trade data.

Iterates chronologically through trades, updates market state,
calls active strategy hooks, manages portfolio.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import polars as pl
from tqdm import tqdm

from .market_state import MarketState, TradeEvent
from .portfolio import Portfolio
from .metrics import compute_metrics

if TYPE_CHECKING:
    from ..strategies.base import Strategy


class Backtester:
    """Event-driven backtester that replays historical trades."""

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: float = 10_000.0,
        slippage_cents: float = 0.01,
        max_position_usd: float = 500.0,
        max_open_positions: int = 20,
        tick_interval_seconds: int = 300,  # on_tick every 5 min
        equity_snapshot_interval: int = 3600,  # snapshot every 1h
    ):
        self.strategy = strategy
        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.market_state = MarketState()
        self.slippage_cents = slippage_cents
        self.max_position_usd = max_position_usd
        self.max_open_positions = max_open_positions
        self.tick_interval = tick_interval_seconds
        self.equity_interval = equity_snapshot_interval

    def load_markets(self, markets_df: pl.DataFrame):
        """Register market metadata with market_state."""
        for row in markets_df.iter_rows(named=True):
            market_id = str(row.get("id") or row.get("condition_id") or "")
            resolution_ts = None
            closed_time = row.get("closedTime") or row.get("endDate")
            if closed_time:
                try:
                    from datetime import datetime, timezone
                    ct = str(closed_time).strip()
                    # Strip timezone suffix, parse as UTC
                    for tz_suffix in ("+00:00", "+00", "Z"):
                        if ct.endswith(tz_suffix):
                            ct = ct[:-len(tz_suffix)]
                            break
                    # Parse up to seconds (ignore fractional)
                    dt = datetime.strptime(ct[:19], "%Y-%m-%d %H:%M:%S")
                    resolution_ts = int(dt.replace(tzinfo=timezone.utc).timestamp())
                except Exception:
                    pass

            resolution = row.get("resolution")
            category = row.get("category") or ""

            self.market_state.register_market(
                market_id=market_id,
                resolution_ts=resolution_ts,
                category=category,
                resolution=str(resolution) if resolution else None,
            )

        self.strategy.on_init(self.market_state, self.portfolio)

    def run(self, trades_df: pl.DataFrame, show_progress: bool = True) -> dict:
        """Run the backtest over a trades DataFrame.

        Expects columns: market (str), timestamp (int), price (float),
        size (float), taker_side (str), outcome (str).
        Optional: maker (str), taker (str).
        """
        start_time = time.time()

        # Ensure sorted by timestamp
        trades = trades_df.sort("timestamp")

        last_tick_ts = 0
        last_equity_ts = 0
        total_rows = len(trades)

        iterator = trades.iter_rows(named=True)
        if show_progress:
            iterator = tqdm(iterator, total=total_rows, desc="Backtesting")

        for row in iterator:
            ts = int(row["timestamp"])
            market_id = str(row.get("market") or row.get("market_id") or "")
            price = float(row.get("price", 0))
            size = float(row.get("size", 0))
            taker_side = str(row.get("taker_side") or row.get("side") or "")
            outcome = str(row.get("outcome") or "Yes")
            maker = str(row.get("maker") or "")
            taker_addr = str(row.get("taker") or "")

            if not market_id or price <= 0:
                continue

            # Build trade event
            trade_event = TradeEvent(
                market_id=market_id, timestamp=ts,
                price=price, size=size, taker_side=taker_side,
                outcome=outcome, maker=maker, taker=taker_addr,
            )

            # Update market state
            snapshot = self.market_state.on_trade(trade_event)

            # Call strategy on_trade
            signals = self.strategy.on_trade(trade_event, snapshot, self.portfolio)
            self._execute_signals(signals, ts)

            # Periodic tick
            if ts - last_tick_ts >= self.tick_interval:
                tick_signals = self.strategy.on_tick(ts, self.market_state, self.portfolio)
                self._execute_signals(tick_signals, ts)
                last_tick_ts = ts

            # Equity snapshot
            if ts - last_equity_ts >= self.equity_interval:
                prices = {mid: s.last_price for mid, s in self.market_state.get_all_snapshots().items()}
                self.portfolio.record_equity(ts, prices)
                last_equity_ts = ts

        # Resolve all remaining positions at actual outcomes
        self._resolve_all()

        # Final equity point
        prices = {mid: s.last_price for mid, s in self.market_state.get_all_snapshots().items()}
        self.portfolio.record_equity(int(time.time()), prices)

        # Compute metrics
        metrics = compute_metrics(self.portfolio)
        elapsed = time.time() - start_time
        metrics["backtest_runtime_seconds"] = elapsed
        metrics["total_trade_events"] = total_rows

        return metrics

    def _execute_signals(self, signals: list[dict], timestamp: int):
        """Execute trading signals from strategy."""
        if not signals:
            return

        for sig in signals:
            action = sig.get("action", "").upper()
            market_id = sig.get("market_id", "")
            side = sig.get("side", "YES")
            qty = sig.get("qty", 0)
            price = sig.get("price", 0)

            if not market_id or qty <= 0 or price <= 0:
                continue

            # Position size limit
            pos_value = qty * price
            if pos_value > self.max_position_usd:
                qty = self.max_position_usd / price

            slippage = self.slippage_cents

            if action == "BUY":
                # Limit concurrent open positions
                if len(self.portfolio.positions) >= self.max_open_positions:
                    continue
                self.portfolio.buy(
                    market_id, side, qty, price, timestamp,
                    slippage=slippage, strategy=self.strategy.name,
                )
            elif action == "SELL":
                snap = self.market_state.get_snapshot(market_id)
                cat = snap.category if snap else ""
                self.portfolio.sell(
                    market_id, side, qty, price, timestamp,
                    slippage=slippage, category=cat,
                )

    def _resolve_all(self):
        """Resolve all open positions using registered market outcomes."""
        open_markets = set()
        for key in list(self.portfolio.positions.keys()):
            market_id = key.split(":")[0]
            open_markets.add(market_id)

        for market_id in open_markets:
            snap = self.market_state.get_snapshot(market_id)
            if snap and snap.resolution:
                self.portfolio.resolve(
                    market_id, snap.resolution,
                    timestamp=snap.resolution_ts or snap.last_trade_ts,
                    category=snap.category,
                )
