"""Latency-aware fill simulation.

After a signal fires at trade i, the order cannot fill until i+k trades
AND latency_seconds have elapsed. Fill price is the VWAP of the next N
trades on the same market/token after the latency window opens. If fewer
than N trades arrive within timeout_seconds, the order is cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import BacktestConfig


@dataclass
class PendingOrder:
    signal_time: int           # timestamp of triggering trade
    signal_trade_index: int    # global trade counter at signal time
    market_id: str
    side: str                  # "YES" or "NO"
    signal_price: float        # price at signal time (YES price)
    direction: str             # "BUY" or "SELL"
    max_shares: float          # requested quantity
    target_price: float        # reversion target (for BUY orders carried through)
    # Accumulator for fill VWAP after latency window opens
    _fill_trades: list = field(default_factory=list)  # [(price, size)]
    _trades_since_signal: int = 0
    _latency_satisfied: bool = False


@dataclass
class Fill:
    signal_time: int
    fill_time: int
    market_id: str
    side: str
    direction: str
    signal_price: float
    fill_price: float          # VWAP of fill trades
    slippage_bps: float
    shares: float
    target_price: float


class FillSimulator:
    """Simulates realistic order execution with latency and fill depth."""

    def __init__(self, config: BacktestConfig):
        self.latency_trades = config.latency_trades
        self.latency_seconds = config.latency_seconds
        self.fill_depth = config.fill_depth_trades
        self.timeout = config.fill_timeout_seconds
        self._pending: list[PendingOrder] = []
        self._trade_counter: int = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def submit(self, signal_time: int, market_id: str, side: str,
               direction: str, signal_price: float, max_shares: float,
               target_price: float = 0.0):
        self._pending.append(PendingOrder(
            signal_time=signal_time,
            signal_trade_index=self._trade_counter,
            market_id=market_id,
            side=side,
            signal_price=signal_price,
            direction=direction,
            max_shares=max_shares,
            target_price=target_price,
        ))

    def on_trade(self, timestamp: int, market_id: str, price: float,
                 size: float) -> tuple[list[Fill], list[PendingOrder]]:
        """Process a new observed trade. Returns (fills, cancelled_orders)."""
        self._trade_counter += 1
        fills: list[Fill] = []
        cancelled: list[PendingOrder] = []
        still_pending: list[PendingOrder] = []

        for order in self._pending:
            if order.market_id != market_id:
                # Check timeout even for non-matching markets
                if timestamp - order.signal_time > self.timeout:
                    cancelled.append(order)
                else:
                    still_pending.append(order)
                continue

            order._trades_since_signal += 1

            # Check if latency window is satisfied
            trades_elapsed = self._trade_counter - order.signal_trade_index
            time_elapsed = timestamp - order.signal_time
            if trades_elapsed >= self.latency_trades and time_elapsed >= self.latency_seconds:
                order._latency_satisfied = True

            # Accumulate fill trades after latency is satisfied
            if order._latency_satisfied:
                order._fill_trades.append((price, size))

            # Check if we have enough fill depth
            if len(order._fill_trades) >= self.fill_depth:
                fill = self._execute_fill(order, timestamp)
                fills.append(fill)
                continue

            # Check timeout
            if timestamp - order.signal_time > self.timeout:
                cancelled.append(order)
                continue

            still_pending.append(order)

        self._pending = still_pending
        return fills, cancelled

    def _execute_fill(self, order: PendingOrder, fill_time: int) -> Fill:
        trades = order._fill_trades[:self.fill_depth]
        total_size = sum(s for _, s in trades)
        if total_size > 0:
            vwap_yes = sum(p * s for p, s in trades) / total_size
        else:
            vwap_yes = trades[0][0] if trades else order.signal_price

        # Convert to the correct side price
        if order.side == "NO":
            fill_price = 1.0 - vwap_yes
        else:
            fill_price = vwap_yes

        fill_price = max(fill_price, 0.001)

        slippage_bps = abs(fill_price - order.signal_price) / order.signal_price * 10_000 if order.signal_price > 0 else 0

        return Fill(
            signal_time=order.signal_time,
            fill_time=fill_time,
            market_id=order.market_id,
            side=order.side,
            direction=order.direction,
            signal_price=order.signal_price,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            shares=order.max_shares,
            target_price=order.target_price,
        )

    def cancel_all_for_market(self, market_id: str) -> list[PendingOrder]:
        """Cancel all pending orders for a market. Returns cancelled orders."""
        cancelled = [o for o in self._pending if o.market_id == market_id]
        self._pending = [o for o in self._pending if o.market_id != market_id]
        return cancelled
