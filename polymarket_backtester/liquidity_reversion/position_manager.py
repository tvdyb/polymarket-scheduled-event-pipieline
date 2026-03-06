"""Position tracking, risk limits, and exit logic."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import BacktestConfig
from .fill_simulator import Fill


@dataclass
class OpenPosition:
    market_id: str
    side: str
    shares: float
    entry_price: float         # fill price (side-adjusted)
    entry_time: int            # fill timestamp
    signal_time: int           # original signal timestamp
    signal_price: float
    target_price: float        # YES-price reversion target
    entry_notional: float      # shares * entry_price


@dataclass
class ClosedPosition:
    market_id: str
    side: str
    shares: float
    signal_time: int
    signal_price: float
    fill_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    pnl: float
    hold_seconds: int
    exit_reason: str           # target_hit, stop_loss, timeout, forced_resolution_exit, market_close
    slippage_bps: float
    entry_notional: float


class PositionManager:
    """Manages open positions, enforces risk limits, handles exits."""

    def __init__(self, config: BacktestConfig):
        self.max_hold = config.max_hold_seconds
        self.max_per_market = config.max_positions_per_market
        self.max_notional_per_market = config.max_notional_per_market
        self.max_total = config.max_total_positions
        self.forced_exit_seconds = config.forced_exit_hours * 3600

        self._positions: dict[str, list[OpenPosition]] = {}  # market_id -> [positions]
        self._closed: list[ClosedPosition] = []
        self._cash: float = config.initial_cash
        self.initial_cash = config.initial_cash

    @property
    def open_positions(self) -> list[OpenPosition]:
        return [p for positions in self._positions.values() for p in positions]

    @property
    def closed_trades(self) -> list[ClosedPosition]:
        return self._closed

    @property
    def total_open(self) -> int:
        return sum(len(ps) for ps in self._positions.values())

    @property
    def cash(self) -> float:
        return self._cash

    def market_position_count(self, market_id: str) -> int:
        return len(self._positions.get(market_id, []))

    def market_notional(self, market_id: str) -> float:
        return sum(p.entry_notional for p in self._positions.get(market_id, []))

    def can_open(self, market_id: str, cost: float) -> str | None:
        """Check if opening a position is allowed. Returns rejection reason or None."""
        if self.total_open >= self.max_total:
            return "max_total_positions"
        if self.market_position_count(market_id) >= self.max_per_market:
            return "max_positions_per_market"
        if self.market_notional(market_id) + cost > self.max_notional_per_market:
            return "max_notional_per_market"
        if cost > self._cash:
            return "insufficient_cash"
        return None

    def open_position(self, fill: Fill) -> OpenPosition:
        """Open a new position from a fill."""
        cost = fill.shares * fill.fill_price
        self._cash -= cost

        pos = OpenPosition(
            market_id=fill.market_id,
            side=fill.side,
            shares=fill.shares,
            entry_price=fill.fill_price,
            entry_time=fill.fill_time,
            signal_time=fill.signal_time,
            signal_price=fill.signal_price,
            target_price=fill.target_price,
            entry_notional=cost,
        )

        if fill.market_id not in self._positions:
            self._positions[fill.market_id] = []
        self._positions[fill.market_id].append(pos)
        return pos

    def check_exits(self, timestamp: int, market_id: str, yes_price: float,
                    resolution_ts: int | None) -> list[ClosedPosition]:
        """Check all positions in a market for exit conditions."""
        if market_id not in self._positions:
            return []

        exits: list[ClosedPosition] = []
        remaining: list[OpenPosition] = []

        for pos in self._positions[market_id]:
            reason = self._should_exit(pos, timestamp, yes_price, resolution_ts)
            if reason:
                exit_price = (1.0 - yes_price) if pos.side == "NO" else yes_price
                exit_price = max(exit_price, 0.0)
                pnl = (exit_price - pos.entry_price) * pos.shares
                proceeds = pos.shares * exit_price
                self._cash += proceeds

                closed = ClosedPosition(
                    market_id=pos.market_id,
                    side=pos.side,
                    shares=pos.shares,
                    signal_time=pos.signal_time,
                    signal_price=pos.signal_price,
                    fill_time=pos.entry_time,
                    entry_price=pos.entry_price,
                    exit_time=timestamp,
                    exit_price=exit_price,
                    pnl=pnl,
                    hold_seconds=timestamp - pos.entry_time,
                    exit_reason=reason,
                    slippage_bps=abs(pos.entry_price - pos.signal_price) / pos.signal_price * 10_000 if pos.signal_price > 0 else 0,
                    entry_notional=pos.entry_notional,
                )
                self._closed.append(closed)
                exits.append(closed)
            else:
                remaining.append(pos)

        if remaining:
            self._positions[market_id] = remaining
        elif market_id in self._positions:
            del self._positions[market_id]

        return exits

    def _should_exit(self, pos: OpenPosition, timestamp: int, yes_price: float,
                     resolution_ts: int | None) -> str | None:
        """Determine if a position should exit and why."""
        # Forced exit near resolution
        if resolution_ts and resolution_ts - timestamp < self.forced_exit_seconds:
            return "forced_resolution_exit"

        # Reversion target hit (target is in YES-price space)
        if pos.side == "YES" and yes_price >= pos.target_price:
            return "target_hit"
        if pos.side == "NO" and yes_price <= pos.target_price:
            return "target_hit"

        # Time expiry
        if timestamp - pos.entry_time > self.max_hold:
            return "timeout"

        return None

    def force_close_all(self, timestamp: int, market_prices: dict[str, float]) -> list[ClosedPosition]:
        """Force-close all remaining positions at current prices."""
        exits: list[ClosedPosition] = []
        for market_id in list(self._positions.keys()):
            yes_price = market_prices.get(market_id, 0.5)
            exits.extend(self.check_exits(timestamp, market_id, yes_price, resolution_ts=0))
        return exits
