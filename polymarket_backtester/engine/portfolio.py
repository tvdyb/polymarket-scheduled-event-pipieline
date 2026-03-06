"""Portfolio tracker: positions, cash, PnL, fills.

Handles Polymarket specifics: binary markets (YES + NO = $1), USDC denomination.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    market_id: str
    side: str       # "YES" or "NO"
    qty: float
    avg_entry: float
    entry_ts: int
    strategy: str = ""


@dataclass
class ClosedTrade:
    market_id: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_ts: int
    exit_ts: int
    pnl: float
    pct_return: float
    strategy: str = ""
    category: str = ""
    hold_time_seconds: int = 0


@dataclass
class EquityPoint:
    timestamp: int
    cash: float
    unrealized_pnl: float
    total_value: float


class Portfolio:
    """Manages positions, cash, and PnL tracking."""

    def __init__(self, initial_cash: float = 10_000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, Position] = {}  # key: f"{market_id}:{side}"
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve: list[EquityPoint] = []
        self.trade_log: list[dict] = []

    def _pos_key(self, market_id: str, side: str) -> str:
        return f"{market_id}:{side}"

    def buy(self, market_id: str, side: str, qty: float, price: float,
            timestamp: int, slippage: float = 0.0, strategy: str = "") -> bool:
        """Open or add to a position. Returns False if insufficient cash."""
        fill_price = price + slippage
        cost = qty * fill_price

        if cost > self.cash:
            return False

        self.cash -= cost
        key = self._pos_key(market_id, side)

        if key in self.positions:
            pos = self.positions[key]
            total_qty = pos.qty + qty
            pos.avg_entry = (pos.avg_entry * pos.qty + fill_price * qty) / total_qty
            pos.qty = total_qty
        else:
            self.positions[key] = Position(
                market_id=market_id, side=side, qty=qty,
                avg_entry=fill_price, entry_ts=timestamp, strategy=strategy,
            )

        self.trade_log.append({
            "timestamp": timestamp, "market_id": market_id,
            "action": "BUY", "side": side, "qty": qty,
            "price": fill_price, "slippage": slippage, "strategy": strategy,
        })
        return True

    def sell(self, market_id: str, side: str, qty: float, price: float,
             timestamp: int, slippage: float = 0.0, category: str = "") -> bool:
        """Close or reduce a position. Returns False if no position."""
        key = self._pos_key(market_id, side)
        pos = self.positions.get(key)
        if not pos or pos.qty <= 0:
            return False

        sell_qty = min(qty, pos.qty)
        fill_price = price - slippage
        proceeds = sell_qty * fill_price
        self.cash += proceeds

        pnl = (fill_price - pos.avg_entry) * sell_qty
        pct = (fill_price - pos.avg_entry) / pos.avg_entry if pos.avg_entry > 0 else 0

        self.closed_trades.append(ClosedTrade(
            market_id=market_id, side=side, qty=sell_qty,
            entry_price=pos.avg_entry, exit_price=fill_price,
            entry_ts=pos.entry_ts, exit_ts=timestamp,
            pnl=pnl, pct_return=pct, strategy=pos.strategy,
            category=category,
            hold_time_seconds=timestamp - pos.entry_ts,
        ))

        pos.qty -= sell_qty
        if pos.qty <= 1e-9:
            del self.positions[key]

        self.trade_log.append({
            "timestamp": timestamp, "market_id": market_id,
            "action": "SELL", "side": side, "qty": sell_qty,
            "price": fill_price, "slippage": slippage, "strategy": pos.strategy,
        })
        return True

    def resolve(self, market_id: str, resolution: str, timestamp: int, category: str = ""):
        """Resolve a market: YES positions pay 1.0 if YES, 0.0 if NO (and vice versa)."""
        for side in ["YES", "NO"]:
            key = self._pos_key(market_id, side)
            pos = self.positions.get(key)
            if not pos:
                continue

            if (side == "YES" and resolution.lower() in ("yes", "true", "1")) or \
               (side == "NO" and resolution.lower() in ("no", "false", "0")):
                exit_price = 1.0
            else:
                exit_price = 0.0

            proceeds = pos.qty * exit_price
            self.cash += proceeds
            pnl = (exit_price - pos.avg_entry) * pos.qty

            self.closed_trades.append(ClosedTrade(
                market_id=market_id, side=side, qty=pos.qty,
                entry_price=pos.avg_entry, exit_price=exit_price,
                entry_ts=pos.entry_ts, exit_ts=timestamp,
                pnl=pnl, pct_return=(exit_price - pos.avg_entry) / pos.avg_entry if pos.avg_entry > 0 else 0,
                strategy=pos.strategy, category=category,
                hold_time_seconds=timestamp - pos.entry_ts,
            ))
            del self.positions[key]

    def unrealized_pnl(self, market_prices: dict[str, float]) -> float:
        """Compute unrealized PnL given current market prices (market_id -> last YES price)."""
        total = 0.0
        for key, pos in self.positions.items():
            market_id = pos.market_id
            yes_price = market_prices.get(market_id, 0.5)
            if pos.side == "YES":
                mark = yes_price
            else:
                mark = 1.0 - yes_price
            total += (mark - pos.avg_entry) * pos.qty
        return total

    def total_value(self, market_prices: dict[str, float]) -> float:
        return self.cash + self.unrealized_pnl(market_prices) + sum(
            (market_prices.get(p.market_id, 0.5) if p.side == "YES" else 1.0 - market_prices.get(p.market_id, 0.5)) * p.qty
            for p in self.positions.values()
        )

    def record_equity(self, timestamp: int, market_prices: dict[str, float]):
        """Snapshot the equity curve."""
        upnl = self.unrealized_pnl(market_prices)
        tv = self.total_value(market_prices)
        self.equity_curve.append(EquityPoint(
            timestamp=timestamp, cash=self.cash,
            unrealized_pnl=upnl, total_value=tv,
        ))

    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)
