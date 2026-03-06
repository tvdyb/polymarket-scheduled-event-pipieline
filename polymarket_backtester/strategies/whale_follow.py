"""Strategy 2: Whale Follow with Lag Filter.

Hypothesis: Top PnL wallets have informational edges. Following their trades
with a delay captures alpha, but only in low-volume markets where their
signal isn't already priced in.
"""

from __future__ import annotations

from collections import deque

from ..engine.market_state import MarketState, MarketSnapshot, TradeEvent
from ..engine.portfolio import Portfolio
from .base import Strategy


class WhaleFollowStrategy(Strategy):
    name = "whale_follow"

    def __init__(
        self,
        whale_addresses: set[str] | None = None,
        min_whale_size: float = 5_000.0,
        follow_delay_seconds: int = 900,    # 15 min
        take_profit: float = 0.05,
        stop_loss: float = 0.05,
        max_market_volume: float = 200_000.0,
        max_position_usd: float = 300.0,
    ):
        self.whale_addrs = whale_addresses or set()
        self.min_size = min_whale_size
        self.follow_delay = follow_delay_seconds
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_volume = max_market_volume
        self.max_position_usd = max_position_usd

        # Pending whale signals: deque of {market_id, side, price, ts, whale}
        self._pending: deque[dict] = deque()
        # Active positions we're tracking for exit
        self._active: dict[str, dict] = {}  # market_id -> {entry_price, side, entry_ts}

    def on_trade(self, trade: TradeEvent, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[dict]:
        signals = []

        # Check exits on active positions
        if trade.market_id in self._active:
            active = self._active[trade.market_id]
            entry = active["entry_price"]
            side = active["side"]

            if side == "YES":
                move = trade.price - entry
            else:
                move = (1.0 - trade.price) - entry

            if move >= self.take_profit or move <= -self.stop_loss:
                signals.append({
                    "action": "SELL",
                    "market_id": trade.market_id,
                    "side": side,
                    "qty": active["qty"],
                    "price": trade.price if side == "YES" else 1.0 - trade.price,
                })
                del self._active[trade.market_id]

        # Detect whale trades by address OR by size alone
        is_whale = (trade.taker in self.whale_addrs or trade.maker in self.whale_addrs) if self.whale_addrs else True
        if is_whale and trade.size >= self.min_size:
                whale_side = "YES" if trade.taker_side == "BUY" else "NO"
                self._pending.append({
                    "market_id": trade.market_id,
                    "side": whale_side,
                    "price_at_signal": trade.price,
                    "ts": trade.timestamp,
                })

        # Check if any pending signals are ready (delay elapsed)
        while self._pending and (trade.timestamp - self._pending[0]["ts"] >= self.follow_delay):
            sig = self._pending.popleft()
            mid = sig["market_id"]

            if mid in self._active:
                continue

            snap = snapshot if mid == trade.market_id else None
            if not snap:
                continue

            # Check if price already moved (signal priced in)
            price_move = abs(snap.last_price - sig["price_at_signal"])
            if price_move > 0.02:
                continue  # already priced in, skip

            # Volume filter
            if snap.volume_24h > self.max_volume:
                continue

            side = sig["side"]
            entry_price = snap.last_price if side == "YES" else 1.0 - snap.last_price
            if entry_price <= 0:
                continue

            qty = self.max_position_usd / entry_price

            signals.append({
                "action": "BUY",
                "market_id": mid,
                "side": side,
                "qty": qty,
                "price": entry_price,
            })
            self._active[mid] = {
                "entry_price": entry_price,
                "side": side,
                "qty": qty,
                "entry_ts": trade.timestamp,
            }

        return signals
