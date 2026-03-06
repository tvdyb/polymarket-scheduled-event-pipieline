"""Strategy 5: Resolution Catalyst Timing / Dead Zone Liquidity.

Hypothesis: 24-48h before known resolution events, liquidity dries up and
bid-ask spreads widen. Providing liquidity (posting limit orders) earns
a systematic spread premium.
"""

from __future__ import annotations

from ..engine.market_state import MarketState, MarketSnapshot, TradeEvent
from ..engine.portfolio import Portfolio
from .base import Strategy


class ResolutionCatalystStrategy(Strategy):
    name = "resolution_catalyst"

    def __init__(
        self,
        entry_hours_before: float = 48.0,
        exit_hours_before: float = 2.0,
        half_spread: float = 0.03,
        max_inventory: float = 1000.0,
        max_position_usd: float = 300.0,
    ):
        self.entry_hours = entry_hours_before
        self.exit_hours = exit_hours_before
        self.half_spread = half_spread
        self.max_inventory = max_inventory
        self.max_position_usd = max_position_usd

        # Track markets we're making in
        self._making: dict[str, dict] = {}  # market_id -> {bid_level, ask_level, net_qty}

    def on_trade(self, trade: TradeEvent, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[dict]:
        signals = []
        mid = trade.market_id

        if not snapshot.resolution_ts:
            return signals

        time_to_res = snapshot.resolution_ts - trade.timestamp
        entry_window = self.entry_hours * 3600
        exit_window = self.exit_hours * 3600

        # Flatten inventory near resolution
        if mid in self._making and time_to_res < exit_window:
            info = self._making[mid]
            if info["net_qty"] > 0:
                signals.append({
                    "action": "SELL",
                    "market_id": mid,
                    "side": "YES",
                    "qty": abs(info["net_qty"]),
                    "price": snapshot.last_price,
                })
            elif info["net_qty"] < 0:
                signals.append({
                    "action": "SELL",
                    "market_id": mid,
                    "side": "NO",
                    "qty": abs(info["net_qty"]),
                    "price": 1.0 - snapshot.last_price,
                })
            del self._making[mid]
            return signals

        # Enter making window
        if time_to_res > entry_window or time_to_res < exit_window:
            return signals

        # Start making if not already
        if mid not in self._making:
            mid_price = snapshot.last_price
            if mid_price < 0.10 or mid_price > 0.90:
                return signals  # don't make near extremes

            self._making[mid] = {
                "bid_level": mid_price - self.half_spread,
                "ask_level": mid_price + self.half_spread,
                "net_qty": 0.0,
                "mid": mid_price,
            }

        info = self._making[mid]

        # Check if this trade crossed our levels (simulated fill)
        if abs(info["net_qty"]) * snapshot.last_price >= self.max_inventory:
            return signals

        # Buyer crosses our ask — we sell YES (go short)
        if trade.taker_side == "BUY" and trade.price >= info["ask_level"]:
            fill_qty = min(trade.size, self.max_position_usd / trade.price)
            info["net_qty"] -= fill_qty

        # Seller crosses our bid — we buy YES (go long)
        elif trade.taker_side == "SELL" and trade.price <= info["bid_level"]:
            fill_qty = min(trade.size, self.max_position_usd / trade.price)
            signals.append({
                "action": "BUY",
                "market_id": mid,
                "side": "YES",
                "qty": fill_qty,
                "price": info["bid_level"],
            })
            info["net_qty"] += fill_qty

        # Re-center levels around current mid
        info["mid"] = snapshot.last_price
        info["bid_level"] = snapshot.last_price - self.half_spread
        info["ask_level"] = snapshot.last_price + self.half_spread

        return signals
