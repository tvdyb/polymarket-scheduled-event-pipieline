"""Strategy 1: Scheduled-Event Momentum Decay.

Hypothesis: Scheduled events that are "almost certain" to resolve YES trade
at a discount to fair value because of residual uncertainty and thin liquidity
near expiry. Buy when market price < model probability - edge_threshold.
"""

from __future__ import annotations

from ..engine.market_state import MarketState, MarketSnapshot, TradeEvent
from ..engine.portfolio import Portfolio
from .base import Strategy


class ScheduledMomentumStrategy(Strategy):
    name = "scheduled_momentum"

    def __init__(
        self,
        hours_before_expiry: float = 24.0,
        edge_threshold: float = 0.03,
        max_position_usd: float = 500.0,
        min_market_volume: float = 1000.0,
        min_price: float = 0.70,        # only buy "almost certain" markets
        max_price: float = 0.97,        # not already resolved
    ):
        self.hours_before = hours_before_expiry
        self.edge_threshold = edge_threshold
        self.max_position_usd = max_position_usd
        self.min_volume = min_market_volume
        self.min_price = min_price
        self.max_price = max_price
        self._entered: set[str] = set()

    def on_trade(self, trade: TradeEvent, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[dict]:
        # Only trade YES side of markets near expiry
        if trade.market_id in self._entered:
            return []
        if not snapshot.resolution_ts:
            return []

        time_to_resolution = snapshot.resolution_ts - trade.timestamp
        window_seconds = self.hours_before * 3600

        # Only act within the entry window
        if time_to_resolution > window_seconds or time_to_resolution < 0:
            return []

        # Volume filter
        if snapshot.volume_24h < self.min_volume:
            return []

        # Price filter: "almost certain" but discounted
        price = snapshot.last_price
        if price < self.min_price or price > self.max_price:
            return []

        # Simple model: fair value is 1.0 minus residual uncertainty
        # Residual uncertainty decays as we approach resolution
        hours_left = max(time_to_resolution / 3600, 0.1)
        # Simple model: residual uncertainty proportional to sqrt(time)
        residual_uncertainty = min(0.30, 0.05 * (hours_left ** 0.5))
        fair_value = 1.0 - residual_uncertainty

        # Edge check
        edge = fair_value - price
        if edge < self.edge_threshold:
            return []

        # Size: spend up to max_position_usd
        qty = self.max_position_usd / price

        self._entered.add(trade.market_id)

        return [{
            "action": "BUY",
            "market_id": trade.market_id,
            "side": "YES",
            "qty": qty,
            "price": price,
        }]

    def on_tick(self, timestamp: int, market_state: MarketState, portfolio: Portfolio) -> list[dict]:
        # No tick-based logic — hold to resolution
        return []
