"""Signal detection: identifies price dislocations from trade flow.

This is the core signal logic — NOT modified per the spec.
Extracted from the original LiquidityReversionStrategy.on_trade entry logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import BacktestConfig


@dataclass
class ImpactSignal:
    """A detected price dislocation that may warrant a fade trade."""
    timestamp: int
    market_id: str
    impact: float              # signed price move (positive = price went up)
    prev_price: float          # price before impact
    trigger_price: float       # price that triggered the signal (YES price)
    fade_side: str             # "YES" or "NO" — the side to buy to fade
    target_price: float        # YES price at which reversion target is hit
    entry_price: float         # price of the fade side token at signal time


class ImpactDetector:
    """Monitors trade-to-trade price impacts and emits fade signals.

    Signal logic (unchanged from original):
    - Track previous price per market
    - If |price_move| >= impact_threshold AND market volume < low_volume_threshold
    - AND price not near extremes — emit a fade signal
    """

    def __init__(self, config: BacktestConfig):
        self.impact_threshold = config.impact_threshold
        self.low_volume_threshold = config.low_volume_threshold
        self.reversion_pct = config.reversion_target_pct
        self._prev_prices: dict[str, float] = {}

    def on_trade(self, timestamp: int, market_id: str, price: float,
                 volume_24h: float) -> ImpactSignal | None:
        """Process a trade and return a signal if impact detected.

        Args:
            price: YES-side price of the trade
            volume_24h: trailing 24h volume for the market
        """
        prev = self._prev_prices.get(market_id)
        self._prev_prices[market_id] = price

        if prev is None:
            return None

        impact = price - prev

        if abs(impact) < self.impact_threshold:
            return None

        if volume_24h > self.low_volume_threshold:
            return None

        # Don't fade near extremes (likely resolving)
        if price < 0.05 or price > 0.95:
            return None

        # Fade the move
        if impact > 0:
            fade_side = "NO"
            target = prev + impact * (1.0 - self.reversion_pct)
        else:
            fade_side = "YES"
            target = prev + impact * (1.0 - self.reversion_pct)

        entry_price = (1.0 - price) if fade_side == "NO" else price

        if entry_price <= 0:
            return None

        return ImpactSignal(
            timestamp=timestamp,
            market_id=market_id,
            impact=impact,
            prev_price=prev,
            trigger_price=price,
            fade_side=fade_side,
            target_price=target,
            entry_price=entry_price,
        )
