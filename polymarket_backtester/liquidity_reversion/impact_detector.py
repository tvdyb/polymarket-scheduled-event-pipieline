"""Signal detection: identifies price dislocations from trade flow.

Compares current trade price against the 1h VWAP (not the previous
single trade) to distinguish real dislocations from bid-ask bounce.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import BacktestConfig


@dataclass
class ImpactSignal:
    """A detected price dislocation that may warrant a fade trade."""
    timestamp: int
    market_id: str
    impact: float              # signed price move vs VWAP
    reference_price: float     # VWAP used as reference
    trigger_price: float       # price that triggered the signal (YES price)
    fade_side: str             # "YES" or "NO" — the side to buy to fade
    target_price: float        # YES price at which reversion target is hit
    entry_price: float         # price of the fade side token at signal time


class ImpactDetector:
    """Monitors trade prices vs rolling VWAP and emits fade signals.

    Signal logic:
    - Compare current trade price against 1h VWAP (not prev single trade)
    - If |deviation| >= impact_threshold AND market volume < threshold
    - AND price not near extremes — emit a fade signal
    """

    def __init__(self, config: BacktestConfig):
        self.impact_threshold = config.impact_threshold
        self.low_volume_threshold = config.low_volume_threshold
        self.reversion_pct = config.reversion_target_pct
        self.min_vwap_trades = config.min_vwap_trades

    def on_trade(self, timestamp: int, market_id: str, price: float,
                 volume_24h: float, vwap_1h: float,
                 trade_count_1h: int) -> ImpactSignal | None:
        """Process a trade and return a signal if impact detected.

        Args:
            price: YES-side price of the trade
            volume_24h: trailing 24h volume for the market
            vwap_1h: 1-hour VWAP for the market
            trade_count_1h: number of trades in last hour
        """
        # Need enough trades for VWAP to be meaningful
        if trade_count_1h < self.min_vwap_trades:
            return None

        impact = price - vwap_1h

        if abs(impact) < self.impact_threshold:
            return None

        if volume_24h > self.low_volume_threshold:
            return None

        # Don't fade near extremes (likely resolving)
        if price < 0.05 or price > 0.95:
            return None

        # Fade the move: revert toward VWAP
        if impact > 0:
            fade_side = "NO"
            target = vwap_1h + impact * (1.0 - self.reversion_pct)
        else:
            fade_side = "YES"
            target = vwap_1h + impact * (1.0 - self.reversion_pct)

        entry_price = (1.0 - price) if fade_side == "NO" else price

        if entry_price <= 0:
            return None

        return ImpactSignal(
            timestamp=timestamp,
            market_id=market_id,
            impact=impact,
            reference_price=vwap_1h,
            trigger_price=price,
            fade_side=fade_side,
            target_price=target,
            entry_price=entry_price,
        )
