"""Strategy 4: Liquidity-Adjusted Mean Reversion.

Hypothesis: In thin markets, large trades cause temporary price dislocations
that revert. Fade large price impacts in low-volume markets.
"""

from __future__ import annotations

from ..engine.market_state import MarketState, MarketSnapshot, TradeEvent
from ..engine.portfolio import Portfolio
from .base import Strategy


class LiquidityReversionStrategy(Strategy):
    name = "liquidity_reversion"

    def __init__(
        self,
        impact_threshold: float = 0.03,
        low_volume_threshold: float = 50_000.0,
        max_hold_seconds: int = 14400,  # 4 hours
        reversion_target_pct: float = 0.60,
        max_position_usd: float = 300.0,
    ):
        self.impact_threshold = impact_threshold
        self.low_volume = low_volume_threshold
        self.max_hold = max_hold_seconds
        self.reversion_pct = reversion_target_pct
        self.max_position_usd = max_position_usd

        # Track pre-trade prices for impact calculation
        self._prev_prices: dict[str, float] = {}
        # Track active fade positions: market_id -> {entry_ts, target_price, side}
        self._fades: dict[str, dict] = {}

    def on_trade(self, trade: TradeEvent, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[dict]:
        signals = []

        # Check for exits on existing fades
        if trade.market_id in self._fades:
            fade = self._fades[trade.market_id]
            target = fade["target_price"]
            side = fade["side"]

            # Hit reversion target
            hit_target = (side == "YES" and trade.price >= target) or \
                         (side == "NO" and trade.price <= target)
            # Time expired
            expired = trade.timestamp - fade["entry_ts"] > self.max_hold

            if hit_target or expired:
                signals.append({
                    "action": "SELL",
                    "market_id": trade.market_id,
                    "side": side,
                    "qty": fade["qty"],
                    "price": trade.price,
                })
                del self._fades[trade.market_id]

        # Check for new fade entries
        prev = self._prev_prices.get(trade.market_id)
        self._prev_prices[trade.market_id] = trade.price

        if prev is None or trade.market_id in self._fades:
            return signals

        # Compute price impact
        impact = trade.price - prev

        if abs(impact) < self.impact_threshold:
            return signals

        # Only in low-volume markets
        if snapshot.volume_24h > self.low_volume:
            return signals

        # Don't fade near extremes (likely resolving)
        if trade.price < 0.05 or trade.price > 0.95:
            return signals

        # Fade the move: if price jumped up, sell (buy NO). If dropped, buy YES.
        if impact > 0:
            # Price went up — fade by buying NO (expect reversion down)
            side = "NO"
            target = prev + impact * (1.0 - self.reversion_pct)
        else:
            # Price went down — fade by buying YES (expect reversion up)
            side = "YES"
            target = prev + impact * (1.0 - self.reversion_pct)

        entry_price = (1.0 - trade.price) if side == "NO" else trade.price
        if entry_price <= 0:
            return signals

        qty = self.max_position_usd / entry_price

        signals.append({
            "action": "BUY",
            "market_id": trade.market_id,
            "side": side,
            "qty": qty,
            "price": entry_price,
        })

        self._fades[trade.market_id] = {
            "entry_ts": trade.timestamp,
            "target_price": target,
            "side": side,
            "qty": qty,
        }

        return signals

    def on_tick(self, timestamp: int, market_state: MarketState, portfolio: Portfolio) -> list[dict]:
        # Time-based exits handled in on_trade for simplicity
        return []
