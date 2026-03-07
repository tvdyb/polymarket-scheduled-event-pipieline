"""Position sizing: notional-based (default) or cost-based (legacy).

Volume cap compares shares against trailing dollar volume / price to
get apples-to-apples comparison in share units.
"""

from __future__ import annotations

from .config import BacktestConfig


class PositionSizer:
    """Compute position size in shares, capped by notional and volume."""

    def __init__(self, config: BacktestConfig):
        self.mode = config.sizing_mode
        self.max_notional = config.max_notional
        self.volume_cap_pct = config.volume_cap_pct

    def compute_size(self, entry_price: float, trailing_volume_1h_usd: float) -> float:
        """Return number of shares to trade.

        In notional mode: shares = max_notional (fixed 300 shares regardless of price).
        In cost mode: shares = max_notional / entry_price (legacy, $300 budget).

        Volume cap converts trailing USD volume to share-equivalent before comparing:
          volume_in_shares = trailing_volume_1h_usd / entry_price
          cap = volume_cap_pct * volume_in_shares
        """
        if entry_price <= 0:
            return 0.0

        if self.mode == "notional":
            shares = self.max_notional
        else:
            shares = self.max_notional / entry_price

        # Volume cap: convert USD volume to shares, then cap
        if trailing_volume_1h_usd > 0 and entry_price > 0:
            volume_in_shares = trailing_volume_1h_usd / entry_price
            volume_cap = self.volume_cap_pct * volume_in_shares
            if shares > volume_cap:
                shares = volume_cap

        return max(shares, 0.0)
