"""Position sizing: notional-based (default) or cost-based (legacy)."""

from __future__ import annotations

from .config import BacktestConfig


class PositionSizer:
    """Compute position size in shares, capped by notional and volume."""

    def __init__(self, config: BacktestConfig):
        self.mode = config.sizing_mode
        self.max_notional = config.max_notional
        self.volume_cap_pct = config.volume_cap_pct

    def compute_size(self, entry_price: float, trailing_volume_1h: float) -> float:
        """Return number of shares to trade.

        In notional mode: shares = max_notional (fixed 300 shares regardless of price).
        In cost mode: shares = max_notional / entry_price (legacy, $300 budget).
        Then cap at volume_cap_pct of trailing 1h volume.
        """
        if entry_price <= 0:
            return 0.0

        if self.mode == "notional":
            shares = self.max_notional
        else:
            shares = self.max_notional / entry_price

        # Volume cap: don't assume fills larger than X% of recent volume
        if trailing_volume_1h > 0:
            volume_cap = self.volume_cap_pct * trailing_volume_1h
            if shares > volume_cap:
                shares = volume_cap

        return max(shares, 0.0)
