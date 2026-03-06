"""Abstract Strategy base class.

All strategies implement on_trade(), on_tick(), and generate_signals().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine.market_state import MarketState, MarketSnapshot, TradeEvent
    from ..engine.portfolio import Portfolio


class Strategy(ABC):
    """Base class for all backtesting strategies."""

    name: str = "base"

    def on_init(self, market_state: MarketState, portfolio: Portfolio):
        """Called once before backtest starts. Override for setup."""
        pass

    @abstractmethod
    def on_trade(self, trade: TradeEvent, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[dict]:
        """Called on every trade event. Return list of signal dicts.

        Signal format:
        {
            "action": "BUY" or "SELL",
            "market_id": str,
            "side": "YES" or "NO",
            "qty": float,
            "price": float,
        }
        """
        ...

    def on_tick(self, timestamp: int, market_state: MarketState, portfolio: Portfolio) -> list[dict]:
        """Called periodically (configurable interval). Return signals."""
        return []
