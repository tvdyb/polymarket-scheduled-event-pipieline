"""Strategy 3: Cross-Market Sentiment Divergence / Inversions.

Hypothesis: Related markets within the same event (e.g., "BTC >100K",
"BTC >110K", "BTC >120K") should have monotonically decreasing YES prices.
Inversions are mispricings.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..engine.market_state import MarketState, MarketSnapshot, TradeEvent
from ..engine.portfolio import Portfolio
from .base import Strategy


def _extract_threshold(question: str) -> float | None:
    """Try to extract a numeric threshold from a market question."""
    # Patterns: ">100K", "above $100,000", "over 110K", "exceed 3%"
    m = re.search(r'[>≥]\s*\$?([\d,]+\.?\d*)\s*[kK]?', question)
    if m:
        val = float(m.group(1).replace(",", ""))
        if question[m.end()-1:m.end()].lower() == 'k':
            val *= 1000
        return val
    m = re.search(r'(?:above|over|exceed|hit|reach)\s+\$?([\d,]+\.?\d*)\s*[kK]?', question, re.I)
    if m:
        val = float(m.group(1).replace(",", ""))
        if m.group(0).rstrip()[-1:].lower() == 'k':
            val *= 1000
        return val
    return None


class CrossMarketArbStrategy(Strategy):
    name = "cross_market_arb"

    def __init__(
        self,
        min_inversion_cents: float = 0.02,
        min_event_markets: int = 3,
        max_hold_seconds: int = 86400,  # 24h
        max_position_usd: float = 300.0,
    ):
        self.min_inversion = min_inversion_cents
        self.min_markets = min_event_markets
        self.max_hold = max_hold_seconds
        self.max_position_usd = max_position_usd

        # event_slug -> [{market_id, threshold, last_price}] sorted by threshold
        self._event_groups: dict[str, list[dict]] = defaultdict(list)
        # Active arb positions
        self._arbs: dict[str, dict] = {}  # key -> {buy_id, sell_id, entry_ts}

    def on_init(self, market_state: MarketState, portfolio: Portfolio):
        """Build event groups from market metadata — called by backtester."""
        # This would be populated from markets.csv event_slug grouping
        pass

    def register_event_group(self, event_slug: str, markets: list[dict]):
        """Register a group of related markets with thresholds.

        markets: list of {market_id, question, threshold}
        """
        ordered = sorted(
            [m for m in markets if m.get("threshold") is not None],
            key=lambda m: m["threshold"],
        )
        if len(ordered) >= self.min_markets:
            self._event_groups[event_slug] = [
                {"market_id": m["market_id"], "threshold": m["threshold"], "last_price": 0.5}
                for m in ordered
            ]

    def on_trade(self, trade: TradeEvent, snapshot: MarketSnapshot, portfolio: Portfolio) -> list[dict]:
        signals = []

        # Update prices in event groups
        for slug, group in self._event_groups.items():
            for entry in group:
                if entry["market_id"] == trade.market_id:
                    entry["last_price"] = snapshot.last_price

            # Check for inversions (higher threshold should have lower YES price)
            if len(group) < self.min_markets:
                continue

            for i in range(len(group) - 1):
                lower = group[i]   # lower threshold (should be MORE likely -> higher price)
                higher = group[i+1]  # higher threshold (should be LESS likely -> lower price)

                if higher["last_price"] > lower["last_price"] + self.min_inversion:
                    # Inversion! Buy the cheap one (lower threshold), sell the expensive one
                    arb_key = f"{lower['market_id']}:{higher['market_id']}"
                    if arb_key in self._arbs:
                        continue

                    buy_price = lower["last_price"]
                    if buy_price <= 0:
                        continue
                    qty = self.max_position_usd / buy_price

                    signals.append({
                        "action": "BUY",
                        "market_id": lower["market_id"],
                        "side": "YES",
                        "qty": qty,
                        "price": buy_price,
                    })

                    self._arbs[arb_key] = {
                        "buy_id": lower["market_id"],
                        "sell_id": higher["market_id"],
                        "entry_ts": trade.timestamp,
                        "qty": qty,
                    }

        # Check exits: reversion to monotonic or time expired
        for arb_key in list(self._arbs.keys()):
            arb = self._arbs[arb_key]
            if trade.timestamp - arb["entry_ts"] > self.max_hold:
                signals.append({
                    "action": "SELL",
                    "market_id": arb["buy_id"],
                    "side": "YES",
                    "qty": arb["qty"],
                    "price": snapshot.last_price if trade.market_id == arb["buy_id"] else 0.5,
                })
                del self._arbs[arb_key]

        return signals
