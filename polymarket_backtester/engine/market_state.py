"""Reconstructs order book state at any timestamp from trade data.

Maintains rolling windows of recent trades per market to compute:
last price, VWAP, volume, spread estimate, time to resolution.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class MarketSnapshot:
    market_id: str
    last_price: float = 0.5
    last_trade_ts: int = 0
    volume_1h: float = 0.0
    volume_4h: float = 0.0
    volume_24h: float = 0.0
    vwap_1h: float = 0.5
    vwap_4h: float = 0.5
    vwap_24h: float = 0.5
    spread_estimate: float = 0.02  # estimated bid-ask spread
    trade_count_1h: int = 0
    resolution_ts: int | None = None
    category: str = ""
    resolution: str | None = None  # "Yes" / "No" / None


@dataclass
class TradeEvent:
    market_id: str
    timestamp: int
    price: float
    size: float
    taker_side: str  # "BUY" or "SELL"
    outcome: str     # "Yes" or "No"
    maker: str = ""
    taker: str = ""


class MarketState:
    """Tracks per-market state from streaming trade events."""

    def __init__(self, window_seconds: int = 86400):
        self._window = window_seconds  # 24h default
        self._trades: dict[str, deque[TradeEvent]] = defaultdict(deque)
        self._snapshots: dict[str, MarketSnapshot] = {}
        self._market_meta: dict[str, dict] = {}  # market_id -> {resolution_ts, category, resolution}

    def register_market(self, market_id: str, resolution_ts: int | None = None,
                        category: str = "", resolution: str | None = None):
        """Register market metadata (call before processing trades)."""
        self._market_meta[market_id] = {
            "resolution_ts": resolution_ts,
            "category": category,
            "resolution": resolution,
        }

    def on_trade(self, trade: TradeEvent) -> MarketSnapshot:
        """Process a new trade event and return updated market snapshot."""
        q = self._trades[trade.market_id]
        q.append(trade)

        # Evict old trades outside window
        cutoff = trade.timestamp - self._window
        while q and q[0].timestamp < cutoff:
            q.popleft()

        snap = self._build_snapshot(trade.market_id, trade.timestamp)
        self._snapshots[trade.market_id] = snap
        return snap

    def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        return self._snapshots.get(market_id)

    def get_all_snapshots(self) -> dict[str, MarketSnapshot]:
        return dict(self._snapshots)

    def _build_snapshot(self, market_id: str, now: int) -> MarketSnapshot:
        trades = self._trades[market_id]
        meta = self._market_meta.get(market_id, {})

        snap = MarketSnapshot(
            market_id=market_id,
            resolution_ts=meta.get("resolution_ts"),
            category=meta.get("category", ""),
            resolution=meta.get("resolution"),
        )

        if not trades:
            return snap

        last = trades[-1]
        snap.last_price = last.price
        snap.last_trade_ts = last.timestamp

        # Compute VWAP and volume for different windows
        for window_secs, attr_vol, attr_vwap, attr_count in [
            (3600, "volume_1h", "vwap_1h", "trade_count_1h"),
            (14400, "volume_4h", "vwap_4h", None),
            (86400, "volume_24h", "vwap_24h", None),
        ]:
            cutoff = now - window_secs
            total_notional = 0.0
            total_size = 0.0
            count = 0
            for t in trades:
                if t.timestamp >= cutoff:
                    total_notional += t.price * t.size
                    total_size += t.size
                    count += 1
            setattr(snap, attr_vol, total_size)
            if total_size > 0:
                setattr(snap, attr_vwap, total_notional / total_size)
            if attr_count:
                setattr(snap, attr_count, count)

        # Estimate spread from recent buy/sell prices
        recent_buys = [t.price for t in trades if t.taker_side == "BUY" and t.timestamp >= now - 3600]
        recent_sells = [t.price for t in trades if t.taker_side == "SELL" and t.timestamp >= now - 3600]
        if recent_buys and recent_sells:
            snap.spread_estimate = max(0.01, max(recent_buys) - min(recent_sells))

        return snap
