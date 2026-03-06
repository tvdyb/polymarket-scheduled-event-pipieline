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


@dataclass
class _WindowAccum:
    """Incremental accumulator for a time window."""
    total_size: float = 0.0
    total_notional: float = 0.0
    count: int = 0


class MarketState:
    """Tracks per-market state from streaming trade events.

    Uses incremental accumulators instead of recomputing from deque each time.
    """

    def __init__(self, window_seconds: int = 86400):
        self._window = window_seconds  # 24h default
        # Per-market deques for eviction, keyed by (market_id, window_secs)
        self._trades_1h: dict[str, deque] = defaultdict(deque)
        self._trades_4h: dict[str, deque] = defaultdict(deque)
        self._trades_24h: dict[str, deque] = defaultdict(deque)
        # Incremental accumulators
        self._acc_1h: dict[str, _WindowAccum] = defaultdict(_WindowAccum)
        self._acc_4h: dict[str, _WindowAccum] = defaultdict(_WindowAccum)
        self._acc_24h: dict[str, _WindowAccum] = defaultdict(_WindowAccum)
        self._snapshots: dict[str, MarketSnapshot] = {}
        self._market_meta: dict[str, dict] = {}

    def register_market(self, market_id: str, resolution_ts: int | None = None,
                        category: str = "", resolution: str | None = None):
        self._market_meta[market_id] = {
            "resolution_ts": resolution_ts,
            "category": category,
            "resolution": resolution,
        }

    def on_trade(self, trade: TradeEvent) -> MarketSnapshot:
        mid = trade.market_id
        ts = trade.timestamp
        notional = trade.price * trade.size

        # Add to all windows
        entry = (ts, trade.price, trade.size, notional, trade.taker_side)

        for deq, acc, window_secs in [
            (self._trades_1h[mid], self._acc_1h[mid], 3600),
            (self._trades_4h[mid], self._acc_4h[mid], 14400),
            (self._trades_24h[mid], self._acc_24h[mid], 86400),
        ]:
            deq.append(entry)
            acc.total_size += trade.size
            acc.total_notional += notional
            acc.count += 1

            # Evict old entries
            cutoff = ts - window_secs
            while deq and deq[0][0] < cutoff:
                old = deq.popleft()
                acc.total_size -= old[2]
                acc.total_notional -= old[3]
                acc.count -= 1

        # Build snapshot from accumulators (O(1))
        meta = self._market_meta.get(mid, {})
        a1 = self._acc_1h[mid]
        a4 = self._acc_4h[mid]
        a24 = self._acc_24h[mid]

        snap = MarketSnapshot(
            market_id=mid,
            last_price=trade.price,
            last_trade_ts=ts,
            volume_1h=a1.total_size,
            volume_4h=a4.total_size,
            volume_24h=a24.total_size,
            vwap_1h=a1.total_notional / a1.total_size if a1.total_size > 0 else trade.price,
            vwap_4h=a4.total_notional / a4.total_size if a4.total_size > 0 else trade.price,
            vwap_24h=a24.total_notional / a24.total_size if a24.total_size > 0 else trade.price,
            trade_count_1h=a1.count,
            spread_estimate=0.02,
            resolution_ts=meta.get("resolution_ts"),
            category=meta.get("category", ""),
            resolution=meta.get("resolution"),
        )

        self._snapshots[mid] = snap
        return snap

    def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        return self._snapshots.get(market_id)

    def get_all_snapshots(self) -> dict[str, MarketSnapshot]:
        return dict(self._snapshots)
