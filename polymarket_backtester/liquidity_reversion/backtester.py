"""Main backtest loop for liquidity reversion strategy.

Architecture:
  ImpactDetector  -> detects signals (untouched signal logic)
  FillSimulator   -> latency + VWAP fill simulation
  PositionSizer   -> notional sizing + volume caps
  PositionManager -> risk limits, exits, position tracking

The loop processes ALL trades sequentially (no sampling).
"""

from __future__ import annotations

import time
from collections import defaultdict

import polars as pl
from tqdm import tqdm

from .config import BacktestConfig
from .fill_simulator import FillSimulator, Fill
from .impact_detector import ImpactDetector
from .position_sizer import PositionSizer
from .position_manager import PositionManager
from .reporting import (
    compute_metrics, write_trade_log, write_equity_curve, print_metrics,
)

# Reuse the efficient market state tracker from the engine
from ..engine.market_state import MarketState, TradeEvent


class LiquidityReversionBacktester:
    """Self-contained backtester for the liquidity reversion strategy."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.market_state = MarketState()
        self.impact_detector = ImpactDetector(self.config)
        self.fill_simulator = FillSimulator(self.config)
        self.position_sizer = PositionSizer(self.config)
        self.position_manager = PositionManager(self.config)

        # Tracking
        self._total_signals = 0
        self._total_fills = 0
        self._filter_counts: dict[str, int] = defaultdict(int)
        self._equity_curve: list[dict] = []
        self._last_equity_day: int = 0
        # Track which markets have active fades in the fill simulator
        self._pending_fade_markets: set[str] = set()

    def load_markets(self, markets_df: pl.DataFrame):
        """Register market metadata (resolution times, categories)."""
        for row in markets_df.iter_rows(named=True):
            market_id = str(row.get("id") or row.get("condition_id") or "")
            resolution_ts = None
            closed_time = row.get("closedTime") or row.get("endDate")
            if closed_time:
                try:
                    from datetime import datetime, timezone
                    ct = str(closed_time).strip()
                    for tz_suffix in ("+00:00", "+00", "Z"):
                        if ct.endswith(tz_suffix):
                            ct = ct[: -len(tz_suffix)]
                            break
                    dt = datetime.strptime(ct[:19], "%Y-%m-%d %H:%M:%S")
                    resolution_ts = int(dt.replace(tzinfo=timezone.utc).timestamp())
                except Exception:
                    pass

            resolution = row.get("resolution")
            category = row.get("category") or ""

            self.market_state.register_market(
                market_id=market_id,
                resolution_ts=resolution_ts,
                category=category,
                resolution=str(resolution) if resolution else None,
            )

    def run(self, trades_df: pl.DataFrame, show_progress: bool = True) -> dict:
        """Run the backtest. Processes ALL trades (no sampling)."""
        start_time = time.time()

        trades = trades_df.sort("timestamp")
        total_rows = len(trades)

        iterator = trades.iter_rows(named=True)
        if show_progress:
            iterator = tqdm(iterator, total=total_rows, desc="Backtesting")

        for row in iterator:
            ts = int(row["timestamp"])
            market_id = str(row.get("market") or row.get("market_id") or "")
            price = float(row.get("price", 0))
            size = float(row.get("size", 0))

            if not market_id or price <= 0:
                continue

            # 1. Update market state
            taker_side = str(row.get("taker_side") or row.get("side") or "")
            outcome = str(row.get("outcome") or "Yes")
            maker = str(row.get("maker") or "")
            taker_addr = str(row.get("taker") or "")

            trade_event = TradeEvent(
                market_id=market_id, timestamp=ts,
                price=price, size=size, taker_side=taker_side,
                outcome=outcome, maker=maker, taker=taker_addr,
            )
            snapshot = self.market_state.on_trade(trade_event)

            # 2. Process pending fills in the fill simulator
            fills, cancelled = self.fill_simulator.on_trade(ts, market_id, price, size)
            for c in cancelled:
                self._pending_fade_markets.discard(c.market_id)
                self._filter_counts["fill_timeout"] += 1
            for fill in fills:
                self._process_fill(fill)

            # 3. Check exits on open positions in this market
            resolution_ts = snapshot.resolution_ts
            self.position_manager.check_exits(ts, market_id, price, resolution_ts)

            # 4. Detect new impact signals
            signal = self.impact_detector.on_trade(
                ts, market_id, price, snapshot.volume_24h
            )

            if signal:
                self._total_signals += 1
                rejection = self._apply_filters(signal, snapshot)
                if rejection:
                    self._filter_counts[rejection] += 1
                else:
                    # Size the position
                    shares = self.position_sizer.compute_size(
                        signal.entry_price, snapshot.volume_1h
                    )
                    if shares <= 0:
                        self._filter_counts["zero_size"] += 1
                    else:
                        # Check risk limits
                        cost = shares * signal.entry_price
                        limit_rejection = self.position_manager.can_open(market_id, cost)
                        if limit_rejection:
                            self._filter_counts[limit_rejection] += 1
                        else:
                            # Submit to fill simulator (delayed execution)
                            self.fill_simulator.submit(
                                signal_time=ts,
                                market_id=market_id,
                                side=signal.fade_side,
                                direction="BUY",
                                signal_price=signal.entry_price,
                                max_shares=shares,
                                target_price=signal.target_price,
                            )
                            self._pending_fade_markets.add(market_id)

            # 5. Daily equity snapshot
            day = ts // 86400
            if day > self._last_equity_day:
                self._record_equity(ts)
                self._last_equity_day = day

        # End of data: force-close all remaining positions
        prices = {mid: s.last_price for mid, s in self.market_state.get_all_snapshots().items()}
        self.position_manager.force_close_all(ts, prices)

        # Final equity point
        self._record_equity(ts)

        elapsed = time.time() - start_time

        metrics = compute_metrics(
            closed=self.position_manager.closed_trades,
            equity_curve=self._equity_curve,
            config=self.config,
            filter_counts=dict(self._filter_counts),
            total_signals=self._total_signals,
            total_fills=self._total_fills,
            runtime_seconds=elapsed,
        )
        metrics["total_trade_events"] = total_rows

        return metrics

    def _process_fill(self, fill: Fill):
        """Handle a completed fill from the simulator."""
        self._total_fills += 1
        self._pending_fade_markets.discard(fill.market_id)

        # Re-check risk limits at fill time (state may have changed)
        cost = fill.shares * fill.fill_price
        rejection = self.position_manager.can_open(fill.market_id, cost)
        if rejection:
            self._filter_counts[f"fill_rejected_{rejection}"] += 1
            return

        self.position_manager.open_position(fill)

    def _apply_filters(self, signal, snapshot) -> str | None:
        """Apply all signal filters. Returns rejection reason or None."""
        # Entry price band
        yes_price = signal.trigger_price
        if yes_price < self.config.entry_price_min or yes_price > self.config.entry_price_max:
            return "price_band"

        # Resolution proximity
        if snapshot.resolution_ts:
            time_to_close = snapshot.resolution_ts - signal.timestamp
            if time_to_close < self.config.resolution_proximity_hours * 3600:
                return "resolution_proximity"

        # Already have a pending order for this market
        if signal.market_id in self._pending_fade_markets:
            return "pending_order_exists"

        return None

    def _record_equity(self, timestamp: int):
        """Snapshot equity curve at daily granularity."""
        open_positions = self.position_manager.total_open
        total_notional = sum(
            p.entry_notional for p in self.position_manager.open_positions
        )
        cumulative_pnl = sum(t.pnl for t in self.position_manager.closed_trades)

        self._equity_curve.append({
            "timestamp": timestamp,
            "cumulative_pnl": cumulative_pnl,
            "open_positions": open_positions,
            "total_notional_exposure": total_notional,
        })
