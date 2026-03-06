"""Polars schema definitions for poly_data CSVs and internal data structures."""

import polars as pl

# ── poly_data schemas ────────────────────────────────────────────────────────

MARKETS_SCHEMA = {
    "id": pl.Utf8,
    "question": pl.Utf8,
    "slug": pl.Utf8,
    "event_slug": pl.Utf8,
    "category": pl.Utf8,
    "neg_risk": pl.Boolean,
    "volume": pl.Float64,
    "closedTime": pl.Utf8,
    "endDate": pl.Utf8,
    "outcomes": pl.Utf8,
    "tokens": pl.Utf8,
    "resolution": pl.Utf8,
}

TRADES_SCHEMA = {
    "market": pl.Utf8,
    "timestamp": pl.Int64,       # UTC epoch seconds
    "price": pl.Float64,
    "size": pl.Float64,          # USDC notional
    "side": pl.Utf8,             # "BUY" or "SELL"
    "taker_side": pl.Utf8,       # "BUY" or "SELL" (taker direction)
    "maker": pl.Utf8,
    "taker": pl.Utf8,
    "outcome": pl.Utf8,          # "Yes" or "No"
    "token_id": pl.Utf8,
}

ORDER_FILLED_SCHEMA = {
    "market": pl.Utf8,
    "timestamp": pl.Int64,
    "price": pl.Float64,
    "size": pl.Float64,
    "side": pl.Utf8,
    "maker": pl.Utf8,
    "taker": pl.Utf8,
}

# ── Internal schemas ─────────────────────────────────────────────────────────

POSITION_SCHEMA = {
    "market_id": pl.Utf8,
    "side": pl.Utf8,             # "YES" or "NO"
    "qty": pl.Float64,
    "avg_entry": pl.Float64,
    "entry_ts": pl.Int64,
}

TRADE_LOG_SCHEMA = {
    "timestamp": pl.Int64,
    "market_id": pl.Utf8,
    "action": pl.Utf8,           # "BUY" or "SELL"
    "side": pl.Utf8,             # "YES" or "NO"
    "qty": pl.Float64,
    "price": pl.Float64,
    "slippage": pl.Float64,
    "strategy": pl.Utf8,
}

EQUITY_SCHEMA = {
    "timestamp": pl.Int64,
    "cash": pl.Float64,
    "unrealized_pnl": pl.Float64,
    "total_value": pl.Float64,
}
