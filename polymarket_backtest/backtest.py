"""Phase 4: Backtest logic — entry/exit on calendar dates, PnL calculation."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import ENTRY_DAYS_BEFORE, EXIT_DAYS_BEFORE


def _parse_date(s) -> datetime | None:
    if not s:
        return None
    if isinstance(s, str):
        s = s.strip()
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
        # Try YYYY-MM-DD
        try:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _find_price_on_date(prices: list[dict], target_date: str, tolerance_days: int = 1) -> tuple[float | None, str | None]:
    """Find price on target_date, or nearest within +-tolerance_days."""
    target = datetime.strptime(target_date, "%Y-%m-%d")
    best_price = None
    best_date = None
    best_diff = None

    for p in prices:
        try:
            d = datetime.strptime(p["date"], "%Y-%m-%d")
        except (ValueError, KeyError):
            continue
        diff = abs((d - target).days)
        if diff <= tolerance_days and (best_diff is None or diff < best_diff):
            best_price = p["price"]
            best_date = p["date"]
            best_diff = diff

    return best_price, best_date


def compute_observed_vol(prices: list[dict], entry_date: str) -> float | None:
    """Compute pre-entry price standard deviation as a diagnostic."""
    pre_entry = [p["price"] for p in prices if p["date"] < entry_date]
    if len(pre_entry) < 3:
        return None
    return pd.Series(pre_entry).std()


def run_backtest(
    markets: list[dict],
    prices: dict[str, list[dict]],
    entry_days_list: list[int] = ENTRY_DAYS_BEFORE,
    exit_days_list: list[int] = EXIT_DAYS_BEFORE,
) -> dict[str, list[dict]]:
    """Run backtest for all (N, M) entry/exit day combinations.

    Returns dict mapping "(N,M)" -> list of trade dicts.
    """
    all_results = {}

    for n_entry in entry_days_list:
        for m_exit in exit_days_list:
            if m_exit >= n_entry:
                continue  # exit must be before entry

            combo_key = f"({n_entry},{m_exit})"
            trades = []

            for market in markets:
                mid = market["id"]
                price_history = prices.get(mid, [])

                if not price_history:
                    continue

                # Determine event date: prefer LLM-extracted, fall back to end_date
                llm_data = market.get("_llm", {})
                event_date_str = llm_data.get("event_date")
                if not event_date_str:
                    event_date_str = market.get("end_date", "")

                event_dt = _parse_date(event_date_str)
                if not event_dt:
                    continue

                event_date_clean = event_dt.strftime("%Y-%m-%d")

                # Check minimum price history length
                if len(price_history) < n_entry + 3:
                    continue

                # Calculate entry and exit dates
                entry_dt = event_dt - timedelta(days=n_entry)
                entry_date_target = entry_dt.strftime("%Y-%m-%d")

                if m_exit == 0:
                    # Hold to resolution — use final price or last available
                    exit_date_target = event_date_clean
                else:
                    exit_dt = event_dt - timedelta(days=m_exit)
                    exit_date_target = exit_dt.strftime("%Y-%m-%d")

                # Find prices
                entry_price, actual_entry_date = _find_price_on_date(price_history, entry_date_target)
                if entry_price is None or entry_price <= 0:
                    continue

                exit_price, actual_exit_date = _find_price_on_date(price_history, exit_date_target)
                if exit_price is None:
                    # Fall back to final resolution price
                    if market.get("final_price") is not None:
                        exit_price = market["final_price"]
                        actual_exit_date = event_date_clean
                    else:
                        # Use last available price
                        if price_history:
                            exit_price = price_history[-1]["price"]
                            actual_exit_date = price_history[-1]["date"]
                        else:
                            continue

                # PnL
                pnl = exit_price - entry_price
                pct_return = (exit_price - entry_price) / entry_price

                # Observed vol diagnostic
                obs_vol = compute_observed_vol(price_history, actual_entry_date or entry_date_target)

                trade = {
                    "market_id": mid,
                    "question": market["question"],
                    "event_type": llm_data.get("event_type", "unknown"),
                    "event_date": event_date_clean,
                    "entry_date": actual_entry_date,
                    "exit_date": actual_exit_date,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "pnl": round(pnl, 4),
                    "pct_return": round(pct_return, 4),
                    "resolution_outcome": market.get("outcome", ""),
                    "observed_pre_entry_std": round(obs_vol, 4) if obs_vol is not None else None,
                    "llm_confidence": llm_data.get("confidence"),
                    "llm_reasoning": llm_data.get("reasoning", ""),
                    "category": market.get("category", ""),
                }
                trades.append(trade)

            all_results[combo_key] = trades

    return all_results
