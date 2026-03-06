"""Phase 4: Backtest logic — entry/exit on calendar dates, PnL calculation.

Exit rule: sell as soon as price moves 5 cents in either direction from entry.
This prevents holding through resolution where prices snap to 0 or 1.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import ENTRY_DAYS_BEFORE, EXIT_DAYS_BEFORE

EXIT_MOVE_THRESHOLD = 0.05  # sell when price moves this far from entry


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


def _find_exit_on_move(
    prices: list[dict],
    entry_date: str,
    entry_price: float,
    deadline_date: str,
    threshold: float = EXIT_MOVE_THRESHOLD,
) -> tuple[float | None, str | None, str]:
    """Scan forward from entry_date for the first price that moves ±threshold from entry_price.

    If no move hits before deadline_date, exit at last price before deadline.
    Returns (exit_price, exit_date, exit_reason).
    """
    post_entry = [p for p in prices if p["date"] >= entry_date and p["date"] <= deadline_date]

    for p in post_entry:
        if abs(p["price"] - entry_price) >= threshold:
            return p["price"], p["date"], "threshold_hit"

    # No threshold hit — exit at last available price before deadline
    if post_entry:
        last = post_entry[-1]
        return last["price"], last["date"], "deadline"

    return None, None, "no_data"


def run_backtest(
    markets: list[dict],
    prices: dict[str, list[dict]],
    entry_days_list: list[int] = ENTRY_DAYS_BEFORE,
    exit_days_list: list[int] = EXIT_DAYS_BEFORE,
) -> dict[str, list[dict]]:
    """Run backtest for all (N, M) entry/exit day combinations.

    Exit rule: sell as soon as price moves ±5c from entry, or at M days
    before event if no move occurs.

    Returns dict mapping "(N,M)" -> list of trade dicts.
    """
    all_results = {}

    for n_entry in entry_days_list:
        for m_exit in exit_days_list:
            if m_exit >= n_entry:
                continue

            combo_key = f"({n_entry},{m_exit})"
            trades = []

            for market in markets:
                mid = market["id"]
                price_history = prices.get(mid, [])

                if not price_history:
                    continue

                llm_data = market.get("_llm", {})
                event_date_str = llm_data.get("event_date")
                if not event_date_str:
                    event_date_str = market.get("end_date", "")

                event_dt = _parse_date(event_date_str)
                if not event_dt:
                    continue

                event_date_clean = event_dt.strftime("%Y-%m-%d")

                if len(price_history) < n_entry + 3:
                    continue

                # Entry
                entry_dt = event_dt - timedelta(days=n_entry)
                entry_date_target = entry_dt.strftime("%Y-%m-%d")
                entry_price, actual_entry_date = _find_price_on_date(price_history, entry_date_target)
                if entry_price is None or entry_price <= 0:
                    continue

                # Deadline: M days before event (or event date if M=0)
                if m_exit == 0:
                    deadline_date = event_date_clean
                else:
                    deadline_date = (event_dt - timedelta(days=m_exit)).strftime("%Y-%m-%d")

                # Exit: first ±5c move, or deadline price
                exit_price, actual_exit_date, exit_reason = _find_exit_on_move(
                    price_history, actual_entry_date, entry_price, deadline_date,
                )
                if exit_price is None:
                    continue

                pnl = exit_price - entry_price
                pct_return = (exit_price - entry_price) / entry_price

                obs_vol = compute_observed_vol(price_history, actual_entry_date or entry_date_target)

                trade = {
                    "market_id": mid,
                    "question": market["question"],
                    "event_type": llm_data.get("event_type", "unknown"),
                    "event_date": event_date_clean,
                    "entry_date": actual_entry_date,
                    "exit_date": actual_exit_date,
                    "exit_reason": exit_reason,
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
