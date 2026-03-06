from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

import requests


TIMESTAMP_KEYS = [
    "startDate", "startTime", "start_date", "start_time",
    "endDate", "endTime", "end_date", "end_time",
    "resolveDate", "resolutionDate", "eventStartDate", "event_start_date",
]


class PolymarketClient:
    def __init__(self, gamma_base: str, clob_base: str, timeout: int = 15) -> None:
        self.gamma_base = gamma_base.rstrip("/")
        self.clob_base = clob_base.rstrip("/")
        self.timeout = timeout

    def fetch_markets(self, closed: bool = True, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        url = f"{self.gamma_base}/markets"
        params = {"closed": str(closed).lower(), "limit": limit, "offset": offset}
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    def fetch_price_history(self, token_id: str, interval: str = "max", fidelity: int = 60) -> list[dict[str, Any]]:
        url = f"{self.clob_base}/prices-history"
        params = {"market": token_id, "interval": interval, "fidelity": fidelity}
        r = requests.get(url, params=params, timeout=self.timeout)
        if r.status_code >= 400:
            return []
        payload = r.json() if r.text else {}
        hist = payload.get("history", []) if isinstance(payload, dict) else []
        return hist if isinstance(hist, list) else []


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # heuristics: ms vs seconds
        v = float(value)
        if v > 1e12:
            v = v / 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def extract_event_time(market: dict[str, Any]) -> datetime | None:
    for key in TIMESTAMP_KEYS:
        dt = parse_datetime(market.get(key))
        if dt:
            return dt

    event = market.get("event")
    if isinstance(event, dict):
        for key in TIMESTAMP_KEYS:
            dt = parse_datetime(event.get(key))
            if dt:
                return dt
    return None


def extract_tokens(market: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    # Gamma sometimes has clobTokenIds as json string list
    token_ids_raw = market.get("clobTokenIds")
    token_ids: list[str] = []
    if isinstance(token_ids_raw, list):
        token_ids = [str(t) for t in token_ids_raw]
    elif isinstance(token_ids_raw, str):
        try:
            parsed = json.loads(token_ids_raw)
            if isinstance(parsed, list):
                token_ids = [str(t) for t in parsed]
        except Exception:
            pass

    outcomes = market.get("outcomes")
    if isinstance(outcomes, str):
        try:
            parsed_outcomes = json.loads(outcomes)
            if isinstance(parsed_outcomes, list):
                outcomes = parsed_outcomes
        except Exception:
            pass

    if isinstance(outcomes, list):
        for i, outcome in enumerate(outcomes):
            if isinstance(outcome, dict):
                name = str(outcome.get("name") or outcome.get("outcome") or f"outcome_{i}")
            else:
                name = str(outcome)
            token = token_ids[i] if i < len(token_ids) else ""
            out.append({"name": name, "token_id": token})

    if not out and token_ids:
        for i, token_id in enumerate(token_ids):
            out.append({"name": "YES" if i == 0 else f"outcome_{i}", "token_id": token_id})

    return [x for x in out if x.get("token_id")]


def normalize_history_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for p in points:
        ts = parse_datetime(p.get("t") or p.get("timestamp") or p.get("time"))
        if not ts:
            continue
        price = p.get("p") or p.get("price")
        if price is None:
            # fallback to OHLC midpoint
            h = p.get("h") or p.get("high")
            l = p.get("l") or p.get("low")
            if h is not None and l is not None:
                try:
                    price = (float(h) + float(l)) / 2.0
                except Exception:
                    price = None
        try:
            price_f = float(price)
        except Exception:
            continue
        normalized.append({"timestamp": ts.isoformat(), "price": price_f})
    normalized.sort(key=lambda x: x["timestamp"])
    return normalized
