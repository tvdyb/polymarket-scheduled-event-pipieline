from __future__ import annotations

from .api import extract_event_time, extract_tokens


def filter_scheduled_markets(markets: list[dict], require_tokens: bool = True) -> list[dict]:
    filtered = []
    for m in markets:
        event_time = extract_event_time(m)
        if not event_time:
            continue
        tokens = extract_tokens(m)
        if require_tokens and not tokens:
            continue
        m2 = dict(m)
        m2["_event_time"] = event_time.isoformat()
        m2["_tokens"] = tokens
        filtered.append(m2)
    return filtered
