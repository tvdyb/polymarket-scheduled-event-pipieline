from __future__ import annotations

from pathlib import Path
import json

from .api import PolymarketClient, normalize_history_points
from .filters import filter_scheduled_markets


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def fetch_and_store(config: dict) -> dict:
    api_cfg = config["api"]
    fetch_cfg = config["fetch"]
    paths = config["paths"]

    client = PolymarketClient(
        gamma_base=api_cfg["gamma_base"],
        clob_base=api_cfg["clob_base"],
        timeout=int(api_cfg.get("request_timeout_seconds", 15)),
    )

    markets = client.fetch_markets(
        closed=bool(fetch_cfg.get("closed", True)),
        limit=int(fetch_cfg.get("limit", 200)),
        offset=int(fetch_cfg.get("offset", 0)),
    )
    max_markets = int(fetch_cfg.get("max_markets", len(markets)))
    markets = markets[:max_markets]

    _ensure_parent(paths["raw_markets"])
    with open(paths["raw_markets"], "w", encoding="utf-8") as f:
        json.dump(markets, f, indent=2)

    scheduled = filter_scheduled_markets(markets)

    histories: dict[str, list[dict]] = {}
    if fetch_cfg.get("include_price_history", True):
        interval = str(fetch_cfg.get("history_interval", "max"))
        fidelity = int(fetch_cfg.get("history_fidelity", 60))
        for m in scheduled:
            for tok in m.get("_tokens", []):
                token_id = tok.get("token_id")
                if not token_id or token_id in histories:
                    continue
                raw_points = client.fetch_price_history(token_id=token_id, interval=interval, fidelity=fidelity)
                histories[token_id] = normalize_history_points(raw_points)

    _ensure_parent(paths["raw_histories"])
    with open(paths["raw_histories"], "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    _ensure_parent(paths["filtered_markets"])
    with open(paths["filtered_markets"], "w", encoding="utf-8") as f:
        json.dump(scheduled, f, indent=2)

    return {
        "markets_fetched": len(markets),
        "scheduled_markets": len(scheduled),
        "histories_fetched": len(histories),
    }
