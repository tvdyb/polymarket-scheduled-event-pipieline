from __future__ import annotations

from pathlib import Path
import json

from .api import PolymarketClient, normalize_history_points
from .filters import filter_scheduled_markets
from .llm_filter import apply_llm_filter


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

    # Keep only markets with accessible history on at least one token (prefer YES token when present)
    scheduled_with_history = []
    for m in scheduled:
        tokens = m.get("_tokens", [])
        yes = next((t for t in tokens if str(t.get("name", "")).lower() in ("yes", "true")), None)
        check_tokens = [yes] if yes else tokens[:1]
        has_history = any(histories.get(t.get("token_id"), []) for t in check_tokens if t)
        if has_history:
            scheduled_with_history.append(m)

    _ensure_parent(paths["raw_histories"])
    with open(paths["raw_histories"], "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    llm_cfg = config.get("llm", {})
    decisions = []
    if llm_cfg.get("enabled", False):
        scheduled_final, decisions = apply_llm_filter(scheduled_with_history, llm_cfg)
    else:
        scheduled_final = scheduled_with_history

    _ensure_parent(paths["filtered_markets"])
    with open(paths["filtered_markets"], "w", encoding="utf-8") as f:
        json.dump(scheduled_final, f, indent=2)

    decisions_path = paths.get("llm_decisions", "data/processed/llm_decisions.json")
    _ensure_parent(decisions_path)
    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)

    error_rejects = sum(1 for d in decisions if d.get("error"))
    policy_rejects = sum(1 for d in decisions if (not d.get("include")) and (not d.get("error")))

    return {
        "markets_fetched": len(markets),
        "scheduled_markets": len(scheduled),
        "scheduled_with_history": len(scheduled_with_history),
        "scheduled_after_llm": len(scheduled_final),
        "llm_policy_rejects": policy_rejects,
        "llm_error_rejects": error_rejects,
        "histories_fetched": len(histories),
        "llm_decisions_path": decisions_path,
    }
