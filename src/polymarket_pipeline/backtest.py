from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import csv
import json
from math import sqrt
from pathlib import Path


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass
class Trade:
    market_id: str
    question: str
    token_id: str
    token_name: str
    event_time: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    return_pct: float


def _pick_entry_exit(history: list[dict], event_time: datetime, entry_mins: int, exit_mins: int):
    entry_cut = event_time - timedelta(minutes=entry_mins)
    exit_cut = event_time - timedelta(minutes=exit_mins)

    entry = None
    exitp = None
    for p in history:
        ts = _dt(p["timestamp"])
        if entry is None and ts >= entry_cut:
            entry = p
        if ts <= exit_cut:
            exitp = p
    return entry, exitp


def run_backtest(config: dict) -> dict:
    paths = config["paths"]
    strat = config["strategy"]
    entry_mins = int(strat.get("entry_minutes_before", 180))
    exit_mins = int(strat.get("exit_minutes_before", 15))
    min_hist = int(strat.get("min_history_points", 5))

    with open(paths["filtered_markets"], "r", encoding="utf-8") as f:
        scheduled = json.load(f)
    with open(paths["raw_histories"], "r", encoding="utf-8") as f:
        histories = json.load(f)

    trades: list[Trade] = []
    for m in scheduled:
        event_time_str = m.get("_event_time")
        if not event_time_str:
            continue
        event_time = _dt(event_time_str)
        market_id = str(m.get("id") or m.get("conditionId") or m.get("slug") or "")
        question = str(m.get("question") or m.get("title") or "")

        yes_token = None
        for tok in m.get("_tokens", []):
            name = str(tok.get("name", "")).lower()
            if name in ("yes", "true"):
                yes_token = tok
                break
        if yes_token is None and m.get("_tokens"):
            yes_token = m["_tokens"][0]
        if not yes_token:
            continue

        token_id = yes_token.get("token_id")
        token_name = yes_token.get("name", "YES")
        hist = histories.get(token_id, [])
        if len(hist) < min_hist:
            continue

        entry, exitp = _pick_entry_exit(hist, event_time, entry_mins, exit_mins)
        if not entry or not exitp:
            continue
        ep = float(entry["price"])
        xp = float(exitp["price"])
        if ep <= 0:
            continue
        ret = (xp - ep) / ep

        trades.append(
            Trade(
                market_id=market_id,
                question=question,
                token_id=token_id,
                token_name=token_name,
                event_time=event_time_str,
                entry_time=entry["timestamp"],
                exit_time=exitp["timestamp"],
                entry_price=ep,
                exit_price=xp,
                return_pct=ret,
            )
        )

    Path(paths["trades_csv"]).parent.mkdir(parents=True, exist_ok=True)
    with open(paths["trades_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys()) if trades else [
            "market_id","question","token_id","token_name","event_time","entry_time","exit_time","entry_price","exit_price","return_pct"
        ])
        writer.writeheader()
        for t in trades:
            writer.writerow(asdict(t))

    returns = [t.return_pct for t in trades]
    n = len(returns)
    avg = sum(returns)/n if n else 0.0
    med = sorted(returns)[n//2] if n else 0.0
    win = sum(1 for r in returns if r > 0)/n if n else 0.0
    std = (sum((r-avg)**2 for r in returns)/n)**0.5 if n else 0.0
    sharpe_like = (avg/std*sqrt(n)) if n and std > 0 else 0.0

    summary = {
        "num_trades": n,
        "avg_return_pct": avg,
        "median_return_pct": med,
        "win_rate": win,
        "sharpe_like": sharpe_like,
        "entry_minutes_before": entry_mins,
        "exit_minutes_before": exit_mins,
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
