from __future__ import annotations

import csv
import json
from pathlib import Path


def build_dataset(config: dict) -> dict:
    paths = config["paths"]

    with open(paths["filtered_markets"], "r", encoding="utf-8") as f:
        scheduled = json.load(f)
    with open(paths["raw_histories"], "r", encoding="utf-8") as f:
        histories = json.load(f)

    rows = []
    for m in scheduled:
        market_id = m.get("id") or m.get("conditionId") or m.get("slug") or ""
        question = m.get("question") or m.get("title") or ""
        event_time = m.get("_event_time")
        for tok in m.get("_tokens", []):
            token_id = tok.get("token_id")
            token_name = tok.get("name")
            hist = histories.get(token_id, [])
            rows.append(
                {
                    "market_id": market_id,
                    "question": question,
                    "event_time": event_time,
                    "token_name": token_name,
                    "token_id": token_id,
                    "history_points": len(hist),
                }
            )

    out_path = Path(paths["dataset_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["market_id", "question", "event_time", "token_name", "token_id", "history_points"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"dataset_rows": len(rows), "output": str(out_path)}
