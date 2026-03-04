from __future__ import annotations

import argparse
import json

from .config import load_config
from .ingest import fetch_and_store
from .dataset import build_dataset
from .backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket scheduled-event pipeline")
    parser.add_argument("command", choices=["fetch", "build-dataset", "backtest", "run-all"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "fetch":
        out = fetch_and_store(cfg)
    elif args.command == "build-dataset":
        out = build_dataset(cfg)
    elif args.command == "backtest":
        out = run_backtest(cfg)
    else:
        out = {
            "fetch": fetch_and_store(cfg),
            "dataset": build_dataset(cfg),
            "backtest": run_backtest(cfg),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
