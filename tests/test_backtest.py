import json
from pathlib import Path

from polymarket_pipeline.backtest import run_backtest


def test_run_backtest_smoke(tmp_path: Path):
    filtered = [
        {
            "id": "m1",
            "question": "Test market",
            "_event_time": "2026-01-01T12:00:00+00:00",
            "_tokens": [{"name": "Yes", "token_id": "tok_yes"}],
        }
    ]
    histories = {
        "tok_yes": [
            {"timestamp": "2026-01-01T08:00:00+00:00", "price": 0.40},
            {"timestamp": "2026-01-01T09:00:00+00:00", "price": 0.42},
            {"timestamp": "2026-01-01T10:00:00+00:00", "price": 0.45},
            {"timestamp": "2026-01-01T11:30:00+00:00", "price": 0.48},
            {"timestamp": "2026-01-01T11:45:00+00:00", "price": 0.50},
        ]
    }

    raw = tmp_path / "raw"
    proc = tmp_path / "proc"
    raw.mkdir()
    proc.mkdir()

    filtered_path = proc / "scheduled_markets.json"
    hist_path = raw / "histories.json"
    trades_path = proc / "trades.csv"
    summary_path = proc / "summary.json"

    filtered_path.write_text(json.dumps(filtered), encoding="utf-8")
    hist_path.write_text(json.dumps(histories), encoding="utf-8")

    cfg = {
        "strategy": {"entry_minutes_before": 180, "exit_minutes_before": 15, "min_history_points": 3},
        "paths": {
            "filtered_markets": str(filtered_path),
            "raw_histories": str(hist_path),
            "trades_csv": str(trades_path),
            "summary_json": str(summary_path),
        },
    }
    summary = run_backtest(cfg)
    assert summary["num_trades"] == 1
    assert trades_path.exists()
    assert summary_path.exists()
