# Polymarket Scheduled Event Pipeline

Practical Python pipeline for researching **scheduled/timed Polymarket contracts** and backtesting a simple strategy:

- **Enter** X minutes before event start
- **Exit** Y minutes before event start

This repo is designed to work even when some API fields are inconsistent by using resilient adapters and documented assumptions.

## What it does

1. **Fetch markets** from Polymarket Gamma API (metadata)
2. **Fetch token price history** (where available)
3. **Filter scheduled events** with parseable start times
4. **Build dataset** joining market metadata + per-token timeseries
5. **Backtest** a pre-event entry/exit strategy across selected events
6. Expose everything via CLI commands

## Project structure

```text
polymarket-scheduled-event-pipeline/
  src/polymarket_pipeline/
    api.py              # Polymarket client + resilient adapters
    config.py           # YAML config loader
    ingest.py           # Fetch markets and prices, save raw JSON
    dataset.py          # Build tabular dataset
    filters.py          # Scheduled-event filters
    backtest.py         # Strategy simulation + metrics
    cli.py              # CLI entrypoint
  config/config.example.yaml
  tests/
  pyproject.toml
  requirements.txt
```

## Quickstart

```bash
cd polymarket-scheduled-event-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy config:

```bash
cp config/config.example.yaml config/config.yaml
```

## CLI usage

### 1) Fetch market metadata + price history

```bash
python -m polymarket_pipeline.cli fetch --config config/config.yaml
```

### 2) Build clean dataset

```bash
python -m polymarket_pipeline.cli build-dataset --config config/config.yaml
```

### 3) Run backtest

```bash
python -m polymarket_pipeline.cli backtest --config config/config.yaml
```

## Assumptions and API resilience

Polymarket/Gamma schemas can vary across markets. This code:

- Tries multiple likely timestamp fields: `startDate`, `startTime`, `endDate`, etc.
- Accepts token containers from `outcomes`, `tokens`, and fallback keys
- Handles missing history gracefully (skips markets/tokens without usable timeseries)
- Uses midpoint of OHLC if direct price is unavailable

## Strategy definition

Given each event start time:

- Enter at first observed price at or after `event_start - entry_minutes_before`
- Exit at last observed price at or before `event_start - exit_minutes_before`
- Long YES token by default

Outputs:

- Trade-level CSV
- Summary JSON (count, win rate, avg return, median return, Sharpe-like ratio)

## Testing

```bash
pytest -q
```

Tests are lightweight unit tests for adapters/filter/backtest logic.

