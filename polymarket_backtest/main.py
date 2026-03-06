"""Orchestrator for the Polymarket Low-Volatility Event Backtest System."""

import argparse
import json
import sys

from .config import ensure_dirs, CACHE_DIR, OUTPUT_DIR
from .fetch_markets import fetch_all_markets, parse_market
from .llm_filter import run_filter_pipeline
from .price_data import fetch_prices_for_markets
from .backtest import run_backtest
from .results import print_console_summary, export_csv, generate_charts


def step_fetch():
    """Phase 1: Fetch historic markets."""
    print("=" * 60)
    print("PHASE 1: Fetching historic markets")
    print("=" * 60)
    raw_markets = fetch_all_markets()
    markets = [parse_market(m) for m in raw_markets]
    # Save parsed markets
    parsed_path = CACHE_DIR / "markets_parsed.jsonl"
    with open(parsed_path, "w", encoding="utf-8") as f:
        for m in markets:
            # Don't serialize _raw to parsed cache
            m_clean = {k: v for k, v in m.items() if k != "_raw"}
            f.write(json.dumps(m_clean) + "\n")
    print(f"Parsed {len(markets)} markets -> {parsed_path}")
    return markets


def step_filter(markets: list[dict] | None = None):
    """Phase 2: Filter pipeline (hard rules + LLM). No price data touched."""
    print("\n" + "=" * 60)
    print("PHASE 2: Filter pipeline (lookahead firewall)")
    print("=" * 60)

    if markets is None:
        parsed_path = CACHE_DIR / "markets_parsed.jsonl"
        if not parsed_path.exists():
            print("No parsed markets found. Run --step fetch first.")
            sys.exit(1)
        markets = []
        with open(parsed_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    markets.append(json.loads(line))

    filtered, stats = run_filter_pipeline(markets)

    # Save filtered market list (locked before prices)
    filtered_path = CACHE_DIR / "markets_filtered.jsonl"
    with open(filtered_path, "w", encoding="utf-8") as f:
        for m in filtered:
            m_clean = {k: v for k, v in m.items() if k != "_raw"}
            f.write(json.dumps(m_clean) + "\n")

    stats_path = CACHE_DIR / "filter_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Filter complete. {len(filtered)} markets saved to {filtered_path}")
    return filtered, stats


def step_prices(markets: list[dict] | None = None):
    """Phase 3: Fetch prices for filtered markets ONLY."""
    print("\n" + "=" * 60)
    print("PHASE 3: Fetching price history (filtered markets only)")
    print("=" * 60)

    if markets is None:
        filtered_path = CACHE_DIR / "markets_filtered.jsonl"
        if not filtered_path.exists():
            print("No filtered markets found. Run --step filter first.")
            sys.exit(1)
        markets = []
        with open(filtered_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    markets.append(json.loads(line))

    prices = fetch_prices_for_markets(markets)

    # Count markets with sufficient data
    markets_with_prices = sum(1 for mid in prices if len(prices[mid]) >= 5)
    price_stats = {"markets_with_prices": markets_with_prices, "total_tokens_fetched": len(prices)}

    stats_path = CACHE_DIR / "price_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(price_stats, f, indent=2)

    print(f"Price data available for {markets_with_prices} markets")
    return prices, price_stats


def step_backtest(markets: list[dict] | None = None, prices: dict | None = None):
    """Phase 4: Run backtest."""
    print("\n" + "=" * 60)
    print("PHASE 4: Running backtest")
    print("=" * 60)

    if markets is None:
        filtered_path = CACHE_DIR / "markets_filtered.jsonl"
        if not filtered_path.exists():
            print("No filtered markets found. Run --step filter first.")
            sys.exit(1)
        markets = []
        with open(filtered_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    markets.append(json.loads(line))

    if prices is None:
        prices_path = CACHE_DIR / "price_histories.json"
        if not prices_path.exists():
            print("No price data found. Run --step prices first.")
            sys.exit(1)
        # Rebuild prices dict keyed by market_id
        with open(prices_path, "r", encoding="utf-8") as f:
            token_prices = json.load(f)
        # Map market_id -> prices using token_id
        prices = {}
        for m in markets:
            token_ids = m.get("clob_token_ids", [])
            if token_ids:
                token_id = token_ids[0]
                if token_id in token_prices:
                    prices[m["id"]] = token_prices[token_id]

    all_results = run_backtest(markets, prices)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "backtest_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({k: len(v) for k, v in all_results.items()}, f, indent=2)

    total_trades = sum(len(v) for v in all_results.values())
    print(f"Backtest complete. {total_trades} total trades across {len(all_results)} combos")
    return all_results


def step_results(all_results: dict | None = None, filter_stats: dict | None = None, price_stats: dict | None = None):
    """Phase 5: Print summary + export CSV + generate charts."""
    print("\n" + "=" * 60)
    print("PHASE 5: Results")
    print("=" * 60)

    if filter_stats is None:
        stats_path = CACHE_DIR / "filter_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                filter_stats = json.load(f)
        else:
            filter_stats = {}

    if price_stats is None:
        stats_path = CACHE_DIR / "price_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                price_stats = json.load(f)
        else:
            price_stats = {}

    if all_results is None:
        # Re-run backtest from cached data
        all_results = step_backtest()

    print_console_summary(all_results, filter_stats, price_stats)
    export_csv(all_results, combo="(7,1)")
    generate_charts(all_results, combo="(7,1)")


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Low-Volatility Event Backtest System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps (run in order, or omit --step for full pipeline):
  fetch      Pull & cache raw markets (no prices)
  filter     Hard rules + LLM classification (no prices)
  prices     Fetch prices for filtered set ONLY
  backtest   Run backtest
  results    Print summary + export CSV + charts
        """,
    )
    parser.add_argument(
        "--step",
        choices=["fetch", "filter", "prices", "backtest", "results"],
        help="Run a specific pipeline step",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.step == "fetch":
        step_fetch()
    elif args.step == "filter":
        step_filter()
    elif args.step == "prices":
        step_prices()
    elif args.step == "backtest":
        step_backtest()
    elif args.step == "results":
        step_results()
    else:
        # Full pipeline
        markets = step_fetch()
        filtered, filter_stats = step_filter(markets)
        prices, price_stats = step_prices(filtered)
        all_results = step_backtest(filtered, prices)
        step_results(all_results, filter_stats, price_stats)


if __name__ == "__main__":
    main()
