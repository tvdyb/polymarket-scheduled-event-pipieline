"""Phase 2: Filter pipeline — hard rules + DeepSeek LLM classification.

All filtering happens BEFORE any price data is fetched. This is the lookahead firewall.
"""

import asyncio
import json
import time
from datetime import datetime, timezone

import openai
from tqdm import tqdm

from .config import (
    SPORTS_KEYWORDS,
    MIN_VOLUME,
    MIN_TRADING_DAYS,
    LLM_CONFIDENCE_THRESHOLD,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
    CACHE_DIR,
)


# ── Step 2a: Hard Rule Filters ──────────────────────────────────────────────

def _parse_date(s) -> datetime | None:
    if not s:
        return None
    if isinstance(s, (int, float)):
        v = float(s)
        if v > 1e12:
            v /= 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (ValueError, OSError):
            return None
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def hard_filter(market: dict) -> bool:
    """Cheap keyword + heuristic filters. Returns True if market passes."""
    text = (market["question"] + " " + market.get("category", "")).lower()
    if any(kw in text for kw in SPORTS_KEYWORDS):
        return False

    start = _parse_date(market.get("start_date"))
    end = _parse_date(market.get("end_date"))
    if start and end:
        if (end - start).days < MIN_TRADING_DAYS:
            return False

    if market.get("volume", 0) < MIN_VOLUME:
        return False

    # Filter out multi-outcome/contestant markets (e.g. "Will X win Best Actress?"
    # where many nominees each have their own contract — most must go to zero)
    # Threshold at 5: catches large nominee pools while keeping legitimate markets
    # that happen to share an event group with a few related markets
    if market.get("event_group_size", 1) > 5:
        return False

    return True


# ── Step 2b: LLM Classification ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a prediction market analyst evaluating whether a market fits a specific \
trading strategy: buying markets for obscure, hard-dated events that most traders \
ignore until the event actually happens.

Given a market question and description, determine:

1. SCHEDULED EVENT: Does this resolve based on a specific scheduled event with a \
known date — like an exam, hearing, product launch, award ceremony, court date, \
or scheduled announcement? The key is that the event date is KNOWABLE IN ADVANCE. \
NOT "will X happen eventually" but "will X happen ON a specific known occasion."

2. LOW ATTENTION: Would this market likely be IGNORED by most traders before the \
event? Niche celebrity events, obscure legal proceedings, minor product launches, \
reality TV outcomes — things that only become salient right before they happen. \
Exclude markets that are inherently high-attention (major elections, Fed decisions, \
big IPOs, anything financial professionals actively track).

3. EVENT DATE: Your best estimate of when the underlying event occurs (not the \
market resolution close date, which is often days later).

Respond in JSON only — no preamble, no explanation outside the JSON:
{
  "is_scheduled_event": true/false,
  "is_low_attention": true/false,
  "include_in_strategy": true/false,
  "event_date": "YYYY-MM-DD or null",
  "event_type": "exam|hearing|launch|award|court|announcement|other|null",
  "reasoning": "one sentence max",
  "confidence": 0.0-1.0
}

include_in_strategy should be true only if BOTH is_scheduled_event AND is_low_attention are true.

Examples of markets to INCLUDE (include_in_strategy: true):
- "Will Kim Kardashian pass the California bar exam?" → Scheduled exam date, celebrity but niche legal event, ignored until results day
- "Will Olivia Rodrigo win Best New Artist at the Grammys?" → Award ceremony has a fixed date, pop culture not finance-tracked
- "Will [person] be sworn in as [position] on January 20th?" → Scheduled inauguration, date is known
- "Will the Supreme Court rule on [case] before June recess?" → Court calendar is scheduled, legal niche
- "Will [movie] gross over $X in its opening weekend?" → Release date is fixed, entertainment niche

Examples of markets to EXCLUDE (include_in_strategy: false):
- "Will the Fed raise rates at the March FOMC meeting?" → Scheduled, but HEAVILY tracked by financial professionals — not low attention
- "Will Bitcoin reach $100K by end of year?" → No specific event date
- "Will [team] win the Super Bowl?" → Major sports, too sharp
- "Will [politician] win the 2024 election?" → Major election, high attention
- "Will [big tech company] hit $X market cap?" → Finance-tracked, not niche"""


MAX_CONCURRENT = 100  # concurrent API requests


async def _classify_market_async(
    client: openai.AsyncOpenAI,
    market: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, dict | None]:
    """Classify a single market with concurrency control. Returns (market, classification)."""
    user_content = f"Question: {market['question']}\n\nDescription: {market.get('description', 'N/A')}"

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=300,
                )
                text = response.choices[0].message.content or ""
                start = text.find("{")
                end = text.rfind("}")
                if start == -1 or end == -1:
                    return market, None
                return market, json.loads(text[start:end + 1])
            except openai.RateLimitError:
                await asyncio.sleep(2.0 * (attempt + 1))
            except Exception as e:
                print(f"  LLM error for '{market['question'][:60]}': {e}")
                return market, None

    return market, None


async def _classify_batch_async(
    markets: list[dict],
    concurrency: int = MAX_CONCURRENT,
) -> list[tuple[dict, dict | None]]:
    """Classify all markets concurrently with a semaphore."""
    client = openai.AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [_classify_market_async(client, m, semaphore) for m in markets]

    results = []
    with tqdm(total=len(tasks), desc="LLM classification") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    await client.close()
    return results


def llm_filter(markets: list[dict]) -> tuple[list[dict], list[dict]]:
    """Classify markets using DeepSeek with concurrent async calls.

    Caches results to llm_classifications.jsonl.
    """
    cache_path = CACHE_DIR / "llm_classifications.jsonl"

    # Load existing cache
    cached = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    cached[entry["market_id"]] = entry

    if not DEEPSEEK_API_KEY:
        print("WARNING: DEEPSEEK_API_KEY not set. Skipping LLM classification.")
        print("  Set it with: export DEEPSEEK_API_KEY='sk-...'")
        return markets, []

    accepted = []
    all_classifications = []
    to_classify = []

    # Separate cached from uncached
    for m in markets:
        mid = m["id"]
        if mid in cached:
            entry = cached[mid]
            all_classifications.append(entry)
            classification = entry.get("classification", {})
            if (classification.get("include_in_strategy") and
                    classification.get("confidence", 0) >= LLM_CONFIDENCE_THRESHOLD):
                m_with_llm = dict(m)
                m_with_llm["_llm"] = classification
                accepted.append(m_with_llm)
        else:
            to_classify.append(m)

    if to_classify:
        print(f"Classifying {len(to_classify)} markets with DeepSeek "
              f"({len(cached)} cached, {MAX_CONCURRENT} concurrent)...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Run async classification
        results = asyncio.run(_classify_batch_async(to_classify))

        # Process results and write cache
        with open(cache_path, "a", encoding="utf-8") as cache_f:
            for market, classification in results:
                entry = {
                    "market_id": market["id"],
                    "question": market["question"],
                    "classification": classification,
                }

                if classification:
                    if (classification.get("include_in_strategy") and
                            classification.get("confidence", 0) >= LLM_CONFIDENCE_THRESHOLD):
                        m_with_llm = dict(market)
                        m_with_llm["_llm"] = classification
                        accepted.append(m_with_llm)

                all_classifications.append(entry)
                cache_f.write(json.dumps(entry) + "\n")

    print(f"LLM filter: {len(accepted)}/{len(markets)} markets accepted")
    return accepted, all_classifications


def run_filter_pipeline(markets: list[dict]) -> tuple[list[dict], dict]:
    """Run the full Phase 2 filter pipeline. Returns (filtered_markets, stats)."""
    total = len(markets)

    # Step 2a: Hard filters
    after_hard = [m for m in markets if hard_filter(m)]
    print(f"After hard filters: {len(after_hard)}/{total}")

    # Step 2b: LLM classification
    accepted, classifications = llm_filter(after_hard)

    stats = {
        "total_markets": total,
        "after_hard_filter": len(after_hard),
        "after_llm_filter": len(accepted),
        "total_classifications": len(classifications),
    }

    return accepted, stats
