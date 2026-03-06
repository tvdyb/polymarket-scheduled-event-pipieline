"""Phase 2: Filter pipeline — hard rules + DeepSeek LLM classification.

All filtering happens BEFORE any price data is fetched. This is the lookahead firewall.

Targets: scheduled binary event contracts with monotonic price potential.
Excludes: sports, multi-outcome competitive markets, awards, elections.
"""

import asyncio
import json
import re
from datetime import datetime, timezone

import openai
from tqdm import tqdm

from .config import (
    SPORTS_KEYWORDS,
    AWARDS_KEYWORDS,
    ELECTION_KEYWORDS,
    MIN_VOLUME,
    MIN_TRADING_DAYS,
    MAX_EVENT_GROUP_SIZE,
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
    """Multi-layer keyword + structural filters. Returns True if market passes."""
    text = (market["question"] + " " + market.get("description", "") + " " + market.get("category", "")).lower()
    question = market["question"].lower()

    # ── Sports filter ───────────────────────────────────────────────────────
    if any(kw in text for kw in SPORTS_KEYWORDS):
        return False

    # Category-level sports filter
    cat = market.get("category", "").lower()
    if cat in ("sports", "football", "soccer", "basketball", "baseball", "hockey",
               "tennis", "cricket", "mma", "nascar", "f1", "golf", "esports"):
        return False

    # Pattern: "Will [Team] win on YYYY-MM-DD?" — sports match result
    if re.search(r"will .+ win on \d{4}-\d{2}-\d{2}", question):
        return False

    # Pattern: "[Team] vs [Team]" anywhere in question
    if re.search(r"\w+ vs\.? \w+", question) and ("win" in question or "draw" in question or "score" in question):
        return False

    # ── Awards/competition filter ───────────────────────────────────────────
    if any(kw in text for kw in AWARDS_KEYWORDS):
        return False

    # ── Election/multi-candidate filter ─────────────────────────────────────
    if any(kw in text for kw in ELECTION_KEYWORDS):
        return False

    # ── Multi-outcome structural filter ─────────────────────────────────────
    # If >2 contracts share the same event, it's a competitive multi-outcome market
    if market.get("event_group_size", 1) > MAX_EVENT_GROUP_SIZE:
        return False

    # ── Minimum trading window ──────────────────────────────────────────────
    start = _parse_date(market.get("start_date"))
    end = _parse_date(market.get("end_date"))
    if start and end:
        if (end - start).days < MIN_TRADING_DAYS:
            return False

    # ── Minimum volume ──────────────────────────────────────────────────────
    if market.get("volume", 0) < MIN_VOLUME:
        return False

    return True


# ── Step 2b: LLM Classification ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a prediction market analyst filtering contracts for a specific trading strategy.

We want SCHEDULED BINARY EVENT contracts where:
- The contract resolves based on ONE specific event (not ongoing/continuous outcomes)
- The price can trend monotonically upward as the event approaches
- Evidence accumulates over time pushing the price toward 1.00
- The contract is NOT constrained by competing sibling contracts

INCLUDE these types:
- Threshold/milestone: "Will Bitcoin hit $100K before July 2026?", "Will US GDP exceed 3%?"
- Policy/regulatory: "Will the Fed cut rates in June?", "Will X bill pass the Senate?"
- Occurrence: "Will there be a Category 5 hurricane in 2026?", "Will X announce layoffs?"
- Deadline-based: "Will X happen before Y date?" where evidence accumulates toward resolution
- Niche scheduled events: "Will Kim K pass the bar exam?", "Will X movie gross over $Y?"

EXCLUDE these types:
- Sports: Any match, game, score, player stat, tournament result
- Awards with multiple nominees: "Will X win Best Picture?" — zero-sum across nominee contracts
- Elections with multiple candidates: "Will X win the primary?" — linked candidate contracts
- "Who will win/be chosen" markets with enumerated options
- Markets that are part of a multi-option slate (multiple contracts under same event where one winning forces others to lose)
- Continuous price tracking: "Up or Down" markets, 5-minute candle markets

Also estimate the EVENT DATE (when the underlying event resolves, not the market close date).

Respond in JSON only:
{
  "is_single_event": true/false,
  "has_monotonic_potential": true/false,
  "is_competitive_multioutcome": true/false,
  "include_in_strategy": true/false,
  "event_date": "YYYY-MM-DD or null",
  "event_type": "threshold|policy|occurrence|deadline|announcement|other|null",
  "reasoning": "one sentence max",
  "confidence": 0.0-1.0
}

include_in_strategy = is_single_event AND has_monotonic_potential AND NOT is_competitive_multioutcome"""


MAX_CONCURRENT = 100


async def _classify_market_async(
    client: openai.AsyncOpenAI,
    market: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, dict | None]:
    """Classify a single market with concurrency control."""
    user_content = (
        f"Question: {market['question']}\n\n"
        f"Description: {market.get('description', 'N/A')[:500]}\n\n"
        f"Category: {market.get('category', 'N/A')}\n"
        f"Event group size: {market.get('event_group_size', 1)}"
    )

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
    """Classify markets using DeepSeek with concurrent async calls."""
    cache_path = CACHE_DIR / "llm_classifications.jsonl"

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

    for m in markets:
        mid = m["id"]
        if mid in cached:
            entry = cached[mid]
            all_classifications.append(entry)
            classification = entry.get("classification", {})
            if classification and (
                classification.get("include_in_strategy")
                and classification.get("confidence", 0) >= LLM_CONFIDENCE_THRESHOLD
            ):
                m_with_llm = dict(m)
                m_with_llm["_llm"] = classification
                accepted.append(m_with_llm)
        else:
            to_classify.append(m)

    if to_classify:
        print(f"Classifying {len(to_classify)} markets with DeepSeek "
              f"({len(cached)} cached, {MAX_CONCURRENT} concurrent)...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        results = asyncio.run(_classify_batch_async(to_classify))

        with open(cache_path, "a", encoding="utf-8") as cache_f:
            for market, classification in results:
                entry = {
                    "market_id": market["id"],
                    "question": market["question"],
                    "classification": classification,
                }

                if classification and (
                    classification.get("include_in_strategy")
                    and classification.get("confidence", 0) >= LLM_CONFIDENCE_THRESHOLD
                ):
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

    after_hard = [m for m in markets if hard_filter(m)]
    print(f"After hard filters: {len(after_hard)}/{total}")

    accepted, classifications = llm_filter(after_hard)

    stats = {
        "total_markets": total,
        "after_hard_filter": len(after_hard),
        "after_llm_filter": len(accepted),
        "total_classifications": len(classifications),
    }

    return accepted, stats
