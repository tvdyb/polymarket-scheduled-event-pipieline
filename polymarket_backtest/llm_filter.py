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

    # ── "Anytime before deadline" filter ─────────────────────────────────────
    # Markets with "by [date]" or "before [date]" can spike at any random moment.
    # No predictable quiet window to exploit.
    if re.search(r"\bby\b\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december|\d{4}|q[1-4]|end of|year|month)", question):
        return False
    if re.search(r"\bbefore\b\s+\w+", question):
        return False
    # "in 2025/2026" open-ended occurrence
    if re.search(r"\bin\s+20\d{2}\s*\??$", question):
        return False

    # Price target / "will reach" / "will hit" / "dip to" — can move any moment
    if re.search(r"\b(hit|reach|above|below|dip to|drop to|fall to|rise to|climb to)\b\s+\$?[\d.,]+", question):
        return False

    # Price in month/year: "Will X reach Y in March?" / "in 2026"
    if re.search(r"\$[\d.,]+\s+in\s+(january|february|march|april|may|june|july|august|september|october|november|december|\d{4})", question):
        return False

    # Geopolitical "can happen any day" patterns
    if re.search(r"\b(strike|attack|invade|capture|bomb|arrest|resign|oust|fire|impeach|indict|assassin)", question):
        return False

    # "Up or Down" / candle markets
    if "up or down" in question:
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

We want contracts that resolve on a SPECIFIC KNOWN DATE based on a SCHEDULED EVENT — where \
nothing price-moving happens until that date arrives. The key insight: we buy N days before \
the scheduled event and the price should be STABLE/BORING until the event occurs.

CRITICAL DISTINCTION — "scheduled date" vs "anytime before deadline":
- GOOD: "Will X beat quarterly earnings?" — earnings report is on a specific date, nothing \
happens until that date. Price is stable, then moves on earnings day.
- GOOD: "Will the Senate vote on X bill on March 15?" — vote is scheduled, price is stable until then.
- BAD: "Will Bitcoin hit $100K before July?" — Bitcoin can spike ANY DAY creating sudden vol. \
There is no quiet period to exploit.
- BAD: "Will GTA6 cost over $100?" — the announcement can come any time.
- BAD: "Will X be arrested by March 31?" — arrest can happen any day, no predictable quiet window.
- BAD: "Will there be a Category 5 hurricane in 2026?" — can happen any time during the season.
- BAD: "Will S&P 500 hit $6,800 in March?" — price can move any moment.
- BAD: "Will Israel strike Gaza on March 1?" — geopolitical event, unpredictable timing.

The test: "Is there a specific scheduled date where all the action happens, with a predictable \
quiet/stable period before it?" If the triggering event can happen at ANY RANDOM TIME before \
the deadline, EXCLUDE it.

INCLUDE:
- Scheduled data releases: earnings reports, economic data releases (GDP, jobs report on known date)
- Scheduled votes/decisions: congressional votes, court rulings on specific docket dates, Fed meetings
- Scheduled events with known dates: product launches on announced dates, award ceremonies, exams
- Regulatory deadlines: "Will FDA approve X by the PDUFA date?" (PDUFA date is known and scheduled)

EXCLUDE:
- "Before/by" deadline markets where the event can happen any time: price targets, arrests, \
resignations, wars, natural disasters, "will X reach Y", "will X happen by Z"
- Sports, awards with nominees, elections with candidates
- Continuous tracking: "Up or Down", 5-minute candles
- Markets where the underlying can move at any random moment

Respond in JSON only:
{
  "has_scheduled_date": true/false,
  "is_stable_before_event": true/false,
  "include_in_strategy": true/false,
  "event_date": "YYYY-MM-DD or null",
  "event_type": "earnings|data_release|scheduled_vote|court_ruling|product_launch|regulatory|other|null",
  "reasoning": "one sentence max",
  "confidence": 0.0-1.0
}

include_in_strategy = has_scheduled_date AND is_stable_before_event. Be STRICT — when in doubt, exclude."""


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
