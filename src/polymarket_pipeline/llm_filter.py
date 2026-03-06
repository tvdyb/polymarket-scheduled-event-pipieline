from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import time

import requests

from .sports import classify_sports_market


def _default_prompt() -> str:
    return (
        "Decide if this Polymarket market is suitable for a pre-event strategy: buy before an anchor event and sell right before that anchor event. "
        "Hard-exclude major sports markets (NFL/NBA/MLB/NHL/UFC/boxing/soccer major leagues/tournaments). "
        "Allow multiple related events close together if there is one practical tradable anchor event window (e.g., NFL draft start for top-10 pick markets). "
        "Infer the most practical event datetime in UTC for trading, not just generic resolution deadlines. "
        "Return ONLY strict JSON with keys: include (bool), reason (string), score (0-100), event_datetime_utc (ISO-8601 or null), event_time_confidence (high|medium|low), event_time_rationale (string)."
    )


def _load_prompt(prompt_path: str | None) -> str:
    if prompt_path and Path(prompt_path).exists():
        return Path(prompt_path).read_text(encoding="utf-8")
    return _default_prompt()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _parse_string_decision(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    if not t:
        return None
    u = t.upper()
    include = None
    if "INCLUDE" in u:
        include = True
    if "EXCLUDE" in u or "REJECT" in u:
        if include is None:
            include = False
    if include is None:
        return None

    score = 50
    import re

    m = re.search(r"score\s*[:=]\s*([0-9]{1,3})", t, flags=re.I)
    if m:
        try:
            score = max(0, min(100, int(m.group(1))))
        except Exception:
            pass

    reason = t
    if "reason" in u:
        m2 = re.search(r"reason\s*[:=]\s*(.*)$", t, flags=re.I | re.S)
        if m2:
            reason = m2.group(1).strip()

    dt = None
    m_dt = re.search(r"event_datetime_utc\s*[:=]\s*([^;\n]+)", t, flags=re.I)
    if m_dt:
        dt = m_dt.group(1).strip().strip('"')

    conf = "low"
    m_conf = re.search(r"event_time_confidence\s*[:=]\s*(high|medium|low)", t, flags=re.I)
    if m_conf:
        conf = m_conf.group(1).lower()

    rationale = ""
    m_rat = re.search(r"event_time_rationale\s*[:=]\s*(.*)$", t, flags=re.I | re.S)
    if m_rat:
        rationale = m_rat.group(1).strip()

    return {
        "include": include,
        "reason": reason[:500],
        "score": score,
        "event_datetime_utc": dt,
        "event_time_confidence": conf,
        "event_time_rationale": rationale[:300],
    }


def _call_anthropic(market: dict[str, Any], sys_prompt: str, model: str, api_key_env: str, timeout: int) -> dict[str, Any]:
    api_key = os.getenv(api_key_env)
    if not api_key:
        return {"include": False, "reason": f"LLM request failed status=missing_key:{api_key_env}", "score": 0}

    payload = {
        "model": model,
        "max_tokens": 140,
        "temperature": 0,
        "system": sys_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "question": market.get("question") or market.get("title"),
                                "event_time": market.get("_event_time"),
                                "tokens": market.get("_tokens", []),
                                "rules": market.get("description") or market.get("rules"),
                                "slug": market.get("slug"),
                            }
                        ),
                    }
                ],
            }
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    for attempt in range(4):
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=timeout)
        if r.status_code == 429 and attempt < 3:
            time.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code >= 400:
            return {"include": False, "reason": f"LLM request failed status={r.status_code}", "score": 0}
        data = r.json()
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        parsed = _parse_json_object(text)
        if not parsed:
            return {"include": False, "reason": "LLM invalid JSON", "score": 0}
        return {
            "include": bool(parsed.get("include", False)),
            "reason": str(parsed.get("reason", "")),
            "score": int(float(parsed.get("score", 0))),
        }

    return {"include": False, "reason": "LLM request failed status=429", "score": 0}


def _call_deepseek(market: dict[str, Any], sys_prompt: str, model: str, api_key_env: str, timeout: int) -> dict[str, Any]:
    api_key = os.getenv(api_key_env)
    if not api_key:
        return {"include": False, "reason": f"LLM request failed status=missing_key:{api_key_env}", "score": 0}

    user_payload = {
        "question": market.get("question") or market.get("title"),
        "event_time": market.get("_event_time"),
        "tokens": market.get("_tokens", []),
        "rules": market.get("description") or market.get("rules"),
        "slug": market.get("slug"),
    }

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 120,
        "messages": [
            {
                "role": "system",
                "content": sys_prompt
                + " Reply in plain text format: INCLUDE|EXCLUDE; score=<0-100>; reason=<short>. No markdown.",
            },
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(4):
        r = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=payload, timeout=timeout)
        if r.status_code == 429 and attempt < 3:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            return {"include": False, "reason": f"LLM request failed status={r.status_code}", "score": 0}
        data = r.json()
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        parsed = _parse_string_decision(text)
        if not parsed:
            parsed = _parse_json_object(text)
        if not parsed:
            return {"include": False, "reason": "LLM unparseable response", "score": 0}
        return {
            "include": bool(parsed.get("include", False)),
            "reason": str(parsed.get("reason", "")),
            "score": int(float(parsed.get("score", 0))),
        }

    return {"include": False, "reason": "LLM request failed status=429", "score": 0}


def apply_llm_filter(markets: list[dict[str, Any]], llm_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provider = str(llm_cfg.get("provider", "anthropic")).lower()
    model = llm_cfg.get("model", "claude-haiku-4-5-20251001")
    prompt_path = llm_cfg.get("prompt_path")
    api_key_env = llm_cfg.get("api_key_env", "ANTHROPIC_API_KEY")
    timeout = int(llm_cfg.get("timeout_seconds", 25))
    sys_prompt = _load_prompt(prompt_path)

    sports_terms = (
        "nfl", "nba", "mlb", "nhl", "ufc", "boxing", "super bowl", "world cup", "premier league", "champions league", "matchday"
    )

    kept: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for m in markets:
        q = str(m.get("question") or m.get("title") or "")
        ql = q.lower()

        # hard exclude major sports
        if any(t in ql for t in sports_terms):
            d = {
                "include": False,
                "score": 0,
                "reason": "Excluded: major sports market",
                "event_datetime_utc": None,
                "event_time_confidence": "low",
                "event_time_rationale": "sports hard filter",
            }
        else:
            if provider == "deepseek":
                d = _call_deepseek(m, sys_prompt, model, api_key_env, timeout)
            else:
                d = _call_anthropic(m, sys_prompt, model, api_key_env, timeout)

        if not d.get("event_datetime_utc"):
            d["event_datetime_utc"] = m.get("_event_time")

        m2 = dict(m)
        m2["_llm_decision"] = d
        decisions.append(
            {
                "question": q,
                "slug": m.get("slug", ""),
                "event_time": m.get("_event_time"),
                "event_datetime_utc": d.get("event_datetime_utc"),
                "event_time_confidence": d.get("event_time_confidence", "low"),
                "event_time_rationale": d.get("event_time_rationale", ""),
                "include": d.get("include", False),
                "score": d.get("score", 0),
                "reason": d.get("reason", ""),
                "error": str(d.get("reason", "")).startswith("LLM"),
            }
        )
        if d.get("include"):
            kept.append(m2)

    return kept, decisions
