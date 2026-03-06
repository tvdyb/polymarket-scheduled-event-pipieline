from __future__ import annotations

import re
from typing import Any

_MAJOR_LEAGUE_TERMS = {
    "nfl", "nba", "mlb", "nhl", "wnba", "ncaa", "ncaa football", "ncaa basketball",
    "premier league", "champions league", "europa league", "serie a", "la liga", "bundesliga",
    "ligue 1", "mls", "epl", "uefa", "fifa", "world cup", "copa america", "euros",
    "cricket world cup", "ipl", "ufc", "mma", "boxing", "pga", "masters tournament",
    "wimbledon", "roland garros", "us open tennis", "australian open", "formula 1", "f1",
    "nascar", "indycar", "super bowl", "stanley cup", "world series", "final four",
}

_SPORTS_GAME_TERMS = {
    "game", "match", "fight", "vs", "v.", "defeat", "win by", "spread", "moneyline",
    "touchdown", "quarterback", "goal scorer", "hat trick", "knockout", "submission",
    "round", "innings", "set", "overtime", "playoff", "playoffs", "semi-final", "semifinal",
    "final", "championship", "cup final", "bracket",
}

_SPORTS_DRAFT_TERMS = {
    "draft", "first overall pick", "top pick", "lottery pick", "combine", "mock draft",
}

_SPORTS_ENTITY_TERMS = {
    "lakers", "celtics", "yankees", "dodgers", "chiefs", "patriots", "cowboys", "arsenal",
    "man city", "real madrid", "barcelona", "inter miami", "maple leafs", "canucks",
    "warriors", "mavericks", "knicks", "uconn", "alabama",
}


def _text_blob(market: dict[str, Any]) -> str:
    bits = [
        str(market.get("question") or ""),
        str(market.get("title") or ""),
        str(market.get("description") or ""),
        str(market.get("rules") or ""),
        str(market.get("slug") or ""),
        str((market.get("event") or {}).get("title") if isinstance(market.get("event"), dict) else ""),
    ]
    return " \n".join(bits).lower()


def _contains_any(blob: str, terms: set[str]) -> bool:
    return any(t in blob for t in terms)


def classify_sports_market(market: dict[str, Any], allow_sports_draft_markets: bool = False) -> tuple[bool, str]:
    blob = _text_blob(market)

    if _contains_any(blob, _MAJOR_LEAGUE_TERMS) or _contains_any(blob, _SPORTS_ENTITY_TERMS):
        return True, "major_league_or_team_signal"

    has_sport_context = bool(re.search(r"\b(team|season|coach|player|tournament|league)\b", blob))
    if has_sport_context and _contains_any(blob, _SPORTS_GAME_TERMS):
        return True, "game_or_prop_signal"

    if _contains_any(blob, {"playoff", "playoffs", "championship", "super bowl", "world series", "stanley cup"}):
        return True, "playoff_or_championship_signal"

    if _contains_any(blob, _SPORTS_DRAFT_TERMS):
        if allow_sports_draft_markets:
            return False, "sports_draft_allowed"
        # only exclude drafts when sports specific
        if has_sport_context or _contains_any(blob, _MAJOR_LEAGUE_TERMS) or _contains_any(blob, _SPORTS_ENTITY_TERMS):
            return True, "sports_draft_signal"

    if re.search(r"\b(week\s+\d+|matchday\s+\d+|game\s+\d+)\b", blob):
        return True, "sports_schedule_signal"

    return False, "non_sports"
