"""Natural-language query understanding (deterministic layer).

Extracts KPI conditions ("packet loss" -> packet_loss), severity hints and metadata filters
(state, city, season, weather) from free text. The LLM can refine this, but the system never depends on it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.telecom_kb import CATEGORIES, CONDITIONS, SEASONS, WEATHER

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR", "guam": "GU", "district of columbia": "DC",
}
SEVERE_WORDS = ("severe", "critical", "very high", "extreme", "massive", "huge", "major", "very poor", "terrible",
                "significant", "drastic", "sharp")
SEASON_ALIASES = {"autumn": "Fall", "fall": "Fall", "winter": "Winter", "spring": "Spring", "summer": "Summer"}
WEATHER_ALIASES = {"rain": "Rain", "rainy": "Rain", "raining": "Rain", "storm": "Rain", "snow": "Snow",
                   "snowy": "Snow", "snowing": "Snow", "blizzard": "Snow", "cloud": "Clouds", "cloudy": "Clouds",
                   "clouds": "Clouds", "clear": "Clear", "sunny": "Clear"}

# multi-condition phrases engineers commonly use
PHRASES = {
    "poor service quality": ["high_latency", "packet_loss", "low_throughput"],
    "service degradation": ["high_latency", "packet_loss", "low_throughput"],
    "slow network": ["high_latency", "low_throughput"],
    "network congestion": ["high_load", "low_throughput"],
    "voice quality": ["high_jitter", "packet_loss"],
    "poor coverage": ["poor_signal"],
}
TELECOM_VOCAB = {
    "cell", "cells", "site", "sites", "network", "kpi", "kpis", "5g", "4g", "lte", "nr", "ran", "enodeb", "gnodeb",
    "tower", "antenna", "radio", "backhaul", "core", "noc", "incident", "incidents", "outage", "region", "traffic",
    "subscriber", "subscribers", "customers", "users", "mobile", "telecom", "operator", "carrier", "spectrum",
    "volte", "sla", "qos", "troubleshoot", "troubleshooting", "degradation", "service", "connectivity", "calls",
}


@dataclass
class ParsedQuery:
    text: str
    conditions: dict[str, int] = field(default_factory=dict)   # key -> severity (1 or 2)
    states: list[str] = field(default_factory=list)
    cities: list[str] = field(default_factory=list)
    seasons: list[str] = field(default_factory=list)
    weather: list[str] = field(default_factory=list)
    domain_terms: int = 0

    @property
    def condition_keys(self) -> list[str]:
        return list(self.conditions)

    def bm25_expansion(self) -> str:
        """Query expansion: canonical condition labels + filters (BM25 field vocabulary)."""
        from utils.telecom_kb import CONDITION_BY_KEY
        parts = [CONDITION_BY_KEY[k].label for k in self.conditions]
        parts += self.seasons + self.weather + self.states + self.cities
        return " ".join(parts)


def _has(term: str, text: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text) is not None


def parse_query(text: str, known_states: set[str] | None = None, known_cities: set[str] | None = None) -> ParsedQuery:
    low = text.lower()
    pq = ParsedQuery(text=text)

    def add(key: str, start: int) -> None:
        window = low[max(0, start - 25):start]
        sev = 2 if any(w in window for w in SEVERE_WORDS) else 1
        pq.conditions[key] = max(pq.conditions.get(key, 0), sev)

    for phrase, keys in PHRASES.items():
        if phrase in low:
            for k in keys:
                add(k, low.index(phrase))
    for c in CONDITIONS:
        for syn in sorted(c.synonyms, key=len, reverse=True):
            m = re.search(rf"(?<![a-z]){re.escape(syn)}(?:s|es)?(?![a-z])", low)
            if m:
                add(c.key, m.start())
                break
    # disambiguation: "critical alarm" also matches "alarm"; keep both (they are distinct KPIs).
    # "handover success declining" -> handover_failure already (direction is implicit in the condition).

    # --- metadata ---------------------------------------------------------------------------------
    for name, code in US_STATES.items():
        if _has(name, low) and (not known_states or code in known_states):
            pq.states.append(code)
    for tok in re.findall(r"\b[A-Z]{2}\b", text):
        if known_states and tok in known_states and tok not in pq.states and tok not in {"OR", "IN", "ME"}:
            pq.states.append(tok)
    for m in re.finditer(r"\b(?:in|at|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", text):
        cand = m.group(1)
        if known_cities:   # longest known prefix
            words = cand.split()
            for n in range(len(words), 0, -1):
                c = " ".join(words[:n])
                if c in known_cities:
                    pq.cities.append(c)
                    break
    for w, s in SEASON_ALIASES.items():
        if _has(w, low) and s not in pq.seasons:
            pq.seasons.append(s)
    for w, s in WEATHER_ALIASES.items():
        if _has(w, low) and s not in pq.weather:
            pq.weather.append(s)

    words = set(re.findall(r"[a-z0-9]+", low))
    pq.domain_terms = len(words & TELECOM_VOCAB) + len(pq.conditions)
    return pq


def categories_from_conditions(conditions: dict[str, int]) -> dict[str, float]:
    scores = {}
    for cat, spec in CATEGORIES.items():
        s = sum(conditions.get(k, 0) * w for k, w in spec["conditions"].items())
        if s:
            scores[cat] = s
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


assert set(SEASON_ALIASES.values()) <= set(SEASONS) and set(WEATHER_ALIASES.values()) <= set(WEATHER)
