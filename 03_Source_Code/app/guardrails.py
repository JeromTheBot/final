"""Input / output guardrails.

Input:  length & character checks, prompt-injection detection, PII redaction, domain-relevance check.
Output: hallucinated case-ID removal, unsafe-instruction filter, mandatory human-approval note for
        service-affecting actions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.query_parser import ParsedQuery

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?|rules)",
    r"disregard (all |the )?(previous|prior|above|system)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now", r"act as (an? )?(unrestricted|jailbroken|dan)", r"\bjailbreak\b", r"developer mode",
    r"<\s*script", r"\bdrop\s+table\b", r";\s*(rm|del|shutdown)\s", r"\bunion\s+select\b",
    r"forget (everything|your instructions)", r"override (the )?(safety|guardrails?)",
]
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
PII_PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}(?!\d)"),
    "IMSI": re.compile(r"(?<!\d)\d{15}(?!\d)"),
    "CARD": re.compile(r"(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)"),
}
UNSAFE_OUTPUT = re.compile(r"\b(disable (all )?(alarms|security|firewall|authentication)|factory reset all|"
                           r"delete (all )?(logs|configuration)|turn off (the )?(core|all sites))\b", re.I)
SERVICE_AFFECTING = re.compile(r"\b(reboot|restart|roll ?back|reset|lock|shut ?down|swap|dispatch)\b", re.I)


class GuardrailError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class GuardrailReport:
    sanitized_query: str
    redactions: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)


def validate_query(query: str, min_chars: int = 8, max_chars: int = 1000,
                   parsed: ParsedQuery | None = None, has_metrics: bool = False) -> GuardrailReport:
    if query is None or not str(query).strip():
        raise GuardrailError("EMPTY_QUERY", "Query must not be empty.")
    q = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(query))      # control characters
    q = re.sub(r"\s+", " ", q).strip()
    rep = GuardrailReport(sanitized_query=q)
    if len(q) < min_chars:
        raise GuardrailError("QUERY_TOO_SHORT", f"Query must be at least {min_chars} characters.")
    if len(q) > max_chars:
        raise GuardrailError("QUERY_TOO_LONG", f"Query must be at most {max_chars} characters.")
    rep.checks["length"] = "pass"

    for pat in _INJ:
        if pat.search(q):
            raise GuardrailError("PROMPT_INJECTION", "The query looks like a prompt-injection attempt and was blocked.")
    rep.checks["prompt_injection"] = "pass"

    for label, pat in PII_PATTERNS.items():
        q, n = pat.subn(f"[{label}]", q)
        if n:
            rep.redactions[label] = n
    rep.sanitized_query = q
    rep.checks["pii"] = "redacted" if rep.redactions else "pass"
    if rep.redactions:
        rep.warnings.append(f"Personal data redacted before processing: {rep.redactions}")

    if parsed is not None and parsed.domain_terms == 0 and not has_metrics:
        raise GuardrailError("OFF_TOPIC", "The query does not appear to be about telecom network operations. "
                                          "Describe network symptoms, e.g. 'high latency and packet loss in TX'.")
    rep.checks["domain_relevance"] = "pass"
    if parsed is not None and not parsed.conditions and not has_metrics:
        rep.warnings.append("No specific KPI symptom recognised; results rely on semantic similarity only.")
    return rep


def sanitize_output(text: str, valid_case_ids: set[str]) -> tuple[str, list[str]]:
    """Remove references to case IDs that were not in the retrieved evidence and block unsafe actions."""
    issues = []

    def _fix(m: re.Match) -> str:
        if m.group(0) in valid_case_ids:
            return m.group(0)
        issues.append(f"removed unsupported citation {m.group(0)}")
        return "[unverified case]"

    text = re.sub(r"NOC-\d{6}", _fix, text)
    if UNSAFE_OUTPUT.search(text):
        issues.append("unsafe instruction removed")
        text = UNSAFE_OUTPUT.sub("[action removed by safety policy]", text)
    return text, issues


def needs_human_approval(actions: list[str]) -> bool:
    return any(SERVICE_AFFECTING.search(a) for a in actions)
