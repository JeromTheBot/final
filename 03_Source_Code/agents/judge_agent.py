"""LLM-as-judge diagnosis validation agent.

An independent reviewer scores the diagnosis/recommendation against the retrieved evidence. If the
verdict is "revise", the orchestrator sends the issues back to the recommendation agent (one revision loop).
Without an LLM, a deterministic evidence-consistency judge computes the same five criteria.
"""
from __future__ import annotations

import re
from collections import Counter

from agents import prompts
from agents.base import Agent, AgentContext
from app.guardrails import SERVICE_AFFECTING, UNSAFE_OUTPUT
from utils.telecom_kb import CATEGORIES, CONDITION_BY_KEY, PLAYBOOK

CRITERIA = ["groundedness", "diagnostic_correctness", "actionability", "relevance", "safety"]
ACTION_VERBS = re.compile(r"^(check|run|verify|audit|tune|validate|enable|activate|review|re-?route|inspect|increase|"
                          r"upgrade|plan|correlate|dispatch|roll|freeze|compare|coordinate|enforce|escalate|rate|"
                          r"use|schedule|scan|analy[sz]e|open|continue|document|adjust|add|apply|monitor|confirm|"
                          r"isolate|restart|reset|replace|switch|configure|deploy|investigate|perform|correct|identify)",
                          re.I)


def heuristic_judge(ctx: AgentContext) -> dict:
    rec, cls = ctx.recommendation, ctx.classification
    actions = rec.get("actions", [])
    k = max(len(ctx.hits), 1)
    prevalence = Counter(c for h in ctx.hits for c in h.record["conditions"].split(",") if c)
    issues = []

    # groundedness: are actions/diagnosis backed by evidence (cases or SOP) and are citations valid?
    cited = [e for a in actions for e in a.get("evidence", [])]
    backed = sum(1 for a in actions if a.get("evidence"))
    g = backed / max(len(actions), 1)
    valid_ids = set(ctx.case_ids)
    bad_cites = [c for c in re.findall(r"NOC-\d{6}", rec.get("diagnosis", "")) if c not in valid_ids]
    if bad_cites:
        g -= 0.3
        issues.append(f"Diagnosis cites cases not in evidence: {bad_cites}")
    if not any(str(c).startswith("NOC-") for c in cited) and ctx.hits:
        g -= 0.15
        issues.append("No action cites a historical case.")

    # diagnostic correctness: category conditions must actually be observed (query or similar cases)
    cat = cls.get("category", "NORMAL")
    cat_conds = set(CATEGORIES.get(cat, {}).get("conditions", {}))
    observed = set(ctx.parsed.conditions) | {c for c, n in prevalence.items() if n / k >= 0.4}
    dc = 1.0 if cat == "NORMAL" and not ctx.parsed.conditions else (len(cat_conds & observed) / max(min(len(cat_conds), 2), 1))
    dc = min(dc, 1.0) * (0.6 + 0.4 * min(cls.get("confidence", 0.5) / 0.6, 1.0))
    if dc < 0.5:
        issues.append(f"Category {cat} is weakly supported by the observed conditions {sorted(observed)}.")

    # actionability: enough concrete, imperative, phased steps
    imperative = sum(1 for a in actions if ACTION_VERBS.match(a.get("step", "").strip()))
    phases = {a.get("phase") for a in actions}
    act = min(len(actions) / 4, 1.0) * 0.5 + (imperative / max(len(actions), 1)) * 0.35 + (0.15 if len(phases) >= 2 else 0)
    if len(actions) < 3:
        issues.append("Fewer than 3 actions.")

    # relevance: do the actions address the symptoms in the query?
    if ctx.parsed.conditions:
        cats_for_query = {c for c, spec in CATEGORIES.items() if set(spec["conditions"]) & set(ctx.parsed.conditions)}
        sop_cats = {e[4:] for e in cited if str(e).startswith("SOP:")}
        rel = len(cats_for_query & sop_cats) / max(min(len(cats_for_query), 2), 1)
        rel = min(rel, 1.0)
        missing = [CONDITION_BY_KEY[c].label for c in ctx.parsed.conditions
                   if not any(c in CATEGORIES[s]["conditions"] for s in sop_cats if s in CATEGORIES)]
        if missing:
            issues.append(f"Symptoms not addressed by any action: {missing}")
            rel = min(rel, 0.75)
    else:
        rel = 0.8

    # safety: no unsafe text; service-affecting steps flagged for approval
    unsafe = any(UNSAFE_OUTPUT.search(a.get("step", "")) for a in actions)
    unflagged = [a["step"] for a in actions if SERVICE_AFFECTING.search(a.get("step", "")) and not a.get("requires_approval")]
    safety = 0.0 if unsafe else (0.6 if unflagged else 1.0)
    if unflagged:
        issues.append("Service-affecting steps not flagged for approval.")

    to5 = lambda x: round(1 + 4 * max(0.0, min(1.0, x)), 2)   # noqa: E731
    scores = {"groundedness": to5(g), "diagnostic_correctness": to5(dc), "actionability": to5(act),
              "relevance": to5(rel), "safety": to5(safety)}
    verdict = "revise" if min(scores.values()) <= 2 or scores["groundedness"] <= 3 else "pass"
    return {"scores": scores, "verdict": verdict, "issues": issues,
            "comment": "Deterministic evidence-consistency review.", "judge": "heuristic"}


class JudgeAgent(Agent):
    name = "judge"
    role = "LLM-as-judge: validates the diagnosis against evidence and requests revisions"

    def run(self, ctx: AgentContext) -> dict:
        result = None
        if self.llm_ok(ctx):
            try:
                rec = ctx.recommendation
                k = max(len(ctx.hits), 1)
                prev = Counter(c for h in ctx.hits for c in h.record["conditions"].split(",") if c)
                out = self.llm.generate_json(
                    "You are a strict, independent telecom NOC reviewer. Output JSON only.",
                    prompts.JUDGE.format(
                        query=ctx.query, query_conditions=list(ctx.parsed.conditions),
                        prevalence={c: round(n / k, 2) for c, n in prev.most_common(8)}, case_ids=ctx.case_ids,
                        category=ctx.classification.get("category_name"), risk=ctx.prediction.get("risk_score"),
                        priority=ctx.prediction.get("priority"), diagnosis=rec.get("diagnosis", ""),
                        actions="\n".join(f"{i + 1}. {a['step']} (evidence {a.get('evidence')}, approval "
                                          f"{a.get('requires_approval')})" for i, a in enumerate(rec.get("actions", [])))),
                    max_tokens=1000, temperature=0.0)
                scores = {c: float(out.get("scores", {}).get(c, 3)) for c in CRITERIA}
                result = {"scores": scores, "verdict": out.get("verdict", "pass"),
                          "issues": [str(i) for i in out.get("issues", [])][:6], "comment": out.get("comment", ""),
                          "judge": f"llm:{self.llm.provider}/{self.llm.model}"}
            except Exception as exc:
                ctx.warnings.append(f"LLM judge fallback: {exc}")
        if result is None:
            result = heuristic_judge(ctx)
        result["overall"] = round(sum(result["scores"].values()) / len(CRITERIA), 2)
        result["revision"] = ctx.revisions
        ctx.judge = result
        return {"verdict": result["verdict"], "overall": result["overall"], "judge": result["judge"],
                "issues": len(result["issues"])}


assert set(PLAYBOOK) >= set(CATEGORIES)
