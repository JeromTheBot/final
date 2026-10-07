"""Incident classification agent.

Evidence fusion of three signals:
  1. rule-based taxonomy scores from the symptoms (query text or explicit KPIs)
  2. similarity-weighted vote of the retrieved historical cases' categories (k-NN)
  3. LLM tie-break when the two disagree / confidence is low (optional)
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from agents import prompts
from agents.base import Agent, AgentContext
from utils.features import classify, detect_conditions, fill_profile
from utils.query_parser import categories_from_conditions
from utils.telecom_kb import CATEGORY_NAMES, NORMAL_CATEGORY


def _norm(d: dict[str, float]) -> dict[str, float]:
    t = sum(d.values())
    return {k: v / t for k, v in d.items()} if t else {}


class ClassificationAgent(Agent):
    name = "classification"
    role = "Classifies the incident into the NOC taxonomy with a confidence score"

    def run(self, ctx: AgentContext) -> dict:
        rule = _norm(categories_from_conditions(ctx.parsed.conditions))
        votes: dict[str, float] = defaultdict(float)
        for h in ctx.hits:
            votes[h.record["category"]] += max(h.score, 1e-3)
        vote = _norm(dict(votes))
        if rule:
            combined = {c: 0.6 * rule.get(c, 0) + 0.4 * vote.get(c, 0) for c in set(rule) | set(vote)}
        else:
            combined = vote or {NORMAL_CATEGORY: 1.0}
        ranked = sorted(combined.items(), key=lambda kv: -kv[1])
        top, conf = ranked[0]
        method = "rules+knn" if rule else "knn"
        rationale = (f"Symptoms {list(ctx.parsed.conditions) or 'n/a'} map to {CATEGORY_NAMES[top]}; "
                     f"{int(round(vote.get(top, 0) * 100))}% of similar-case evidence agrees.")
        notes = {"rule_scores": {k: round(v, 3) for k, v in rule.items()},
                 "knn_votes": {k: round(v, 3) for k, v in vote.items()}}
        if self.llm_ok(ctx) and len(ranked) > 1 and (conf < 0.45 or (rule and vote and
                                                                     max(rule, key=rule.get) != max(vote, key=vote.get))):
            try:
                out = self.llm.generate_json(
                    "You are a telecom incident classifier. Output JSON only.",
                    prompts.CLASSIFY_TIEBREAK.format(
                        candidates={CATEGORY_NAMES[k]: round(v, 2) for k, v in ranked[:3]} | {"keys": [k for k, _ in ranked[:3]]},
                        conditions=list(ctx.parsed.conditions),
                        neighbors=[h.record["category"] for h in ctx.hits]), max_tokens=200)
                if out.get("category") in dict(ranked[:3]):
                    top = out["category"]
                    conf = combined[top]
                    rationale = out.get("rationale", rationale)
                    method += "+llm"
            except Exception as exc:
                notes["llm_error"] = str(exc)[:200]
        secondary = [k for k, v in ranked[1:3] if v >= 0.15 and k != NORMAL_CATEGORY]
        ctx.classification = {"category": top, "category_name": CATEGORY_NAMES[top], "confidence": round(conf, 3),
                              "secondary": [{"category": k, "category_name": CATEGORY_NAMES[k],
                                             "score": round(combined[k], 3)} for k in secondary],
                              "conditions": dict(ctx.parsed.conditions), "method": method, "rationale": rationale,
                              "distribution": {CATEGORY_NAMES[k]: round(v, 3) for k, v in ranked[:5]}}
        notes |= {"category": top, "confidence": round(conf, 3), "method": method}
        return notes


def classify_metrics(metrics: dict, th: dict) -> dict:
    """Pure rule-based classification of a single KPI record (used by /classify and /prioritize)."""
    df = pd.DataFrame([fill_profile(metrics, th)])
    cond = detect_conditions(df, th)
    cat, scores = classify(cond, df["Weather"])
    return {"category": cat.iloc[0], "category_name": CATEGORY_NAMES[cat.iloc[0]],
            "conditions": {k: int(v) for k, v in cond.iloc[0].items() if v},
            "scores": {k: round(float(v), 2) for k, v in scores.iloc[0].items() if v}}
