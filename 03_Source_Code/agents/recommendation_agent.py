"""Troubleshooting recommendation agent (RAG over SOP playbook + similar cases).

LLM path: grounded generation with mandatory citations, then output guardrails.
Deterministic path: SOP actions ranked by category relevance and *feedback-learned* effectiveness.
"""
from __future__ import annotations

import json

from agents import prompts
from agents.base import Agent, AgentContext
from app.guardrails import SERVICE_AFFECTING, needs_human_approval, sanitize_output
from utils.telecom_kb import CATEGORIES, CATEGORY_NAMES, CONDITION_BY_KEY, PLAYBOOK

ESCALATION = {
    "P1": "Page on-call RAN/transport lead now; open a major-incident bridge within 15 min; update every 30 min.",
    "P2": "Assign to L2 NOC within 30 min; escalate to field operations if not mitigated in 4 h.",
    "P3": "Queue for L1/L2 within the shift; review trend at next daily ops call.",
    "P4": "Log and monitor; no escalation unless KPIs degrade further.",
}


def _target(kpi: str, th: dict) -> str:
    st = th["stats"].get(kpi)
    if not st:
        return kpi
    low_is_bad = any(c.column == kpi and c.direction == "low" for c in CONDITION_BY_KEY.values())
    return f"{kpi}: {'>=' if low_is_bad else '<='} {st['median']:.1f} (fleet median)"


class RecommendationAgent(Agent):
    name = "recommendation"
    role = "Generates a grounded, prioritised troubleshooting plan"

    def run(self, ctx: AgentContext) -> dict:
        cat = ctx.classification.get("category", "NORMAL")
        cats = [cat] + [s["category"] for s in ctx.classification.get("secondary", [])]
        if ctx.revisions:   # judge feedback: make sure every stated symptom is covered by some SOP
            covered = {k for c in cats for k in CATEGORIES.get(c, {}).get("conditions", {})}
            for k in ctx.parsed.conditions:
                if k not in covered:
                    best = max(CATEGORIES, key=lambda c: CATEGORIES[c]["conditions"].get(k, 0))
                    if best not in cats:
                        cats.append(best)
                        covered |= set(CATEGORIES[best]["conditions"])
        ctx.playbook = self.svc.retriever.search_playbook(ctx.query, categories=cats, k=2)
        rec = None
        if self.llm_ok(ctx):
            try:
                rec = self._llm(ctx)
                rec["method"] = "llm_grounded"
            except Exception as exc:
                ctx.warnings.append(f"LLM recommendation fallback: {exc}")
        if rec is None:
            rec = self._deterministic(ctx, cats)
            rec["method"] = "sop_playbook+feedback"
        # ---- output guardrails ----
        valid = set(ctx.case_ids)
        issues = []
        for a in rec["actions"]:
            a["step"], i1 = sanitize_output(a["step"], valid)
            a["rationale"], i2 = sanitize_output(a.get("rationale", ""), valid)
            a["evidence"] = [e for e in a.get("evidence", []) if e in valid or str(e).startswith("SOP:")]
            a["requires_approval"] = bool(a.get("requires_approval")) or bool(SERVICE_AFFECTING.search(a["step"]))
            issues += i1 + i2
        rec["diagnosis"], i3 = sanitize_output(rec.get("diagnosis", ""), valid)
        rec["summary"], i4 = sanitize_output(rec.get("summary", ""), valid)
        issues += i3 + i4
        rec["guardrail_issues"] = issues
        rec["human_approval_required"] = needs_human_approval([a["step"] for a in rec["actions"]])
        rec["sops_used"] = [p["name"] for p in ctx.playbook]
        ctx.recommendation = rec
        return {"n_actions": len(rec["actions"]), "method": rec["method"], "guardrail_issues": len(issues),
                "revision": ctx.revisions}

    # ------------------------------------------------------------------------------------------------
    def _deterministic(self, ctx: AgentContext, cats: list[str]) -> dict:
        th = self.svc.thresholds
        pred = ctx.prediction or {}
        prio = pred.get("priority") or "P3"
        conds = set(ctx.parsed.conditions)
        actions = []
        for rank, c in enumerate(cats[:4]):
            fb = self.svc.feedback.action_scores(c)
            acts = sorted(PLAYBOOK[c]["actions"], key=lambda a: -fb.get(a, 0))   # stable: SOP order on ties
            cat_conds = set(CATEGORIES.get(c, {}).get("conditions", {}))
            evidence_cases = [h.case_id for h in ctx.hits
                              if cat_conds & set(h.record["conditions"].split(","))][:2]
            n = 4 if rank == 0 else (2 if rank == 1 else 1)
            for j, a in enumerate(acts[:n]):
                phase = "immediate" if rank == 0 and j < 2 else "short_term"
                matched = [CONDITION_BY_KEY[k].label for k in conds & cat_conds]
                actions.append({"step": a, "phase": phase,
                                "rationale": (f"Addresses {', '.join(matched)} ({CATEGORY_NAMES[c]})" if matched
                                              else f"Standard step for {CATEGORY_NAMES[c]}")
                                             + (" - previously rated effective by engineers" if fb.get(a, 0) > 0 else ""),
                                "evidence": evidence_cases + [f"SOP:{c}"],
                                "requires_approval": bool(SERVICE_AFFECTING.search(a)),
                                "feedback_score": round(fb.get(a, 0), 2)})
        actions.append({"step": PLAYBOOK[cats[0]]["actions"][-1], "phase": "preventive",
                        "rationale": "Prevent recurrence of this incident pattern", "evidence": [f"SOP:{cats[0]}"],
                        "requires_approval": False})
        # de-duplicate while preserving order
        seen, uniq = set(), []
        for a in actions:
            if a["step"] not in seen:
                seen.add(a["step"])
                uniq.append(a)
        kpis = []
        for c in cats[:2]:
            kpis += [k for k in PLAYBOOK[c]["kpis"] if k not in kpis]
        cited = ", ".join(ctx.case_ids[:3])
        rc = ctx.root_cause.get("summary", "")
        return {
            "summary": (f"{prio} {ctx.classification.get('category_name', 'incident')} "
                        f"(risk {pred.get('risk_score', 'n/a')}/100, predicted fault occurrence "
                        f"{pred.get('fault_rate_pct', 'n/a')}%). {len(ctx.hits)} similar historical cases retrieved"
                        + (f" ({cited})." if cited else ".")),
            "diagnosis": rc + (f" Closest precedents: {cited}." if cited else ""),
            "actions": uniq[:9],
            "monitoring": [_target(k, th) for k in kpis],
            "escalation": ESCALATION.get(prio, ESCALATION["P3"]),
        }

    def _llm(self, ctx: AgentContext) -> dict:
        cases = "\n".join(f"- {h.case_id} (similarity {h.score:.2f}): {h.record['text'][:600]}" for h in ctx.hits)
        sops = "\n".join(f"- SOP:{p['category']} {p['text']}" for p in ctx.playbook)
        revision = ""
        if ctx.judge_feedback:
            revision = ("\nA reviewer rejected the previous draft. Fix these issues:\n- "
                        + "\n- ".join(ctx.judge_feedback) + "\n")
        p = ctx.prediction
        out = self.llm.generate_json(prompts.SYSTEM_GUARD, prompts.RECOMMEND.format(
            query=ctx.query, category=ctx.classification.get("category_name"), priority=p.get("priority"),
            risk=p.get("risk_score"), fault_rate=p.get("fault_rate_pct"), early=p.get("early_warning_fault_rate_pct"),
            root_cause=ctx.root_cause.get("summary", ""),
            impact=json.dumps({k: ctx.impact.get(k) for k in ("customer_impact_level", "users_affected_estimate",
                                                              "sla_breach_risk", "services_affected")}),
            cases=cases or "none", sops=sops, revision=revision), max_tokens=2500)
        actions = []
        for a in out.get("actions", [])[:7]:
            if isinstance(a, str):
                a = {"step": a}
            actions.append({"step": str(a.get("step", "")), "rationale": str(a.get("rationale", "")),
                            "evidence": [str(e) for e in a.get("evidence", [])],
                            "phase": a.get("phase", "short_term"),
                            "requires_approval": bool(a.get("requires_approval", False))})
        if not actions:
            raise ValueError("LLM returned no actions")
        mon = out.get("monitoring") or []
        return {"summary": str(out.get("summary", "")), "diagnosis": str(out.get("diagnosis", "")),
                "actions": actions, "monitoring": [str(m) for m in mon],
                "escalation": str(out.get("escalation") or ESCALATION.get(p.get("priority", "P3")))}
