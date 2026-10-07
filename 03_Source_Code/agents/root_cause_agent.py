"""Root-cause & network-impact analysis agent.

Triangulates three independent signals:
  * statistical deviations of the incident's KPIs from the fleet baseline (direction-aware z-scores)
  * ML explanations - which KPIs push the predicted fault rate up (local ablation contributions)
  * historical evidence - how often each condition appears among the most similar past cases
and maps them onto SOP root causes. Impact quantifies users, traffic, availability and SLA exposure.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from agents import prompts
from agents.base import Agent, AgentContext
from utils.features import NUMERIC_COLUMNS
from utils.telecom_kb import CATEGORY_NAMES, CONDITION_BY_KEY, CONDITIONS, PLAYBOOK

DIRECTION = {c.column: c.direction for c in CONDITIONS}


def kpi_deviations(profile: dict, th: dict, top: int = 6) -> list[dict]:
    out = []
    for col in NUMERIC_COLUMNS:
        st = th["stats"][col]
        z = (float(profile[col]) - st["mean"]) / (st["std"] or 1)
        d = DIRECTION.get(col, "high")
        badness = -z if d == "low" else (abs(z) if d == "both" else z)
        out.append({"kpi": col, "value": round(float(profile[col]), 2), "fleet_median": round(st["median"], 2),
                    "z_score": round(z, 2), "badness": round(badness, 2)})
    out.sort(key=lambda d: -d["badness"])
    return [d for d in out[:top] if d["badness"] > 0.5]


class RootCauseAgent(Agent):
    name = "root_cause_impact"
    role = "Performs root-cause and network-impact analysis"

    def run(self, ctx: AgentContext) -> dict:
        th, svc = self.svc.thresholds, self.svc
        cat = ctx.classification.get("category", "NORMAL")
        cats = [cat] + [s["category"] for s in ctx.classification.get("secondary", [])]
        devs = kpi_deviations(ctx.profile, th)
        drivers = svc.models.fault.explain(pd.DataFrame([ctx.profile]))
        cnt = Counter(c for h in ctx.hits for c in h.record["conditions"].split(",") if c)
        k = max(len(ctx.hits), 1)
        prevalence = {CONDITION_BY_KEY[c].label: round(n / k, 2) for c, n in cnt.most_common(8)}

        causes = []
        for i, c in enumerate(cats):
            for j, cause in enumerate(PLAYBOOK[c]["root_causes"]):
                causes.append({"cause": cause, "category": CATEGORY_NAMES[c],
                               "likelihood": "high" if i == 0 and j < 2 else ("medium" if i == 0 or j == 0 else "low"),
                               "evidence": f"SOP:{c}"})
        causes = causes[:6]
        summary = self._template_summary(ctx, devs, drivers, prevalence)
        method = "deterministic"
        if self.llm_ok(ctx):
            try:
                out = self.llm.generate_json(prompts.SYSTEM_GUARD, prompts.ROOT_CAUSE.format(
                    query=ctx.query, category=ctx.classification.get("category_name"),
                    confidence=ctx.classification.get("confidence"),
                    deviations=[(d["kpi"], d["value"], d["z_score"]) for d in devs],
                    drivers=[(d["feature"], d["contribution"]) for d in drivers],
                    k=len(ctx.hits), prevalence=prevalence,
                    sop_causes=[c["cause"] for c in causes]), max_tokens=1000)
                summary = out.get("summary") or summary
                if out.get("likely_root_causes"):
                    causes = out["likely_root_causes"][:6]
                method = "llm_grounded"
            except Exception as exc:
                ctx.warnings.append(f"LLM root-cause fallback: {exc}")
        ctx.root_cause = {"summary": summary, "likely_root_causes": causes, "kpi_deviations": devs,
                          "model_drivers": drivers, "condition_prevalence_in_similar_cases": prevalence,
                          "method": method}
        ctx.impact = self._impact(ctx)
        return {"top_deviation": devs[0]["kpi"] if devs else None,
                "top_driver": drivers[0]["feature"] if drivers else None, "method": method}

    @staticmethod
    def _template_summary(ctx, devs, drivers, prevalence) -> str:
        name = ctx.classification.get("category_name", "an unclassified issue")
        dev_txt = ", ".join(f"{d['kpi']} at {d['value']} (fleet median {d['fleet_median']})" for d in devs[:3])
        drv_txt = ", ".join(f"{d['feature']} ({d['contribution']:+.1f} pp)" for d in drivers[:3] if d["contribution"] > 0)
        prev_txt = ", ".join(f"{k} in {int(v * 100)}%" for k, v in list(prevalence.items())[:3])
        parts = [f"The pattern is most consistent with {name}."]
        if dev_txt:
            parts.append(f"Largest deviations from the fleet baseline: {dev_txt}.")
        if drv_txt:
            parts.append(f"The fault-risk model attributes elevated fault likelihood mainly to {drv_txt}.")
        if prev_txt:
            parts.append(f"Among similar historical cases: {prev_txt}.")
        return " ".join(parts)

    def _impact(self, ctx: AgentContext) -> dict:
        th, p = self.svc.thresholds, ctx.profile
        prio = ctx.prediction.get("priority", "P4")
        avail_gap = th["stats"]["Cell Availability (%)"]["median"] - float(p["Cell Availability (%)"])
        mttr = float(np.median([h.record["MTTR (hours)"] for h in ctx.hits])) if ctx.hits else float(p["MTTR (hours)"])
        users = int(p["User Count"])
        sla = "High" if prio in ("P1", "P2") or avail_gap > 5 else ("Medium" if prio == "P3" else "Low")
        cases = self.svc.retriever.cases
        scope = self.svc.retriever.filter_mask(ctx.filters)
        same_cat = int((cases.loc[scope, "category"] == ctx.classification.get("category")).sum())
        level = "Severe" if users > 7500 and prio in ("P1", "P2") else ("Significant" if prio in ("P1", "P2", "P3")
                                                                         else "Limited")
        services = []
        conds = ctx.parsed.conditions
        if {"call_drops", "high_jitter", "handover_failure"} & set(conds):
            services.append("Voice / VoLTE")
        if {"low_throughput", "high_latency", "packet_loss", "high_load"} & set(conds):
            services.append("Mobile broadband / video")
        if {"setup_failure", "poor_signal", "low_availability"} & set(conds):
            services.append("Network access / attach")
        if {"auth_failures", "security_incidents"} & set(conds):
            services.append("Subscriber authentication")
        return {"customer_impact_level": level, "users_affected_estimate": users,
                "data_volume_at_risk_gb": round(float(p["Data Usage (GB)"]), 1),
                "availability_gap_pp": round(avail_gap, 2), "expected_mttr_hours": round(mttr, 2),
                "sla_breach_risk": sla, "services_affected": services or ["General data service"],
                "pattern_prevalence": {"same_category_cases_in_scope": same_cat, "scope_size": int(scope.sum()),
                                       "share": round(same_cat / max(int(scope.sum()), 1), 3)}}
