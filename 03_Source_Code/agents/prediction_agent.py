"""Risk & performance prediction agent.

Builds a KPI profile for the incident (explicit KPIs if given; otherwise a *case-based profile*: median of
the retrieved similar cases overlaid with the symptoms in the query) and scores it with the ML models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents.base import Agent, AgentContext
from utils.features import (NUMERIC_COLUMNS, SEVERITY_BY_PRIORITY, detect_conditions, fill_profile,
                            percentile_rank, priority_from_risk, profile_from_conditions, risk_score)


def build_profile(ctx: AgentContext, th: dict) -> tuple[dict, str]:
    if ctx.metrics_input:
        return fill_profile(ctx.metrics_input, th), "user_kpis"
    if ctx.hits:
        recs = pd.DataFrame([h.record for h in ctx.hits])
        prof = {c: float(recs[c].median()) for c in NUMERIC_COLUMNS}
        prof["Season"] = recs["Season"].mode()[0]
        prof["Weather"] = recs["Weather"].mode()[0]
        prof.update(profile_from_conditions(ctx.parsed.conditions, th))
        return fill_profile(prof, th), "similar_cases+query_symptoms"
    return fill_profile(profile_from_conditions(ctx.parsed.conditions, th), th), "query_symptoms"


def score_profile(svc, profile: dict) -> dict:
    th = svc.thresholds
    X = pd.DataFrame([profile])
    fp = svc.models.fault.predict(X)
    pp = svc.models.perf.predict(X)
    cond = detect_conditions(X, th)
    fault_rate = float(fp["fault_rate"][0])
    risk = float(risk_score(X, cond, th, fault_rate=np.array([fault_rate]))[0])
    prio = str(priority_from_risk(np.array([risk]), cond["critical_alarms"].to_numpy())[0])
    perf = {}
    for t, v in pp.items():
        fleet_med = th["stats"][t]["median"]
        perf[t] = {"predicted": round(float(v[0]), 2), "fleet_median": round(fleet_med, 2),
                   "percentile": round(float(percentile_rank(float(v[0]), t, th)) * 100, 1)}
    return {"fault_rate_pct": round(fault_rate, 2),
            "early_warning_fault_rate_pct": round(float(fp["early_warning_fault_rate"][0]), 2),
            "high_risk_probability": round(float(fp["high_risk_probability"][0]), 3),
            "fault_rate_percentile": round(float(percentile_rank(fault_rate, "Fault Occurrence Rate (%)", th)) * 100, 1),
            "risk_score": round(risk, 1), "priority": prio, "severity": SEVERITY_BY_PRIORITY[prio],
            "performance_forecast": perf,
            "conditions": {k: int(v) for k, v in cond.iloc[0].items() if v}}


class RiskPredictionAgent(Agent):
    name = "risk_prediction"
    role = "Predicts fault risk, network performance and assigns incident priority"

    def run(self, ctx: AgentContext) -> dict:
        ctx.profile, ctx.profile_source = build_profile(ctx, self.svc.thresholds)
        pred = score_profile(self.svc, ctx.profile)
        if ctx.hits:
            pred["similar_cases"] = {
                "avg_fault_rate_pct": round(float(np.mean([h.record["Fault Occurrence Rate (%)"] for h in ctx.hits])), 2),
                "avg_risk_score": round(float(np.mean([h.record["risk_score"] for h in ctx.hits])), 1),
                "share_p1_p2": round(float(np.mean([h.record["priority"] in ("P1", "P2") for h in ctx.hits])), 2)}
        pred["profile_source"] = ctx.profile_source
        ctx.prediction = pred
        self.svc.metrics.record_priority(pred["priority"])
        return {"risk": pred["risk_score"], "priority": pred["priority"], "fault_rate": pred["fault_rate_pct"],
                "profile_source": ctx.profile_source}
