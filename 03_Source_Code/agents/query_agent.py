"""Query-understanding agent: symptoms + metadata from free text (lexicon first, LLM refinement optional)."""
from __future__ import annotations

import pandas as pd

from agents import prompts
from agents.base import Agent, AgentContext
from utils.features import detect_conditions, fill_profile
from utils.telecom_kb import CONDITION_BY_KEY


class QueryUnderstandingAgent(Agent):
    name = "query_understanding"
    role = "Extracts KPI symptoms, severity and location/season filters from the query"

    def run(self, ctx: AgentContext) -> dict:
        notes = {"lexicon_conditions": dict(ctx.parsed.conditions)}
        if ctx.metrics_input:   # explicit KPIs are the strongest evidence of symptoms
            df = pd.DataFrame([fill_profile(ctx.metrics_input, self.svc.thresholds)])
            measured = {k: int(v) for k, v in detect_conditions(df, self.svc.thresholds).iloc[0].items() if v}
            for k, v in measured.items():
                ctx.parsed.conditions[k] = max(ctx.parsed.conditions.get(k, 0), v)
            notes["kpi_conditions"] = measured
        if self.llm_ok(ctx):
            try:
                out = self.llm.generate_json(
                    "You extract structured telecom symptoms. Output JSON only.",
                    prompts.QUERY_UNDERSTANDING.format(keys=", ".join(CONDITION_BY_KEY), query=ctx.query),
                    max_tokens=300)
                added = {}
                for k, v in (out.get("conditions") or {}).items():
                    if k in CONDITION_BY_KEY and k not in ctx.parsed.conditions:
                        ctx.parsed.conditions[k] = 2 if int(v or 1) >= 2 else 1
                        added[k] = ctx.parsed.conditions[k]
                notes["llm_added_conditions"] = added
                notes["intent"] = out.get("intent")
            except Exception as exc:
                notes["llm_error"] = str(exc)[:200]
        if ctx.auto_filters:
            f = ctx.filters
            for attr, vals in (("state", ctx.parsed.states), ("city", ctx.parsed.cities),
                               ("season", ctx.parsed.seasons), ("weather", ctx.parsed.weather)):
                if vals and not getattr(f, attr):
                    setattr(f, attr, list(vals))
                    ctx.auto_filters_applied[attr] = list(vals)
        notes["auto_filters"] = ctx.auto_filters_applied
        notes["conditions"] = dict(ctx.parsed.conditions)
        return notes
