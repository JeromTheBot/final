"""Agent framework: a shared blackboard (AgentContext) + agents that read/write it and leave a trace."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.retrieval import Hit, SearchFilters
from utils.query_parser import ParsedQuery

log = logging.getLogger("netops.agents")


@dataclass
class AgentContext:
    query: str
    parsed: ParsedQuery
    filters: SearchFilters
    metrics_input: dict | None = None          # explicit KPI values (dataset column names)
    top_k: int = 5
    alpha: float = 0.6
    mode: str = "hybrid"
    use_llm: bool = True
    auto_filters: bool = True
    # ---- written by agents ----
    auto_filters_applied: dict = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    retrieval_stats: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    profile: dict = field(default_factory=dict)
    profile_source: str = ""
    prediction: dict = field(default_factory=dict)
    root_cause: dict = field(default_factory=dict)
    impact: dict = field(default_factory=dict)
    playbook: list[dict] = field(default_factory=list)
    recommendation: dict = field(default_factory=dict)
    judge: dict = field(default_factory=dict)
    judge_feedback: list[str] = field(default_factory=list)
    revisions: int = 0
    trace: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def case_ids(self) -> list[str]:
        return [h.case_id for h in self.hits]


class Agent:
    name = "agent"
    role = ""

    def __init__(self, services: Any):
        self.svc = services

    @property
    def llm(self):
        return self.svc.llm

    def llm_ok(self, ctx: AgentContext) -> bool:
        return ctx.use_llm and self.svc.llm.available

    def run(self, ctx: AgentContext) -> dict:   # returns notes for the trace
        raise NotImplementedError

    def __call__(self, ctx: AgentContext) -> AgentContext:
        t0 = time.perf_counter()
        status, notes, used_llm = "ok", {}, False
        calls_before = self.svc.llm.stats.calls
        try:
            notes = self.run(ctx) or {}
        except Exception as exc:  # an agent failure must never take down the whole pipeline
            log.exception("Agent %s failed", self.name)
            status, notes = "error", {"error": str(exc)[:300]}
            ctx.warnings.append(f"{self.name} failed: {exc}")
        used_llm = self.svc.llm.stats.calls > calls_before
        ctx.trace.append({"agent": self.name, "role": self.role, "status": status, "used_llm": used_llm,
                          "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "notes": notes})
        return ctx
