"""Retrieval agent: hybrid search over 100K historical incidents with metadata filters and filter relaxation."""
from __future__ import annotations

from agents.base import Agent, AgentContext


class RetrievalAgent(Agent):
    name = "retrieval"
    role = "Finds the most similar historical network cases (BM25 + dense vectors + RRF + re-ranking)"

    def run(self, ctx: AgentContext) -> dict:
        r = self.svc.retriever
        bm25_q = f"{ctx.query} {ctx.parsed.bm25_expansion()}"
        kw = dict(top_k=ctx.top_k, alpha=ctx.alpha, mode=ctx.mode, query_conditions=ctx.parsed.condition_keys,
                  bm25_query=bm25_q, feedback_scores=self.svc.feedback.case_scores(),
                  feedback_weight=self.svc.settings.feedback_weight)
        hits, stats = r.search(ctx.query, ctx.filters, **kw)
        relaxed = []
        # auto-detected filters are "soft": relax them one by one if they empty the result set
        for attr in ("city", "weather", "season", "state"):
            if hits or attr not in ctx.auto_filters_applied:
                continue
            setattr(ctx.filters, attr, None)
            relaxed.append(attr)
            hits, stats = r.search(ctx.query, ctx.filters, **kw)
        if relaxed:
            ctx.warnings.append(f"No cases matched auto-detected filter(s) {relaxed}; filter relaxed.")
        ctx.hits, ctx.retrieval_stats = hits, stats | {"relaxed_filters": relaxed,
                                                       "filters": ctx.filters.active()}
        if not hits:
            ctx.warnings.append("No historical cases matched the filters.")
        return {"n_hits": len(hits), "candidates": stats.get("candidates_after_filter"),
                "latency_ms": stats.get("latency_ms"), "top": [(h.case_id, h.score) for h in hits[:3]]}
