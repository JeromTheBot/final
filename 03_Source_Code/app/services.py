"""Service layer: wires artifacts, models, agents, guardrails and feedback together (used by the API)."""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from agents.base import AgentContext
from agents.classification_agent import classify_metrics
from agents.orchestrator import Orchestrator
from agents.prediction_agent import score_profile
from app.config import Settings, get_settings
from app.feedback import FeedbackStore
from app.guardrails import GuardrailReport, validate_query
from app.llm import LLMClient, get_llm
from app.ml_models import ModelBundle
from app.monitoring import METRICS, Metrics
from app.retrieval import Hit, HybridRetriever, SearchFilters
from utils.features import COLUMN_TO_ALIAS, fill_profile, load_thresholds
from utils.query_parser import parse_query
from utils.telecom_kb import CATEGORY_NAMES, CONDITION_BY_KEY, SEASONS, WEATHER

log = logging.getLogger("netops.service")

KEY_KPIS = ["Cell Availability (%)", "Throughput (Mbps)", "Latency (ms)", "Packet Loss Rate (%)", "Jitter (ms)",
            "Call Drop Rate (%)", "Handover Success Rate (%)", "Signal Strength (dBm)",
            "Connection Setup Success Rate (%)", "Alarm Count", "Critical Alarm Count", "User Count",
            "Authentication Failures", "Fault Occurrence Rate (%)", "MTTR (hours)"]


def _py(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def hit_to_dict(h: Hit, include_text: bool = True) -> dict:
    r = h.record
    d = {"case_id": h.case_id, "score": h.score,
         "scores": {"dense": round(h.dense_score, 4), "bm25": round(h.bm25_score, 3), "dense_rank": h.dense_rank,
                    "bm25_rank": h.bm25_rank, "condition_overlap": round(h.condition_overlap, 2),
                    "feedback_boost": round(h.feedback_boost, 3)},
         "location": {"city": r["City"], "state": r["State"], "zip": _py(r["Zip"])},
         "season": r["Season"], "weather": r["Weather"], "category": r["category"], "category_name": r["category_name"],
         "priority": r["priority"], "severity": r["severity"], "risk_score": _py(r["risk_score"]),
         "conditions": [CONDITION_BY_KEY[c].label for c in r["conditions"].split(",") if c],
         "kpis": {COLUMN_TO_ALIAS[k]: round(float(r[k]), 3) for k in KEY_KPIS},
         "issue_reported": r["Issue Reported"]}
    if include_text:
        d["text"] = r["text"]
    return d


class NetOpsService:
    def __init__(self, settings: Settings | None = None, retriever: HybridRetriever | None = None,
                 llm: LLMClient | None = None, metrics: Metrics | None = None):
        t0 = time.time()
        self.settings = s = settings or get_settings()
        self.thresholds = load_thresholds(s.artifacts_dir / "thresholds.json")
        self.retriever = retriever or HybridRetriever.load(s.artifacts_dir, rrf_k=s.rrf_k,
                                                           candidate_pool=s.candidate_pool)
        self.models = ModelBundle.load(s.artifacts_dir)
        self.llm = llm or get_llm()
        self.feedback = FeedbackStore(s.feedback_db)
        self.metrics = metrics or METRICS
        self.orchestrator = Orchestrator(self)
        self.index_meta = json.loads((s.artifacts_dir / "index_meta.json").read_text())
        self.known_states = set(self.thresholds["states"])
        self.known_cities = set(self.retriever.cases["City"].unique())
        self.predict_risk({}, explain=True)          # warm-up: first sklearn/OpenMP call is slow
        self.retriever.embedder.encode(["warm up"])
        log.info("NetOpsService ready in %.1fs: %d cases, embedder=%s, llm=%s", time.time() - t0,
                 len(self.retriever.cases), self.retriever.embedder.name, self.llm.describe())

    # ---- helpers -----------------------------------------------------------------------------------
    def _context(self, req) -> tuple[AgentContext, GuardrailReport]:
        kpis = req.kpis.to_columns() if getattr(req, "kpis", None) else None
        query = (req.query or "").strip()
        if not query and kpis:   # KPI-only request: synthesise a description so retrieval still works
            df = pd.DataFrame([fill_profile(kpis, self.thresholds)])
            from utils.features import detect_conditions
            conds = [CONDITION_BY_KEY[k].label for k, v in detect_conditions(df, self.thresholds).iloc[0].items() if v]
            query = "Cell with " + (", ".join(conds) if conds else "normal KPIs")
        parsed = parse_query(query, self.known_states, self.known_cities)
        try:
            report = validate_query(query, self.settings.min_query_chars, self.settings.max_query_chars,
                                    parsed=parsed, has_metrics=bool(kpis))
        except Exception:
            self.metrics.event("guardrail_blocked")
            raise
        if report.redactions:
            self.metrics.event("pii_redacted")
            parsed = parse_query(report.sanitized_query, self.known_states, self.known_cities)
        f = req.filters.model_dump() if getattr(req, "filters", None) else {}
        ctx = AgentContext(query=report.sanitized_query, parsed=parsed, filters=SearchFilters(**f), metrics_input=kpis,
                           top_k=req.top_k, alpha=req.alpha, mode=req.mode, use_llm=req.use_llm,
                           auto_filters=req.auto_filters)
        ctx.warnings += report.warnings
        return ctx, report

    def _base_response(self, ctx: AgentContext, report: GuardrailReport) -> dict:
        return {"query": ctx.query,
                "understanding": {"conditions": {k: {"label": CONDITION_BY_KEY[k].label,
                                                     "severity": "severe" if v == 2 else "degraded"}
                                                 for k, v in ctx.parsed.conditions.items()},
                                  "auto_filters_applied": ctx.auto_filters_applied},
                "guardrails": {"checks": report.checks, "redactions": report.redactions},
                "retrieval": ctx.retrieval_stats,
                "similar_cases": [hit_to_dict(h) for h in ctx.hits],
                "warnings": ctx.warnings}

    # ---- use cases ---------------------------------------------------------------------------------
    def search(self, req) -> dict:
        ctx, report = self._context(req)
        self.orchestrator.query_agent(ctx)
        self.orchestrator.retrieval(ctx)
        return self._base_response(ctx, report) | {"agent_trace": ctx.trace}

    def classify(self, req) -> dict:
        ctx, report = self._context(req)
        for a in (self.orchestrator.query_agent, self.orchestrator.retrieval, self.orchestrator.classifier,
                  self.orchestrator.predictor):
            a(ctx)
        out = self._base_response(ctx, report)
        out["classification"] = ctx.classification
        out["priority"] = {k: ctx.prediction.get(k) for k in ("priority", "severity", "risk_score")}
        out["agent_trace"] = ctx.trace
        return out

    def recommend(self, req) -> dict:
        """Requirement 1: retrieval + classification + prioritisation + basic recommendation."""
        ctx, report = self._context(req)
        self.orchestrator.run_basic(ctx)
        return self._finish(ctx, report, full=False)

    def analyze(self, req) -> dict:
        """Requirement 2: full multi-agent pipeline incl. root cause, impact, predictions and LLM-as-judge."""
        ctx, report = self._context(req)
        self.orchestrator.run_full(ctx)
        return self._finish(ctx, report, full=True)

    def _finish(self, ctx: AgentContext, report: GuardrailReport, full: bool) -> dict:
        out = self._base_response(ctx, report)
        out["classification"] = ctx.classification
        out["prioritization"] = {k: ctx.prediction.get(k) for k in ("priority", "severity", "risk_score")}
        out["recommendation"] = ctx.recommendation
        if full:
            out["prediction"] = ctx.prediction
            out["root_cause"] = ctx.root_cause
            out["impact"] = ctx.impact
            out["judge"] = ctx.judge
            out["sops"] = [{"category": p["category"], "name": p["name"], "similarity": round(p["similarity"], 3)}
                           for p in ctx.playbook]
        out["agent_trace"] = ctx.trace
        out["llm"] = self.llm.describe() | {"used": any(t["used_llm"] for t in ctx.trace)}
        out["response_id"] = self.feedback.log_response(
            ctx.query, ctx.classification.get("category", ""), ctx.case_ids,
            [a["step"] for a in ctx.recommendation.get("actions", [])],
            {"priority": ctx.prediction.get("priority"), "judge": ctx.judge.get("overall")})
        out["warnings"] = ctx.warnings
        return out

    def predict_risk(self, kpis: dict, explain: bool = True) -> dict:
        prof = fill_profile(kpis, self.thresholds)
        out = score_profile(self, prof)
        out["classification"] = classify_metrics(kpis, self.thresholds)
        out["conditions"] = {CONDITION_BY_KEY[k].label: ("severe" if v == 2 else "degraded")
                             for k, v in out["conditions"].items()}
        if explain:
            out["drivers"] = self.models.fault.explain(pd.DataFrame([prof]))
        out["model"] = {k: self.models.metrics["fault_risk"][k] for k in ("algorithm", "r2", "mae", "roc_auc",
                                                                           "early_warning_r2")}
        return out

    def predict_performance(self, kpis: dict, explain: bool = True) -> dict:
        prof = fill_profile(kpis, self.thresholds)
        X = pd.DataFrame([prof])
        pp = self.models.perf.predict(X)
        out = {"forecast": {}}
        for t, v in pp.items():
            st = self.thresholds["stats"][t]
            pred = round(float(v[0]), 3)
            item = {"predicted": pred, "fleet_median": round(st["median"], 3),
                    "model_r2": self.models.metrics["performance"][t]["r2"],
                    "model_mae": self.models.metrics["performance"][t]["mae"],
                    "interval_approx": [round(pred - self.models.metrics["performance"][t]["mae"] * 1.5, 2),
                                        round(pred + self.models.metrics["performance"][t]["mae"] * 1.5, 2)]}
            if explain:
                item["drivers"] = self.models.perf.explain(X, t)
            out["forecast"][COLUMN_TO_ALIAS[t]] = item
        out["note"] = ("Forecasts use operating conditions only (load, radio, environment, config); "
                       "see model_r2 for reliability.")
        return out

    def prioritize(self, items) -> dict:
        rows = []
        for it in items:
            if it.case_id:
                rec = self.retriever.get(it.case_id)
                if rec is None:
                    rows.append({"id": it.id, "error": f"unknown case_id {it.case_id}"})
                    continue
                kpis = {k: rec[k] for k in self.thresholds["stats"] if k in rec} | {"Season": rec["Season"],
                                                                                   "Weather": rec["Weather"]}
            else:
                kpis = it.kpis.to_columns()
            r = self.predict_risk(kpis, explain=False)
            prof = fill_profile(kpis, self.thresholds)
            rows.append({"id": it.id, "case_id": it.case_id, "description": it.description,
                         "priority": r["priority"], "severity": r["severity"], "risk_score": r["risk_score"],
                         "predicted_fault_rate_pct": r["fault_rate_pct"],
                         "high_risk_probability": r["high_risk_probability"],
                         "category": r["classification"]["category_name"],
                         "conditions": list(r["conditions"]), "users": int(prof["User Count"])})
        ok = [r for r in rows if "error" not in r]
        order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
        ok.sort(key=lambda r: (order[r["priority"]], -r["risk_score"], -r["users"]))
        for i, r in enumerate(ok, 1):
            r["rank"] = i
        return {"ranked": ok, "errors": [r for r in rows if "error" in r],
                "summary": {p: sum(r["priority"] == p for r in ok) for p in order}}

    def metadata(self) -> dict:
        c = self.retriever.cases
        return {"n_cases": len(c), "states": sorted(c["State"].unique().tolist()), "seasons": SEASONS,
                "weather": WEATHER, "categories": CATEGORY_NAMES,
                "conditions": {k: v.label for k, v in CONDITION_BY_KEY.items()},
                "priorities": ["P1", "P2", "P3", "P4"], "index": self.index_meta,
                "llm": self.llm.describe(), "model_metrics": self.models.metrics,
                "category_distribution": c["category_name"].value_counts().to_dict(),
                "priority_distribution": c["priority"].value_counts().sort_index().to_dict()}
