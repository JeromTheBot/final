"""Multi-agent orchestrator (supervisor + shared blackboard).

    query ─► QueryUnderstanding ─► Retrieval ─► Classification ─► RiskPrediction ─► RootCause/Impact
                                                                                         │
                         ┌──────────── revise (max 1) ◄── Judge ◄── Recommendation ◄─────┘
                         └──────────────────────────────► Recommendation ─► Judge ─► response

The "basic" pipeline (Requirement 1, /recommend) skips the judge and runs the deterministic planners.
"""
from __future__ import annotations

import logging

from agents.base import AgentContext
from agents.classification_agent import ClassificationAgent
from agents.judge_agent import JudgeAgent
from agents.prediction_agent import RiskPredictionAgent
from agents.query_agent import QueryUnderstandingAgent
from agents.recommendation_agent import RecommendationAgent
from agents.retrieval_agent import RetrievalAgent
from agents.root_cause_agent import RootCauseAgent

log = logging.getLogger("netops.orchestrator")


class Orchestrator:
    MAX_REVISIONS = 1

    def __init__(self, services):
        self.svc = services
        self.query_agent = QueryUnderstandingAgent(services)
        self.retrieval = RetrievalAgent(services)
        self.classifier = ClassificationAgent(services)
        self.predictor = RiskPredictionAgent(services)
        self.root_cause = RootCauseAgent(services)
        self.recommender = RecommendationAgent(services)
        self.judge = JudgeAgent(services)

    def run_basic(self, ctx: AgentContext) -> AgentContext:
        for agent in (self.query_agent, self.retrieval, self.classifier, self.predictor, self.root_cause,
                      self.recommender):
            agent(ctx)
        return ctx

    def run_full(self, ctx: AgentContext) -> AgentContext:
        self.run_basic(ctx)
        self.judge(ctx)
        while ctx.judge.get("verdict") == "revise" and ctx.revisions < self.MAX_REVISIONS:
            ctx.revisions += 1
            ctx.judge_feedback = ctx.judge.get("issues", [])
            self.svc.metrics.event("judge_revision")
            log.info("Judge requested revision: %s", ctx.judge_feedback)
            self.recommender(ctx)
            self.judge(ctx)
        return ctx
