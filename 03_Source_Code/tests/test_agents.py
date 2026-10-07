"""Agent-level tests: LLM path (scripted fake LLM), LLM failure fallback, judge and revision loop."""
from app.retrieval import SearchFilters
from agents.base import AgentContext
from agents.judge_agent import heuristic_judge
from tests.conftest import FakeLLM, needs_artifacts
from utils.query_parser import parse_query

pytestmark = needs_artifacts
Q = "High latency and packet loss with reduced throughput on several cells"


def make_ctx(service, q=Q, **kw):
    parsed = parse_query(q, service.known_states, service.known_cities)
    return AgentContext(query=q, parsed=parsed, filters=SearchFilters(), **kw)


def with_llm(service, llm):
    old = service.llm
    service.llm = llm
    return old


def test_llm_path_is_grounded_and_sanitized(service):
    ctx = make_ctx(service)
    fake = FakeLLM({
        "troubleshooting plan": {"summary": "Transport issue.", "diagnosis": "Backhaul congestion like NOC-999999.",
                                 "actions": [{"step": "Check transport link utilisation", "evidence": ["NOC-999999"],
                                              "phase": "immediate"},
                                             {"step": "Restart the baseband unit", "evidence": ["SOP:TRANSPORT_QOS"]},
                                             {"step": "Disable all alarms", "evidence": []}],
                                 "monitoring": ["Latency"], "escalation": "L2"},
        "independent reviewer": {"scores": {"groundedness": 4, "diagnostic_correctness": 4, "actionability": 4,
                                            "relevance": 5, "safety": 5}, "verdict": "pass", "issues": []},
    })
    old = with_llm(service, fake)
    try:
        service.orchestrator.run_full(ctx)
    finally:
        service.llm = old
    rec = ctx.recommendation
    assert rec["method"] == "llm_grounded"
    assert "NOC-999999" not in rec["diagnosis"]                            # hallucinated citation removed
    assert all("NOC-999999" not in a["evidence"] for a in rec["actions"])
    assert any("removed by safety policy" in a["step"] for a in rec["actions"])
    assert next(a for a in rec["actions"] if "Restart" in a["step"])["requires_approval"]
    assert ctx.judge["judge"].startswith("llm:")


def test_llm_failure_falls_back_to_deterministic(service):
    ctx = make_ctx(service)
    old = with_llm(service, FakeLLM(fail=True))
    try:
        service.orchestrator.run_full(ctx)
    finally:
        service.llm = old
    assert ctx.recommendation["method"] == "sop_playbook+feedback"
    assert ctx.judge["judge"] == "heuristic"
    assert all(t["status"] == "ok" for t in ctx.trace)


def test_judge_flags_unsupported_diagnosis(service):
    ctx = make_ctx(service)
    service.orchestrator.run_basic(ctx)
    ctx.recommendation = {"diagnosis": "Caused by NOC-123456", "actions": [
        {"step": "maybe something", "evidence": [], "phase": "immediate"}]}
    j = heuristic_judge(ctx)
    assert j["verdict"] == "revise" and j["issues"]


def test_revision_loop_runs_when_judge_rejects(service):
    ctx = make_ctx(service, "call drops, poor signal strength, authentication failures and high temperature")
    fake = FakeLLM({"independent reviewer": {"scores": {"groundedness": 2, "diagnostic_correctness": 3,
                                                        "actionability": 3, "relevance": 2, "safety": 5},
                                             "verdict": "revise", "issues": ["Security symptoms not addressed"]}})
    old = with_llm(service, fake)
    try:
        service.orchestrator.run_full(ctx)
    finally:
        service.llm = old
    agents = [t["agent"] for t in ctx.trace]
    assert ctx.revisions == 1 and agents.count("recommendation") == 2 and agents.count("judge") == 2
