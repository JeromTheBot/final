"""DeepEval quality gates (run only when an LLM API key is configured).

    NETOPS_LLM_PROVIDER=openai OPENAI_API_KEY=... pytest tests/test_deepeval.py -m deepeval
"""
import os

import pytest

from tests.conftest import needs_artifacts

LLM_KEYS = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY")
pytestmark = [pytest.mark.deepeval, needs_artifacts,
              pytest.mark.skipif(os.getenv("NETOPS_LLM_PROVIDER", "none") == "none" or
                                 not any(os.getenv(k) for k in LLM_KEYS), reason="no LLM configured")]

CASES = [
    ("A cell is experiencing high latency, packet loss, and reduced throughput. What corrective actions?",
     "Check backhaul utilisation and errors, run per-hop latency/loss tests, validate QoS and re-route traffic."),
    ("Customers are facing frequent dropped calls and handover failures",
     "Audit neighbour relations, tune handover parameters and check radio link failure timers."),
]


@pytest.mark.parametrize("query,reference", CASES)
def test_recommendation_quality(service, query, reference):
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    from app.schemas import QueryRequest
    from evaluation.deepeval_eval import build_wrapper, deepeval_metrics, render_output
    d = service.analyze(QueryRequest(query=query))
    tc = LLMTestCase(input=query, actual_output=render_output(d), expected_output=reference,
                     retrieval_context=[h["text"] for h in d["similar_cases"]])
    metrics = [m for m in deepeval_metrics(build_wrapper(service.llm))
               if getattr(m, "name", "") in ("Recommendation Quality", "Diagnosis Correctness")
               or type(m).__name__ in ("AnswerRelevancyMetric", "FaithfulnessMetric")]
    assert_test(tc, metrics)
