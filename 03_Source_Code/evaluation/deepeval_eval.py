"""Recommendation-quality evaluation with DeepEval (+ deterministic metrics that always run).

    python -m evaluation.deepeval_eval              # offline metrics (+ DeepEval if an LLM key is set)
    python -m evaluation.deepeval_eval --no-deepeval

DeepEval metrics (LLM judge = the configured provider, wrapped as a DeepEvalBaseLLM):
  AnswerRelevancy, Faithfulness, ContextualRelevancy, GEval "Recommendation Quality", GEval "Diagnosis Correctness"
Deterministic metrics:
  category accuracy, symptom-extraction F1, SOP coverage, action keyword recall, citation validity,
  heuristic-judge scores, latency
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def build_wrapper(llm):
    from deepeval.models import DeepEvalBaseLLM

    class NetOpsJudgeLLM(DeepEvalBaseLLM):
        def __init__(self):
            self.llm = llm

        def load_model(self):
            return self.llm

        def generate(self, prompt: str, schema=None, **kw):
            if schema is not None:
                data = self.llm.generate_json("Return only valid JSON matching the requested schema.", prompt,
                                              max_tokens=1500, temperature=0.0)
                return schema(**data)
            return self.llm.generate("You are a precise evaluator.", prompt, max_tokens=1500, temperature=0.0)

        async def a_generate(self, prompt: str, schema=None, **kw):
            return self.generate(prompt, schema)

        def get_model_name(self):
            return f"{self.llm.provider}:{self.llm.model}"

    return NetOpsJudgeLLM()


def deepeval_metrics(model):
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCaseParams as P
    return [
        AnswerRelevancyMetric(model=model, threshold=0.6, async_mode=False),
        FaithfulnessMetric(model=model, threshold=0.6, async_mode=False),
        ContextualRelevancyMetric(model=model, threshold=0.5, async_mode=False),
        GEval(name="Recommendation Quality", model=model, threshold=0.6, async_mode=False,
              evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
              evaluation_steps=["Check the actions address every symptom in the input.",
                                "Check actions are specific, ordered and executable by a NOC engineer.",
                                "Compare with the expected output: are the key corrective actions present?",
                                "Penalise risky actions that are not flagged for approval."]),
        GEval(name="Diagnosis Correctness", model=model, threshold=0.6, async_mode=False,
              evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.RETRIEVAL_CONTEXT],
              criteria="Is the diagnosis consistent with the KPIs and historical cases in the retrieval context, "
                       "without unsupported claims?"),
    ]


def render_output(d: dict) -> str:
    rec = d["recommendation"]
    acts = "\n".join(f"{i + 1}. {a['step']}" for i, a in enumerate(rec["actions"]))
    return f"{rec['summary']}\nDiagnosis: {rec['diagnosis']}\nActions:\n{acts}"


def offline_metrics(g: dict, d: dict) -> dict:
    cats = g["expected_categories"]
    pred_conds = set(d["understanding"]["conditions"])
    exp = set(g["expected_conditions"])
    tp = len(pred_conds & exp)
    prec = tp / len(pred_conds) if pred_conds else 0
    rec_ = tp / len(exp) if exp else 1
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0
    sops = {e[4:] for a in d["recommendation"]["actions"] for e in a["evidence"] if e.startswith("SOP:")}
    text = render_output(d).lower()
    kw = [k for k in g["action_keywords"] if k.lower() in text]
    ids = {h["case_id"] for h in d["similar_cases"]}
    cites = [e for a in d["recommendation"]["actions"] for e in a["evidence"] if e.startswith("NOC-")]
    case_rel = [len(exp & {c for c in _keys(h)}) / len(exp) for h in d["similar_cases"]] if exp else [1]
    return {"category_correct": d["classification"]["category"] in cats,
            "category_top1_strict": d["classification"]["category"] == cats[0],
            "condition_f1": round(f1, 3),
            "sop_coverage": round(len(sops & set(g["expected_sops"])) / len(g["expected_sops"]), 3),
            "action_keyword_recall": round(len(kw) / len(g["action_keywords"]), 3),
            "citation_validity": 1.0 if all(c in ids for c in cites) else 0.0,
            "retrieval_symptom_match@k": round(statistics.mean(case_rel), 3),
            "judge_overall": d["judge"]["overall"], "judge_verdict": d["judge"]["verdict"],
            **{f"judge_{k}": v for k, v in d["judge"]["scores"].items()}}


_LABEL2KEY = None


def _keys(hit: dict) -> set[str]:
    global _LABEL2KEY
    if _LABEL2KEY is None:
        from utils.telecom_kb import CONDITION_BY_KEY
        _LABEL2KEY = {c.label: k for k, c in CONDITION_BY_KEY.items()}
    return {_LABEL2KEY[l] for l in hit["conditions"] if l in _LABEL2KEY}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-deepeval", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    from app.schemas import QueryRequest
    from app.services import NetOpsService
    svc = NetOpsService()
    golden = json.loads((ROOT / "golden_queries.json").read_text(encoding="utf-8"))
    if args.limit:
        golden = golden[:args.limit]

    use_deepeval = not args.no_deepeval and svc.llm.available
    model = build_wrapper(svc.llm) if use_deepeval else None
    rows = []
    for g in golden:
        t0 = time.perf_counter()
        d = svc.analyze(QueryRequest(query=g["query"], top_k=5))
        latency = (time.perf_counter() - t0) * 1000
        row = {"id": g["id"], "query": g["query"], "category": d["classification"]["category"],
               "priority": d["prediction"]["priority"], "risk": d["prediction"]["risk_score"],
               "latency_ms": round(latency, 1), "method": d["recommendation"]["method"],
               **offline_metrics(g, d)}
        if use_deepeval:
            from deepeval.test_case import LLMTestCase
            tc = LLMTestCase(input=g["query"], actual_output=render_output(d), expected_output=g["reference"],
                             retrieval_context=[h["text"] for h in d["similar_cases"]] +
                                               [f"SOP {s['name']}" for s in d["sops"]])
            for m in deepeval_metrics(model):
                try:
                    m.measure(tc)
                    row[f"deepeval_{m.__name__ if hasattr(m, '__name__') else m.name}"] = round(m.score, 3)
                except Exception as exc:
                    row[f"deepeval_{getattr(m, 'name', type(m).__name__)}_error"] = str(exc)[:120]
        rows.append(row)
        print(f"{g['id']} cat={row['category']:<16} ok={row['category_correct']} f1={row['condition_f1']} "
              f"sop={row['sop_coverage']} kw={row['action_keyword_recall']} judge={row['judge_overall']} "
              f"{row['latency_ms']:.0f}ms")

    numeric = [k for k in rows[0] if isinstance(rows[0][k], (int, float, bool)) and k not in ("risk",)]
    summary = {k: round(statistics.mean(float(r[k]) for r in rows if k in r), 3) for k in numeric}
    summary |= {"n_queries": len(rows), "llm": svc.llm.describe(), "deepeval": use_deepeval,
                "embedding_model": svc.retriever.embedder.name, "n_cases_indexed": len(svc.retriever.cases),
                "p95_latency_ms": round(sorted(r["latency_ms"] for r in rows)[int(0.95 * (len(rows) - 1))], 1)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "recommendation_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
