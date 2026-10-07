"""Retrieval & embedding-model evaluation.

    python -m evaluation.retrieval_eval                 # strategies on the live index + model comparison
    python -m evaluation.retrieval_eval --skip-models   # strategies only

Labelled queries are generated from symptom combinations and phrased with *paraphrases* (not the
index vocabulary), so keyword matching cannot trivially win. A case is graded by the fraction of the
queried symptoms it actually exhibits (graded relevance); "relevant" = shows all of them.
Metrics: Precision@5, nDCG@10, MRR@10, Recall of symptoms, latency.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from app.config import get_settings
from app.retrieval import BM25Index, HybridRetriever, SearchFilters, keyword_text, make_embedder, summary_text
from utils.query_parser import parse_query

RESULTS = Path(__file__).resolve().parent / "results"
PARAPHRASES = {
    "high_latency": ["slow response times", "large network delays", "latency spikes"],
    "packet_loss": ["packets being dropped", "lost packets on the link", "packet loss"],
    "low_throughput": ["slow download speeds", "low data rates", "throughput has dropped"],
    "call_drops": ["calls keep dropping", "frequent dropped calls"],
    "handover_failure": ["handovers between cells failing", "mobility handoff problems"],
    "poor_signal": ["weak radio signal", "bad coverage"],
    "high_jitter": ["choppy voice and video", "jitter is very high"],
    "critical_alarms": ["critical alarms firing", "many critical alarms"],
    "low_availability": ["cells going down", "site outages"],
    "setup_failure": ["users unable to connect", "connection setup failing"],
    "high_load": ["heavy traffic congestion", "too many users on the cell"],
    "auth_failures": ["authentication failures for subscribers", "auth failure spike"],
    "security_incidents": ["security attacks detected", "intrusion attempts"],
    "config_failure": ["failed configuration changes", "config pushes failing"],
}
TEMPLATES = ["We are seeing {a} and {b} on several cells.", "Cell reports {a}, together with {b}.",
             "Customers complain about {a}; engineers also observe {b}.", "{a} plus {b} in the region - similar cases?"]
TEMPLATES_1 = ["We are seeing {a} on several cells.", "Customers complain about {a}."]


def make_queries(n: int = 40, seed: int = 7) -> list[dict]:
    rnd = random.Random(seed)
    keys = list(PARAPHRASES)
    out = []
    for i in range(n):
        k = rnd.choice([1, 2, 2, 3])
        conds = rnd.sample(keys, k)
        phr = [rnd.choice(PARAPHRASES[c]) for c in conds]
        if k == 1:
            q = rnd.choice(TEMPLATES_1).format(a=phr[0])
        elif k == 2:
            q = rnd.choice(TEMPLATES).format(a=phr[0], b=phr[1])
        else:
            q = rnd.choice(TEMPLATES).format(a=f"{phr[0]}, {phr[1]}", b=phr[2])
        out.append({"id": f"R{i:02d}", "query": q, "conditions": conds})
    return out


def grade(cases: pd.DataFrame, rows: list[int], conds: list[str]) -> list[float]:
    return [float(np.mean([cases.at[r, f"c_{c}"] > 0 for c in conds])) for r in rows]


def metrics_for(gains: list[float], k5: int = 5, k10: int = 10) -> dict:
    g = gains[:k10]
    p5 = float(np.mean([x == 1.0 for x in gains[:k5]])) if gains else 0.0
    dcg = sum(x / np.log2(i + 2) for i, x in enumerate(g))
    ndcg = dcg / sum(1 / np.log2(i + 2) for i in range(len(g))) if g else 0.0
    mrr = next((1 / (i + 1) for i, x in enumerate(g) if x == 1.0), 0.0)
    return {"p@5": p5, "ndcg@10": ndcg, "mrr@10": mrr, "symptom_recall@10": float(np.mean(g)) if g else 0.0}


def run_strategy(r: HybridRetriever, queries, mode: str, rerank: bool, alpha: float = 0.6) -> dict:
    agg, lat = [], []
    for q in queries:
        pq = parse_query(q["query"])
        t0 = time.perf_counter()
        hits, _ = r.search(q["query"], SearchFilters(), top_k=10, alpha=alpha, mode=mode,
                           query_conditions=pq.condition_keys if rerank else None,
                           bm25_query=f"{q['query']} {pq.bm25_expansion()}")
        lat.append((time.perf_counter() - t0) * 1000)
        agg.append(metrics_for(grade(r.cases, [h.row for h in hits], q["conditions"])))
    out = {k: round(float(np.mean([a[k] for a in agg])), 4) for k in agg[0]}
    out["latency_ms_p50"] = round(float(np.median(lat)), 1)
    return out


def build_subset_retriever(cases: pd.DataFrame, model_name: str) -> tuple[HybridRetriever, float]:
    emb = make_embedder(model_name, "tfidf" if model_name == "tfidf-lsa" else "sentence-transformers")
    texts = [summary_text(t) for t in cases["text"]]
    if model_name == "tfidf-lsa":
        emb.fit(texts)
    t0 = time.time()
    vec = emb.encode(texts, batch_size=64)
    enc_s = time.time() - t0
    idx = faiss.IndexFlatIP(vec.shape[1])
    idx.add(vec)
    bm25 = BM25Index().fit(keyword_text(cases))
    pb = np.zeros((1, vec.shape[1]), dtype=np.float32)
    return HybridRetriever(cases, emb, idx, bm25, pb), enc_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--subset", type=int, default=4000)
    ap.add_argument("--models", nargs="*", default=["sentence-transformers/all-MiniLM-L6-v2",
                                                     "BAAI/bge-small-en-v1.5",
                                                     "sentence-transformers/all-mpnet-base-v2", "tfidf-lsa"])
    args = ap.parse_args()
    s = get_settings()
    queries = make_queries()
    report: dict = {"queries": queries, "strategies": {}, "embedding_models": {}}

    live = HybridRetriever.load(s.artifacts_dir)
    print(f"Live index: {len(live.cases)} cases, {live.embedder.name}")
    for name, mode, rr in [("keyword_bm25", "keyword", False), ("semantic_dense", "semantic", False),
                           ("hybrid_rrf", "hybrid", False), ("hybrid_rrf+symptom_rerank", "hybrid", True)]:
        report["strategies"][name] = m = run_strategy(live, queries, mode, rr)
        print(f"{name:<28} {m}")
    for a in (0.3, 0.5, 0.7, 0.9):
        report["strategies"][f"hybrid_alpha_{a}"] = m = run_strategy(live, queries, "hybrid", True, alpha=a)
        print(f"alpha={a:<4} {m}")

    if not args.skip_models:
        sub = pd.read_parquet(s.artifacts_dir / "cases.parquet").sample(args.subset, random_state=3).reset_index(drop=True)
        for model in args.models:
            try:
                r, enc_s = build_subset_retriever(sub, model)
            except Exception as exc:
                print(f"skip {model}: {exc}")
                continue
            res = {"dim": r.embedder.dim, "encode_docs_per_s": round(len(sub) / enc_s, 1),
                   "dense_only": run_strategy(r, queries, "semantic", False),
                   "hybrid+rerank": run_strategy(r, queries, "hybrid", True)}
            report["embedding_models"][model] = res
            print(f"{model:<45} {res}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_eval.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
