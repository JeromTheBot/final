"""Step 2 of the offline pipeline: embeddings + FAISS vector index + BM25 keyword index.

    python -m data.build_index                # all records (NETOPS_INDEX_ROWS, default 100K)
    python -m data.build_index --rows 20000   # quicker build for development
    python -m data.build_index --backend tfidf  # fully offline (no model download)
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import faiss
import joblib
import numpy as np
import pandas as pd

from app.config import get_settings
from app.retrieval import BM25Index, TfidfLsaEmbedder, keyword_text, make_embedder, summary_text
from utils.logging_utils import setup_logging
from utils.telecom_kb import playbook_documents

log = logging.getLogger("netops.index")


def main() -> None:
    setup_logging()
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=s.index_rows)
    ap.add_argument("--backend", default=s.embedding_backend, choices=["auto", "sentence-transformers", "tfidf"])
    ap.add_argument("--model", default=s.embedding_model)
    args = ap.parse_args()

    cases = pd.read_parquet(s.artifacts_dir / "cases.parquet")
    if args.rows < len(cases):
        cases = cases.sample(args.rows, random_state=s.random_seed).sort_index()
    cases = cases.reset_index(drop=True)
    summaries = [summary_text(t) for t in cases["text"]]
    log.info("Indexing %d cases", len(cases))

    t0 = time.time()
    if args.backend == "tfidf":
        emb = TfidfLsaEmbedder().fit(summaries)
        joblib.dump(emb, s.artifacts_dir / "tfidf_lsa.joblib")
    else:
        emb = make_embedder(args.model, args.backend, s.artifacts_dir)
        if isinstance(emb, TfidfLsaEmbedder):
            emb.fit(summaries)
            joblib.dump(emb, s.artifacts_dir / "tfidf_lsa.joblib")
    vectors = emb.encode(summaries, batch_size=128, show_progress=True)
    log.info("Embedded with %s (dim=%d) in %.0fs", emb.name, emb.dim, time.time() - t0)

    index = faiss.IndexFlatIP(vectors.shape[1])   # exact cosine search (vectors are L2-normalised)
    index.add(vectors)
    faiss.write_index(index, str(s.artifacts_dir / "incidents.faiss"))

    t1 = time.time()
    bm25 = BM25Index().fit(keyword_text(cases))
    joblib.dump(bm25, s.artifacts_dir / "bm25.joblib")
    log.info("BM25 index built in %.1fs (vocab=%d)", time.time() - t1, len(bm25.vectorizer.vocabulary_))

    pb = emb.encode([d["text"] for d in playbook_documents()])
    np.save(s.artifacts_dir / "playbook_emb.npy", pb)

    cases.to_parquet(s.artifacts_dir / "cases_indexed.parquet", index=False)
    meta = {"embedding_model": emb.name, "backend": "tfidf" if emb.name == "tfidf-lsa" else "sentence-transformers",
            "dim": emb.dim, "n_cases": len(cases), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "index_type": "faiss.IndexFlatIP", "sparse": "BM25 (k1=1.5, b=0.75, uni+bi-grams)"}
    (s.artifacts_dir / "index_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Done in %.0fs: %s", time.time() - t0, meta)


if __name__ == "__main__":
    main()
