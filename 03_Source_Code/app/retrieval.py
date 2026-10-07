"""Hybrid retrieval: dense (FAISS) + sparse (BM25) + Reciprocal Rank Fusion + structured re-ranking.

Two corpora are indexed:
  * incidents - 100K historical network records turned into narratives (case-based reasoning)
  * playbook  - troubleshooting SOPs (grounding for recommendations)
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from utils.telecom_kb import CONDITION_BY_KEY, playbook_documents

log = logging.getLogger("netops.retrieval")

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b"


# ---------------------------------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------------------------------
class Embedder:
    name: str = "base"
    dim: int = 0

    def encode(self, texts: list[str], batch_size: int = 128, show_progress: bool = False) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode(self, texts, batch_size=128, show_progress=False):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=show_progress, convert_to_numpy=True).astype(np.float32)


class TfidfLsaEmbedder(Embedder):
    """Offline fallback: TF-IDF + truncated SVD (latent semantic analysis). No model download needed."""

    def __init__(self, dim: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self.name, self.dim = "tfidf-lsa", dim

    def fit(self, texts: list[str]) -> "TfidfLsaEmbedder":
        self.svd.fit(self.vec.fit_transform(texts))
        return self

    def encode(self, texts, batch_size=128, show_progress=False):
        z = self.svd.transform(self.vec.transform(texts))
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
        return z.astype(np.float32)


def make_embedder(model_name: str, backend: str = "auto", artifacts_dir: Path | None = None) -> Embedder:
    if backend in ("auto", "sentence-transformers") and model_name != "tfidf-lsa":
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:  # no internet / model not cached
            if backend == "sentence-transformers":
                raise
            log.warning("Sentence-transformer unavailable (%s); falling back to TF-IDF/LSA", exc)
    if artifacts_dir and (artifacts_dir / "tfidf_lsa.joblib").exists():
        return joblib.load(artifacts_dir / "tfidf_lsa.joblib")
    return TfidfLsaEmbedder()


# ---------------------------------------------------------------------------------------------------
# Sparse BM25 (vectorised with scipy - scores 100K docs in a few ms)
# ---------------------------------------------------------------------------------------------------
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), min_df=1,
                                          lowercase=True, dtype=np.float32)
        self.weights: sparse.csc_matrix | None = None

    def fit(self, texts: list[str]) -> "BM25Index":
        tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        # BM25 term weight per (doc, term) - precomputed so a query is a sparse column sum
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[rows] / avgdl)
        data = idf[tf.indices] * tf.data * (self.k1 + 1) / denom
        self.weights = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).tocsc()
        return self

    def query_terms(self, query: str) -> np.ndarray:
        analyzer = self.vectorizer.build_analyzer()
        vocab = self.vectorizer.vocabulary_
        return np.array(sorted({vocab[t] for t in analyzer(query) if t in vocab}), dtype=np.int64)

    def scores(self, query: str) -> np.ndarray:
        ids = self.query_terms(query)
        if ids.size == 0:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, ids].sum(axis=1)).ravel()


# ---------------------------------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------------------------------
@dataclass
class SearchFilters:
    state: list[str] | None = None
    city: list[str] | None = None
    season: list[str] | None = None
    weather: list[str] | None = None
    category: list[str] | None = None
    priority: list[str] | None = None
    min_risk: float | None = None
    required_conditions: list[str] | None = None
    issue_reported: str | None = None

    def active(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], "")}


@dataclass
class Hit:
    case_id: str
    row: int
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    condition_overlap: float = 0.0
    feedback_boost: float = 0.0
    record: dict = field(default_factory=dict)


def keyword_text(df: pd.DataFrame) -> list[str]:
    """Field used for BM25: location/season/weather + condition labels + category (no numeric noise)."""
    labels = {c: CONDITION_BY_KEY[c].label for c in CONDITION_BY_KEY}
    out = []
    for season, city, state, weather, conds, cat, prio, sev in zip(
            df["Season"], df["City"], df["State"], df["Weather"], df["conditions"], df["category_name"],
            df["priority"], df["severity"]):
        cond_txt = " ; ".join(labels[c] for c in conds.split(",") if c)
        out.append(f"{season} {city} {state} {weather} {cond_txt} {cat} {prio} {sev}")
    return out


def summary_text(full_text: str) -> str:
    """Field used for dense embeddings: the part before the raw KPI dump (fits MiniLM's 256-token window)."""
    return full_text.split(" KPIs:")[0]


class HybridRetriever:
    def __init__(self, cases: pd.DataFrame, embedder: Embedder, faiss_index, bm25: BM25Index,
                 playbook_emb: np.ndarray, rrf_k: int = 60, candidate_pool: int = 200):
        self.cases = cases.reset_index(drop=True)
        self.embedder, self.index, self.bm25 = embedder, faiss_index, bm25
        self.rrf_k, self.candidate_pool = rrf_k, candidate_pool
        self.playbook = playbook_documents()
        self.playbook_emb = playbook_emb
        self.case_pos = {cid: i for i, cid in enumerate(self.cases["case_id"])}
        self._cond_cols = [c for c in self.cases.columns if c.startswith("c_")]
        self._cond_matrix = (self.cases[self._cond_cols].to_numpy() > 0)
        self._lower = {col: self.cases[col].str.lower().to_numpy() for col in ("State", "City", "Season", "Weather")}

    # ---- persistence -------------------------------------------------------------------------------
    @classmethod
    def load(cls, artifacts_dir: Path, embedder: Embedder | None = None, **kw) -> "HybridRetriever":
        import faiss
        meta = json.loads((artifacts_dir / "index_meta.json").read_text())
        cases = pd.read_parquet(artifacts_dir / "cases_indexed.parquet")
        if embedder is None:
            embedder = make_embedder(meta["embedding_model"], meta.get("backend", "auto"), artifacts_dir)
            if embedder.name != meta["embedding_model"]:
                raise RuntimeError(f"Index was built with {meta['embedding_model']} but {embedder.name} loaded; "
                                   "rebuild with `python -m data.build_index`.")
        index = faiss.read_index(str(artifacts_dir / "incidents.faiss"))
        bm25 = joblib.load(artifacts_dir / "bm25.joblib")
        pb = np.load(artifacts_dir / "playbook_emb.npy")
        return cls(cases, embedder, index, bm25, pb, **kw)

    # ---- filtering ---------------------------------------------------------------------------------
    def filter_mask(self, f: SearchFilters) -> np.ndarray:
        m = np.ones(len(self.cases), dtype=bool)
        for attr, col in (("state", "State"), ("city", "City"), ("season", "Season"), ("weather", "Weather")):
            vals = getattr(f, attr)
            if vals:
                m &= np.isin(self._lower[col], [v.lower() for v in vals])
        if f.category:
            m &= self.cases["category"].isin(f.category).to_numpy()
        if f.priority:
            m &= self.cases["priority"].isin(f.priority).to_numpy()
        if f.min_risk is not None:
            m &= (self.cases["risk_score"] >= f.min_risk).to_numpy()
        if f.issue_reported:
            m &= (self.cases["Issue Reported"] == f.issue_reported.lower()).to_numpy()
        if f.required_conditions:
            for c in f.required_conditions:
                col = f"c_{c}"
                if col in self.cases:
                    m &= (self.cases[col] > 0).to_numpy()
        return m

    # ---- retrievers --------------------------------------------------------------------------------
    def _dense(self, qv: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        import faiss
        n_ok = int(mask.sum())
        if n_ok == len(mask):
            scores, ids = self.index.search(qv[None, :], k)
        else:
            sel = faiss.IDSelectorBatch(np.flatnonzero(mask).astype(np.int64))
            scores, ids = self.index.search(qv[None, :], min(k, n_ok), params=faiss.SearchParameters(sel=sel))
        keep = ids[0] >= 0
        return ids[0][keep], scores[0][keep]

    def _sparse(self, query: str, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.bm25.scores(query)
        s = np.where(mask, s, -np.inf)
        k = min(k, int(mask.sum()))
        if k == 0:
            return np.array([], dtype=int), np.array([])
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        top = top[s[top] > 0]
        return top, s[top]

    def search(self, query: str, filters: SearchFilters | None = None, top_k: int = 5, alpha: float = 0.6,
               mode: str = "hybrid", query_conditions: list[str] | None = None, bm25_query: str | None = None,
               feedback_scores: dict[str, float] | None = None, feedback_weight: float = 0.15) -> tuple[list[Hit], dict]:
        t0 = time.perf_counter()
        filters = filters or SearchFilters()
        mask = self.filter_mask(filters)
        stats = {"candidates_after_filter": int(mask.sum()), "mode": mode, "alpha": alpha}
        if not mask.any():
            return [], stats | {"latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        pool = max(self.candidate_pool, top_k * 4)

        dense_ids, dense_sc = (np.array([], int), np.array([]))
        sparse_ids, sparse_sc = (np.array([], int), np.array([]))
        if mode in ("hybrid", "semantic"):
            qv = self.embedder.encode([query])[0]
            dense_ids, dense_sc = self._dense(qv, mask, pool)
        if mode in ("hybrid", "keyword"):
            sparse_ids, sparse_sc = self._sparse(bm25_query or query, mask, pool)

        fused: dict[int, Hit] = {}
        w_dense = {"hybrid": alpha, "semantic": 1.0, "keyword": 0.0}[mode]
        for rank, (i, s) in enumerate(zip(dense_ids, dense_sc)):
            h = fused.setdefault(int(i), Hit(self.cases.at[int(i), "case_id"], int(i), 0.0))
            h.dense_score, h.dense_rank = float(s), rank + 1
            h.score += w_dense / (self.rrf_k + rank + 1)
        for rank, (i, s) in enumerate(zip(sparse_ids, sparse_sc)):
            h = fused.setdefault(int(i), Hit(self.cases.at[int(i), "case_id"], int(i), 0.0))
            h.bm25_score, h.bm25_rank = float(s), rank + 1
            h.score += (1 - w_dense) / (self.rrf_k + rank + 1)

        hits = list(fused.values())
        if not hits:
            return [], stats | {"latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        # normalise fused RRF score to 0..1, then structured re-ranking
        top_rrf = max(h.score for h in hits) or 1.0
        qset = set(query_conditions or [])
        col_idx = {c[2:]: j for j, c in enumerate(self._cond_cols)}
        for h in hits:
            base = h.score / top_rrf
            if qset:
                case_conds = {k for k, j in col_idx.items() if self._cond_matrix[h.row, j]}
                h.condition_overlap = len(qset & case_conds) / len(qset)
                base = 0.65 * base + 0.35 * h.condition_overlap
            if feedback_scores:
                h.feedback_boost = float(np.tanh(feedback_scores.get(h.case_id, 0.0))) * feedback_weight
            h.score = round(base + h.feedback_boost, 4)
        hits.sort(key=lambda h: -h.score)
        hits = hits[:top_k]
        for h in hits:
            h.record = self.record(h.row)
        stats["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return hits, stats

    def search_playbook(self, query: str, categories: list[str] | None = None, k: int = 2) -> list[dict]:
        qv = self.embedder.encode([query])[0]
        sims = self.playbook_emb @ qv
        order = np.argsort(-sims)
        out, seen = [], set()
        for cat in categories or []:            # categories decided by the classifier come first
            for i, d in enumerate(self.playbook):
                if d["category"] == cat and cat not in seen:
                    out.append(d | {"similarity": float(sims[i])})
                    seen.add(cat)
        for i in order:
            if len(out) >= max(k, len(seen)):
                break
            d = self.playbook[i]
            if d["category"] not in seen and d["category"] != "NORMAL":
                out.append(d | {"similarity": float(sims[i])})
                seen.add(d["category"])
        return out[:max(k, len(categories or []))]

    def record(self, row: int) -> dict:
        r = self.cases.iloc[row]
        drop = {c for c in self._cond_cols}
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if k not in drop}

    def get(self, case_id: str) -> dict | None:
        i = self.case_pos.get(case_id)
        return None if i is None else self.record(i)


_ws = re.compile(r"\s+")


def clean_ws(s: str) -> str:
    return _ws.sub(" ", s).strip()
