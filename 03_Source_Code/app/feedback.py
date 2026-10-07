"""Feedback loop: engineers rate answers; ratings flow back into retrieval ranking and action ordering.

  * case-level score   -> re-ranking boost/penalty for historical cases that were (un)helpful
  * action-level score -> recommended actions that worked are promoted next time for that category
  * category corrections -> logged as labelled examples for the evaluation set / future fine-tuning
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    response_id TEXT PRIMARY KEY, ts REAL, query TEXT, category TEXT, case_ids TEXT, actions TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, response_id TEXT, rating INTEGER, helpful INTEGER,
    correct_category TEXT, helpful_case_ids TEXT, unhelpful_case_ids TEXT, effective_actions TEXT, comment TEXT);
"""


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
        self._cache: tuple[float, dict, dict] | None = None

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    # ---- writes -----------------------------------------------------------------------------------
    def log_response(self, query: str, category: str, case_ids: list[str], actions: list[str], payload: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO responses VALUES (?,?,?,?,?,?,?)",
                      (rid, time.time(), query, category, json.dumps(case_ids), json.dumps(actions),
                       json.dumps(payload, default=str)[:200_000]))
        return rid

    def add_feedback(self, response_id: str, rating: int, helpful: bool | None = None,
                     correct_category: str | None = None, helpful_case_ids: list[str] | None = None,
                     unhelpful_case_ids: list[str] | None = None, effective_actions: list[str] | None = None,
                     comment: str = "") -> int:
        resp = self.get_response(response_id)
        if resp is None:
            raise KeyError(response_id)
        if helpful is None:
            helpful = rating >= 4
        # default attribution: a rating applies to all cases that were shown
        if helpful_case_ids is None and unhelpful_case_ids is None:
            (helpful_case_ids, unhelpful_case_ids) = (resp["case_ids"], []) if helpful else ([], resp["case_ids"])
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO feedback (ts,response_id,rating,helpful,correct_category,helpful_case_ids,"
                "unhelpful_case_ids,effective_actions,comment) VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), response_id, int(rating), int(bool(helpful)), correct_category,
                 json.dumps(helpful_case_ids or []), json.dumps(unhelpful_case_ids or []),
                 json.dumps(effective_actions or []), comment[:2000]))
            fid = cur.lastrowid
        self._cache = None
        return fid

    # ---- reads ------------------------------------------------------------------------------------
    def get_response(self, response_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT response_id, ts, query, category, case_ids, actions FROM responses "
                          "WHERE response_id=?", (response_id,)).fetchone()
        if not r:
            return None
        return {"response_id": r[0], "ts": r[1], "query": r[2], "category": r[3],
                "case_ids": json.loads(r[4]), "actions": json.loads(r[5])}

    def _scores(self) -> tuple[dict, dict]:
        if self._cache and time.time() - self._cache[0] < 5:
            return self._cache[1], self._cache[2]
        case_scores: dict[str, float] = {}
        action_scores: dict[str, dict[str, float]] = {}
        with self._conn() as c:
            rows = c.execute("SELECT f.rating, f.helpful_case_ids, f.unhelpful_case_ids, f.effective_actions, "
                             "r.category, r.actions, f.helpful FROM feedback f JOIN responses r "
                             "ON f.response_id=r.response_id").fetchall()
        for rating, hc, uc, ea, cat, shown_actions, helpful in rows:
            w = (rating - 3) / 2.0          # 1..5 -> -1..+1
            for cid in json.loads(hc):
                case_scores[cid] = case_scores.get(cid, 0) + max(w, 0.25)
            for cid in json.loads(uc):
                case_scores[cid] = case_scores.get(cid, 0) - max(-w, 0.25)
            acts = json.loads(ea) or (json.loads(shown_actions) if helpful else [])
            for a in acts:
                d = action_scores.setdefault(cat, {})
                d[a] = d.get(a, 0) + (1.0 if helpful else -0.5)
        self._cache = (time.time(), case_scores, action_scores)
        return case_scores, action_scores

    def case_scores(self) -> dict[str, float]:
        return self._scores()[0]

    def action_scores(self, category: str) -> dict[str, float]:
        return self._scores()[1].get(category, {})

    def stats(self) -> dict:
        with self._conn() as c:
            n_resp = c.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            n_fb, avg, helpful = c.execute("SELECT COUNT(*), AVG(rating), AVG(helpful) FROM feedback").fetchone()
            by_cat = c.execute("SELECT r.category, COUNT(*), AVG(f.rating) FROM feedback f JOIN responses r "
                               "ON f.response_id=r.response_id GROUP BY r.category").fetchall()
            corrections = c.execute("SELECT r.query, r.category, f.correct_category FROM feedback f JOIN responses r "
                                    "ON f.response_id=r.response_id WHERE f.correct_category IS NOT NULL "
                                    "AND f.correct_category != r.category ORDER BY f.ts DESC LIMIT 20").fetchall()
            recent = c.execute("SELECT f.ts, r.query, f.rating, f.comment FROM feedback f JOIN responses r "
                               "ON f.response_id=r.response_id ORDER BY f.ts DESC LIMIT 10").fetchall()
        cs, _ = self._scores()
        return {"responses_logged": n_resp, "feedback_count": n_fb,
                "avg_rating": round(avg, 2) if avg else None, "helpful_rate": round(helpful, 3) if helpful else None,
                "by_category": [{"category": a, "count": b, "avg_rating": round(c_, 2)} for a, b, c_ in by_cat],
                "category_corrections": [{"query": q, "predicted": p, "corrected": cc} for q, p, cc in corrections],
                "cases_with_feedback": len(cs),
                "recent": [{"ts": t, "query": q, "rating": r, "comment": cm} for t, q, r, cm in recent]}
