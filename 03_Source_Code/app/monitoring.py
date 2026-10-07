"""Lightweight in-process observability: request metrics, latency percentiles, error tracking, alerts.

Exposed at GET /metrics (JSON) and visualised on the Streamlit monitoring page. In production these
counters would be exported to Prometheus / Azure Monitor; the interface is kept deliberately small.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque

import numpy as np


class Metrics:
    def __init__(self, window: int = 2000):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: Counter = Counter()
        self.errors: Counter = Counter()
        self.status: Counter = Counter()
        self.latency: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.events: Counter = Counter()          # guardrail blocks, llm fallbacks, judge revisions...
        self.priorities: deque = deque(maxlen=window)   # (ts, priority) of analysed incidents
        self.recent_errors: deque = deque(maxlen=50)

    def observe(self, path: str, status: int, latency_ms: float) -> None:
        with self._lock:
            self.requests[path] += 1
            self.status[str(status)] += 1
            self.latency[path].append(latency_ms)
            if status >= 500:
                self.errors[path] += 1

    def event(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.events[name] += n

    def record_priority(self, p: str) -> None:
        with self._lock:
            self.priorities.append((time.time(), p))

    def error(self, path: str, msg: str) -> None:
        with self._lock:
            self.recent_errors.append({"ts": time.time(), "path": path, "error": msg[:300]})

    def snapshot(self, p95_alert_ms: float = 5000, error_rate_alert: float = 0.05) -> dict:
        with self._lock:
            per_path = {}
            for path, lat in self.latency.items():
                arr = np.fromiter(lat, float)
                per_path[path] = {"count": self.requests[path], "errors": self.errors[path],
                                  "p50_ms": round(float(np.percentile(arr, 50)), 1),
                                  "p95_ms": round(float(np.percentile(arr, 95)), 1),
                                  "max_ms": round(float(arr.max()), 1)}
            total = sum(self.requests.values())
            errs = sum(self.errors.values())
            hour_ago = time.time() - 3600
            prio_hour = Counter(p for t, p in self.priorities if t >= hour_ago)
            alerts = []
            for path, m in per_path.items():
                if m["p95_ms"] > p95_alert_ms and m["count"] >= 5:
                    alerts.append({"level": "warning", "msg": f"High p95 latency on {path}: {m['p95_ms']} ms"})
            if total >= 20 and errs / total > error_rate_alert:
                alerts.append({"level": "critical", "msg": f"Error rate {errs / total:.1%} above {error_rate_alert:.0%}"})
            if prio_hour.get("P1", 0) >= 5:
                alerts.append({"level": "critical", "msg": f"{prio_hour['P1']} P1 incidents analysed in the last hour"})
            if self.events.get("guardrail_blocked", 0) >= 10:
                alerts.append({"level": "info", "msg": f"{self.events['guardrail_blocked']} queries blocked by guardrails"})
            return {"uptime_s": round(time.time() - self.started), "total_requests": total, "total_errors": errs,
                    "status_codes": dict(self.status), "endpoints": per_path, "events": dict(self.events),
                    "priorities_last_hour": dict(prio_hour), "alerts": alerts,
                    "recent_errors": list(self.recent_errors)[-10:]}


METRICS = Metrics()
