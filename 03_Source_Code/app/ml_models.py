"""Predictive models: network fault-risk prediction and network performance prediction.

* FaultRiskModel     - regresses Fault Occurrence Rate (%) and classifies "high fault risk" (>= fleet P75)
* PerformanceModel   - forecasts Throughput, Latency and Call Drop Rate from operating conditions
Both expose `explain()` - local, model-agnostic feature contributions (median-substitution ablation),
which feed the root-cause agent.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from utils.features import NUMERIC_COLUMNS

CAT_FEATURES = ["Season", "Weather"]
FAULT_TARGET = "Fault Occurrence Rate (%)"
PERF_TARGETS = ["Throughput (Mbps)", "Latency (ms)", "Call Drop Rate (%)"]
FAULT_FEATURES = [c for c in NUMERIC_COLUMNS if c != FAULT_TARGET] + CAT_FEATURES
# Early-warning variant: drops near-coincident symptoms (critical alarms, throughput collapse) so it
# forecasts fault risk from *leading* indicators only (latency, loss, auth failures, load, environment...).
EARLY_FEATURES = [f for f in FAULT_FEATURES if f not in ("Critical Alarm Count", "Throughput (Mbps)")]
# performance models must not see the KPIs they predict, nor fault rate (a consequence, not a cause)
PERF_EXCLUDE = set(PERF_TARGETS) | {FAULT_TARGET, "Critical Alarm Count", "Authentication Failures"}


def _pipeline(estimator, features: list[str]) -> Pipeline:
    cats = [f for f in features if f in CAT_FEATURES]
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats)],
                            remainder="passthrough", verbose_feature_names_out=False)
    est = estimator.set_params(categorical_features=list(range(len(cats))))
    return Pipeline([("pre", pre), ("model", est)])


def perf_features(target: str) -> list[str]:
    # Throughput depends strongly on alarm state - we allow alarm *count* but not the critical subset,
    # so the forecast reflects operating conditions rather than the outcome itself.
    return [c for c in NUMERIC_COLUMNS if c not in PERF_EXCLUDE] + CAT_FEATURES


def local_contributions(predict, X: pd.DataFrame, baseline: dict, features: list[str], top: int = 6) -> list[dict]:
    """Signed effect of each feature: f(x) - f(x with feature set to fleet median)."""
    base_pred = float(predict(X)[0])
    variants = []
    for f in features:
        v = X.copy()
        v[f] = baseline[f]
        variants.append(v)
    preds = predict(pd.concat(variants, ignore_index=True))
    contrib = [{"feature": f, "value": X[f].iloc[0] if not isinstance(X[f].iloc[0], np.generic) else X[f].iloc[0].item(),
                "baseline": baseline[f], "contribution": round(base_pred - float(p), 3)}
               for f, p in zip(features, preds)]
    contrib.sort(key=lambda d: -abs(d["contribution"]))
    return [c for c in contrib[:top] if abs(c["contribution"]) > 1e-6]


class FaultRiskModel:
    def __init__(self):
        self.reg = _pipeline(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42),
                             FAULT_FEATURES)
        self.clf = _pipeline(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=42),
                             FAULT_FEATURES)
        self.early = _pipeline(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42),
                               EARLY_FEATURES)
        self.high_risk_threshold: float = 0.0
        self.baseline: dict = {}

    def fit(self, df: pd.DataFrame) -> "FaultRiskModel":
        X, y = df[FAULT_FEATURES], df[FAULT_TARGET]
        self.high_risk_threshold = float(y.quantile(0.75))
        self.reg.fit(X, y)
        self.clf.fit(X, (y >= self.high_risk_threshold).astype(int))
        self.early.fit(df[EARLY_FEATURES], y)
        self.baseline = {f: (df[f].median() if f not in CAT_FEATURES else df[f].mode()[0]) for f in FAULT_FEATURES}
        return self

    def predict(self, X: pd.DataFrame) -> dict:
        fr = np.clip(self.reg.predict(X[FAULT_FEATURES]), 0, 100)
        p = self.clf.predict_proba(X[FAULT_FEATURES])[:, 1]
        early = np.clip(self.early.predict(X[EARLY_FEATURES]), 0, 100)
        return {"fault_rate": fr, "high_risk_probability": p, "early_warning_fault_rate": early}

    def explain(self, X: pd.DataFrame, top: int = 6) -> list[dict]:
        return local_contributions(lambda d: self.reg.predict(d[FAULT_FEATURES]), X[FAULT_FEATURES],
                                   self.baseline, [f for f in FAULT_FEATURES if f not in CAT_FEATURES], top)


class PerformanceModel:
    def __init__(self):
        self.models = {t: _pipeline(HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, random_state=42),
                                    perf_features(t)) for t in PERF_TARGETS}
        self.baseline: dict = {}

    def fit(self, df: pd.DataFrame) -> "PerformanceModel":
        for t, m in self.models.items():
            m.fit(df[perf_features(t)], df[t])
        feats = sorted({f for t in PERF_TARGETS for f in perf_features(t)})
        self.baseline = {f: (df[f].median() if f not in CAT_FEATURES else df[f].mode()[0]) for f in feats}
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {t: m.predict(X[perf_features(t)]) for t, m in self.models.items()}

    def explain(self, X: pd.DataFrame, target: str, top: int = 5) -> list[dict]:
        feats = perf_features(target)
        return local_contributions(lambda d: self.models[target].predict(d[feats]), X[feats], self.baseline,
                                   [f for f in feats if f not in CAT_FEATURES], top)


class ModelBundle:
    def __init__(self, fault: FaultRiskModel, perf: PerformanceModel, metrics: dict):
        self.fault, self.perf, self.metrics = fault, perf, metrics

    def save(self, d: Path) -> None:
        joblib.dump({"fault": self.fault, "perf": self.perf}, d / "models.joblib")
        (d / "model_metrics.json").write_text(json.dumps(self.metrics, indent=2))

    @classmethod
    def load(cls, d: Path) -> "ModelBundle":
        obj = joblib.load(d / "models.joblib")
        metrics = json.loads((d / "model_metrics.json").read_text())
        return cls(obj["fault"], obj["perf"], metrics)
