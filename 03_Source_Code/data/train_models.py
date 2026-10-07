"""Step 3 of the offline pipeline: train & evaluate the predictive models.

    python -m data.train_models

Output: data/artifacts/models.joblib, data/artifacts/model_metrics.json
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.ml_models import (FAULT_FEATURES, FAULT_TARGET, PERF_TARGETS, FaultRiskModel, ModelBundle,
                           PerformanceModel, perf_features)
from utils.logging_utils import setup_logging

log = logging.getLogger("netops.train")


def main() -> None:
    setup_logging()
    s = get_settings()
    t0 = time.time()
    df = pd.read_parquet(s.artifacts_dir / "cases.parquet")
    train, test = train_test_split(df, test_size=0.2, random_state=s.random_seed)

    fault = FaultRiskModel().fit(train)
    pred = fault.predict(test)
    y = test[FAULT_TARGET].to_numpy()
    yb = (y >= fault.high_risk_threshold).astype(int)
    baseline_mae = mean_absolute_error(y, DummyRegressor().fit(train[FAULT_FEATURES], train[FAULT_TARGET])
                                       .predict(test[FAULT_FEATURES]))
    imp = permutation_importance(fault.reg, test[FAULT_FEATURES].sample(4000, random_state=1),
                                 test.loc[test[FAULT_FEATURES].sample(4000, random_state=1).index, FAULT_TARGET],
                                 n_repeats=3, random_state=1)
    importance = sorted(zip(FAULT_FEATURES, imp.importances_mean), key=lambda kv: -kv[1])
    metrics = {"fault_risk": {
        "target": FAULT_TARGET, "algorithm": "HistGradientBoosting (regressor + classifier)",
        "n_train": len(train), "n_test": len(test),
        "r2": round(r2_score(y, pred["fault_rate"]), 4), "mae": round(mean_absolute_error(y, pred["fault_rate"]), 3),
        "baseline_mae_mean_predictor": round(baseline_mae, 3),
        "early_warning_r2": round(r2_score(y, pred["early_warning_fault_rate"]), 4),
        "early_warning_mae": round(mean_absolute_error(y, pred["early_warning_fault_rate"]), 3),
        "high_risk_threshold": round(fault.high_risk_threshold, 2),
        "roc_auc": round(roc_auc_score(yb, pred["high_risk_probability"]), 4),
        "f1": round(f1_score(yb, pred["high_risk_probability"] >= 0.5), 4),
        "accuracy": round(accuracy_score(yb, pred["high_risk_probability"] >= 0.5), 4),
        "permutation_importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in importance[:10]],
    }, "performance": {}}

    perf = PerformanceModel().fit(train)
    pp = perf.predict(test)
    for t in PERF_TARGETS:
        yt = test[t].to_numpy()
        metrics["performance"][t] = {"r2": round(r2_score(yt, pp[t]), 4), "mae": round(mean_absolute_error(yt, pp[t]), 3),
                                     "baseline_mae": round(float(np.mean(np.abs(yt - train[t].mean()))), 3),
                                     "n_features": len(perf_features(t))}
    # "Issue Reported" sanity check - documented finding: the flag is statistically independent of KPIs
    metrics["notes"] = ["'Issue Reported' is ~50/50 and independent of every KPI (label noise) - not used as a target.",
                        "Fault Occurrence Rate is (near-)deterministic given all KPIs (R2~1.0): the full model is used for "
                        "what-if analysis; the early-warning model (no critical alarms / throughput) is the honest "
                        "forecast from leading indicators.",
                        "Fault Occurrence Rate is driven mainly by Critical Alarm Count, Authentication Failures "
                        "and Call Drop Rate."]
    ModelBundle(fault, perf, metrics).save(s.artifacts_dir)
    log.info("Fault model: R2=%.3f MAE=%.2f AUC=%.3f | early-warning R2=%.3f", metrics["fault_risk"]["r2"],
             metrics["fault_risk"]["mae"], metrics["fault_risk"]["roc_auc"], metrics["fault_risk"]["early_warning_r2"])
    for t, m in metrics["performance"].items():
        log.info("Perf %-22s R2=%.3f MAE=%.2f (baseline MAE %.2f)", t, m["r2"], m["mae"], m["baseline_mae"])
    log.info("Training done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
