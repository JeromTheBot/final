"""Step 1 of the offline pipeline: download -> clean -> enrich -> persist.

    python -m data.ingest

Output: data/artifacts/cases.parquet  (one row per historical network record + derived fields)
        data/artifacts/thresholds.json (fleet percentiles used for condition detection & risk)
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from app.config import get_settings
from utils.features import (CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, build_narratives, classify, compute_thresholds,
                            detect_conditions, normalise_columns, priority_from_risk, risk_score, save_thresholds,
                            SEVERITY_BY_PRIORITY)
from utils.logging_utils import setup_logging
from utils.telecom_kb import CATEGORY_NAMES

log = logging.getLogger("netops.ingest")


def load_raw() -> pd.DataFrame:
    s = get_settings()
    if s.raw_csv.exists():
        log.info("Loading cached dataset %s", s.raw_csv)
        df = pd.read_csv(s.raw_csv)
    else:
        log.info("Downloading dataset from %s", s.dataset_uri)
        try:  # corporate proxies often need the OS trust store
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        df = pd.read_csv(s.dataset_uri)
        s.raw_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(s.raw_csv, index=False)
    return normalise_columns(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    # physical bounds (guards against corrupt rows; no-ops on the clean HF dataset)
    for col in [c for c in NUMERIC_COLUMNS if c.endswith("(%)")]:
        df[col] = df[col].clip(0, 100)
    df["Signal Strength (dBm)"] = df["Signal Strength (dBm)"].clip(-140, -30)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    df["Season"] = df["Season"].str.title()
    df["Weather"] = df["Weather"].str.title()
    df["State"] = df["State"].str.upper()
    df["Issue Reported"] = df["Issue Reported"].str.lower()
    log.info("Cleaned: %d -> %d rows", before, len(df))
    return df


def enrich(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    th = compute_thresholds(df)
    cond = detect_conditions(df, th)
    category, _ = classify(cond, df["Weather"])
    risk = risk_score(df, cond, th)
    prio = priority_from_risk(risk, cond["critical_alarms"].to_numpy())
    df = df.copy()
    df.insert(0, "case_id", [f"NOC-{i:06d}" for i in range(len(df))])
    df["category"] = category.values
    df["category_name"] = df["category"].map(CATEGORY_NAMES)
    df["risk_score"] = risk
    df["priority"] = prio
    df["severity"] = df["priority"].map(SEVERITY_BY_PRIORITY)
    df["conditions"] = [",".join(k for k, v in row.items() if v) for row in cond.to_dict("records")]
    df["n_conditions"] = (cond > 0).sum(axis=1).values
    for k in cond.columns:
        df[f"c_{k}"] = cond[k].values
    df["text"] = build_narratives(df, cond, category, prio, risk)
    return df, th


def main() -> None:
    setup_logging()
    s = get_settings()
    t0 = time.time()
    df = clean(load_raw())
    df, th = enrich(df)
    save_thresholds(th, s.artifacts_dir / "thresholds.json")
    df.to_parquet(s.artifacts_dir / "cases.parquet", index=False)
    log.info("Category mix:\n%s", df["category_name"].value_counts().to_string())
    log.info("Priority mix:\n%s", df["priority"].value_counts().sort_index().to_string())
    log.info("Ingestion finished in %.1fs (%d cases)", time.time() - t0, len(df))
    print(df["text"].iloc[0])


if __name__ == "__main__":
    np.random.seed(get_settings().random_seed)
    main()
