import numpy as np
import pandas as pd
import pytest

from app.retrieval import BM25Index
from utils.features import (NUMERIC_COLUMNS, classify, compute_thresholds, detect_conditions, fill_profile,
                            priority_from_risk, profile_from_conditions, risk_score)
from utils.query_parser import categories_from_conditions, parse_query


@pytest.fixture(scope="module")
def fleet():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({c: rng.normal(50, 10, n) for c in NUMERIC_COLUMNS})
    df["Signal Strength (dBm)"] = rng.uniform(-120, -60, n)
    df["Season"], df["Weather"], df["State"] = "Winter", "Clear", "TX"
    return df


def test_conditions_and_classification(fleet):
    th = compute_thresholds(fleet)
    row = fill_profile({"Latency (ms)": 200, "Packet Loss Rate (%)": 200, "Jitter (ms)": 200}, th)
    cond = detect_conditions(pd.DataFrame([row]), th)
    assert cond.loc[0, "high_latency"] == 2 and cond.loc[0, "packet_loss"] == 2
    cat, _ = classify(cond)
    assert cat.iloc[0] == "TRANSPORT_QOS"


def test_low_direction_conditions(fleet):
    th = compute_thresholds(fleet)
    row = fill_profile({"Signal Strength (dBm)": -125, "Connection Setup Success Rate (%)": 0}, th)
    cond = detect_conditions(pd.DataFrame([row]), th)
    assert cond.loc[0, "poor_signal"] == 2 and cond.loc[0, "setup_failure"] == 2
    assert classify(cond)[0].iloc[0] == "RAN_COVERAGE"


def test_normal_profile_is_normal(fleet):
    th = compute_thresholds(fleet)
    cond = detect_conditions(pd.DataFrame([fill_profile({}, th)]), th)
    assert classify(cond)[0].iloc[0] == "NORMAL"


def test_risk_monotonic_and_priority(fleet):
    th = compute_thresholds(fleet)
    rows = pd.DataFrame([fill_profile({"Fault Occurrence Rate (%)": v}, th) for v in (20, 50, 90)])
    r = risk_score(rows, detect_conditions(rows, th), th)
    assert r[0] < r[1] < r[2] and 0 <= r.min() and r.max() <= 100
    assert list(priority_from_risk(np.array([90, 75, 60, 10]))) == ["P1", "P2", "P3", "P4"]


def test_profile_from_conditions(fleet):
    th = compute_thresholds(fleet)
    p = profile_from_conditions({"high_latency": 2, "poor_signal": 1}, th)
    assert p["Latency (ms)"] >= th["conditions"]["high_latency"]["high"]
    assert p["Signal Strength (dBm)"] <= th["conditions"]["poor_signal"]["high"]


@pytest.mark.parametrize("q,expected", [
    ("high latency, packet loss and reduced throughput", {"high_latency", "packet_loss", "low_throughput"}),
    ("increasing fault occurrence and declining handover success", {"high_fault_rate", "handover_failure"}),
    ("customers facing frequent dropped calls", {"call_drops"}),
    ("weak signal and users cannot connect", {"poor_signal", "setup_failure"}),
    ("spike in authentication failures and security attacks", {"auth_failures", "security_incidents"}),
])
def test_query_parser_conditions(q, expected):
    assert set(parse_query(q).conditions) == expected


def test_query_parser_metadata_and_severity():
    p = parse_query("Severe packet loss in Texas during winter snow", known_states={"TX"})
    assert p.conditions["packet_loss"] == 2 and p.states == ["TX"]
    assert p.seasons == ["Winter"] and p.weather == ["Snow"]


def test_categories_from_conditions():
    top = next(iter(categories_from_conditions({"handover_failure": 1, "call_drops": 1})))
    assert top == "MOBILITY"


def test_bm25_ranks_matching_doc_first():
    docs = ["high latency packet loss winter", "poor signal strength summer", "handover failures call drops"]
    idx = BM25Index().fit(docs)
    s = idx.scores("call drops and handover failures")
    assert int(np.argmax(s)) == 2 and s[0] == 0
    assert idx.scores("zzzz unknown").sum() == 0
