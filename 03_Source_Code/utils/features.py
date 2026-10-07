"""Feature engineering shared by offline ingestion and online inference.

One code path (DataFrame in, DataFrame out) is used for both 100K-row batch processing and single
requests, so the API can never drift from how the index was built.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.telecom_kb import CATEGORIES, CATEGORY_NAMES, CONDITIONS, NORMAL_CATEGORY

NUMERIC_COLUMNS = [
    "Cell Availability (%)", "MTTR (hours)", "Throughput (Mbps)", "Latency (ms)", "Packet Loss Rate (%)",
    "Call Drop Rate (%)", "Handover Success Rate (%)", "Alarm Count", "Critical Alarm Count",
    "Parameter Changes", "Successful Configuration Changes (%)", "Data Usage (GB)", "User Count",
    "Signal Strength (dBm)", "Jitter (ms)", "Connection Setup Success Rate (%)", "Security Incidents",
    "Authentication Failures", "Temperature (°C)", "Humidity (%)", "Fault Occurrence Rate (%)",
]
CATEGORICAL_COLUMNS = ["Season", "Weather", "Issue Reported", "City", "State", "Zip"]

# Short snake-case aliases used in the public API (JSON-friendly).
API_ALIASES = {
    "cell_availability": "Cell Availability (%)", "mttr_hours": "MTTR (hours)",
    "throughput_mbps": "Throughput (Mbps)", "latency_ms": "Latency (ms)",
    "packet_loss_rate": "Packet Loss Rate (%)", "call_drop_rate": "Call Drop Rate (%)",
    "handover_success_rate": "Handover Success Rate (%)", "alarm_count": "Alarm Count",
    "critical_alarm_count": "Critical Alarm Count", "parameter_changes": "Parameter Changes",
    "successful_config_changes": "Successful Configuration Changes (%)", "data_usage_gb": "Data Usage (GB)",
    "user_count": "User Count", "signal_strength_dbm": "Signal Strength (dBm)", "jitter_ms": "Jitter (ms)",
    "connection_setup_success_rate": "Connection Setup Success Rate (%)",
    "security_incidents": "Security Incidents", "authentication_failures": "Authentication Failures",
    "temperature_c": "Temperature (°C)", "humidity": "Humidity (%)",
    "fault_occurrence_rate": "Fault Occurrence Rate (%)",
    "season": "Season", "weather": "Weather", "city": "City", "state": "State",
}
COLUMN_TO_ALIAS = {v: k for k, v in API_ALIASES.items()}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix encoding artefacts in the raw header (e.g. 'Temperature (�C)')."""
    ren = {c: "Temperature (°C)" for c in df.columns if c.startswith("Temperature")}
    return df.rename(columns=ren)


# ---------------------------------------------------------------------------------------------------
# Thresholds (fit on the dataset once, persisted to JSON)
# ---------------------------------------------------------------------------------------------------
def compute_thresholds(df: pd.DataFrame) -> dict:
    th: dict = {"conditions": {}, "stats": {}, "quantiles": {}}
    for c in CONDITIONS:
        s = df[c.column]
        if c.direction == "high":
            th["conditions"][c.key] = {"high": float(s.quantile(c.high_q)), "severe": float(s.quantile(c.severe_q))}
        elif c.direction == "low":
            th["conditions"][c.key] = {"high": float(s.quantile(c.high_q)), "severe": float(s.quantile(c.severe_q))}
        else:  # both tails
            th["conditions"][c.key] = {"high": float(s.quantile(c.high_q)), "severe": float(s.quantile(c.severe_q)),
                                       "low": float(s.quantile(1 - c.high_q)),
                                       "low_severe": float(s.quantile(1 - c.severe_q))}
    for col in NUMERIC_COLUMNS:
        s = df[col]
        th["stats"][col] = {"mean": float(s.mean()), "std": float(s.std() or 1.0), "median": float(s.median()),
                            "p10": float(s.quantile(.1)), "p25": float(s.quantile(.25)),
                            "p75": float(s.quantile(.75)), "p90": float(s.quantile(.9)),
                            "min": float(s.min()), "max": float(s.max())}
        th["quantiles"][col] = [float(x) for x in np.quantile(s, np.linspace(0, 1, 101))]
    th["states"] = sorted(df["State"].dropna().unique().tolist())
    return th


def save_thresholds(th: dict, path: Path) -> None:
    path.write_text(json.dumps(th, indent=1), encoding="utf-8")


def load_thresholds(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile_rank(value: float | np.ndarray, col: str, th: dict) -> np.ndarray:
    """Empirical CDF (0..1) of a value against the fleet distribution."""
    q = np.asarray(th["quantiles"][col])
    return np.interp(value, q, np.linspace(0, 1, len(q)))


# ---------------------------------------------------------------------------------------------------
# Conditions -> category -> risk -> priority
# ---------------------------------------------------------------------------------------------------
def detect_conditions(df: pd.DataFrame, th: dict) -> pd.DataFrame:
    """Return an int8 frame (0 = normal, 1 = degraded, 2 = severe) per condition key."""
    out = {}
    for c in CONDITIONS:
        if c.column not in df:
            out[c.key] = np.zeros(len(df), dtype=np.int8)
            continue
        s = df[c.column].astype(float)
        t = th["conditions"][c.key]
        if c.direction == "high":
            sev = np.where(s >= t["severe"], 2, np.where(s >= t["high"], 1, 0))
        elif c.direction == "low":
            sev = np.where(s <= t["severe"], 2, np.where(s <= t["high"], 1, 0))
        else:
            hi = np.where(s >= t["severe"], 2, np.where(s >= t["high"], 1, 0))
            lo = np.where(s <= t["low_severe"], 2, np.where(s <= t["low"], 1, 0))
            sev = np.maximum(hi, lo)
        out[c.key] = np.where(s.isna(), 0, sev).astype(np.int8)
    return pd.DataFrame(out, index=df.index)


def category_scores(cond: pd.DataFrame, weather: pd.Series | None = None) -> pd.DataFrame:
    scores = {}
    for cat, spec in CATEGORIES.items():
        total = sum(cond[k].astype(float) * w for k, w in spec["conditions"].items())
        scores[cat] = total
    sc = pd.DataFrame(scores, index=cond.index)
    if weather is not None:   # adverse weather is evidence for an environmental cause
        sc["ENVIRONMENT"] += weather.isin(["Snow", "Rain"]).astype(float) * 0.5 * (sc["ENVIRONMENT"] > 0)
    return sc


def classify(cond: pd.DataFrame, weather: pd.Series | None = None) -> tuple[pd.Series, pd.DataFrame]:
    sc = category_scores(cond, weather)
    top = sc.idxmax(axis=1)
    top = top.where(sc.max(axis=1) >= 1.0, NORMAL_CATEGORY)
    return top, sc


def risk_score(df: pd.DataFrame, cond: pd.DataFrame, th: dict, fault_rate: np.ndarray | None = None) -> np.ndarray:
    """0-100 composite risk.

    50% fault likelihood   (percentile of (predicted) fault occurrence rate)
    30% degradation breadth (how many KPIs are degraded / severe)
    20% customer impact    (user load and loss of availability)
    """
    fr = fault_rate if fault_rate is not None else df["Fault Occurrence Rate (%)"].to_numpy(float)
    fault_pct = percentile_rank(fr, "Fault Occurrence Rate (%)", th)
    breadth = np.minimum(1.0, cond.sum(axis=1).to_numpy(float) / 8.0)
    users = df["User Count"].to_numpy(float) if "User Count" in df else np.full(len(df), th["stats"]["User Count"]["median"])
    avail = (df["Cell Availability (%)"].to_numpy(float) if "Cell Availability (%)" in df
             else np.full(len(df), th["stats"]["Cell Availability (%)"]["median"]))
    impact = 0.5 * percentile_rank(users, "User Count", th) + 0.5 * (1 - percentile_rank(avail, "Cell Availability (%)", th))
    return np.round(100 * (0.5 * fault_pct + 0.3 * breadth + 0.2 * impact), 1)


def priority_from_risk(risk: np.ndarray | float, critical_alarm_sev: np.ndarray | int = 0) -> np.ndarray:
    r = np.asarray(risk, dtype=float)
    crit = np.asarray(critical_alarm_sev) >= 2
    # Calibrated on the fleet distribution: ~12% P1, ~23% P2, ~35% P3, ~30% P4 (NOC queues stay workable).
    p = np.select([(r >= 83) | (crit & (r >= 78)), r >= 70, r >= 50], ["P1", "P2", "P3"], default="P4")
    return p


SEVERITY_BY_PRIORITY = {"P1": "Critical", "P2": "Major", "P3": "Minor", "P4": "Low"}


# ---------------------------------------------------------------------------------------------------
# Narrative generation (structured row -> searchable incident text)
# ---------------------------------------------------------------------------------------------------
def _fmt(v: float, unit: str) -> str:
    if unit in ("ms", "Mbps", "GB"):
        return f"{v:.0f} {unit}"
    if unit == "dBm":
        return f"{v:.0f} dBm"
    if unit == "%":
        return f"{v:.1f}%"
    if unit == "h":
        return f"{v:.1f} h"
    if unit == "°C":
        return f"{v:.1f}°C"
    return f"{v:.0f}"


def build_narratives(df: pd.DataFrame, cond: pd.DataFrame, category: pd.Series, priority: np.ndarray,
                     risk: np.ndarray) -> list[str]:
    cond_meta = [(c.key, c.label, c.column, c.unit) for c in CONDITIONS]
    texts = []
    recs = df.to_dict("records")
    cond_vals = cond.to_numpy()
    for i, r in enumerate(recs):
        present = []
        for j, (key, label, col, unit) in enumerate(cond_meta):
            sev = cond_vals[i, j]
            if sev:
                present.append(f"{'severe ' if sev == 2 else ''}{label} ({_fmt(r[col], unit)})")
        cond_txt = "; ".join(present) if present else "all KPIs within normal fleet range"
        texts.append(
            f"{r['Season']} network incident at {r['City']}, {r['State']}. Weather {r['Weather'].lower()}, "
            f"{r['Temperature (°C)']:.0f}°C, humidity {r['Humidity (%)']:.0f}%. "
            f"Observed conditions: {cond_txt}. "
            f"Classified as {CATEGORY_NAMES[category.iloc[i]]}, priority {priority[i]}, risk {risk[i]:.0f}/100. "
            f"KPIs: availability {r['Cell Availability (%)']:.1f}%, throughput {r['Throughput (Mbps)']:.0f} Mbps, "
            f"latency {r['Latency (ms)']:.0f} ms, packet loss {r['Packet Loss Rate (%)']:.1f}%, "
            f"jitter {r['Jitter (ms)']:.0f} ms, call drop {r['Call Drop Rate (%)']:.2f}%, "
            f"handover success {r['Handover Success Rate (%)']:.1f}%, signal {r['Signal Strength (dBm)']:.0f} dBm, "
            f"setup success {r['Connection Setup Success Rate (%)']:.1f}%. "
            f"Alarms {r['Alarm Count']:.0f} (critical {r['Critical Alarm Count']:.1f}), "
            f"{r['Parameter Changes']:.0f} parameter changes ({r['Successful Configuration Changes (%)']:.1f}% successful), "
            f"{r['User Count']:.0f} users, {r['Data Usage (GB)']:.0f} GB data, "
            f"{r['Security Incidents']:.0f} security incidents, {r['Authentication Failures']:.0f} authentication failures. "
            f"Fault occurrence {r['Fault Occurrence Rate (%)']:.1f}%, MTTR {r['MTTR (hours)']:.1f} h. "
            f"Issue reported: {r['Issue Reported']}."
        )
    return texts


def condition_list(cond_row: pd.Series | dict) -> list[str]:
    return [k for k, v in dict(cond_row).items() if v]


def fill_profile(partial: dict, th: dict) -> dict:
    """Complete a partial KPI dict with fleet medians so models can score it."""
    full = {col: th["stats"][col]["median"] for col in NUMERIC_COLUMNS}
    full.update({"Season": "Fall", "Weather": "Clear", "City": "", "State": "", "Issue Reported": "yes"})
    full.update({k: v for k, v in partial.items() if v is not None})
    return full


def profile_from_conditions(conditions: dict[str, int], th: dict) -> dict:
    """Synthesise a 'typical' KPI profile for a set of NL-described conditions (median elsewhere)."""
    from utils.telecom_kb import CONDITION_BY_KEY
    prof = {}
    for key, sev in conditions.items():
        c = CONDITION_BY_KEY[key]
        st = th["stats"][c.column]
        if c.direction == "low":
            prof[c.column] = st["p10"] if sev >= 2 else (st["p10"] + st["p25"]) / 2
        else:
            prof[c.column] = st["p90"] if sev >= 2 else (st["p75"] + st["p90"]) / 2
            if c.direction == "both":
                prof[c.column] = th["conditions"][key]["high"]
    return prof
