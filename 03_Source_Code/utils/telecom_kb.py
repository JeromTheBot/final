"""Telecom domain knowledge: KPI condition definitions, incident taxonomy and the troubleshooting playbook.

The dataset has no free text, so this module is what turns KPI numbers into the language engineers use
("high latency", "handover failures", ...). Thresholds come from the data (percentiles computed during
ingestion), so "high" means "worse than 75% of the fleet" and "severe" means "worse than 90%".
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Condition:
    key: str
    column: str
    direction: str            # "high" -> bad when large, "low" -> bad when small
    label: str                # phrase used in narratives
    synonyms: tuple[str, ...] = field(default_factory=tuple)
    unit: str = ""
    high_q: float = 0.75      # quantile at which the condition is "present"
    severe_q: float = 0.90    # quantile at which it is "severe"


CONDITIONS: list[Condition] = [
    Condition("high_latency", "Latency (ms)", "high", "high latency",
              ("latency", "delay", "lag", "rtt", "ping", "slow response", "response time"), "ms"),
    Condition("packet_loss", "Packet Loss Rate (%)", "high", "packet loss",
              ("packet loss", "packets lost", "dropped packets", "plr", "losing packets"), "%"),
    Condition("low_throughput", "Throughput (Mbps)", "low", "reduced throughput",
              ("throughput", "slow speed", "low speed", "bandwidth", "data rate", "slow data", "slow internet",
               "speed"), "Mbps", 0.25, 0.10),
    Condition("call_drops", "Call Drop Rate (%)", "high", "call drops",
              ("call drop", "dropped call", "drops call", "call failure", "calls drop", "cdr", "call drops"), "%"),
    Condition("handover_failure", "Handover Success Rate (%)", "low", "handover failures",
              ("handover", "handoff", "hand-over", "mobility", "ho failure"), "%", 0.25, 0.10),
    Condition("poor_signal", "Signal Strength (dBm)", "low", "poor signal strength",
              ("signal", "coverage", "rsrp", "rssi", "weak signal", "dbm", "no signal", "dead zone"), "dBm", 0.25, 0.10),
    Condition("high_jitter", "Jitter (ms)", "high", "high jitter",
              ("jitter", "choppy", "voice quality", "stutter", "volte quality"), "ms"),
    Condition("alarm_storm", "Alarm Count", "high", "elevated alarm count",
              ("alarm", "alarms", "alarm count", "alarm storm"), ""),
    Condition("critical_alarms", "Critical Alarm Count", "high", "critical alarms",
              ("critical alarm", "major alarm", "critical alarms"), ""),
    Condition("low_availability", "Cell Availability (%)", "low", "low cell availability",
              ("availability", "outage", "cell down", "site down", "unavailable", "downtime", "not available"),
              "%", 0.25, 0.10),
    Condition("setup_failure", "Connection Setup Success Rate (%)", "low", "connection setup failures",
              ("connection setup", "setup failure", "attach failure", "cannot connect", "can't connect", "rrc",
               "access failure", "unable to connect", "connection failure"), "%", 0.25, 0.10),
    Condition("config_churn", "Parameter Changes", "high", "frequent parameter changes",
              ("parameter change", "configuration change", "config change", "reconfiguration", "parameter tuning"), ""),
    Condition("config_failure", "Successful Configuration Changes (%)", "low", "failed configuration changes",
              ("failed configuration", "config failure", "configuration failure", "rollback", "failed change"),
              "%", 0.25, 0.10),
    Condition("high_load", "User Count", "high", "high user load",
              ("congestion", "congested", "overload", "heavy traffic", "many users", "capacity", "peak hour",
               "crowded", "user load", "busy"), ""),
    Condition("high_data_usage", "Data Usage (GB)", "high", "high data usage",
              ("data usage", "traffic volume", "heavy usage", "data consumption"), "GB"),
    Condition("security_incidents", "Security Incidents", "high", "security incidents",
              ("security", "intrusion", "attack", "breach", "ddos", "malicious", "threat"), ""),
    Condition("auth_failures", "Authentication Failures", "high", "authentication failures",
              ("authentication", "auth failure", "login failure", "sim failure", "credential", "auth"), ""),
    Condition("high_fault_rate", "Fault Occurrence Rate (%)", "high", "high fault occurrence",
              ("fault", "faults", "fault occurrence", "failure rate", "fault rate", "network fault"), "%"),
    Condition("slow_repair", "MTTR (hours)", "high", "long repair time (MTTR)",
              ("mttr", "repair time", "slow restoration", "long outage", "time to repair", "slow recovery"), "h"),
    Condition("extreme_temperature", "Temperature (°C)", "both", "extreme temperature",
              ("heat", "hot", "heatwave", "cold", "freezing", "temperature", "overheating"), "°C", 0.95, 0.99),
    Condition("high_humidity", "Humidity (%)", "high", "high humidity",
              ("humidity", "moisture", "humid", "water ingress"), "%", 0.90, 0.97),
]
CONDITION_BY_KEY = {c.key: c for c in CONDITIONS}

# --- Incident taxonomy -----------------------------------------------------------------------------
CATEGORIES: dict[str, dict] = {
    "RAN_COVERAGE": {"name": "Radio Access & Coverage",
                     "conditions": {"poor_signal": 1.0, "setup_failure": 1.0, "low_availability": 0.4}},
    "CAPACITY": {"name": "Capacity & Congestion",
                 "conditions": {"high_load": 1.0, "high_data_usage": 0.7, "low_throughput": 1.0}},
    "TRANSPORT_QOS": {"name": "Transport & QoS Degradation",
                      "conditions": {"high_latency": 1.0, "packet_loss": 1.0, "high_jitter": 0.8}},
    "MOBILITY": {"name": "Mobility & Handover",
                 "conditions": {"handover_failure": 1.2, "call_drops": 1.0}},
    "EQUIPMENT_FAULT": {"name": "Equipment Fault & Alarms",
                        "conditions": {"critical_alarms": 1.0, "alarm_storm": 0.8, "high_fault_rate": 0.8,
                                       "low_availability": 0.8, "slow_repair": 0.6}},
    "CONFIGURATION": {"name": "Configuration & Change Management",
                      "conditions": {"config_churn": 1.0, "config_failure": 1.2}},
    "SECURITY": {"name": "Security & Authentication",
                 "conditions": {"security_incidents": 1.0, "auth_failures": 1.0}},
    "ENVIRONMENT": {"name": "Environmental / Weather Impact",
                    "conditions": {"extreme_temperature": 1.0, "high_humidity": 0.8}},
}
NORMAL_CATEGORY = "NORMAL"
CATEGORY_NAMES = {k: v["name"] for k, v in CATEGORIES.items()} | {NORMAL_CATEGORY: "Normal Operation"}

SEASONS = ["Winter", "Spring", "Summer", "Fall"]
WEATHER = ["Clear", "Clouds", "Rain", "Snow"]

# --- Troubleshooting playbook (RAG knowledge base #2) -------------------------------------------
# Distilled from common 3GPP / NOC operational practice. Each entry is indexed and retrieved alongside
# historical cases so that recommendations are grounded in an SOP, not invented by the LLM.
PLAYBOOK: dict[str, dict] = {
    "RAN_COVERAGE": {
        "root_causes": ["Antenna tilt/azimuth misalignment or physical damage",
                        "Coverage holes from new obstructions or neighbour cell outage",
                        "Uplink interference raising noise floor", "Low transmit power / PA degradation"],
        "actions": ["Run drive test or MDT trace to confirm RSRP/SINR footprint",
                    "Check VSWR and RET (remote electrical tilt) settings; correct tilt/azimuth",
                    "Scan for uplink interference (PIM, external jammers) and mitigate",
                    "Verify neighbour-cell availability and adjust neighbour lists",
                    "Increase reference signal power within licence limits or plan a small cell"],
        "kpis": ["Signal Strength (dBm)", "Connection Setup Success Rate (%)", "Cell Availability (%)"],
    },
    "CAPACITY": {
        "root_causes": ["Busy-hour traffic exceeding cell capacity", "Uneven load distribution between layers/bands",
                        "Insufficient backhaul capacity", "Special events or seasonal traffic surges"],
        "actions": ["Enable/tune load balancing (MLB) and inter-frequency offload",
                    "Activate additional carriers / carrier aggregation or 5G NR layer",
                    "Review PRB utilisation and scheduler parameters",
                    "Upgrade backhaul or enable traffic shaping for heavy users",
                    "Plan cell split or small-cell densification for persistent hotspots"],
        "kpis": ["User Count", "Data Usage (GB)", "Throughput (Mbps)"],
    },
    "TRANSPORT_QOS": {
        "root_causes": ["Backhaul/transport link congestion or errors", "Microwave link fading (weather)",
                        "Misconfigured QoS/QCI mapping", "Router buffer bloat or routing flaps"],
        "actions": ["Check transport link utilisation, CRC errors and interface drops",
                    "Run TWAMP/ping tests per hop to localise latency and loss",
                    "Validate QoS/DSCP marking and queue configuration end to end",
                    "Re-route traffic over redundant path; open ticket with transport team",
                    "Inspect microwave link margins and alignment if weather-correlated"],
        "kpis": ["Latency (ms)", "Packet Loss Rate (%)", "Jitter (ms)"],
    },
    "MOBILITY": {
        "root_causes": ["Missing or incorrect neighbour relations", "Handover thresholds/hysteresis mis-tuned",
                        "Coverage overlap too small or too large (ping-pong)", "Timer (T304/T310) misconfiguration"],
        "actions": ["Audit ANR neighbour lists and add missing relations",
                    "Tune A3 offset, hysteresis and time-to-trigger parameters",
                    "Analyse handover failure causes (too early / too late / wrong cell)",
                    "Check RLF counters and adjust radio link timers",
                    "Validate X2/Xn interface health between neighbour sites"],
        "kpis": ["Handover Success Rate (%)", "Call Drop Rate (%)"],
    },
    "EQUIPMENT_FAULT": {
        "root_causes": ["Hardware failure (RRU, baseband board, power supply)", "Software bug after upgrade",
                        "Power or cooling issues at site", "Fibre/CPRI link failure"],
        "actions": ["Correlate and de-duplicate alarms; identify root alarm in the alarm chain",
                    "Dispatch field engineer for hardware swap if critical alarms persist",
                    "Check site power, battery backup and cooling status",
                    "Roll back recent software if fault started after an upgrade",
                    "Open a priority trouble ticket and track MTTR against SLA"],
        "kpis": ["Critical Alarm Count", "Alarm Count", "Fault Occurrence Rate (%)", "MTTR (hours)"],
    },
    "CONFIGURATION": {
        "root_causes": ["Unvalidated parameter changes", "Conflicting SON / manual changes",
                        "Failed or partial configuration push", "Configuration drift from golden baseline"],
        "actions": ["Freeze non-urgent changes on affected cells (change freeze)",
                    "Compare running config against golden baseline; revert drifted parameters",
                    "Review change log and roll back failed changes",
                    "Coordinate SON features to avoid conflicting actions",
                    "Enforce pre/post-change KPI checks in the change process"],
        "kpis": ["Parameter Changes", "Successful Configuration Changes (%)"],
    },
    "SECURITY": {
        "root_causes": ["Brute-force or credential-stuffing attempts", "SIM cloning / rogue devices",
                        "Signalling storm or DDoS on core interfaces", "Expired certificates or key mismatch"],
        "actions": ["Escalate to SOC; review authentication logs for anomalous IMSIs",
                    "Rate-limit / blacklist offending subscribers or IPs",
                    "Check certificate validity and AUSF/HSS key synchronisation",
                    "Enable signalling firewall rules for abnormal traffic",
                    "Document incident for compliance reporting"],
        "kpis": ["Security Incidents", "Authentication Failures"],
    },
    "ENVIRONMENT": {
        "root_causes": ["Equipment overheating or freezing outside operating range",
                        "Rain fade on microwave links", "Water ingress / humidity corrosion",
                        "Snow/ice loading on antennas"],
        "actions": ["Check shelter/cabinet temperature and HVAC status",
                    "Inspect connectors and weatherproofing for moisture ingress",
                    "Increase microwave fade margin or switch to adaptive modulation",
                    "Schedule preventive maintenance before extreme-weather seasons",
                    "Use weather-correlated alerting to pre-position field crews"],
        "kpis": ["Temperature (°C)", "Humidity (%)"],
    },
    "NORMAL": {
        "root_causes": ["No KPI deviates significantly from the fleet baseline"],
        "actions": ["Continue routine monitoring", "Verify the reported symptom with a targeted test",
                    "Check for customer-side or device-specific issues"],
        "kpis": ["Cell Availability (%)", "Throughput (Mbps)"],
    },
}


def playbook_documents() -> list[dict]:
    """Flatten the playbook into retrievable SOP documents."""
    docs = []
    for cat, entry in PLAYBOOK.items():
        name = CATEGORY_NAMES[cat]
        text = (f"SOP {name}. Typical root causes: {'; '.join(entry['root_causes'])}. "
                f"Corrective actions: {'; '.join(entry['actions'])}. KPIs to monitor: {', '.join(entry['kpis'])}.")
        docs.append({"category": cat, "name": name, "text": text, **entry})
    return docs
