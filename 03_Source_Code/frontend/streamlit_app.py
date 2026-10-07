"""Streamlit front-end for the Telecom NetOps Assistant (talks to the FastAPI microservice).

    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.getenv("NETOPS_API_URL", "http://localhost:8000")
API_KEY = os.getenv("NETOPS_API_KEY", "")
ROOT = Path(__file__).resolve().parent.parent
SAMPLES = [
    "A cell is experiencing high latency, packet loss, and reduced throughput. What similar network conditions have occurred before?",
    "Multiple cells are experiencing poor service quality and increasing alarm counts. What factors could be contributing to the problem?",
    "A network region has increasing fault occurrence and declining handover success. What corrective actions should be considered?",
    "Customers are facing frequent dropped calls in Texas during winter",
    "Spike in authentication failures and security incidents on the core network",
]
PRIO_COLORS = {"P1": "#d62728", "P2": "#ff7f0e", "P3": "#e1b000", "P4": "#2ca02c"}

st.set_page_config(page_title="NetOps Assistant", page_icon="📡", layout="wide")


def call(method: str, path: str, payload: dict | None = None, timeout: int = 120):
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        r = requests.request(method, f"{API}{path}", json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        st.error(f"API not reachable at {API}: {exc}")
        return None
    if r.status_code == 422:
        body = r.json()
        if body.get("error") == "guardrail_violation":
            st.warning(f"🛡️ Guardrail ({body['code']}): {body['message']}")
        else:
            st.warning(f"Invalid input: {json.dumps(body.get('detail', body))[:600]}")
        return None
    if r.status_code >= 400:
        st.error(f"API error {r.status_code}: {r.text[:400]}")
        return None
    return r.json()


@st.cache_data(ttl=300)
def metadata():
    try:
        return requests.get(f"{API}/metadata", timeout=30).json()
    except Exception:
        return None


def prio_badge(p: str) -> str:
    return f"<span style='background:{PRIO_COLORS.get(p, '#888')};color:white;padding:2px 10px;border-radius:10px;" \
           f"font-weight:600'>{p}</span>"


md = metadata()
st.title("📡 AI-Powered Telecom Network Operations Assistant")
if md is None:
    st.error(f"Cannot reach the API at {API}. Start it with `uvicorn app.main:app --port 8000`.")
    st.stop()
st.caption(f"{md['n_cases']:,} historical 5G incidents indexed · embeddings: {md['index']['embedding_model']} · "
           f"LLM: {md['llm']['provider']} {'' if md['llm']['available'] else '(offline deterministic mode)'}")

tab_ask, tab_prio, tab_pred, tab_mon, tab_eval = st.tabs(
    ["🔎 Ask the assistant", "📋 Incident prioritisation", "📈 Risk & performance predictor",
     "🖥️ Monitoring & feedback", "🧪 Evaluation"])

# ------------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Search settings")
    pipeline = st.radio("Pipeline", ["Full multi-agent analysis", "Basic recommendation", "Similarity search only"])
    mode = st.selectbox("Retrieval mode", ["hybrid", "semantic", "keyword"])
    alpha = st.slider("Semantic weight (alpha)", 0.0, 1.0, 0.6, 0.05, disabled=mode != "hybrid")
    top_k = st.slider("Similar cases (top-k)", 1, 15, 5)
    use_llm = st.toggle("Use LLM (if configured)", value=True)
    auto_filters = st.toggle("Auto-detect location/season filters", value=True)
    st.header("Metadata filters")
    f_state = st.multiselect("State", md["states"])
    f_city = st.text_input("City (exact)", "")
    f_season = st.multiselect("Season", md["seasons"])
    f_weather = st.multiselect("Weather", md["weather"])
    f_cat = st.multiselect("Category", list(md["categories"]), format_func=lambda k: md["categories"][k])
    f_prio = st.multiselect("Priority", md["priorities"])
    f_conds = st.multiselect("Must show conditions", list(md["conditions"]), format_func=lambda k: md["conditions"][k])
    f_risk = st.slider("Minimum risk score", 0, 100, 0)


def current_filters() -> dict:
    f = {"state": f_state or None, "city": [f_city] if f_city.strip() else None, "season": f_season or None,
         "weather": f_weather or None, "category": f_cat or None, "priority": f_prio or None,
         "required_conditions": f_conds or None, "min_risk": f_risk or None}
    return {k: v for k, v in f.items() if v}


# ------------------------------------------------------------------------------------------------- ask
with tab_ask:
    c1, c2 = st.columns([3, 1])
    with c2:
        sample = st.selectbox("Sample questions", ["—"] + SAMPLES, index=0)
    with c1:
        query = st.text_area("Describe the network problem", value="" if sample == "—" else sample, height=90,
                             placeholder="e.g. Several cells in TX show high latency and packet loss since the storm")
    with st.expander("Optional: live KPI readings for the affected cell"):
        k1, k2, k3, k4 = st.columns(4)
        kpis = {}
        if st.checkbox("Include KPIs"):
            kpis = {"latency_ms": k1.number_input("Latency (ms)", 0.0, 500.0, 120.0),
                    "packet_loss_rate": k2.number_input("Packet loss (%)", 0.0, 100.0, 10.0),
                    "throughput_mbps": k3.number_input("Throughput (Mbps)", 0.0, 5000.0, 300.0),
                    "signal_strength_dbm": k4.number_input("Signal (dBm)", -150.0, -20.0, -95.0),
                    "handover_success_rate": k1.number_input("Handover success (%)", 0.0, 100.0, 95.0),
                    "call_drop_rate": k2.number_input("Call drop (%)", 0.0, 100.0, 1.0),
                    "critical_alarm_count": k3.number_input("Critical alarms", 0.0, 100.0, 6.0),
                    "user_count": k4.number_input("Users", 0.0, 100000.0, 5000.0)}
    run = st.button("Analyse", type="primary", use_container_width=True)
    if run:
        payload = {"query": query, "filters": current_filters() or None, "top_k": top_k, "alpha": alpha,
                   "mode": mode, "use_llm": use_llm, "auto_filters": auto_filters, "kpis": kpis or None}
        path = {"Full multi-agent analysis": "/analyze", "Basic recommendation": "/recommend",
                "Similarity search only": "/search"}[pipeline]
        with st.spinner("Agents at work: retrieving cases, classifying, predicting, reasoning..."):
            st.session_state["result"] = call("POST", path, payload, timeout=300)
            st.session_state["fb_sent"] = False

    d = st.session_state.get("result")
    if d:
        for w in d.get("warnings", []):
            st.info(w)
        if d.get("guardrails", {}).get("redactions"):
            st.warning(f"🛡️ Personal data redacted: {d['guardrails']['redactions']}")
        und = d.get("understanding", {})
        if und.get("conditions"):
            st.markdown("**Understood symptoms:** " + " · ".join(
                f"`{v['label']}`{' (severe)' if v['severity'] == 'severe' else ''}" for v in und["conditions"].values())
                + (f" &nbsp; **Auto filters:** {und['auto_filters_applied']}" if und.get("auto_filters_applied") else ""))

        if "classification" in d:
            cls, pz = d["classification"], d.get("prioritization", {})
            m = st.columns(5)
            m[0].metric("Incident category", cls["category_name"], f"confidence {cls['confidence']:.0%}",
                        delta_color="off")
            m[1].markdown(f"**Priority**<br>{prio_badge(pz.get('priority', '?'))} {pz.get('severity', '')}",
                          unsafe_allow_html=True)
            m[2].metric("Risk score", f"{pz.get('risk_score', 0):.0f}/100")
            if "prediction" in d:
                m[3].metric("Predicted fault occurrence", f"{d['prediction']['fault_rate_pct']:.1f}%",
                            f"early-warning {d['prediction']['early_warning_fault_rate_pct']:.1f}%", delta_color="off")
            if "judge" in d:
                j = d["judge"]
                m[4].metric("Judge validation", f"{j['overall']:.1f}/5", j["verdict"].upper(),
                            delta_color="normal" if j["verdict"] == "pass" else "inverse")

        if "recommendation" in d:
            rec = d["recommendation"]
            st.subheader("🛠️ Recommended troubleshooting plan")
            st.success(rec["summary"])
            st.markdown(f"**Diagnosis.** {rec['diagnosis']}")
            acts = pd.DataFrame([{"#": i + 1, "Phase": a.get("phase", ""), "Action": a["step"],
                                  "Why": a.get("rationale", ""), "Evidence": ", ".join(a.get("evidence", [])),
                                  "Needs approval": "⚠️ yes" if a.get("requires_approval") else ""}
                                 for i, a in enumerate(rec["actions"])])
            st.dataframe(acts, hide_index=True, use_container_width=True)
            cA, cB = st.columns(2)
            cA.markdown("**Monitor:**\n" + "\n".join(f"- {x}" for x in rec.get("monitoring", [])))
            cB.markdown(f"**Escalation:** {rec.get('escalation', '')}")
            if rec.get("human_approval_required"):
                cB.warning("Service-affecting steps require change approval before execution.")
            st.caption(f"Generated by: {rec.get('method')} · SOPs: {', '.join(rec.get('sops_used', []))}")

        if "root_cause" in d:
            st.subheader("🔬 Root-cause & network-impact analysis")
            rc, imp = d["root_cause"], d["impact"]
            st.markdown(rc["summary"])
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown("**Likely root causes**")
                for c in rc["likely_root_causes"]:
                    st.markdown(f"- **{c.get('likelihood', '').upper()}** · {c['cause']}")
            with r2:
                st.markdown("**KPI deviation from fleet (z-score)**")
                if rc["kpi_deviations"]:
                    st.bar_chart(pd.DataFrame(rc["kpi_deviations"]).set_index("kpi")["badness"], horizontal=True)
                st.markdown("**Fault-model drivers (pp)**")
                if rc["model_drivers"]:
                    st.bar_chart(pd.DataFrame(rc["model_drivers"]).set_index("feature")["contribution"],
                                 horizontal=True)
            with r3:
                st.markdown("**Impact**")
                st.metric("Customer impact", imp["customer_impact_level"])
                st.metric("Users affected (est.)", f"{imp['users_affected_estimate']:,}")
                st.metric("SLA breach risk", imp["sla_breach_risk"])
                st.markdown(f"Expected MTTR **{imp['expected_mttr_hours']} h** · availability gap "
                            f"**{imp['availability_gap_pp']} pp** · services: {', '.join(imp['services_affected'])}")
                pp = imp["pattern_prevalence"]
                st.caption(f"Pattern prevalence: {pp['same_category_cases_in_scope']:,} of {pp['scope_size']:,} "
                           f"cases in scope ({pp['share']:.1%})")
            pf = d["prediction"]["performance_forecast"]
            st.markdown("**Network performance forecast** (operating-condition model)")
            st.dataframe(pd.DataFrame(pf).T, use_container_width=True)

        if d.get("similar_cases"):
            st.subheader(f"📚 Similar historical cases ({len(d['similar_cases'])})")
            st.dataframe(pd.DataFrame([{"Case": h["case_id"], "Score": h["score"], "Category": h["category_name"],
                                        "Priority": h["priority"], "Risk": h["risk_score"],
                                        "Where": f"{h['location']['city']}, {h['location']['state']}",
                                        "Season": h["season"], "Weather": h["weather"],
                                        "Conditions": ", ".join(h["conditions"])} for h in d["similar_cases"]]),
                         hide_index=True, use_container_width=True)
            for h in d["similar_cases"]:
                with st.expander(f"{h['case_id']} · {h['category_name']} · {h['priority']} · score {h['score']}"):
                    st.write(h["text"])
                    st.json(h["scores"])

        if "judge" in d:
            with st.expander("⚖️ LLM-as-judge validation details"):
                st.json(d["judge"])
        if d.get("agent_trace"):
            with st.expander("🤖 Multi-agent trace (transparency)"):
                st.dataframe(pd.DataFrame([{"Agent": t["agent"], "Role": t["role"], "Status": t["status"],
                                            "LLM": "✓" if t["used_llm"] else "", "ms": t["latency_ms"],
                                            "Notes": json.dumps(t["notes"])[:300]} for t in d["agent_trace"]]),
                             hide_index=True, use_container_width=True)

        if d.get("response_id"):
            st.subheader("👍 Was this helpful?")
            with st.form("feedback"):
                f1, f2 = st.columns(2)
                rating = f1.slider("Rating", 1, 5, 4)
                correct = f2.selectbox("Correct category (if wrong)", ["—"] + list(md["categories"]),
                                       format_func=lambda k: md["categories"].get(k, k))
                eff = st.multiselect("Actions that worked", [a["step"] for a in d["recommendation"]["actions"]])
                comment = st.text_input("Comment")
                if st.form_submit_button("Send feedback") and not st.session_state.get("fb_sent"):
                    ok = call("POST", "/feedback", {"response_id": d["response_id"], "rating": rating,
                                                    "correct_category": None if correct == "—" else correct,
                                                    "effective_actions": eff or None, "comment": comment})
                    if ok:
                        st.session_state["fb_sent"] = True
                        st.success("Thanks! Feedback will re-rank cases and actions for future queries.")

# ------------------------------------------------------------------------------------------------- prioritise
with tab_prio:
    st.markdown("Rank a batch of open incidents by **priority, risk score and user impact**. Pull candidate cases "
                "with a search (uses sidebar filters) or upload a CSV of live KPIs.")
    pq = st.text_input("Find candidate incidents", "cells with critical alarms, call drops or high latency")
    n = st.slider("How many incidents", 5, 30, 12)
    up = st.file_uploader("…or upload CSV (columns: id + any KPI names such as latency_ms, packet_loss_rate)", type="csv")
    if st.button("Prioritise", type="primary"):
        if up is not None:
            df = pd.read_csv(up)
            items = []
            for _, r in df.iterrows():
                k = {c: float(r[c]) for c in df.columns if c != "id" and pd.notna(r[c])}
                items.append({"id": str(r.get("id", len(items))), "kpis": k})
        else:
            res = call("POST", "/search", {"query": pq, "top_k": min(n, 20), "filters": current_filters() or None,
                                           "use_llm": False})
            items = [{"id": h["case_id"], "case_id": h["case_id"]} for h in (res or {}).get("similar_cases", [])]
        if items:
            out = call("POST", "/prioritize", {"incidents": items})
            if out:
                s = out["summary"]
                c = st.columns(4)
                for i, p in enumerate(["P1", "P2", "P3", "P4"]):
                    c[i].metric(p, s.get(p, 0))
                df = pd.DataFrame(out["ranked"])
                st.dataframe(df[["rank", "id", "priority", "severity", "risk_score", "predicted_fault_rate_pct",
                                 "high_risk_probability", "category", "users", "conditions"]],
                             hide_index=True, use_container_width=True)
                for e in out["errors"]:
                    st.warning(e)

# ------------------------------------------------------------------------------------------------- predictor
with tab_pred:
    st.markdown("What-if analysis: adjust KPIs and see **fault risk** and **performance** forecasts with drivers.")
    a, b, c = st.columns(3)
    k = {"critical_alarm_count": a.slider("Critical alarm count", 0.0, 14.0, 6.0, 0.5),
         "alarm_count": a.slider("Alarm count", 0, 26, 10),
         "authentication_failures": a.slider("Authentication failures", 0, 63, 8),
         "call_drop_rate": a.slider("Call drop rate (%)", 0.0, 7.0, 0.5, 0.1),
         "latency_ms": b.slider("Latency (ms)", 5, 175, 77),
         "packet_loss_rate": b.slider("Packet loss (%)", 0.1, 17.0, 6.2, 0.1),
         "jitter_ms": b.slider("Jitter (ms)", 0, 50, 25),
         "signal_strength_dbm": b.slider("Signal strength (dBm)", -120, -60, -90),
         "user_count": c.slider("User count", 100, 9999, 5000),
         "temperature_c": c.slider("Temperature (°C)", -10, 40, 17),
         "humidity": c.slider("Humidity (%)", 20, 90, 55),
         "season": c.selectbox("Season", md["seasons"], index=2),
         "weather": c.selectbox("Weather", md["weather"], index=0)}
    risk = call("POST", "/predict/risk", {"kpis": k})
    perf = call("POST", "/predict/performance", {"kpis": k})
    if risk and perf:
        m = st.columns(5)
        m[0].metric("Predicted fault occurrence", f"{risk['fault_rate_pct']:.1f}%",
                    f"P{risk['fault_rate_percentile']:.0f} of fleet", delta_color="off")
        m[1].metric("Early-warning model", f"{risk['early_warning_fault_rate_pct']:.1f}%")
        m[2].metric("High-risk probability", f"{risk['high_risk_probability']:.0%}")
        m[3].metric("Risk score", f"{risk['risk_score']:.0f}/100")
        m[4].markdown(f"**Priority**<br>{prio_badge(risk['priority'])} {risk['classification']['category_name']}",
                      unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            st.markdown("**Fault-risk drivers** (pp vs fleet median)")
            if risk.get("drivers"):
                st.bar_chart(pd.DataFrame(risk["drivers"]).set_index("feature")["contribution"], horizontal=True)
        with r:
            st.markdown("**Performance forecast**")
            st.dataframe(pd.DataFrame({k_: {x: v[x] for x in ("predicted", "fleet_median", "model_r2",
                                                                "interval_approx")}
                                       for k_, v in perf["forecast"].items()}).T, use_container_width=True)
            st.caption(perf["note"])

# ------------------------------------------------------------------------------------------------- monitoring
with tab_mon:
    if st.button("Refresh"):
        st.rerun()
    mt = call("GET", "/metrics")
    fb = call("GET", "/feedback/stats")
    if mt:
        for a_ in mt["alerts"]:
            (st.error if a_["level"] == "critical" else st.warning if a_["level"] == "warning" else st.info)(
                f"🔔 {a_['msg']}")
        if not mt["alerts"]:
            st.success("No active alerts")
        c = st.columns(5)
        c[0].metric("Requests", mt["total_requests"])
        c[1].metric("Errors (5xx)", mt["total_errors"])
        c[2].metric("Uptime", f"{mt['uptime_s'] // 60} min")
        c[3].metric("LLM calls", mt["llm"]["calls"], f"{mt['llm']['failures']} failures", delta_color="inverse")
        c[4].metric("Guardrail blocks", mt["events"].get("guardrail_blocked", 0))
        if mt["endpoints"]:
            st.markdown("**Endpoint latency**")
            st.dataframe(pd.DataFrame(mt["endpoints"]).T, use_container_width=True)
        st.markdown(f"**Events:** {mt['events']} · **Priorities analysed (last hour):** {mt['priorities_last_hour']}")
    if fb:
        st.subheader("Feedback loop")
        c = st.columns(4)
        c[0].metric("Responses logged", fb["responses_logged"])
        c[1].metric("Feedback received", fb["feedback_count"])
        c[2].metric("Average rating", fb["avg_rating"] or "–")
        c[3].metric("Cases re-weighted", fb["cases_with_feedback"])
        if fb["by_category"]:
            st.dataframe(pd.DataFrame(fb["by_category"]), hide_index=True)
        if fb["category_corrections"]:
            st.markdown("**Category corrections (new labelled examples)**")
            st.dataframe(pd.DataFrame(fb["category_corrections"]), hide_index=True)
    st.subheader("Model quality")
    st.json(md["model_metrics"], expanded=False)

# ------------------------------------------------------------------------------------------------- evaluation
with tab_eval:
    res_dir = ROOT / "evaluation" / "results"
    rec_f, ret_f = res_dir / "recommendation_eval.json", res_dir / "retrieval_eval.json"
    if rec_f.exists():
        r = json.loads(rec_f.read_text())
        st.subheader("Recommendation quality (golden set)")
        st.json(r["summary"], expanded=False)
        st.dataframe(pd.DataFrame(r["rows"]).drop(columns=["query"]), hide_index=True, use_container_width=True)
    if ret_f.exists():
        r = json.loads(ret_f.read_text())
        st.subheader("Retrieval strategies")
        st.dataframe(pd.DataFrame(r["strategies"]).T, use_container_width=True)
        if r.get("embedding_models"):
            st.subheader("Embedding model comparison")
            st.dataframe(pd.DataFrame({m: {"dim": v["dim"], "docs/s": v["encode_docs_per_s"],
                                           **{f"dense {k}": x for k, x in v["dense_only"].items()},
                                           **{f"hybrid {k}": x for k, x in v["hybrid+rerank"].items()}}
                                       for m, v in r["embedding_models"].items()}).T, use_container_width=True)
    if not rec_f.exists() and not ret_f.exists():
        st.info("Run `python -m evaluation.retrieval_eval` and `python -m evaluation.deepeval_eval` first.")
