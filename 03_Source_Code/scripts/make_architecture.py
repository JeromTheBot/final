"""Render the high-level architecture diagram -> 01_Architecture.pdf / .jpeg (submission root)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2]
NAVY, TEAL, GREEN, PURPLE, ORANGE, GREY, RED = "#16325c", "#0f7c8c", "#3d8b37", "#6a3d9a", "#d9771e", "#5b6770", "#b03a2e"


def box(ax, x, y, w, h, title, lines, color, fs=8.6, title_fs=10.2):
    tb = 0.034
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.008",
                                fc="white", ec=color, lw=1.8))
    ax.add_patch(FancyBboxPatch((x, y + h - tb), w, tb, boxstyle="round,pad=0.003,rounding_size=0.008",
                                fc=color, ec=color, lw=1.8))
    ax.text(x + w / 2, y + h - tb / 2, title, ha="center", va="center", color="white", fontsize=title_fs,
            fontweight="bold")
    step = min(0.025, (h - tb - 0.012) / max(len(lines), 1))
    for i, l in enumerate(lines):
        ax.text(x + 0.008, y + h - tb - 0.008 - i * step, l, ha="left", va="top", fontsize=fs, color="#222")


def arrow(ax, a, b, color=GREY, style="-|>", ls="-", rad=0.0, lw=1.6, text=None, tpos=0.5, toff=(0, 0.012)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(a[0] + (b[0] - a[0]) * tpos + toff[0], a[1] + (b[1] - a[1]) * tpos + toff[1], text, fontsize=7.6,
                color=color, ha="center", style="italic")


def band(ax, y, h, label, color):
    ax.add_patch(FancyBboxPatch((0.01, y), 0.98, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc=color, ec="none", alpha=0.07))
    ax.text(0.018, y + h - 0.012, label, fontsize=10.5, fontweight="bold", color=color, va="top")


def main():
    fig = plt.figure(figsize=(22, 13.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.975, "AI-Powered Telecom Network Operations Assistant — High-Level Architecture",
            ha="center", fontsize=19, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.948, "From the 5G NetOps dataset (100K records) to a validated, grounded troubleshooting answer",
            ha="center", fontsize=11.5, color=GREY)

    # ---------------- offline pipeline ----------------
    band(ax, 0.655, 0.275, "① OFFLINE DATA & INDEXING PIPELINE  (python -m data.ingest / data.build_index / data.train_models)", TEAL)
    y, h = 0.672, 0.225
    box(ax, 0.02, y, 0.135, h, "Data Source", ["Hugging Face", "fenar/5g_netops", "100K rows x 27 cols (CSV.xz)",
                                               "KPIs, alarms, config,", "security, weather,", "city / state / season"], NAVY)
    box(ax, 0.17, y, 0.15, h, "Ingestion & Cleaning", ["dedupe, type coercion", "median imputation", "physical-bound clipping",
                                                       "header encoding fix", "categorical normalisation",
                                                       "case_id = NOC-######"], TEAL)
    box(ax, 0.335, y, 0.19, h, "Feature Engineering", ["fleet percentile thresholds (P75/P90)", "21 KPI conditions (degraded/severe)",
                                                       "8-class incident taxonomy", "risk score 0-100 -> P1..P4",
                                                       "row -> incident narrative text", "(record = 1 chunk)"], TEAL)
    box(ax, 0.54, y, 0.215, h, "Indexing (Knowledge Layer)", ["Dense: all-MiniLM-L6-v2 (384-d)", "  -> FAISS IndexFlatIP (cosine)",
                                                              "Sparse: BM25 (uni+bi-grams)", "  on keyword field (scipy CSC)",
                                                              "SOP playbook (9 docs) embedded", "metadata columns for filters"], GREEN)
    box(ax, 0.77, y, 0.21, h, "ML Training", ["Fault-risk: HistGradientBoosting", "  regressor + high-risk classifier",
                                              "  + early-warning variant", "Performance: throughput, latency,",
                                              "  call-drop regressors", "metrics -> model_metrics.json"], PURPLE)
    for x0, x1 in ((0.155, 0.17), (0.32, 0.335), (0.525, 0.54)):
        arrow(ax, (x0, y + h / 2), (x1, y + h / 2), TEAL, lw=2)

    # artifacts strip
    ax.add_patch(FancyBboxPatch((0.17, 0.612), 0.81, 0.032, boxstyle="round,pad=0.004", fc="#eef3f7", ec=GREY, lw=1))
    ax.text(0.575, 0.628, "Artifact store (data/artifacts):  cases_indexed.parquet · incidents.faiss · bm25.joblib · "
            "playbook_emb.npy · thresholds.json · models.joblib · model_metrics.json · feedback.db",
            ha="center", va="center", fontsize=9, color=NAVY)
    for x in (0.43, 0.65, 0.875):
        arrow(ax, (x, y), (x, 0.645), GREY, lw=1.2)

    # ---------------- online path ----------------
    band(ax, 0.17, 0.43, "② ONLINE QUERY PATH  (FastAPI microservice + multi-agent orchestrator)", NAVY)
    box(ax, 0.02, 0.43, 0.115, 0.13, "NOC Engineer", ["natural-language", "question and/or", "live KPI readings"], GREY)
    box(ax, 0.02, 0.24, 0.115, 0.15, "Streamlit UI", ["Ask / Prioritise", "Predictor (what-if)", "Monitoring", "Evaluation",
                                                      "Feedback form"], ORANGE)
    box(ax, 0.155, 0.24, 0.13, 0.32, "FastAPI Layer", ["/search  /classify", "/recommend  /prioritize", "/analyze",
                                                        "/predict/risk", "/predict/performance", "/feedback  /metrics",
                                                        "Pydantic schemas", "X-API-Key (optional)", "request-id + timing"], NAVY)
    box(ax, 0.30, 0.24, 0.12, 0.32, "Input Guardrails", ["length / charset", "prompt-injection", "  patterns -> 422", "PII redaction",
                                                         "  (email, phone,", "  IMSI, card)", "domain relevance", "filter validation"], RED)
    arrow(ax, (0.077, 0.43), (0.077, 0.39), GREY)
    arrow(ax, (0.135, 0.315), (0.155, 0.315), ORANGE, text="REST/JSON", toff=(0, 0.01))
    arrow(ax, (0.285, 0.40), (0.30, 0.40), NAVY, lw=2)

    # agents
    ax.add_patch(FancyBboxPatch((0.435, 0.19), 0.545, 0.39, boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc="#f7f4fb", ec=PURPLE, lw=1.6, ls="--"))
    ax.text(0.707, 0.565, "Multi-Agent Orchestrator (supervisor + shared blackboard context, full trace)",
            ha="center", fontsize=10.5, fontweight="bold", color=PURPLE)
    aw, ah = 0.12, 0.145
    agents = [
        (0.445, 0.395, "1 Query Understanding", ["lexicon symptoms", "severity, state/city/", "season/weather", "(+LLM refine)"]),
        (0.578, 0.395, "2 Retrieval (RAG)", ["BM25 + dense FAISS", "RRF fusion (alpha)", "metadata filters", "symptom re-rank",
                                             ]),
        (0.711, 0.395, "3 Classification", ["rules + k-NN vote", "of similar cases", "confidence", "(+LLM tie-break)"]),
        (0.844, 0.395, "4 Risk Prediction", ["case-based KPI profile", "fault-risk + perf. ML", "risk score -> P1..P4"]),
        (0.844, 0.205, "5 Root Cause & Impact", ["KPI z-score deviations", "ML driver ablation", "case prevalence", "users / SLA / MTTR"]),
        (0.645, 0.205, "6 Recommendation", ["SOP RAG + cases", "grounded LLM / SOP", "feedback-ranked", "output guardrails"]),
        (0.445, 0.205, "7 LLM-as-Judge", ["groundedness, correct-", "ness, actionability,", "relevance, safety", "pass / revise"]),
    ]
    for x, yy, t, ls in agents:
        box(ax, x, yy, aw if "Judge" not in t and "Recommendation" not in t else 0.15, ah, t, ls, PURPLE, fs=8.1, title_fs=9.2)
    arrow(ax, (0.42, 0.47), (0.445, 0.47), RED, lw=2)
    for x0, x1 in ((0.565, 0.578), (0.698, 0.711), (0.831, 0.844)):
        arrow(ax, (x0, 0.468), (x1, 0.468), PURPLE, lw=1.8)
    arrow(ax, (0.904, 0.395), (0.904, 0.35), PURPLE, lw=1.8)
    arrow(ax, (0.844, 0.278), (0.795, 0.278), PURPLE, lw=1.8)
    arrow(ax, (0.645, 0.278), (0.595, 0.278), PURPLE, lw=1.8)
    arrow(ax, (0.55, 0.205), (0.72, 0.205), RED, rad=0.25, ls="--")
    ax.text(0.635, 0.183, "revise (max 1) with judge issues", fontsize=8, color=RED, ha="center", style="italic")
    arrow(ax, (0.445, 0.215), (0.42, 0.215), GREEN, lw=2.2)
    ax.text(0.36, 0.205, "validated JSON response -> UI", fontsize=8, color=GREEN, ha="center", style="italic")

    # connections to artifacts
    arrow(ax, (0.638, 0.612), (0.638, 0.54), GREEN, ls="--", text="indexes", tpos=0.5, toff=(0.025, 0))
    arrow(ax, (0.905, 0.612), (0.905, 0.54), PURPLE, ls="--", text="models", tpos=0.5, toff=(0.025, 0))

    # ---------------- cross-cutting ----------------
    band(ax, 0.015, 0.125, "③ CROSS-CUTTING SERVICES", GREEN)
    box(ax, 0.02, 0.022, 0.225, 0.095, "LLM Provider (optional, pluggable)", ["OpenAI gpt-4o-mini · Azure OpenAI",
                                                                              "Gemini 2.5 Flash · Claude Haiku 4.5",
                                                                              "no key -> deterministic offline mode"], ORANGE, fs=8.3)
    box(ax, 0.26, 0.022, 0.225, 0.095, "Feedback Loop (SQLite)", ["ratings, helpful cases, effective actions",
                                                                 "-> case re-rank boost, action ordering",
                                                                 "-> category corrections = new labels"], GREEN, fs=8.3)
    box(ax, 0.50, 0.022, 0.225, 0.095, "Monitoring & Observability", ["structured logs + request IDs",
                                                                     "p50/p95 latency, errors, events",
                                                                     "alerts: latency, error rate, P1 burst"], NAVY, fs=8.3)
    box(ax, 0.74, 0.022, 0.24, 0.095, "Evaluation", ["DeepEval: relevancy, faithfulness, GEval",
                                                    "retrieval: P@5, nDCG@10, MRR; models",
                                                    "pytest: 60+ unit / API / agent tests"], RED, fs=8.3)
    ax.text(0.5, 0.147, "LLM: used by agents 1, 3, 5, 6, 7 when configured  ·  Feedback: re-ranks agent 2 and "
            "orders actions in agent 6  ·  Monitoring: wraps every API call", ha="center", fontsize=9, color=GREY,
            style="italic")
    fig.savefig(OUT / "01_Architecture.pdf")
    fig.savefig(OUT / "01_Architecture.jpeg", dpi=110, pil_kwargs={"quality": 92})
    print("written", OUT / "01_Architecture.pdf")


if __name__ == "__main__":
    main()
