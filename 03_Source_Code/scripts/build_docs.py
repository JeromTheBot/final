"""Build the submission documents.

    python -m scripts.build_docs

* fills <!-- markers --> in docs/*.md with live results (evaluation JSON, model metrics, API examples, dataset stats)
* writes resolved Markdown to docs/build/ and PDFs to the submission root:
  02_Design_Document.pdf, 05_API_Docs.pdf, 06_Dataset_Details.pdf (+ 06_Dataset_Sample.csv)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

SRC = Path(__file__).resolve().parent.parent
SUB = SRC.parent
DOCS, ART, RES = SRC / "docs", SRC / "data" / "artifacts", SRC / "evaluation" / "results"
NAVY = colors.HexColor("#16325c")


# ------------------------------------------------------------------------------------------------- helpers
def md_table(rows: list[dict], cols: list[str] | None = None, fmt=lambda v: v) -> str:
    if not rows:
        return "_no data_"
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(fmt(r.get(c, ""))) for c in cols) + " |")
    return "\n".join(out)


def r3(v):
    return f"{v:.3f}" if isinstance(v, float) else v


def trim(obj, depth=0, max_list=2, max_str=260):
    if isinstance(obj, dict):
        return {k: trim(v, depth + 1, max_list, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        t = [trim(v, depth + 1, max_list, max_str) for v in obj[:max_list]]
        if len(obj) > max_list:
            t.append(f"... {len(obj) - max_list} more")
        return t
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "..."
    return obj


# ------------------------------------------------------------------------------------------------- fillers
def eval_sections() -> dict[str, str]:
    out = {}
    ret = json.loads((RES / "retrieval_eval.json").read_text()) if (RES / "retrieval_eval.json").exists() else None
    if ret:
        rows = [{"strategy": k, **v} for k, v in ret["strategies"].items()]
        out["EVAL_RETRIEVAL"] = ("**Retrieval strategy results** (live 100K index, 40 paraphrased labelled queries):\n\n"
                                 + md_table(rows, ["strategy", "p@5", "ndcg@10", "mrr@10", "symptom_recall@10",
                                                   "latency_ms_p50"], r3))
        if ret.get("embedding_models"):
            rows = [{"model": m.split("/")[-1], "dim": v["dim"], "docs/s": v["encode_docs_per_s"],
                     "dense nDCG@10": v["dense_only"]["ndcg@10"], "dense P@5": v["dense_only"]["p@5"],
                     "dense MRR": v["dense_only"]["mrr@10"], "hybrid nDCG@10": v["hybrid+rerank"]["ndcg@10"],
                     "hybrid P@5": v["hybrid+rerank"]["p@5"]} for m, v in ret["embedding_models"].items()]
            out["EVAL_EMBEDDINGS"] = md_table(rows, fmt=r3)
    rec = json.loads((RES / "recommendation_eval.json").read_text()) if (RES / "recommendation_eval.json").exists() else None
    if rec:
        s = rec["summary"]
        keys = [k for k in s if isinstance(s[k], (int, float)) and not isinstance(s[k], bool)]
        out["EVAL_RECO"] = ("**Recommendation evaluation results** (15 golden queries, mode: "
                            f"{'LLM + DeepEval' if s.get('deepeval') else 'deterministic/offline'}, "
                            f"{s.get('n_cases_indexed')} cases indexed):\n\n"
                            + md_table([{"metric": k, "value": s[k]} for k in keys], fmt=r3) + "\n\n"
                            + md_table(rec["rows"], ["id", "category", "category_correct", "condition_f1",
                                                     "sop_coverage", "action_keyword_recall", "judge_overall",
                                                     "latency_ms"], r3))
    if (ART / "model_metrics.json").exists():
        m = json.loads((ART / "model_metrics.json").read_text())
        f = m["fault_risk"]
        rows = [{"model": "Fault risk (full)", "R2": f["r2"], "MAE": f["mae"], "baseline MAE": f["baseline_mae_mean_predictor"]},
                {"model": "Fault risk (early warning)", "R2": f["early_warning_r2"], "MAE": f["early_warning_mae"],
                 "baseline MAE": f["baseline_mae_mean_predictor"]}]
        rows += [{"model": f"Performance: {t}", "R2": v["r2"], "MAE": v["mae"], "baseline MAE": v["baseline_mae"]}
                 for t, v in m["performance"].items()]
        out["EVAL_MODELS"] = ("**Model results** (80/20 split, 20K test rows). High-risk classifier: ROC-AUC "
                              f"{f['roc_auc']}, F1 {f['f1']}, threshold {f['high_risk_threshold']} % fault rate.\n\n"
                              + md_table(rows, fmt=r3) + "\n\nTop permutation importances (fault model): "
                              + ", ".join(f"{p['feature']} ({p['importance']:.3f})" for p in f["permutation_importance"][:6])
                              + ". The performance models beat the mean baseline but have modest R² because the "
                                "synthetic KPIs are largely independent; forecasts are shown with R² and an approximate interval.")
    return out


def example_sections() -> dict[str, str]:
    out = {}
    ex = DOCS / "examples"
    for f in sorted(ex.glob("*.json")) if ex.exists() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        resp = d["response"]
        if f.stem == "analyze":
            resp = {k: resp[k] for k in ("understanding", "classification", "prioritization", "prediction",
                                          "root_cause", "impact", "recommendation", "judge", "agent_trace",
                                          "response_id") if k in resp}
            resp["similar_cases"] = d["response"]["similar_cases"]
        body = json.dumps(trim(d["request"]["body"], max_str=400), indent=1) if d["request"]["body"] else ""
        out[f"EXAMPLE:{f.stem}"] = (f"Request `{d['request']['method']} {d['request']['path']}`\n\n```json\n{body}\n```\n\n"
                                    f"Response `{d['status']}` (abridged)\n\n```json\n"
                                    f"{json.dumps(trim(resp, max_list=3 if f.stem != 'analyze' else 2), indent=1)}\n```")
    return out


def dataset_sections() -> dict[str, str]:
    from utils.features import NUMERIC_COLUMNS
    from utils.telecom_kb import CONDITIONS
    out = {}
    if not (ART / "cases.parquet").exists():
        return out
    df = pd.read_parquet(ART / "cases.parquet")
    th = json.loads((ART / "thresholds.json").read_text())
    desc = {"Season": "Winter/Spring/Summer/Fall", "Weather": "Clear/Clouds/Rain/Snow", "Issue Reported": "yes/no flag",
            "City": "city name (synthetic)", "State": "US state/territory code", "Zip": "postal code"}
    rows = []
    for c in NUMERIC_COLUMNS:
        st = th["stats"][c]
        rows.append({"field": c, "type": "numeric", "min": f"{st['min']:.2f}", "median": f"{st['median']:.2f}",
                     "max": f"{st['max']:.2f}", "mean": f"{st['mean']:.2f}"})
    for c, d_ in desc.items():
        rows.append({"field": c, "type": "categorical", "min": "", "median": f"{df[c].nunique()} values",
                     "max": "", "mean": d_})
    out["DATASET_FIELDS"] = md_table(rows)
    trows = [{"condition": c.key, "KPI": c.column, "bad when": c.direction,
              "degraded at": f"{th['conditions'][c.key]['high']:.2f}",
              "severe at": f"{th['conditions'][c.key]['severe']:.2f}",
              "share degraded+": f"{(df[f'c_{c.key}'] > 0).mean():.1%}"} for c in CONDITIONS]
    out["DATASET_THRESHOLDS"] = md_table(trows)
    cat = df["category_name"].value_counts()
    pr = df["priority"].value_counts().sort_index()
    out["DATASET_DISTRIBUTION"] = (md_table([{"category": k, "cases": v, "share": f"{v / len(df):.1%}"} for k, v in cat.items()])
                                   + "\n\n" + md_table([{"priority": k, "cases": v, "share": f"{v / len(df):.1%}"}
                                                        for k, v in pr.items()]))
    out["DATASET_EXAMPLE"] = f"```\n{df['text'].iloc[0]}\n```"
    sample_cols = [c for c in df.columns if not c.startswith("c_")]
    df[sample_cols].head(200).to_csv(SUB / "06_Dataset_Sample.csv", index=False)
    return out


def readme_sections() -> dict[str, str]:
    out = {}
    f = DOCS / "examples" / "analyze.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))["response"]
        p, c, rc, imp, j = d["prediction"], d["classification"], d["root_cause"], d["impact"], d["judge"]
        lines = ["**Result (abridged, from `docs/examples/analyze.json`):**", "",
                 f"- **Understood symptoms:** {', '.join(v['label'] for v in d['understanding']['conditions'].values())}",
                 f"- **Classification:** {c['category_name']} (confidence {c['confidence']:.0%}, method {c['method']})",
                 f"- **Priority:** {p['priority']} {p['severity']}, risk score {p['risk_score']}/100, predicted fault "
                 f"occurrence {p['fault_rate_pct']}% (early-warning {p['early_warning_fault_rate_pct']}%), "
                 f"high-risk probability {p['high_risk_probability']:.0%}",
                 f"- **Root cause:** {rc['summary']}",
                 f"- **Impact:** {imp['customer_impact_level']} customer impact, about {imp['users_affected_estimate']:,} users, "
                 f"SLA breach risk {imp['sla_breach_risk']}, expected MTTR {imp['expected_mttr_hours']} h, services: "
                 f"{', '.join(imp['services_affected'])}",
                 "- **Similar historical cases:** " + "; ".join(
                     f"{h['case_id']} ({h['category_name']}, {h['location']['city']} {h['location']['state']}, "
                     f"{h['season']}, score {h['score']})" for h in d["similar_cases"][:3]),
                 "", "**Recommended actions:**", ""]
        for i, a in enumerate(d["recommendation"]["actions"], 1):
            lines.append(f"{i}. [{a['phase']}] {a['step']}{' ⚠️ needs approval' if a['requires_approval'] else ''} "
                         f"— evidence: {', '.join(a['evidence'])}")
        lines += ["", f"**Escalation:** {d['recommendation']['escalation']}", "",
                  f"**LLM-as-judge:** {j['verdict']} · overall {j['overall']}/5 · "
                  + ", ".join(f"{k} {v}" for k, v in j["scores"].items()) + f" ({j['judge']})"]
        out["README_SAMPLE"] = "\n".join(lines)
    res = []
    if (RES / "recommendation_eval.json").exists():
        s_ = json.loads((RES / "recommendation_eval.json").read_text())["summary"]
        res.append(f"- **Golden-set recommendation eval** ({s_['n_queries']} queries): category accuracy "
                   f"{s_['category_correct']:.0%}, symptom-extraction F1 {s_['condition_f1']:.2f}, SOP coverage "
                   f"{s_['sop_coverage']:.0%}, action-keyword recall {s_['action_keyword_recall']:.0%}, citation validity "
                   f"{s_['citation_validity']:.0%}, judge {s_['judge_overall']:.2f}/5, mean latency {s_['latency_ms']:.0f} ms.")
    if (RES / "retrieval_eval.json").exists():
        r = json.loads((RES / "retrieval_eval.json").read_text())["strategies"]
        res.append("- **Retrieval** (nDCG@10 on paraphrased queries): " + ", ".join(
            f"{k} {v['ndcg@10']:.3f}" for k, v in r.items() if not k.startswith("hybrid_alpha")))
    if (ART / "model_metrics.json").exists():
        m = json.loads((ART / "model_metrics.json").read_text())
        res.append(f"- **Models:** fault-risk R2 {m['fault_risk']['r2']}, early-warning R2 {m['fault_risk']['early_warning_r2']}, "
                   f"high-risk AUC {m['fault_risk']['roc_auc']}; throughput R2 {m['performance']['Throughput (Mbps)']['r2']}.")
    out["README_RESULTS"] = "**Latest results:**\n\n" + "\n".join(res) if res else ""
    return out


def resolve(md: str, fills: dict[str, str]) -> str:
    def rep(m):
        return fills.get(m.group(1).strip(), f"_({m.group(1).strip()}: run the evaluation scripts to populate)_")
    return re.sub(r"<!--\s*([A-Z_]+(?::[a-z_]+)?)\s*-->", rep, md)


# ------------------------------------------------------------------------------------------------- PDF
def styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.4, leading=12.6, spaceAfter=4)
    return {
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY,
                             spaceAfter=10, alignment=TA_CENTER),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=NAVY,
                             spaceBefore=10, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=NAVY,
                             spaceBefore=6),
        "body": base,
        "cell": ParagraphStyle("c", parent=base, fontSize=7.6, leading=9.4, spaceAfter=0),
        "cellh": ParagraphStyle("ch", parent=base, fontSize=7.8, leading=9.4, spaceAfter=0, fontName="Helvetica-Bold",
                                textColor=colors.white),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=6.9, leading=8.2, backColor=colors.HexColor("#f3f5f8"),
                               borderPadding=4, leftIndent=4, spaceAfter=6),
    }


def inline(t: str) -> str:
    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = re.split(r"(`[^`]+`)", t)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<font face='Courier' color='#9c2b2b'>{esc(p[1:-1])}</font>")
            continue
        p = esc(p)
        p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
        p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", p)
        p = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"<i>\1</i>", p)
        p = re.sub(r"(https?://[^\s)|]+)", r"<link href='\1' color='blue'>\1</link>", p)
        out.append(p)
    return "".join(out)


def md_to_flowables(md: str, st: dict, width: float) -> list:
    lines, out, i = md.splitlines(), [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            text = "\n".join(buf)
            if len(buf) > 140:
                text = "\n".join(buf[:140]) + "\n..."
            # wrap very long lines
            text = "\n".join(l if len(l) < 150 else l[:147] + "..." for l in text.splitlines())
            out.append(Preformatted(text, st["code"]))
            i = j + 1
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            ncol = max(len(r) for r in rows)
            data = [[Paragraph(inline(c), st["cellh" if k == 0 else "cell"]) for c in r + [""] * (ncol - len(r))]
                    for k, r in enumerate(rows)]
            lens = [max(len(r[c]) if c < len(r) else 0 for r in rows) for c in range(ncol)]
            tot = sum(min(l, 60) + 6 for l in lens)
            cw = [width * (min(l, 60) + 6) / tot for l in lens]
            t = Table(data, colWidths=cw, repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c3cf")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
            out += [t, Spacer(1, 6)]
        elif ln.startswith("# "):
            out.append(Paragraph(inline(ln[2:]), st["h1"]))
            i += 1
        elif ln.startswith("## "):
            out.append(Paragraph(inline(ln[3:]), st["h2"]))
            i += 1
        elif ln.startswith("### "):
            out.append(Paragraph(inline(ln[4:]), st["h3"]))
            i += 1
        elif re.match(r"^\s*(-|\d+\.)\s", ln):
            items, numbered = [], bool(re.match(r"^\s*\d+\.", ln))
            while i < len(lines) and re.match(r"^\s*(-|\d+\.)\s", lines[i]):
                txt = re.sub(r"^\s*(-|\d+\.)\s", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip() and \
                        not re.match(r"^\s*(-|\d+\.)\s", lines[i]):
                    txt += " " + lines[i].strip()
                    i += 1
                items.append(ListItem(Paragraph(inline(txt), st["body"]), leftIndent=12))
            out.append(ListFlowable(items, bulletType="1" if numbered else "bullet", start="1" if numbered else None,
                                    bulletFontSize=8, leftIndent=14))
        elif ln.strip() == "":
            i += 1
        else:
            buf = [ln]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|\s*(-|\d+\.)\s)", lines[i]):
                buf.append(lines[i])
                i += 1
            out.append(Paragraph(inline(" ".join(b.strip() for b in buf)), st["body"]))
    return out


def build_pdf(md: str, path: Path, title: str, extra_first: list | None = None) -> None:
    st = styles()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title=title, author="Jerom Santhiyagu")
    width = A4[0] - 3.2 * cm

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.6 * cm, 0.9 * cm, f"{title} - Telecom NetOps Assistant (PU Bootcamp Capstone, Project 4)")
        canvas.drawRightString(A4[0] - 1.6 * cm, 0.9 * cm, f"Page {d.page}")
        canvas.restoreState()

    flow = md_to_flowables(md, st, width)
    if extra_first:
        flow = flow[:1] + extra_first + flow[1:]
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print("written", path.name)


def main() -> None:
    fills = eval_sections() | example_sections() | dataset_sections() | readme_sections()
    readme = resolve((DOCS / "README.template.md").read_text(encoding="utf-8"), fills)
    (SUB / "README.md").write_text(readme, encoding="utf-8")
    (SRC / "README.md").write_text(readme.replace("](03_Source_Code/", "]("), encoding="utf-8")
    print("written README.md")
    (DOCS / "build").mkdir(exist_ok=True)
    jobs = [("design_document.md", "02_Design_Document.pdf", "Design Document"),
            ("api_documentation.md", "05_API_Docs.pdf", "API Documentation"),
            ("dataset_details.md", "06_Dataset_Details.pdf", "Dataset Details")]
    for src, pdf, title in jobs:
        md = resolve((DOCS / src).read_text(encoding="utf-8"), fills)
        (DOCS / "build" / src).write_text(md, encoding="utf-8")
        extra = None
        if src == "design_document.md" and (SUB / "01_Architecture.jpeg").exists():
            w = A4[0] - 3.2 * cm
            extra = [Image(str(SUB / "01_Architecture.jpeg"), width=w, height=w * 1452 / 2420),
                     Paragraph("<i>Figure 1 - High-level architecture (full size: 01_Architecture.pdf)</i>",
                               styles()["body"]), Spacer(1, 6)]
        build_pdf(md, SUB / pdf, title, extra)


if __name__ == "__main__":
    main()
