# 📡 AI-Powered Telecom Network Operations Assistant

**PU Bootcamp GenAI Capstone · Project 4** · Jerom Santhiyagu (Prodapt)

NOC engineers describe a network problem in plain English, optionally with live KPIs. The assistant retrieves the
most similar historical 5G network cases (hybrid **BM25 + semantic** search with **metadata filters**),
**classifies** and **prioritises** the incident, predicts **fault risk** and **network performance**, explains the
**root cause and network impact**, and produces a **grounded troubleshooting plan**. A team of **7 collaborating
agents** builds the plan, an **LLM-as-judge** validates it, and engineer **feedback** improves later answers.
It runs fully offline in a deterministic mode, and an LLM (OpenAI / Azure OpenAI / Gemini / Claude) is used
when a key is configured.

## Submission contents

| # | Deliverable | File |
|---|---|---|
| 1 | Architecture diagram | `01_Architecture.pdf` / `01_Architecture.jpeg` |
| 2 | Design document (decisions and trade-offs) | `02_Design_Document.pdf` |
| 3 | Full executable code (microservice) | `03_Source_Code/` |
| 4 | README (setup and sample usage) | `README.md` (this file) |
| 5 | API documentation | `05_API_Docs.pdf` (+ Swagger at `/docs`) |
| 6 | Dataset details | `06_Dataset_Details.pdf`, `06_Dataset_Sample.csv` |
| 7 | Demo video script (8-min demo plan) | `07_Demo_Script.md` (record the MP4 from this) |
| 8 | Presentation deck | `08_Presentation.pptx` |

## Requirement coverage

| Requirement 1 (Basic) | Where |
|---|---|
| Basic RAG for network incident similarity search | `app/retrieval.py`, `POST /search`, `/recommend` |
| Hybrid search (keywords + semantic) | BM25 + FAISS + Reciprocal Rank Fusion (`mode`, `alpha`) |
| Network incident classification agent | `agents/classification_agent.py`, `POST /classify` |
| Metadata filtering (state, city, season, network conditions) | `filters` on all query endpoints + auto-detection from the query |
| Basic network troubleshooting recommendation | `agents/recommendation_agent.py`, `POST /recommend` |
| Input validation guardrails | `app/schemas.py`, `app/guardrails.py` |
| Incident prioritisation | risk score → P1–P4, `POST /prioritize` |
| API endpoints | FastAPI `app/main.py` (13 endpoints) |

| Requirement 2 (Advanced) | Where |
|---|---|
| DeepEval for recommendation quality evaluation | `evaluation/deepeval_eval.py`, `tests/test_deepeval.py` |
| Network fault risk prediction | `app/ml_models.py`, `POST /predict/risk` |
| LLM-as-judge for network diagnosis validation | `agents/judge_agent.py` (pass/revise loop) |
| Multi-agent network troubleshooting collaboration | `agents/orchestrator.py` (7 agents) |
| Network performance prediction | `POST /predict/performance` |
| Root-cause and network-impact analysis | `agents/root_cause_agent.py` |
| Feedback loop for continuous improvement | `app/feedback.py`, `POST /feedback` |
| Simple front-end interface | `frontend/streamlit_app.py` (Streamlit) |

## Project setup

**Prerequisites:** Python 3.11+ (tested on 3.14) and about 2 GB of disk. Internet is needed once, to download the
dataset and the embedding model. A fully offline build is available with `--backend tfidf`.

```bash
cd 03_Source_Code
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # optional: add ONE LLM key (OPENAI_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY)
```

**Build the data artifacts** (one time: ingest about 6 s, index about 20 min for 100K rows on a laptop CPU, train
about 10 s):

```bash
python -m data.ingest                  # download + clean + enrich 100K records
python -m data.build_index             # embeddings + FAISS + BM25   (add --rows 20000 for a quick build)
python -m data.train_models            # fault-risk and performance models
```

Or run everything, tests included: `powershell -ExecutionPolicy Bypass -File run_pipeline.ps1` (Windows) /
`./run_pipeline.sh` (Linux/macOS).

**Run the microservice and the UI** (two terminals):

```bash
uvicorn app.main:app --port 8000                 # API  -> http://localhost:8000/docs
streamlit run frontend/streamlit_app.py          # UI   -> http://localhost:8501
```

**Docker:** `docker compose up --build` (API on 8000, UI on 8501).

**LLM configuration (optional).** `NETOPS_LLM_PROVIDER=auto` picks the first key it finds: Azure OpenAI, then
OpenAI, then Gemini, then Anthropic. Default models are `gpt-4o-mini`, `gemini-2.5-flash` and
`claude-haiku-4-5-20251001`, and can be overridden with `NETOPS_OPENAI_MODEL` etc. With no key the system uses its
deterministic agents. Every response reports which path ran (`llm.used`, `recommendation.method`, `judge.judge`).

## Sample usage

```bash
curl -s -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"query": "A network region has increasing fault occurrence and declining handover success. What corrective actions should be considered?"}'
```

<!-- README_SAMPLE -->

More examples, with full request and response bodies, are in `05_API_Docs.pdf` and `03_Source_Code/docs/examples/`.

## Testing and evaluation

```bash
python -m pytest                                  # unit + API + agent tests (offline, deterministic)
python -m evaluation.retrieval_eval               # retrieval strategies + embedding-model comparison
python -m evaluation.deepeval_eval                # golden-set recommendation eval (+ DeepEval when an LLM key is set)
pytest tests/test_deepeval.py -m deepeval         # DeepEval quality gates (needs an LLM key)
python -m scripts.capture_examples                # refresh API examples
python -m scripts.build_docs                      # rebuild PDFs from docs/*.md + latest results
```

<!-- README_RESULTS -->

## Project structure

```
03_Source_Code/
├── app/                 FastAPI service: config, schemas, services, retrieval, ML models, LLM client,
│                        guardrails, feedback store, monitoring
├── agents/              7 agents + orchestrator + prompts (query, retrieval, classification, risk prediction,
│                        root cause & impact, recommendation, LLM-as-judge)
├── data/                ingest.py, build_index.py, train_models.py, raw/ (dataset), artifacts/ (indexes, models)
├── utils/               telecom_kb.py (conditions, taxonomy, SOP playbook), features.py, query_parser.py, logging
├── frontend/            streamlit_app.py
├── evaluation/          golden_queries.json, retrieval_eval.py, deepeval_eval.py, results/
├── tests/               pytest suite (guardrails, features, API, agents, DeepEval)
├── docs/                design / API / dataset docs (Markdown sources), examples/
├── scripts/             architecture diagram, docs and deck builders, example capture
├── requirements.txt  Dockerfile  docker-compose.yml  .env.example  run_pipeline.ps1/.sh
```

## Design highlights

- **KPIs into language.** 21 data-driven KPI conditions (fleet percentiles) turn each numeric record into a
  searchable incident narrative. One record is one chunk.
- **Two-field hybrid index.** Dense MiniLM embeddings of the narrative summary plus BM25 over a keyword field, fused
  with RRF, re-ranked by symptom overlap, and nudged by feedback.
- **Case-based KPI profile.** Text-only questions can still be scored by the ML models.
- **Grounding and safety.** Every action cites a retrieved case or SOP. Hallucinated IDs are stripped,
  service-affecting steps are flagged for approval, and PII is redacted before anything reaches an LLM.
- **Honest modelling.** The fault rate is near-deterministic in this dataset, so an early-warning model is reported
  alongside the full one. `Issue Reported` was found to be label noise.

## Troubleshooting

- **SSL errors downloading the dataset or model** behind a corporate proxy: `truststore` is used automatically. Or
  build fully offline with `python -m data.build_index --backend tfidf`.
- **"Index was built with X but Y loaded"**: rebuild the index after changing `NETOPS_EMBEDDING_MODEL`.
- **UI says the API is not reachable**: start uvicorn first, or set `NETOPS_API_URL`.
