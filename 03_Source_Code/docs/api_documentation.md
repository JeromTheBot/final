# API Documentation — Telecom NetOps Assistant

Base URL: `http://localhost:8000` · Interactive Swagger UI at `/docs`, ReDoc at `/redoc`, OpenAPI schema at `/openapi.json`.

**Authentication.** Optional. If `NETOPS_API_KEY` is set, every POST endpoint requires the header `X-API-Key: <key>`.
**Response headers.** `X-Request-ID` (pass your own or one is generated) and `X-Response-Time-ms`.
**Errors.** `422` means a validation or guardrail violation (`{"error":"guardrail_violation","code":"PROMPT_INJECTION|OFF_TOPIC|QUERY_TOO_SHORT|QUERY_TOO_LONG|EMPTY_QUERY","message":...}`, or a FastAPI `detail` list for schema errors). `401` means a bad API key, `404` an unknown case or response, and `500` an internal error (logged with the request ID).

## Endpoint summary

| Method | Path | Purpose | Requirement |
|---|---|---|---|
| GET | `/health` | Liveness, index size, embedder, LLM status | ops |
| GET | `/metadata` | Filter values, taxonomy, conditions, index and model metrics | ops |
| GET | `/metrics` | Request counts, p50/p95 latency, events, alerts | monitoring |
| POST | `/search` | Hybrid / semantic / keyword similarity search with metadata filters | R1: RAG, hybrid, filters |
| POST | `/classify` | Incident classification agent + priority | R1: classification |
| POST | `/recommend` | Similar cases + classification + priority + basic troubleshooting plan | R1: recommendation, prioritisation |
| POST | `/prioritize` | Rank a batch of incidents (case IDs or live KPIs) | R1: prioritisation |
| POST | `/analyze` | Full multi-agent pipeline: root cause, impact, predictions, plan, LLM-as-judge | R2 |
| POST | `/predict/risk` | Fault-risk prediction with drivers | R2 |
| POST | `/predict/performance` | Throughput / latency / call-drop forecast | R2 |
| POST | `/feedback` | Rate a response (feeds the feedback loop) | R2 |
| GET | `/feedback/stats` | Feedback aggregates, category corrections | R2 |
| GET | `/cases/{case_id}` | One historical case | — |

## Common request body — `QueryRequest` (`/search`, `/classify`, `/recommend`, `/analyze`)

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string ≤ 2000 | "" | Natural-language problem description (required unless `kpis` is given) |
| `filters` | object | null | See *Filters* |
| `kpis` | object | null | Live KPI readings (see *KPIInput*); missing values are filled with fleet medians |
| `top_k` | int 1–20 | 5 | Number of similar cases |
| `alpha` | float 0–1 | 0.6 | Semantic weight in hybrid fusion |
| `mode` | `hybrid` \| `semantic` \| `keyword` | hybrid | Retrieval strategy |
| `use_llm` | bool | true | Use the configured LLM (ignored when none is configured) |
| `auto_filters` | bool | true | Apply state/city/season/weather mentioned in the query as soft filters |

**Filters** (fields are AND-ed, values within a field are OR-ed): `state` [2-letter codes], `city` [exact names],
`season` [Winter, Spring, Summer, Fall], `weather` [Clear, Clouds, Rain, Snow], `category` [RAN_COVERAGE, CAPACITY,
TRANSPORT_QOS, MOBILITY, EQUIPMENT_FAULT, CONFIGURATION, SECURITY, ENVIRONMENT, NORMAL], `priority` [P1..P4],
`min_risk` 0–100, `required_conditions` [e.g. high_latency, packet_loss, call_drops, handover_failure, poor_signal,
...] (the full list is in `/metadata`), `issue_reported` yes/no.

**KPIInput** (all optional): `cell_availability`, `mttr_hours`, `throughput_mbps`, `latency_ms`, `packet_loss_rate`,
`call_drop_rate`, `handover_success_rate`, `alarm_count`, `critical_alarm_count`, `parameter_changes`,
`successful_config_changes`, `data_usage_gb`, `user_count`, `signal_strength_dbm` (−150…−20), `jitter_ms`,
`connection_setup_success_rate`, `security_incidents`, `authentication_failures`, `temperature_c`, `humidity`,
`season`, `weather`. Percentages are 0–100. Unknown fields are rejected.

## POST /search

<!-- EXAMPLE:search -->

## POST /classify

<!-- EXAMPLE:classify -->

## POST /recommend

<!-- EXAMPLE:recommend -->

## POST /analyze

Response sections: `understanding`, `guardrails`, `retrieval`, `similar_cases[]`, `classification`,
`prioritization`, `prediction` (fault rate, early-warning rate, high-risk probability, risk score, priority,
`performance_forecast`), `root_cause` (summary, likely_root_causes, kpi_deviations, model_drivers,
condition_prevalence_in_similar_cases), `impact`, `recommendation` (summary, diagnosis, actions[{step, phase,
rationale, evidence, requires_approval}], monitoring, escalation, human_approval_required), `judge` (scores,
overall, verdict, issues), `sops`, `agent_trace[]`, `llm`, `response_id`, `warnings`.

<!-- EXAMPLE:analyze -->

## POST /prioritize

Body: `{"incidents": [{"id": str, "case_id": "NOC-######" | null, "kpis": KPIInput | null, "description": str}]}`
(1–500 items, each needs `case_id` or `kpis`). Items are ranked by priority, then risk score, then users.

<!-- EXAMPLE:prioritize -->

## POST /predict/risk

Body: `{"kpis": KPIInput, "explain": true}`

<!-- EXAMPLE:predict_risk -->

## POST /predict/performance

<!-- EXAMPLE:predict_performance -->

## POST /feedback

Body: `response_id` (from `/recommend` or `/analyze`), `rating` 1–5, optional `helpful`, `correct_category`,
`helpful_case_ids`, `unhelpful_case_ids`, `effective_actions`, `comment`.

<!-- EXAMPLE:feedback -->

## Guardrail example

<!-- EXAMPLE:guardrail -->

## GET /metrics

<!-- EXAMPLE:metrics -->

## curl quick start

```bash
curl -s -X POST localhost:8000/analyze -H "Content-Type: application/json" \
  -d '{"query":"A network region has increasing fault occurrence and declining handover success. What corrective actions should be considered?"}'
curl -s -X POST localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query":"dropped calls in winter","filters":{"state":["TX"],"priority":["P1","P2"]},"top_k":3}'
curl -s -X POST localhost:8000/predict/risk -H "Content-Type: application/json" \
  -d '{"kpis":{"critical_alarm_count":12,"authentication_failures":40,"call_drop_rate":4}}'
```
