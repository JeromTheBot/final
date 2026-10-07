"""FastAPI microservice - AI-Powered Telecom Network Operations Assistant.

    uvicorn app.main:app --port 8000        (Swagger UI: http://localhost:8000/docs)
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.guardrails import GuardrailError
from app.monitoring import METRICS
from app.schemas import FeedbackRequest, KPIRequest, PrioritizeRequest, QueryRequest
from app.services import NetOpsService
from utils.logging_utils import setup_logging

setup_logging()
log = logging.getLogger("netops.api")
_service: NetOpsService | None = None


def get_service() -> NetOpsService:
    global _service
    if _service is None:
        _service = NetOpsService()
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_service()           # load indexes & models once at start-up
    yield


app = FastAPI(
    title="Telecom NetOps Assistant API",
    version="1.0.0",
    description="Semantic + keyword retrieval of historical 5G network incidents, incident classification, "
                "prioritisation, fault-risk & performance prediction, root-cause/impact analysis, multi-agent "
                "troubleshooting recommendations with LLM-as-judge validation, and a feedback loop.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover - safety net
        log.exception("Unhandled error on %s", request.url.path)
        METRICS.error(request.url.path, str(exc))
        response = JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    ms = (time.perf_counter() - t0) * 1000
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    METRICS.observe(path, response.status_code, ms)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s -> %s (%.0f ms) rid=%s", request.method, path, response.status_code, ms, rid)
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    key = get_settings().api_key
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.exception_handler(GuardrailError)
async def guardrail_handler(request: Request, exc: GuardrailError):
    return JSONResponse(status_code=422, content={"error": "guardrail_violation", "code": exc.code,
                                                  "message": exc.message})


# ---------------------------------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health():
    svc = get_service()
    return {"status": "ok", "cases_indexed": len(svc.retriever.cases), "embedding_model": svc.retriever.embedder.name,
            "llm": svc.llm.describe()}


@app.get("/metadata", tags=["ops"], summary="Filter values, taxonomy, index and model information")
def metadata():
    return get_service().metadata()


@app.get("/metrics", tags=["ops"], summary="Service metrics, latency percentiles and alerts")
def metrics():
    snap = METRICS.snapshot()
    snap["llm"] = get_service().llm.describe()
    return snap


# ---- Requirement 1 ---------------------------------------------------------------------------------
@app.post("/search", tags=["retrieval"], dependencies=[Depends(require_api_key)],
          summary="Hybrid (BM25 + semantic) similarity search over historical incidents with metadata filters")
def search(req: QueryRequest):
    return get_service().search(req)


@app.post("/classify", tags=["analysis"], dependencies=[Depends(require_api_key)],
          summary="Incident classification agent (query and/or KPIs) with priority")
def classify(req: QueryRequest):
    return get_service().classify(req)


@app.post("/recommend", tags=["analysis"], dependencies=[Depends(require_api_key)],
          summary="Basic troubleshooting recommendation: similar cases + classification + priority + actions")
def recommend(req: QueryRequest):
    return get_service().recommend(req)


@app.post("/prioritize", tags=["analysis"], dependencies=[Depends(require_api_key)],
          summary="Rank a batch of incidents (case IDs or live KPIs) by priority and risk")
def prioritize(req: PrioritizeRequest):
    return get_service().prioritize(req.incidents)


# ---- Requirement 2 ---------------------------------------------------------------------------------
@app.post("/analyze", tags=["analysis"], dependencies=[Depends(require_api_key)],
          summary="Full multi-agent analysis: root cause, impact, predictions, recommendations, LLM-as-judge")
def analyze(req: QueryRequest):
    return get_service().analyze(req)


@app.post("/predict/risk", tags=["prediction"], dependencies=[Depends(require_api_key)],
          summary="Network fault-risk prediction with explanation")
def predict_risk(req: KPIRequest):
    return get_service().predict_risk(req.kpis.to_columns(), req.explain)


@app.post("/predict/performance", tags=["prediction"], dependencies=[Depends(require_api_key)],
          summary="Network performance prediction (throughput, latency, call-drop rate)")
def predict_performance(req: KPIRequest):
    return get_service().predict_performance(req.kpis.to_columns(), req.explain)


@app.post("/feedback", tags=["feedback"], dependencies=[Depends(require_api_key)],
          summary="Rate a response; feeds the continuous-improvement loop")
def feedback(req: FeedbackRequest):
    try:
        fid = get_service().feedback.add_feedback(**req.model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown response_id")
    METRICS.event("feedback_received")
    return {"status": "recorded", "feedback_id": fid}


@app.get("/feedback/stats", tags=["feedback"])
def feedback_stats():
    return get_service().feedback.stats()


@app.get("/cases/{case_id}", tags=["retrieval"], summary="Fetch one historical case")
def get_case(case_id: str):
    rec = get_service().retriever.get(case_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="case not found in index")
    return rec
