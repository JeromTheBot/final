"""Pydantic request models (API validation layer = first guardrail)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from utils.features import API_ALIASES
from utils.telecom_kb import CATEGORY_NAMES, CONDITION_BY_KEY, SEASONS, WEATHER


class Filters(BaseModel):
    """Metadata filters (all optional, values within a field are OR-ed, fields are AND-ed)."""
    model_config = ConfigDict(extra="forbid")
    state: list[str] | None = Field(None, description="2-letter state codes, e.g. ['TX','CA']")
    city: list[str] | None = None
    season: list[str] | None = Field(None, description=f"Any of {SEASONS}")
    weather: list[str] | None = Field(None, description=f"Any of {WEATHER}")
    category: list[str] | None = Field(None, description=f"Any of {list(CATEGORY_NAMES)}")
    priority: list[Literal["P1", "P2", "P3", "P4"]] | None = None
    min_risk: float | None = Field(None, ge=0, le=100)
    required_conditions: list[str] | None = Field(None, description="Network conditions every case must show, "
                                                                    "e.g. ['high_latency','packet_loss']")
    issue_reported: Literal["yes", "no"] | None = None

    @field_validator("state")
    @classmethod
    def _state(cls, v):
        if v is None:
            return v
        out = [s.strip().upper() for s in v]
        for s in out:
            if len(s) != 2 or not s.isalpha():
                raise ValueError(f"invalid state code '{s}'")
        return out

    @field_validator("season")
    @classmethod
    def _season(cls, v):
        if v is None:
            return v
        out = [s.strip().title() for s in v]
        bad = [s for s in out if s not in SEASONS]
        if bad:
            raise ValueError(f"unknown season(s) {bad}; allowed {SEASONS}")
        return out

    @field_validator("weather")
    @classmethod
    def _weather(cls, v):
        if v is None:
            return v
        out = [s.strip().title() for s in v]
        bad = [s for s in out if s not in WEATHER]
        if bad:
            raise ValueError(f"unknown weather {bad}; allowed {WEATHER}")
        return out

    @field_validator("category")
    @classmethod
    def _cat(cls, v):
        if v is None:
            return v
        out = [s.strip().upper() for s in v]
        bad = [s for s in out if s not in CATEGORY_NAMES]
        if bad:
            raise ValueError(f"unknown category {bad}; allowed {list(CATEGORY_NAMES)}")
        return out

    @field_validator("required_conditions")
    @classmethod
    def _conds(cls, v):
        if v is None:
            return v
        bad = [s for s in v if s not in CONDITION_BY_KEY]
        if bad:
            raise ValueError(f"unknown condition(s) {bad}; allowed {list(CONDITION_BY_KEY)}")
        return v

    @field_validator("city")
    @classmethod
    def _city(cls, v):
        if v is None:
            return v
        for c in v:
            if len(c) > 60 or not all(ch.isalnum() or ch in " .-'" for ch in c):
                raise ValueError(f"invalid city name '{c[:60]}'")
        return [c.strip() for c in v]


class KPIInput(BaseModel):
    """Live KPI readings for a cell (any subset; missing values are filled with fleet medians)."""
    model_config = ConfigDict(extra="forbid")
    cell_availability: float | None = Field(None, ge=0, le=100)
    mttr_hours: float | None = Field(None, ge=0, le=500)
    throughput_mbps: float | None = Field(None, ge=0, le=20000)
    latency_ms: float | None = Field(None, ge=0, le=5000)
    packet_loss_rate: float | None = Field(None, ge=0, le=100)
    call_drop_rate: float | None = Field(None, ge=0, le=100)
    handover_success_rate: float | None = Field(None, ge=0, le=100)
    alarm_count: float | None = Field(None, ge=0, le=10000)
    critical_alarm_count: float | None = Field(None, ge=0, le=10000)
    parameter_changes: float | None = Field(None, ge=0, le=10000)
    successful_config_changes: float | None = Field(None, ge=0, le=100)
    data_usage_gb: float | None = Field(None, ge=0, le=100000)
    user_count: float | None = Field(None, ge=0, le=1_000_000)
    signal_strength_dbm: float | None = Field(None, ge=-150, le=-20)
    jitter_ms: float | None = Field(None, ge=0, le=5000)
    connection_setup_success_rate: float | None = Field(None, ge=0, le=100)
    security_incidents: float | None = Field(None, ge=0, le=10000)
    authentication_failures: float | None = Field(None, ge=0, le=100000)
    temperature_c: float | None = Field(None, ge=-60, le=70)
    humidity: float | None = Field(None, ge=0, le=100)
    season: Literal["Winter", "Spring", "Summer", "Fall"] | None = None
    weather: Literal["Clear", "Clouds", "Rain", "Snow"] | None = None

    def to_columns(self) -> dict:
        return {API_ALIASES[k]: v for k, v in self.model_dump().items() if v is not None}


class QueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "query": "A cell is experiencing high latency, packet loss, and reduced throughput. "
                 "What similar network conditions have occurred before?",
        "filters": {"season": ["Winter"]}, "top_k": 5}})
    query: str = Field("", max_length=2000, description="Natural-language description of the network problem")
    filters: Filters | None = None
    kpis: KPIInput | None = Field(None, description="Optional live KPI readings for the affected cell")
    top_k: int = Field(5, ge=1, le=20)
    alpha: float = Field(0.6, ge=0, le=1, description="Weight of semantic vs keyword ranking in hybrid fusion")
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    use_llm: bool = True
    auto_filters: bool = Field(True, description="Apply state/city/season/weather mentioned in the query as filters")

    @model_validator(mode="after")
    def _need_input(self):
        if not self.query.strip() and self.kpis is None:
            raise ValueError("Provide a 'query' and/or 'kpis'.")
        return self


class KPIRequest(BaseModel):
    kpis: KPIInput
    explain: bool = True


class IncidentItem(BaseModel):
    id: str = Field(..., max_length=64)
    case_id: str | None = Field(None, pattern=r"^NOC-\d{6}$")
    kpis: KPIInput | None = None
    description: str | None = Field(None, max_length=500)

    @model_validator(mode="after")
    def _need(self):
        if not self.case_id and self.kpis is None:
            raise ValueError("each incident needs 'case_id' or 'kpis'")
        return self


class PrioritizeRequest(BaseModel):
    incidents: list[IncidentItem] = Field(..., min_length=1, max_length=500)


class FeedbackRequest(BaseModel):
    response_id: str = Field(..., pattern=r"^[0-9a-f]{12}$")
    rating: int = Field(..., ge=1, le=5)
    helpful: bool | None = None
    correct_category: str | None = None
    helpful_case_ids: list[str] | None = None
    unhelpful_case_ids: list[str] | None = None
    effective_actions: list[str] | None = None
    comment: str = Field("", max_length=2000)

    @field_validator("correct_category")
    @classmethod
    def _cat(cls, v):
        if v is not None and v not in CATEGORY_NAMES:
            raise ValueError(f"unknown category; allowed {list(CATEGORY_NAMES)}")
        return v
