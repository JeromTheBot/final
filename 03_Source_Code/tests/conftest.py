import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NETOPS_LLM_PROVIDER", "none")   # tests are deterministic & offline by default

ARTIFACTS = ROOT / "data" / "artifacts"
HAVE_ARTIFACTS = all((ARTIFACTS / f).exists() for f in ("incidents.faiss", "bm25.joblib", "models.joblib",
                                                         "cases_indexed.parquet", "thresholds.json"))
needs_artifacts = pytest.mark.skipif(not HAVE_ARTIFACTS, reason="run the data pipeline first (see README)")


class FakeLLM:
    """Scripted LLM used to test the LLM code paths without network access."""
    provider, model = "fake", "fake-1"

    def __init__(self, responses: dict | None = None, fail: bool = False):
        from app.llm import LLMStats
        self.stats = LLMStats()
        self.responses = responses or {}
        self.fail = fail

    @property
    def available(self):
        return True

    def describe(self):
        return {"provider": "fake", "model": "fake-1", "available": True, "calls": self.stats.calls,
                "failures": self.stats.failures, "avg_latency_ms": 0}

    def generate_json(self, system, user, **kw):
        self.stats.calls += 1
        if self.fail:
            self.stats.failures += 1
            raise RuntimeError("LLM down")
        for key, resp in self.responses.items():
            if key in user or key in system:
                return resp
        return {}

    def generate(self, system, user, **kw):
        import json
        return json.dumps(self.generate_json(system, user, **kw))


@pytest.fixture(scope="session")
def service(tmp_path_factory):
    if not HAVE_ARTIFACTS:
        pytest.skip("artifacts missing")
    from app.config import get_settings
    from app.services import NetOpsService
    s = get_settings().model_copy(update={"feedback_db": tmp_path_factory.mktemp("fb") / "feedback.db"})
    return NetOpsService(settings=s)


@pytest.fixture(scope="session")
def client(service):
    import app.main as main
    from fastapi.testclient import TestClient
    main._service = service
    with TestClient(main.app) as c:
        yield c
