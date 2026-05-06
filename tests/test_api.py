from io import BytesIO
from pathlib import Path
import shutil
import uuid

from fastapi.testclient import TestClient

import deployment.api.main as api_main


class DummyAgent:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.calls = []

    def run(self, image_path: str, pdf_path: str):
        self.calls.append((image_path, pdf_path))
        if self.should_fail:
            raise RuntimeError("boom")
        return {
            "diagnosis": {
                "top_finding": "Normal",
                "probabilities": [0.1, 0.1, 0.1, 0.1, 0.6],
                "uncertainty_std": 0.02,
            },
            "confidence": 0.6,
            "history_data": {"metadata": {"patient_id": "P1"}},
            "pubmed_citations": [],
            "escalation_required": False,
        }


def make_temp_dir() -> Path:
    base = Path(uuid.uuid4().hex)
    base.mkdir(exist_ok=True)
    return base


def test_health_is_lazy(monkeypatch):
    base_dir = make_temp_dir()
    try:
        monkeypatch.setattr(api_main, "TEMP_ROOT", Path("."))
        monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
        app = api_main.create_app()

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "models_loaded": False}
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_analyze_cleans_request_dir_on_success(monkeypatch):
    base_dir = make_temp_dir()
    try:
        agent = DummyAgent()
        monkeypatch.setattr(api_main, "TEMP_ROOT", Path("."))
        monkeypatch.setattr(api_main, "build_agent", lambda: agent)
        app = api_main.create_app()

        with TestClient(app) as client:
            response = client.post(
                "/analyze",
                files={
                    "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                    "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
                },
            )

        assert response.status_code == 200
        image_path, pdf_path = agent.calls[0]
        request_dir = Path(image_path).parent
        assert not Path(image_path).exists()
        assert not Path(pdf_path).exists()
        assert not request_dir.exists()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_analyze_cleans_request_dir_on_failure(monkeypatch):
    base_dir = make_temp_dir()
    try:
        agent = DummyAgent(should_fail=True)
        monkeypatch.setattr(api_main, "TEMP_ROOT", Path("."))
        monkeypatch.setattr(api_main, "build_agent", lambda: agent)
        app = api_main.create_app()

        with TestClient(app) as client:
            response = client.post(
                "/analyze",
                files={
                    "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                    "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
                },
            )

        assert response.status_code == 500
        assert response.json()["detail"].startswith("Analysis failed:")
        if agent.calls:
            request_dir = Path(agent.calls[0][0]).parent
            assert not request_dir.exists()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
