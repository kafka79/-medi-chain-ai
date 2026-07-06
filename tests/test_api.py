from io import BytesIO
import pytest
from fastapi.testclient import TestClient

import os
os.environ["REDIS_URL"] = "memory://"
os.environ["STORAGE_MODE"] = "local"
os.environ["TESTING"] = "true"

import deployment.api.main as api_main


class DummyAgent:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.calls = []

    async def run(self, image_path: str, pdf_path: str):
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

class DummyStorageProvider:
    def __init__(self, *args, **kwargs):
        self.files = {}
        
    def save(self, file_obj, rel_path):
        self.files[rel_path] = b"mock_data"
        return rel_path
        
    def load(self, rel_path):
        return rel_path
        
    def delete(self, rel_path):
        keys = list(self.files.keys())
        for k in keys:
            if k.startswith(rel_path):
                del self.files[k]
                
    def cleanup(self, max_age_seconds=None, *args, **kwargs):
        self.files.clear()

def test_health_is_lazy(monkeypatch):
    monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(api_main, "storage", DummyStorageProvider())
    # Flaw #3-structural Fix: Isolate from host .env to prevent MAX_CONCURRENT_REQUESTS leakage
    monkeypatch.setattr(api_main, "MAX_CONCURRENT_REQUESTS", 2)
    app = api_main.create_app()

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "models_loaded": True,
            "concurrency_limit": 2,
            "version": "1.3.0",
        }

def test_analyze_cleans_request_dir_on_success(monkeypatch):
    agent = DummyAgent()
    storage_mock = DummyStorageProvider()
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    monkeypatch.setattr(api_main, "storage", storage_mock)
    
    # Also mock ehr_gateway to prevent real requests
    class DummyEHR:
        async def push_report(self, fhir_json, is_retry=False): return True
    monkeypatch.setattr(api_main, "ehr_gateway", DummyEHR())
    
    app = api_main.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/analyze?sync=true",
            files={
                "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
            },
            headers={"X-API-Key": "dev-secret-key-123"}
        )

    assert response.status_code == 200
    assert len(agent.calls) == 1
    # Check that cleanup task deleted the files
    assert len(storage_mock.files) == 0

def test_analyze_cleans_request_dir_on_failure(monkeypatch):
    agent = DummyAgent(should_fail=True)
    storage_mock = DummyStorageProvider()
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    monkeypatch.setattr(api_main, "storage", storage_mock)
    app = api_main.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/analyze?sync=true",
            files={
                "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
            },
            headers={"X-API-Key": "dev-secret-key-123"}
        )

    assert response.status_code == 500
    assert response.json()["detail"].startswith("Analysis failed:")
    # Even on failure, files should be deleted
    assert len(storage_mock.files) == 0


@pytest.mark.asyncio
async def test_cleanup_circuit_breaker(monkeypatch):
    import asyncio
    # Mock storage.cleanup to fail consecutively
    class FailingStorage:
        def cleanup(self, max_age_seconds=None):
            raise RuntimeError("storage error")
            
    monkeypatch.setattr(api_main, "storage", FailingStorage())
    
    sleep_calls = []
    async def mock_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 5:
            # Raise an error to break the infinite loop once we verify the backoff sequence
            raise asyncio.CancelledError("Test completed successfully")
        
    monkeypatch.setattr(asyncio, "sleep", mock_sleep)
    
    # Run the cleanup task, which should run with exponential backoff
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await api_main.cleanup_old_temp_files()
        
    assert "Test completed successfully" in str(exc_info.value)
    # Verify exponential backoff: 600s, 1200s, 2400s, 3600s, 3600s
    assert sleep_calls == [600, 1200, 2400, 3600, 3600]

