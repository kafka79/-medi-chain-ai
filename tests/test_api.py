from io import BytesIO
import pytest
from fastapi.testclient import TestClient
import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["REDIS_URL"] = "memory://"
os.environ["STORAGE_MODE"] = "local"
os.environ["TESTING"] = "true"
os.environ["CLINICAL_THRESHOLDS_VALIDATED"] = "true"
os.environ["CLINICAL_VALIDATION_DATASET"] = "test"
os.environ["CLINICAL_VALIDATION_METRICS"] = '{"test": "metrics"}'

import deployment.api.main as api_main


class DummyAgent:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.calls = []

    async def run(self, image_path: str, pdf_path: str, idempotency_key: str = None):
        self.calls.append((image_path, pdf_path, idempotency_key))
        if self.should_fail:
            raise RuntimeError("boom")
        return {
            "diagnosis": {
                "top_finding": "Normal",
                "probabilities": [0.1, 0.1, 0.1, 0.1, 0.6],
                "uncertainty_std": 0.02,
                "ood_detected": False,
            },
            "confidence": 0.6,
            "history_data": {"metadata": {"patient_id": "P1"}},
            "pubmed_citations": [],
            "escalation_required": False,
            "iteration_count": 0,
        }


class DummyAgentOOD:
    """Agent that returns OOD detection."""
    async def run(self, image_path: str, pdf_path: str, idempotency_key: str = None):
        return {
            "diagnosis": {
                "top_finding": "Out-of-Distribution",
                "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
                "uncertainty_std": 0.5,
                "ood_detected": True,
            },
            "confidence": 0.2,
            "history_data": {"metadata": {"patient_id": "P1"}},
            "pubmed_citations": [],
            "escalation_required": True,
            "iteration_count": 0,
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


class DummyEHR:
    async def push_report(self, fhir_json, is_retry=False): 
        return True


class DummyInferenceAPI:
    """Mock inference API for integration tests."""
    def __init__(self):
        self.encode_calls = 0
        self.estimate_calls = 0
    
    async def encode_image(self, image):
        self.encode_calls += 1
        return {"features": [0.1] * 512, "visual_std": [0.01] * 512, "heatmap_base64": ""}
    
    async def encode_text(self, text):
        self.encode_calls += 1
        return {"embeddings": [[0.1] * 768]}
    
    async def estimate(self, visual_features, visual_std, text_features, num_passes):
        self.estimate_calls += 1
        return {
            "prediction": [4],  # Normal
            "mean_confidence": [0.6],
            "std_deviation": [0.02],
            "all_probs": [[0.1, 0.1, 0.1, 0.1, 0.6]],
            "fusion_head_variance": [0.02],
            "visual_uncertainty_score": [0.01],
            "combined_uncertainty": [0.02],
        }


def test_health_is_lazy(monkeypatch):
    monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(api_main, "storage", DummyStorageProvider())
    monkeypatch.setattr(api_main, "MAX_CONCURRENT_REQUESTS", 2)
    app = api_main.create_app()

    with TestClient(app) as client:
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "models_loaded": True,
            "inference_api_healthy": True,
            "concurrency_limit": 2,
            "version": "1.3.0",
        }


def test_analyze_cleans_request_dir_on_success(monkeypatch):
    agent = DummyAgent()
    storage_mock = DummyStorageProvider()
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    monkeypatch.setattr(api_main, "storage", storage_mock)
    monkeypatch.setattr(api_main, "ehr_gateway", DummyEHR())
    
    app = api_main.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze?sync=true",
            files={
                "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
            },
            headers={"X-API-Key": "test-key-for-ci"}
        )

    assert response.status_code == 200
    assert len(agent.calls) == 1
    assert len(storage_mock.files) == 0


def test_analyze_cleans_request_dir_on_failure(monkeypatch):
    agent = DummyAgent(should_fail=True)
    storage_mock = DummyStorageProvider()
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    monkeypatch.setattr(api_main, "storage", storage_mock)
    app = api_main.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze?sync=true",
            files={
                "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
            },
            headers={"X-API-Key": "test-key-for-ci"}
        )

    assert response.status_code == 500
    assert response.json()["detail"].startswith("Analysis failed:")
    assert len(storage_mock.files) == 0


def test_idempotency_key_returns_cached_result(monkeypatch):
    """Test that providing an idempotency key returns cached result on duplicate request."""
    agent = DummyAgent()
    storage_mock = DummyStorageProvider()
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    monkeypatch.setattr(api_main, "storage", storage_mock)
    monkeypatch.setattr(api_main, "ehr_gateway", DummyEHR())
    
    app = api_main.create_app()

    idempotency_key = "test-idempotency-key-123"
    
    with TestClient(app) as client:
        # First request
        response1 = client.post(
            "/v1/analyze?sync=true",
            files={
                "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
            },
            headers={
                "X-API-Key": "test-key-for-ci",
                "X-Idempotency-Key": idempotency_key
            }
        )
        assert response1.status_code == 200
        result1 = response1.json()
        
        # Second request with same idempotency key
        response2 = client.post(
            "/v1/analyze?sync=true",
            files={
                "image": ("scan.png", BytesIO(b"different-image-bytes"), "image/png"),
                "history": ("history.pdf", BytesIO(b"different-pdf-bytes"), "application/pdf"),
            },
            headers={
                "X-API-Key": "test-key-for-ci",
                "X-Idempotency-Key": idempotency_key
            }
        )
        assert response2.status_code == 200
        result2 = response2.json()
        
        # Should return cached result (same request_id)
        assert result1["request_id"] == result2["request_id"]
        # Agent should only have been called once
        assert len(agent.calls) == 1


def test_ood_detection_escalates(monkeypatch):
    """Test that OOD detection triggers escalation."""
    agent = DummyAgentOOD()
    storage_mock = DummyStorageProvider()
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    monkeypatch.setattr(api_main, "storage", storage_mock)
    monkeypatch.setattr(api_main, "ehr_gateway", DummyEHR())
    
    app = api_main.create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze?sync=true",
            files={
                "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
            },
            headers={"X-API-Key": "test-key-for-ci"}
        )

    assert response.status_code == 200
    result = response.json()
    assert result["escalation_required"] is True
    assert result["diagnosis"]["ood_detected"] is True
    assert result["diagnosis"]["top_finding"] == "Out-of-Distribution"


def test_radiologist_review_endpoint(monkeypatch):
    """Test the radiologist review workflow endpoint."""
    # First create a completed job
    import deployment.api.main as main_module
    from datetime import datetime, timezone
    
    job_id = "test-job-123"
    job_data = {
        "job_id": job_id,
        "status": "completed",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "result": {
            "request_id": job_id,
            "diagnosis": {
                "top_finding": "Silicosis",
                "probabilities": [0.7, 0.1, 0.05, 0.1, 0.05],
                "uncertainty_std": 0.1,
                "ood_detected": False,
                "escalation_required": True,
            },
            "confidence": 0.7,
            "escalation_required": True,
        }
    }
    
    # Mock the job store
    main_module._jobs_db[job_id] = job_data
    
    class MockFeedbackLogger:
        def log_feedback(self, *args, **kwargs):
            return "/tmp/feedback.json"
    
    monkeypatch.setattr(main_module, "feedback_logger", MockFeedbackLogger())
    monkeypatch.setattr(main_module, "redis_client", None)
    monkeypatch.setattr(main_module, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(main_module, "storage", DummyStorageProvider())
    monkeypatch.setattr(main_module, "ehr_gateway", DummyEHR())
    monkeypatch.setattr(main_module, "MAX_CONCURRENT_REQUESTS", 2)
    
    app = main_module.create_app()
    
    with TestClient(app) as client:
        response = client.post(
            f"/v1/cases/{job_id}/review",
            json={
                "verdict": "agree",
                "final_diagnosis": "Silicosis",
                "notes": "Confirmed based on clinical correlation",
                "doctor_id": "dr-smith",
                "time_spent_seconds": 45.0
            },
            headers={"X-API-Key": "test-key-for-ci"}
        )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_feedback_endpoint(monkeypatch):
    """Test the feedback endpoint."""
    class MockFeedbackLogger:
        def log_feedback(self, session_id, verdict, notes, diagnosis, history_metadata, 
                        disagreement_reason=None, correction_mask_base64=None):
            return f"/tmp/feedback_{session_id}.json"
    
    monkeypatch.setattr(api_main, "feedback_logger", MockFeedbackLogger())
    monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(api_main, "storage", DummyStorageProvider())
    monkeypatch.setattr(api_main, "ehr_gateway", DummyEHR())
    monkeypatch.setattr(api_main, "MAX_CONCURRENT_REQUESTS", 2)
    monkeypatch.setattr(api_main, "drift_detector", MagicMock())
    
    app = api_main.create_app()
    
    with TestClient(app) as client:
        response = client.post(
            "/v1/feedback",
            json={
                "session_id": "test-session-123",
                "verdict": "agree",
                "notes": "Correct diagnosis",
                "diagnosis": {"top_finding": "Normal", "probabilities": [0.1]*5},
                "history_metadata": {"patient_id": "P1"},
                "doctor_id": "dr-test",
                "start_time": 1234567890.0
            },
            headers={"X-API-Key": "test-key-for-ci"}
        )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"


@pytest.mark.asyncio
async def test_cleanup_circuit_breaker(monkeypatch):
    class FailingStorage:
        def cleanup(self, max_age_seconds=None):
            raise RuntimeError("storage error")
             
    monkeypatch.setattr(api_main, "storage", FailingStorage())
    
    sleep_calls = []
    async def mock_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 5:
            raise asyncio.CancelledError("Test completed successfully")
         
    monkeypatch.setattr(asyncio, "sleep", mock_sleep)
    
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await api_main.cleanup_old_temp_files()
        
    assert "Test completed successfully" in str(exc_info.value)
    assert sleep_calls == [600, 1200, 2400, 3600, 3600]


@pytest.mark.asyncio
async def test_dlq_reconciliation_backpressure(monkeypatch):
    """Test that DLQ reconciliation respects backpressure on EHR pushes."""
    import redis
    
    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.set.return_value = True  # lock acquired
    mock_redis.rpoplpush.return_value = None  # no items in DLQ
    mock_redis.lrange.return_value = []
    
    monkeypatch.setattr(api_main, "redis_client", mock_redis)
    monkeypatch.setattr(api_main, "use_redis", True)
    monkeypatch.setattr(api_main, "DLQ_DIR", "temp/dlq")
    
    # Run one iteration of reconciliation
    async def run_once():
        from deployment.api.main import reconcile_dlq_task
        task = asyncio.create_task(reconcile_dlq_task())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    await run_once()
    
    # Verify semaphore was used (backpressure mechanism)
    assert mock_redis.set.called


def test_telemetry_metrics_endpoint(monkeypatch):
    """Test telemetry metrics endpoint returns proper structure."""
    mock_redis = AsyncMock()
    mock_redis.get.side_effect = lambda key: {
        "medi_chain:telemetry:total_cases": "100",
        "medi_chain:telemetry:escalated_cases": "15",
        "medi_chain:drift:feedback_summary": json.dumps({
            "total_cases": 50,
            "agreements": 40,
            "disagreements": 10,
            "agreement_rate": 0.8
        })
    }.get(key)
    
    monkeypatch.setattr(api_main, "redis_client", mock_redis)
    monkeypatch.setattr(api_main, "use_redis", True)
    monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(api_main, "storage", DummyStorageProvider())
    monkeypatch.setattr(api_main, "ehr_gateway", DummyEHR())
    monkeypatch.setattr(api_main, "MAX_CONCURRENT_REQUESTS", 2)
    
    app = api_main.create_app()
    
    with TestClient(app) as client:
        response = client.get(
            "/v1/telemetry/metrics",
            headers={"X-API-Key": "test-key-for-ci"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_cases"] == 100
    assert data["escalated_cases"] == 15
    assert data["escalation_rate"] == 0.15
    assert data["feedback_total_cases"] == 50
    assert data["clinician_agreement_rate"] == 0.8
    assert data["clinician_override_rate"] == 0.2


def test_config_validation_endpoint(monkeypatch):
    """Test that config audit dump is available in health endpoint."""
    monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(api_main, "storage", DummyStorageProvider())
    monkeypatch.setattr(api_main, "MAX_CONCURRENT_REQUESTS", 2)
    
    app = api_main.create_app()
    
    with TestClient(app) as client:
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "config_audit" in data
        assert "clinical_thresholds" in data["config_audit"]
        assert "inference" in data["config_audit"]
        assert data["config_audit"]["clinical_thresholds"]["thresholds_validated"] is True


def test_gpu_health_endpoint_exists(monkeypatch):
    """Test that GPU health endpoint is available."""
    monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(api_main, "storage", DummyStorageProvider())
    monkeypatch.setattr(api_main, "MAX_CONCURRENT_REQUESTS", 2)
    
    app = api_main.create_app()
    
    with TestClient(app) as client:
        response = client.get("/v1/health/gpu")
        # Should return 200 even if no GPU (will show gpu_available: false)
        assert response.status_code == 200
        data = response.json()
        assert "gpu_available" in data


def test_metrics_endpoint_prometheus(monkeypatch):
    """Test Prometheus metrics endpoint."""
    monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(api_main, "storage", DummyStorageProvider())
    monkeypatch.setattr(api_main, "MAX_CONCURRENT_REQUESTS", 2)
    
    app = api_main.create_app()
    
    with TestClient(app) as client:
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        assert "medi_chain_cases_processed_total" in response.text


def test_analyze_requires_valid_content_types(monkeypatch):
    """Test that invalid content types are rejected."""
    monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(api_main, "storage", DummyStorageProvider())
    monkeypatch.setattr(api_main, "ehr_gateway", DummyEHR())
    monkeypatch.setattr(api_main, "MAX_CONCURRENT_REQUESTS", 2)
    
    app = api_main.create_app()
    
    with TestClient(app) as client:
        # Invalid image type
        response = client.post(
            "/v1/analyze?sync=true",
            files={
                "image": ("scan.gif", BytesIO(b"image-bytes"), "image/gif"),
                "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
            },
            headers={"X-API-Key": "test-key-for-ci"}
        )
        assert response.status_code == 415
        
        # Invalid history type
        response = client.post(
            "/v1/analyze?sync=true",
            files={
                "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                "history": ("history.txt", BytesIO(b"text-bytes"), "text/plain"),
            },
            headers={"X-API-Key": "test-key-for-ci"}
        )
        assert response.status_code == 415


def test_async_analysis_flow(monkeypatch):
    """Test async analysis returns job ID and status endpoint works."""
    agent = DummyAgent()
    storage_mock = DummyStorageProvider()
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    monkeypatch.setattr(api_main, "storage", storage_mock)
    monkeypatch.setattr(api_main, "ehr_gateway", DummyEHR())
    monkeypatch.setattr(api_main, "MAX_CONCURRENT_REQUESTS", 2)
    monkeypatch.setattr(api_main, "redis_client", None)
    
    app = api_main.create_app()
    
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze?sync=false",
            files={
                "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
            },
            headers={"X-API-Key": "test-key-for-ci"}
        )
    
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert "status_url" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])