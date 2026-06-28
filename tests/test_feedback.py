from io import BytesIO
import pytest
from fastapi.testclient import TestClient
import os

import deployment.api.main as api_main

class DummyAgent:
    def __init__(self, escalation=False):
        self.escalation = escalation

    async def run(self, image_path: str, pdf_path: str):
        return {
            "diagnosis": {
                "top_finding": "Normal",
                "probabilities": [0.1, 0.1, 0.1, 0.1, 0.6],
                "uncertainty_std": 0.02,
            },
            "confidence": 0.6,
            "history_data": {"metadata": {"patient_id": "P1"}},
            "pubmed_citations": [],
            "escalation_required": self.escalation,
        }

def test_analyze_escalation_header(monkeypatch, mock_inference_api):
    # Set up agent that triggers escalation
    agent = DummyAgent(escalation=True)
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    
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
        assert response.json()["escalation_required"] is True
        assert "X-Requires-Human-Review" not in response.headers

def test_analyze_no_escalation_header(monkeypatch, mock_inference_api):
    # Set up agent that does not trigger escalation
    agent = DummyAgent(escalation=False)
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    
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
        assert response.json()["escalation_required"] is False
        assert "X-Requires-Human-Review" not in response.headers

def test_feedback_rate_limiting_and_validation(monkeypatch, mock_redis):
    app = api_main.create_app()
    
    # Mock feedback_logger to not do real file writes
    class DummyFeedbackLogger:
        def log_feedback(self, **kwargs):
            return "mocked_path.csv"
    monkeypatch.setattr(api_main, "feedback_logger", DummyFeedbackLogger())

    with TestClient(app) as client:
        payload = {
            "session_id": "session-123",
            "verdict": "agree",
            "notes": "Looks good",
            "diagnosis": {"top_finding": "Normal"},
            "history_metadata": {"patient_id": "P1"},
            "doctor_id": "dr-smith"
        }
        
        # Test valid request
        response = client.post(
            "/feedback",
            json=payload,
            headers={"X-API-Key": "dev-secret-key-123"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # Test invalid doctor_id format (must start with 'dr-')
        bad_payload = payload.copy()
        bad_payload["doctor_id"] = "smith"  # invalid, doesn't start with dr-
        response_bad = client.post(
            "/feedback",
            json=bad_payload,
            headers={"X-API-Key": "dev-secret-key-123"}
        )
        assert response_bad.status_code == 422
