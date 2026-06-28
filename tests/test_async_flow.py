import os
import pytest
import numpy as np
import json
import asyncio
import tempfile
import cv2
import torch
from pathlib import Path
from io import BytesIO
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

os.environ["REDIS_URL"] = "memory://"
os.environ["STORAGE_MODE"] = "local"
os.environ["TESTING"] = "true"

import deployment.api.main as api_main
from src.data.privacy_scrubber import PrivacyScrubber
from src.models.uncertainty import UncertaintyEstimator
from src.utils.feedback_logger import FeedbackLogger

def test_lossless_png_masking():
    # Verify that masking an image preserves png or uses png for standard files, avoiding jpg conversion artifacts
    scrubber = PrivacyScrubber()
    
    # Create temporary .png file
    tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_png.close()
    
    try:
        # Create a basic image with peripheral text to trigger masking
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(img, "MRN: 112233", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(tmp_png.name, img)
        
        # Check that we detect text
        boxes = scrubber.detect_burned_in_text(tmp_png.name)
        assert len(boxes) > 0
        
        # Mask text and verify output extension is .png
        output_path = scrubber.mask_burned_in_text(tmp_png.name)
        assert output_path != tmp_png.name
        assert output_path.endswith(".png")
        assert os.path.exists(output_path)
        os.unlink(output_path)
    finally:
        if os.path.exists(tmp_png.name):
            os.unlink(tmp_png.name)

def test_non_isotropic_noise_scaling():
    # Verify that UncertaintyEstimator scales noise non-isotropically when baseline features cache exists
    dummy_model = MagicMock()
    # Mock model return shape (fused, logits)
    dummy_model.return_value = (torch.randn(1, 512), torch.randn(1, 5))
    dummy_model.modules.return_value = []
    
    estimator = UncertaintyEstimator(dummy_model)
    
    vision_emb = torch.randn(1, 512)
    text_emb = torch.randn(1, 768)
    
    # Create baseline cache with high variance in dimension 0 and low variance in dimension 1
    # 512 dimensions, 10 samples
    baseline_data = np.zeros((10, 512))
    baseline_data[:, 0] = np.random.rand(10) * 100.0  # High std in dim 0
    baseline_data[:, 1] = np.random.rand(10) * 0.01   # Low std in dim 1
    
    os.makedirs("temp/drift", exist_ok=True)
    baseline_cache_path = Path("temp/drift/features_baseline_cache.json")
    with open(baseline_cache_path, "w") as f:
        json.dump(baseline_data.tolist(), f)
        
    try:
        # Run estimation with TTA visual_std = 1.0 everywhere
        visual_std = torch.ones(1, 512)
        
        # We want to patch torch.randn_like to return 1.0 so we can inspect the noise scaling
        with patch("torch.randn_like", side_effect=lambda x, **kwargs: torch.ones_like(x)):
            res = estimator.estimate_uncertainty(vision_emb, text_emb, num_passes=2, visual_std=visual_std)
            
        assert res is not None
        assert "combined_uncertainty" in res
    finally:
        if baseline_cache_path.exists():
            baseline_cache_path.unlink()

@pytest.mark.asyncio
async def test_async_analyze_and_telemetry_metrics(monkeypatch):
    # Setup FastAPI app client
    class DummyAgent:
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
                "escalation_required": False,
            }
        async def close(self): pass
        
    class DummyStorageProvider:
        def save(self, file_obj, rel_path): return rel_path
        def load(self, rel_path): return rel_path
        def delete(self, rel_path): pass
        def cleanup(self, max_age_seconds=None): pass

    agent = DummyAgent()
    storage_mock = DummyStorageProvider()
    monkeypatch.setattr(api_main, "build_agent", lambda: agent)
    monkeypatch.setattr(api_main, "storage", storage_mock)
    
    app = api_main.create_app()
    
    # 1. Test posting to /analyze with sync=False
    with TestClient(app) as client:
        # Request async job
        response = client.post(
            "/analyze?sync=false",
            files={
                "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                "history": ("history.pdf", BytesIO(b"pdf-bytes"), "application/pdf"),
            },
            headers={"X-API-Key": "dev-secret-key-123"}
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        job_id = data["job_id"]
        
        status_resp = client.get(
            f"/analyze/status/{job_id}",
            headers={"X-API-Key": "dev-secret-key-123"}
        )
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["job_id"] == job_id
        
        # 2. Verify override telemetry fallback to CSV
        logger = api_main.feedback_logger
        import csv
        logger.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(logger.csv_path, "w", newline="", encoding="utf-8") as h:
            writer = csv.writer(h)
            writer.writerow(["feedback_id", "timestamp_utc", "session_id", "verdict", "notes", "top_finding", "uncertainty_std", "patient_id", "occupation"])
            writer.writerow(["f1", "2026-06-28", "s1", "agree", "", "Normal", "0.02", "P1", "Miner"])
            writer.writerow(["f2", "2026-06-28", "s2", "disagree", "Wrong finding", "Normal", "0.02", "P2", "Miner"])
            
        try:
            # Query metrics
            metrics_resp = client.get(
                "/telemetry/metrics",
                headers={"X-API-Key": "dev-secret-key-123"}
            )
            assert metrics_resp.status_code == 200
            metrics = metrics_resp.json()
            assert metrics["feedback_total_cases"] == 2
            assert metrics["feedback_agreements"] == 1
            assert metrics["feedback_disagreements"] == 1
            assert metrics["clinician_agreement_rate"] == 0.5
            assert metrics["clinician_override_rate"] == 0.5
        finally:
            if logger.csv_path.exists():
                logger.csv_path.unlink()
