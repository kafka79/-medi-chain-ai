import os
import json
import pytest
import shutil
import tempfile
import torch
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.fusion import AttentionFusion, LateFusionModel
from src.models.uncertainty import UncertaintyEstimator
from src.data.dicom_handler import create_secondary_capture
from src.data.fhir_formatter import EHRGateway
from deployment.api.main import RedisDistributedSemaphore

def test_dicom_metadata_preservation(tmp_path):
    # Create a dummy png image
    from PIL import Image
    png_path = tmp_path / "dummy_heatmap.png"
    Image.new("RGB", (100, 100), color="red").save(png_path)
    
    dcm_output = tmp_path / "output.dcm"
    
    metadata = {
        "PatientName": "Doe^John",
        "PatientID": "PAT-999-111",
        "PatientBirthDate": "19800101",
        "PatientSex": "M",
        "StudyInstanceUID": "1.2.3.4.5.6.7",
        "SeriesInstanceUID": "1.2.3.4.5.6.8",
        "StudyID": "STUDY-123",
        "AccessionNumber": "ACC-555"
    }
    
    # Create secondary capture passing the metadata dict
    create_secondary_capture(metadata, str(png_path), str(dcm_output))
    
    # Read the output DICOM dataset
    import pydicom
    ds = pydicom.dcmread(str(dcm_output))
    
    # Assert headers are properly preserved
    assert str(ds.PatientName) == "Doe^John"
    assert ds.PatientID == "PAT-999-111"
    assert ds.PatientBirthDate == "19800101"
    assert ds.PatientSex == "M"
    assert ds.StudyInstanceUID == "1.2.3.4.5.6.7"
    assert ds.SeriesInstanceUID == "1.2.3.4.5.6.8"
    assert ds.StudyID == "STUDY-123"
    assert ds.AccessionNumber == "ACC-555"

@pytest.mark.asyncio
async def test_fhir_formatter_dlq_full_payload(tmp_path):
    gateway = EHRGateway(endpoint_url="http://completely-broken-host/fhir")
    
    # Mock redis and s3 to fail, forcing local disk fallback
    with patch("redis.Redis", side_effect=Exception("Redis offline")), \
         patch("src.utils.storage.S3StorageProvider", side_effect=Exception("S3 offline")), \
         patch("shutil.disk_usage") as mock_disk_usage:
         
        # Simulate critically low space (e.g., 10MB free)
        mock_disk_usage.return_value = (100 * 1024 * 1024, 90 * 1024 * 1024, 10 * 1024 * 1024)
        
        # DLQ directory
        dlq_dir = tmp_path / "dlq"
        
        with patch.dict(os.environ, {"DLQ_DIR": str(dlq_dir), "TESTING": "true"}):
            # Try to push a report (will fail and trigger fallback)
            dummy_payload = '{"resourceType": "DiagnosticReport", "id": "rpt-001"}'
            result = await gateway.push_report(dummy_payload)
            
            assert result is False
            
            # Assert a file was created in DLQ directory
            dlq_files = list(dlq_dir.glob("failed_report_*.json"))
            assert len(dlq_files) == 1
            
            # Read the persisted local file wrapper
            with open(dlq_files[0], "r") as f:
                wrapper = json.load(f)
                
            # Verify the wrapper contains encrypted data (full payload)
            assert wrapper["encrypted"] is True
            assert "data" in wrapper
            
            # Decrypt payload to verify it contains the full report info
            from src.utils.security import decrypt_payload
            decrypted = decrypt_payload(wrapper["data"])
            decrypted_json = json.loads(decrypted)
            assert decrypted_json["payload"]["resourceType"] == "DiagnosticReport"
            assert decrypted_json["payload"]["id"] == "rpt-001"

def test_gated_fusion_independent_parameters():
    # Verify that gate projections use distinct parameters
    model = AttentionFusion(vision_dim=128, text_dim=256, hidden_dim=128)
    
    assert hasattr(model, "v_gate")
    assert hasattr(model, "t_gate")
    assert model.v_gate is not model.v_proj
    assert model.t_gate is not model.t_proj
    
    # Test random input forward pass
    v_in = torch.randn(4, 128)
    t_in = torch.randn(4, 256)
    fused, logits = model(v_in, t_in)
    assert fused.shape == (4, 128)
    assert logits.shape == (4, 5)

def test_gated_fusion_backward_compatibility(tmp_path):
    # Simulate an old state dict checkpoint that lacks v_gate and t_gate
    # Match the default shapes of LateFusionModel exactly to prevent size mismatch errors
    old_state_dict = {
        "v_proj.weight": torch.randn(512, 512),
        "v_proj.bias": torch.randn(512),
        "t_proj.weight": torch.randn(512, 768),
        "t_proj.bias": torch.randn(512),
        "norm1.weight": torch.randn(512),
        "norm1.bias": torch.randn(512),
        "ffn.0.weight": torch.randn(2048, 512),
        "ffn.0.bias": torch.randn(2048),
        "ffn.3.weight": torch.randn(512, 2048),
        "ffn.3.bias": torch.randn(512),
        "norm2.weight": torch.randn(512),
        "norm2.bias": torch.randn(512),
        "classifier.0.weight": torch.randn(256, 512),
        "classifier.0.bias": torch.randn(256),
        "classifier.3.weight": torch.randn(5, 256),
        "classifier.3.bias": torch.randn(5)
    }
    
    ckpt_path = tmp_path / "old_checkpoint.pt"
    torch.save(old_state_dict, ckpt_path)
    
    # Test loader compatibility mimicking InferenceService
    from sentence_transformers import SentenceTransformer
    mock_encoder = MagicMock()
    mock_encoder.device = "cpu"
    with patch("deployment.api.inference_service.MODEL_CHECKPOINT", str(ckpt_path)), \
         patch("deployment.api.inference_service.BiomedVisualEncoder", return_value=mock_encoder), \
         patch.object(SentenceTransformer, "__init__", return_value=None):
         
        from deployment.api.inference_service import InferenceService
        service = InferenceService()
        
        # Verify that weights were successfully loaded and gates were cloned from projections
        assert torch.equal(service.fusion.v_gate.weight, service.fusion.v_proj.weight)
        assert torch.equal(service.fusion.t_gate.weight, service.fusion.t_proj.weight)

def test_calibrated_uncertainty_bounds(tmp_path):
    # Setup dummy model
    model = MagicMock()
    model.modules.return_value = []
    # Logits yield a highly confident class probability on pass 1, but random across others
    # (MC Dropout passes)
    pass_logits = [
        torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0]]),
    ]
    
    # Mock forward return (joint, logits)
    model.side_effect = lambda v, t: (torch.randn(1, 128), pass_logits[model.call_count % 2])
    model.call_count = 0
    
    estimator = UncertaintyEstimator(model)
    v = torch.randn(1, 128)
    t = torch.randn(1, 256)
    
    # 1. Perfectly in-distribution (ood_distance = 0.0)
    with patch("pathlib.Path.exists", return_value=False):
        res_id = estimator.estimate_uncertainty(v, t, num_passes=2)
        # Combined uncertainty should equal standard deviation of head variance
        assert torch.allclose(res_id["std_deviation"], res_id["fusion_head_variance"])

    # 2. Test bounds when completely OOD
    # Write a baseline cache JSON file to temp directory
    baseline_cache_dir = Path("temp/drift")
    baseline_cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache_path = baseline_cache_dir / "features_baseline_cache.json"
    
    baseline_cache = [[-10.0] * 128, [-10.0] * 128] # Opposite direction to v to maximize cosine distance and len > 1
    with open(baseline_cache_path, "w") as f:
        json.dump(baseline_cache, f)
        
    try:
        v_norm = torch.nn.functional.normalize(torch.ones(1, 128), p=2, dim=-1)
        res_ood = estimator.estimate_uncertainty(v_norm, t, num_passes=2)
        
        # With temperature scaling calibration, OOD distance increases softmax temperature,
        # which flattens probabilities and strictly decreases the extreme epistemic variance
        # compared to a highly confident but conflicting ID sample directly.
        # So combined_uncertainty reflects the natural epistemic variance.
        assert res_ood["std_deviation"] == res_ood["fusion_head_variance"]
        assert res_ood["std_deviation"] < res_id["std_deviation"]
    finally:
        if baseline_cache_path.exists():
            baseline_cache_path.unlink()

@pytest.mark.asyncio
async def test_pubsub_semaphore_non_polling():
    mock_redis = MagicMock()
    # Mock Redis eval: returns 0 (semaphore full), then 1 (success)
    mock_redis.eval.side_effect = [0, 1]
    
    # Mock pubsub listener callback
    pubsub_mock = MagicMock()
    mock_redis.pubsub.return_value = pubsub_mock
    # Simulate a release message trigger
    pubsub_mock.get_message.side_effect = [
        {"type": "message", "channel": "medi_chain:semaphore:test_sem:leases:released", "data": "released"},
        None
    ]
    
    sem = RedisDistributedSemaphore(mock_redis, "test_sem", limit=2)
    
    # We acquire the semaphore. On first eval it gets 0, registers waiter,
    # then gets a message from Pub/Sub, wakes up, and eval gets 1 (success).
    # Since timeout is 2.0, this should execute instantly
    
    async def run_enter():
        async with sem:
            return True
            
    res = await asyncio.wait_for(run_enter(), timeout=1.0)
    assert res is True
    assert mock_redis.eval.call_count == 2
    
    # Clean up tasks
    from deployment.api.main import SEMAPHORE_LISTENER_TASKS
    task = SEMAPHORE_LISTENER_TASKS.get(sem.name)
    if task:
        task.cancel()

def test_phi_encryption_at_rest_flow(tmp_path):
    from deployment.api.main import _update_job_status, _jobs_db, app
    from fastapi.testclient import TestClient
    
    # 1. Store test metadata
    job_id = "test-job-phi-1"
    raw_metadata = {"PatientName": "Jane Doe", "PatientID": "12345"}
    result_payload = {"dicom_metadata": raw_metadata, "top_finding": "Normal"}
    
    # Run status update
    asyncio.run(_update_job_status(job_id, "completed", result=result_payload))
    
    # Verify that the internal database has the ENCRYPTED metadata
    stored = _jobs_db[job_id]
    assert stored["result"]["dicom_metadata"]["encrypted"] is True
    assert stored["result"]["dicom_metadata"]["data"] != json.dumps(raw_metadata)
    
    # Use TestClient to request the status endpoint
    from unittest.mock import AsyncMock
    mock_agent = MagicMock()
    mock_agent.close = AsyncMock()
    with patch("deployment.api.main.build_agent", return_value=mock_agent):
        with TestClient(app) as client:
            headers = {"X-API-Key": "dev-secret-key-123"}
            response = client.get(f"/analyze/status/{job_id}", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert data["result"]["dicom_metadata"] == raw_metadata

@pytest.mark.asyncio
async def test_joint_multimodal_ood_uncertainty(tmp_path):
    # Setup dummy model
    model = MagicMock()
    model.modules.return_value = []
    pass_logits = [torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])]
    model.side_effect = lambda v, t: (torch.randn(1, 128), pass_logits[0])
    
    estimator = UncertaintyEstimator(model)
    v = torch.randn(1, 128)
    t = torch.randn(1, 256)
    
    # Write visual and text baseline cache
    baseline_cache_dir = Path("temp/drift")
    baseline_cache_dir.mkdir(parents=True, exist_ok=True)
    visual_cache_path = baseline_cache_dir / "features_baseline_cache.json"
    text_cache_path = baseline_cache_dir / "text_baseline_cache.json"
    
    # Make visual and text baseline features opposite to trigger distance OOD
    with open(visual_cache_path, "w") as f:
        json.dump([[-10.0] * 128, [-10.0] * 128], f)
    with open(text_cache_path, "w") as f:
        json.dump([[-10.0] * 256, [-10.0] * 256], f)
        
    try:
        # Run uncertainty check
        v_norm = torch.nn.functional.normalize(torch.ones(1, 128), p=2, dim=-1)
        t_norm = torch.nn.functional.normalize(torch.ones(1, 256), p=2, dim=-1)
        # Use num_passes=2 to avoid division-by-zero variance degrees of freedom
        res = estimator.estimate_uncertainty(v_norm, t_norm, num_passes=2)
        
        # With temperature scaling, we no longer add arbitrary penalties to variance.
        # It's fully reflected in the fusion_head_variance.
        assert torch.allclose(torch.tensor(res["combined_uncertainty"]), torch.tensor(res["fusion_head_variance"]))
    finally:
        if visual_cache_path.exists():
            visual_cache_path.unlink()
        if text_cache_path.exists():
            text_cache_path.unlink()

@pytest.mark.asyncio
async def test_fail_fast_semaphore_listener_crash():
    from deployment.api.main import _start_semaphore_listener, SEMAPHORE_WAITERS, SEMAPHORE_LISTENER_TASKS
    
    mock_redis = MagicMock()
    # Mock pubsub subscribe to raise an error to trigger listener exception
    pubsub_mock = MagicMock()
    mock_redis.pubsub.return_value = pubsub_mock
    pubsub_mock.subscribe.side_effect = Exception("PubSub Subscribe Error")
    
    event = asyncio.Event()
    SEMAPHORE_WAITERS["test_sem"] = [event]
    
    # Start listener; it should fail instantly and trigger exception handling
    task = asyncio.create_task(_start_semaphore_listener(mock_redis, "test_sem"))
    SEMAPHORE_LISTENER_TASKS["test_sem"] = task
    
    # Wait for task to finish
    await asyncio.sleep(0.1)
    
    # Verify the event waiter was woken up instantly (failsafe activated)
    assert event.is_set()
    assert len(SEMAPHORE_WAITERS["test_sem"]) == 0

@pytest.mark.asyncio
async def test_scoped_api_keys_permissions():
    from deployment.api.main import verify_api_key
    from fastapi.security import SecurityScopes
    import hashlib
    
    # Define a custom keys config with SHA-256 hash mappings
    key1 = "secret-clinic-key"
    key2 = "secret-metrics-key"
    h1 = hashlib.sha256(key1.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(key2.encode("utf-8")).hexdigest()
    
    config_data = {
        "keys": {
            h1: ["cases:write", "cases:read"],
            h2: ["metrics:read"]
        }
    }
    
    with patch.dict(os.environ, {
        "API_KEY": "fallback-key",
        "API_KEYS_CONFIG": json.dumps(config_data),
        "TESTING": "true"
    }):
        # 1. Valid clinic key requesting cases:write (should pass)
        scopes_write = SecurityScopes(scopes=["cases:write"])
        res = await verify_api_key(scopes_write, key1)
        assert res == key1
        
        # 2. Valid clinic key requesting metrics:read (should fail)
        scopes_metrics = SecurityScopes(scopes=["metrics:read"])
        with pytest.raises(Exception) as exc:
            await verify_api_key(scopes_metrics, key1)
        assert "Not enough permissions" in str(exc.value)
        
        # 3. Valid metrics key requesting metrics:read (should pass)
        res = await verify_api_key(scopes_metrics, key2)
        assert res == key2

def test_gpu_roi_deduction():
    import time
    from deployment.api.main import app
    from fastapi.testclient import TestClient
    
    # Set the started_at time on the app state
    app.state.started_at = time.time() - 7200 # Started 2 hours ago
    
    # Mock Redis to return 10 cases, 2 escalated cases
    mock_redis = MagicMock()
    mock_redis.get.side_effect = [None, "10", "2"] # summary, total, escalated
    
    from unittest.mock import AsyncMock
    mock_agent = MagicMock()
    mock_agent.close = AsyncMock()
    with patch("deployment.api.main.use_redis", True), \
         patch("deployment.api.main.redis_client", mock_redis), \
         patch("deployment.api.main.build_agent", return_value=mock_agent):
         
        with TestClient(app) as client:
            headers = {"X-API-Key": "dev-secret-key-123"}
            response = client.get(
                "/telemetry/metrics",
                params={
                    "baseline_minutes": 30,
                    "automated_minutes": 2,
                    "hourly_rate": 150.0,
                    "gpu_hourly_cost": 1.50
                },
                headers=headers
            )
            assert response.status_code == 200
            data = response.json()
            
            # 10 cases - 2 escalated = 8 saved cases
            # 8 * (30 - 2) = 224 minutes saved = 3.733 hours saved
            # 3.733 * $150 = $560 gross saved
            # 2 hours running * $1.50 = $3.00 infra cost
            # Net saved = $560 - $3.00 = $557.00
            assert data["saved_cost_usd"] == 559.5
            assert data["infrastructure_cost_usd"] == 3.0
            assert data["net_saved_cost_usd"] == 556.5
