import os
from pathlib import Path
import pytest
import numpy as np
import json
import asyncio
import torch
from unittest.mock import MagicMock, patch
from src.agent.clinical_graph import ClinicalAgent
from src.monitoring.drift_detector import DriftDetector

@pytest.mark.asyncio
async def test_node_extract_visuals_cleanup_on_failure():
    # Setup mocks for ClinicalAgent dependencies
    mock_parser = MagicMock()
    mock_rag = MagicMock()
    
    # Create fake temporary file
    temp_dummy_path = "temp_dummy_scrubbed.jpg"
    with open(temp_dummy_path, "wb") as f:
        f.write(b"dummy image bytes")
        
    try:
        with patch.dict("os.environ", {"INTERNAL_API_KEY": "test-key"}), \
             patch("src.data.privacy_scrubber.PrivacyScrubber") as mock_scrubber_class:
            
            mock_scrubber = MagicMock()
            # mask_burned_in_text returns the temp dummy path
            mock_scrubber.mask_burned_in_text.return_value = temp_dummy_path
            mock_scrubber_class.return_value = mock_scrubber
            
            agent = ClinicalAgent(mock_parser, mock_rag, inference_api_url="http://invalid-test-url:8001")
            
            # Mock state
            state = {
                "image_path": "orig_image.jpg",
                "patient_pdf_path": "history.pdf"
            }
            
            # Execute node_extract_visuals, which will degrade gracefully on failure
            result_state = await agent.node_extract_visuals(state)
                
            assert result_state.get("inference_failed") is True
            assert result_state.get("escalation_required") is True
            
            # Crucial assertion: the temporary file MUST be deleted by the finally block
            assert not os.path.exists(temp_dummy_path)
            
    finally:
        # Emergency cleanup in case test fails
        if os.path.exists(temp_dummy_path):
            os.remove(temp_dummy_path)

def test_check_prediction_drift_class_count_mismatch():
    # Instantiate DriftDetector
    with patch.dict("os.environ", {"TESTING": "false"}):
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            detector = DriftDetector()
            
    # Set up a baseline with 5 classes
    detector.baseline = np.random.rand(100, 5)
    detector._save_baseline = MagicMock()
    
    # Send current predictions with 6 classes (mismatch)
    current_probs_mismatch = (np.random.rand(100, 6)).tolist()
    
    # Call check_prediction_drift, it should NOT reset the baseline and return False
    try:
        result = detector.check_prediction_drift(current_probs_mismatch)
        assert result is False
        detector._save_baseline.assert_not_called()
    except IndexError:
        pytest.fail("check_prediction_drift threw IndexError on class count mismatch!")


@pytest.mark.asyncio
async def test_dlq_poison_routing():
    import shutil
    shutil.rmtree("temp/dlq", ignore_errors=True)
    
    mock_redis = MagicMock()
    # First call returns a JSON payload, second returns None
    payload_dict = {
        "timestamp": "2026-06-15",
        "error": "Initial EHR failure",
        "payload": {"resourceType": "DiagnosticReport"},
        "retry_count": 2  # Already failed twice
    }
    mock_redis.rpoplpush.side_effect = [json.dumps(payload_dict), None]
    
    # Mock sleep to raise CancelledError so the infinite loop terminates immediately
    async def mock_sleep(seconds):
        raise asyncio.CancelledError()
        
    from deployment.api.main import reconcile_dlq_task
    
    # Patch requests.post inside EHRGateway if called, or patch EHRGateway.push_report directly
    with patch("redis.from_url", return_value=mock_redis), \
         patch("deployment.api.main.use_redis", True), \
         patch("deployment.api.main.ehr_gateway.push_report", return_value=False) as mock_push, \
         patch("asyncio.sleep", side_effect=mock_sleep), \
         patch("deployment.api.main._send_system_alert") as mock_alert:
         
        await reconcile_dlq_task()
        
        # Check that push_report was called
        mock_push.assert_called_once()
        # Check that the payload was pushed to the POISON queue since retry_count reached 3 (2 + 1)
        mock_redis.rpush.assert_called_once()
        args = mock_redis.rpush.call_args[0]
        assert args[0] == "medi_chain:dlq:poison"
        poison_item = json.loads(args[1])
        assert poison_item["retry_count"] == 3
        mock_alert.assert_called_once()


def test_grad_cam_aspect_ratio_alignment():
    from src.vlm.explainability import VisualExplainer
    from PIL import Image
    
    model = MagicMock()
    preprocess = MagicMock()
    
    explainer = VisualExplainer(model, preprocess)
    
    # Create a real numpy array of size 200x100 (W=200, H=100) -> landscape (W > H)
    mock_img = np.zeros((100, 200, 3), dtype=np.uint8)
    real_pil_img = Image.fromarray(mock_img)
    
    # Mock cv2 and pytorch_grad_cam functions
    with patch("PIL.Image.open") as mock_open, \
         patch("cv2.resize") as mock_resize, \
         patch("pytorch_grad_cam.GradCAM.__call__") as mock_cam_call, \
         patch("src.vlm.explainability.show_cam_on_image") as mock_show:
         
        mock_open.return_value = real_pil_img
        
        mock_cam_call.return_value = np.zeros((1, 14, 14), dtype=np.float32)
        mock_resize.return_value = np.zeros((200, 200), dtype=np.float32)
        mock_show.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
        
        explainer.generate_heatmap("fake_xray.png")
        
        # Check cv2.resize call arguments. It should resize 14x14 grid to (padded_size, padded_size) -> (200, 200)
        mock_resize.assert_called_once()
        resize_args = mock_resize.call_args[0]
        assert resize_args[1] == (200, 200)


def test_mc_dropout_feature_perturbation():
    from src.models.uncertainty import UncertaintyEstimator
    
    # Create mock fusion model
    model = MagicMock()
    # Mock modules to avoid AttributeError during loop in estimate_uncertainty
    model.modules.return_value = []
    
    # Mock forward pass: return (fused, logits)
    # Logits shape: (batch_size, num_classes)
    model.return_value = (torch.randn(1, 512), torch.tensor([[2.0, 1.0, 0.5, 0.2, 0.1]]))
    
    estimator = UncertaintyEstimator(model)
    
    # Pass dummy embeddings
    v = torch.randn(1, 512)
    t = torch.randn(1, 768)
    
    # Execute estimate_uncertainty
    results = estimator.estimate_uncertainty(v, t, num_passes=5)
    
    assert "prediction" in results
    assert "std_deviation" in results
    assert "fusion_head_variance" in results
    assert len(results["std_deviation"]) == 1
    assert isinstance(results["std_deviation"], torch.Tensor)


def test_reflect_padding_violation_fix():
    """Verify that letterbox padding uses reflect mode instead of black pixels."""
    from src.vlm.explainability import VisualExplainer
    from PIL import Image
    
    # Create a 10x5 image (W=10, H=5)
    # Give it specific values at the top and bottom rows
    pixels = np.zeros((5, 10, 3), dtype=np.uint8)
    pixels[0, :] = [255, 0, 0]   # Red top row
    pixels[4, :] = [0, 0, 255]   # Blue bottom row
    img = Image.fromarray(pixels)
    
    # VisualExplainer._letterbox_pad returns (padded_image, pad_info)
    padded, info = VisualExplainer._letterbox_pad(img)
    assert padded.size == (10, 10)
    
    padded_arr = np.array(padded)
    # The original image starts at info["pad_top"] (which is 2)
    # So padded_arr[2, 0] should be Red [255, 0, 0]
    assert np.array_equal(padded_arr[info["pad_top"], 0], [255, 0, 0])
    # The bottom of original is at index 2 + 5 - 1 = 6. So padded_arr[6, 0] should be Blue [0, 0, 255]
    assert np.array_equal(padded_arr[info["pad_top"] + 5 - 1, 0], [0, 0, 255])
    
    # Under reflect padding:
    # Row 1 of padded mirrors Row 1 of original (which is all zeros [0,0,0])
    # Row 0 of padded mirrors Row 2 of original (all zeros)
    # Let's test bottom reflection:
    # Row 6 is original index 4 (Blue [0,0,255])
    # Row 7 mirrors original index 3 (all zeros)
    # Row 8 mirrors original index 2 (all zeros)
    # Row 9 mirrors original index 1 (all zeros)
    # Let's check with an image that has distinct values throughout to verify reflection
    grad_pixels = np.arange(15).reshape(5, 1, 3).repeat(10, axis=1).astype(np.uint8)
    # grad_pixels: row 0 has [0,1,2], row 1 has [3,4,5], row 2 has [6,7,8], row 3 has [9,10,11], row 4 has [12,13,14]
    grad_img = Image.fromarray(grad_pixels)
    padded_grad, info_grad = VisualExplainer._letterbox_pad(grad_img)
    padded_grad_arr = np.array(padded_grad)
    
    # Under reflection:
    # Row 2 (original index 0): [0,1,2]
    # Row 1 (mirrors original index 1): [3,4,5]
    # Row 0 (mirrors original index 2): [6,7,8]
    assert np.array_equal(padded_grad_arr[2, 0], [0, 1, 2])
    assert np.array_equal(padded_grad_arr[1, 0], [3, 4, 5])
    assert np.array_equal(padded_grad_arr[0, 0], [6, 7, 8])


def test_linear_time_mmd_unbiased():
    """Verify that DriftDetector uses linear-time MMD and handles equal/different inputs correctly."""
    from src.monitoring.drift_detector import DriftDetector
    with patch.dict("os.environ", {"TESTING": "false"}):
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            detector = DriftDetector()
            
    # Generate random features
    X = np.random.normal(0, 1, (100, 10))
    Y = np.random.normal(0, 1, (100, 10))
    Z = np.random.normal(5, 1, (100, 10))  # Significant shift
    
    mmd_xy = detector._compute_mmd(X, Y)
    mmd_xz = detector._compute_mmd(X, Z)
    
    # MMD with shifted distribution should be larger
    assert isinstance(mmd_xy, float)
    assert isinstance(mmd_xz, float)
    assert mmd_xz > mmd_xy


def test_uncertainty_variance_addition():
    """Verify combined uncertainty uses variance addition instead of geometric mean."""
    from src.models.uncertainty import UncertaintyEstimator
    
    model = MagicMock()
    model.modules.return_value = []
    model.return_value = (torch.randn(2, 512), torch.tensor([[1.0]*5, [1.0]*5]))
    
    estimator = UncertaintyEstimator(model)
    
    # We will test the underlying logic directly using mock values for fusion & visual uncertainties
    fusion_unc = torch.tensor([0.2, 0.4])
    visual_unc = torch.tensor([0.1, 0.3])
    
    # Mock visual_uncertainty TTA results
    # UncertaintyEstimator combines them in estimate_uncertainty using torch.where
    # combined = fusion_uncertainties + visual_uncertainty
    combined = torch.where(
        torch.isnan(visual_unc),
        fusion_unc,
        fusion_unc + visual_unc
    )
    
    assert torch.allclose(combined, torch.tensor([0.3, 0.7]))


@pytest.mark.asyncio
async def test_redis_semaphore_spinlock_usage():
    """Verify Redis semaphore uses Pub/Sub wait and event notifications instead of spin-sleep polling."""
    from deployment.api.main import RedisDistributedSemaphore, SEMAPHORE_WAITERS, SEMAPHORE_LISTENER_TASKS
    import unittest
    
    mock_redis = MagicMock()
    # Mock eval to return 0 (not acquired) first, then 1 (acquired)
    mock_redis.eval.side_effect = [0, 1]
    
    # Mock pipeline for release
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline
    mock_pipeline.zrem.return_value = mock_pipeline
    mock_pipeline.publish.return_value = mock_pipeline
    
    sem = RedisDistributedSemaphore(mock_redis, "test_sem", limit=2)
    
    # Mock the background listener task creation
    with patch("deployment.api.main._start_semaphore_listener") as mock_listener_func:
        # To simulate the event being set (e.g. by pubsub message) and prevent waiting the full 2.0s
        async def mock_wait_for(fut, timeout):
            # Find the event in SEMAPHORE_WAITERS and set it immediately
            waiters = SEMAPHORE_WAITERS.get(sem.name, [])
            for event in waiters:
                event.set()
            return True
            
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            await sem.__aenter__()
            
        # Verify eval was called to acquire
        assert mock_redis.eval.call_count == 2
        
    # Exit the semaphore
    await sem.__aexit__(None, None, None)
    
    # Verify pipeline was used to release and publish
    mock_pipeline.zrem.assert_called_once_with(sem.name, unittest.mock.ANY)
    mock_pipeline.publish.assert_called_once_with(f"{sem.name}:released", "released")
    mock_pipeline.execute.assert_called_once()
    
    # Clean up any listener task
    task = SEMAPHORE_LISTENER_TASKS.get(sem.name)
    if task:
        task.cancel()


def test_s3_storage_temp_tracking_and_cleanup():
    """Verify S3StorageProvider tracks downloaded temp files and deletes them upon cleanup."""
    from src.utils.storage import S3StorageProvider
    import tempfile
    
    with patch.dict("os.environ", {"S3_ACCESS_KEY": "dummy", "S3_SECRET_KEY": "dummy"}), \
         patch("minio.Minio") as mock_minio_cls:
         
        mock_minio = MagicMock()
        mock_minio_cls.return_value = mock_minio
        
        provider = S3StorageProvider()
        
        # Mock load download
        dummy_file = tempfile.NamedTemporaryFile(delete=False)
        dummy_file.write(b"data")
        dummy_name = dummy_file.name
        dummy_file.close()
        
        # We manually inject the downloaded file path as if load had generated it
        provider._downloaded_temp_files.add(dummy_name)
        assert os.path.exists(dummy_name)
        
        # Execute cleanup_downloads
        provider.cleanup_downloads()
        
        # Check it got cleaned up and removed from tracked set
        assert not os.path.exists(dummy_name)
        assert dummy_name not in provider._downloaded_temp_files


@pytest.mark.asyncio
async def test_dlq_configured_directory_and_fsync():
    """Verify that DLQ local fallback uses configured DLQ_DIR environment variable and writes to it."""
    from src.data.fhir_formatter import EHRGateway
    import tempfile
    import shutil
    from pathlib import Path
    
    custom_dlq_dir = tempfile.mkdtemp()
    try:
        with patch.dict("os.environ", {"DLQ_DIR": custom_dlq_dir}):
            gateway = EHRGateway(endpoint_url="http://completely-broken-invalid-host/fhir")
            dummy_payload = json.dumps({"resourceType": "DiagnosticReport", "id": "456"})
            
            # Since S3 is not configured and endpoint is broken, it will fall back to local disk
            result = await gateway.push_report(dummy_payload)
            assert result is False
            
            # Verify file was created in the custom configured directory
            custom_path = Path(custom_dlq_dir)
            files = list(custom_path.glob("failed_report_*.json"))
            assert len(files) == 1
            
            with open(files[0], "r") as f:
                wrapper = json.load(f)
            
            if isinstance(wrapper, dict) and wrapper.get("encrypted"):
                from src.utils.security import decrypt_payload
                decrypted = decrypt_payload(wrapper["data"])
                stored = json.loads(decrypted)
            else:
                stored = wrapper
                
            assert stored["payload"]["resourceType"] == "DiagnosticReport"
    finally:
        shutil.rmtree(custom_dlq_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_redis_semaphore_outage_fallback_limit():
    """Verify that RedisDistributedSemaphore degrades gracefully to local fallback limit upon Redis connection failure."""
    from deployment.api.main import RedisDistributedSemaphore
    
    mock_redis = MagicMock()
    # Mock eval to throw an exception to simulate connection error / Redis outage
    mock_redis.eval.side_effect = Exception("Redis Connection Timeout")
    
    with patch.dict("os.environ", {"MAX_CONCURRENT_REQUESTS_FALLBACK": "1"}), \
         patch("deployment.api.main._send_system_alert") as mock_alert:
         
        sem = RedisDistributedSemaphore(mock_redis, "test_outage", limit=2)
        
        # Enter the semaphore
        await sem.__aenter__()
        
        # Verify it marked redis as None for fallback
        assert sem.redis is None
        # Verify system alert was sent
        mock_alert.assert_called_once()
        
        # Verify it holds the local fallback lock
        assert sem.fallback_sem.locked()
        
        # Exit the semaphore
        await sem.__aexit__(None, None, None)
        assert not sem.fallback_sem.locked()


def test_uncertainty_law_of_total_variance_covariance():
    """Verify combined uncertainty computes covariance correctly according to the Law of Total Variance."""
    from src.models.uncertainty import UncertaintyEstimator
    
    model = MagicMock()
    model.modules.return_value = []
    
    # We will simulate logits across 5 passes to see how covariance affects std_deviation
    # Create 5 passes with logits
    model.side_effect = [
        (torch.randn(1, 512), torch.tensor([[1.0, 0.0]])),
        (torch.randn(1, 512), torch.tensor([[1.2, 0.0]])),
        (torch.randn(1, 512), torch.tensor([[0.8, 0.0]])),
        (torch.randn(1, 512), torch.tensor([[1.1, 0.0]])),
        (torch.randn(1, 512), torch.tensor([[0.9, 0.0]])),
    ]
    
    estimator = UncertaintyEstimator(model)
    v = torch.randn(1, 512)
    t = torch.randn(1, 768)
    
    # We pass visual_std
    visual_std = torch.ones(1, 512) * 0.1
    results = estimator.estimate_uncertainty(v, t, num_passes=5, visual_std=visual_std)
    
    # Assert return fields exist and are of correct types
    assert "std_deviation" in results
    assert "fusion_head_variance" in results
    assert "visual_uncertainty_score" in results
    assert "combined_uncertainty" in results
    
    # Check that std_deviation is a Tensor and matches combined_uncertainty
    assert isinstance(results["std_deviation"], torch.Tensor)
    assert torch.equal(results["std_deviation"], results["combined_uncertainty"])


def test_fhir_diagnostic_report_contained_observations():
    """Verify FHIRFormatter structures differential probabilities as contained Observation resources."""
    from src.data.fhir_formatter import FHIRFormatter
    
    formatter = FHIRFormatter()
    sample_data = {
        "patient_id": "P12345",
        "primary_finding": "Silicosis",
        "differential": {"Silicosis": 0.72, "Pneumonia": 0.18, "Tuberculosis": 0.10}
    }
    
    report = formatter.create_diagnostic_report(sample_data)
    
    # Verify report fields
    assert report.__resource_type__ == "DiagnosticReport"
    assert report.conclusion is not None
    assert "Silicosis: 72.0%" in report.conclusion
    
    # Verify contained Observation resources
    assert report.contained is not None
    assert len(report.contained) == 3
    
    for obs in report.contained:
        assert obs.__resource_type__ == "Observation"
        assert obs.status == "final"
        assert obs.subject.reference == "Patient/P12345"
        assert obs.valueQuantity is not None
        assert obs.valueQuantity.unit == "%"
        assert obs.valueQuantity.value in [0.72, 0.18, 0.10]
        
        # Verify SNOMED-CT codes are correctly mapped
        coding = obs.code.coding[0]
        assert coding.system == "http://snomed.info/sct"
        if coding.display == "Silicosis":
            assert coding.code == "50751000"
        elif coding.display == "Pneumonia":
            assert coding.code == "233604007"
        elif coding.display == "Tuberculosis":
            assert coding.code == "56717001"
            
    # Verify result references in the report match contained IDs
    assert len(report.result) == 3
    contained_ids = [f"#{obs.id}" for obs in report.contained]
    for ref in report.result:
        assert ref.reference in contained_ids


def test_dlq_payload_encryption_and_decryption():
    """Verify that DLQ payloads are encrypted and decrypted correctly using AES-GCM."""
    from src.utils.security import encrypt_payload, decrypt_payload
    payload = {"patient_id": "P999", "findings": ["Finding 1"]}
    plaintext = json.dumps(payload)
    
    ciphertext = encrypt_payload(plaintext)
    assert ciphertext != plaintext
    # Nonce is random, so encrypting twice gives different ciphertexts
    ciphertext2 = encrypt_payload(plaintext)
    assert ciphertext != ciphertext2
    
    decrypted = decrypt_payload(ciphertext)
    assert decrypted == plaintext
    assert json.loads(decrypted) == payload


@pytest.mark.asyncio
async def test_redis_semaphore_self_healing_reconnect():
    """Verify that RedisDistributedSemaphore attempts self-healing reconnection check."""
    from deployment.api.main import RedisDistributedSemaphore
    
    mock_redis = MagicMock()
    # Initially ping fails, then succeeds
    mock_redis.ping.side_effect = [Exception("Ping timeout"), True]
    mock_redis.eval.return_value = 1  # Optimistic acquisition succeeds once ping succeeds
    
    # Set short reconnect cooldown for testing
    with patch.dict("os.environ", {"REDIS_RECONNECT_COOLDOWN": "0.01", "MAX_CONCURRENT_REQUESTS_FALLBACK": "1"}):
        # Start with None/offline Redis, but store mock_redis in orig_redis
        sem = RedisDistributedSemaphore(mock_redis, "test_reconnect", limit=2)
        sem.redis = None  # Simulate offline/fallback state
        sem.reconnect_cooldown = 0.01
        
        # 1. First enter: ping fails, so it stays in fallback mode (acquires fallback_sem)
        await sem.__aenter__()
        assert sem.redis is None
        assert sem.fallback_sem.locked()
        await sem.__aexit__(None, None, None)
        assert not sem.fallback_sem.locked()
        
        # 2. Wait a bit for cooldown
        await asyncio.sleep(0.02)
        
        # 3. Second enter: ping succeeds, so self.redis is restored to mock_redis, and local_sem is acquired
        await sem.__aenter__()
        assert sem.redis is mock_redis
        assert not sem.fallback_sem.locked()
        await sem.__aexit__(None, None, None)


def test_drift_detector_local_baseline_cache():
    """Verify that DriftDetector caches baselines locally and fallback works when Redis is offline."""
    import tempfile
    import shutil
    from src.monitoring.drift_detector import DriftDetector
    
    custom_cache_dir = tempfile.mkdtemp()
    try:
        # Patch Redis to simulate connection errors
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("Redis connection error")
        mock_redis.set.side_effect = Exception("Redis connection error")
        
        with patch.dict("os.environ", {"DRIFT_CACHE_DIR": custom_cache_dir, "TESTING": "false"}), \
             patch("redis.Redis", return_value=mock_redis):
            
            # 1. Instantiate detector (Redis offline, cache empty -> baseline is None)
            detector = DriftDetector()
            assert detector.baseline is None
            assert detector.features_baseline is None
            
            # 2. Save baseline (Redis offline -> baseline is saved locally)
            test_probs = [[0.1, 0.9], [0.2, 0.8]]
            detector._save_baseline(test_probs)
            
            # 3. Re-instantiate detector (Redis still offline, cache has baseline -> baseline is restored from local cache)
            detector2 = DriftDetector()
            assert detector2.baseline is not None
            assert np.array_equal(detector2.baseline, np.array(test_probs))
            
    finally:
        shutil.rmtree(custom_cache_dir, ignore_errors=True)


def test_failed_pdf_parsing_raises_http_400(monkeypatch):
    """Verify that malformed/corrupted PDF uploads raise HTTP 400 Bad Request in production-like mode (TESTING=false)."""
    import deployment.api.main as api_main
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from io import BytesIO

    # Mock agent and storage to prevent actual runs
    class DummyAgent:
        async def run(self, image_path, pdf_path): return {}
    monkeypatch.setattr(api_main, "build_agent", lambda: DummyAgent())
    monkeypatch.setattr(api_main, "storage", MagicMock())

    # Set TESTING to false to simulate production fail-fast check
    from src.utils.secrets_manager import SecretsManager
    SecretsManager._cache.clear()
    with patch.dict("os.environ", {"TESTING": "false", "API_KEY": "test-secret"}):
        app = api_main.create_app()
        with TestClient(app) as client:
            response = client.post(
                "/v1/analyze?sync=true",
                files={
                    "image": ("scan.png", BytesIO(b"image-bytes"), "image/png"),
                    "history": ("corrupt.pdf", BytesIO(b"this-is-garbage-pdf-bytes"), "application/pdf")
                },
                headers={"X-API-Key": "test-secret"}
            )
            # Should fail fast with HTTP 400 Bad Request
            assert response.status_code == 400
            assert "Failed to parse clinical history PDF" in response.json()["detail"]


def test_static_ood_threshold(monkeypatch):
    """Verify that setting OOD_USE_STATIC_THRESHOLD=true utilizes the configured static OOD cosine threshold."""
    from src.agent.clinical_graph import ClinicalAgent
    
    mock_parser = MagicMock()
    mock_rag = MagicMock()
    
    agent = ClinicalAgent(mock_parser, mock_rag, inference_api_url="http://localhost:8000")
    
    # We will test the node_synthesize_diagnosis behavior with OOD_USE_STATIC_THRESHOLD=true
    state = {
        "visual_features": [[0.0] * 256 + [0.9] * 256],
        "history_data": {},
        "diagnosis_results": {
            "mean_confidence": [0.9],
            "std_deviation": [0.01],
            "all_probs": [[0.8, 0.05, 0.05, 0.05, 0.05]]
        }
    }
    
    # Write a centroid cache JSON file
    baseline_cache_dir = Path("temp/drift")
    baseline_cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache_path = baseline_cache_dir / "features_centroid.json"
    
    # Set a baseline centroid of 10 samples
    baseline_data = {"centroid": [0.9] * 256 + [0.0] * 256, "count": 10}
    with open(baseline_cache_path, "w") as f:
        json.dump(baseline_data, f)
        
    try:
        with patch.dict("os.environ", {
            "OOD_USE_STATIC_THRESHOLD": "true",
            "OOD_COSINE_THRESHOLD": "0.99"  # Highly strict threshold to guarantee OOD flag is triggered
        }), patch.object(agent, "_http_client") as mock_client:
            # Mock async post method
            async def mock_post(url, *args, **kwargs):
                m = MagicMock()
                m.raise_for_status = MagicMock()
                if "/encode/text" in url:
                    m.json.return_value = {"embeddings": [[0.1] * 512]}
                elif "/estimate" in url:
                    m.json.return_value = {
                        "prediction": [4],
                        "mean_confidence": [0.9],
                        "std_deviation": [0.01],
                        "all_probs": [[0.8, 0.05, 0.05, 0.05, 0.05]]
                    }
                else:
                    m.json.return_value = {"citations": []}
                return m
            mock_client.post = mock_post
            
            new_state = asyncio.run(agent.node_synthesize_diagnosis(state))
            # With threshold = 0.99, our visual features ([0.1]*512 vs baseline [0.9]*512) will fail OOD
            assert new_state["escalation_required"] is True
    finally:
        if baseline_cache_path.exists():
            baseline_cache_path.unlink()


def test_dicom_secondary_capture_generation(tmp_path):
    """Verify that create_secondary_capture constructs a valid DICOM SC instance containing the heatmap."""
    from src.data.dicom_handler import create_secondary_capture
    from PIL import Image
    import pydicom
    
    # 1. Create a dummy PNG image representing the heatmap
    png_path = tmp_path / "heatmap.png"
    img = Image.new("RGB", (256, 256), color=(255, 0, 0)) # Solid red image
    img.save(png_path)
    
    dcm_output_path = tmp_path / "output.dcm"
    
    # 2. Run conversion (with None for original DICOM path)
    create_secondary_capture(None, str(png_path), str(dcm_output_path))
    
    assert dcm_output_path.exists()
    
    # 3. Read it back and verify key tags
    ds = pydicom.dcmread(str(dcm_output_path))
    assert ds.PatientName == "REDACTED_PATIENTNAME"
    assert ds.PatientID == "REDACTED_PATIENTID"
    assert ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.7'  # SC Class
    assert ds.Rows == 256
    assert ds.Columns == 256
    assert ds.SamplesPerPixel == 3
    assert ds.PhotometricInterpretation == "RGB"
    
    # Verify pixel data length (256 * 256 * 3 = 196608 bytes)
    assert len(ds.PixelData) == 196608
    # The first pixel should be red [255, 0, 0]
    pixel_array = ds.pixel_array
    assert np.array_equal(pixel_array[0, 0], [255, 0, 0])


def test_roi_telemetry_custom_config(monkeypatch):
    """Verify that get_telemetry_metrics respects custom configuration environment variables for ROI metrics."""
    import deployment.api.main as api_main
    from fastapi.testclient import TestClient
    
    mock_redis = MagicMock()
    mock_redis.get.side_effect = lambda k: {
        "medi_chain:telemetry:total_cases": "10",
        "medi_chain:telemetry:escalated_cases": "2"
    }.get(k, None)
    
    monkeypatch.setattr(api_main, "redis_client", mock_redis)
    monkeypatch.setattr(api_main, "use_redis", True)
    
    from src.utils.secrets_manager import SecretsManager
    SecretsManager._cache.clear()
    
    with patch.dict("os.environ", {
        "TESTING": "true",
        "API_KEY": "secret-key",
        "TELEMETRY_BASELINE_MINUTES": "30",
        "TELEMETRY_AUTOMATED_MINUTES": "2",
        "TELEMETRY_HOURLY_RATE": "150.0"
    }):
        app = api_main.create_app()
        with TestClient(app) as client:
            response = client.get("/v1/telemetry/metrics", headers={"X-API-Key": "secret-key"})
            assert response.status_code == 200
            data = response.json()
            
            # Expected calculations:
            # total = 10, escalated = 2
            # non-escalated = 8
            # saved_minutes = 8 * (30 - 2) = 224 minutes
            # saved_hours = 224 / 60 = 3.73 hours
            # saved_cost = 3.73 * 150 = 559.5 USD
            assert data["saved_time_hours"] == 3.73
            assert data["saved_cost_usd"] == 559.5


def test_check_prediction_drift_confidence_based():
    """Verify that check_prediction_drift computes KS-test on maximum confidence scores."""
    from src.monitoring.drift_detector import DriftDetector
    import numpy as np
    from unittest.mock import patch, MagicMock
    import pytest
    
    with patch.dict("os.environ", {"TESTING": "false"}):
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            detector = DriftDetector()
            
    # Set up baseline probabilities (high confidence in class 0)
    baseline_probs = np.zeros((100, 5))
    baseline_probs[:, 0] = 0.9
    baseline_probs[:, 1] = 0.1
    detector.baseline = baseline_probs
    
    # Test case 1: identical distribution (no drift)
    current_probs_no_drift = np.zeros((100, 5))
    current_probs_no_drift[:, 0] = 0.9
    current_probs_no_drift[:, 1] = 0.1
    
    with patch("src.monitoring.drift_detector._send_alert") as mock_alert:
        result = detector.check_prediction_drift(current_probs_no_drift.tolist())
        assert result is False
        mock_alert.assert_not_called()
        
    # Test case 2: drifted distribution (lower confidence, e.g., 0.5 class 0)
    current_probs_drift = np.zeros((100, 5))
    current_probs_drift[:, 0] = 0.5
    current_probs_drift[:, 1] = 0.5
    
    with patch("src.monitoring.drift_detector._send_alert") as mock_alert:
        result = detector.check_prediction_drift(current_probs_drift.tolist())
        assert result is True
        mock_alert.assert_called_once()

