import os
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
            
            # Execute node_extract_visuals, which will throw error when trying to POST to invalid url
            with pytest.raises(RuntimeError) as exc_info:
                await agent.node_extract_visuals(state)
                
            assert "Visual encoder failed" in str(exc_info.value)
            
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
    
    model = MagicMock()
    preprocess = MagicMock()
    
    explainer = VisualExplainer(model, preprocess)
    
    # Create a real numpy array of size 200x100 (W=200, H=100) -> landscape (W > H)
    # So crop box is H x H -> 100 x 100
    mock_img = np.zeros((100, 200, 3), dtype=np.uint8)
    
    # Mock cv2 and pytorch_grad_cam functions
    with patch("PIL.Image.open") as mock_open, \
         patch("cv2.resize") as mock_resize, \
         patch("pytorch_grad_cam.GradCAM.__call__") as mock_cam_call, \
         patch("src.vlm.explainability.show_cam_on_image") as mock_show:
         
        # Make Image.open return a mock image object that can be converted to numpy array
        mock_pil_img = MagicMock()
        mock_open.return_value = mock_pil_img
        
        # Patch np.array(Image.open(...)) to return our mock_img array
        with patch("numpy.array", return_value=mock_img):
            mock_cam_call.return_value = np.zeros((1, 14, 14), dtype=np.float32)
            mock_resize.return_value = np.zeros((100, 100), dtype=np.float32)
            mock_show.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
            
            explainer.generate_heatmap("fake_xray.png")
            
            # Check cv2.resize call arguments. It should resize 14x14 grid to (box_w, box_h) -> (100, 100)
            mock_resize.assert_called_once()
            resize_args = mock_resize.call_args[0]
            assert resize_args[1] == (100, 100)


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
