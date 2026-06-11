import os
import pytest
import numpy as np
import json
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
    
    # Call check_prediction_drift, it should reset the baseline and return False instead of raising IndexError
    try:
        result = detector.check_prediction_drift(current_probs_mismatch)
        assert result is False
        detector._save_baseline.assert_called_once_with(current_probs_mismatch)
    except IndexError:
        pytest.fail("check_prediction_drift threw IndexError on class count mismatch!")
