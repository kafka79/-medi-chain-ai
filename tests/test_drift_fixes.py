import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch
from src.monitoring.drift_detector import DriftDetector

@pytest.mark.asyncio
async def test_drift_detector_async_offloading_and_lua():
    # Instantiate DriftDetector
    # Set TESTING to false during instantiating to let it initialize properly
    with patch.dict("os.environ", {"TESTING": "false"}):
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            detector = DriftDetector()
            
    # Mock class methods
    detector.check_prediction_drift = MagicMock()
    detector.check_covariate_shift = MagicMock()
    detector.check_performance_drift = MagicMock()
    
    probs = [0.8, 0.1, 0.1, 0.0, 0.0]
    features = [0.5] * 128
    
    # 1. Test when threshold is NOT met (Lua eval returns None)
    mock_redis.eval.return_value = None
    
    await detector.add_prediction(probs, features)
    
    # Verify Lua script was called on redis
    mock_redis.eval.assert_called_once()
    detector.check_prediction_drift.assert_not_called()
    detector.check_covariate_shift.assert_not_called()
    detector.check_performance_drift.assert_not_called()
    
    mock_redis.eval.reset_mock()
    
    # 2. Test when threshold IS met (Lua eval returns both prediction and feature lists)
    mock_probs_list = [json.dumps(probs)] * 100
    mock_features_list = [json.dumps(features)] * 100
    mock_redis.eval.return_value = [mock_probs_list, mock_features_list]
    
    await detector.add_prediction(probs, features)
    
    # Verify Lua script was called
    mock_redis.eval.assert_called_once()
    
    # Verify the checks were invoked with correct parsed arguments
    detector.check_prediction_drift.assert_called_once()
    called_probs = detector.check_prediction_drift.call_args[0][0]
    assert len(called_probs) == 100
    assert called_probs[0] == probs
    
    detector.check_covariate_shift.assert_called_once()
    called_features = detector.check_covariate_shift.call_args[0][0]
    assert len(called_features) == 100
    assert called_features[0] == features
    
    detector.check_performance_drift.assert_called_once()

@pytest.mark.asyncio
async def test_drift_detector_visual_features_none():
    with patch.dict("os.environ", {"TESTING": "false"}):
        with patch("redis.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            detector = DriftDetector()
            
    detector.check_prediction_drift = MagicMock()
    detector.check_covariate_shift = MagicMock()
    detector.check_performance_drift = MagicMock()
    
    # Mock Redis returning predictions but empty/None features list
    mock_probs_list = [json.dumps([0.8, 0.1, 0.1, 0.0, 0.0])] * 100
    mock_redis.eval.return_value = [mock_probs_list, None]
    
    await detector.add_prediction([0.8, 0.1, 0.1, 0.0, 0.0], None)
    
    detector.check_prediction_drift.assert_called_once()
    # check_covariate_shift should NOT be called if current_features is None or empty
    detector.check_covariate_shift.assert_not_called()
    detector.check_performance_drift.assert_called_once()
