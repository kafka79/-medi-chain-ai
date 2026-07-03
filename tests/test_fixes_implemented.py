import os
import json
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from src.models.fusion import load_diagnostic_classes
from src.monitoring.xai_explainer import XAIExplainer
from src.rag.evaluator import RAGEvaluator
from src.models.uncertainty import UncertaintyEstimator

def test_dynamic_class_loading(tmp_path):
    # Test loading from config path
    config_file = tmp_path / "classes.json"
    classes_list = ["COVID", "Pneumothorax", "Normal"]
    with open(config_file, "w") as f:
        json.dump(classes_list, f)
        
    with patch.dict(os.environ, {"DIAGNOSTIC_CLASSES_PATH": str(config_file), "DIAGNOSTIC_CLASSES": ""}):
        loaded = load_diagnostic_classes()
        assert loaded == classes_list

    # Test loading from environment variable
    with patch.dict(os.environ, {"DIAGNOSTIC_CLASSES": '["A", "B"]'}):
        loaded = load_diagnostic_classes()
        assert loaded == ["A", "B"]

def test_xai_explainer_dynamic_synthesis():
    explainer = XAIExplainer()
    history = {
        "chief_complaint": "chronic cough",
        "history_present_illness": "Patient reports severe chest pain and cough for two weeks.",
        "labs": "Elevated WBC count",
        "metadata": {
            "age": "60",
            "gender": "Male",
            "occupation": "Driller",
            "exposure_years": "20"
        }
    }
    
    citations = [{"title": "Silica dust hazards", "pmid": "999888", "text": "Study content"}]
    
    rationale = explainer.explain(
        diagnosis="Silicosis",
        confidence=0.88,
        uncertainty=0.03,
        probabilities=[0.88, 0.05, 0.03, 0.02, 0.02],
        history_data=history,
        pubmed_citations=citations
    )
    
    assert "Driller" in rationale
    assert "20 years" in rationale
    assert "chronic cough" in rationale
    assert "chest pain" in rationale
    assert "labs: Elevated WBC count" in rationale
    assert "Silica dust hazards" in rationale
    assert "PMID: 999888" in rationale
    assert "Margin of 83.0%" in rationale

@pytest.mark.asyncio
async def test_evaluator_lazy_connection():
    with patch("pymilvus.connections.connect") as mock_connect, \
         patch("src.rag.evaluator._send_alert") as mock_alert:
         
        # Make connections.connect fail on first 2 calls, then succeed on 3rd
        mock_connect.side_effect = [ValueError("Milvus offline"), ValueError("Milvus offline"), None]
        
        # In CI testing defaults to True which skips eager connection, so we patch to False
        # Set MILVUS_CONN_FAIL_COOLDOWN to 0 to bypass the circuit breaker during this test
        with patch.dict(os.environ, {"TESTING": "false", "MILVUS_CONN_FAIL_COOLDOWN": "0"}):
            evaluator = RAGEvaluator(milvus_host="localhost", milvus_port="19530")
            
            # Connection failed during init but didn't block or raise
            assert evaluator.collection is None
            
            # Trigger lazy connection check: should fail on first try (which is the 2nd overall fail)
            with pytest.raises(RuntimeError) as exc:
                await evaluator._ensure_connected()
            assert "uninitialized" in str(exc.value)
            
            # Called once during eager init and once during lazy connection
            assert mock_alert.call_count == 2
            
            # Patch pymilvus.Collection to avoid Milvus actual class instantiation issues in mock
            with patch("src.rag.evaluator.Collection") as mock_collection_class:
                mock_collection = MagicMock()
                mock_collection_class.return_value = mock_collection
                
                # Try lazy connection again: 3rd overall call will succeed
                await evaluator._ensure_connected()
                assert evaluator.collection is not None

def test_async_alert_sending():
    from src.monitoring.drift_detector import _send_alert
    
    with patch("src.monitoring.drift_detector.http_requests.post") as mock_post, \
         patch("src.monitoring.drift_detector.DRIFT_ALERT_WEBHOOK_URL", "http://alert-endpoint/webhook"):
        _send_alert("Title", "Message")
        
        # Since alert sending is offloaded to a background thread pool,
        # we sleep briefly to allow the thread executor to pick it up and execute.
        import time
        time.sleep(0.5)
        
        mock_post.assert_called_once()

def test_mc_dropout_perturbation_with_visual_std():
    model = MagicMock()
    model.modules.return_value = []
    # Mock return: (fused, logits)
    model.return_value = (torch.randn(1, 512), torch.tensor([[1.5, 0.5, 0.2, 0.1, 0.0]]))
    
    estimator = UncertaintyEstimator(model)
    v = torch.randn(1, 512)
    t = torch.randn(1, 768)
    visual_std = torch.randn(1, 512).abs() * 0.1
    
    # Run estimate_uncertainty with visual_std
    results = estimator.estimate_uncertainty(v, t, num_passes=5, visual_std=visual_std)
    assert "prediction" in results
    assert isinstance(results["std_deviation"], torch.Tensor)
