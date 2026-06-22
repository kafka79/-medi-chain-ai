"""
Flaw #21 Fix: Unit tests for ClinicalAgent LangGraph orchestration.
Tests the retry logic, escalation path, should_continue conditional edge,
and OOD detection without requiring real model inference.
"""
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Mock the PrivacyScrubber at its source module before clinical_graph imports it
_mock_linear = MagicMock()
_mock_linear.out_features = 5


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies so tests run without GPU, Milvus, or inference API."""
    with patch("src.data.privacy_scrubber.PrivacyScrubber") as mock_scrubber_cls, \
         patch("src.models.fusion.LateFusionModel") as mock_fusion_cls:
        
        # Mock the fusion model's classifier for the startup assertion
        mock_model_instance = MagicMock()
        mock_model_instance.classifier.__getitem__ = MagicMock(return_value=_mock_linear)
        mock_fusion_cls.return_value = mock_model_instance
        
        # Mock the scrubber
        mock_scrubber_cls.return_value = MagicMock()
        
        yield


def _make_agent():
    """Helper to create a ClinicalAgent with mocked parser and RAG."""
    # Import inside function so the patches are active
    from src.agent.clinical_graph import ClinicalAgent
    
    mock_parser = MagicMock()
    mock_rag = MagicMock()
    mock_rag.search.return_value = [{"pmid": "12345", "text": "test", "title": "Test Paper"}]
    
    agent = ClinicalAgent(
        history_parser=mock_parser,
        rag_evaluator=mock_rag,
        inference_api_url="http://fake-inference:8001"
    )
    return agent


class TestShouldContinue:
    """Tests for the should_continue conditional edge logic."""
    
    def test_ends_on_high_confidence(self):
        agent = _make_agent()
        state = {
            "confidence": 0.95,
            "diagnosis": {"uncertainty_std": 0.02},
            "iteration_count": 1,
            "escalation_required": False,
        }
        assert agent.should_continue(state) == "end"
    
    def test_retries_on_low_confidence(self):
        agent = _make_agent()
        state = {
            "confidence": 0.4,
            "diagnosis": {"uncertainty_std": 0.05},
            "iteration_count": 1,
            "escalation_required": False,
        }
        assert agent.should_continue(state) == "retry"
    
    def test_retries_on_high_uncertainty(self):
        agent = _make_agent()
        state = {
            "confidence": 0.8,
            "diagnosis": {"uncertainty_std": 0.25},
            "iteration_count": 1,
            "escalation_required": False,
        }
        assert agent.should_continue(state) == "retry"
    
    def test_ends_after_max_retries(self):
        agent = _make_agent()
        state = {
            "confidence": 0.4,
            "diagnosis": {"uncertainty_std": 0.25},
            "iteration_count": 3,  # MAX_RETRY_ITERATIONS default
            "escalation_required": False,
        }
        assert agent.should_continue(state) == "end"
    
    def test_ends_immediately_on_escalation(self):
        agent = _make_agent()
        state = {
            "confidence": 0.4,
            "diagnosis": {"uncertainty_std": 0.25},
            "iteration_count": 1,
            "escalation_required": True,
        }
        assert agent.should_continue(state) == "end"


class TestNodeSelfVerify:
    """Tests for the self_verify node logic."""
    
    async def test_increments_iteration_count(self):
        agent = _make_agent()
        state = {
            "confidence": 0.95,
            "diagnosis": {"uncertainty_std": 0.01},
            "iteration_count": 0,
            "escalation_required": False,
        }
        result = await agent.node_self_verify(state)
        assert result["iteration_count"] == 1
    
    async def test_triggers_escalation_after_max_retries(self):
        agent = _make_agent()
        state = {
            "confidence": 0.3,
            "diagnosis": {"uncertainty_std": 0.3},
            "iteration_count": 2,  # Will become 3 -> max reached
            "escalation_required": False,
        }
        result = await agent.node_self_verify(state)
        assert result["escalation_required"] is True
    
    async def test_no_escalation_on_confident_result(self):
        agent = _make_agent()
        state = {
            "confidence": 0.95,
            "diagnosis": {"uncertainty_std": 0.01},
            "iteration_count": 0,
            "escalation_required": False,
        }
        result = await agent.node_self_verify(state)
        assert result.get("escalation_required") is not True
    
    async def test_preserves_ood_escalation(self):
        """If OOD detection already flagged escalation, self_verify should preserve it."""
        agent = _make_agent()
        state = {
            "confidence": 0.95,
            "diagnosis": {"uncertainty_std": 0.01},
            "iteration_count": 0,
            "escalation_required": True,  # Already flagged by OOD
        }
        result = await agent.node_self_verify(state)
        assert result["escalation_required"] is True


class TestNodeSynthesizeDiagnosis:
    """Tests for the synthesize_diagnosis node with mocked inference API."""
    
    async def test_normal_prediction(self):
        agent = _make_agent()
        
        # Mock text encoding response
        text_resp = MagicMock()
        text_resp.status_code = 200
        text_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        text_resp.raise_for_status = MagicMock()
        
        # Mock uncertainty estimation response — high confidence, normal prediction
        est_resp = MagicMock()
        est_resp.status_code = 200
        est_resp.json.return_value = {
            "prediction": [4],  # Normal
            "mean_confidence": [0.92],
            "std_deviation": [0.02],
            "all_probs": [[0.01, 0.02, 0.03, 0.02, 0.92]],
        }
        est_resp.raise_for_status = MagicMock()
        
        # Flaw #5-structural: patch the persistent _http_client.post instead of AsyncClient context manager
        agent._http_client.post = AsyncMock(side_effect=[text_resp, est_resp])
        
        state = {
            "visual_features": [[0.1] * 512],
            "history_data": {"chief_complaint": "routine checkup", "history_present_illness": "", "labs": ""},
        }
        
        result = await agent.node_synthesize_diagnosis(state)
        assert result["diagnosis"]["top_finding"] == "Normal"
        assert result["diagnosis"]["ood_detected"] is False
        assert result["confidence"] == 0.92
    
    async def test_ood_detection_triggers_escalation(self):
        """Flaw #6: When max(softmax) < OOD_CONFIDENCE_THRESHOLD, should flag OOD."""
        agent = _make_agent()
        
        text_resp = MagicMock()
        text_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        text_resp.raise_for_status = MagicMock()
        
        # Mock a very low-confidence, spread-out prediction (possible OOD input)
        est_resp = MagicMock()
        est_resp.json.return_value = {
            "prediction": [1],
            "mean_confidence": [0.25],
            "std_deviation": [0.08],
            "all_probs": [[0.20, 0.25, 0.20, 0.15, 0.20]],  # max=0.25 < 0.4 threshold
        }
        est_resp.raise_for_status = MagicMock()
        
        # Flaw #5-structural: patch the persistent _http_client.post instead of AsyncClient context manager
        agent._http_client.post = AsyncMock(side_effect=[text_resp, est_resp])
        
        state = {
            "visual_features": [[0.1] * 512],
            "history_data": {"chief_complaint": "mass in lung", "history_present_illness": "", "labs": ""},
        }
        
        result = await agent.node_synthesize_diagnosis(state)
        assert result["diagnosis"]["ood_detected"] is True
        assert result["diagnosis"]["top_finding"] == "Out-of-Distribution"
        assert result["escalation_required"] is True


class TestDiagnosticClassesConsistency:
    """Flaw #13: Ensure DIAGNOSTIC_CLASSES is consistent with model."""
    
    def test_class_count_matches_constant(self):
        from src.agent.clinical_graph import DIAGNOSTIC_CLASSES, NUM_CLASSES
        assert len(DIAGNOSTIC_CLASSES) == NUM_CLASSES
        assert NUM_CLASSES == 5
    
    def test_known_classes_present(self):
        from src.agent.clinical_graph import DIAGNOSTIC_CLASSES
        expected = {"Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"}
        assert set(DIAGNOSTIC_CLASSES) == expected
