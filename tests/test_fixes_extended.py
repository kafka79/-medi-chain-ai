import os
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from src.vlm.explainability import VisualExplainer
from src.data.privacy_scrubber import PrivacyScrubber

def test_explainability_dynamic_vit_reshape():
    # Construct a dummy model
    model = MagicMock()
    # Mock visual branch of model
    model.visual = MagicMock()
    preprocess = MagicMock()
    
    explainer = VisualExplainer(model, preprocess)
    
    # Test reshape_transform with standard sequence length (197 tokens -> 14x14 grid)
    # Shape: (Batch, Tokens, Dim) -> (2, 197, 64)
    dummy_tensor = torch.randn(2, 197, 64)
    reshaped = explainer.reshape_transform(dummy_tensor)
    
    # Expect output shape: (Batch, Dim, Grid, Grid) -> (2, 64, 14, 14)
    assert reshaped.shape == (2, 64, 14, 14)
    
    # Test reshape_transform with dynamic sequence length (e.g., 50 tokens -> 7x7 grid)
    # Shape: (Batch, Tokens, Dim) -> (1, 50, 128)
    dummy_tensor_dyn = torch.randn(1, 50, 128)
    reshaped_dyn = explainer.reshape_transform(dummy_tensor_dyn)
    
    # Expect output shape: (1, 128, 7, 7)
    assert reshaped_dyn.shape == (1, 128, 7, 7)

def test_explainability_dynamic_layer_resolution():
    model = MagicMock()
    preprocess = MagicMock()
    
    # Mock Strategy A
    model.visual.trunk.blocks = [MagicMock()]
    explainer = VisualExplainer(model, preprocess)
    assert explainer.target_layers is not None
    
    # Mock Strategy B
    del model.visual.trunk
    model.visual.transformer.resblocks = [MagicMock()]
    explainer_b = VisualExplainer(model, preprocess)
    assert explainer_b.target_layers is not None

def test_privacy_scrubber_dynamic_ocr_bounds():
    scrubber = PrivacyScrubber()
    
    # Mock cv2.imread and shape
    with patch("cv2.imread") as mock_imread, patch("cv2.threshold") as mock_threshold, patch("cv2.dilate") as mock_dilate, patch("cv2.findContours") as mock_find:
        # Create a mock high-resolution image shape: 2000x2000
        mock_img = MagicMock()
        mock_img.shape = (2000, 2000)
        mock_imread.return_value = mock_img
        
        mock_threshold.return_value = (None, MagicMock())
        mock_dilate.return_value = MagicMock()
        
        # Mock a large contour representing a patient identifier on high resolution scan
        # (x_c, y_c, w_c, h_c) -> e.g. height=45, width=150
        # In full image space, we'll return a bounding rect
        mock_contour = MagicMock()
        mock_find.return_value = ([mock_contour], None)
        
        with patch("cv2.boundingRect") as mock_bounding:
            # high resolution text: height=50, width=200
            mock_bounding.return_value = (50, 5, 200, 50)
            
            # Run detect_burned_in_text
            boxes = scrubber.detect_burned_in_text("fake_path.png")
            
            # Since high resolution bounds are calculated dynamically:
            # min_h_c = max(4, int(2000 * 0.008)) = 16
            # max_h_c = max(15, int(2000 * 0.035)) = 70
            # min_w_c = max(6, int(2000 * 0.012)) = 24
            # height 50 is inside [16, 70] and width 200 >= 24, so it should be collected
            # It processes both top and bottom zones, so it returns 2 boxes due to mock.
            assert len(boxes) == 2
            assert boxes[0] == (50, 5, 200, 50)
            assert boxes[1] == (50, 1705, 200, 50)


def test_attention_fusion_transformer_block():
    from src.models.fusion import AttentionFusion
    model = AttentionFusion(vision_dim=512, text_dim=768, hidden_dim=512)
    # Check that self.ffn is present
    assert hasattr(model, "ffn")
    assert hasattr(model, "norm2")
    
    # Run forward pass
    vision_emb = torch.randn(4, 512)
    text_emb = torch.randn(4, 768)
    fused, logits = model(vision_emb, text_emb)
    
    assert fused.shape == (4, 512)
    assert logits.shape == (4, 5)

def test_privacy_scrubber_eager_ner_failure():
    # Force eager loading
    with patch.dict(os.environ, {"NER_LAZY_LOAD": "false", "TESTING": ""}), \
         patch("transformers.pipeline", side_effect=ValueError("HF Hub unreachable")):
        with pytest.raises(RuntimeError) as exc_info:
            PrivacyScrubber()
        assert "Privacy scrubber failed to load the NER model at startup" in str(exc_info.value)

def test_ehr_gateway_redis_dlq_fallback():
    from src.data.fhir_formatter import EHRGateway
    
    mock_redis_client = MagicMock()
    
    with patch("redis.Redis", return_value=mock_redis_client) as mock_redis_class:
        gateway = EHRGateway(endpoint_url="http://completely-broken-invalid-host/fhir")
        
        # Trigger EHR failure
        result = gateway.push_report('{"resourceType": "DiagnosticReport"}')
        
        # Should return False but not raise
        assert result is False
        
        # Verify Redis rpush was called to backup DLQ
        mock_redis_class.assert_called_once()
        mock_redis_client.rpush.assert_called_once()
        call_args = mock_redis_client.rpush.call_args[0]
        assert call_args[0] == "medi_chain:dlq"
        assert "DiagnosticReport" in call_args[1]
