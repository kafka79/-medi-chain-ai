import pytest
from src.agent.clinical_graph import ClinicalAgent
from unittest.mock import MagicMock
import torch

def test_agent_loop_protection():
    """
    Assert that the agent does not enter an infinite loop.
    We mock the components to simulate a low-confidence scenario.
    """
    # Mock components
    encoder = MagicMock()
    encoder.encode_image.return_value = torch.randn(1, 512)
    
    parser = MagicMock()
    parser.parse_pdf.return_value = {"chief_complaint": "chronic cough"}
    
    rag = MagicMock()
    rag.search.return_value = [{"pmid": "1", "title": "test", "text": "test"}]
    
    fusion = MagicMock()
    # Mock fusion to always return low confidence (forcing retry)
    # logits for 5 classes
    mock_logits = torch.tensor([[5.0, 1.0, 1.0, 1.0, 1.0]]) # This would be high confidence though
    # Let's make it ambiguous
    mock_logits = torch.tensor([[1.1, 1.1, 1.1, 1.1, 1.1]]) 
    fusion.return_value = (torch.randn(1, 256), mock_logits)
    
    text_encoder = MagicMock()
    text_encoder.encode.return_value = torch.randn(1, 768)
    
    uncertainty = MagicMock()
    # Force high uncertainty
    uncertainty.estimate_uncertainty.return_value = {
        "prediction": torch.tensor([0]),
        "mean_confidence": torch.tensor([0.2]), # Low confidence
        "std_deviation": torch.tensor([0.25]), # High uncertainty
        "all_probs": torch.softmax(mock_logits, dim=1)
    }
    
    agent = ClinicalAgent(encoder, parser, rag, fusion, text_encoder, uncertainty)
    
    # Run agent
    # It should hit max 3 iterations and then end or escalate
    result = agent.run("dummy_image.png", "dummy_history.pdf")
    
    assert result['iteration_count'] <= 3
    assert result['escalation_required'] is True
    print(f"Agent terminated correctly after {result['iteration_count']} iterations.")

if __name__ == "__main__":
    test_agent_loop_protection()
