import pytest
from src.agent.clinical_graph import ClinicalAgent
from unittest.mock import MagicMock, patch, AsyncMock
import torch
import os

async def test_agent_loop_protection():
    """
    Assert that the agent does not enter an infinite loop.
    We mock the components to simulate a low-confidence scenario.
    """
    parser = MagicMock()
    parser.parse_pdf.return_value = {"chief_complaint": "chronic cough"}
    
    rag = MagicMock()
    rag.search.return_value = [{"pmid": "1", "title": "test", "text": "test"}]
    
    agent = ClinicalAgent(history_parser=parser, rag_evaluator=rag, inference_api_url="http://dummy")
    agent.scrubber = MagicMock()
    agent.scrubber.mask_burned_in_text.return_value = "dummy_image.png"
    agent.scrubber.scrub_history_data.side_effect = lambda x: x
    
    mock_logits = torch.tensor([[1.1, 1.1, 1.1, 1.1, 1.1]])
    all_probs = torch.softmax(mock_logits, dim=1)[0].tolist()

    async def side_effect(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        if "encode/image" in url:
            mock_resp.json.return_value = {"features": [[0.1]*512], "heatmap_base64": ""}
        elif "encode/text" in url:
            mock_resp.json.return_value = {"embeddings": [[0.2]*768]}
        elif "estimate" in url:
            mock_resp.json.return_value = {
                "prediction": [0],
                "mean_confidence": [0.2], # Low confidence
                "std_deviation": [0.25], # High uncertainty
                "all_probs": [all_probs]
            }
        return mock_resp
        
    agent._http_client.post = AsyncMock(side_effect=side_effect)
    
    # Create dummy files
    with open("dummy_image.png", "wb") as f:
        f.write(b"fake image data")
    with open("dummy_history.pdf", "wb") as f:
        f.write(b"fake pdf data")
        
    try:
        result = await agent.run("dummy_image.png", "dummy_history.pdf")
    finally:
        await agent.close()
        if os.path.exists("dummy_image.png"):
            os.remove("dummy_image.png")
        if os.path.exists("dummy_history.pdf"):
            os.remove("dummy_history.pdf")
        
        assert result['iteration_count'] <= 3
        assert result['escalation_required'] is True
        print(f"Agent terminated correctly after {result['iteration_count']} iterations.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent_loop_protection())
