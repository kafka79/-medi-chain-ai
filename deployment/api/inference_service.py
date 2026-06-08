import os
import sys
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import secrets
import torch
import uvicorn
import shutil
import tempfile
import asyncio
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.vlm.visual_encoder import BiomedVisualEncoder
from sentence_transformers import SentenceTransformer
from src.models.fusion import LateFusionModel

app = FastAPI(
    title="MEdi Chain AI - Inference Microservice",
    description="Dedicated GPU-bound microservice for raw AI inference, decoupling heavy PyTorch models from the Web API."
)

inference_semaphore = asyncio.Semaphore(1)

class InferenceService:
    def __init__(self):
        self.encoder = BiomedVisualEncoder()
        self.text_encoder = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        
        self.fusion = LateFusionModel()
        checkpoint_path = "models/fusion_model.pt"
        if os.path.exists(checkpoint_path):
            self.fusion.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            print(f"Loaded weights from {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint {checkpoint_path} not found! Using random weights.")
            
        self.fusion = self.fusion.to(self.encoder.device)
        self.fusion.eval()
        from src.models.uncertainty import UncertaintyEstimator
        self.uncertainty = UncertaintyEstimator(self.fusion)

service = None

@app.on_event("startup")
def load_models():
    global service
    service = InferenceService()

class TextPayload(BaseModel):
    text: str

class EstimatePayload(BaseModel):
    visual_features: Any
    text_features: Any
    num_passes: int = 20

INTERNAL_API_KEY_HEADER = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)

async def verify_internal_api_key(api_key: str = Security(INTERNAL_API_KEY_HEADER)):
    expected_key = os.getenv("INTERNAL_API_KEY", "internal-secret-token")
    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=403, detail="Invalid Internal API Key")
    return api_key

@app.post("/encode/image")
async def encode_image(image: UploadFile = File(...), api_key: str = Depends(verify_internal_api_key)):
    if not service:
        raise HTTPException(503, "Models not loaded")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name
        
    try:
        async with inference_semaphore:
            features = service.encoder.encode_image(tmp_path)
        return {"features": features.tolist()}
    finally:
        os.unlink(tmp_path)

@app.post("/encode/text")
async def encode_text(payload: TextPayload, api_key: str = Depends(verify_internal_api_key)):
    if not service:
        raise HTTPException(503, "Models not loaded")
    
    async with inference_semaphore:
        emb = service.text_encoder.encode([payload.text], convert_to_tensor=True)
    return {"embeddings": emb.tolist()}

@app.post("/estimate")
async def estimate_uncertainty(payload: EstimatePayload, api_key: str = Depends(verify_internal_api_key)):
    import numpy as np
    if not service:
        raise HTTPException(503, "Models not loaded")
        
    v = torch.tensor(payload.visual_features, dtype=torch.float32, device=service.encoder.device)
    if v.ndim == 1:
        v = v.unsqueeze(0)
        
    t = torch.tensor(payload.text_features, dtype=torch.float32, device=service.encoder.device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    
    async with inference_semaphore:
        results = service.uncertainty.estimate_uncertainty(v, t, num_passes=payload.num_passes)
        
    # Convert numpy/torch arrays to lists for JSON
    clean_results = {}
    for k, val in results.items():
        if isinstance(val, (torch.Tensor, np.ndarray)):
            clean_results[k] = val.tolist()
        else:
            clean_results[k] = val
            
    return clean_results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
