from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
from src.agent.clinical_graph import ClinicalAgent
from src.vlm.visual_encoder import BiomedVisualEncoder
from src.data.pdf_parser import ClinicalPDFParser
from src.rag.evaluator import RAGEvaluator
from src.models.fusion import LateFusionModel
from src.models.uncertainty import UncertaintyEstimator
from src.evaluation.report_generator import ClinicalReportGenerator
from sentence_transformers import SentenceTransformer

app = FastAPI(title="MEdi Chain AI - API", version="1.0.0")

# Lazy load models
encoder = BiomedVisualEncoder()
parser = ClinicalPDFParser()
text_encoder = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
fusion = LateFusionModel()
uncertainty = UncertaintyEstimator(fusion)
rag = RAGEvaluator()
agent = ClinicalAgent(encoder, parser, rag, fusion, text_encoder, uncertainty)

@app.post("/analyze")
async def analyze_case(image: UploadFile = File(...), history: UploadFile = File(...)):
    """
    Multimodal analysis endpoint.
    Accepts an X-ray image and a clinical history PDF.
    """
    # Save files locally
    img_path = f"temp_{image.filename}"
    pdf_path = f"temp_{history.filename}"
    
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(history.file, buffer)
        
    try:
        # Run agent
        result = agent.run(img_path, pdf_path)
        
        # Clean up
        os.remove(img_path)
        os.remove(pdf_path)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
