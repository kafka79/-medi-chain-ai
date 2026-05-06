import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sentence_transformers import SentenceTransformer

from src.agent.clinical_graph import ClinicalAgent
from src.data.pdf_parser import ClinicalPDFParser
from src.data.fhir_formatter import EHRGateway
from src.models.fusion import LateFusionModel
from src.models.uncertainty import UncertaintyEstimator
from src.rag.evaluator import RAGEvaluator
from src.vlm.visual_encoder import BiomedVisualEncoder
from src.utils.storage import LocalStorageProvider
from src.monitoring.drift_detector import DriftDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medi-chain-api")

# Global dependencies (Ready for DI)
storage = LocalStorageProvider()
drift_detector = DriftDetector()
ehr_gateway = EHRGateway()

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))

# Global semaphore for model inference
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def build_agent() -> ClinicalAgent:
    logger.info("Initializing models (BiomedVisualEncoder, SapBERT, LateFusion)...")
    encoder = BiomedVisualEncoder()
    parser = ClinicalPDFParser()
    text_encoder = SentenceTransformer(MODEL_NAME)
    fusion = LateFusionModel()
    uncertainty = UncertaintyEstimator(fusion)
    rag = RAGEvaluator(
        milvus_host=os.getenv("MILVUS_HOST", "localhost"),
        milvus_port=os.getenv("MILVUS_PORT", "19530"),
    )
    return ClinicalAgent(encoder, parser, rag, fusion, text_encoder, uncertainty)

async def cleanup_old_temp_files():
    """Background task to clean up storage older than 1 hour."""
    while True:
        try:
            now = time.time()
            root = Path(storage.root)
            if root.exists():
                for item in root.iterdir():
                    if item.is_dir() and (now - item.stat().st_mtime > 3600):
                        logger.info(f"Cleaning up old storage directory: {item}")
                        storage.delete(item.name)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
        await asyncio.sleep(600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize the agent
    app.state.agent = build_agent()
    # Start background cleanup
    cleanup_task = asyncio.create_task(cleanup_old_temp_files())
    yield
    cleanup_task.cancel()
    app.state.agent = None

def create_app() -> FastAPI:
    app = FastAPI(
        title="MEdi Chain AI - API", 
        version="1.3.0", 
        lifespan=lifespan,
        description="Enterprise-ready multimodal diagnostic API."
    )

    @app.post("/analyze")
    async def analyze_case(
        request: Request,
        background_tasks: BackgroundTasks,
        image: UploadFile = File(...),
        history: UploadFile = File(...),
    ):
        agent = request.app.state.agent
        if agent is None:
            raise HTTPException(status_code=503, detail="Models not loaded.")

        request_id = uuid.uuid4().hex
        img_rel = f"{request_id}/image_{Path(image.filename or 'upload').name}"
        pdf_rel = f"{request_id}/history_{Path(history.filename or 'upload').name}"

        try:
            # Use StorageProvider (addresses 'Stateful Temp Storage')
            img_path = storage.save(image.file, img_rel)
            pdf_path = storage.save(history.file, pdf_rel)
            
            async with inference_semaphore:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, agent.run, img_path, pdf_path)
                
            # Monitor for drift (addresses 'Model Drifting')
            drift_detector.add_prediction(result['diagnosis']['probabilities'])
            
            # Simulated EHR Push (addresses 'Integration Friction')
            background_tasks.add_task(ehr_gateway.push_report, str(result))
                
            return JSONResponse(content=result)
        except Exception as exc:
            logger.error(f"Analysis failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            background_tasks.add_task(storage.delete, request_id)

    @app.get("/health")
    async def health_check(request: Request):
        return {
            "status": "ok",
            "models_loaded": request.app.state.agent is not None,
            "concurrency_limit": MAX_CONCURRENT_REQUESTS,
        }

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
