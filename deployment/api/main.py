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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, BackgroundTasks, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import secrets
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agent.clinical_graph import ClinicalAgent
from src.data.pdf_parser import ClinicalPDFParser
from src.data.fhir_formatter import EHRGateway
from src.rag.evaluator import RAGEvaluator
from src.utils.storage import S3StorageProvider
from src.monitoring.drift_detector import DriftDetector
from src.utils.feedback_logger import FeedbackLogger
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medi-chain-api")

# Check if Redis is actually responsive before using it for rate limiting
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
use_redis = False
if redis_url.startswith("redis://"):
    try:
        import redis
        import urllib.parse
        parsed = urllib.parse.urlparse(redis_url)
        r = redis.Redis(host=parsed.hostname, port=parsed.port or 6379, socket_connect_timeout=1)
        r.ping()
        use_redis = True
        logger.info("Successfully connected to Redis for rate limiting.")
    except Exception as e:
        logger.warning(f"Redis not responsive ({e}). Falling back to in-memory rate limiting.")

if not use_redis:
    redis_url = "memory://"

limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url)

# Global dependencies (Ready for DI)
TEMP_ROOT = Path("temp/storage")

# Initialize storage provider: fall back to LocalStorageProvider if MinIO connection fails
try:
    s3_storage = S3StorageProvider(endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"))
    if s3_storage.client is not None:
        storage = s3_storage
        logger.info("Using S3/MinIO Storage Provider.")
    else:
        from src.utils.storage import LocalStorageProvider
        storage = LocalStorageProvider()
        logger.warning("MinIO client not initialized. Falling back to Local Storage Provider.")
except Exception as e:
    from src.utils.storage import LocalStorageProvider
    storage = LocalStorageProvider()
    logger.warning(f"Failed to initialize S3 storage ({e}). Falling back to Local Storage Provider.")

drift_detector = DriftDetector()
ehr_gateway = EHRGateway()
feedback_logger = FeedbackLogger()

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))

# Global semaphore for model inference
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)



API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    expected_key = os.getenv("API_KEY", "dev-secret-key-123")
    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

def build_agent() -> ClinicalAgent:
    logger.info("Initializing ClinicalAgent with remote inference API...")
    parser = ClinicalPDFParser()
    rag = RAGEvaluator(
        milvus_host=os.getenv("MILVUS_HOST", "localhost"),
        milvus_port=os.getenv("MILVUS_PORT", "19530"),
        inference_api_url=os.getenv("INFERENCE_API_URL", "http://inference-api:8001")
    )
    return ClinicalAgent(
        history_parser=parser, 
        rag_evaluator=rag,
        inference_api_url=os.getenv("INFERENCE_API_URL", "http://inference-api:8001")
    )

async def cleanup_old_temp_files():
    """Background task to clean up storage older than 1 hour."""
    while True:
        try:
            await asyncio.to_thread(storage.cleanup, max_age_seconds=3600)
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
    
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.post("/analyze")
    @limiter.limit("10/minute")
    async def analyze_case(
        request: Request,
        background_tasks: BackgroundTasks,
        image: UploadFile = File(...),
        history: UploadFile = File(...),
        api_key: str = Depends(verify_api_key)
    ):
        if image.content_type not in ["image/jpeg", "image/png", "application/dicom"]:
            raise HTTPException(415, "Unsupported image type. Use JPEG, PNG, or DICOM.")
        if history.content_type != "application/pdf":
            raise HTTPException(415, "Unsupported history document type. Use PDF.")
        agent = request.app.state.agent
        if agent is None:
            raise HTTPException(status_code=503, detail="Models not loaded.")

        request_id = uuid.uuid4().hex
        img_rel = f"{request_id}/image_{Path(image.filename or 'upload').name}"
        pdf_rel = f"{request_id}/history_{Path(history.filename or 'upload').name}"

        try:
            # Use StorageProvider (addresses 'Stateful Temp Storage')
            img_path = await asyncio.to_thread(storage.save, image.file, img_rel)
            pdf_path = await asyncio.to_thread(storage.save, history.file, pdf_rel)
            
            # Hydrate S3 blobs to local filesystem for ML models
            local_img_path = await asyncio.to_thread(storage.load, img_path)
            local_pdf_path = await asyncio.to_thread(storage.load, pdf_path)
            
            async with inference_semaphore:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, agent.run, local_img_path, local_pdf_path)
                
            # Monitor for drift (prediction drift and covariate shift)
            background_tasks.add_task(drift_detector.add_prediction, result['diagnosis']['probabilities'], result.get('visual_features'))
            
            # EHR Push is officially handled by the Presentation Layer (app.py)
            # to prevent duplicate background transmissions.
                
            return JSONResponse(content=result)
        except Exception as exc:
            logger.error(f"Analysis failed: {exc}")
            return JSONResponse(status_code=500, content={"detail": f"Analysis failed: {exc}"})
        finally:
            background_tasks.add_task(storage.delete, request_id)
            if 'local_img_path' in locals() and os.path.exists(local_img_path):
                try: os.unlink(local_img_path)
                except Exception: pass
            if 'local_pdf_path' in locals() and os.path.exists(local_pdf_path):
                try: os.unlink(local_pdf_path)
                except Exception: pass

    class FeedbackPayload(BaseModel):
        session_id: str
        verdict: str
        notes: str
        diagnosis: dict
        history_metadata: dict
        doctor_id: str = "dr-anonymous"

    @app.post("/feedback")
    async def receive_feedback(
        payload: FeedbackPayload,
        api_key: str = Depends(verify_api_key)
    ):
        try:
            path = await asyncio.to_thread(
                feedback_logger.log_feedback,
                session_id=payload.session_id,
                verdict=payload.verdict,
                notes=payload.notes,
                diagnosis=payload.diagnosis,
                history_metadata=payload.history_metadata
            )
            return JSONResponse(content={"status": "success", "saved_path": str(path)})
        except Exception as e:
            logger.error(f"Feedback logging failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to log feedback: {e}")

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
