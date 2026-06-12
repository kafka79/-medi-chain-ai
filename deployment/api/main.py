import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import shutil
import tempfile
import sys
import time
import uuid
import hashlib
import json
from datetime import datetime, timezone
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
from pydantic import BaseModel, Field, field_validator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medi-chain-api")
audit_logger = logging.getLogger("medi-chain-api.audit")
audit_logger.setLevel(logging.INFO)


def _configure_audit_logger():
    """Attach a JSON-lines file handler for request/response audit events."""
    default_to_file = "false" if os.getenv("TESTING") == "true" else "true"
    if os.getenv("API_AUDIT_LOG_TO_FILE", default_to_file).lower() != "true":
        return

    audit_path = Path(os.getenv("API_AUDIT_LOG_PATH", "outputs/audit/api_audit.log"))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = str(audit_path.resolve())
    for handler in audit_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved_path:
            return

    handler = logging.FileHandler(resolved_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(handler)


def _write_audit_event(event: dict):
    audit_logger.info(json.dumps(event, separators=(",", ":"), default=str))


def _file_sha256(path: str) -> Optional[str]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_model_metadata() -> dict:
    return {
        "api_version": APP_VERSION,
        "checkpoint": MODEL_CHECKPOINT,
        "checkpoint_sha256": _file_sha256(MODEL_CHECKPOINT),
        "visual_encoder": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "text_encoder": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    }


_configure_audit_logger()

# Flaw #2 Fix: Fail fast if API_KEY is not set in production
_api_key = os.getenv("API_KEY")
if not _api_key and os.getenv("TESTING") != "true" and os.getenv("STORAGE_MODE") != "local":
    raise RuntimeError(
        "CRITICAL: API_KEY environment variable is not set. "
        "Refusing to start with default credentials in a production-like environment. "
        "Set API_KEY in your .env file."
    )

# Check if Redis is actually responsive before using it for rate limiting
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
use_redis = False
redis_client = None
is_production = os.getenv("TESTING") != "true" and os.getenv("STORAGE_MODE") != "local"

if redis_url.startswith("redis://") or redis_url.startswith("rediss://"):
    try:
        import redis
        redis_client = redis.from_url(redis_url, socket_connect_timeout=1)
        redis_client.ping()
        use_redis = True
        logger.info("Successfully connected to Redis for rate limiting.")
    except Exception as e:
        redis_client = None
        if not is_production:
            logger.warning(f"Redis not responsive ({e}). Falling back to in-memory rate limiting for local dev/testing.")
        else:
            logger.critical(f"Redis not responsive ({e}) in production environment. Aborting startup to prevent un-synchronized rate limiting.")
            raise RuntimeError(f"Redis rate limiter connection failed in production: {e}")
else:
    if is_production:
        logger.critical("REDIS_URL must start with 'redis://' or 'rediss://' in production. Memory fallback is disabled.")
        raise RuntimeError("Redis is required for rate limiting in production.")

if not use_redis:
    redis_url = "memory://"

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=redis_url,
    enabled=os.getenv("TESTING") != "true",
)

# Global dependencies (Ready for DI)
TEMP_ROOT = Path("temp/storage")

# Initialize storage provider: fail fast on MinIO connection failures in production-like environments
storage_mode = os.getenv("STORAGE_MODE", "s3")
if storage_mode == "local" or os.getenv("TESTING") == "true":
    from src.utils.storage import LocalStorageProvider
    storage = LocalStorageProvider()
    logger.info("Using Local Storage Provider for testing/local development.")
else:
    try:
        s3_storage = S3StorageProvider(endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"))
        if s3_storage.client is not None:
            storage = s3_storage
            logger.info("Using S3/MinIO Storage Provider.")
        else:
            raise RuntimeError("MinIO client not initialized. Cannot proceed in S3 mode.")
    except Exception as e:
        logger.critical(f"Failed to initialize S3 storage ({e}). Aborting startup.")
        raise RuntimeError(f"S3 storage initialization failed: {e}")

drift_detector = DriftDetector()
ehr_gateway = EHRGateway()
feedback_logger = FeedbackLogger(redis_client=redis_client, storage_provider=storage)

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))

# Global semaphore for model inference
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Flaw #17: Application-level version metadata for audit trail
APP_VERSION = "1.3.0"
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "models/fusion_model.pt")

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    # Flaw #2 Fix: No hardcoded default. If env var is missing in dev/test, use a test-only key.
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        if os.getenv("TESTING") == "true":
            expected_key = "test-key-for-ci"
        else:
            raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY not set.")
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

# Flaw #11 Fix: Circuit breaker — stop cleanup loop after consecutive failures
MAX_CLEANUP_FAILURES = 5

async def cleanup_old_temp_files():
    """Background task to clean up storage older than 1 hour.
    Flaw #11 Fix: Stops after MAX_CLEANUP_FAILURES consecutive errors instead of spinning forever."""
    consecutive_failures = 0
    while True:
        try:
            await asyncio.to_thread(storage.cleanup, max_age_seconds=3600)
            consecutive_failures = 0  # Reset on success
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Error in cleanup task ({consecutive_failures}/{MAX_CLEANUP_FAILURES}): {e}")
            if consecutive_failures >= MAX_CLEANUP_FAILURES:
                logger.critical(
                    f"Cleanup task circuit breaker tripped after {MAX_CLEANUP_FAILURES} consecutive failures. "
                    f"Stopping cleanup loop. Manual intervention required."
                )
                return  # Exit the loop permanently
        await asyncio.sleep(600)

async def reconcile_dlq_task():
    """Background task that polls the Redis DLQ 'medi_chain:dlq' and retries pushes to the EHR."""
    if not use_redis:
        logger.info("[DLQ Reconciler] Redis is not enabled. Skipping DLQ reconciliation worker.")
        return

    import redis
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    try:
        r = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
    except Exception as e:
        logger.error(f"[DLQ Reconciler] Failed to connect to Redis: {e}")
        return
        
    logger.info("[DLQ Reconciler] Started background DLQ reconciliation worker.")
    
    while True:
        try:
            item_json = r.lpop("medi_chain:dlq")
            if item_json:
                try:
                    item = json.loads(item_json)
                    payload = item.get("payload")
                    if payload:
                        if not isinstance(payload, str):
                            fhir_json = json.dumps(payload)
                        else:
                            fhir_json = payload
                        
                        logger.info(f"[DLQ Reconciler] Attempting to reconcile report from DLQ...")
                        loop = asyncio.get_running_loop()
                        success = await loop.run_in_executor(None, ehr_gateway.push_report, fhir_json)
                        if success:
                            logger.info("[DLQ Reconciler] Successfully reconciled report.")
                        else:
                            logger.warning("[DLQ Reconciler] Re-push failed. Payload rewritten to DLQ.")
                except Exception as parse_err:
                    logger.error(f"[DLQ Reconciler] Failed to process DLQ item: {parse_err}. Item content: {item_json}")
            else:
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("[DLQ Reconciler] DLQ reconciler task cancelled.")
            break
        except Exception as e:
            logger.error(f"[DLQ Reconciler] Error in DLQ reconciliation loop: {e}")
            await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize the agent
    app.state.agent = build_agent()
    # Start background cleanup
    cleanup_task = asyncio.create_task(cleanup_old_temp_files())
    # Start background DLQ reconciler
    dlq_task = asyncio.create_task(reconcile_dlq_task())
    yield
    cleanup_task.cancel()
    dlq_task.cancel()
    app.state.agent = None

def create_app() -> FastAPI:
    app = FastAPI(
        title="MEdi Chain AI - API", 
        version=APP_VERSION, 
        lifespan=lifespan,
        description="Enterprise-ready multimodal diagnostic API."
    )
    
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.middleware("http")
    async def audit_requests(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = None
        status_code = 500
        error_type = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            _write_audit_event({
                "event": "http_request",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "status_code": status_code,
                "duration_ms": duration_ms,
                "content_length": request.headers.get("content-length"),
                "response_content_length": response.headers.get("content-length") if response else None,
                "error_type": error_type,
            })

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

        request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        request_temp_dir = None

        try:
            # Ensure the base temp dir exists
            TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            # Create a request-specific temporary folder in temp/storage
            request_temp_dir = tempfile.mkdtemp(dir=str(TEMP_ROOT))
            
            # Form clean request-scoped local file paths
            local_img_path = os.path.join(request_temp_dir, Path(image.filename or "image.jpg").name)
            local_pdf_path = os.path.join(request_temp_dir, Path(history.filename or "history.pdf").name)
            
            # Save files directly into local ephemeral storage
            with open(local_img_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            with open(local_pdf_path, "wb") as f:
                shutil.copyfileobj(history.file, f)
            
            async with inference_semaphore:
                result = await agent.run(local_img_path, local_pdf_path)
                
            # Monitor for drift (prediction drift and covariate shift)
            background_tasks.add_task(drift_detector.add_prediction, result['diagnosis']['probabilities'], result.get('visual_features'))
            
            # Flaw #5 Fix: Surface escalation_required as a first-class response field
            escalation = result.get('escalation_required', False)

            # Flaw #17 Fix: Include model version metadata in every response for audit trail
            response_payload = {
                "request_id": request_id,
                "diagnosis": result.get("diagnosis", {}),
                "confidence": result.get("confidence", 0.0),
                "heatmap_base64": result.get("heatmap_base64", ""),
                "pubmed_citations": result.get("pubmed_citations", []),
                "escalation_required": escalation,
                "iteration_count": result.get("iteration_count", 0),
                "model_metadata": get_model_metadata(),
            }

            # Flaw #5 Fix: Set explicit HTTP header when human review is required
            headers = {}
            if escalation:
                headers["X-Requires-Human-Review"] = "true"
                logger.warning(f"[{request_id}] Escalation triggered — insufficient evidence for automated diagnosis.")

            return JSONResponse(content=response_payload, headers=headers)
        except Exception as exc:
            logger.error(f"Analysis failed: {exc}")
            return JSONResponse(status_code=500, content={"detail": f"Analysis failed: {exc}"})
        finally:
            if request_temp_dir and os.path.exists(request_temp_dir):
                try:
                    shutil.rmtree(request_temp_dir, ignore_errors=True)
                except Exception as cleanup_err:
                    logger.error(f"Failed to clean up temporary directory {request_temp_dir}: {cleanup_err}")

    class FeedbackPayload(BaseModel):
        session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
        verdict: str = Field(..., min_length=1, max_length=32)
        notes: str = Field("", max_length=2000)
        diagnosis: dict
        history_metadata: dict
        doctor_id: str = Field("dr-anonymous", max_length=128)

        @field_validator("verdict")
        @classmethod
        def verdict_must_be_known(cls, value: str) -> str:
            normalized = value.strip().lower()
            allowed = {"agree", "disagree", "uncertain", "needs_review", "match", "mismatch"}
            if normalized not in allowed:
                raise ValueError(f"verdict must be one of: {', '.join(sorted(allowed))}")
            return normalized

        @field_validator("notes")
        @classmethod
        def notes_must_not_be_blank_noise(cls, value: str) -> str:
            return value.strip()

        @field_validator("doctor_id")
        @classmethod
        def doctor_id_must_use_doctor_prefix(cls, value: str) -> str:
            if not value.startswith("dr-") or not all(ch.isalnum() or ch in "-_." for ch in value):
                raise ValueError("doctor_id must start with 'dr-'")
            return value

    @app.post("/feedback")
    @limiter.limit("30/minute")  # Flaw #4 Fix: Rate limit feedback endpoint to prevent spam/poisoning
    async def receive_feedback(
        request: Request,
        background_tasks: BackgroundTasks,
        payload: FeedbackPayload,
        api_key: str = Depends(verify_api_key)
    ):
        try:
            agreement = payload.verdict in {"agree", "match"}
            path = await asyncio.to_thread(
                feedback_logger.log_feedback,
                session_id=payload.session_id,
                verdict=payload.verdict,
                notes=payload.notes,
                diagnosis=payload.diagnosis,
                history_metadata=payload.history_metadata
            )
            background_tasks.add_task(drift_detector.update_feedback_summary, agreement)
            _write_audit_event({
                "event": "feedback_received",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": getattr(request.state, "request_id", None),
                "session_id_hash": hashlib.sha256(payload.session_id.encode("utf-8")).hexdigest(),
                "doctor_id_hash": hashlib.sha256(payload.doctor_id.encode("utf-8")).hexdigest(),
                "verdict": payload.verdict,
                "agreement": agreement,
            })
            return JSONResponse(content={"status": "success", "saved_path": str(path)})
        except Exception as e:
            logger.error(f"Feedback logging failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to log feedback: {e}")

    @app.get("/feedback/discrepancies")
    @limiter.limit("10/minute")
    async def get_discrepancies(
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """Retrieves aggregated discrepancy records across all stateless replicas using Redis/S3."""
        try:
            records = []
            if redis_client is not None:
                # Load from shared Redis list
                raw_records = redis_client.lrange("medi_chain:feedback:records", 0, -1)
                for r_raw in raw_records:
                    rec = json.loads(r_raw)
                    if rec.get("verdict") in {"disagree", "mismatch"}:
                        records.append(rec)
            else:
                # Fallback to reading the local CSV if Redis isn't configured
                import csv
                csv_path = feedback_logger.csv_path
                if csv_path.exists():
                    with csv_path.open("r", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        for row in reader:
                            if row.get("verdict") in {"disagree", "mismatch"}:
                                records.append(row)
            return JSONResponse(content={"discrepancies": records})
        except Exception as e:
            logger.error(f"Failed to load discrepancies: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load discrepancies: {e}")

    @app.get("/health")
    async def health_check(request: Request):
        return {
            "status": "ok",
            "models_loaded": request.app.state.agent is not None,
            "concurrency_limit": MAX_CONCURRENT_REQUESTS,
            "version": APP_VERSION,
        }

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
