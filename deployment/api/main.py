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
from typing import Optional, Any, Dict
import re
from dataclasses import dataclass, asdict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, BackgroundTasks, Security, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import APIKeyHeader, SecurityScopes
import secrets
from filelock import FileLock
import uvicorn
import contextvars
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, field_validator

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config.settings import (
    validate_production_config, dump_all_configs,
    get_clinical_thresholds, get_inference_config, get_semaphore_config,
    get_drift_config, get_redis_config, get_storage_config,
    get_api_config, get_security_config, get_app_settings
)
from src.agent.clinical_graph import ClinicalAgent
from src.data.pdf_parser import ClinicalPDFParser
from src.data.privacy_scrubber import PrivacyScrubber
from src.data.fhir_formatter import EHRGateway, MockEHRGateway
from src.rag.evaluator import RAGEvaluator
from src.utils.storage import S3StorageProvider, LocalStorageProvider
from src.monitoring.drift_detector import DriftDetector
from src.utils.feedback_logger import FeedbackLogger
from src.utils.secrets_manager import SecretsManager
from src.utils.security import encrypt_payload, decrypt_payload
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Prometheus metrics
from prometheus_client import REGISTRY

def _get_or_create_counter(name, documentation, labelnames=()):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation, labelnames)

def _get_or_create_histogram(name, documentation, buckets):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, documentation, buckets=buckets)

PROM_CASES_PROCESSED = _get_or_create_counter("medi_chain_cases_processed_total", "Total number of clinical cases processed.")
PROM_ESCALATIONS = _get_or_create_counter("medi_chain_escalations_total", "Total number of clinical cases escalated.")
PROM_FAILURES = _get_or_create_counter("medi_chain_failures_total", "Total number of clinical cases failed.")
PROM_FEEDBACK = _get_or_create_counter("medi_chain_feedback_total", "Clinician feedback verdicts.", ["verdict"])
PROM_SIGN_OFF_TIME = _get_or_create_histogram(
    "medi_chain_sign_off_time_seconds", 
    "Radiologist sign-off time in seconds.",
    buckets=(10, 30, 45, 60, 120, 180, 300, 600, 1200, 3600)
)
PROM_PROCESSING_TIME = _get_or_create_histogram(
    "medi_chain_processing_duration_seconds",
    "Processing time of the /analyze endpoint in seconds.",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60)
)
PROM_GPU_HEALTH = _get_or_create_histogram(
    "medi_chain_gpu_health",
    "GPU health metrics",
    buckets=(0, 1, 2, 3, 4, 5)
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medi-chain-api")
audit_logger = logging.getLogger("medi-chain-api.audit")
audit_logger.setLevel(logging.INFO)


def _configure_audit_logger():
    audit_logger.handlers = []
    audit_logger.propagate = False
    
    log_mode = os.getenv("API_AUDIT_LOG_MODE", "stdout").lower()
    if log_mode == "file":
        audit_path = Path(os.getenv("API_AUDIT_LOG_PATH", "outputs/audit/api_audit.log"))
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(audit_path.resolve()), encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
        
    handler.setFormatter(logging.Formatter("[AUDIT] %(message)s"))
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


def save_heatmap_from_base64(heatmap_base64: str, job_id: str) -> Optional[str]:
    if not heatmap_base64:
        return None
    try:
        import base64
        import io
        from pathlib import Path
        
        if "," in heatmap_base64:
            header, data = heatmap_base64.split(",", 1)
        else:
            data = heatmap_base64
        img_bytes = base64.b64decode(data)
        
        relative_path = f"heatmaps/{job_id}.png"
        if storage_mode == "s3":
            storage.save(io.BytesIO(img_bytes), relative_path)
            logger.info(f"Saved heatmap PNG to S3: {relative_path}")
            return relative_path
        else:
            output_dir = Path("outputs/heatmaps")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{job_id}.png"
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            logger.info(f"Saved heatmap PNG to persistent path: {output_path}")
            return str(output_path)
    except Exception as e:
        logger.error(f"Failed to save heatmap PNG for job {job_id}: {e}")
        return None


_configure_audit_logger()

# Load unified configuration
clinical_config = get_clinical_thresholds()
inference_config = get_inference_config()
semaphore_config = get_semaphore_config()
drift_config = get_drift_config()
redis_config = get_redis_config()
storage_config = get_storage_config()
api_config = get_api_config()
security_config = get_security_config()
app_settings = get_app_settings()

# Validate production config
errors = validate_production_config()
if errors:
    for err in errors:
        logger.critical(f"CONFIG ERROR: {err}")
    if not app_settings.testing and app_settings.storage_mode != "local":
        raise RuntimeError("Production config validation failed: " + "; ".join(errors))

# Extract commonly used settings
TESTING = app_settings.testing
STORAGE_MODE = app_settings.storage_mode
MODEL_CHECKPOINT = app_settings.model_checkpoint
APP_VERSION = api_config.version
MAX_CONCURRENT_REQUESTS = api_config.max_concurrent_requests
EHR_GATEWAY_URL = app_settings.ehr_gateway_url
DRIFT_ALERT_WEBHOOK_URL = drift_config.alert_webhook_url
DLQ_DIR = app_settings.dlq_dir
DRIFT_CACHE_DIR = app_settings.drift_cache_dir

# API Key validation (fail-fast in production)
_api_key = SecretsManager.get_secret("API_KEY")
if not _api_key and not TESTING and STORAGE_MODE != "local":
    raise RuntimeError(
        "CRITICAL: API_KEY environment variable is not set. "
        "Refusing to start with default credentials in a production-like environment. "
        "Set API_KEY in your .env file."
    )

# DLQ Encryption Key validation
_dlq_encryption_key = security_config.dlq_encryption_key
if not _dlq_encryption_key and not TESTING and STORAGE_MODE != "local":
    raise RuntimeError(
        "CRITICAL: SECURITY_DLQ_ENCRYPTION_KEY environment variable is not set. "
        "Refusing to start with default key fallback in a production-like environment. "
        "Set SECURITY_DLQ_ENCRYPTION_KEY in your .env file."
    )

# DICOM Encryption Key validation (separate from DLQ)
_dicom_encryption_key = security_config.dicom_encryption_key
if not _dicom_encryption_key and not TESTING and STORAGE_MODE != "local":
    raise RuntimeError(
        "CRITICAL: SECURITY_DICOM_ENCRYPTION_KEY environment variable is not set. "
        "Set SECURITY_DICOM_ENCRYPTION_KEY in your .env file."
    )

# Internal API Key validation
_internal_api_key = security_config.internal_api_key or app_settings.internal_api_key
if not _internal_api_key and not TESTING:
    raise RuntimeError("INTERNAL_API_KEY environment variable is required.")

# Redis connection with HA support
redis_url = redis_config.url
use_redis = False
redis_client = None
is_production = not TESTING and STORAGE_MODE != "local"

sentinel_hosts_str = redis_config.sentinel_hosts
cluster_nodes_str = redis_config.cluster_nodes

if sentinel_hosts_str or cluster_nodes_str or redis_url.startswith("redis://") or redis_url.startswith("rediss://"):
    try:
        import redis
        if sentinel_hosts_str:
            from redis.sentinel import Sentinel
            sentinels = []
            for s in sentinel_hosts_str.split(","):
                if ":" in s:
                    host, port = s.split(":")
                    sentinels.append((host, int(port)))
                else:
                    sentinels.append((s, 26379))
            service_name = redis_config.sentinel_service_name
            sentinel_client = Sentinel(sentinels, socket_connect_timeout=redis_config.socket_connect_timeout, decode_responses=True)
            redis_client = sentinel_client.master_for(service_name, socket_connect_timeout=redis_config.socket_connect_timeout, decode_responses=True)
            logger.info("Successfully configured Redis Sentinel for High Availability.")
        elif cluster_nodes_str:
            from redis.cluster import RedisCluster, ClusterNode
            nodes = []
            for node in cluster_nodes_str.split(","):
                if ":" in node:
                    host, port = node.split(":")
                    nodes.append(ClusterNode(host, int(port)))
                else:
                    nodes.append(ClusterNode(node, 6379))
            redis_client = RedisCluster(startup_nodes=nodes, socket_connect_timeout=redis_config.socket_connect_timeout, decode_responses=True)
            logger.info("Successfully configured Redis Cluster for High Availability.")
        else:
            redis_client = redis.from_url(redis_url, socket_connect_timeout=redis_config.socket_connect_timeout)
            
        redis_client.ping()
        use_redis = True
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        redis_client = None
        if not is_production:
            logger.warning(f"Redis not responsive ({e}). Falling back to in-memory rate limiting for local dev/testing.")
        else:
            logger.critical(f"Redis not responsive ({e}) in production environment. Aborting startup to prevent un-synchronized rate limiting.")
            raise RuntimeError(f"Redis rate limiter connection failed in production: {e}")
else:
    if is_production:
        logger.critical("REDIS_URL must start with 'redis://' or 'rediss://', or Sentinel/Cluster variables must be set in production.")
        raise RuntimeError("Redis is required for rate limiting in production.")

if not use_redis:
    redis_url = "memory://"

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=redis_url,
    enabled=not TESTING,
)

# Global dependencies
TEMP_ROOT = Path("temp/storage")

# Initialize storage provider
storage_mode = STORAGE_MODE
if storage_mode == "local" or TESTING:
    storage = LocalStorageProvider()
    logger.info("Using Local Storage Provider for testing/local development.")
else:
    try:
        s3_storage = S3StorageProvider(endpoint=storage_config.minio_endpoint)
        if s3_storage.client is not None:
            storage = s3_storage
            logger.info("Using S3/MinIO Storage Provider.")
        else:
            raise RuntimeError("MinIO client not initialized. Cannot proceed in S3 mode.")
    except Exception as e:
        logger.critical(f"Failed to initialize S3 storage ({e}). Aborting startup.")
        raise RuntimeError(f"S3 storage initialization failed: {e}")

drift_detector = DriftDetector()

if "mock-ehr-gateway.internal" in EHR_GATEWAY_URL:
    ehr_gateway = MockEHRGateway(endpoint_url=EHR_GATEWAY_URL)
else:
    ehr_gateway = EHRGateway(endpoint_url=EHR_GATEWAY_URL)

feedback_logger = FeedbackLogger(redis_client=redis_client, storage_provider=storage)
gateway_scrubber = PrivacyScrubber()
gateway_pdf_parser = ClinicalPDFParser()

SEMAPHORE_WAITERS = {}
SEMAPHORE_LISTENER_TASKS = {}


async def _start_semaphore_listener(r_client, semaphore_name: str):
    """Background task to listen for semaphore release events in Redis and notify local waiters."""
    pubsub_name = f"{semaphore_name}:released"
    try:
        pubsub = r_client.pubsub()
        pubsub.subscribe(pubsub_name)
        
        loop = asyncio.get_running_loop()
        while True:
            message = await loop.run_in_executor(
                None,
                lambda: pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            )
            if message:
                waiters = SEMAPHORE_WAITERS.get(semaphore_name, [])
                SEMAPHORE_WAITERS[semaphore_name] = []
                for event in waiters:
                    if not event.is_set():
                        event.set()
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in semaphore listener for {semaphore_name}: {e}")
        waiters = SEMAPHORE_WAITERS.get(semaphore_name, [])
        SEMAPHORE_WAITERS[semaphore_name] = []
        for event in waiters:
            if not event.is_set():
                event.set()
    finally:
        try:
            pubsub.unsubscribe(pubsub_name)
            pubsub.close()
        except Exception:
            pass
        SEMAPHORE_LISTENER_TASKS.pop(semaphore_name, None)


class RedisDistributedSemaphore:
    """Distributed semaphore backed by Redis sorted-set leases with proper lease TTL management."""
    
    def __init__(self, r_client, name: str, limit: int):
        self.orig_redis = r_client
        self.redis = r_client
        self.name = f"medi_chain:semaphore:{name}:leases"
        self.limit = limit
        self.local_sem = asyncio.Semaphore(limit)
        
        expected_replicas = int(os.getenv("EXPECTED_NUM_REPLICAS", "5"))
        safe_fallback = max(1, limit // expected_replicas)
        fallback_limit = int(os.getenv("MAX_CONCURRENT_REQUESTS_FALLBACK", str(safe_fallback)))
        self.fallback_sem = asyncio.Semaphore(fallback_limit)
        
        self.client_id_var = contextvars.ContextVar(f"sem_client_id_{name}", default=None)
        self.refresher_task = None
        self.last_reconnect_attempt = 0
        self.lease_ttl = semaphore_config.lease_ttl_seconds
        self.refresh_interval = max(1, self.lease_ttl // 3)
        self.reconnect_cooldown = semaphore_config.reconnect_cooldown_seconds
        self.max_refresh_retries = semaphore_config.max_lease_refresh_retries

    async def _refresh_lease_loop(self, client_id: str):
        try:
            while True:
                await asyncio.sleep(self.refresh_interval)
                if self.redis is not None:
                    loop = asyncio.get_running_loop()
                    for attempt in range(self.max_refresh_retries):
                        try:
                            await loop.run_in_executor(
                                None,
                                lambda: self.redis.zadd(self.name, {client_id: time.time()})
                            )
                            break
                        except Exception as e:
                            if attempt == self.max_refresh_retries - 1:
                                logger.warning(f"Failed to refresh lease after {self.max_refresh_retries} attempts: {e}")
                                self.redis = None
                            else:
                                await asyncio.sleep(0.5 * (attempt + 1))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Lease refresh loop error: {e}")

    async def __aenter__(self):
        # Self-healing check
        if self.redis is None and self.orig_redis is not None:
            now = time.time()
            if now - self.last_reconnect_attempt > self.reconnect_cooldown:
                self.last_reconnect_attempt = now
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.orig_redis.ping)
                    self.redis = self.orig_redis
                    logger.info("Redis connection self-healed, returning to distributed semaphore.")
                except Exception as reconnect_err:
                    logger.warning(f"Redis self-healing reconnect attempt failed: {reconnect_err}")

        if self.redis is None:
            await self.fallback_sem.acquire()
            return self

        await self.local_sem.acquire()
        client_id = uuid.uuid4().hex
        self.client_id_var.set(client_id)
        
        if self.name not in SEMAPHORE_LISTENER_TASKS:
            task = asyncio.create_task(_start_semaphore_listener(self.redis, self.name))
            SEMAPHORE_LISTENER_TASKS[self.name] = task
            
        acquire_script = f"""
        local leases_key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local lease_ttl = tonumber(ARGV[3])
        local client_id = ARGV[4]

        redis.call('ZREMRANGEBYSCORE', leases_key, '-inf', now - lease_ttl)
        if redis.call('ZCARD', leases_key) < limit then
            redis.call('ZADD', leases_key, now, client_id)
            return 1
        end
        return 0
        """
        
        loop = asyncio.get_running_loop()
        while True:
            try:
                now = time.time()
                res = await loop.run_in_executor(
                    None,
                    lambda: self.redis.eval(acquire_script, 1, self.name, self.limit, now, self.lease_ttl, client_id)
                )
                if res == 1:
                    break
            except Exception as e:
                logger.error(f"Redis semaphore failed, degrading to local fallback: {e}")
                self.redis = None
                self.client_id_var.set(None)
                _send_system_alert(
                    "Redis Semaphore Connection Outage",
                    f"Degrading to local fallback semaphore: {e}"
                )
                await self.fallback_sem.acquire()
                self.local_sem.release()
                return self
            
            event = asyncio.Event()
            if self.name not in SEMAPHORE_WAITERS:
                SEMAPHORE_WAITERS[self.name] = []
            SEMAPHORE_WAITERS[self.name].append(event)
            
            try:
                await asyncio.wait_for(event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            
        self.refresher_task = asyncio.create_task(self._refresh_lease_loop(client_id))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        client_id = self.client_id_var.get()
        if client_id:
            self.client_id_var.set(None)
            if self.refresher_task:
                self.refresher_task.cancel()
                try:
                    await self.refresher_task
                except asyncio.CancelledError:
                    pass
            if self.redis is not None:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self.redis.pipeline()
                        .zrem(self.name, client_id)
                        .publish(f"{self.name}:released", "released")
                        .execute()
                    )
                except Exception as e:
                    logger.error(f"Redis release failed: {e}")
            self.local_sem.release()
        else:
            self.fallback_sem.release()


# Global semaphore for model inference with proper lease TTL
inference_semaphore = RedisDistributedSemaphore(
    redis_client if use_redis else None, 
    "inference", 
    MAX_CONCURRENT_REQUESTS
)


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
IDEMPOTENCY_KEY_HEADER = APIKeyHeader(name="X-Idempotency-Key", auto_error=False)


async def verify_api_key(security_scopes: SecurityScopes, api_key: str = Security(API_KEY_HEADER)):
    import hashlib
    expected_key = SecretsManager.get_secret("API_KEY")
    if not expected_key:
        if TESTING:
            expected_key = "test-key-for-ci"
        else:
            raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY not set.")

    key_scopes_map = {}
    if expected_key:
        h_expected = hashlib.sha256(expected_key.encode("utf-8")).hexdigest()
        key_scopes_map[h_expected] = ["cases:write", "cases:read", "metrics:read", "feedback:write", "feedback:read"]

    config_json = security_config.api_keys_config
    if config_json:
        try:
            config = json.loads(config_json)
            for k_hash, scopes in config.get("keys", {}).items():
                key_scopes_map[k_hash] = scopes
        except Exception as e:
            logger.error(f"Failed to parse API_KEYS_CONFIG: {e}")

    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header is missing")

    h_incoming = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    if h_incoming not in key_scopes_map:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    allowed_scopes = key_scopes_map[h_incoming]
    for scope in security_scopes.scopes:
        if scope not in allowed_scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Not enough permissions. Required scope: {scope}"
            )
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


MAX_CLEANUP_FAILURES = 5

async def cleanup_old_temp_files():
    """Background task to clean up storage older than configured max age.
    Implements exponential backoff on consecutive failures."""
    consecutive_failures = 0
    base_sleep = storage_config.cleanup_max_age_seconds
    while True:
        run_cleanup = True
        if use_redis and redis_client:
            try:
                loop = asyncio.get_running_loop()
                acquired = await loop.run_in_executor(
                    None,
                    lambda: redis_client.set("medi_chain:locks:cleanup", "locked", ex=500, nx=True)
                )
                if not acquired:
                    run_cleanup = False
                    logger.info("Another replica is already running storage cleanup. Skipping this run.")
            except Exception as e:
                logger.critical(f"Failed to acquire Redis cleanup lock: {e}. Skipping cleanup task as fallback to avoid concurrent write races on shared volumes.")
                run_cleanup = False
        else:
            run_cleanup = True
            logger.info("Redis is down. Running staggered temp file cleanup without distributed lock.")
        
        if run_cleanup:
            try:
                await asyncio.to_thread(storage.cleanup, max_age_seconds=storage_config.cleanup_max_age_seconds)
                consecutive_failures = 0
                sleep_time = base_sleep
            except Exception as e:
                consecutive_failures += 1
                sleep_time = min(base_sleep * (2 ** (consecutive_failures - 1)), 3600)
                logger.error(f"Error in cleanup task ({consecutive_failures} consecutive failures). Retrying in {sleep_time} seconds: {e}")
                if consecutive_failures >= MAX_CLEANUP_FAILURES:
                    logger.critical(
                        f"Cleanup task experienced {consecutive_failures} consecutive failures. "
                        f"Storage cleanup is impaired. System requires manual check/reboot."
                    )
        else:
            sleep_time = base_sleep
            
        import random
        is_testing = TESTING
        actual_sleep = sleep_time + (random.randint(0, 60) if not (use_redis and redis_client) and not is_testing else 0)
        await asyncio.sleep(actual_sleep)


import concurrent.futures
_alert_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="system-alert-sender")

def _send_system_alert_sync(title: str, message: str):
    logger.critical(f"CRITICAL SYSTEM ALERT: {title} — {message}")
    webhook_url = DRIFT_ALERT_WEBHOOK_URL
    if webhook_url:
        try:
            import requests
            payload = {
                "text": f"🚨 *{title}*\n{message}\n_Timestamp: {datetime.now(timezone.utc).isoformat()}_"
            }
            requests.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send connection alert webhook: {e}")

def _send_system_alert(title: str, message: str):
    _alert_executor.submit(_send_system_alert_sync, title, message)


async def reconcile_dlq_task():
    """Background task that polls the Redis DLQ and local disk DLQ and retries pushes to the EHR."""
    
    r = None
    if use_redis:
        import redis
        try:
            r = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
            logger.info("[DLQ Reconciler] Connected to Redis for DLQ reconciliation.")
        except Exception as e:
            logger.error(f"[DLQ Reconciler] Failed to connect to Redis: {e}. Falling back to local DLQ only.")
            
    logger.info("[DLQ Reconciler] Started background concurrent DLQ reconciliation worker.")
    
    dlq_semaphore = asyncio.Semaphore(5)
    loop = asyncio.get_running_loop()
    active_tasks = set()
    
    # Rate limiter for EHR pushes to prevent thundering herd
    ehr_rate_limiter = asyncio.Semaphore(3)
    
    async def process_reconciliation(item: dict, source_type: str, identifier: Optional[str], item_raw_json: Optional[str] = None):
        async with ehr_rate_limiter:
            try:
                payload = item.get("payload")
                if not payload:
                    if source_type == "redis" and r and item_raw_json:
                        await loop.run_in_executor(None, r.lrem, "medi_chain:dlq:processing", 1, item_raw_json)
                    return
                
                if not isinstance(payload, str):
                    fhir_json = json.dumps(payload)
                else:
                    fhir_json = payload
                    
                logger.info(f"[DLQ Reconciler] Attempting to reconcile report from {source_type}...")
                
                success = await ehr_gateway.push_report(fhir_json, True)
                
                if success:
                    logger.info(f"[DLQ Reconciler] Successfully reconciled report from {source_type}.")
                    if source_type == "local" and identifier:
                        try:
                            file_path = Path(identifier)
                            if file_path.exists():
                                file_path.unlink()
                        except Exception as e:
                            logger.error(f"[DLQ Reconciler] Failed to delete resolved local DLQ file {identifier}: {e}")
                    elif source_type == "redis" and r and item_raw_json:
                        try:
                            await loop.run_in_executor(None, r.lrem, "medi_chain:dlq:processing", 1, item_raw_json)
                        except Exception as e:
                            logger.error(f"[DLQ Reconciler] Failed to remove processed item from Redis processing queue: {e}")
                else:
                    retry_count = item.get("retry_count", 0) + 1
                    item["retry_count"] = retry_count
                    
                    if retry_count >= 3:
                        logger.critical(f"[DLQ Reconciler] Report push failed {retry_count} times. Routing to POISON DLQ.")
                        
                        # Escalation path 1: save locally under temp/dlq/poison/
                        try:
                            poison_dir = Path(DLQ_DIR) / "poison"
                            poison_dir.mkdir(parents=True, exist_ok=True)
                            filename = f"poison_report_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
                            local_path = poison_dir / filename
                            payload_json = json.dumps(item, indent=2)
                            encrypted_payload = encrypt_payload(payload_json)
                            wrapper = {"encrypted": True, "data": encrypted_payload}
                            with open(local_path, "w") as f:
                                json.dump(wrapper, f, indent=2)
                            logger.info(f"[DLQ Reconciler] Saved poison report locally to {local_path}")
                        except Exception as local_err:
                            logger.error(f"[DLQ Reconciler] Failed to save poison report locally: {local_err}")
                        
                        # Escalation path 2: push to Redis poison DLQ if Redis is enabled
                        if use_redis and r:
                            try:
                                await loop.run_in_executor(None, r.rpush, "medi_chain:dlq:poison", json.dumps(item))
                                if item_raw_json:
                                    await loop.run_in_executor(None, r.lrem, "medi_chain:dlq:processing", 1, item_raw_json)
                            except Exception as redis_err:
                                logger.error(f"[DLQ Reconciler] Failed to push to Redis poison DLQ: {redis_err}")
                        
                        # Escalation path 3: Delete local file if it was a local item
                        if source_type == "local" and identifier:
                            try:
                                file_path = Path(identifier)
                                if file_path.exists():
                                    file_path.unlink()
                            except Exception as e:
                                logger.error(f"[DLQ Reconciler] Failed to delete poisoned local DLQ file {identifier}: {e}")
                        
                        _send_system_alert(
                            "DLQ Poison Threshold Exceeded",
                            f"Report push failed {retry_count} times and has been escalated to poison DLQ. Error: {item.get('error')}"
                        )
                    else:
                        if source_type == "redis" and r:
                            logger.warning(f"[DLQ Reconciler] Re-push failed. Pushing payload back to the tail of the Redis DLQ (Retry count: {retry_count}).")
                            try:
                                await loop.run_in_executor(None, r.rpush, "medi_chain:dlq", json.dumps(item))
                                if item_raw_json:
                                    await loop.run_in_executor(None, r.lrem, "medi_chain:dlq:processing", 1, item_raw_json)
                            except Exception as redis_err:
                                logger.error(f"[DLQ Reconciler] Failed to re-queue back to Redis DLQ: {redis_err}")
                        elif source_type == "local" and identifier:
                            logger.warning(f"[DLQ Reconciler] Local re-push failed. Saving updated retry count to local file {identifier}.")
                            try:
                                import tempfile
                                payload_json = json.dumps(item, indent=2)
                                encrypted_payload = encrypt_payload(payload_json)
                                wrapper = {"encrypted": True, "data": encrypted_payload}
                                wrapper_json = json.dumps(wrapper, indent=2)
                                
                                target_path = Path(identifier).with_suffix(".json")
                                dlq_dir = target_path.parent
                                
                                total, used, free = shutil.disk_usage(dlq_dir)
                                if free < 50 * 1024 * 1024:
                                    raise RuntimeError("Critically low disk space on local partition")
                                    
                                fd, tmp_path = tempfile.mkstemp(dir=str(dlq_dir), suffix=".tmp")
                                try:
                                    with os.fdopen(fd, "w") as f:
                                        f.write(wrapper_json)
                                        f.flush()
                                        os.fsync(f.fileno())
                                    os.replace(tmp_path, str(target_path))
                                except Exception:
                                    if os.path.exists(tmp_path):
                                        os.remove(tmp_path)
                                    raise
                                
                                Path(identifier).unlink(missing_ok=True)
                                logger.info(f"[DLQ Reconciler] Successfully saved updated retry count to local DLQ file {target_path.name}")
                            except Exception as local_err:
                                logger.error(f"[DLQ Reconciler] Failed to write updated local DLQ retry count: {local_err}")
            except Exception as err:
                logger.error(f"[DLQ Reconciler] Error processing DLQ item: {err}")
            finally:
                dlq_semaphore.release()
            
    while True:
        try:
            run_reconcile = True
            if use_redis and r:
                try:
                    acquired = await loop.run_in_executor(
                        None,
                        lambda: r.set("medi_chain:locks:dlq_reconciler", "locked", ex=30, nx=True)
                    )
                    if not acquired:
                        run_reconcile = False
                except Exception as lock_err:
                    logger.warning(f"[DLQ Reconciler] Failed to acquire Redis lock: {lock_err}")
            
            if not run_reconcile:
                await asyncio.sleep(10)
                continue

            await dlq_semaphore.acquire()
            
            item_found = False
            
            # 1. Try to fetch from Redis DLQ if configured
            if use_redis and r:
                try:
                    item_json = await loop.run_in_executor(None, r.rpoplpush, "medi_chain:dlq", "medi_chain:dlq:processing")
                    if item_json:
                        item = json.loads(item_json)
                        task = asyncio.create_task(process_reconciliation(item, "redis", None, item_json))
                        active_tasks.add(task)
                        task.add_done_callback(active_tasks.discard)
                        item_found = True
                except Exception as redis_err:
                    logger.error(f"[DLQ Reconciler] Redis pull error: {redis_err}")
                    
            # 2. Try to fetch from local DLQ files if no Redis item was found
            if not item_found:
                dlq_dir = Path(DLQ_DIR)
                if dlq_dir.exists() and dlq_dir.is_dir():
                    try:
                        from itertools import islice
                        local_files = [f for f in islice(dlq_dir.glob("failed_report_*.json"), 50)]
                    except Exception as glob_err:
                        logger.error(f"[DLQ Reconciler] Glob error: {glob_err}")
                        local_files = []
                        
                    for file_path in local_files:
                        processing_path = file_path.with_suffix(".processing")
                        try:
                            file_path.rename(processing_path)
                        except FileNotFoundError:
                            continue
                        except Exception as e:
                            logger.error(f"[DLQ Reconciler] Failed to rename local DLQ file {file_path.name}: {e}")
                            continue
                        
                        try:
                            with open(processing_path, "r") as f:
                                wrapper = json.load(f)
                            if isinstance(wrapper, dict) and wrapper.get("encrypted"):
                                decrypted_data = decrypt_payload(wrapper["data"])
                                item = json.loads(decrypted_data)
                            else:
                                item = wrapper
                            task = asyncio.create_task(process_reconciliation(item, "local", str(processing_path)))
                            active_tasks.add(task)
                            task.add_done_callback(active_tasks.discard)
                            item_found = True
                            break
                        except Exception as e:
                            logger.error(f"[DLQ Reconciler] Failed to read local DLQ file {processing_path.name}: {e}")
                            try:
                                processing_path.rename(file_path)
                            except Exception:
                                pass
                                    
            if not item_found:
                dlq_semaphore.release()
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("[DLQ Reconciler] DLQ reconciler task cancelled.")
            break
        except Exception as e:
            logger.error(f"[DLQ Reconciler] Error in DLQ reconciliation loop: {e}")
            if not item_found:
                try:
                    dlq_semaphore.release()
                except ValueError:
                    pass
            await asyncio.sleep(10)

    if active_tasks:
        logger.info(f"[DLQ Reconciler] Awaiting {len(active_tasks)} pending reconciliation tasks...")
        await asyncio.gather(*active_tasks, return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log configuration audit on startup
    logger.info("Configuration audit dump:")
    audit_config = dump_all_configs()
    logger.info(json.dumps(audit_config, indent=2, default=str))
    
    # Eagerly initialize the agent
    app.state.agent = build_agent()
    
    # Conditionally start background tasks
    run_workers = api_config.enable_background_workers
    cleanup_task = None
    dlq_task = None
    
    if run_workers:
        cleanup_task = asyncio.create_task(cleanup_old_temp_files())
        dlq_task = asyncio.create_task(reconcile_dlq_task())
        logger.info("Background cleanup and DLQ tasks started inside the API process.")
    else:
        logger.info("Background tasks are disabled inside the API process (API_ENABLE_BACKGROUND_WORKERS=false).")
        
    yield
    
    if cleanup_task:
        cleanup_task.cancel()
    if dlq_task:
        dlq_task.cancel()
        
    # Gracefully close the agent's persistent httpx client
    if app.state.agent and hasattr(app.state.agent, 'close'):
        try:
            await app.state.agent.close()
        except Exception as e:
            logger.warning(f"Error closing agent HTTP client: {e}")
    app.state.agent = None


# In-memory job store (fallback when Redis unavailable)
_jobs_db = {}
_jobs_db_lock = asyncio.Lock()

# Idempotency store
_idempotency_store = {}
_idempotency_lock = asyncio.Lock()
IDEMPOTENCY_TTL_SECONDS = 86400  # 24 hours


async def _update_job_status(job_id: str, status: str, result: dict = None, error: str = None):
    data = {
        "job_id": job_id,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    if result is not None:
        result_copy = json.loads(json.dumps(result))
        if "dicom_metadata" in result_copy and result_copy["dicom_metadata"] is not None:
            try:
                plain_str = json.dumps(result_copy["dicom_metadata"])
                encrypted_str = encrypt_payload(plain_str)
                result_copy["dicom_metadata"] = {
                    "encrypted": True,
                    "data": encrypted_str
                }
            except Exception as encrypt_err:
                logger.error(f"Failed to encrypt DICOM metadata for storage: {encrypt_err}")
        data["result"] = result_copy
    if error is not None:
        data["error"] = error
        
    if use_redis and redis_client:
        try:
            await asyncio.to_thread(redis_client.set, f"medi_chain:jobs:{job_id}", json.dumps(data), ex=86400)
        except Exception as e:
            logger.error(f"Failed to update job {job_id} in Redis: {e}")
            
    async with _jobs_db_lock:
        _jobs_db[job_id] = data


async def _check_idempotency(idempotency_key: str) -> Optional[dict]:
    """Check if idempotency key exists and return cached result."""
    async with _idempotency_lock:
        if idempotency_key in _idempotency_store:
            entry = _idempotency_store[idempotency_key]
            if time.time() - entry["timestamp"] < IDEMPOTENCY_TTL_SECONDS:
                return entry["result"]
            else:
                del _idempotency_store[idempotency_key]
    return None


async def _store_idempotency(idempotency_key: str, result: dict):
    """Store result for idempotency key."""
    async with _idempotency_lock:
        _idempotency_store[idempotency_key] = {
            "result": result,
            "timestamp": time.time()
        }


async def process_analyze_job(
    job_id: str,
    local_img_path: str,
    local_pdf_path: str,
    request_temp_dir: str,
    agent: ClinicalAgent,
    dicom_metadata: Optional[dict] = None,
    idempotency_key: Optional[str] = None
):
    await _update_job_status(job_id, "running")
    try:
        start_t = time.time()
        async with inference_semaphore:
            result = await agent.run(local_img_path, local_pdf_path, idempotency_key=idempotency_key)
        processing_time = time.time() - start_t
        PROM_PROCESSING_TIME.observe(processing_time)
            
        await drift_detector.add_prediction(
            result['diagnosis']['probabilities'],
            result.get('visual_features')
        )
        
        PROM_CASES_PROCESSED.inc()
        escalation = result.get('escalation_required', False)
        if escalation:
            PROM_ESCALATIONS.inc()
        if use_redis and redis_client:
            try:
                pipeline = redis_client.pipeline()
                pipeline.incr("medi_chain:telemetry:total_cases")
                if escalation:
                    pipeline.incr("medi_chain:telemetry:escalated_cases")
                    pipeline.zadd("medi_chain:review_queue", {job_id: time.time()})
                await asyncio.to_thread(pipeline.execute)
            except Exception as telemetry_err:
                logger.error(f"Failed to update escalation telemetry: {telemetry_err}")
                
        # Save heatmap persistently
        heatmap_path = save_heatmap_from_base64(result.get("heatmap_base64", ""), job_id)
        heatmap_url = storage.get_download_url(heatmap_path) if heatmap_path else ""

        response_payload = {
            "request_id": job_id,
            "diagnosis": result.get("diagnosis", {}),
            "confidence": result.get("confidence", 0.0),
            "heatmap_url": heatmap_url,
            "pubmed_citations": result.get("pubmed_citations", []),
            "escalation_required": escalation,
            "iteration_count": result.get("iteration_count", 0),
            "model_metadata": get_model_metadata(),
            "dicom_metadata": dicom_metadata,
        }
        await _update_job_status(job_id, "completed", result=response_payload)
        
        # Store for idempotency
        if idempotency_key:
            await _store_idempotency(idempotency_key, response_payload)
    except Exception as exc:
        logger.error(f"Async job {job_id} failed: {exc}")
        await _update_job_status(job_id, "failed", error=str(exc))
        
        PROM_CASES_PROCESSED.inc() 
        PROM_FAILURES.inc()
        if use_redis and redis_client:
            try:
                pipeline = redis_client.pipeline()
                pipeline.incr("medi_chain:telemetry:total_cases")
                pipeline.incr("medi_chain:telemetry:failed_cases")
                await asyncio.to_thread(pipeline.execute)
            except Exception as telemetry_err:
                logger.error(f"Failed to update failed cases telemetry: {telemetry_err}")
    finally:
        if request_temp_dir and os.path.exists(request_temp_dir):
            try:
                shutil.rmtree(request_temp_dir, ignore_errors=True)
            except Exception as cleanup_err:
                logger.error(f"Failed to clean up temporary directory {request_temp_dir}: {cleanup_err}")


def create_app() -> FastAPI:
    app = FastAPI(
        title="MEdi Chain AI - API", 
        version=APP_VERSION, 
        lifespan=lifespan,
        description="Enterprise-ready multimodal diagnostic API with clinical validation.",
        docs_url="/v1/docs",
        redoc_url="/v1/redoc",
        openapi_url="/v1/openapi.json",
    )
    app.state.started_at = time.time()
    
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

    @app.post("/v1/analyze")
    @limiter.limit("10/minute")
    async def analyze_case(
        request: Request,
        background_tasks: BackgroundTasks,
        image: UploadFile = File(...),
        history: UploadFile = File(...),
        sync: bool = False,
        idempotency_key: Optional[str] = Security(IDEMPOTENCY_KEY_HEADER),
        api_key: str = Security(verify_api_key, scopes=["cases:write"])
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

        # Check idempotency
        if idempotency_key:
            cached_result = await _check_idempotency(idempotency_key)
            if cached_result:
                logger.info(f"Returning cached result for idempotency key: {idempotency_key}")
                return JSONResponse(content=cached_result, headers={"X-Request-ID": request_id})

        try:
            TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            request_temp_dir = tempfile.mkdtemp(dir=str(TEMP_ROOT))
            
            history_filename = Path(history.filename or "history.pdf").name
            history_json_name = Path(history_filename).with_suffix(".json").name
            local_img_path = os.path.join(request_temp_dir, Path(image.filename or "image.jpg").name)
            local_pdf_path = os.path.join(request_temp_dir, history_json_name)

            # Gateway-Level De-identification
            raw_img_temp = tempfile.NamedTemporaryFile(suffix=Path(image.filename or "image.jpg").suffix, delete=False)
            raw_pdf_temp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            raw_img_temp_name = raw_img_temp.name
            raw_pdf_temp_name = raw_pdf_temp.name
            raw_img_temp.close()
            raw_pdf_temp.close()

            try:
                with open(raw_img_temp_name, "wb") as f:
                    shutil.copyfileobj(image.file, f)
                with open(raw_pdf_temp_name, "wb") as f:
                    shutil.copyfileobj(history.file, f)

                # Extract DICOM metadata before raw image scrubbing
                dicom_metadata = None
                if image.content_type == "application/dicom" or raw_img_temp_name.lower().endswith(('.dcm', '.dicom')):
                    try:
                        import pydicom
                        ds = pydicom.dcmread(raw_img_temp_name, stop_before_pixels=True)
                        dicom_metadata = {
                            "PatientName": str(getattr(ds, "PatientName", "REDACTED_PATIENTNAME")),
                            "PatientID": str(getattr(ds, "PatientID", "REDACTED_PATIENTID")),
                            "PatientBirthDate": str(getattr(ds, "PatientBirthDate", "")),
                            "PatientSex": str(getattr(ds, "PatientSex", "")),
                            "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "")),
                            "SeriesInstanceUID": str(getattr(ds, "SeriesInstanceUID", "")),
                        }
                    except Exception as e:
                        logger.warning(f"Failed to extract DICOM metadata: {e}")

                # Scrub PHI from raw files
                scrubbed_img_path = gateway_scrubber.mask_burned_in_text(raw_img_temp_name)
                scrubbed_pdf_path = gateway_scrubber.scrub_pdf(raw_pdf_temp_name)

                # Move scrubbed files to request temp dir
                shutil.move(scrubbed_img_path, local_img_path)
                shutil.move(scrubbed_pdf_path, local_pdf_path)

            finally:
                # Always clean up raw temp files
                for tmp_path in [raw_img_temp_name, raw_pdf_temp_name]:
                    try:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    except Exception:
                        pass

            if sync:
                result = await agent.run(local_img_path, local_pdf_path, idempotency_key=idempotency_key)
                
                await drift_detector.add_prediction(
                    result['diagnosis']['probabilities'],
                    result.get('visual_features')
                )
                
                PROM_CASES_PROCESSED.inc()
                escalation = result.get('escalation_required', False)
                if escalation:
                    PROM_ESCALATIONS.inc()
                if use_redis and redis_client:
                    try:
                        pipeline = redis_client.pipeline()
                        pipeline.incr("medi_chain:telemetry:total_cases")
                        if escalation:
                            pipeline.incr("medi_chain:telemetry:escalated_cases")
                            pipeline.zadd("medi_chain:review_queue", {request_id: time.time()})
                        await asyncio.to_thread(pipeline.execute)
                    except Exception as telemetry_err:
                        logger.error(f"Failed to update escalation telemetry: {telemetry_err}")
                
                heatmap_path = save_heatmap_from_base64(result.get("heatmap_base64", ""), request_id)
                heatmap_url = storage.get_download_url(heatmap_path) if heatmap_path else ""

                response_payload = {
                    "request_id": request_id,
                    "diagnosis": result.get("diagnosis", {}),
                    "confidence": result.get("confidence", 0.0),
                    "heatmap_url": heatmap_url,
                    "pubmed_citations": result.get("pubmed_citations", []),
                    "escalation_required": escalation,
                    "iteration_count": result.get("iteration_count", 0),
                    "model_metadata": get_model_metadata(),
                    "dicom_metadata": dicom_metadata,
                }
                
                if idempotency_key:
                    await _store_idempotency(idempotency_key, response_payload)
                
                return JSONResponse(content=response_payload, headers={"X-Request-ID": request_id})
            else:
                background_tasks.add_task(
                    process_analyze_job,
                    request_id,
                    local_img_path,
                    local_pdf_path,
                    request_temp_dir,
                    agent,
                    dicom_metadata,
                    idempotency_key
                )
                return JSONResponse(
                    content={"request_id": request_id, "status": "accepted", "message": "Analysis queued for processing."},
                    headers={"X-Request-ID": request_id}
                )
        except ValueError as ve:
            logger.error(f"Validation error for request {request_id}: {ve}")
            if request_temp_dir and os.path.exists(request_temp_dir):
                try:
                    shutil.rmtree(request_temp_dir, ignore_errors=True)
                except Exception as cleanup_err:
                    logger.error(f"Failed to clean up temporary directory {request_temp_dir}: {cleanup_err}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as exc:
            logger.error(f"Request {request_id} failed: {exc}")
            if request_temp_dir and os.path.exists(request_temp_dir):
                try:
                    shutil.rmtree(request_temp_dir, ignore_errors=True)
                except Exception as cleanup_err:
                    logger.error(f"Failed to clean up temporary directory {request_temp_dir}: {cleanup_err}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    @app.get("/v1/analyze/status/{job_id}")
    @limiter.limit("60/minute")
    async def get_job_status(
        job_id: str,
        request: Request,
        api_key: str = Security(verify_api_key, scopes=["cases:read"])
    ):
        job_data = None
        if use_redis and redis_client:
            try:
                raw = await asyncio.to_thread(redis_client.get, f"medi_chain:jobs:{job_id}")
                if raw:
                    job_data = json.loads(raw)
            except Exception as e:
                logger.error(f"Failed to read job {job_id} from Redis: {e}")
                
        if job_data is None:
            async with _jobs_db_lock:
                job_data = _jobs_db.get(job_id)
                
        if job_data is None:
            raise HTTPException(status_code=404, detail="Job not found")
            
        # Decrypt DICOM patient metadata if encrypted
        if job_data and "result" in job_data:
            job_data = json.loads(json.dumps(job_data))
            raw_meta = job_data["result"].get("dicom_metadata")
            if isinstance(raw_meta, dict) and raw_meta.get("encrypted"):
                try:
                    decrypted_str = decrypt_payload(raw_meta["data"])
                    job_data["result"]["dicom_metadata"] = json.loads(decrypted_str)
                except Exception as decrypt_err:
                    logger.error(f"Failed to decrypt DICOM metadata for status: {decrypt_err}")
        
        return JSONResponse(content=job_data)

    @app.get("/v1/cases/review-queue")
    @limiter.limit("30/minute")
    async def get_review_queue(
        request: Request,
        limit: int = 50,
        api_key: str = Security(verify_api_key, scopes=["cases:read"])
    ):
        """Fetch the list of escalated cases pending radiologist review."""
        if not use_redis or not redis_client:
            raise HTTPException(status_code=503, detail="Review queue requires Redis.")
            
        try:
            # Fetch oldest pending cases from the sorted set
            raw_queue = await asyncio.to_thread(redis_client.zrange, "medi_chain:review_queue", 0, limit - 1, withscores=True)
            queue_items = [{"job_id": str(item[0], 'utf-8') if isinstance(item[0], bytes) else item[0], "timestamp": item[1]} for item in raw_queue]
            return JSONResponse(content={"queue": queue_items, "count": len(queue_items)})
        except Exception as e:
            logger.error(f"Failed to fetch review queue: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch review queue")

    @dataclass
    class ReviewPayload:
        verdict: str  # agree, disagree, modify
        final_diagnosis: Optional[str] = None
        notes: str = ""
        doctor_id: str = "dr-anonymous"
        time_spent_seconds: Optional[float] = None

    @app.post("/v1/cases/{job_id}/review")
    @limiter.limit("30/minute")
    async def submit_radiologist_review(
        job_id: str,
        request: Request,
        payload: ReviewPayload,
        api_key: str = Security(verify_api_key, scopes=["cases:write", "feedback:write"])
    ):
        """Submit radiologist review for an escalated case. Updates FHIR report status and logs feedback."""
        # Validate verdict
        allowed_verdicts = {"agree", "disagree", "modify"}
        verdict_normalized = payload.verdict.strip().lower()
        if verdict_normalized not in allowed_verdicts:
            raise HTTPException(400, f"verdict must be one of: {', '.join(sorted(allowed_verdicts))}")
        
        # Load job
        job_data = None
        if use_redis and redis_client:
            try:
                raw = await asyncio.to_thread(redis_client.get, f"medi_chain:jobs:{job_id}")
                if raw:
                    job_data = json.loads(raw)
            except Exception as e:
                logger.error(f"Failed to read job {job_id} from Redis: {e}")
                
        if job_data is None:
            async with _jobs_db_lock:
                job_data = _jobs_db.get(job_id)
                
        if job_data is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_data.get("status") != "completed":
            raise HTTPException(400, "Can only review completed cases")
        
        result = job_data.get("result", {})
        diagnosis = result.get("diagnosis", {})
        was_escalated = result.get("escalation_required", False)
        
        # Log feedback
        agreement = verdict_normalized == "agree"
        session_id = job_id
        history_metadata = {"job_id": job_id, "escalated": was_escalated}
        
        try:
            path = await asyncio.to_thread(
                feedback_logger.log_feedback,
                session_id=session_id,
                verdict=verdict_normalized,
                notes=payload.notes,
                diagnosis=diagnosis,
                history_metadata=history_metadata,
                doctor_id=payload.doctor_id,
            )
        except Exception as e:
            logger.error(f"Feedback logging failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to log feedback: {e}")
        
        # Update drift detector feedback summary
        await drift_detector.update_feedback_summary(agreement)
        
        # Record metrics
        PROM_FEEDBACK.labels(verdict=verdict_normalized).inc()
        if payload.time_spent_seconds is not None:
            PROM_SIGN_OFF_TIME.observe(payload.time_spent_seconds)
        
        # Update FHIR report status to final if radiologist agrees or modifies
        if verdict_normalized in {"agree", "modify"}:
            try:
                from src.data.fhir_formatter import FHIRFormatter
                fhir_formatter = FHIRFormatter()
                hl7_oru = fhir_formatter.generate_hl7_oru(
                    diagnosis_data=result,
                    verdict=verdict_normalized,
                    final_diagnosis=payload.final_diagnosis,
                    notes=payload.notes,
                    doctor_id=payload.doctor_id
                )
                logger.info(f"Generated HL7 ORU for job {job_id}:\n{hl7_oru}")
                
                # Actively transmit to EHR Gateway
                if EHR_GATEWAY_URL:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        mllp_url = f"{EHR_GATEWAY_URL.rstrip('/')}/mllp_bridge"
                        try:
                            ehr_resp = await client.post(
                                mllp_url, 
                                content=hl7_oru, 
                                headers={"Content-Type": "application/hl7-v2"}, 
                                timeout=5.0
                            )
                            ehr_resp.raise_for_status()
                            logger.info(f"Successfully transmitted HL7 to EHR gateway at {mllp_url}")
                        except Exception as ehr_err:
                            logger.error(f"Failed to transmit HL7 to EHR gateway: {ehr_err}")
                
                _write_audit_event({
                    "event": "radiologist_review",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "job_id": job_id,
                    "doctor_id_hash": hashlib.sha256(payload.doctor_id.encode("utf-8")).hexdigest(),
                    "verdict": verdict_normalized,
                    "agreement": agreement,
                    "final_diagnosis": payload.final_diagnosis,
                    "time_spent_seconds": payload.time_spent_seconds,
                    "hl7_oru_generated": True
                })
            except Exception as e:
                logger.error(f"Failed to update FHIR report after review: {e}")
                
        # Remove from review queue
        if use_redis and redis_client:
            try:
                await asyncio.to_thread(redis_client.zrem, "medi_chain:review_queue", job_id)
            except Exception as e:
                logger.error(f"Failed to remove {job_id} from review queue: {e}")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Radiologist review recorded",
            "saved_path": str(path),
            "verdict": verdict_normalized,
        })

    class FeedbackPayload(BaseModel):
        session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
        verdict: str = Field(..., min_length=1, max_length=32)
        notes: str = Field("", max_length=2000)
        diagnosis: dict
        history_metadata: dict
        doctor_id: str = Field("dr-anonymous", max_length=128)
        start_time: Optional[float] = None
        disagreement_reason: Optional[str] = Field(None, max_length=256)
        correction_mask_base64: Optional[str] = None

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
            value_stripped = value.strip()
            is_valid = (
                value_stripped.startswith("dr-") or
                "@" in value_stripped or
                re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", value_stripped) is not None or
                (value_stripped.isdigit() and len(value_stripped) == 10)
            )
            if not is_valid:
                raise ValueError("doctor_id must start with 'dr-', or be a valid email/UUID/NPI.")
            return value_stripped

    @app.get("/v1/storage/{path:path}")
    async def get_storage_file(path: str, api_key: str = Security(verify_api_key, scopes=["cases:read"])):
        from fastapi.responses import FileResponse
        safe_path = Path(path).resolve()
        cwd = Path.cwd()
        if not (safe_path.is_relative_to(cwd / "outputs") or safe_path.is_relative_to(cwd / "temp")):
            raise HTTPException(status_code=403, detail="Forbidden")
        if not safe_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(safe_path))

    @app.post("/v1/feedback")
    @limiter.limit("30/minute")
    async def receive_feedback(
        request: Request,
        background_tasks: BackgroundTasks,
        payload: FeedbackPayload,
        api_key: str = Security(verify_api_key, scopes=["feedback:write"])
    ):
        try:
            agreement = payload.verdict in {"agree", "match"}
            path = await asyncio.to_thread(
                feedback_logger.log_feedback,
                session_id=payload.session_id,
                verdict=payload.verdict,
                notes=payload.notes,
                diagnosis=payload.diagnosis,
                history_metadata=payload.history_metadata,
                disagreement_reason=payload.disagreement_reason,
                correction_mask_base64=payload.correction_mask_base64
            )
            background_tasks.add_task(drift_detector.update_feedback_summary, agreement)
            
            PROM_FEEDBACK.labels(verdict=payload.verdict).inc()
            if payload.start_time is not None:
                latency = time.time() - payload.start_time
                PROM_SIGN_OFF_TIME.observe(latency)
                
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

    @app.get("/v1/feedback/discrepancies")
    @limiter.limit("10/minute")
    async def get_discrepancies(
        request: Request,
        api_key: str = Security(verify_api_key, scopes=["feedback:read"])
    ):
        try:
            records = []
            if redis_client is not None:
                raw_records = await asyncio.to_thread(redis_client.lrange, "medi_chain:feedback:records", 0, -1)
                for r_raw in raw_records:
                    rec = json.loads(r_raw)
                    if rec.get("verdict") in {"disagree", "mismatch"}:
                        records.append(rec)
            else:
                def read_local_csv():
                    import csv
                    csv_path = feedback_logger.csv_path
                    local_records = []
                    if csv_path.exists():
                        with csv_path.open("r", encoding="utf-8") as handle:
                            reader = csv.DictReader(handle)
                            for row in reader:
                                if row.get("verdict") in {"disagree", "mismatch"}:
                                    local_records.append(row)
                    return local_records
                records = await asyncio.to_thread(read_local_csv)
            return JSONResponse(content={"discrepancies": records})
        except Exception as e:
            logger.error(f"Failed to load discrepancies: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load discrepancies: {e}")

    @app.get("/v1/metrics/clinical")
    @limiter.limit("10/minute")
    async def get_clinical_metrics(
        request: Request,
        api_key: str = Security(verify_api_key, scopes=["metrics:read"])
    ):
        try:
            from src.config.settings import get_clinical_thresholds
            thresholds = get_clinical_thresholds()
            if not thresholds.thresholds_validated:
                return JSONResponse(status_code=404, content={"message": "Clinical thresholds and validation metrics have not been generated."})
            
            return JSONResponse(content={
                "validation_dataset": thresholds.validation_dataset,
                "validation_date": thresholds.validation_date,
                "metrics": thresholds.validation_metrics
            })
        except Exception as e:
            logger.error(f"Failed to fetch clinical metrics: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error retrieving metrics")


    @app.get("/v1/telemetry/metrics")
    @limiter.limit("10/minute")
    async def get_telemetry_metrics(
        request: Request,
        baseline_minutes: Optional[int] = None,
        hourly_rate: Optional[float] = None,
        gpu_hourly_cost: Optional[float] = None,
        api_key: str = Security(verify_api_key, scopes=["metrics:read"])
    ):
        h_rate = hourly_rate if hourly_rate is not None else float(os.getenv("TELEMETRY_HOURLY_RATE", "250.0"))
        gpu_cost = gpu_hourly_cost if gpu_hourly_cost is not None else float(os.getenv("GPU_HOURLY_COST", "0.0" if TESTING else "0.90"))

        if baseline_minutes is not None and (baseline_minutes <= 0 or baseline_minutes > 1440):
            raise HTTPException(status_code=400, detail="baseline_minutes must be between 1 and 1440 (24 hours).")
        if h_rate < 0.0 or h_rate > 10000.0:
            raise HTTPException(status_code=400, detail="hourly_rate must be between 0.0 and 10000.0.")
        if gpu_cost < 0.0 or gpu_cost > 1000.0:
            raise HTTPException(status_code=400, detail="gpu_hourly_cost must be between 0.0 and 1000.0.")

        started_at = getattr(request.app.state, "started_at", None)
        if started_at is None:
            started_at = time.time() - 3600
        elapsed_hours = max(0.01, (time.time() - started_at) / 3600.0)
        infra_cost = round(elapsed_hours * gpu_cost, 2)

        feedback_total = 0
        agreements = 0
        disagreements = 0
        agreement_rate = 1.0
        override_rate = 0.0
        
        summary_raw = None
        if use_redis and redis_client:
            try:
                summary_raw = await asyncio.to_thread(redis_client.get, "medi_chain:drift:feedback_summary")
            except Exception as telemetry_err:
                logger.error(f"Failed to load drift feedback summary from Redis: {telemetry_err}")
                
        if summary_raw:
            try:
                summary = json.loads(summary_raw)
                feedback_total = int(summary.get("total_cases", 0))
                agreements = int(summary.get("agreements", 0))
                disagreements = int(summary.get("disagreements", 0))
                agreement_rate = float(summary.get("agreement_rate", 1.0))
                override_rate = 1.0 - agreement_rate if feedback_total > 0 else 0.0
            except Exception as parse_err:
                logger.error(f"Failed to parse Redis feedback summary: {parse_err}")
        else:
            def read_local_csv_telemetry():
                import csv
                csv_path = feedback_logger.csv_path
                fb_total = 0
                fb_agreements = 0
                fb_disagreements = 0
                if csv_path.exists():
                    with csv_path.open("r", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        for row in reader:
                            fb_total += 1
                            v = row.get("verdict", "").strip().lower()
                            if v in {"agree", "match"}:
                                fb_agreements += 1
                            elif v in {"disagree", "mismatch"}:
                                fb_disagreements += 1
                return fb_total, fb_agreements, fb_disagreements

            feedback_total, agreements, disagreements = await asyncio.to_thread(read_local_csv_telemetry)
            if feedback_total > 0:
                agreement_rate = agreements / feedback_total
                override_rate = disagreements / feedback_total

        if not use_redis or not redis_client:
            return {
                "total_cases": 0,
                "escalated_cases": 0,
                "escalation_rate": 0.0,
                "saved_time_hours": 0.0,
                "saved_cost_usd": 0.0,
                "infrastructure_cost_usd": infra_cost,
                "net_saved_cost_usd": round(-infra_cost, 2),
                "feedback_total_cases": feedback_total,
                "feedback_agreements": agreements,
                "feedback_disagreements": disagreements,
                "clinician_agreement_rate": round(agreement_rate, 4),
                "clinician_override_rate": round(override_rate, 4),
                "message": "Full telemetry requires active Redis connection. Feedback metrics loaded from local cache."
            }
        
        try:
            total = int((await asyncio.to_thread(redis_client.get, "medi_chain:telemetry:total_cases")) or 0)
            escalated = int((await asyncio.to_thread(redis_client.get, "medi_chain:telemetry:escalated_cases")) or 0)
            rate = escalated / total if total > 0 else 0.0
            
            processing_sum = 0.0
            processing_count = 0.0
            try:
                for sample in PROM_PROCESSING_TIME.collect()[0].samples:
                    if sample.name.endswith('_sum'):
                        processing_sum = sample.value
                    elif sample.name.endswith('_count'):
                        processing_count = sample.value
            except Exception:
                pass
            
            signoff_sum = 0.0
            signoff_count = 0.0
            try:
                for sample in PROM_SIGN_OFF_TIME.collect()[0].samples:
                    if sample.name.endswith('_sum'):
                        signoff_sum = sample.value
                    elif sample.name.endswith('_count'):
                        signoff_count = sample.value
            except Exception:
                pass
                
            actual_baseline_mins = (signoff_sum / signoff_count) / 60.0 if signoff_count > 0 else float(os.getenv("TELEMETRY_BASELINE_MINUTES", "18"))
            b_mins = baseline_minutes if baseline_minutes is not None else actual_baseline_mins
            
            a_mins = (processing_sum / processing_count) / 60.0 if processing_count > 0 else float(os.getenv("TELEMETRY_AUTOMATED_MINUTES", "1"))
            
            saved_minutes = (total - escalated) * (b_mins - a_mins)
            saved_hours = round(saved_minutes / 60, 2)
            saved_cost = round(saved_hours * h_rate, 2)
            net_saved_cost = round(saved_cost - infra_cost, 2)
            
            return {
                "total_cases": total,
                "escalated_cases": escalated,
                "escalation_rate": round(rate, 4),
                "saved_time_hours": saved_hours,
                "saved_cost_usd": saved_cost,
                "infrastructure_cost_usd": infra_cost,
                "net_saved_cost_usd": net_saved_cost,
                "feedback_total_cases": feedback_total,
                "feedback_agreements": agreements,
                "feedback_disagreements": disagreements,
                "clinician_agreement_rate": round(agreement_rate, 4),
                "clinician_override_rate": round(override_rate, 4),
            }
        except Exception as e:
            logger.error(f"Failed to load telemetry metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load telemetry metrics: {e}")

    @app.get("/v1/metrics")
    async def metrics_endpoint():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/health")
    async def health_check(request: Request):
        # Check inference API connectivity
        inference_healthy = False
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{os.getenv('INFERENCE_API_URL', 'http://inference-api:8001')}/health")
                inference_healthy = resp.status_code == 200
        except Exception:
            pass
        
        return {
            "status": "ok",
            "models_loaded": request.app.state.agent is not None,
            "inference_api_healthy": inference_healthy,
            "concurrency_limit": MAX_CONCURRENT_REQUESTS,
            "version": APP_VERSION,
            "config_audit": get_config().audit_dump(),
        }

    @app.get("/v1/health/gpu")
    async def gpu_health():
        """GPU health endpoint for inference API monitoring."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"gpu_available": False, "message": "CUDA not available"}
            
            device_count = torch.cuda.device_count()
            gpu_info = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                total = props.total_memory / (1024 ** 3)
                free = total - reserved
                
                gpu_info.append({
                    "device": i,
                    "name": props.name,
                    "total_memory_gb": round(total, 2),
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "free_gb": round(free, 2),
                    "utilization_pct": round((reserved / total) * 100, 1),
                })
            
            return {
                "gpu_available": True,
                "device_count": device_count,
                "gpus": gpu_info,
                "cuda_version": torch.version.cuda,
            }
        except Exception as e:
            logger.error(f"GPU health check failed: {e}")
            return {"gpu_available": False, "error": str(e)}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)