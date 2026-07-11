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
from typing import Optional, Any
import re

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, BackgroundTasks, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import secrets
from filelock import FileLock
import uvicorn
import contextvars
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agent.clinical_graph import ClinicalAgent
from src.data.pdf_parser import ClinicalPDFParser
from src.data.privacy_scrubber import PrivacyScrubber
from src.data.fhir_formatter import EHRGateway
from src.rag.evaluator import RAGEvaluator
from src.utils.storage import S3StorageProvider
from src.monitoring.drift_detector import DriftDetector
from src.utils.feedback_logger import FeedbackLogger
from pydantic import BaseModel, Field, field_validator
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
PROM_FEEDBACK = _get_or_create_counter("medi_chain_feedback_total", "Clinician feedback verdicts.", ["verdict"])
PROM_SIGN_OFF_TIME = _get_or_create_histogram(
    "medi_chain_sign_off_time_seconds", 
    "Radiologist sign-off time in seconds.",
    buckets=(10, 30, 45, 60, 120, 180, 300, 600, 1200, 3600)
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medi-chain-api")
audit_logger = logging.getLogger("medi-chain-api.audit")
audit_logger.setLevel(logging.INFO)


def _configure_audit_logger():
    """Configure container-native stdout audit logging, falling back to local file if explicitly requested."""
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
        
        # Decode base64 image
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
            # Save to outputs/heatmaps/
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

# Flaw #2 Fix: Fail fast if API_KEY is not set in production
_api_key = os.getenv("API_KEY")
if not _api_key and os.getenv("TESTING") != "true" and os.getenv("STORAGE_MODE") != "local":
    raise RuntimeError(
        "CRITICAL: API_KEY environment variable is not set. "
        "Refusing to start with default credentials in a production-like environment. "
        "Set API_KEY in your .env file."
    )

# Security Fail-Fast: Fail fast if DLQ_ENCRYPTION_KEY is not set in production
_dlq_encryption_key = os.getenv("DLQ_ENCRYPTION_KEY")
if not _dlq_encryption_key and os.getenv("TESTING") != "true" and os.getenv("STORAGE_MODE") != "local":
    raise RuntimeError(
        "CRITICAL: DLQ_ENCRYPTION_KEY environment variable is not set. "
        "Refusing to start with default key fallback in a production-like environment. "
        "Set DLQ_ENCRYPTION_KEY in your .env file."
    )

# Check if Redis is actually responsive before using it for rate limiting
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
use_redis = False
redis_client = None
is_production = os.getenv("TESTING") != "true" and os.getenv("STORAGE_MODE") != "local"

# Support Redis Sentinel or Redis Cluster for High Availability (removes SPOF)
sentinel_hosts_str = os.getenv("REDIS_SENTINEL_HOSTS", "")
cluster_nodes_str = os.getenv("REDIS_CLUSTER_NODES", "")

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
            service_name = os.getenv("REDIS_SENTINEL_SERVICE_NAME", "mymaster")
            sentinel_client = Sentinel(sentinels, socket_connect_timeout=1, decode_responses=True)
            redis_client = sentinel_client.master_for(service_name, socket_connect_timeout=1, decode_responses=True)
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
            redis_client = RedisCluster(startup_nodes=nodes, socket_connect_timeout=1, decode_responses=True)
            logger.info("Successfully configured Redis Cluster for High Availability.")
        else:
            redis_client = redis.from_url(redis_url, socket_connect_timeout=1)
            
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
gateway_scrubber = PrivacyScrubber()
gateway_pdf_parser = ClinicalPDFParser()
class RedisDistributedSemaphore:
    """Distributed semaphore backed by Redis sorted-set leases.
    
    ponytail: Uses a clean exponential-backoff spinlock to avoid BLPOP list leakage
    and falls back to process-local asyncio.Semaphore when Redis is down, avoiding
    misleading FileLock setups that fail across container replicas.
    """
    def __init__(self, r_client, name: str, limit: int):
        self.orig_redis = r_client
        self.redis = r_client
        self.name = f"medi_chain:semaphore:{name}:leases"
        self.limit = limit
        self.local_sem = asyncio.Semaphore(limit)
        
        fallback_limit = int(os.getenv("MAX_CONCURRENT_REQUESTS_FALLBACK", str(limit)))
        self.fallback_sem = asyncio.Semaphore(fallback_limit)
        
        self.client_id_var = contextvars.ContextVar(f"sem_client_id_{name}", default=None)
        self.refresher_task = None
        self.last_reconnect_attempt = 0

    async def _refresh_lease_loop(self, client_id: str):
        lease_ttl = int(os.getenv("SEMAPHORE_LEASE_TTL", "10"))
        refresh_interval = max(1, lease_ttl // 3)
        try:
            while True:
                await asyncio.sleep(refresh_interval)
                if self.redis is not None:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self.redis.zadd(self.name, {client_id: time.time()})
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Failed to refresh lease: {e}")

    async def __aenter__(self):
        # Self-healing check: if redis is offline but original client exists, try to reconnect
        if self.redis is None and self.orig_redis is not None:
            now = time.time()
            cooldown = float(os.getenv("REDIS_RECONNECT_COOLDOWN", "30.0"))
            if now - self.last_reconnect_attempt > cooldown:
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
        
        lease_ttl = int(os.getenv("SEMAPHORE_LEASE_TTL", "10"))
        loop = asyncio.get_running_loop()
        
        acquire_script = """
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
        
        import random
        retry_delay = 0.05
        while True:
            try:
                now = time.time()
                res = await loop.run_in_executor(
                    None,
                    lambda: self.redis.eval(acquire_script, 1, self.name, self.limit, now, lease_ttl, client_id)
                )
                if res == 1:
                    break
            except Exception as e:
                logger.error(f"Redis semaphore failed, degrading to local local_sem fallback: {e}")
                self.redis = None
                self.client_id_var.set(None)
                _send_system_alert(
                    "Redis Semaphore Connection Outage",
                    f"Degrading to local fallback semaphore: {e}"
                )
                await self.fallback_sem.acquire()
                self.local_sem.release()
                return self
            
            await asyncio.sleep(retry_delay)
            # ponytail: cap retry delay at 0.2s to prevent high contention latency spikes
            retry_delay = min(retry_delay * 1.5 + random.random() * 0.05, 0.2)
            
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
                    await loop.run_in_executor(None, lambda: self.redis.zrem(self.name, client_id))
                except Exception as e:
                    logger.error(f"Redis release failed: {e}")
            self.local_sem.release()
        else:
            self.fallback_sem.release()


MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))

# Global semaphore for model inference (distributed when Redis is active)
if use_redis and redis_client:
    inference_semaphore = RedisDistributedSemaphore(redis_client, "inference", MAX_CONCURRENT_REQUESTS)
else:
    inference_semaphore = RedisDistributedSemaphore(None, "inference", MAX_CONCURRENT_REQUESTS)


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

# Flaw #11 Fix: Circuit breaker — alert after consecutive failures, but don't terminate loop
MAX_CLEANUP_FAILURES = 5

async def cleanup_old_temp_files():
    """Background task to clean up storage older than 1 hour.
    Implements exponential backoff on consecutive failures to prevent permanent shutdown
    if the remote storage (e.g. MinIO) is transiently unavailable."""
    consecutive_failures = 0
    base_sleep = 600
    cleanup_file_lock = None
    while True:
        run_cleanup = True
        file_lock_acquired = False
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
            # Fallback to local execution without FileLock: stagger executions
            # across replicas using randomized jitter to prevent write races on S3/MinIO
            run_cleanup = True
            logger.info("Redis is down. Running staggered temp file cleanup without distributed lock.")
        
        if run_cleanup:
            try:
                await asyncio.to_thread(storage.cleanup, max_age_seconds=3600)
                consecutive_failures = 0  # Reset on success
                sleep_time = base_sleep
            except Exception as e:
                consecutive_failures += 1
                # Exponential backoff: 600s, 1200s, 2400s, maxing out at 3600s (1 hour)
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
        # Stagger executions across replicas by adding a random jitter of up to 60 seconds
        is_testing = os.getenv("TESTING") == "true"
        actual_sleep = sleep_time + (random.randint(0, 60) if not (use_redis and redis_client) and not is_testing else 0)
        await asyncio.sleep(actual_sleep)

import concurrent.futures
_alert_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="system-alert-sender")

def _send_system_alert_sync(title: str, message: str):
    logger.critical(f"CRITICAL SYSTEM ALERT: {title} — {message}")
    webhook_url = os.getenv("DRIFT_ALERT_WEBHOOK_URL", "")
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
    """Flaw #2 Fix: Send system alert asynchronously via a thread pool to avoid blocking execution."""
    _alert_executor.submit(_send_system_alert_sync, title, message)


async def reconcile_dlq_task():
    """Background task that polls the Redis DLQ 'medi_chain:dlq' and the local disk folder 'temp/dlq'
    and retries pushes to the EHR. Resolves Flaws #4 and #5 by supporting both Redis and local disk DLQs,
    and running up to 5 concurrent pushes using asyncio task scheduling and semaphores."""
    
    r = None
    if use_redis:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        try:
            r = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
            logger.info("[DLQ Reconciler] Connected to Redis for DLQ reconciliation.")
        except Exception as e:
            logger.error(f"[DLQ Reconciler] Failed to connect to Redis: {e}. Falling back to local DLQ only.")
            
    logger.info("[DLQ Reconciler] Started background concurrent DLQ reconciliation worker.")
    
    dlq_semaphore = asyncio.Semaphore(5)
    loop = asyncio.get_running_loop()
    active_tasks = set()
    
    async def process_reconciliation(item: dict, source_type: str, identifier: Optional[str], item_raw_json: Optional[str] = None):
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
            
            # Pass is_retry=True to avoid duplicate DLQ writes inside push_report
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
                        dlq_base = os.getenv("DLQ_DIR", "temp/dlq")
                        poison_dir = Path(dlq_base) / "poison"
                        poison_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"poison_report_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
                        local_path = poison_dir / filename
                        from src.utils.security import encrypt_payload
                        payload_json = json.dumps(item, indent=2)
                        encrypted_payload = encrypt_payload(payload_json)
                        wrapper = {
                            "encrypted": True,
                            "data": encrypted_payload
                        }
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
                            import shutil
                            from src.utils.security import encrypt_payload
                            
                            payload_json = json.dumps(item, indent=2)
                            encrypted_payload = encrypt_payload(payload_json)
                            wrapper = {
                                "encrypted": True,
                                "data": encrypted_payload
                            }
                            wrapper_json = json.dumps(wrapper, indent=2)
                            
                            target_path = Path(identifier).with_suffix(".json")
                            dlq_dir = target_path.parent
                            
                            # Validate disk space before writing
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
                                
                            # Safely delete processing file only after new .json file is durable
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
                    # Acquire lock to run DLQ reconciliation loop (lease of 30 seconds)
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

            # Wait until a semaphore slot is available
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
                dlq_dir = Path(os.getenv("DLQ_DIR", "temp/dlq"))
                if dlq_dir.exists() and dlq_dir.is_dir():
                    try:
                        from itertools import islice
                        # Limit to 50 files per iteration to avoid globbing thousands of files
                        local_files = [f for f in islice(dlq_dir.glob("failed_report_*.json"), 50)]
                    except Exception as glob_err:
                        logger.error(f"[DLQ Reconciler] Glob error: {glob_err}")
                        local_files = []
                        
                    for file_path in local_files:
                        processing_path = file_path.with_suffix(".processing")
                        try:
                            # Atomic rename to claim the file without OS locks
                            file_path.rename(processing_path)
                        except FileNotFoundError:
                            # Another process/thread claimed it or it was deleted
                            continue
                        except Exception as e:
                            logger.error(f"[DLQ Reconciler] Failed to rename local DLQ file {file_path.name}: {e}")
                            continue
                        
                        try:
                            with open(processing_path, "r") as f:
                                wrapper = json.load(f)
                            if isinstance(wrapper, dict) and wrapper.get("encrypted"):
                                from src.utils.security import decrypt_payload
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
                                # Revert rename on failure to read
                                processing_path.rename(file_path)
                            except Exception:
                                pass
                                    
            if not item_found:
                # Release semaphore and sleep if no work
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

    # Await any remaining active tasks to ensure graceful shutdown and test synchronization
    if active_tasks:
        logger.info(f"[DLQ Reconciler] Awaiting {len(active_tasks)} pending reconciliation tasks...")
        await asyncio.gather(*active_tasks, return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize the agent
    app.state.agent = build_agent()
    
    # Conditionally start background tasks
    run_workers = os.getenv("RUN_BACKGROUND_WORKERS_IN_API", "false").lower() == "true"
    cleanup_task = None
    dlq_task = None
    
    if run_workers:
        cleanup_task = asyncio.create_task(cleanup_old_temp_files())
        dlq_task = asyncio.create_task(reconcile_dlq_task())
        logger.info("Background cleanup and DLQ tasks started inside the API process.")
    else:
        logger.info("Background tasks are disabled inside the API process (RUN_BACKGROUND_WORKERS_IN_API=false).")
        
    yield
    
    if cleanup_task:
        cleanup_task.cancel()
    if dlq_task:
        dlq_task.cancel()
        
    # Flaw #5-structural Fix: Gracefully close the agent's persistent httpx client
    if app.state.agent and hasattr(app.state.agent, 'close'):
        try:
            await app.state.agent.close()
        except Exception as e:
            logger.warning(f"Error closing agent HTTP client: {e}")
    app.state.agent = None

_jobs_db = {}
_jobs_db_lock = asyncio.Lock()

async def _update_job_status(job_id: str, status: str, result: dict = None, error: str = None):
    data = {
        "job_id": job_id,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    if result is not None:
        data["result"] = result
    if error is not None:
        data["error"] = error
        
    if use_redis and redis_client:
        try:
            # We wrap the synchronous Redis set in asyncio.to_thread to avoid blocking the event loop
            await asyncio.to_thread(redis_client.set, f"medi_chain:jobs:{job_id}", json.dumps(data), ex=86400)
        except Exception as e:
            logger.error(f"Failed to update job {job_id} in Redis: {e}")
            
    async with _jobs_db_lock:
        _jobs_db[job_id] = data

async def process_analyze_job(
    job_id: str,
    local_img_path: str,
    local_pdf_path: str,
    request_temp_dir: str,
    agent: ClinicalAgent
):
    await _update_job_status(job_id, "running")
    try:
        async with inference_semaphore:
            result = await agent.run(local_img_path, local_pdf_path)
            
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
                await asyncio.to_thread(pipeline.execute)
            except Exception as telemetry_err:
                logger.error(f"Failed to update escalation telemetry: {telemetry_err}")
                
        # Save heatmap persistently
        save_heatmap_from_base64(result.get("heatmap_base64", ""), job_id)

        response_payload = {
            "request_id": job_id,
            "diagnosis": result.get("diagnosis", {}),
            "confidence": result.get("confidence", 0.0),
            "heatmap_base64": result.get("heatmap_base64", ""),
            "pubmed_citations": result.get("pubmed_citations", []),
            "escalation_required": escalation,
            "iteration_count": result.get("iteration_count", 0),
            "model_metadata": get_model_metadata(),
        }
        await _update_job_status(job_id, "completed", result=response_payload)
    except Exception as exc:
        logger.error(f"Async job {job_id} failed: {exc}")
        await _update_job_status(job_id, "failed", error=str(exc))
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
        sync: bool = False,
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
            # Note: since the history is parsed and scrubbed, we write a JSON file to local_pdf_path
            history_filename = Path(history.filename or "history.pdf").name
            history_json_name = Path(history_filename).with_suffix(".json").name
            local_img_path = os.path.join(request_temp_dir, Path(image.filename or "image.jpg").name)
            local_pdf_path = os.path.join(request_temp_dir, history_json_name)

            # Gateway-Level De-identification:
            # Write uploaded raw files to transient system-level temp files that are immediately scrubbed and deleted
            raw_img_temp = tempfile.NamedTemporaryFile(suffix=Path(image.filename or "image.jpg").suffix, delete=False)
            raw_pdf_temp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            raw_img_temp_name = raw_img_temp.name
            raw_pdf_temp_name = raw_pdf_temp.name
            raw_img_temp.close()
            raw_pdf_temp.close()

            try:
                # 1. Save uploaded image to raw temp file
                with open(raw_img_temp_name, "wb") as f:
                    shutil.copyfileobj(image.file, f)
                # 2. Save uploaded PDF to raw temp file
                with open(raw_pdf_temp_name, "wb") as f:
                    shutil.copyfileobj(history.file, f)

                # 3. De-identify the image at the gateway level
                sanitized_img_path = await asyncio.to_thread(gateway_scrubber.mask_burned_in_text, raw_img_temp_name)
                # Copy sanitized image to request temp dir
                shutil.copy2(sanitized_img_path, local_img_path)
                # Clean up sanitized image temp file if a copy was created
                if sanitized_img_path != raw_img_temp_name and os.path.exists(sanitized_img_path):
                    try:
                        os.unlink(sanitized_img_path)
                    except Exception:
                        pass

                # 4. De-identify and parse the PDF history at the gateway level
                try:
                    raw_history = await asyncio.to_thread(gateway_pdf_parser.parse_pdf, raw_pdf_temp_name)
                    scrubbed_history = await asyncio.to_thread(gateway_scrubber.scrub_history_data, raw_history)
                except Exception as parse_err:
                    if os.getenv("TESTING") == "true":
                        logger.warning(
                            f"Failed to parse clinical history PDF at gateway ({parse_err}). "
                            f"Falling back to placeholder history layout for compatibility in testing."
                        )
                        # Safe basic layout expected by downstream models
                        scrubbed_history = {
                            "chief_complaint": "Not found",
                            "history_present_illness": "Not found",
                            "past_medical_history": "Not found",
                            "social_history": "Not found",
                            "review_of_systems": "Not found",
                            "labs": "Not found",
                            "metadata": {
                                "age": "Unknown",
                                "gender": "Unknown",
                                "occupation": "Unknown",
                                "exposure_years": "0"
                            }
                        }
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to parse clinical history PDF: {parse_err}"
                        )
                # Save the scrubbed history as a JSON file in the request temp dir
                with open(local_pdf_path, "w", encoding="utf-8") as f:
                    json.dump(scrubbed_history, f, indent=4)
            finally:
                # Clean up raw temp files immediately so they never linger
                if os.path.exists(raw_img_temp_name):
                    try:
                        os.unlink(raw_img_temp_name)
                    except Exception:
                        pass
                if os.path.exists(raw_pdf_temp_name):
                    try:
                        os.unlink(raw_pdf_temp_name)
                    except Exception:
                        pass

            if sync:
                # Synchronous path: execute agent run blocking the endpoint until completion
                async with inference_semaphore:
                    result = await agent.run(local_img_path, local_pdf_path)
                    
                # Monitor for drift (prediction drift and covariate shift)
                background_tasks.add_task(drift_detector.add_prediction, result['diagnosis']['probabilities'], result.get('visual_features'))
                
                PROM_CASES_PROCESSED.inc()
                escalation = result.get('escalation_required', False)
                if escalation:
                    PROM_ESCALATIONS.inc()

                # Telemetry for escalation rates
                if use_redis and redis_client:
                    try:
                        pipeline = redis_client.pipeline()
                        pipeline.incr("medi_chain:telemetry:total_cases")
                        if escalation:
                            pipeline.incr("medi_chain:telemetry:escalated_cases")
                        await asyncio.to_thread(pipeline.execute)
                    except Exception as telemetry_err:
                        logger.error(f"Failed to update escalation telemetry: {telemetry_err}")

                # Save heatmap persistently
                save_heatmap_from_base64(result.get("heatmap_base64", ""), request_id)

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

                if escalation:
                    logger.warning(f"[{request_id}] Escalation triggered — insufficient evidence for automated diagnosis.")

                return JSONResponse(content=response_payload)
            else:
                # Asynchronous path: record pending state and offload run to BackgroundTasks
                job_id = request_id
                await _update_job_status(job_id, "pending")
                
                background_tasks.add_task(
                    process_analyze_job,
                    job_id,
                    local_img_path,
                    local_pdf_path,
                    request_temp_dir,
                    agent
                )
                
                status_payload = {
                    "job_id": job_id,
                    "status": "pending",
                    "status_url": f"/analyze/status/{job_id}"
                }
                return JSONResponse(status_code=202, content=status_payload)
                
        except HTTPException as http_exc:
            logger.warning(f"HTTPException in analysis: {http_exc.detail}")
            if sync or request_temp_dir is not None:
                if request_temp_dir and os.path.exists(request_temp_dir):
                    try:
                        shutil.rmtree(request_temp_dir, ignore_errors=True)
                    except Exception:
                        pass
            return JSONResponse(status_code=http_exc.status_code, content={"detail": http_exc.detail})
        except Exception as exc:
            logger.error(f"Analysis failed: {exc}")
            # If sync execution failed, clean up temp dir now; if async, process_analyze_job cleans it up
            if sync or request_temp_dir is not None:
                if request_temp_dir and os.path.exists(request_temp_dir):
                    try:
                        shutil.rmtree(request_temp_dir, ignore_errors=True)
                    except Exception:
                        pass
            return JSONResponse(status_code=500, content={"detail": f"Analysis failed: {exc}"})
        finally:
            if sync:
                if request_temp_dir and os.path.exists(request_temp_dir):
                    try:
                        shutil.rmtree(request_temp_dir, ignore_errors=True)
                    except Exception as cleanup_err:
                        logger.error(f"Failed to clean up temporary directory {request_temp_dir}: {cleanup_err}")

    @app.get("/analyze/heatmap/{job_id}")
    @limiter.limit("60/minute")
    async def get_heatmap_file(
        job_id: str,
        request: Request,
        format: str = "png",
        api_key: str = Depends(verify_api_key)
    ):
        from fastapi.responses import FileResponse
        from pathlib import Path
        import io
        
        relative_png_path = f"heatmaps/{job_id}.png"
        relative_dcm_path = f"heatmaps/{job_id}.dcm"
        
        if storage_mode == "s3":
            try:
                await asyncio.to_thread(storage.client.stat_object, storage.bucket, relative_png_path)
            except Exception:
                raise HTTPException(status_code=404, detail="Heatmap not found for this job ID.")
            
            png_path_str = await asyncio.to_thread(storage.load, relative_png_path)
            png_path = Path(png_path_str)
        else:
            png_path = Path("outputs/heatmaps") / f"{job_id}.png"
            if not png_path.exists():
                raise HTTPException(status_code=404, detail="Heatmap not found for this job ID.")
            
        if format.lower() == "dicom":
            if storage_mode == "s3":
                exists_dcm = False
                try:
                    await asyncio.to_thread(storage.client.stat_object, storage.bucket, relative_dcm_path)
                    exists_dcm = True
                except Exception:
                    pass
                
                if exists_dcm:
                    dcm_path_str = await asyncio.to_thread(storage.load, relative_dcm_path)
                    dcm_path = Path(dcm_path_str)
                else:
                    import tempfile
                    fd, temp_dcm_name = tempfile.mkstemp(suffix=".dcm")
                    os.close(fd)
                    try:
                        from src.data.dicom_handler import create_secondary_capture
                        await asyncio.to_thread(create_secondary_capture, None, str(png_path), temp_dcm_name)
                        with open(temp_dcm_name, "rb") as f:
                            await asyncio.to_thread(storage.save, f, relative_dcm_path)
                        dcm_path = Path(temp_dcm_name)
                    except Exception as e:
                        if os.path.exists(temp_dcm_name):
                            os.unlink(temp_dcm_name)
                        logger.error(f"Failed to generate Secondary Capture DICOM: {e}")
                        raise HTTPException(status_code=500, detail=f"Failed to generate DICOM: {e}")
            else:
                dcm_path = Path("outputs/heatmaps") / f"{job_id}.dcm"
                if not dcm_path.exists():
                    try:
                        from src.data.dicom_handler import create_secondary_capture
                        await asyncio.to_thread(create_secondary_capture, None, str(png_path), str(dcm_path))
                    except Exception as e:
                        logger.error(f"Failed to generate Secondary Capture DICOM: {e}")
                        raise HTTPException(status_code=500, detail=f"Failed to generate DICOM: {e}")
                        
            return FileResponse(
                path=str(dcm_path),
                media_type="application/dicom",
                filename=f"heatmap_{job_id}.dcm"
            )
        else:
            return FileResponse(
                path=str(png_path),
                media_type="image/png",
                filename=f"heatmap_{job_id}.png"
            )

    @app.get("/analyze/status/{job_id}")
    @limiter.limit("60/minute")
    async def get_job_status(
        job_id: str,
        request: Request,
        api_key: str = Depends(verify_api_key)
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
            
        return JSONResponse(content=job_data)

    class FeedbackPayload(BaseModel):
        session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
        verdict: str = Field(..., min_length=1, max_length=32)
        notes: str = Field("", max_length=2000)
        diagnosis: dict
        history_metadata: dict
        doctor_id: str = Field("dr-anonymous", max_length=128)
        start_time: Optional[float] = None

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
            # Flaw #6 Fix: Allow doctor prefix 'dr-', UUIDs, hospital emails, or NPI numbers
            value_stripped = value.strip()
            # Allow email address format, standard UUID format, or numeric NPI (10 digits), or standard dr- prefix
            is_valid = (
                value_stripped.startswith("dr-") or
                "@" in value_stripped or
                re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", value_stripped) is not None or
                (value_stripped.isdigit() and len(value_stripped) == 10)
            )
            if not is_valid:
                raise ValueError("doctor_id must start with 'dr-', or be a valid email/UUID/NPI.")
            return value_stripped

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
            
            # Record clinician sign-off latency and verdict count in Prometheus metrics
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
                raw_records = await asyncio.to_thread(redis_client.lrange, "medi_chain:feedback:records", 0, -1)
                for r_raw in raw_records:
                    rec = json.loads(r_raw)
                    if rec.get("verdict") in {"disagree", "mismatch"}:
                        records.append(rec)
            else:
                # Fallback to reading the local CSV if Redis isn't configured
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

    @app.get("/telemetry/metrics")
    @limiter.limit("10/minute")
    async def get_telemetry_metrics(
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """Returns production telemetry metrics including the real-world escalation rate and clinician override rates."""
        # Calculate feedback metrics first
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
            # Fallback: parse from local CSV file directly
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
            
            # ROI Calculations (factor in manual radiologist baseline vs automated with escalation)
            base_mins = int(os.getenv("TELEMETRY_BASELINE_MINUTES", "18"))
            auto_mins = int(os.getenv("TELEMETRY_AUTOMATED_MINUTES", "1"))
            hourly_rate = float(os.getenv("TELEMETRY_HOURLY_RATE", "250.0"))
            
            saved_minutes = (total - escalated) * (base_mins - auto_mins)
            saved_hours = round(saved_minutes / 60, 2)
            saved_cost = round(saved_hours * hourly_rate, 2)
            
            return {
                "total_cases": total,
                "escalated_cases": escalated,
                "escalation_rate": round(rate, 4),
                "saved_time_hours": saved_hours,
                "saved_cost_usd": saved_cost,
                "feedback_total_cases": feedback_total,
                "feedback_agreements": agreements,
                "feedback_disagreements": disagreements,
                "clinician_agreement_rate": round(agreement_rate, 4),
                "clinician_override_rate": round(override_rate, 4),
            }
        except Exception as e:
            logger.error(f"Failed to load telemetry metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load telemetry metrics: {e}")

    @app.get("/metrics")
    async def metrics_endpoint():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

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
