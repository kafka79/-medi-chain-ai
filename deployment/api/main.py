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
class RedisDistributedSemaphore:
    """Distributed semaphore backed by Redis sorted-set leases.

    Panel Flaw #2 Fix: The previous implementation used a spinlock that polled
    Redis every 100ms (``while not acquired: await asyncio.sleep(0.1)``).
    Under high concurrency this generated a storm of network round-trips.

    This version replaces the spinlock with Redis BLPOP on a notification key.
    When a lease is released in ``__aexit__``, the releaser pushes a token to
    the notification list.  Waiters call ``BLPOP`` which is a Redis-native
    blocking operation — the connection idles until a token arrives, consuming
    zero CPU and zero network bandwidth while waiting.
    
    Model Server Concurrency Storm Outage Fallback:
    If Redis goes down or is None, we degrade gracefully to a process-local limit of 1
    (via MAX_CONCURRENT_REQUESTS_FALLBACK) instead of silently falling back to the full limit.
    """
    def __init__(self, r_client, name: str, limit: int):
        self.orig_redis = r_client
        self.redis = r_client
        self.name = f"medi_chain:semaphore:{name}:leases"
        self.notify_key = f"medi_chain:semaphore:{name}:notify"
        self.limit = limit
        self.local_sem = asyncio.Semaphore(limit)
        fallback_limit = int(os.getenv("MAX_CONCURRENT_REQUESTS_FALLBACK", "1"))
        self.fallback_sem = asyncio.Semaphore(fallback_limit)
        self.client_id_var = contextvars.ContextVar(f"sem_client_id_{name}", default=None)
        self.last_reconnect_attempt = 0.0
        
        # Ensure temp/storage folder exists for lock files
        lock_dir = Path("temp/storage")
        lock_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_file_lock = FileLock(lock_dir / f"semaphore_{name}.lock")

    async def __aenter__(self):
        # Attempt self-healing reconnection check
        if self.redis is None and self.orig_redis is not None:
            now = time.time()
            reconnect_cooldown = float(os.getenv("REDIS_RECONNECT_COOLDOWN", "10.0"))
            if now - self.last_reconnect_attempt >= reconnect_cooldown:
                self.last_reconnect_attempt = now
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.orig_redis.ping)
                    self.redis = self.orig_redis
                    logger.info("Redis connectivity restored. Recovering distributed capacity.")
                except Exception as e:
                    logger.warning(f"Redis reconnection attempt failed: {e}")

        # Local/File fallback mode (coordinated across replicas via shared FileLock)
        if self.redis is None:
            await self.fallback_sem.acquire()
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.fallback_file_lock.acquire)
            except Exception as e:
                logger.error(f"Fallback file lock acquire failed: {e}")
                self.fallback_sem.release()
                raise
            return self

        # Normal distributed mode
        await self.local_sem.acquire()
        client_id = uuid.uuid4().hex
        self.client_id_var.set(client_id)
        
        acquired = False
        loop = asyncio.get_running_loop()
        lease_ttl = int(os.getenv("SEMAPHORE_LEASE_TTL", "60"))
        blpop_timeout = int(os.getenv("SEMAPHORE_BLPOP_TIMEOUT", "5"))
        
        acquire_script = """
        local leases_key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local lease_ttl = tonumber(ARGV[3])
        local client_id = ARGV[4]

        -- 1. Remove expired leases
        redis.call('ZREMRANGEBYSCORE', leases_key, '-inf', now - lease_ttl)

        -- 2. Count active leases
        local active_count = redis.call('ZCARD', leases_key)

        -- 3. If under limit, acquire lease
        if active_count < limit then
            redis.call('ZADD', leases_key, now, client_id)
            redis.call('EXPIRE', leases_key, 86400)
            return 1
        else
            return 0
        end
        """
        
        # First attempt — optimistic acquisition without blocking
        try:
            now = time.time()
            res = await loop.run_in_executor(
                None,
                lambda: self.redis.eval(acquire_script, 1, self.name, self.limit, now, lease_ttl, client_id)
            )
            if res == 1:
                acquired = True
        except Exception as e:
            logger.error(f"Redis acquisition failed, degrading to local/file fallback semaphore: {e}")
            self.client_id_var.set(None)
            self.local_sem.release()
            self.redis = None
            self.last_reconnect_attempt = time.time()
            _send_system_alert("Redis Semaphore Outage", f"Degrading to local/file fallback. Error: {e}")
            await self.fallback_sem.acquire()
            try:
                await loop.run_in_executor(None, self.fallback_file_lock.acquire)
            except Exception as fe:
                logger.error(f"Fallback file lock acquire failed after Redis exception: {fe}")
                self.fallback_sem.release()
                raise
            return self
        
        # If not acquired, block on BLPOP notifications instead of polling
        while not acquired:
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self.redis.blpop(self.notify_key, timeout=blpop_timeout)
                )
                now = time.time()
                res = await loop.run_in_executor(
                    None,
                    lambda: self.redis.eval(acquire_script, 1, self.name, self.limit, now, lease_ttl, client_id)
                )
                if res == 1:
                    acquired = True
            except Exception as e:
                logger.error(f"Redis execution failed during blpop/retry, degrading to local/file fallback: {e}")
                self.client_id_var.set(None)
                self.local_sem.release()
                self.redis = None
                self.last_reconnect_attempt = time.time()
                _send_system_alert("Redis Semaphore Outage", f"Degrading to local/file fallback during wait. Error: {e}")
                await self.fallback_sem.acquire()
                try:
                    await loop.run_in_executor(None, self.fallback_file_lock.acquire)
                except Exception as fe:
                    logger.error(f"Fallback file lock acquire failed during wait exception: {fe}")
                    self.fallback_sem.release()
                    raise
                return self
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        client_id = self.client_id_var.get()
        if client_id:
            self.client_id_var.set(None)
            if self.redis is not None:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: self.redis.zrem(self.name, client_id))
                    await loop.run_in_executor(
                        None,
                        lambda: (
                            self.redis.lpush(self.notify_key, "1"),
                            self.redis.ltrim(self.notify_key, 0, self.limit * 2)
                        )
                    )
                except Exception as e:
                    logger.error(f"Redis semaphore release failed: {e}")
            self.local_sem.release()
        else:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.fallback_file_lock.release)
            except Exception as e:
                logger.error(f"Fallback file lock release failed: {e}")
            finally:
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
            
        await asyncio.sleep(sleep_time)

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
            success = await loop.run_in_executor(None, ehr_gateway.push_report, fhir_json, True)
            
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
                            from src.utils.security import encrypt_payload
                            payload_json = json.dumps(item, indent=2)
                            encrypted_payload = encrypt_payload(payload_json)
                            wrapper = {
                                "encrypted": True,
                                "data": encrypted_payload
                            }
                            with open(identifier, "w") as f:
                                json.dump(wrapper, f, indent=2)
                            # Rename back to .json so it can be picked up again
                            orig_path = Path(identifier).with_suffix(".json")
                            Path(identifier).rename(orig_path)
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
    # Start background cleanup
    cleanup_task = asyncio.create_task(cleanup_old_temp_files())
    # Start background DLQ reconciler
    dlq_task = asyncio.create_task(reconcile_dlq_task())
    yield
    cleanup_task.cancel()
    dlq_task.cancel()
    # Flaw #5-structural Fix: Gracefully close the agent's persistent httpx client
    if app.state.agent and hasattr(app.state.agent, 'close'):
        try:
            await app.state.agent.close()
        except Exception as e:
            logger.warning(f"Error closing agent HTTP client: {e}")
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

            # Telemetry for escalation rates
            if use_redis and redis_client:
                try:
                    pipeline = redis_client.pipeline()
                    pipeline.incr("medi_chain:telemetry:total_cases")
                    if escalation:
                        pipeline.incr("medi_chain:telemetry:escalated_cases")
                    pipeline.execute()
                except Exception as telemetry_err:
                    logger.error(f"Failed to update escalation telemetry: {telemetry_err}")

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

            # Flaw #5 Fix: Remove the X-Requires-Human-Review HTTP header setting.
            # Escalation is now a first-class field in the JSON response payload.
            headers = {}
            if escalation:
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

    @app.get("/telemetry/metrics")
    @limiter.limit("10/minute")
    async def get_telemetry_metrics(
        request: Request,
        api_key: str = Depends(verify_api_key)
    ):
        """Returns production telemetry metrics including the real-world escalation rate."""
        if not use_redis or not redis_client:
            return {
                "total_cases": 0,
                "escalated_cases": 0,
                "escalation_rate": 0.0,
                "message": "Telemetry requires active Redis connection."
            }
        
        try:
            total = int(redis_client.get("medi_chain:telemetry:total_cases") or 0)
            escalated = int(redis_client.get("medi_chain:telemetry:escalated_cases") or 0)
            rate = escalated / total if total > 0 else 0.0
            
            # ROI Calculations (factor in manual radiologist baseline vs automated with escalation)
            # Baseline: 18 mins per case. Automated: < 1 min per case (we use 1 min).
            # Escalated: still takes 18 mins.
            # Saved time per non-escalated case = 17 mins.
            saved_minutes = (total - escalated) * 17
            saved_hours = round(saved_minutes / 60, 2)
            saved_cost = round(saved_hours * 250, 2) # At $250/hour
            
            return {
                "total_cases": total,
                "escalated_cases": escalated,
                "escalation_rate": round(rate, 4),
                "saved_time_hours": saved_hours,
                "saved_cost_usd": saved_cost
            }
        except Exception as e:
            logger.error(f"Failed to load telemetry metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load telemetry metrics: {e}")

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
