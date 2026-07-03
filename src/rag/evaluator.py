from pymilvus import connections, Collection
import os
import logging
import time
import asyncio
from datetime import datetime, timezone
import requests

logger = logging.getLogger("rag-evaluator")

import concurrent.futures
import threading

_alert_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-alert-sender")

def _send_alert_sync(title: str, message: str):
    logger.critical(f"CRITICAL SYSTEM ALERT: {title} — {message}")
    webhook_url = os.getenv("DRIFT_ALERT_WEBHOOK_URL", "")
    if webhook_url:
        try:
            payload = {
                "text": f"🚨 *{title}*\n{message}\n_Timestamp: {datetime.now(timezone.utc).isoformat()}_"
            }
            requests.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send connection alert webhook: {e}")

def _send_alert(title: str, message: str):
    """Flaw #2 Fix: Send alerts asynchronously via a thread pool to avoid blocking execution."""
    _alert_executor.submit(_send_alert_sync, title, message)

class RAGEvaluator:
    def __init__(self, milvus_host="localhost", milvus_port="19530", inference_api_url: str = None):
        self.collection_name = "pubmed_abstracts"
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.inference_api_url = inference_api_url or os.getenv("INFERENCE_API_URL", "http://inference-api:8001")
        self.internal_api_key = os.getenv("INTERNAL_API_KEY", "")
        if not self.internal_api_key and os.getenv("TESTING") != "true":
            raise RuntimeError("INTERNAL_API_KEY environment variable is required.")

        # Configure SSL / TLS settings
        self.ssl_verify = os.getenv("INTERNAL_SSL_VERIFY", "true")
        if self.ssl_verify.lower() == "true":
            self.ssl_verify = True
        elif self.ssl_verify.lower() == "false":
            self.ssl_verify = False
        
        ssl_cert_file = os.getenv("INTERNAL_SSL_CERT_FILE", None)
        ssl_key_file = os.getenv("INTERNAL_SSL_KEY_FILE", None)
        if ssl_cert_file and ssl_key_file:
            self.ssl_cert = (ssl_cert_file, ssl_key_file)
        elif ssl_cert_file:
            self.ssl_cert = ssl_cert_file
        else:
            self.ssl_cert = None

        self.collection = None
        self._conn_lock = threading.Lock()
        self._async_conn_lock = None
        
        # Circuit breaker parameters
        self.last_conn_failure_time = 0.0
        self.conn_fail_cooldown = float(os.getenv("MILVUS_CONN_FAIL_COOLDOWN", "60.0"))
        
        if os.getenv("TESTING") == "true":
            return
        
        # Flaw #4 Fix: Eager connection check on startup, but swallow connection errors to prevent startup crash/blocking
        try:
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info("Successfully established eager connection to Milvus.")
        except Exception as e:
            self.last_conn_failure_time = time.time()
            msg = f"Eager Milvus connection failed during RAGEvaluator init: {e}. Will attempt lazy reconnection on query."
            logger.warning(msg)
            _send_alert("Milvus Connection Failure", msg)

    async def _ensure_connected(self):
        """Flaw #4 Fix: Thread-safe lazy initialization check, run asynchronously to avoid blocking loop."""
        if self.collection is not None:
            return
            
        if self._async_conn_lock is None:
            self._async_conn_lock = asyncio.Lock()
            
        async with self._async_conn_lock:
            if self.collection is not None:
                return
                
            # Circuit breaker to prevent connection storms and alert cascades
            now = time.time()
            if now - self.last_conn_failure_time < self.conn_fail_cooldown:
                logger.warning("[Milvus Connection] Circuit breaker active. Skipping connection attempt.")
                raise RuntimeError("Milvus RAG collection is offline or uninitialized (circuit breaker active).")
                
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._sync_connect)
            except Exception as e:
                self.last_conn_failure_time = time.time()
                msg = f"Milvus lazy connection failed: {e}"
                logger.error(msg)
                # Send non-blocking alert on connection failure
                _send_alert("Milvus Connection Failure", msg)
                raise RuntimeError("Milvus RAG collection is offline or uninitialized.")

    def _sync_connect(self):
        logger.info(f"Attempting lazy Milvus connection to {self.milvus_host}:{self.milvus_port}...")
        connections.connect("default", host=self.milvus_host, port=self.milvus_port)
        self.collection = Collection(self.collection_name)
        self.collection.load()
        logger.info("Lazily loaded Milvus RAG collection successfully.")

    async def search(self, query, k=5):
        """Perform search in Milvus. Raises RuntimeError if offline."""
        await self._ensure_connected()
        
        import httpx
        try:
            async with httpx.AsyncClient(verify=self.ssl_verify, cert=self.ssl_cert) as client:
                resp = await client.post(
                    f"{self.inference_api_url}/encode/text",
                    json={"text": query},
                    headers={"X-Internal-API-Key": self.internal_api_key},
                    timeout=10
                )
            resp.raise_for_status()
            vector = resp.json()["embeddings"][0]
        except Exception as e:
            print(f"[RAGEvaluator] Error calling inference API: {e}")
            raise RuntimeError(f"RAG text encoding failed: {e}")
        
        import asyncio
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.collection.search(
                data=[vector],
                anns_field="vector",
                param=search_params,
                limit=k,
                output_fields=["pmid", "text", "title"]
            )
        )
        
        parsed_results = []
        for hit in results[0]:
            parsed_results.append({
                "pmid": hit.entity.get("pmid"),
                "text": hit.entity.get("text"),
                "title": hit.entity.get("title", "Unknown Title")
            })
        return parsed_results

    async def evaluate_hit_rate(self, test_cases):
        """
        Evaluate Hit-Rate@5.
        test_cases: List of dicts {'query': str, 'expected_pmid': str}
        """
        hits = 0
        total = len(test_cases)
        
        print(f"Evaluating Hit-Rate@5 on {total} cases...")
        for case in test_cases:
            results = await self.search(case['query'], k=5)
            # Check if expected_pmid is in results
            pmids = [res.get('pmid') for res in results]
            if case['expected_pmid'] in pmids:
                hits += 1
                
        hit_rate = hits / total if total > 0 else 0
        print(f"Hit-Rate@5: {hit_rate:.2f}")
        return hit_rate

if __name__ == "__main__":
    import asyncio
    # Hand-labelled gold standard queries (Example)
    test_queries = [
        {"query": "What are the common radiological findings in silicosis?", "expected_pmid": "123456"},
        {"query": "Differential diagnosis between pneumonia and tuberculosis on CXR", "expected_pmid": "789012"},
        # Add 18 more...
    ]
    
    evaluator = RAGEvaluator()
    asyncio.run(evaluator.evaluate_hit_rate(test_queries))
