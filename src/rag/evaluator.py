from pymilvus import connections, Collection
import os
import logging
import time
from datetime import datetime, timezone
import requests

logger = logging.getLogger("rag-evaluator")

def _send_alert(title: str, message: str):
    """Send critical system alert to external webhook (Slack, PagerDuty, etc.)."""
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

class RAGEvaluator:
    def __init__(self, milvus_host="localhost", milvus_port="19530", inference_api_url: str = None):
        self.collection_name = "pubmed_abstracts"
        self.inference_api_url = inference_api_url or os.getenv("INFERENCE_API_URL", "http://inference-api:8001")
        self.internal_api_key = os.getenv("INTERNAL_API_KEY", "")
        if not self.internal_api_key and os.getenv("TESTING") != "true":
            raise RuntimeError("INTERNAL_API_KEY environment variable is required.")

        # Configure SSL / TLS settings
        self.ssl_verify = os.getenv("INTERNAL_SSL_VERIFY", "false")
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
        if os.getenv("TESTING") == "true":
            return
        
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                # Add retry loop to give Milvus database time to spin up in Docker Compose
                connections.connect("default", host=milvus_host, port=milvus_port)
                self.collection = Collection(self.collection_name)
                self.collection.load()
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    msg = f"Could not connect to Milvus/Collection after {max_attempts} attempts: {e}"
                    print(f"Warning: {msg}")
                    _send_alert("Milvus Connection Failure", msg)
                else:
                    time.sleep(1)

    def search(self, query, k=5):
        """Perform search in Milvus. Raises RuntimeError if offline."""
        if not self.collection:
            raise RuntimeError("Milvus RAG collection is offline or uninitialized.")
            
        try:
            resp = requests.post(
                f"{self.inference_api_url}/encode/text",
                json={"text": query},
                headers={"X-Internal-API-Key": self.internal_api_key},
                verify=self.ssl_verify,
                cert=self.ssl_cert
            )
            resp.raise_for_status()
            vector = resp.json()["embeddings"][0]
        except Exception as e:
            print(f"[RAGEvaluator] Error calling inference API: {e}")
            raise RuntimeError(f"RAG text encoding failed: {e}")
        
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["pmid", "text", "title"]
        )
        
        parsed_results = []
        for hit in results[0]:
            parsed_results.append({
                "pmid": hit.entity.get("pmid"),
                "text": hit.entity.get("text"),
                "title": hit.entity.get("title", "Unknown Title")
            })
        return parsed_results

    def evaluate_hit_rate(self, test_cases):
        """
        Evaluate Hit-Rate@5.
        test_cases: List of dicts {'query': str, 'expected_pmid': str}
        """
        hits = 0
        total = len(test_cases)
        
        print(f"Evaluating Hit-Rate@5 on {total} cases...")
        for case in test_cases:
            results = self.search(case['query'], k=5)
            # Check if expected_pmid is in results
            pmids = [res.get('pmid') for res in results]
            if case['expected_pmid'] in pmids:
                hits += 1
                
        hit_rate = hits / total if total > 0 else 0
        print(f"Hit-Rate@5: {hit_rate:.2f}")
        return hit_rate

if __name__ == "__main__":
    # Hand-labelled gold standard queries (Example)
    test_queries = [
        {"query": "What are the common radiological findings in silicosis?", "expected_pmid": "123456"},
        {"query": "Differential diagnosis between pneumonia and tuberculosis on CXR", "expected_pmid": "789012"},
        # Add 18 more...
    ]
    
    evaluator = RAGEvaluator()
    evaluator.evaluate_hit_rate(test_queries)
