from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import os

class RAGEvaluator:
    def __init__(self, milvus_host="localhost", milvus_port="19530", inference_api_url: str = None):
        self.collection_name = "pubmed_abstracts"
        import os
        self.inference_api_url = inference_api_url or os.getenv("INFERENCE_API_URL", "http://inference-api:8001")
        self.internal_api_key = os.getenv("INTERNAL_API_KEY", "internal-secret-token")
        
        try:
            connections.connect("default", host=milvus_host, port=milvus_port)
            self.collection = Collection(self.collection_name)
            self.collection.load()
        except Exception as e:
            print(f"Warning: Could not connect to Milvus/Collection: {e}")
            self.collection = None

    def search(self, query, k=5):
        """Perform search in Milvus."""
        if not self.collection:
            return []
            
        import requests
        try:
            resp = requests.post(
                f"{self.inference_api_url}/encode/text",
                json={"text": query},
                headers={"X-Internal-API-Key": self.internal_api_key}
            )
            resp.raise_for_status()
            vector = resp.json()["embeddings"][0]
        except Exception as e:
            print(f"[RAGEvaluator] Error calling inference API: {e}")
            return []
        
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
