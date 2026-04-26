import pandas as pd
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os

class PubMedIndexer:
    def __init__(self, milvus_host="localhost", milvus_port="19530"):
        self.collection_name = "pubmed_abstracts"
        self.embedding_model = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
        
        print(f"Connecting to Milvus at {milvus_host}:{milvus_port}...")
        connections.connect("default", host=milvus_host, port=milvus_port)

    def create_collection(self, dim=768):
        """Create a Milvus collection for PubMed abstracts."""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, "PubMed abstracts chunked")
        collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": "IP", # Inner Product for SapBERT
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        return collection

    def chunk_text(self, text, chunk_size=256, overlap=32):
        """Simple token-based chunking."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def index_data(self, csv_path):
        """Index abstracts from CSV into Milvus."""
        df = pd.read_csv(csv_path)
        collection = self.create_collection()
        
        print(f"Indexing {len(df)} articles...")
        all_pmids = []
        all_texts = []
        all_vectors = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            chunks = self.chunk_text(row['abstract'])
            pmid = str(row['pmid'])
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                # Truncate text for Milvus max_length
                text_snippet = chunk[:1990] 
                
                all_pmids.append(pmid)
                all_texts.append(text_snippet)
                
        # Batch embedding for efficiency
        print("Generating embeddings...")
        batch_size = 64
        for i in tqdm(range(0, len(all_texts), batch_size)):
            batch_texts = all_texts[i:i+batch_size]
            embeddings = self.embedding_model.encode(batch_texts, convert_to_tensor=True)
            all_vectors.extend(embeddings.cpu().numpy().tolist())
            
            # Flush to Milvus every 5000 items
            if len(all_vectors) >= 5000:
                self._insert_to_milvus(collection, all_pmids[:len(all_vectors)], all_texts[:len(all_vectors)], all_vectors)
                all_pmids = all_pmids[len(all_vectors):]
                all_texts = all_texts[len(all_vectors):]
                all_vectors = []
        
        # Final insert
        if all_vectors:
            self._insert_to_milvus(collection, all_pmids, all_texts, all_vectors)
            
        print("Indexing complete.")
        collection.flush()

    def _insert_to_milvus(self, collection, pmids, texts, vectors):
        data = [pmids, texts, vectors]
        collection.insert(data)

if __name__ == "__main__":
    csv_file = "data/rag/pubmed_abstracts.csv"
    if os.path.exists(csv_file):
        indexer = PubMedIndexer()
        indexer.index_data(csv_file)
    else:
        print(f"Error: {csv_file} not found. Run pubmed_fetcher.py first.")
