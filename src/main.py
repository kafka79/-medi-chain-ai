import argparse
import sys
from src.data.iu_xray_pipeline import IUXrayPipeline
from src.data.history_generator import ClinicalHistoryGenerator
from src.rag.pubmed_fetcher import PubMedFetcher
from src.rag.indexer import PubMedIndexer

def main():
    parser = argparse.ArgumentParser(description="MEdi chain ai - Phase 1 Runner")
    parser.add_argument("--mode", type=str, choices=["data", "rag", "history", "all"], 
                        default="all", help="Which component to run")
    
    args = parser.parse_args()
    
    if args.mode in ["data", "all"]:
        print("--- Running IU-Xray Data Pipeline ---")
        pipeline = IUXrayPipeline()
        pipeline.run()
        
    if args.mode in ["history", "all"]:
        print("--- Running Synthetic History Generator ---")
        generator = ClinicalHistoryGenerator()
        for condition in ["silicosis", "asbestosis", "pneumonia"]:
            generator.create_pdf(generator.generate_patient_data(condition))
            
    if args.mode in ["rag", "all"]:
        print("--- Running PubMed RAG Pipeline ---")
        fetcher = PubMedFetcher()
        query = "(radiology[MeSH Terms] OR 'chest x-ray') AND (pneumonia OR silicosis)"
        ids = fetcher.search_ids(query, max_results=100)
        df = fetcher.fetch_details(ids)
        df.to_csv("data/rag/pubmed_abstracts.csv", index=False)
        
        indexer = PubMedIndexer()
        indexer.index_data("data/rag/pubmed_abstracts.csv")

if __name__ == "__main__":
    main()
