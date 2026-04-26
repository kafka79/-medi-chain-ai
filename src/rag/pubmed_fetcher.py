import requests
import time
import os
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

class PubMedFetcher:
    def __init__(self, api_key=None):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = api_key

    def search_ids(self, query, max_results=50000):
        """Search for PubMed IDs matching a query."""
        url = f"{self.base_url}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "usehistory": "y",
            "retmode": "json"
        }
        if self.api_key:
            params["api_key"] = self.api_key
            
        print(f"Searching PubMed for: {query}")
        response = requests.get(url, params=params)
        data = response.json()
        
        id_list = data.get("esearchresult", {}).get("idlist", [])
        print(f"Found {len(id_list)} IDs.")
        return id_list

    def fetch_details(self, id_list, batch_size=200):
        """Fetch abstract details for a list of IDs."""
        results = []
        
        print(f"Fetching details for {len(id_list)} articles...")
        for i in tqdm(range(0, len(id_list), batch_size)):
            batch_ids = id_list[i:i+batch_size]
            url = f"{self.base_url}efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(batch_ids),
                "retmode": "xml"
            }
            if self.api_key:
                params["api_key"] = self.api_key
            
            # Rate limiting: 3 requests per second without API key, 10 with
            time.sleep(0.4 if not self.api_key else 0.1)
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for article in root.findall(".//PubmedArticle"):
                    pmid = article.find(".//PMID").text
                    title = article.find(".//ArticleTitle")
                    title_text = title.text if title is not None else ""
                    
                    abstract = article.find(".//AbstractText")
                    abstract_text = abstract.text if abstract is not None else ""
                    
                    if abstract_text:
                        results.append({
                            "pmid": pmid,
                            "title": title_text,
                            "abstract": abstract_text
                        })
            
            # Temporary safety break for demo purposes
            if len(results) >= 50000:
                break
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    fetcher = PubMedFetcher()
    # Example query focused on radiology and respiratory diseases
    query = "(radiology[MeSH Terms] OR 'chest x-ray' OR 'thoracic imaging') AND (pneumonia OR silicosis OR asbestosis OR tuberculosis)"
    ids = fetcher.search_ids(query, max_results=1000) # Limited for initial run
    df = fetcher.fetch_details(ids)
    
    os.makedirs("data/rag", exist_ok=True)
    df.to_csv("data/rag/pubmed_abstracts.csv", index=False)
    print(f"Saved {len(df)} abstracts to data/rag/pubmed_abstracts.csv")
