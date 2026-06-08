import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from src.vlm.visual_encoder import BiomedVisualEncoder
from src.models.fusion import LateFusionModel

def warmup():
    print("Pre-loading models for production warm-up...")
    
    # Pre-download SapBERT
    MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    SentenceTransformer(MODEL_NAME)
    
    # Pre-initialize visual encoder (BiomedCLIP)
    BiomedVisualEncoder()
    
    # Initialize fusion and create dummy checkpoint
    import torch
    fusion = LateFusionModel()
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/fusion_model.pt"
    if not os.path.exists(checkpoint_path):
        torch.save(fusion.state_dict(), checkpoint_path)
        print(f"Created mock checkpoint at {checkpoint_path}")
    
    # Pre-download NER for privacy scrubbing
    print("Pre-downloading NER model for Privacy Scrubber...")
    from transformers import pipeline
    pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    
    print("Models pre-loaded successfully.")

if __name__ == "__main__":
    warmup()
