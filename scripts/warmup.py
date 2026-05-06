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
    
    # Initialize fusion
    LateFusionModel()
    
    print("Models pre-loaded successfully.")

if __name__ == "__main__":
    warmup()
