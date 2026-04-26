import torch
import torch.nn as nn
from PIL import Image
import open_clip
import time
import logging
from typing import Union, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomedVisualEncoder:
    """
    Visual Encoder using Microsoft's BiomedCLIP for medical imaging (IU-Xray).
    Extracts 512-dimensional embeddings specialized for biomedical contexts.
    """
    def __init__(self, model_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        
        logger.info(f"Loading BiomedCLIP model: {model_id} on {self.device}")
        try:
            self.model, self.preprocess = open_clip.create_model_from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP model: {e}")
            raise

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, str, List[Union[Image.Image, str]]]) -> torch.Tensor:
        """
        Encodes one or more images into embeddings.
        
        Args:
            image: A PIL Image, a path to an image, or a list of either.
            
        Returns:
            torch.Tensor: Embeddings of shape (N, 512).
        """
        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            processed_images.append(self.preprocess(img))

        image_input = torch.stack(processed_images).to(self.device)
        
        # BiomedCLIP visual branch
        embeddings = self.model.encode_image(image_input)
        
        # Normalize embeddings
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings

    def benchmark(self, dummy_image_path: Optional[str] = None, iterations: int = 10):
        """
        Benchmarks the encoding speed and VRAM usage.
        """
        logger.info(f"Starting benchmark for {iterations} iterations...")
        
        if dummy_image_path:
            img = Image.open(dummy_image_path).convert("RGB")
        else:
            # Create a dummy PIL image
            img = Image.new('RGB', (224, 224), color = (73, 109, 137))
            
        # Warmup
        _ = self.encode_image(img)
        
        start_time = time.time()
        for _ in range(iterations):
            _ = self.encode_image(img)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        logger.info(f"Average encoding time: {avg_time:.4f} seconds per image")
        
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            logger.info(f"VRAM usage: {vram:.2f} MB")
        
        return {
            "avg_time": avg_time,
            "vram_mb": torch.cuda.memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0
        }

if __name__ == "__main__":
    # Quick test
    encoder = BiomedVisualEncoder()
    results = encoder.benchmark()
    print(f"Benchmark Results: {results}")
