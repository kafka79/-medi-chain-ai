import numpy as np
import cv2
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import os
import math
import logging

logger = logging.getLogger("explainability")

class CLIPClassifierWrapper(nn.Module):
    """
    Wraps the visual encoder of BiomedCLIP and computes cosine similarity
    with the target class text embeddings to output classification-like logits.
    """
    def __init__(self, visual_model, text_embeddings):
        super().__init__()
        self.visual = visual_model
        # Register text_embeddings as a buffer so it moves with the module to the correct device
        self.register_buffer("text_embeddings", text_embeddings)
        
    def forward(self, x):
        image_features = self.visual(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Output shape: (batch, num_classes)
        logits = image_features @ self.text_embeddings.T
        return logits


class VisualExplainer:
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess
        
        # Resolve target layers dynamically to support different OpenCLIP versions and ViT trunk structures
        self.target_layers = None
        
        # Strategy A: BiomedCLIP/OpenCLIP visual.trunk (standard in newer versions)
        if hasattr(model, "visual") and hasattr(model.visual, "trunk"):
            try:
                self.target_layers = [model.visual.trunk.blocks[-1].norm1]
            except Exception:
                pass
                
        # Strategy B: Fallback to visual.transformer
        if not self.target_layers and hasattr(model, "visual") and hasattr(model.visual, "transformer"):
            try:
                self.target_layers = [model.visual.transformer.resblocks[-1]]
            except Exception:
                pass

        # Strategy C: General fallback searching for any module named norm1 or resblocks
        if not self.target_layers:
            for name, module in model.named_modules():
                if "blocks" in name and "norm1" in name:
                    self.target_layers = [module]
            if not self.target_layers:
                for name, module in model.named_modules():
                    if "resblocks" in name:
                        self.target_layers = [module]
                        
        # Raise an exception if layer resolution fails
        if not self.target_layers:
            raise RuntimeError(
                "Failed to dynamically resolve Vision Transformer (ViT) target layers for Grad-CAM. "
                "Check visual encoder model attributes (e.g. trunk, transformer)."
            )

        # Encode diagnostic classes text to compute visual-text similarity for Grad-CAM
        from src.models.fusion import DIAGNOSTIC_CLASSES
        import open_clip
        
        self.class_embeddings = None
        if hasattr(model, "encode_text"):
            try:
                # Load tokenizer dynamically
                model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                tokenizer = open_clip.get_tokenizer(model_id)
                tokens = tokenizer(DIAGNOSTIC_CLASSES)
                device = next(model.parameters()).device
                tokens = tokens.to(device)
                with torch.no_grad():
                    text_features = model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                self.class_embeddings = text_features
                logger.info("VisualExplainer successfully pre-encoded diagnostic classes text labels.")
            except Exception as e:
                logger.warning(f"Could not pre-encode text labels for visual explainer: {e}. Falling back to visual projections.")

    def reshape_transform(self, tensor, height=14, width=14):
        # Result of ViT backbone is (Batch, Tokens, Dim)
        # We need to reshape to (Batch, Dim, Height, Width)
        # B/16 uses 224/16 = 14x14 patches + 1 cls token
        seq_len = tensor.size(1) - 1
        grid_size = int(round(math.sqrt(seq_len)))
        
        # Verify shape integrity
        if grid_size * grid_size != seq_len:
            # Fallback dynamic calculation if sequence length is unexpected
            logger.warning(f"Reshape sequence length {seq_len} is not a perfect square. Falling back to default grid dimensions.")
            grid_size = 14
            
        result = tensor[:, 1:, :].reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
        
        # Bring the channels to the first dimension
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def generate_heatmap(self, image_path, target_category=None, output_path=None):
        """Generate Grad-CAM heatmap for an image."""
        with Image.open(image_path) as pil_img:
            rgb_img = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
            input_tensor = self.preprocess(pil_img).unsqueeze(0).to(next(self.model.parameters()).device)

        if self.class_embeddings is not None:
            # Construct similarity-based wrapper so ClassifierOutputTarget targets actual similarity scores
            wrapper = CLIPClassifierWrapper(self.model.visual, self.class_embeddings)
            cam = GradCAM(model=wrapper, 
                          target_layers=self.target_layers, 
                          reshape_transform=self.reshape_transform)
        else:
            # Fallback for mock/test environments
            cam = GradCAM(model=self.model.visual, 
                          target_layers=self.target_layers, 
                          reshape_transform=self.reshape_transform)

        # If target_category is None, it targets the highest scoring class
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None

        # Generate grayscale CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Resize grayscale CAM up to original image dimensions with aspect-ratio preservation (mapping the CenterCrop box)
        H, W = rgb_img.shape[0], rgb_img.shape[1]
        cam_full = np.zeros((H, W), dtype=np.float32)
        
        # Center-crop mapping (BiomedCLIP standard preprocessor resizes the shortest edge to 224, then center-crops to 224x224)
        if W < H:
            # Portrait: cropped to a W x W square in the vertical center of the image
            box_w = W
            box_h = W
            top = (H - W) // 2
            bottom = top + W
            left = 0
            right = W
        else:
            # Landscape: cropped to an H x H square in the horizontal center of the image
            box_w = H
            box_h = H
            top = 0
            bottom = H
            left = (W - H) // 2
            right = left + H
            
        cam_resized = cv2.resize(grayscale_cam, (box_w, box_h))
        cam_full[top:bottom, left:right] = cam_resized

        visualization = show_cam_on_image(rgb_img, cam_full, use_rgb=True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Saved heatmap to {output_path}")
            
        return visualization

if __name__ == "__main__":
    import open_clip
    # Load model
    model_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, preprocess = open_clip.create_model_from_pretrained(model_id)
    model.eval()
    
    explainer = VisualExplainer(model, preprocess)
    
    image_path = "data/raw/sample_xray.png"
    if os.path.exists(image_path):
        output_path = "outputs/heatmaps/sample_heatmap.png"
        explainer.generate_heatmap(image_path, output_path=output_path)
    else:
        print(f"Error: {image_path} not found.")
