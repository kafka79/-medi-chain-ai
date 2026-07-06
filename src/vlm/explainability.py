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
        
        # Strategy A: Check user configuration via environment variable (e.g. "visual.trunk.blocks[-1].norm1")
        target_layer_path = os.getenv("GRADCAM_TARGET_LAYER")
        if target_layer_path:
            try:
                curr = model
                for part in target_layer_path.split("."):
                    if "[" in part and part.endswith("]"):
                        name, idx_str = part[:-1].split("[")
                        idx = int(idx_str)
                        curr = getattr(curr, name)[idx]
                    else:
                        curr = getattr(curr, part)
                self.target_layers = [curr]
                logger.info(f"Successfully resolved custom Grad-CAM target layer: {target_layer_path}")
            except Exception as e:
                logger.error(f"Failed to resolve custom Grad-CAM target layer path '{target_layer_path}': {e}. Falling back to default strategies.")

        # Strategy B: BiomedCLIP/OpenCLIP visual.trunk (standard in newer versions)
        if not self.target_layers and hasattr(model, "visual") and hasattr(model.visual, "trunk"):
            try:
                self.target_layers = [model.visual.trunk.blocks[-1].norm1]
            except Exception:
                pass
                
        # Strategy C: Fallback to visual.transformer
        if not self.target_layers and hasattr(model, "visual") and hasattr(model.visual, "transformer"):
            try:
                self.target_layers = [model.visual.transformer.resblocks[-1]]
            except Exception:
                pass

        # Strategy D: General fallback searching for any module named norm1 or resblocks
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
        """Generate Grad-CAM heatmap for an image.
        
        Flaw #1-structural Fix: Uses letterbox-padding-aware preprocessing
        instead of center-crop. The full image is padded to a square, then
        resized to 224×224, ensuring NO peripheral regions are discarded.
        The heatmap reverse-mapping strips the padding to overlay correctly.
        """
        with Image.open(image_path) as pil_img:
            rgb_img = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
            
            # Letterbox-pad the image to a square before preprocessing
            padded_pil, pad_info = self._letterbox_pad(pil_img)
            input_tensor = self.preprocess(padded_pil).unsqueeze(0).to(next(self.model.parameters()).device)

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

        # Generate grayscale CAM (on the padded square input)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Reverse the letterbox transform: resize CAM to the padded square size,
        # then crop out the padding to get the CAM at original image dimensions.
        H, W = rgb_img.shape[0], rgb_img.shape[1]
        pad_top, pad_left, padded_size = pad_info["pad_top"], pad_info["pad_left"], pad_info["padded_size"]
        
        # Resize CAM to the padded square dimensions using bicubic interpolation
        cam_padded = cv2.resize(grayscale_cam, (padded_size, padded_size), interpolation=cv2.INTER_CUBIC)
        
        # Crop out padding to recover original aspect ratio
        cam_original = cam_padded[pad_top:pad_top + H, pad_left:pad_left + W]
        
        # Safety: ensure exact match (rounding can cause ±1px)
        if cam_original.shape != (H, W):
            cam_original = cv2.resize(cam_original, (W, H), interpolation=cv2.INTER_CUBIC)
            
        # Apply edge-preserving bilateral filtering to avoid blurry visual artifacts
        cam_u8 = (cam_original * 255.0).astype(np.uint8)
        cam_filtered = cv2.bilateralFilter(cam_u8, d=9, sigmaColor=75, sigmaSpace=75)
        cam_original = cam_filtered.astype(np.float32) / 255.0

        visualization = show_cam_on_image(rgb_img, cam_original, use_rgb=True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Saved heatmap to {output_path}")
            
        return visualization

    @staticmethod
    def _letterbox_pad(pil_img: Image.Image) -> tuple:
        """Pad an image to a square using reflect-padding, preserving all content.

        Panel Flaw #4 Fix: Replaced solid black (0,0,0) padding with reflect
        padding. Black borders create high-contrast artificial edges that
        attract ViT self-attention to the padding boundary instead of the
        actual pathology, distorting both predictions and Grad-CAM heatmaps.
        Reflect-padding mirrors edge pixels, producing smooth continuations
        that are invisible to the attention mechanism.

        Returns:
            (padded_image, pad_info) where pad_info contains pad_top, pad_left, padded_size
        """
        w, h = pil_img.size
        max_dim = max(w, h)

        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top

        img_array = np.array(pil_img.convert("RGB"))
        padded_array = np.pad(
            img_array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='reflect'
        )
        padded = Image.fromarray(padded_array)

        return padded, {"pad_top": pad_top, "pad_left": pad_left, "padded_size": max_dim}

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
