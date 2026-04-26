import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import os

class VisualExplainer:
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess
        
        # In Vit (BiomedCLIP), targets are usually transformer blocks
        # Target the last layer of the visual backbone
        # This depends on the specific architecture of BiomedCLIP
        self.target_layers = [model.visual.trunk.blocks[-1].norm1]

    def reshape_transform(self, tensor, height=14, width=14):
        # Result of ViT backbone is (Batch, Tokens, Dim)
        # We need to reshape to (Batch, Dim, Height, Width)
        # B/16 uses 224/16 = 14x14 patches + 1 cls token
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        
        # Bring the channels to the first dimension
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def generate_heatmap(self, image_path, target_category=None, output_path=None):
        """Generate Grad-CAM heatmap for an image."""
        rgb_img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
        input_tensor = self.preprocess(Image.open(image_path)).unsqueeze(0).to(next(self.model.parameters()).device)

        # Construct the CAM object with reshape_transform for ViT
        cam = GradCAM(model=self.model.visual, 
                      target_layers=self.target_layers, 
                      reshape_transform=self.reshape_transform)

        # If target_category is None, it targets the highest scoring class
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None

        # Generate grayscale CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Overlay on image (ensure they are the same size)
        img_resized = cv2.resize(rgb_img, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
        visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Saved heatmap to {output_path}")
            
        return visualization

if __name__ == "__main__":
    import open_clip
    import os
    
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
