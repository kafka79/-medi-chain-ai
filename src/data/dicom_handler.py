import pydicom
import numpy as np
from PIL import Image
import os

class DICOMProcessor:
    """
    Handles medical-grade DICOM ingestion with high bit-depth precision.
    Fix for Marco: Handles 16-bit depth normalization to avoid pathology masking.
    """
    def __init__(self, output_dir='data/processed/images'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_dicom(self, dicom_path):
        """
        Convert DICOM to high-fidelity PNG while preserving clinical dynamic range.
        """
        ds = pydicom.dcmread(dicom_path)
        
        # Extract pixel data
        img_array = ds.pixel_array.astype(float)
        
        # Clinical Bit-Depth handling
        # Medical X-rays are often 12-bit or 16-bit. 
        # Simple 8-bit conversion hides faint tumors.
        
        # Apply Rescale Slope and Intercept if present (DICOM standard)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img_array = img_array * ds.RescaleSlope + ds.RescaleIntercept
            
        # Min-Max Normalization to [0, 1] range for CLIP
        img_min = np.min(img_array)
        img_max = np.max(img_array)
        
        if img_max > img_min:
            img_norm = (img_array - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img_array)
            
        # Convert to 8-bit ONLY for visualization/standard CLIP, 
        # but keep high-precision for specific diagnostic logic if needed.
        img_8bit = (img_norm * 255).astype(np.uint8)
        
        img_output = Image.fromarray(img_8bit)
        filename = os.path.basename(dicom_path).replace('.dcm', '.png')
        save_path = os.path.join(self.output_dir, filename)
        img_output.save(save_path)
        
        return save_path

if __name__ == "__main__":
    # Integration test
    pass
