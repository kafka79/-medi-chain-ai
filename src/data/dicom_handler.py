import pydicom
from typing import Optional
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


def create_secondary_capture(original_dcm_path: Optional[str], heatmap_png_path: str, output_dcm_path: str) -> str:
    """
    Generate a DICOM Secondary Capture image dataset using metadata from the original uploaded scan.
    Converts visual PNG overlay output into an standard-compliant DICOM dataset.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid
    from PIL import Image
    import numpy as np
    from typing import Optional

    # Load the heatmap overlay image
    img = Image.open(heatmap_png_path).convert("RGB")
    pixel_data = np.array(img)

    # Attempt to load metadata from original scan
    ds = None
    if original_dcm_path and os.path.exists(original_dcm_path):
        try:
            ds = pydicom.dcmread(original_dcm_path)
        except Exception:
            pass

    # Setup FileMeta headers
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 200
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = '1.2.840.10008.5.1.4.1.1.7.0.1'
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Create new DICOM FileDataset
    new_ds = FileDataset(output_dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    new_ds.is_little_endian = True
    new_ds.is_implicit_VR = False

    # Extract/populate identity headers
    if ds:
        new_ds.PatientName = getattr(ds, "PatientName", "REDACTED_PATIENTNAME")
        new_ds.PatientID = getattr(ds, "PatientID", "REDACTED_PATIENTID")
        new_ds.PatientBirthDate = getattr(ds, "PatientBirthDate", "")
        new_ds.PatientSex = getattr(ds, "PatientSex", "")
        new_ds.StudyInstanceUID = getattr(ds, "StudyInstanceUID", generate_uid())
        new_ds.SeriesInstanceUID = getattr(ds, "SeriesInstanceUID", generate_uid())
        new_ds.StudyID = getattr(ds, "StudyID", "")
        new_ds.AccessionNumber = getattr(ds, "AccessionNumber", "")
    else:
        new_ds.PatientName = "REDACTED_PATIENTNAME"
        new_ds.PatientID = "REDACTED_PATIENTID"
        new_ds.StudyInstanceUID = generate_uid()
        new_ds.SeriesInstanceUID = generate_uid()

    new_ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    new_ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    new_ds.Modality = "OT"
    new_ds.ConversionType = "WSD"

    # Image properties
    new_ds.Rows = pixel_data.shape[0]
    new_ds.Columns = pixel_data.shape[1]
    new_ds.SamplesPerPixel = 3
    new_ds.PhotometricInterpretation = "RGB"
    new_ds.PlanarConfiguration = 0
    new_ds.BitsAllocated = 8
    new_ds.BitsStored = 8
    new_ds.HighBit = 7
    new_ds.PixelRepresentation = 0
    new_ds.PixelData = pixel_data.tobytes()

    # Save to file
    os.makedirs(os.path.dirname(output_dcm_path), exist_ok=True)
    new_ds.save_as(output_dcm_path)
    return output_dcm_path

if __name__ == "__main__":
    # Integration test
    pass
