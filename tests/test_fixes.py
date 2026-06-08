import os
import json
import pytest
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pydicom

from src.monitoring.xai_explainer import XAIExplainer
from src.data.privacy_scrubber import PrivacyScrubber
from src.data.fhir_formatter import EHRGateway

def test_dynamic_xai_explainer():
    explainer = XAIExplainer()
    
    # Patient 1: Silicosis with dust exposure history
    history_silicosis = {
        "chief_complaint": "severe shortness of breath",
        "metadata": {
            "age": "45",
            "gender": "Male",
            "occupation": "Coal Miner",
            "exposure_years": "15"
        }
    }
    
    rationale_silicosis = explainer.explain(
        diagnosis="Silicosis",
        confidence=0.85,
        uncertainty=0.04,
        probabilities=[0.85, 0.05, 0.05, 0.03, 0.02],
        history_data=history_silicosis
    )
    
    assert "Miner" in rationale_silicosis
    assert "15" in rationale_silicosis
    assert "severe shortness of breath" in rationale_silicosis
    
    # Patient 2: Pneumonia
    history_pneumonia = {
        "chief_complaint": "high fever and productive cough",
        "metadata": {
            "age": "28",
            "gender": "Female",
            "occupation": "Teacher",
            "exposure_years": "0"
        }
    }
    
    rationale_pneumonia = explainer.explain(
        diagnosis="Pneumonia",
        confidence=0.91,
        uncertainty=0.02,
        probabilities=[0.02, 0.91, 0.03, 0.02, 0.02],
        history_data=history_pneumonia
    )
    
    assert "consolidation" in rationale_pneumonia.lower()
    assert "Teacher" not in rationale_pneumonia  # Asbestosis/Silicosis templates use occupation, Pneumonia uses age/gender
    assert "28-year-old" in rationale_pneumonia
    assert "high fever" in rationale_pneumonia

def test_dicom_metadata_scrubbing():
    # Create a dummy DICOM file
    tmp_dcm = tempfile.NamedTemporaryFile(suffix=".dcm", delete=False)
    tmp_dcm.close()
    
    try:
        # Construct minimal valid DICOM structure
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.UID("1.2.840.10008.5.1.4.1.1.7")
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        
        ds = pydicom.dataset.FileDataset(tmp_dcm.name, {}, file_meta=file_meta, preamble=b"\0"*128)
        ds.PatientName = "John Doe"
        ds.PatientID = "12345"
        ds.PatientBirthDate = "19800101"
        ds.PatientSex = "M"
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.save_as(tmp_dcm.name)
        
        # Scrub it
        scrubber = PrivacyScrubber()
        scrubbed_path = scrubber.mask_burned_in_text(tmp_dcm.name)
        
        # Verify it created a new temp path and redacted fields
        assert scrubbed_path != tmp_dcm.name
        assert os.path.exists(scrubbed_path)
        
        scrubbed_ds = pydicom.dcmread(scrubbed_path)
        assert scrubbed_ds.PatientName == "REDACTED_PATIENTNAME"
        assert scrubbed_ds.PatientID == "REDACTED_PATIENTID"
        assert scrubbed_ds.PatientBirthDate == "REDACTED_PATIENTBIRTHDATE"
        
        # Cleanup scrubbed file
        if os.path.exists(scrubbed_path):
            os.unlink(scrubbed_path)
            
    finally:
        if os.path.exists(tmp_dcm.name):
            os.unlink(tmp_dcm.name)

def test_opencv_peripheral_redaction():
    # Create a dummy image
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_img.close()
    
    try:
        # Create a black image
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Write text in top zone (peripheral)
        cv2.putText(img, "MRN: 987654", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Write text in center zone (clinical zone - shouldn't be touched by peripheral filter)
        cv2.putText(img, "PAT HOLOGY", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(tmp_img.name, img)
        
        scrubber = PrivacyScrubber()
        boxes = scrubber.detect_burned_in_text(tmp_img.name)
        
        # Verify that we detected the MRN in the peripheral zone (y < 60)
        # but completely ignored the "PAT HOLOGY" in the center (y = 200)
        assert len(boxes) > 0
        for x, y, w, h in boxes:
            assert y < 60 or y > 340  # Top 15% (60px) or Bottom 15% (340px+)
            assert not (150 < y < 250)
            
    finally:
        if os.path.exists(tmp_img.name):
            os.unlink(tmp_img.name)

def test_local_first_dlq_fallback():
    # Use a mock EHR gateway pointing to a completely broken URL
    gateway = EHRGateway(endpoint_url="http://completely-broken-invalid-host/fhir")
    
    # Clear local DLQ folder
    dlq_dir = Path("temp/dlq")
    if dlq_dir.exists():
        for f in dlq_dir.glob("failed_report_*.json"):
            os.unlink(f)
            
    dummy_payload = json.dumps({"resourceType": "DiagnosticReport", "id": "123"})
    result = gateway.push_report(dummy_payload)
    
    # The push must return False (failure to send) but NOT raise an exception
    assert result is False
    
    # Verify local file was created in temp/dlq
    files = list(dlq_dir.glob("failed_report_*.json"))
    assert len(files) == 1
    
    with open(files[0], "r") as f:
        stored = json.load(f)
        
    assert stored["payload"]["resourceType"] == "DiagnosticReport"
    assert "payload" in stored
    assert "error" in stored
    
    # Clean up
    os.unlink(files[0])
