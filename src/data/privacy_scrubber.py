import re
import logging
import threading
from typing import Dict, Any

logger = logging.getLogger("privacy-scrubber")

class PrivacyScrubber:
    """
    Addresses 'De-identification Edge Cases'.
    Performs deep cleaning of extracted clinical text to remove PII/PHI.
    """
    def __init__(self):
        # Basic patterns for names, IDs, and dates
        self.patterns = {
            "mrn": re.compile(r"\b\d{6,10}\b"),
            "dob": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "date": re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b", re.IGNORECASE),
            "zipcode": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
            "doctor": re.compile(r"\bDr\.\s+[A-Z][a-zA-Z]*\b"),
            "patient": re.compile(r"\bPatient:\s+[A-Z][a-zA-Z]*\b")
        }
        self.ner = None
        self.ner_load_error = None
        self._ner_lock = threading.Lock()
        self._ner_loaded = False
        
        # Pre-initialize/download NER in the background to prevent blocking application lifespan startup
        threading.Thread(target=self._init_ner, daemon=True).start()

    def _init_ner(self):
        """Thread-safe lazy initialization of the Hugging Face NER pipeline."""
        with self._ner_lock:
            if self._ner_loaded:
                return
            logger.info("Initializing Hugging Face NER pipeline in background thread...")
            try:
                from transformers import pipeline
                # Using aggregation_strategy="simple" merges B-PER and I-PER into a single PER entity
                self.ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
                self.ner_load_error = None
                logger.info("Hugging Face NER pipeline loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load NER pipeline: {e}")
                self.ner = None
                self.ner_load_error = e
            finally:
                self._ner_loaded = True

    def scrub_text(self, text: str) -> str:
        """Removes PII patterns from text."""
        # Ensure NER is loaded (blocks if background thread is still initializing)
        if not self._ner_loaded:
            self._init_ner()
            
        if self.ner is None or self.ner_load_error is not None:
            raise RuntimeError(
                f"Critical HIPAA Risk: Privacy scrubber failed to load the NER model. "
                f"Aborting process to prevent patient data leakage. Error: {self.ner_load_error}"
            )
            
        scrubbed = text
        for label, pattern in self.patterns.items():
            scrubbed = pattern.sub(f"[{label.upper()}_REDACTED]", scrubbed)
            
        try:
            entities = self.ner(scrubbed)
            # Sort entities in reverse order to replace from end to start without affecting indices
            for ent in sorted(entities, key=lambda x: x['start'], reverse=True):
                if ent['entity_group'] in ['PER', 'ORG', 'LOC']:
                    scrubbed = scrubbed[:ent['start']] + f"[NER_{ent['entity_group']}_REDACTED]" + scrubbed[ent['end']:]
        except Exception as e:
            logger.error(f"NER scrubbing failed: {e}")
            raise RuntimeError(f"Critical HIPAA Risk: NER scrubbing failed at runtime. Aborting. Error: {e}")
                
        return scrubbed

    def scrub_history_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deep scrubs the history dictionary."""
        scrubbed_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                scrubbed_data[key] = self.scrub_text(value)
            elif isinstance(value, dict):
                scrubbed_data[key] = self.scrub_history_data(value)
            else:
                scrubbed_data[key] = value
        return scrubbed_data

    def detect_burned_in_text(self, image_path: str):
        """
        Detects regions likely to contain burned-in patient identifiers.
        To avoid naively masking pathology (like calcifications or nodules) in the lung fields,
        we restrict detection exclusively to the outer peripheral zones (top 15% and bottom 15%
        of the image height) where patient names, MRNs, DOBs, and scanning timestamps are standard.
        """
        import cv2
        import numpy as np
        
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return []
            
            h, w = img.shape
            boxes = []
            
            # Restrict detection to top 15% and bottom 15% of the image height
            zones = [
                (0, int(h * 0.15)),          # Top zone
                (int(h * 0.85), h)           # Bottom zone
            ]
            
            for start_y, end_y in zones:
                if start_y >= end_y:
                    continue
                # Segment of the image for the current zone
                zone_img = img[start_y:end_y, :]
                
                # Thresholding using Otsu's binarization to segment text contours
                thresh = cv2.threshold(zone_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Apply horizontal dilation to merge individual character contours into words/lines
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
                dilated = cv2.dilate(thresh, kernel, iterations=1)
                
                # Find connected contours that could represent letters or words
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                    
                    # Filtering contours using basic heuristics for typical text dimensions:
                    # - Height must be between 8 and 35 pixels
                    # - Width must be at least 12 pixels
                    if 8 <= h_c <= 35 and w_c >= 12:
                        # Translate zone-relative y-coordinate to full image space
                        boxes.append((x_c, start_y + y_c, w_c, h_c))
                        
            return boxes
        except Exception as e:
            logger.error(f"Error detecting burned-in text: {e}")
            return []

    def mask_burned_in_text(self, image_path: str) -> str:
        """
        Scrubs PHI from images:
        - For DICOM (.dcm), reads dataset headers and anonymizes standard identifier tags.
        - For standard images, masks burned-in text within peripheral zones by drawing black boxes.
        Saves the sanitized image to a safe tempfile and returns its path.
        """
        if str(image_path).lower().endswith(".dcm"):
            try:
                import pydicom
                import tempfile
                
                ds = pydicom.dcmread(image_path)
                # Key standard Patient / Institution identity tags to sanitize
                phi_tags = [
                    "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                    "InstitutionName", "AccessionNumber", "PatientAddress", 
                    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName"
                ]
                for tag in phi_tags:
                    if tag in ds:
                        ds.data_element(tag).value = f"REDACTED_{tag.upper()}"
                        
                tmp = tempfile.NamedTemporaryFile(suffix=".dcm", delete=False)
                sanitized_path = tmp.name
                tmp.close()
                
                ds.save_as(sanitized_path)
                logger.info(f"Successfully scrubbed DICOM metadata headers. Saved to {sanitized_path}")
                return sanitized_path
            except Exception as e:
                logger.error(f"Failed to scrub DICOM metadata headers: {e}")
                return image_path
            
        import cv2
        import shutil
        import tempfile
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path
                
            boxes = self.detect_burned_in_text(image_path)
            if not boxes:
                return image_path
                
            for (x, y, w, h) in boxes:
                # Mask text box by drawing a solid black rectangle over it
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
                
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            output_path = tmp.name
            tmp.close()
                
            cv2.imwrite(output_path, img)
            logger.info(f"Masked {len(boxes)} PHI text regions. Sanitized image saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to mask burned-in text: {e}")
            return image_path
